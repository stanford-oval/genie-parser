# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: sawyerb <sawyerb@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
Created on May 2, 2017

@author: sawyerb
'''

import tensorflow as tf
import numpy as np
import re
import math
from tensorflow.python.util import nest

from .base_model import BaseModel

from .config import Config

from util import beam_search

# TODO put params in config.py
NUM_ENCODER_BLOCKS = 6;
NUM_DECODER_BLOCKS = 6;
NUM_HEADS = 5;

class TransformerAligner(BaseModel):

    def build(self):
        self.add_placeholders()

        # Calculate padding and attention bias mask for attention layers.
        # (batch_size, max_length)
        input_padding = get_padding_mask(self.input_length_placeholder, self.config.max_length)
        attention_bias = get_padding_bias(input_padding)

        # Add embeddings (input/output vocab_size, input/output embed_size)
        self.add_input_embeddings()
        self.add_output_embeddings()

        # Add the encoder (batch_size, max_length, encoder_hidden_size)
        final_enc_state = self.add_encoder_op(input_padding, attention_bias)

        # Add the training decoder (batch_size, max_length, output_size)
        train_logits = self.add_decoder_op(final_enc_state, attention_bias)

        # Calculate training loss and training op
        if self.config.decoder_sequence_loss > 0:
            with tf.name_scope('sequence_loss'):
                seq_loss = self.add_loss_op(train_logits)
        else:
            seq_loss = 0

        with tf.name_scope('training_loss'):
            seq_loss *= self.config.decoder_sequence_loss
            self.loss = seq_loss + self.add_regularization_loss()

        self.train_op = self.add_training_op(self.loss)

        # Add inference decoder (batch_size, max_length, output_size)
        with tf.variable_scope('eval_decoder'):
            preds, _ = self.add_predict_op(final_enc_state, attention_bias)

        self.eval_loss = self.loss
        self.preds = self.finalize_predictions(preds)
        #     eval_logits = self.add_predict_op(final_enc_state, attention_bias)

        # # Calculate evaluation loss
        # if self.config.decoder_sequence_loss > 0:
        #     with tf.name_scope('sequence_loss'):
        #         eval_seq_loss = self.add_loss_op(eval_logits)
        # else:
        #     eval_seq_loss = 0

        # with tf.name_scope('eval_loss'):
        #     self.eval_loss = self.config.decoder_sequence_loss * eval_seq_loss

        # Finalize predictions by taking argmax and adding beam size of 1
        # self.raw_preds = eval_logits
        # preds = tf.argmax(eval_logits, axis=2)
        # self.preds = self.finalize_predictions(preds)
        if not isinstance(self.preds, dict):
            self.preds = {
                self.config.grammar.primary_output: self.preds
            }

        print_weights(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def add_placeholders(self):
        self.add_input_placeholders()
        self.add_output_placeholders()
        self.add_extra_placeholders()

    def add_input_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='input_sequence')
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='input_length')
        self.constituency_parse_placeholder = tf.placeholder(tf.bool, shape=(None, 2*self.config.max_length-1), name='input_constituency_parse')

    def add_output_placeholders(self):
        self.output_placeholders = dict()
        for key in self.config.grammar.output_size:
            self.output_placeholders[key] = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='output_sequence')
            self.primary_output_placeholder = self.output_placeholders[self.config.grammar.primary_output]
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='output_length')

    def add_extra_placeholders(self):
        self.batch_number_placeholder = tf.placeholder(tf.int32, shape=(), name='batch_number')
        self.epoch_placeholder = tf.placeholder(tf.int32, shape=(), name='epoch_number')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_probability')
        self.action_counts = None       # Don't need

    def create_feed_dict(self, inputs_batch, input_length_batch, parses_batch,
                         labels_sequence_batch=None, labels_batch=None, label_length_batch=None,
                         dropout=1, batch_number=0, epoch=0):

        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.input_length_placeholder] = input_length_batch
        feed_dict[self.constituency_parse_placeholder] = parses_batch
        if labels_batch is not None:
            for key, batch in labels_batch.items():
                feed_dict[self.output_placeholders[key]] = batch
        if label_length_batch is not None:
            feed_dict[self.output_length_placeholder] = label_length_batch
        feed_dict[self.dropout_placeholder] = dropout
        feed_dict[self.batch_number_placeholder] = batch_number
        feed_dict[self.epoch_placeholder] = epoch

        return feed_dict

    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]

    @property
    def output_size(self):
        return self.config.grammar.output_size[self.config.grammar.primary_output]

    def add_input_embeddings(self):
        xavier = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('embed', initializer=xavier):
            with tf.variable_scope('input'):

                shape = (self.config.dictionary_size, self.config.input_embed_size)
                init = None

                if self.config.train_input_embeddings:
                    if self.config.input_embedding_matrix is not None:
                        init = tf.constant_initializer(self.config.input_embedding_matrix)

                    self.input_embed_matrix = tf.get_variable('embedding',
                                                              shape=shape,
                                                              initializer=init)
                else:
                    self.input_embed_matrix = tf.constant(self.config.input_embedding_matrix)

    def add_output_embeddings(self):
        xavier = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('embed', initializer=xavier):
            with tf.variable_scope('output'):
                self.output_embed_matrices = dict()
                pretrained = self.config.output_embedding_matrix

                for key, size in self.config.grammar.output_size.items():
                    if self.config.grammar.is_copy_type(key):
                        continue
                    shape = (size, self.config.output_embed_size)
                    if key == self.config.grammar.primary_output and self.config.train_output_embeddings:
                        embed_matrix = tf.get_variable('embedding_' + key,
                                                        shape=shape)
                    else:
                        embed_matrix = tf.constant(pretrained[key], name='embedding_' + key)

                    self.output_embed_matrices[key] = embed_matrix

        output_embed_matrix = self.output_embed_matrices[self.config.grammar.primary_output]
        self.output_embed_matrix = output_embed_matrix

    def add_encoder_op(self, input_padding, attention_bias):
        ''' Add the encoder operation, given inputs in self.input_placeholder.
        Produces the final encoding hidden state.
        Args:
            attention_bias: bias for the self-attention layer. [batch_size, 1, 1, max_length]
            inputs_padding: padding mask. [batch_size, max_length]
        Returns:
            A 3D tensor (batch_size, max_length, encoder_hidden_size)
        '''

        with tf.variable_scope('encoder'):

            # (batch size, max input length, embed_size)
            input_embeds = tf.nn.embedding_lookup([self.input_embed_matrix], self.input_placeholder)

            # Now project the input down to a small size
            input_embeds = tf.layers.dense(input_embeds, self.config.input_projection)

            # Positionally encode them.
            with tf.variable_scope("positional_encoding"):
                input_embeds += positional_encoding(self.config.max_length,
                                                        self.config.input_projection)
            # Apply dropout immediately before encoding.
            input_embeds = tf.nn.dropout(input_embeds,
                                             self.dropout_placeholder)
            # Encoder block. (batch_size, max_length, encoder_hidden_size)
            final_enc_state = self.add_encoder_block(input_embeds,
                                                     attention_bias,
                                                     input_padding)
        return final_enc_state

    def add_encoder_block(self, inputs, attention_bias, input_padding):
        ''' Adds the encoder block of the Transformer network.
        Args:
            inputs: (batch_size, max_length, embed_size)
            attention_bias: bias for the self-attention layer. [batch_size, 1, 1, max_length]
            inputs_padding: padding mask. [batch_size, max_length]
        Return:
            A 3D tensor (batch_size, max_length, hidden_size)
        '''

        hidden_size = self.config.encoder_hidden_size

        for i in range(NUM_ENCODER_BLOCKS):
            with tf.variable_scope('layer_{}'.format(i)):
                with tf.variable_scope('self_attention'):
                    # (batch_size, max_length, encoder_hidden_size)
                    mh_out = multihead_attention(query=inputs, key=inputs,
                                                 attn_bias=attention_bias,
                                                 attn_size=hidden_size,
                                                 num_heads=NUM_HEADS,
                                                 dropout_rate=self.dropout_placeholder)

                    # Residual connection and normalize
                    if (i != 0): mh_out += inputs
                    mh_out = normalize(mh_out, scope="mh_norm_{}".format(i))

                with tf.variable_scope('ffn'):
                    ff_out = feedforward(mh_out, 4*hidden_size, hidden_size,
                                         padding=input_padding,
                                         dropout=self.dropout_placeholder)

                    # Residual connection and normalize
                    ff_out += mh_out
                    inputs = normalize(ff_out, scope="ff_norm_{}".format(i))

        return inputs

    def final_output_projection(self, final_state):
        # Final linear projection (batch_size, max_length, output_size)
        return tf.layers.dense(final_state, self.output_size)

    def add_decoder_op(self, enc_final_state, enc_dec_attention_bias):
        ''' Adds the decoder op for training, comprised of output embeddings,
        the decoder blocks, and the final feed forward layer.

        Args:
            enc_final_state: final encoder state. (batch_size, max_length, encoder_hidden_size)
            enc_dec_attention_bias: bias for the encoder-decoder attention layer.
                (batch_size, 1, 1, max_length)
        Returns:
            A 3D Tensor of logits (batch_size, max_length, output_size)
        '''

        with tf.variable_scope('train_decoder'):
            # HACK to add the go vector: this cuts off the last token.
            go_vector = tf.ones((self.batch_size, 1), dtype=tf.int32) * self.config.grammar.start
            output_ids = tf.concat([go_vector, self.primary_output_placeholder[:, :-1]],
                                    axis=1)

            # (batch_size, max_length, output_embed_size)
            output_embeddings = tf.nn.embedding_lookup([self.output_embed_matrix],
                                                        output_ids)
            # Positionally encode them.
            with tf.variable_scope("positional_encoding"):
                output_embeddings += positional_encoding(self.config.max_length,
                                                         self.config.output_embed_size)
            # Apply dropout immediately before decoding.
            output_embeddings = tf.nn.dropout(output_embeddings,
                                              self.dropout_placeholder)

            # Get attention bias (upper triangular matrix of large negatives)
            decoder_self_attention_bias = get_decoder_self_attention_bias(self.config.max_length)

            # Decoder block. (batch_size, max_length, decoder_hidden_size)
            final_dec_state = self.add_decoder_block(enc_final_state,
                                                     output_embeddings,
                                                     decoder_self_attention_bias,
                                                     enc_dec_attention_bias)

            return self.final_output_projection(final_dec_state)

    def get_symbols_to_logits_fn(self):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = positional_encoding(self.config.max_length + 1,
                                            self.config.output_embed_size)
        decoder_self_attention_bias = get_decoder_self_attention_bias(self.config.max_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
                ids: Current decoded sequences.
                    int tensor (batch_size * beam_size, i + 1)
                i: Loop index
                cache: dictionary of values storing the encoder output,
                        encoder-decoder attention bias, and previous decoder
                        attention values.
            Returns:
                Tuple of
                    (logits with shape [batch_size * beam_size, vocab_size],
                    updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Embed decoder inputs and positionally encode them.
            # (batch_size * beam_size, 1, output_embed_size)
            decoder_input = tf.nn.embedding_lookup([self.output_embed_matrix],
                                                        decoder_input)
            decoder_input += timing_signal[i:i + 1]

            # Get just the ith self attention bias (1, 1, 1, i+1)
            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            # (batch_size * beam_size, 1, decoder_hidden_size)
            decoder_outputs = self.add_decoder_block(cache.get("encoder_outputs"),
                                                     decoder_input,
                                                     self_attention_bias,
                                                     cache.get("encoder_decoder_attention_bias"),
                                                     cache)

            # Final linear projection (batch_size, 1, output_size)
            logits = self.final_output_projection(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def add_predict_op(self, enc_final_state, enc_dec_attention_bias):
        ''' Adds the op for predictions and evaluation. Similar to the decoder
        op but each token is decoded iteratively until all inputs have hit EOS.

        Args:
            enc_final_state: final encoder state. (batch_size, max_length, encoder_hidden_size)
            enc_dec_attention_bias: bias for the encoder-decoder attention layer.
                (batch_size, 1, 1, max_length)
        Returns:
            A 3D Tensor of logits (batch_size, max_length, output_size)
        '''

        dec_hidden_size = self.config.decoder_hidden_size

        symbols_to_logits_fn = self.get_symbols_to_logits_fn()

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "K": tf.zeros([self.batch_size, 0, dec_hidden_size]),
                "V": tf.zeros([self.batch_size, 0, dec_hidden_size]),
            } for layer in range(NUM_DECODER_BLOCKS)}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = enc_final_state
        cache["encoder_decoder_attention_bias"] = enc_dec_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        # (batch_size, beam_size, max_length), (batch_size, beam_size)
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.output_size,
            beam_size=1,
            alpha=0.6,
            max_decode_length=self.config.max_length,
            eos_id=self.config.grammar.end)

        # Get the top sequence for each batch element
        # (batch_size, max_length), (batch_size)
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]
        return top_decoded_ids, top_scores

    def add_decoder_block(self, enc_final_state, dec_state,
                          decoder_self_attention_bias, attention_bias,
                          cache=None):
        ''' Adds the decoder block.

        Args:
            enc_final_state: (batch_size, max_length, encoder_hidden_size)
            dec_state: (batch_size, max_length, output_embed_size)
            decoder_self_attention_bias: bias for decoder self-attention layer.
                (1, 1, max_length, max_length)
            attention_bias: bias for encoder-decoder attention layer.
                (batch_size, 1, 1, max_length)
            cache: (Used for fast decoding) A nested dictionary storing previous
                    decoder self-attention values in the form:
                {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                           "v": tensor with shape [batch_size, i, value_channels]},
                ...}

        Returns:
            A 3D Tensor (batch_size, max_length, decoder_hidden_size)
        '''
        hidden_size = self.config.decoder_hidden_size

        for i in range(NUM_DECODER_BLOCKS):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None

            with tf.variable_scope(layer_name):
                with tf.variable_scope('self_attention'):
                    # (batch_size, max_length, decoder_hidden_size)
                    mh_out = multihead_attention(query=dec_state, key=dec_state,
                                                 attn_bias=decoder_self_attention_bias,
                                                 attn_size=hidden_size,
                                                 num_heads=NUM_HEADS,
                                                 dropout_rate=self.dropout_placeholder,
                                                 cache=layer_cache)

                    # Residual connection and normalize
                    if (i != 0): mh_out += dec_state
                    dec_state = normalize(mh_out, scope="self_norm_{}".format(i))

                with tf.variable_scope('enc_dec_attention'):
                    mh_out = multihead_attention(query=dec_state, key=enc_final_state,
                                                 attn_bias=attention_bias,
                                                 attn_size=hidden_size,
                                                 num_heads=NUM_HEADS,
                                                 dropout_rate=self.dropout_placeholder)

                    # Residual connection and normalize
                    mh_out += dec_state
                    mh_out = normalize(mh_out, scope="enc_attention_norm_{}".format(i))

                with tf.variable_scope('ffn'):
                    ff_out = feedforward(mh_out, 4*hidden_size, hidden_size,
                                         dropout=self.dropout_placeholder)

                    # Residual connection and normalize
                    ff_out += mh_out
                    dec_state = normalize(ff_out, scope="ff_norm_{}".format(i))

        return dec_state

    def finalize_predictions(self, preds):
        return tf.expand_dims(preds, axis=1)

    def add_loss_op(self, logits):
        # TODO use max margin loss
        length_diff = self.config.max_length - tf.shape(logits)[1]
        padding = tf.convert_to_tensor([[0, 0], [0, length_diff], [0, 0]], name='padding')
        logits = tf.pad(logits, padding, mode='constant')
        logits.set_shape((None, self.config.max_length, logits.shape[2]))
        logits = logits + 1e-5 # add epsilon to avoid division by 0

        mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(logits, self.primary_output_placeholder, mask)

    def _add_l2_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search(where, w.name)]
        regularizer = tf.contrib.layers.l2_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def _add_l1_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search(where, w.name)]
        if not weights:
            return 0
        regularizer = tf.contrib.layers.l1_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def add_regularization_loss(self):
        return self._add_l2_helper('/kernel:0$', self.config.l2_regularization) + \
            self._add_l2_helper('/embedding(_[a-z]+)?:0$', self.config.embedding_l2_regularization) + \
            self._add_l1_helper('/kernel:0$', self.config.l1_regularization)

    def add_training_op(self, loss):
        optclass = getattr(tf.train, self.config.optimizer + 'Optimizer')
        assert issubclass(optclass, tf.train.Optimizer)

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step,
                                                   1000, self.config.learning_rate_decay)
        optimizer = optclass(learning_rate)

        gradient_var_pairs = optimizer.compute_gradients(loss)
        vars = [x[1] for x in gradient_var_pairs]
        gradients = [x[0] for x in gradient_var_pairs]
        if self.config.gradient_clip > 0:
            clipped, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
        else:
            clipped = gradients

        self.grad_norm = tf.global_norm(clipped)
        train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=global_step)
        return train_op

    def __init__(self, config : Config):
        self.config = config


def positional_encoding(length, embed_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
        Calculates the position encoding as a mix of sine and cosine functions with
        geometrically increasing wavelengths.
        Defined and formulized in Attention is All You Need, section 3.5.
    Args:
        length: Sequence length.
        hidden_size: embedding size
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position
    Returns:
        A 2D Tensor (length, embed_size)
    """
    position = tf.to_float(tf.range(length))
    num_timescales = embed_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal

def multihead_attention(query, key, attn_bias, attn_size=None, num_heads=8,
                        dropout_rate=1, cache=None):
    # TODO change comments, query/key might not be max_length due to cache
    '''
    Args:
        query: (batch_size, max_length, C_q)
        key: (batch_size, max_length, C_k)
        attn_bias: (batch_size, 1, 1, max_length)
        attn_size: attention size and also final output size
        num_heads: number of attention layers
        dropout_rate: 1 - rate of dropout
        cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]}
                where i is the current decoded length.
    Returns
        A 3D tensor (batch_size, max_length, attn_size)
    '''

    def split_heads(x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to ensure the inner dimensions hold the correct values
        during the matrix multiplication.
        Args:
            x: (batch_size, length, attn_size)
        Returns:
            A 4D Tensor (batch_size, num_heads, length, attn_size/num_heads)
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            depth = (attn_size // num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(x):
        """Combine tensor that has been split.
        Args:
            x: (batch_size, num_heads, length, attn_size/num_heads)
        Returns:
            A 3D Tensor (batch_size, length, attn_size)
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3]) # --> # [batch, # length, # num_heads, # depth]
            return tf.reshape(x, [batch_size, length, attn_size])

    # (batch_size, max_length, attn_size)
    Q = tf.layers.dense(query, attn_size, use_bias=False)
    K = tf.layers.dense(key, attn_size, use_bias=False)
    V = tf.layers.dense(key, attn_size, use_bias=False)

    if cache is not None:
        # Combine cached keys and values with new keys and values.
        K = tf.concat([cache["K"], K], axis=1)
        V = tf.concat([cache["V"], V], axis=1)

        # Update cache
        cache["K"] = K
        cache["V"] = V

    # (batch_size, num_heads, max_length, attn_size/num_heads)
    Q_ = split_heads(Q)
    K_ = split_heads(K)
    V_ = split_heads(V)

    # Scale Q down before taking a large dot product.
    Q_ *= ((attn_size // num_heads) ** -0.5)

    # (batch_size, num_heads, max_length, max_length)
    outputs = tf.matmul(Q_, K_, transpose_b=True)
    outputs += attn_bias
    outputs = tf.nn.softmax(outputs)

    # Dropout
    outputs = tf.nn.dropout(outputs, dropout_rate)

    # Weighted sum
    outputs = tf.matmul(outputs, V_) # (batch_size, num_heads, max_length, attn_size/num_heads)

    # Recombine heads
    outputs = combine_heads(outputs)

    # Final linear projection
    outputs = tf.layers.dense(outputs, attn_size, use_bias=False)

    return outputs

def feedforward(inputs, ff_size, output_size, padding=None, dropout=1):
    ''' Feed forward net implemented as two dense layers.
    Optionally removes padding before the layers and adds them back in after.

    Args:
        inputs: (batch_size, max_length, hidden_size)
        ff_size: size of feed forward layer
        output_size: size of layer output
        padding: padding mask. [batch_size, max_length]
    Returns:
        A 3D tensor with same shape/type as inputs
    '''

    # Retrieve dynamically known shapes
    batch_size = tf.shape(inputs)[0]
    max_length = tf.shape(inputs)[1]
    hidden_size = inputs.get_shape().as_list()[-1]

    if padding is not None:
        with tf.name_scope("remove_padding"):
            # Flatten padding to [batch_size*max_length]
            pad_mask = tf.reshape(padding, [-1])

            nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

            # Reshape to [batch_size*max_length, hidden_size] to remove padding
            inputs = tf.reshape(inputs, [-1, hidden_size])
            inputs = tf.gather_nd(inputs, indices=nonpad_ids)

            # Reshape from 2 dimensions to 3 dimensions.
            inputs.set_shape([None, hidden_size])
            inputs = tf.expand_dims(inputs, axis=0)

    output = tf.layers.dense(inputs, ff_size, activation=tf.nn.relu)
    output = tf.nn.dropout(output, dropout)
    output = tf.layers.dense(output, output_size)

    if padding is not None:
        with tf.name_scope("re_add_padding"):
            output = tf.squeeze(output, axis=0)
            output = tf.scatter_nd(indices=nonpad_ids, updates=output,
                                   shape=[batch_size * max_length, hidden_size])
            output = tf.reshape(output, [batch_size, max_length, hidden_size])

    return output

def normalize(inputs, scope='normalize'):
    '''Applies layer normalization.
    Returns: A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope):
        return tf.contrib.layers.layer_norm(inputs)

def get_padding_mask(lengths, max_length):
    ''' Create padding mask.
    Args:
        lengths: A 1D tensor (batch_size) indicating how long is each input.

    Returns:
        A 2D float tensor (batch_size, max_length) where 0 is real input and 1
        is padding.
    '''
    with tf.variable_scope('padding'):
        padding_mask = tf.sequence_mask(lengths, max_length, dtype=tf.int32)
    return tf.to_float(1 - padding_mask)

def get_padding_bias(mask):
    ''' Given padding mask, calculates padding bias.
    Args:
        mask: 2D Tensor (batch_size, max_length) where 0 is real input and
        1 is padding,

    Returns:
        Attention bias 4D Tensor (batch_size, 1, 1 length)
    '''
    attention_bias = mask * -1e9
    return tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)

def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder's self attention that maintains model's
    autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections (position i attending to positions > i).
    Args:
        length: int max length of sequences.
    Returns:
        float tensor of shape [1, 1, length, length]
    """
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = -1e9 * (1.0 - valid_locs)
    return decoder_bias

def print_weights(weights):
    size = 0
    def get_size(w):
        shape = w.get_shape()
        if shape.ndims == 2:
            return int(shape[0])*int(shape[1])
        else:
            return int(shape[0])
    for w in weights:
        sz = get_size(w)
        print('weight', w, sz)
        size += sz
    print('total model size', size)
