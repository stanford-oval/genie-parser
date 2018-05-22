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
from tensorflow.python.util import nest

from .base_model import BaseModel

from .config import Config

class TransformerAligner(BaseModel):

    def build(self):
        self.add_placeholders()
        xavier = tf.contrib.layers.xavier_initializer()

        # Embed the inputs and positionally encode them.
        # shape: (batch_size, max_input_length, embed_size)
        input_embeddings = self.add_input_embeddings(xavier)
        input_embeddings += positional_encoding(self.input_placeholder,
                                                self.config.input_embed_size)
        # Apply dropout immediately before encoding.
        input_embeddings = tf.layers.dropout(input_embeddings,
                                             self.dropout_placeholder)

        # TODO: should we treat ouptut the same as input?
        output_embeddings = self.add_output_embeddings(xavier)
        output_embeddings += positional_encoding(self.input_placeholder,
                                                self.config.output_embed_size)
        # Apply dropout immediately before encoding.
        output_embeddings = tf.layers.dropout(output_embeddings,
                                             self.dropout_placeholder)

        # Encoder block
        # shape (batch_size, max_input_length, hidden_size)
        with tf.variable_scope('encoder', initializer=xavier):
            final_encoder_state = self.add_encoder_op(input_embeddings)

        self.add_output_embeddings(xavier)

        # Decoder block
        with tf.variable_scope('decoder'):
            preds = self.add_decoder_op(self.final_encoder_state, self.output_embeddings)

        self.pred = self.finalize_predictions(preds)

        # TODO What is this decoder action count loss?
        if self.config.decoder_action_count_loss > 0:
            # only consider the first 23 values (excluding any query, action, parameter or filter)
            #count_layer = tf.layers.Dense(self.config.grammar.output_size, name='action_count_layer')
            count_layer = tf.layers.Dense(23, name='action_count_layer')
            action_count_logits = count_layer(tf.concat(nest.flatten(final_encoder_state), axis=1))
            #self.action_counts = action_count_logits > 0
            self.action_counts = None
        else:
            self.action_counts = None

        if self.config.decoder_action_count_loss > 0:
            with tf.name_scope('action_count_loss'):
                binarized_label = tf.cast(self.output_action_counts[:,:23] >= 1, dtype=tf.float32)
                #binarized_predictions = tf.cast(self.action_counts >= 0.5, dtype=tf.float32)
                #action_count_loss = tf.nn.l2_loss(tf.cast(self.output_action_counts, dtype=tf.float32) - self.action_counts)

                action_count_loss = tf.losses.hinge_loss(labels=binarized_label, logits=action_count_logits,
                                                         reduction=tf.losses.Reduction.NONE)
                action_count_loss = tf.reduce_sum(action_count_loss, axis=1)
                action_count_loss = tf.reduce_mean(action_count_loss)
        else:
            action_count_loss = 0

        if self.config.decoder_sequence_loss > 0:
            with tf.name_scope('sequence_loss'):
                sequence_loss = self.add_loss_op(preds)
        else:
            sequence_loss = 0

        with tf.name_scope('training_loss'):
            self.loss = self.config.decoder_action_count_loss * action_count_loss + \
                self.config.decoder_sequence_loss * sequence_loss + \
                self.add_regularization_loss()

        # TODO Need extra eval loss? Why not use feed_dict?
        with tf.name_scope('eval_loss'):
            self.eval_loss = self.config.decoder_action_count_loss * action_count_loss + \
                self.config.decoder_sequence_loss * sequence_loss

        self.train_op = self.add_training_op(self.loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        size = 0
        def get_size(w):
            shape = w.get_shape()
            if shape.ndims == 2:
                return int(shape[0])*int(shape[1])
            else:
                assert shape.ndims == 1
                return int(shape[0])
        for w in weights:
            sz = get_size(w)
            print('weight', w, sz)
            size += sz
        print('total model size', size)

    def add_placeholders(self):
        self.add_input_placeholders()
        self.add_output_placeholders()
        self.add_extra_placeholders()

    def add_input_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='input_sequence')

    def add_output_placeholders(self):
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='output_sequence')
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='output_length')
        self.output_action_counts = tf.placeholder(dtype=tf.int32, shape=(None, self.config.grammar.output_size), name='output_action_counts')

    def add_extra_placeholders(self):
        self.batch_number_placeholder = tf.placeholder(tf.int32, shape=(), name='batch_number')
        self.epoch_placeholder = tf.placeholder(tf.int32, shape=(), name='epoch_number')
        self.dropout_placeholder = tf.placeholder_with_default(0.0, shape=(), name='dropout_probability')

    def create_feed_dict(self, inputs_batch, parses_batch,
                         labels_batch=None, label_length_batch=None,
                         dropout=0, batch_number=0, epoch=0):
        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        if labels_batch is not None:
            feed_dict[self.output_placeholder] = labels_batch
        if label_length_batch is not None:
            feed_dict[self.output_length_placeholder] = label_length_batch
        feed_dict[self.dropout_placeholder] = dropout
        feed_dict[self.batch_number_placeholder] = batch_number
        feed_dict[self.epoch_placeholder] = epoch

        if self.config.decoder_action_count_loss > 0 and labels_batch is not None:
            action_count_batch = np.zeros((len(labels_batch), self.config.grammar.output_size), dtype=np.int32)
            for i in range(len(labels_batch)):
                action_count_batch[i] = np.bincount(labels_batch[i][:label_length_batch[i]],
                                                    minlength=self.config.grammar.output_size)
            feed_dict[self.output_action_counts] = action_count_batch
        return feed_dict

    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]

    def add_input_embeddings(self, xavier):
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

                # shape: (dictionary size, embed_size)
                assert self.input_embed_matrix.get_shape() == shape

        # shape: (batch size, max input length, embed_size)
        inputs = tf.nn.embedding_lookup([self.input_embed_matrix], self.input_placeholder)
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.input_embed_size)

        # now project the input down to a small size
        # input_projection = tf.layers.Dense(units=self.config.input_projection,
        #                                    name='input_projection')
        # return input_projection(inputs)
        return inputs

    def add_output_embeddings(self, xavier):
        with tf.variable_scope('embed', initializer=xavier):
            with tf.variable_scope('output'):
                shape = (self.config.output_size, self.config.output.embed_size)
                init = None

                if self.config.train_output_embeddings:
                    if self.config.output_embedding_matrix is not None:
                        init = tf.constant_initializer(self.config.output_embedding_matrix)

                    self.output_embed_matrix = tf.get_variable('embedding',
                                                              shape=shape,
                                                              initializer=init)
                else:
                    self.output_embed_matrix = tf.constant(self.config.output_embedding_matrix)

                # shape: (dictionary size, embed_size)
                assert self.output_embed_matrix.get_shape() == shape

        # shape: (batch size, max output length, embed_size)
        outputs = tf.nn.embedding_lookup([self.output_embed_matrix], self.output_placeholder)
        assert outputs.get_shape()[1] == (self.config.max_length, self.config.output_embed_size)

        # return output_projection(output)
        return outputs


    def add_encoder_op(self, inputs):
        ''' Adds the encoder block of the Transformer network.
        Args:
            inputs: A 3D tensor with shape (batch_size, max_input_length, embed_size)
        Return:
            A 3D tensor with shape (batch_size, max_input_length, hidden_size)
        '''

        hidden_size = self.config.encoder_hidden_size

        for i in range(self.config.num_encoder_blocks):
            with tf.variable_scope('encoder_block_{}'.format(i)):
                # shape (batch_size, max_input_length, encoder_hidden_size)
                mh_out = multihead_attention(query=inputs, key=inputs,
                                             output_size=hidden_size,
                                             num_heads=self.config.num_heads,
                                             dropout_rate=self.dropout_placeholder,
                                             mask_future=False)

                # Residual connection and normalize
                mh_out += inputs
                mh_out = normalize(mh_out)

                # No change in shape
                ff_out = feedforward(mh_out, 4*hidden_size, hidden_size)

                # Residual connection and normalize
                ff_out += mh_out
                inputs = normalize(ff_out)

        return inputs

    def add_output_embeddings(self, xavier):
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

    def add_decoder_op(self, enc_final_state, dec_state):
        ''' Adds the decoder op.

        The decoder takes as inputs:
            encoder final state (batch_size, max_input_length, encoder_hidden_size)
            decoder initial embeddings (batch_size, max_input_length, output_embed_size)

        The start token is self.config.grammar.start
        The end token is self.config.grammar.end

        Returns final_outputs (final_outputs, final_state, final_sequence_lengths)
        '''

        # TODO do decoder based on how the evaluation is set up

        # go_vec = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        # output_embed_matrix = self.output_embed_matrices[self.config.grammar.primary_output]
        # output_embed_size = output_embed_matrix.shape[-1]
        # primary_output_size = self.config.grammar.output_size[self.config.grammar.primary_output]
        # output_ids_with_go = tf.concat([tf.expand_dims(go_vec, axis=1), self.primary_output_placeholder], axis=1)
        
        # TODO: figure out if hidden_size is correct
        hidden_size = self.config.decoder_hidden_size
        dec_state = dec_initial_state

        for i in range(self.config.num_decoder_blocks):
            with tf.variable_scope('decoder_block_{}'.format(i)):
                # shape (batch_size, max_input_length, encoder_hidden_size)
                mh_out = multihead_attention(query=dec_state, key=dec_state,
                                             output_size=hidden_size,
                                             num_heads=self.config.num_heads,
                                             dropout_rate=self.dropout_placeholder,
                                             mask_future=False)

                # Residual connection and normalize
                mh_out += dec_state
                dec_state = normalize(mh_out)

                mh_out = multihead_attention(query=dec_state, key=enc_final_state,
                                             output_size=hidden_size,
                                             num_heads=self.config.num_heads,
                                             dropout_rate=self.dropout_placeholder,
                                             mask_future=False)

                # Residual connection and normalize
                mh_out += dec_state
                mh_out = normalize(mh_out)

                # No change in shape
                ff_out = feedforward(mh_out, 4*hidden_size, hidden_size)

                # Residual connection and normalize
                ff_out += mh_out
                dec_state = normalize(ff_out)

        params = {"inputs": dec_state, "filters": self.config.grammar.output_size, "kernel_size": 1,
                  "activation": tf.nn.softmax, "use_bias": True}
        dec_state = tf.layers.conv1d(**params)

        return dec_state

        # if training:
        #     outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
        #     # FIXME

        # else:
        #     # FIXME

        # if not training:
        #     # self.attention_scores = FIXME

        # return final_outputs

    def finalize_predictions(self, preds):
        raise NotImplementedError()

    def add_loss_op(self, preds, training = True):
        raise NotImplementedError()

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


def positional_encoding(inputs, num_units, scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (batch size, max input length).
      num_units: Output dimensionality
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    batch_size, max_input_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(max_input_length), 0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(max_input_length)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        return outputs

def multihead_attention(query, key, attn_size=None, num_heads=8,
                        dropout_rate=0, mask_future=False,
                        scope='multihead_attention', reuse=None):
    '''
    Args:
        query: A 3D tensor with shape (N, T_q, C_q)
        key: A 3D tensor with shape (N, T_k, C_k)
        attn_size: attention size and also final output size
        num_heads: number of attention layers
        dropout_rate: rate of dropout during attention
        mask_future: whether to mask input from future time steps for decoding
        scope: Optional scope for 'variable_scope'
        reuse: Whether to reuse the weights of the same scope
    Returns
        A 3D tensor with shape (N, T_q, attn_size)
    '''

    with tf.variable_scope(scope, reuse=reuse):
        # (N, {T_q, T_k}, attn_size)
        Q = tf.layers.dense(query, attn_size, activation=tf.nn.relu)
        K = tf.layers.dense(key, attn_size, activation=tf.nn.relu)
        V = tf.layers.dense(key, attn_size, activation=tf.nn.relu)

        # (num_heads*N, {T_q, T_k}, attn_size/num_heads)
        # each N is one attention layer / "head"
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # (num_heads*N, T_q, T_k)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Mask keys
        key_masks = tf.sign(tf.abs(tf.reduce_sum(key, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (num_heads*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(query)[1], 1])
        # shape (num_heads*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Mask future time steps
        if mask_future:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            # shape (num_heads*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (num_heads*N, T_q, T_k)

        outputs = tf.nn.softmax(outputs) # (num_heads*N, T_q, T_k)

        # Mask query
        query_masks = tf.sign(tf.abs(tf.reduce_sum(query, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (num_heads*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        # shape (num_heads*N, T_q, T_k)

        outputs *= query_masks # (N, T_q, T_k) because of broadcasting

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate)

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # (num_heads*N, T_q, attn_size/num_heads)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    return outputs


def feedforward(inputs, ff_size, output_size, scope='feedforward', reuse=None):
    ''' Point-wise feed forward net implemented as two 1xd convolutions.

    Args:
        inputs: A 3D tensor with shape (batch_size, max_input_length, _)
        ff_size: size of feed forward layer
        output_size: size of layer output
        scope: Optional scope for 'variable_scope'
        reuse: Whether to reuse the weights of the same scope
    Returns:
        A 3D tensor with same shape/type as inputs
    '''

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": ff_size, "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": output_size, "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

    return outputs

def normalize(inputs, epsilon = 1e-8, scope="ln", reuse=None):
    '''Applies layer normalization.

    Args:
        inputs: A tensor with 2 or more dimensions
        epsilon: a very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: whether to reuse the weights of the same layer
    Returns: A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs
