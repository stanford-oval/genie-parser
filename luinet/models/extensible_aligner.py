# Copyright 2018 The Board of Trustees of the Leland Stanford Junior University
# 
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
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
from setuptools.dist import sequence
'''
Created on Jan 31, 2018

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from . import common

from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BasicDecoder, \
    Helper, Decoder, TrainingHelper, GreedyEmbeddingHelper, \
    BasicDecoderOutput

from collections import namedtuple

# Part of this code was copied from Tensorflow
#
# Copyright 2016 The TensorFlow Authors.
#
# The original code was licensed under the Apache 2.0 license.

class PrimarySequenceTrainingHelper(TrainingHelper):
    def __init__(self, inputs, sequence_length, time_major=False, name=None, primary_sequence=None):
        super().__init__(inputs, sequence_length, time_major, name)

        self._primary_sequence = primary_sequence

    def sample(self, time, outputs, name=None, **unused_kwargs):
        return tf.argmax(outputs[self._primary_sequence], axis=-1, output_type=tf.int32)

class GreedyExtensibleDecoder(Decoder):
    """Sampling decoder that produces multiple output sequences."""

    def __init__(self, cell, embeddings, sequence_keys, output_layers, start_tokens, end_token,
                 initial_state, input_max_length):
        self._cell = cell
        self._initial_state = initial_state
        
        self._embeddings = embeddings
        self._output_layers = output_layers
        self._input_max_length = input_max_length
        self._primary_sequence = sequence_keys[0]
        self._sequence_keys = sequence_keys

        self._start_tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embed(self._start_tokens)

    def _embed(self, what):
        return tf.nn.embedding_lookup(self._embeddings[self._primary_sequence], what)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def output_size(self):
        size = self._cell.output_size
        # To use layer's compute_output_shape, we need to convert the
        # RNNCell's output_size entries into shapes with an unknown
        # batch size.  We then pass this through the layer's
        # compute_output_shape and read off all but the first (batch)
        # dimensions to get the output size of the rnn with the layer
        # applied to the top.
        output_shape_with_unknown_batch = nest.map_structure(
            lambda s: tf.TensorShape([None]).concatenate(s),
            size)
        
        layer_output_shapes = dict()
        sample_id_shapes = dict()
        for key in self._sequence_keys:
            if isinstance(self._output_layers[key], common.EmbeddingPointerLayer):
                layer_output_shapes[key] = tf.expand_dims(self._output_layers[key].output_size, axis=0)
            elif isinstance(self._output_layers[key], (common.AttentivePointerLayer, common.DNNPointerLayer, common.ImprovedAttentivePointerLayer)):
                layer_output_shapes[key] = tf.TensorShape([self._input_max_length])
            else: 
                layer_output_shape = self._output_layers[key].compute_output_shape(output_shape_with_unknown_batch)
                # now remove the first dimension (batch size)
                layer_output_shapes[key] = nest.map_structure(lambda s: s[1:], layer_output_shape)
            sample_id_shapes[key] = tf.TensorShape([])
        
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=layer_output_shapes,
            sample_id=sample_id_shapes)

    @property
    def output_dtype(self):
        layer_output_dtype = dict()
        sample_id_dtype = dict()
        for key in self._sequence_keys:
            layer_output_dtype[key] = tf.float32
            sample_id_dtype[key] = tf.int32
        
        return BasicDecoderOutput(rnn_output=layer_output_dtype, sample_id=sample_id_dtype)

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs, self._initial_state,)

    def step(self, time, inputs, state, name=None):
        with tf.name_scope(name, "GreedyExtensibleDecoderStep", (time, inputs, state)):
            rnn_outputs, cell_state = self._cell(inputs, state)
            
            decoder_outputs = dict()
            sample_ids = dict()
            for key in self._sequence_keys:
                with tf.name_scope('decode_' + key):
                    decoder_outputs[key] = self._output_layers[key](rnn_outputs)
                    sample_ids[key] = tf.argmax(decoder_outputs[key], axis=-1, output_type=tf.int32)

            primary_sample_id = sample_ids[self._primary_sequence]
            with tf.name_scope('check_finished'):
                finished = tf.equal(primary_sample_id, self._end_token)
                all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: self._embed(primary_sample_id))
            
            outputs = BasicDecoderOutput(rnn_output=decoder_outputs,
                                         sample_id=sample_ids)
            return (outputs, cell_state, next_inputs, finished)


class TupleOutputLayer(tf.layers.Layer):
    def __init__(self, layers, sequence_keys, input_max_length):
        super().__init__()

        self._layers = layers
        self._sequence_keys = sequence_keys
        self._input_max_length = input_max_length

    def call(self, inputs):
        outputs = dict()
        for key in self._sequence_keys:
            outputs[key] = self._layers[key](inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        layer_output_shapes = dict()
        for key in self._sequence_keys:
            if isinstance(self._layers[key], common.EmbeddingPointerLayer):
                layer_output_shapes[key] = tf.TensorShape([None, self._layers[key].output_size])
            elif isinstance(self._layers[key], (common.AttentivePointerLayer, common.DNNPointerLayer, common.ImprovedAttentivePointerLayer)):
                layer_output_shapes[key] = tf.TensorShape([None, self._input_max_length])
            else: 
                layer_output_shapes[key] = self._layers[key].compute_output_shape(input_shape)
        return layer_output_shapes


class ExtensibleGrammarAligner(BaseAligner):
    '''
    A model that implements Seq2Seq with an extensible (pointer-based) grammar:
    that is, it uses a sequence loss during training, and a greedy decoder during inference 
    
    Maybe. Maybe It will use RL-loss during training, and something else during
    inference. It's hard to say
    '''

    def __init__(self, config):
        super().__init__(config)

    def add_decoder_op(self, enc_final_state, enc_hidden_states, training):
        cell_dec = common.make_multi_rnn_cell(self.config.rnn_layers, self.config.rnn_cell_type,
                                              self.config.decoder_hidden_size,
                                              self.dropout_placeholder)
        enc_hidden_states, enc_final_state = common.unify_encoder_decoder(cell_dec,
                                                                          enc_hidden_states,
                                                                          enc_final_state)
        
        #if self.config.connect_output_decoder:
        #    cell_dec = common.ParentFeedingCellWrapper(cell_dec, enc_final_state)
        #else:
        #    cell_dec = common.InputIgnoringCellWrapper(cell_dec, enc_final_state)
        if self.config.apply_attention:
            cell_dec, enc_final_state = common.apply_attention(cell_dec,
                                                               enc_hidden_states,
                                                               enc_final_state,
                                                               self.input_length_placeholder,
                                                               self.batch_size,
                                                               self.config.attention_probability_fn,
                                                               dropout=self.dropout_placeholder)
        
        
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        
        primary_embedding_matrix = self.output_embed_matrices[self.config.grammar.primary_output]

        output_layers = dict()
        for key, size in self.config.grammar.output_size.items():
            with tf.variable_scope('output_' + key):
                if key == self.config.grammar.primary_output:
                    output_layers[key] = tf.layers.Dense(size, use_bias=False)
                elif self.config.grammar.is_copy_type(key):
                    output_layers[key] = common.DNNPointerLayer(enc_hidden_states) #common.ImprovedAttentivePointerLayer(enc_hidden_states)  #common.AttentivePointerLayer(enc_hidden_states)
                else:
                    output_layers[key] = common.EmbeddingPointerLayer(self.config.decoder_hidden_size/2, self.output_embed_matrices[key],
                                                                      dropout=self.dropout_placeholder)
                output_layers[key].build(tf.TensorShape((None, self.config.decoder_hidden_size)))
        
        sequence_keys = list(self.config.grammar.output_size.keys())
        assert sequence_keys[0] == self.config.grammar.primary_output

        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.primary_output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([primary_embedding_matrix], output_ids_with_go)
            tuple_layer = TupleOutputLayer(output_layers, sequence_keys, self.config.max_length)
            
            if self.config.scheduled_sampling > 0:
                raise NotImplementedError("Scheduled sampling cannot work with the extensible grammar")
            
            helper = PrimarySequenceTrainingHelper(outputs, self.output_length_placeholder+1,
                                                   primary_sequence=self.config.grammar.primary_output)
            
            decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer=tuple_layer)
            decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.config.max_length,
                                                                               swap_memory=True)

            final_output = BasicDecoderOutput(sample_id=dict(), rnn_output=dict())
            for key in sequence_keys:
                final_output.sample_id[key] = None
                final_output.rnn_output[key] = decoder_output.rnn_output[key]
            
        else:
            decoder = GreedyExtensibleDecoder(cell_dec, self.output_embed_matrices, sequence_keys,
                                              output_layers, go_vector, self.config.grammar.end, enc_final_state,
                                              input_max_length=self.config.max_length)
            final_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.config.max_length,
                                                                               swap_memory=True)

        if self.config.apply_attention:
            # convert alignment history from time-major to batch major
            self.attention_scores = tf.transpose(final_state.alignment_history.stack(), [1, 0, 2])
        return final_output
        
    def finalize_predictions(self, result):
        finalized = dict()
        for key in self.config.output_size:
            with tf.name_scope('finalize_' + key):
                finalized[key] = result.sample_id[key]
                # add a dimension of 1 between the batch size and the sequence length to emulate a beam width of 1 
                finalized[key] = tf.expand_dims(finalized[key], axis=1)
        return finalized
    
    def _max_margin_loss(self, logits, gold, mask, size):
        # add epsilon to avoid division by 0
        logits = logits + 1e-5
        
        # as in Crammer and Singer, "On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines"

        flat_mask = tf.reshape(tf.cast(mask, tf.float32), (self.batch_size * self.config.max_length,))
        flat_preds = tf.reshape(logits, (self.batch_size * self.config.max_length, size))
        flat_gold = tf.reshape(gold, (self.batch_size * self.config.max_length,))

        flat_indices = tf.range(self.batch_size * self.config.max_length, dtype=tf.int32)
        flat_gold_indices = tf.stack((flat_indices, flat_gold), axis=1)

        one_hot_gold = tf.one_hot(gold, depth=size, dtype=tf.float32)
        marginal_scores = logits - one_hot_gold + 1

        marginal_scores = tf.reshape(marginal_scores, (self.batch_size * self.config.max_length, size))
        max_margin = tf.reduce_max(marginal_scores, axis=1)

        gold_score = tf.gather_nd(flat_preds, flat_gold_indices)
        margin = max_margin - gold_score

        margin = margin * flat_mask

        margin = tf.reshape(margin, (self.batch_size, self.config.max_length))
        return tf.reduce_sum(margin, axis=1) / tf.cast(self.output_length_placeholder, dtype=tf.float32)
    
    def _sequence_softmax_loss(self, logits, gold, mask):
        return tf.contrib.seq2seq.sequence_loss(logits, gold, tf.cast(mask, tf.float32), average_across_batch=False)
    
    def add_loss_op(self, result):
        sequence_keys = list(self.config.grammar.output_size.keys())

        all_logits = result.rnn_output
        primary_logits = all_logits[self.config.grammar.primary_output]
        with tf.control_dependencies([tf.assert_positive(tf.shape(primary_logits)[1], data=[tf.shape(primary_logits)])]):
            length_diff = self.config.max_length - tf.shape(primary_logits)[1]

        total_loss = 0
        sequence_mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.bool)
        padding = tf.convert_to_tensor([[0, 0], [0, length_diff], [0, 0]], dtype=tf.int32, name='padding')
        for key in sequence_keys:
            with tf.name_scope('loss_' + key):
                logits = all_logits[key]
                preds = tf.pad(logits, padding, mode='constant')
                preds.set_shape((None, self.config.max_length, logits.shape[2]))
        
                if key == self.config.grammar.primary_output:
                    mask = sequence_mask
                    masked_gold = self.output_placeholders[key]
                else:
                    mask = tf.greater_equal(self.output_placeholders[key], 0)
                    # put an arbitrary token in the masked position of the gold, so that we don't exceed the output size
                    # and crash the sequence loss computation
                    masked_gold = tf.where(mask, self.output_placeholders[key], tf.zeros_like(self.output_placeholders[key]))
                size = self.config.grammar.output_size[key]
        
                if key == self.config.grammar.primary_output:
                    loss = 20 * self._max_margin_loss(preds, masked_gold, mask, size)
                else:
                    loss = self._sequence_softmax_loss(preds, masked_gold, mask)
                total_loss += loss

        return tf.reduce_mean(total_loss, axis=0)
