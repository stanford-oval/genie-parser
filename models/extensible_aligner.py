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
        # tensorflow does not cope well with dictionaries, it needs tuples or namedtuple
        # so we create a namedtuple here one the fly
        self._sequence_tuple_type = namedtuple('GreedyExtensibleDecoderInnerTuple', sequence_keys)

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
            elif isinstance(self._output_layers[key], common.AttentivePointerLayer):
                layer_output_shapes[key] = tf.TensorShape([self._input_max_length])
            else: 
                layer_output_shape = self._output_layers[key]._compute_output_shape(output_shape_with_unknown_batch)
                # now remove the first dimension (batch size)
                layer_output_shapes[key] = nest.map_structure(lambda s: s[1:], layer_output_shape)
            sample_id_shapes[key] = tf.TensorShape([])
        
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._sequence_tuple_type(**layer_output_shapes),
            sample_id=self._sequence_tuple_type(**sample_id_shapes))

    @property
    def output_dtype(self):
        layer_output_dtype = self._sequence_tuple_type(*(tf.float32 for _ in self._sequence_keys))
        sample_id_dtype = self._sequence_tuple_type(*(tf.int32 for _ in self._sequence_keys))
        
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
            
            outputs = BasicDecoderOutput(rnn_output=self._sequence_tuple_type(**decoder_outputs),
                                         sample_id=self._sequence_tuple_type(**sample_ids))
            return (outputs, cell_state, next_inputs, finished)


class ExtensibleGrammarAligner(BaseAligner):
    '''
    A model that implements Seq2Seq with an extensible (pointer-based) grammar:
    that is, it uses a sequence loss during training, and a greedy decoder during inference 
    
    Maybe. Maybe It will use RL-loss during training, and something else during
    inference. It's hard to say
    '''
    
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
                                                               self.config.attention_probability_fn)
        
        
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        
        primary_embedding_matrix = self.output_embed_matrices[self.config.grammar.primary_output]

        output_layers = dict()
        for key, size in self.config.grammar.output_size.items():
            with tf.variable_scope('output_' + key):
                if key == self.config.grammar.primary_output:
                    output_layers[key] = tf.layers.Dense(size, use_bias=False)
                elif self.config.grammar.is_copy_type(key):
                    output_layers[key] = common.AttentivePointerLayer(enc_hidden_states)
                else:
                    output_layers[key] = common.EmbeddingPointerLayer(self.config.decoder_hidden_size/2, self.output_embed_matrices[key],
                                                                      dropout=self.dropout_placeholder)
                output_layers[key].build(tf.TensorShape((None, self.config.decoder_hidden_size)))
        
        final_output = BasicDecoderOutput(sample_id=dict(), rnn_output=dict())
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.primary_output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([primary_embedding_matrix], output_ids_with_go)
            
            if self.config.scheduled_sampling > 0:
                raise NotImplementedError("Scheduled sampling cannot work with the extensible grammar")
            
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
            
            decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer=None)
            decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.config.max_length,
                                                                               swap_memory=True)
            
            for key, layer in output_layers.items():
                # rnn output is batch x time x hiddensize
                # output is batch x time x outputsize
                output = layer(decoder_output.rnn_output)
                final_output.sample_id[key] = None
                final_output.rnn_output[key] = output
            
        else:
            sequence_keys = [self.config.grammar.primary_output] + self.config.grammar.copy_terminal_list + self.config.grammar.extensible_terminal_list
            decoder = GreedyExtensibleDecoder(cell_dec, self.output_embed_matrices, sequence_keys,
                                              output_layers, go_vector, self.config.grammar.end, enc_final_state,
                                              input_max_length=self.config.max_length)
            decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.config.max_length,
                                                                               swap_memory=True)

            # decoder_output is composed of namedtuples, because tensorflow does not like dicts
            # bring it back into dict form
            for key in sequence_keys:
                final_output.rnn_output[key] = getattr(decoder_output.rnn_output, key)
                final_output.sample_id[key] = getattr(decoder_output.sample_id, key)
        
        if self.config.apply_attention:
            # convert alignment history from time-major to batch major
            self.attention_scores = tf.transpose(final_state.alignment_history.stack(), [1, 0, 2])
        return final_output
        
    def finalize_predictions(self, result):
        finalized = dict()
        for key in self.config.output_size:
            # add a dimension of 1 between the batch size and the sequence length to emulate a beam width of 1 
            finalized[key] = tf.expand_dims(result.sample_id[key], axis=1)
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
    
    def _copy_softmax_loss(self, key, logits, gold, mask):
        flat_gold = tf.reshape(gold, (self.batch_size * self.config.max_length,))
        flat_input_token_gold = tf.gather(self.config.grammar.copy_token_to_input_maps[key], flat_gold)
        print('flat_input_token_gold', flat_input_token_gold)
        input_token_gold = tf.reshape(flat_input_token_gold, (self.batch_size, self.config.max_length, 1))
        print('input_token_gold', input_token_gold)
        input_token_equal = tf.equal(tf.expand_dims(self.input_placeholder, axis=1), input_token_gold)
        print('input_token_equal', input_token_equal)
        input_token_equal = tf.cast(input_token_equal, dtype=tf.float32)
        with tf.control_dependencies([tf.assert_positive(tf.reduce_sum(input_token_equal, axis=2), data=(self.input_placeholder, input_token_gold, input_token_equal),
                                                         summarize=1000)]):
            input_token_prob = input_token_equal / tf.reduce_sum(input_token_equal, axis=2, keep_dims=True)
        print('input_token_prob', input_token_prob)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_token_prob, logits=logits)
        print('loss', loss)
        
        mask = tf.cast(mask, dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1) + 1e-6
        element_average_loss = tf.reduce_sum(loss * mask, axis=1) / mask_sum
        print('average_loss', element_average_loss)
        return element_average_loss
    
    def add_loss_op(self, result):
        sequence_keys = [self.config.grammar.primary_output] + self.config.grammar.copy_terminal_list + self.config.grammar.extensible_terminal_list

        all_logits = result.rnn_output
        primary_logits = all_logits[self.config.grammar.primary_output]
        with tf.control_dependencies([tf.assert_positive(tf.shape(primary_logits)[1], data=[tf.shape(primary_logits)])]):
            length_diff = self.config.max_length - tf.shape(primary_logits)[1]

        total_loss = 0
        for key in sequence_keys:
            with tf.name_scope('loss_' + key):
                logits = all_logits[key]
                preds = tf.pad(logits, [[0, 0], [0, length_diff], [0, 0]], mode='constant')
                preds.set_shape((None, self.config.max_length, logits.shape[2]))
        
                if key == self.config.grammar.primary_output:
                    mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.bool)
                else:
                    mask = tf.greater_equal(self.output_placeholders[key], 0)
                # put an arbitrary token in the masked position of the gold, so that we don't exceed the output size
                # and crash the sequence loss computation
                masked_gold = tf.where(mask, self.output_placeholders[key], tf.zeros_like(self.output_placeholders[key]))
                size = self.config.grammar.output_size[key]
        
                if key == self.config.grammar.primary_output:
                    loss = self._max_margin_loss(preds, masked_gold, mask, size)
                elif self.config.grammar.is_copy_type(key):
                    loss = self._copy_softmax_loss(key, preds, masked_gold, mask)
                else:
                    loss = self._sequence_softmax_loss(preds, masked_gold, mask)
                total_loss += tf.reduce_mean(loss, axis=0)

        return total_loss
