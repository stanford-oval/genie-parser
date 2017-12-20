# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner

from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper
from tensorflow.python.util import nest

class DotProductLayer(tf.layers.Layer):
    def __init__(self, against):
        super().__init__()
        self._against = against
        self._depth_size = self._against.get_shape()[1]
        self._output_size = self._against.get_shape()[0]
    
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError("Input to DotProductLayer must have the last dimension defined")
        if input_shape[-1].value != self._depth_size:
            self._space_transform = self.add_variable('kernel',
                                                      shape=(input_shape[-1].value, self._depth_size),
                                                      dtype=self.dtype,
                                                      trainable=True)
        else:
            self._space_transform = None
    
    def call(self, input):
        if self._space_transform:
            input = tf.matmul(input, self._space_transform)
        
        # input is batch by depth
        # self._against is output by depth
        # result is batch by output
        return tf.matmul(input, self._against, transpose_b=True)

    def _compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self._output_size)

def pad_up_to(vector, size):
    rank = vector.get_shape().ndims - 1
    
    length_diff = tf.reshape(size - tf.shape(vector)[1], shape=(1,))
    with tf.control_dependencies([tf.assert_non_negative(length_diff, data=(vector, size, tf.shape(vector)))]):
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0,0]*(rank-1)], axis=0), shape=((rank+1), 2))
        return tf.pad(vector, padding, mode='constant')

class ParentFeedingCellWrapper(tf.contrib.rnn.RNNCell):
    '''
    A cell wrapper that concatenates a fixed Tensor to the input
    before calling the wrapped cell
    '''
    def __init__(self, wrapped : tf.contrib.rnn.RNNCell, parent_state):
        super().__init__()
        self._wrapped = wrapped
        self._flat_parent_state = tf.concat(nest.flatten(parent_state), axis=1)
        
    def call(self, input, state):
        concat_input = tf.concat((self._flat_parent_state, input), axis=1)
        return self._wrapped.call(concat_input, state)
    
    @property
    def output_size(self):
        return self._wrapped.output_size
    
    @property
    def state_size(self):
        return self._wrapped.state_size

class InputIgnoringCellWrapper(tf.contrib.rnn.RNNCell):
    '''
    A cell wrapper that replaces the cell input with a fixed Tensor
    and ignores whatever input is passed in
    '''
    def __init__(self, wrapped : tf.contrib.rnn.RNNCell, constant_input):
        super().__init__()
        self._wrapped = wrapped
        self._flat_constant_input = tf.concat(nest.flatten(constant_input), axis=1)
        
    def call(self, input, state):
        return self._wrapped.call(self._flat_constant_input, state)
    
    @property
    def output_size(self):
        return self._wrapped.output_size
    
    @property
    def state_size(self):
        return self._wrapped.state_size

class Seq2SeqAligner(BaseAligner):
    '''
    A model that implements Seq2Seq: that is, it uses a sequence loss
    during training, and a greedy decoder during inference 
    '''
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i, True) for i in range(self.config.rnn_layers)])
        
        encoder_hidden_size = int(enc_hidden_states.get_shape()[-1])
        decoder_hidden_size = int(cell_dec.output_size)
        
        # if encoder and decoder have different sizes, add a projection layer
        if encoder_hidden_size != decoder_hidden_size:
            assert False, (encoder_hidden_size, decoder_hidden_size)
            with tf.variable_scope('hidden_projection'):
                kernel = tf.get_variable('kernel', (encoder_hidden_size, decoder_hidden_size), dtype=tf.float32)
            
                # apply a relu to the projection for good measure
                enc_final_state = nest.map_structure(lambda x: tf.nn.relu(tf.matmul(x, kernel)), enc_final_state)
                enc_hidden_states = tf.nn.relu(tf.tensordot(enc_hidden_states, kernel, [[2], [1]]))
        else:
            # flatten and repack the state
            enc_final_state = nest.pack_sequence_as(cell_dec.state_size, nest.flatten(enc_final_state))
        
        if self.config.connect_output_decoder:
            cell_dec = ParentFeedingCellWrapper(cell_dec, enc_final_state)
        else:
            cell_dec = InputIgnoringCellWrapper(cell_dec, enc_final_state)
        if self.config.apply_attention:
            attention = LuongAttention(self.config.decoder_hidden_size, enc_hidden_states, self.input_length_placeholder,
                                       probability_fn=tf.nn.softmax)
            cell_dec = AttentionWrapper(cell_dec, attention,
                                        cell_input_fn=lambda inputs, _: inputs,
                                        attention_layer_size=self.config.decoder_hidden_size,
                                        alignment_history=True,
                                        initial_cell_state=enc_final_state)
            enc_final_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
        else:
            helper = GreedyEmbeddingHelper(output_embed_matrix, go_vector, self.config.grammar.end)
        
        if self.config.use_dot_product_output:
            output_layer = DotProductLayer(output_embed_matrix)
        else:
            output_layer = tf.layers.Dense(self.config.grammar.output_size, use_bias=False)
        
        decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer=output_layer)
        final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.config.max_length,
                                                                          swap_memory=True)
        if self.config.apply_attention:
            # convert alignment history from time-major to batch major
            self.attention_scores = tf.transpose(final_state.alignment_history.stack(), [1, 0, 2])
        return final_outputs
        
    def finalize_predictions(self, preds):
        # add a dimension of 1 between the batch size and the sequence length to emulate a beam width of 1 
        return tf.expand_dims(preds.sample_id, axis=1)
    
    def add_loss_op(self, result):
        logits = result.rnn_output
        with tf.control_dependencies([tf.assert_positive(tf.shape(logits)[1], data=[tf.shape(logits)])]):
            length_diff = tf.reshape(self.config.max_length - tf.shape(logits)[1], shape=(1,))
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0, 0]], axis=0), shape=(3, 2))
        preds = tf.pad(logits, padding, mode='constant')
        
        # add epsilon to avoid division by 0
        preds = preds + 1e-5

        mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(preds, self.output_placeholder, mask)

        with tf.control_dependencies([tf.assert_non_negative(loss, data=[preds, mask], summarize=256*60*300)]):
            return tf.identity(loss)
