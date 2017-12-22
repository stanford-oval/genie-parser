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
Created on Dec 22, 2017

@author: gcampagn
'''

import tensorflow as tf
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper

def make_rnn_cell(cell_type, hidden_size, dropout):
    if cell_type == "lstm":
        cell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
    elif cell_type == "gru":
        cell = tf.contrib.rnn.GRUBlockCellV2(hidden_size)
    elif cell_type == "basic-tanh":
        cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
    else:
        raise ValueError("Invalid RNN Cell type")
    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                         variational_recurrent=True,
                                         output_keep_prob=dropout,
                                         state_keep_prob=dropout,
                                         dtype=tf.float32)
    return cell

def make_multi_rnn_cell(num_layers, cell_type, hidden_size, dropout):
    return tf.contrib.rnn.MultiRNNCell([make_rnn_cell(cell_type, hidden_size, dropout) for _ in range(num_layers)])

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
    
def unify_encoder_decoder(cell_dec, enc_hidden_states, enc_final_state):
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
    
    return enc_hidden_states, enc_final_state

def apply_attention(cell_dec, enc_hidden_states, enc_final_state, input_length, batch_size, attention_probability_fn):
    if attention_probability_fn == 'softmax':
        probability_fn = tf.nn.softmax
        score_mask_value = float('-inf')
    elif attention_probability_fn == 'hardmax':
        probability_fn = tf.contrib.seq2seq.hardmax
        score_mask_value = float('-inf')
    elif attention_probability_fn == 'sparsemax':
        def sparsemax(attentionscores):
            attentionscores = tf.contrib.sparsemax.sparsemax(attentionscores)
            with tf.control_dependencies([tf.assert_non_negative(attentionscores),
                                          tf.assert_less_equal(attentionscores, 1., summarize=60)]):
                return tf.identity(attentionscores)
        probability_fn = sparsemax
        # sparsemax does not deal with -inf properly, and has significant numerical stability issues
        # with large numbers (positive or negative)
        score_mask_value = -1e+5
    else:
        raise ValueError("Invalid attention_probability_fn " + str(attention_probability_fn))
    
    with tf.variable_scope('attention', initializer=tf.initializers.identity(dtype=tf.float32)):
        attention = LuongAttention(int(cell_dec.output_size), enc_hidden_states,
                                   memory_sequence_length=input_length,
                                   probability_fn=probability_fn,
                                   score_mask_value=score_mask_value)
    cell_dec = AttentionWrapper(cell_dec, attention,
                                cell_input_fn=lambda inputs, _: inputs,
                                attention_layer_size=int(cell_dec.output_size),
                                alignment_history=True,
                                initial_cell_state=enc_final_state)
    enc_final_state = cell_dec.zero_state(batch_size, dtype=tf.float32)
    
    return cell_dec, enc_final_state