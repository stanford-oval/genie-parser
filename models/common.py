# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
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
'''
Created on Dec 22, 2017

@author: gcampagn
'''

import tensorflow as tf
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper

from collections import namedtuple

StackRNNState = namedtuple('StackRNNState', ('hidden_state', 'stacks'))

class StackRNNCell(tf.contrib.rnn.RNNCell):
    '''
    Stack-Augmented RNN Cell, from:
    "Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets"
    '''
    
    def __init__(self, num_units, num_stacks=2, stack_size=50, stack_top_k=2, activation=tf.sigmoid):
        super().__init__()
        
        self._num_units = num_units
        self._num_stacks = num_stacks
        self._stack_size = stack_size
        self._stack_top_k = stack_top_k

        self._action_layers = [tf.layers.Dense(3, activation=tf.nn.softmax, name=('action_layer_%d' % i)) for i in range(self._num_stacks)] # push, pop, nop
        self._push_layers = [tf.layers.Dense(self._num_units, activation=tf.sigmoid, use_bias=False, name=('push_layer_%d' % i)) for i in range(self._num_stacks)]
        self._cell_layer = tf.layers.Dense(self._num_units, activation=activation, name='cell_layer')
        
    def call(self, input, state):
        batch_size = tf.shape(input)[0]
             
        new_stacks = []
        for i in range(self._num_stacks):
            with tf.name_scope('stack_%d' % i):
                action = self._action_layers[i](state.hidden_state)
                push_prob = tf.expand_dims(action[:,0], axis=1)
                pop_prob = tf.expand_dims(action[:,1], axis=1)
                nop_prob = tf.expand_dims(action[:,2], axis=1)
                
                stack = tf.reshape(state.stacks[i], (batch_size, self._stack_size, self._num_units))
                print('stack', stack)
                
                stack_0 = stack[:,0]
                stack_1 = stack[:,1]
                with tf.name_scope('new_stack_0'): 
                    new_stack_0 = tf.expand_dims(push_prob * self._push_layers[i](state.hidden_state) +
                                                 pop_prob * stack_1 + nop_prob * stack_0, axis=1)
                print('new_stack_0', new_stack_0)
                with tf.name_scope('new_stack_i'):
                    stack_push = (stack[:,:-1])
                    print('stack_push', stack_push)
                    stack_pop = tf.concat((stack[:,2:], tf.zeros((batch_size, 1, self._num_units), dtype=tf.float32)), axis=1)
                    print('stack_pop', stack_pop)
                    stack_nop = stack[:,1:]
                    print('stack_nop', stack_nop)
                    
                    new_stack_i = tf.expand_dims(push_prob, axis=1) * stack_push + \
                        tf.expand_dims(pop_prob, axis=1) * stack_pop + \
                        tf.expand_dims(nop_prob, axis=1) * stack_nop
                new_stack = tf.concat((new_stack_0, new_stack_i), axis=1)
                print('new_stack', new_stack)
                new_stacks.append(new_stack)
        
        stack_tops = [tf.reshape(stack[:,:self._stack_top_k], (batch_size, self._stack_top_k * self._num_units)) for stack in new_stacks]
        flat_input = tf.concat([input, state.hidden_state] + stack_tops, axis=1)
        new_hidden_state = self._cell_layer(flat_input)

        return new_hidden_state, StackRNNState(hidden_state=new_hidden_state,
                                               stacks=tuple(tf.reshape(stack, (batch_size, self._stack_size * self._num_units)) for stack in new_stacks))
    
    @property
    def output_size(self):
        return self._num_units
    
    @property
    def state_size(self):
        return StackRNNState(hidden_state=tf.TensorShape((self._num_units,)),
                             stacks=tuple(tf.TensorShape((self._stack_size * self._num_units)) for _ in range(self._num_stacks)))

def make_rnn_cell(cell_type, hidden_size, dropout):
    if cell_type == "lstm":
        cell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
    elif cell_type == "gru":
        cell = tf.contrib.rnn.GRUBlockCellV2(hidden_size)
    elif cell_type == "basic-tanh":
        cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
    elif cell_type == 'stackrnn':
        cell = StackRNNCell(hidden_size)
    else:
        raise ValueError("Invalid RNN Cell type")
    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                         variational_recurrent=True,
                                         output_keep_prob=dropout,
                                         state_keep_prob=dropout if cell_type != 'gru' else 1.0,
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
        self.built = True
    
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            if self._space_transform is not None:
                inputs = tf.tensordot(inputs, self._space_transform, [[len(shape) - 1], [0]])
            outputs = tf.tensordot(inputs, self._against, [[len(shape) - 1], [1]])
            
            # Reshape the output back to the original ndim of the input.
            output_shape = tf.TensorShape(shape[:-1] + [self._output_size])
            assert output_shape.is_compatible_with(outputs.shape)
            outputs.set_shape(output_shape)
            return outputs
        else:
            if self._space_transform is not None:
                inputs = tf.matmul(inputs, self._space_transform)
        
            # input is batch by depth
            # self._against is output by depth
            # result is batch by output
            return tf.matmul(inputs, self._against, transpose_b=True)

    def _compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self._output_size)


class EmbeddingPointerLayer(tf.layers.Layer):
    """
    A pointer layer that chooses from an embedding matrix (of size O x D, where
    D is the depth of the embedding and O the space of choices), using an MLP
    """
    def __init__(self, hidden_size, embeddings, activation=tf.tanh, dropout=1):
        super().__init__()
        
        self._embeddings = embeddings
        self._embedding_size = embeddings.get_shape()[-1]
        self._hidden_size = hidden_size
        self._activation = activation
        self._dropout = dropout
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        
        self.kernel1 = self.add_variable('kernel1', shape=(self._embedding_size, self._hidden_size), dtype=self.dtype)
        self.kernel2 = self.add_variable('kernel2', shape=(input_shape[-1], self._hidden_size), dtype=self.dtype)
        self.bias = self.add_variable('bias', shape=(self._hidden_size,), dtype=self.dtype)
        self.output_projection = self.add_variable('output_projection', shape=(self._hidden_size,), dtype=self.dtype)
        self.built = True
        
    def call(self, inputs):
        with tf.name_scope('EmbeddingPointerLayer', (inputs,)):
            matmul1 = tf.tensordot(self._embeddings, self.kernel1, [[1], [0]])
            
            input_shape = inputs.shape
            matmul2 = tf.tensordot(inputs, self.kernel2, [[input_shape.ndims-1], [0]])
            
            for _ in range(input_shape.ndims-1):
                matmul1 = tf.expand_dims(matmul1, axis=0)
            
            matmul2 = tf.expand_dims(matmul2, axis=input_shape.ndims-1)
            
            neuron_input = matmul1 + matmul2
            neuron_input = tf.nn.bias_add(neuron_input, self.bias)
            activation = self._activation(neuron_input)
            activation = tf.nn.dropout(activation, keep_prob=self._dropout)
            
            scores = tf.tensordot(activation, self.output_projection, [[input_shape.ndims], [0]])
            return scores
    
    @property
    def output_size(self):
        return tf.shape(self._embeddings)[0]


class AttentivePointerLayer(tf.layers.Layer):
    """
    A pointer layer that chooses from the encoding of the inputs, using Luong (multiplicative) Attention
    """
    def __init__(self, enc_hidden_states):
        super().__init__()
        
        self._enc_hidden_states = enc_hidden_states
        self._num_units = enc_hidden_states.shape[-1]
        
    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', (self._num_units, self._num_units), dtype=self.dtype)
        self.built = True
        
    def call(self, inputs):
        with tf.name_scope('AttentivePointerLayer', (inputs,)):
            is_2d = False
            if inputs.shape.ndims < 3:
                is_2d = True
                inputs = tf.expand_dims(inputs, axis=1)
            
            score = tf.matmul(inputs, self._enc_hidden_states, transpose_b=True)
            if is_2d:
                score = tf.squeeze(score, axis=1)
            print('score', score)
            return score


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
        
            enc_final_state = nest.map_structure(lambda x: tf.matmul(x, kernel), enc_final_state)
            enc_hidden_states = tf.tensordot(enc_hidden_states, kernel, [[2], [1]])
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
