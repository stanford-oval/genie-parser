'''
Sentence encoders.

Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_encoder import BaseEncoder

from tensorflow.python.util import nest

class RNNEncoder(BaseEncoder):
    '''
    Use an RNN to encode the sentence
    '''

    def __init__(self, cell_type, num_layers, *args, **kw):
        super().__init__(*args, **kw)
        self._num_layers = num_layers
        self._cell_type = cell_type
    
    def _make_rnn_cell(self, i):
        if self._cell_type == "lstm":
            cell = tf.contrib.rnn.LSTMCell(self.output_size)
        elif self._cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell(self.output_size)
        elif self._cell_type == "basic-tanh":
            cell = tf.contrib.rnn.BasicRNNCell(self.output_size)
        else:
            raise ValueError("Invalid RNN Cell type")
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._dropout, seed=8 + 33 * i)
        return cell
    
    def encode(self, inputs, input_length, _parses):
        with tf.name_scope('LSTMEncoder'):
            cell_enc = tf.contrib.rnn.MultiRNNCell([self._make_rnn_cell(i) for i in range(self._num_layers)])

            return tf.nn.dynamic_rnn(cell_enc, inputs, sequence_length=input_length,
                                     dtype=tf.float32)


class BiRNNEncoder(RNNEncoder):
    def encode(self, inputs, input_length, _parses):
        with tf.name_scope('BiLSTMEncoder'):
            fw_cell_enc = tf.contrib.rnn.MultiRNNCell([self._make_rnn_cell(i) for i in range(self._num_layers)])
            bw_cell_enc = tf.contrib.rnn.MultiRNNCell([self._make_rnn_cell(i) for i in range(self._num_layers)])

            outputs, output_state = tf.nn.bidirectional_dynamic_rnn(fw_cell_enc, bw_cell_enc, inputs, input_length,
                                                                    dtype=tf.float32)

            fw_output_state, bw_output_state = output_state
            # concat each element of the final state, so that we're compatible with a unidirectional
            # decoder
            output_state = nest.pack_sequence_as(fw_output_state, [tf.concat((x, y), axis=1) for x, y in zip(nest.flatten(fw_output_state), nest.flatten(bw_output_state))])

            return tf.concat(outputs, axis=2), output_state


class BagOfWordsEncoder(BaseEncoder):
    '''
    Use a bag of words model to encode the sentence
    '''
    
    def __init__(self, cell_type, *args, **kw):
        super().__init__(*args, **kw)
        self._cell_type = cell_type
    
    def encode(self, inputs, _input_length, _parses):
        with tf.variable_scope('BagOfWordsEncoder'):
            W = tf.get_variable('W', (self.embed_size, self.output_size))
            b = tf.get_variable('b', shape=(self.output_size,), initializer=tf.constant_initializer(0, tf.float32))

            enc_hidden_states = tf.tanh(tf.tensordot(inputs, W, [[2], [0]]) + b)
            enc_final_state = tf.reduce_sum(enc_hidden_states, axis=1)

            #assert enc_hidden_states.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
            if self._cell_type == 'lstm':
                enc_final_state = (tf.contrib.rnn.LSTMStateTuple(enc_final_state, enc_final_state),)

            enc_output = tf.nn.dropout(enc_hidden_states, keep_prob=self._dropout, seed=12345)

            return enc_output, enc_final_state
