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
Sentence encoders.

Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_encoder import BaseEncoder
from . import common

from tensorflow.python.util import nest

class RNNEncoder(BaseEncoder):
    '''
    Use an RNN to encode the sentence
    '''

    def __init__(self, cell_type, num_layers, *args, **kw):
        super().__init__(*args, **kw)
        self._num_layers = num_layers
        self._cell_type = cell_type
    
    def encode(self, inputs, input_length, _parses):
        with tf.name_scope('RNNEncoder'):
            cell_enc = common.make_multi_rnn_cell(self._num_layers, self._cell_type, self.output_size, self._dropout)
            return tf.nn.dynamic_rnn(cell_enc, inputs, sequence_length=input_length,
                                     dtype=tf.float32)


class BiRNNEncoder(RNNEncoder):
    def encode(self, inputs, input_length, _parses):
        with tf.name_scope('BiRNNEncoder'):
            fw_cell_enc = common.make_multi_rnn_cell(self._num_layers, self._cell_type, self.output_size, self._dropout)
            bw_cell_enc = common.make_multi_rnn_cell(self._num_layers, self._cell_type, self.output_size, self._dropout)

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
            layer = tf.layers.Dense(units=self.output_size, activation=tf.tanh)
            enc_hidden_states = layer(inputs)
            enc_final_state = tf.reduce_sum(enc_hidden_states, axis=1)

            #assert enc_hidden_states.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
            if self._cell_type == 'lstm':
                enc_final_state = (tf.contrib.rnn.LSTMStateTuple(enc_final_state, enc_final_state),)

            enc_output = tf.nn.dropout(enc_hidden_states, keep_prob=self._dropout)

            return enc_output, enc_final_state
