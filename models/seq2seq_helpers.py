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
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper

from .grammar_decoder import GrammarBasicDecoder
from .config import Config

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

class Seq2SeqDecoder(object):
    def __init__(self, config : Config, input_placeholder, input_length_placeholder, output_placeholder, output_length_placeholder, batch_number_placeholder, max_length=None):
        self.config = config
        self.batch_number_placeholder = batch_number_placeholder
        self.input_placeholder = input_placeholder
        self.input_length_placeholder = input_length_placeholder
        self.output_placeholder = output_placeholder
        self.output_length_placeholder = output_length_placeholder
        self.max_length = max_length or self.config.max_length
        
    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]
    
    def decode(self, cell_dec, enc_final_state, output_size, output_embed_matrix, training, grammar_helper=None):
        if self.config.use_dot_product_output:
            output_layer = DotProductLayer(output_embed_matrix)
        else:
            output_layer = tf.layers.Dense(output_size, use_bias=False)

        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
        else:
            helper = GreedyEmbeddingHelper(output_embed_matrix, go_vector, self.config.grammar.end)
        
        if self.config.use_grammar_constraints:
            decoder = GrammarBasicDecoder(self.config.grammar, cell_dec, helper, enc_final_state, output_layer=output_layer, training_output = self.output_placeholder if training else None,
                                          grammar_helper=grammar_helper)
        else:
            decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer=output_layer)

        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=self.max_length)
        
        return final_outputs
