'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf
from tensorflow.python.layers import core as tf_core_layers

from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper

from .grammar_decoder import GrammarBasicDecoder
from .config import Config


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
        linear_layer = tf_core_layers.Dense(output_size, use_bias=False)

        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
        else:
            helper = GreedyEmbeddingHelper(output_embed_matrix, go_vector, self.config.grammar.end)
        
        if self.config.use_grammar_constraints:
            decoder = GrammarBasicDecoder(self.config.grammar, cell_dec, helper, enc_final_state, output_layer = linear_layer, training_output = self.output_placeholder if training else None,
                                          grammar_helper=grammar_helper)
        else:
            decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer = linear_layer)

        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=self.max_length)
        
        return final_outputs
