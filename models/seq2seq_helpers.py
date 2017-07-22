'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf
from tensorflow.python.layers import core as tf_core_layers

from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper, LuongAttention, AttentionWrapper, BeamSearchDecoder

class Seq2SeqDecoder(object):
    def __init__(self, config, input_placeholder, input_length_placeholder, output_placeholder, output_length_placeholder):
        self.config = config
        self.input_placeholder = input_placeholder
        self.input_length_placeholder = input_length_placeholder
        self.output_placeholder = output_placeholder
        self.output_length_placeholder = output_length_placeholder
        
    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]
        
    def decode(self, cell_dec, enc_hidden_states, enc_final_state, output_embed_matrix, training):
        if self.config.apply_attention:
            attention = LuongAttention(self.config.hidden_size, enc_hidden_states, self.input_length_placeholder,
                                       probability_fn=tf.nn.softmax)
            cell_dec = AttentionWrapper(cell_dec, attention,
                                        cell_input_fn=lambda inputs, _: inputs,
                                        attention_layer_size=self.config.hidden_size,
                                        initial_cell_state=enc_final_state)
            enc_final_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        linear_layer = tf_core_layers.Dense(self.config.output_size)

        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
            decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer = linear_layer)
        else:
            if self.config.beam_size > 0:
                decoder = BeamSearchDecoder(cell_dec, output_embed_matrix, go_vector, self.config.grammar.end,
                                            tf.contrib.seq2seq.tile_batch(enc_final_state, self.config.batch_size),
                                            self.config.batch_size)
            else:
                helper = GreedyEmbeddingHelper(output_embed_matrix, go_vector, self.config.grammar.end)
                decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer = linear_layer)

        return tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=self.config.max_length)