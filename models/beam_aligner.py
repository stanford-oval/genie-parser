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

from tensorflow.python.layers import core as tf_core_layers
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper, BeamSearchDecoder

from .base_aligner import BaseAligner

class BeamAligner(BaseAligner):
    '''
    A Beam Search based semantic parser, using beam search for
    both training and inference
    '''
    
    def __init__(self, config):
        super().__init__(config)
        
        if config.beam_size <= 1:
            raise ValueError("Must specify a beam size of more than 1 with seq2seq model")

    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])
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
        
        print(enc_final_state)
        decoder = BeamSearchDecoder(cell_dec, output_embed_matrix, go_vector, self.config.grammar.end,
                                    tf.contrib.seq2seq.tile_batch(enc_final_state, self.config.beam_size),
                                    self.config.beam_size, output_layer=linear_layer)
        
        if self.config.use_grammar_constraints:
            raise NotImplementedError("Grammar constraints are not implemented for the beam search yet")
        
        final_outputs, _, final_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.config.max_length)
        
        # final_outputs.predicted_ids is [batch_size, max_time, beam_width] for some dumb reason
        # transpose it to be [batch_size, beam_width, max_time] which is what makes sense
        predicted_ids = tf.transpose(final_outputs.predicted_ids, [0, 2, 1])
        
        if training:
            return final_outputs.beam_search_decoder_output.scores[:,-1,:], predicted_ids
        else:
            return predicted_ids
        
    
    def add_loss_op(self, preds):
        scores, predicted_ids = preds
        assert predicted_ids.get_shape()[1] == self.config.beam_size
        
        # pad the predictions up to max_length
        assert self.config.grammar.dictionary['<<PAD>>'] == 0
        actual_length = tf.shape(predicted_ids)[1]
        length_diff = tf.reshape(self.config.max_length - actual_length, shape=(1,))
        
        # Padding works as:
        # [before batch, after batch, before beam, after beam, before time, after time]
        padding = tf.reshape(tf.concat([[0, 0, 0, 0, 0], length_diff], axis=0), shape=(3, 2))
        preds = tf.pad(predicted_ids, padding, mode='constant')
        preds.set_shape((None, self.config.beam_size, self.config.max_length))
        
        equal = tf.equal(preds, tf.expand_dims(self.output_placeholder, axis=1))
        pred_compat = tf.reduce_all(equal, axis=2)
        # shape is [batch_size, beam_width]
        assert pred_compat.get_shape()[1] == self.config.beam_size
        
        # normalize pred compat to be a probability distribution
        pred_compat = tf.cast(pred_compat, dtype=tf.float32)
        pred_sum = tf.reduce_sum(pred_compat, axis=1)
        # shape is [batch_size,]
        
        # the loss of each example in the minibatch
        # this is the softmax cross entropy if the good program is in the beam,
        # and an arbitrary high value otherwise
        # in the case we're in the beam, the cross entropy loss will raise the score of the good
        # entry in the beam and lower everything else
        # because the beam is sorted by score, the good entry in the beam will move up
        #
        # if we're out of the beam it does not really matter what we do,
        # the gradient wrt the inputs is 0 and there will be no update
        #
        # the proper way to implement Beam Search Optimization is to use a decoder at training time
        # that recovers the gold sequence if it falls off the beam
        # but we don't implement that yet
        print(scores)
        stochastic_loss = tf.where(pred_sum > 0,
            tf.nn.softmax_cross_entropy_with_logits(labels=(pred_compat / tf.expand_dims(pred_sum, axis=1)), logits=scores),
            tf.ones((self.batch_size,)) * 1e+5)

        return tf.reduce_mean(stochastic_loss)
        
        
        