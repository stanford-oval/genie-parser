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
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from . import common

from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper, ScheduledEmbeddingTrainingHelper

class Seq2SeqAligner(BaseAligner):
    '''
    A model that implements Seq2Seq: that is, it uses a sequence loss
    during training, and a greedy decoder during inference 
    '''
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, training):
        cell_dec = common.make_multi_rnn_cell(self.config.rnn_layers, self.config.rnn_cell_type,
                                              self.config.output_embed_size,
#                                              + self.config.encoder_hidden_size,
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
                                                               self.config.attention_probability_fn,
                                                               self.dropout_placeholder)
        
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([self.output_embed_matrix], output_ids_with_go)
            if self.config.scheduled_sampling > 0:
                sampling_probability = tf.minimum(self.config.scheduled_sampling*tf.cast((self.epoch_placeholder+1)//2, tf.float32), 1)
                helper = ScheduledEmbeddingTrainingHelper(inputs=outputs, sequence_length=self.output_length_placeholder+1,
                                                          embedding=self.output_embed_matrix,
                                                          sampling_probability=sampling_probability)
            else:
                helper = TrainingHelper(outputs, self.output_length_placeholder+1)
        else:
            helper = GreedyEmbeddingHelper(self.output_embed_matrix, go_vector, self.config.grammar.end)
        
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
    
    def _max_margin_loss(self, preds, mask):
        # as in Crammer and Singer, "On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines"

        flat_mask = tf.reshape(mask, (self.batch_size * self.config.max_length,))
        flat_preds = tf.reshape(preds, (self.batch_size * self.config.max_length, self.config.output_size))
        flat_gold = tf.reshape(self.output_placeholder, (self.batch_size * self.config.max_length,))

        flat_indices = tf.range(self.batch_size * self.config.max_length, dtype=tf.int32)
        flat_gold_indices = tf.stack((flat_indices, flat_gold), axis=1)

        one_hot_gold = tf.one_hot(self.output_placeholder, depth=self.config.output_size, dtype=tf.float32)
        marginal_scores = preds - one_hot_gold + 1

        marginal_scores = tf.reshape(marginal_scores, (self.batch_size * self.config.max_length, self.config.output_size))
        max_margin = tf.reduce_max(marginal_scores, axis=1)

        gold_score = tf.gather_nd(flat_preds, flat_gold_indices)
        margin = max_margin - gold_score

        margin = margin * flat_mask

        margin = tf.reshape(margin, (self.batch_size, self.config.max_length))

        return tf.reduce_mean(tf.reduce_sum(margin, axis=1) / tf.cast(self.output_length_placeholder, dtype=tf.float32), axis=0)

    def add_loss_op(self, result):
        logits = result.rnn_output
        length_diff = self.config.max_length - tf.shape(logits)[1]
        padding = tf.convert_to_tensor([[0, 0], [0, length_diff], [0, 0]], name='padding')
        preds = tf.pad(logits, padding, mode='constant')
        preds.set_shape((None, self.config.max_length, result.rnn_output.shape[2]))
        
        mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.float32)
        
        # add epsilon to avoid division by 0
        preds = preds + 1e-5

        return self._max_margin_loss(preds, mask)

        #return tf.contrib.seq2seq.sequence_loss(preds, self.output_placeholder,
        #                                        tf.expand_dims(self.output_weight_placeholder, axis=1) * mask)
