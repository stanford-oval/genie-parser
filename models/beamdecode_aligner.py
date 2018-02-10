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
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import FinalBeamSearchDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoderOutput

'''
Created on Feb 9, 2018

@author: gcampagn
'''

import tensorflow as tf

from . import common
from .seq2seq_aligner import Seq2SeqAligner

from tensorflow.contrib.seq2seq import BeamSearchDecoder
    
class BeamDecodeAligner(Seq2SeqAligner):
    def add_decoder_op(self, enc_final_state, enc_hidden_states, training):
        if training:
            return super().add_decoder_op(enc_final_state, enc_hidden_states, training)

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
            enc_hidden_states = tf.contrib.seq2seq.tile_batch(enc_hidden_states, multiplier=self.config.beam_size)
            tiled_enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, multiplier=self.config.beam_size)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.input_length_placeholder, multiplier=self.config.beam_size)
            
            cell_dec, enc_final_state = common.apply_attention(cell_dec,
                                                               enc_hidden_states,
                                                               tiled_enc_final_state,
                                                               tiled_sequence_length,
                                                               self.batch_size * self.config.beam_size,
                                                               self.config.attention_probability_fn,
                                                               self.dropout_placeholder,
                                                               alignment_history=False)
            #enc_final_state = enc_final_state.clone(cell_state=tiled_enc_final_state)
        else:
            enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, multiplier=self.config.beam_size)
        print('enc final_state', enc_final_state)
        
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        
        output_layer = tf.layers.Dense(self.config.grammar.output_size, use_bias=False)
        
        decoder = BeamSearchDecoder(cell_dec, self.output_embed_matrix,
                                    start_tokens=go_vector,
                                    end_token=self.config.grammar.end, 
                                    initial_state=enc_final_state,
                                    beam_width=self.config.beam_size,
                                    output_layer=output_layer)
        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                maximum_iterations=self.config.max_length,
                                                                swap_memory=True)
        return final_outputs

    def finalize_predictions(self, preds):
        return preds.predicted_ids
    
    def add_loss_op(self, result):
        if isinstance(result, FinalBeamSearchDecoderOutput):
            return tf.convert_to_tensor(0.0)
        else:
            return super().add_loss_op(result)
