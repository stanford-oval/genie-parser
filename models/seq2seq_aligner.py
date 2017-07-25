'''
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from .seq2seq_helpers import Seq2SeqDecoder, AttentionSeq2SeqDecoder

class Seq2SeqAligner(BaseAligner):
    '''
    A model that implements Seq2Seq: that is, it uses a sequence loss
    during training, and a greedy decoder during inference 
    '''
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])
        if self.config.apply_attention:
            decoder = AttentionSeq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                              self.output_placeholder, self.output_length_placeholder)
        else:
            decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                     self.output_placeholder, self.output_length_placeholder)
        return decoder.decode(cell_dec, enc_hidden_states, enc_final_state, output_embed_matrix, training)