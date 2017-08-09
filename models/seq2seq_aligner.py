'''
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from .seq2seq_helpers import Seq2SeqDecoder, AttentionSeq2SeqDecoder

from tensorflow.python.util import nest

def pad_up_to(vector, size):
    rank = vector.get_shape().ndims - 1
    
    length_diff = tf.reshape(size - tf.shape(vector)[1], shape=(1,))
    with tf.control_dependencies([tf.assert_non_negative(length_diff, data=(vector, size, tf.shape(vector)))]):
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0,0]*(rank-1)], axis=0), shape=((rank+1), 2))
        return tf.pad(vector, padding, mode='constant')

class Seq2SeqAligner(BaseAligner):
    '''
    A model that implements Seq2Seq: that is, it uses a sequence loss
    during training, and a greedy decoder during inference 
    '''
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i, True) for i in range(self.config.rnn_layers)])
        
        encoder_hidden_size = int(enc_hidden_states.get_shape()[-1])
        decoder_hidden_size = int(cell_dec.output_size)
        
        # if encoder and decoder have different sizes, add a projection layer
        if encoder_hidden_size != decoder_hidden_size:
            with tf.variable_scope('hidden_projection'):
                kernel = tf.get_variable('kernel', (encoder_hidden_size, decoder_hidden_size), dtype=tf.float32)
            
                # apply a relu to the projection for good measure
                enc_final_state = nest.map_structure(lambda x: tf.nn.relu(tf.matmul(x, kernel)), enc_final_state)
                enc_hidden_states = tf.tensordot(enc_hidden_states, kernel, [[2], [1]])
        
        if self.config.apply_attention:
            decoder = AttentionSeq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                              self.output_placeholder, self.output_length_placeholder, self.batch_number_placeholder)
        else:
            decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                     self.output_placeholder, self.output_length_placeholder, self.batch_number_placeholder)
        return decoder.decode(cell_dec, enc_hidden_states, enc_final_state, self.config.grammar.output_size, output_embed_matrix, training)
    
    def finalize_predictions(self, preds):
        # add a dimension of 1 between the batch size and the sequence length to emulate a beam width of 1 
        return tf.expand_dims(preds.sample_id, axis=1)
    
    def add_loss_op(self, result):
        logits = result.rnn_output
        with tf.control_dependencies([tf.assert_positive(tf.shape(logits)[1], data=[tf.shape(logits)])]):
            length_diff = tf.reshape(self.config.max_length - tf.shape(logits)[1], shape=(1,))
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0, 0]], axis=0), shape=(3, 2))
        preds = tf.pad(logits, padding, mode='constant')
        
        # add epsilon to avoid division by 0
        preds = preds + 1e-5

        mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(preds, self.output_placeholder, mask)

        with tf.control_dependencies([tf.assert_non_negative(loss, data=[preds, mask], summarize=256*60*300)]):
            return tf.identity(loss)
