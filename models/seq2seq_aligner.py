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
    
    def _do_decode(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])
        if self.config.apply_attention:
            decoder = AttentionSeq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                              self.output_placeholder, self.output_length_placeholder, self.batch_number_placeholder)
        else:
            decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                     self.output_placeholder, self.output_length_placeholder, self.batch_number_placeholder)
        return decoder.decode(cell_dec, enc_hidden_states, enc_final_state, self.config.grammar.output_size, output_embed_matrix, training)
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        # in training mode, try with both the training and the inference decoder, to apply sentence
        # level scheduled sampling
        if training and self.config.scheduled_sampling > 0:
            out1 = self._do_decode(enc_final_state, enc_hidden_states, output_embed_matrix, True)
            out2 = self._do_decode(enc_final_state, enc_hidden_states, output_embed_matrix, False)
            
            # make sure both structures have the same length, by padding all the way to max length
            out1 = nest.map_structure(lambda x: pad_up_to(x, self.config.max_length), out1)
            out2 = nest.map_structure(lambda x: pad_up_to(x, self.config.max_length), out2)
            
            nest.assert_same_structure(out1, out2)

            choice = tf.random_uniform((self.batch_size,), 0, 1, dtype=tf.float32, seed=123457)      
            scheduled_sample_prob = self.config.scheduled_sampling * tf.cast(self.batch_number_placeholder, dtype=tf.float32)      
            outflat = [tf.where(choice > scheduled_sample_prob, x, y) for x, y in zip(nest.flatten(out1), nest.flatten(out2))]
            return nest.pack_sequence_as(out1, outflat)
        else:
            return self._do_decode(enc_final_state, enc_hidden_states, output_embed_matrix, training)
    
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
