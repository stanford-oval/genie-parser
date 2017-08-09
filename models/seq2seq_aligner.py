'''
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from .seq2seq_helpers import Seq2SeqDecoder

from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper
from tensorflow.python.util import nest

def pad_up_to(vector, size):
    rank = vector.get_shape().ndims - 1
    
    length_diff = tf.reshape(size - tf.shape(vector)[1], shape=(1,))
    with tf.control_dependencies([tf.assert_non_negative(length_diff, data=(vector, size, tf.shape(vector)))]):
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0,0]*(rank-1)], axis=0), shape=((rank+1), 2))
        return tf.pad(vector, padding, mode='constant')

class ParentFeedingCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, wrapped : tf.contrib.rnn.RNNCell, parent_state):
        super().__init__()
        self._wrapped = wrapped
        self._flat_parent_state = tf.concat(nest.flatten(parent_state), axis=1)
        
    def call(self, input, state):
        concat_input = tf.concat((self._flat_parent_state, input), axis=1)
        return self._wrapped.call(concat_input, state)
    
    @property
    def output_size(self):
        return self._wrapped.output_size
    
    @property
    def state_size(self):
        return self._wrapped.state_size

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
            assert False, (encoder_hidden_size, decoder_hidden_size)
            with tf.variable_scope('hidden_projection'):
                kernel = tf.get_variable('kernel', (encoder_hidden_size, decoder_hidden_size), dtype=tf.float32)
            
                # apply a relu to the projection for good measure
                enc_final_state = nest.map_structure(lambda x: tf.nn.relu(tf.matmul(x, kernel)), enc_final_state)
                enc_hidden_states = tf.nn.relu(tf.tensordot(enc_hidden_states, kernel, [[2], [1]]))
        else:
            # flatten and repack the state
            enc_final_state = nest.pack_sequence_as(cell_dec.state_size, nest.flatten(enc_final_state))
        
        cell_dec = ParentFeedingCellWrapper(cell_dec, enc_final_state)
        if self.config.apply_attention:
            attention = LuongAttention(self.config.decoder_hidden_size, enc_hidden_states, self.input_length_placeholder,
                                       probability_fn=tf.nn.softmax)
            cell_dec = AttentionWrapper(cell_dec, attention,
                                        cell_input_fn=lambda inputs, _: inputs,
                                        attention_layer_size=self.config.decoder_hidden_size,
                                        initial_cell_state=enc_final_state)
            enc_final_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                 self.output_placeholder, self.output_length_placeholder, self.batch_number_placeholder)
        return decoder.decode(cell_dec, enc_final_state, self.config.grammar.output_size, output_embed_matrix, training)
    
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
