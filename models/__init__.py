'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_model import BaseModel
from .base_aligner import BaseAligner
from .config import Config

class LSTMAligner(BaseAligner):
    def add_encoder_op(self, inputs, scope=None):
        cell_enc = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(id) for id in range(self.config.rnn_layers)])
        #cell_enc = tf.contrib.rnn.AttentionCellWrapper(cell_enc, 5, state_is_tuple=True)

        return tf.nn.dynamic_rnn(cell_enc, inputs, sequence_length=self.input_length_placeholder,
                                 dtype=tf.float32, scope=scope)
        # assert enc_preds.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
        # if self.config.input_cell == "lstm":
        #     assert enc_final_state[0][0].get_shape()[1:] == (self.config.hidden_size,)
        #     assert enc_final_state[0][1].get_shape()[1:] == (self.config.hidden_size,)
        # else:
        #     assert enc_final_state.get_shape()[1:] == (self.config.hidden_size,)


class BagOfWordsAligner(BaseAligner):
    def add_encoder_op(self, inputs, scope=None):
        W = tf.get_variable('W', (self.config.embed_size, self.config.hidden_size))
        b = tf.get_variable('b', shape=(self.config.hidden_size,), initializer=tf.constant_initializer(0, tf.float32))

        enc_hidden_states = tf.tanh(tf.tensordot(inputs, W, [[2], [0]]) + b)
        enc_hidden_states.set_shape((None, self.config.max_length, self.config.hidden_size))
        enc_final_state = tf.reduce_sum(enc_hidden_states, axis=1)

        #assert enc_hidden_states.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
        
        if self.config.rnn_cell_type == 'lstm':
            enc_final_state = (tf.contrib.rnn.LSTMStateTuple(enc_final_state, enc_final_state),)
        
        return enc_hidden_states, enc_final_state
    
def create_model(config):
    if config.model_type == 'bagofwords':
        model = BagOfWordsAligner(config)
    elif config.model_type == 'seq2seq':
        model = LSTMAligner(config)
    else:
        raise ValueError("Invalid model type %s" % (config.model_type,))
    
    return model