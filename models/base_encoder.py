'''
Created on Jul 27, 2017

@author: gcampagn
'''

import tensorflow as tf

class BaseEncoder(object):
    '''
    Base class for an operation that takes a sentence
    (embedded in a some high dimensional vector space) and
    produces a dense representation of it
    '''
    
    def __init__(self, embed_size, output_size, output_dropout):
        self._embed_size = embed_size
        self._output_size = output_size
        self._output_dropout = output_dropout
    
    @property
    def output_size(self):
        return self._output_size
    
    @property
    def embed_size(self):
        return self._embed_size
    
    def encode(self, inputs : tf.Tensor, input_length : tf.Tensor):
        '''
        Encode the given sentence
        
        Args:
            inputs: a tf.Tensor of shape [batch_size, max_time, embed_size]
            input_length: a tf.Tensor of shape [batch_size] and dtype tf.int32
            
        Returns:
            A tuple (enc_hidden_states, enc_final_states), where
            enc_hidden_states is a tensor of shape [batch_size, max_time, output_size]
            and enc_final_states a tensor of shape [batch_size, output_size]
        '''
        raise NotImplementedError
