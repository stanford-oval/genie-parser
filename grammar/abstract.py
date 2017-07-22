'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf
import numpy as np

class AbstractGrammar(object):
    '''
    Base class for a Grammar that defines the output of a Sequence to Sequence
    (or other X to Sequence) parser
    
    A Grammar defines the following attributes:
        - tokens: the list of string tokens in the grammar
        - dictionary: a mapping from token to its ID
        
    All Grammars must include a mapping for <<GO>>, <<EOS>> and <<PAD>>
    '''
    
    def __init__(self):
        self.tokens = []
        self.dictionary = dict()
        
    @property
    def output_size(self):
        return len(self.tokens)
        
    @property
    def start(self):
        ''' The ID of the start token when decoding '''
        return self.dictionary['<<GO>>']
    
    @property
    def end(self):
        ''' The ID of the end token, which signals end of decoding '''
        return self.dictionary['<<EOS>>']
    
    def constrain(self, logits : tf.Tensor, curr_state, batch_size : tf.Tensor, dtype=tf.int32):
        '''
        Apply grammar constraints to a Tensor of sequence outputs, and returns
        the next predicted token
        
        Args:
            logits: the logits produced by the current step of sequence decoding
            curr_state: the state returned by the previous call of constrain() or None
            batch_size: 
            dtype: the tensor dtype for the token ID to return
            
        Returns:
            A tuple of (token_id, next_state); token_id is a tensor of shape (batch_size,) and
            dtype dtype
        '''
        
        if curr_state is None:
            return tf.ones((batch_size,), dtype=dtype) * self.start, ()
        else:
            return tf.cast(tf.argmax(logits, axis=1), dtype=dtype), ()

    def decode_output(self, sequence):
        '''
        Decode a sequence of logits into a sequence of token identifiers
        
        Args:
            sequence: a numpy.ndarray of shape (max_length, output_size)
            
        Returns:
            The decoded sequence, as a list of integer token IDs
        '''
         
        output = []
        for logits in sequence:
            assert logits.shape == (self.output_size,)
            word_idx = np.argmax(logits)
            if word_idx > 0:
                output.append(word_idx)
        return output
    
    def compare(self, seq1, seq2):
        '''
        Compare two sequence, to check if they represent semantically equivalent outputs
        '''
        return seq1 == seq2