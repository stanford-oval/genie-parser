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
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf
import numpy as np
from tensor2tensor.layers import common_layers

class AbstractGrammar(object):
    '''
    Base class for a Grammar that defines the output of a Sequence to Sequence
    (or other X to Sequence) parser
    
    A Grammar defines the following attributes:
        - tokens: the list of string tokens in the grammar
        - dictionary: a mapping from token to its ID

    All Grammars must include a mapping for <pad>, <s> and </s>
    Furthermore, <pad> must have ID 0 and </s> must have ID 1
    '''

    def __init__(self, **kw):
        self.tokens = []
        self.dictionary = dict()
        self.num_control_tokens = 3

    def set_input_dictionary(self, input_dictionary):
        # AbstractGrammar does nothing with it
        pass

    @property
    def start(self):
        ''' The ID of the start token when decoding '''
        return self.dictionary['<s>']

    @property
    def end(self):
        ''' The ID of the end token, which signals end of decoding '''
        return self.dictionary['</s>']

    def dump_tokens(self):
        for t in self.tokens:
            print(t)
    
    @property
    def output_size(self):
        raise NotImplementedError()
    
    def is_copy_type(self, output):
        return False
    
    def verify_program(self, program):
        # nothing to do by default
        pass
    
    def tokenize_to_vector(self, input_sentence, program, max_length):
        raise NotImplementedError()
    
    def vectorize_program(self, input_sentence, program, direction, max_length):
        raise NotImplementedError()
    
    def reconstruct_to_vector(self, sequences, direction, ignore_errors=False):
        raise NotImplementedError()
    
    def reconstruct_program(self, input_sentence, sequence, direction, ignore_errors=False):
        if direction != 'linear':
            raise ValueError("Invalid " + direction + " direction for simple grammar")
        if isinstance(sequence, dict):
            sequence = sequence['targets']
        program = []
        for x in sequence:
            if x == self.end:
                break
            program.append(self.tokens[x])
        return program
    
    def print_prediction(self, input_sentence, sequence):
        raise NotImplementedError()
        
    def prediction_to_string(self, sequence):
        raise NotImplementedError()
    
    def get_embeddings(self, input_words, input_embeddings):
        embeddings = dict()
        for key, size in self.output_size.items():
            embeddings[key] = np.identity(size, dtype=np.float32)
        return embeddings

    def eval_metrics(self):
        def accuracy(predictions, labels, features):
            batch_size = tf.shape(predictions)[0]
            predictions, labels = common_layers.pad_with_zeros(predictions, labels)
            weights = tf.ones((batch_size,), dtype=tf.float32)
            ok = tf.to_float(tf.reduce_all(tf.equal(predictions, labels, axis=1), axis=1))
            return ok, weights
        
        def grammar_accuracy(predictions, labels, features):
            batch_size = tf.shape(predictions)[0]
            weights = tf.ones((batch_size,), dtype=tf.float32)
            
            return tf.cond(tf.shape(predictions)[1] > 0,
                           lambda: tf.to_float(predictions[:,0] > 0),
                           lambda: tf.zeros_like(weights)), weights
        
        return {
            "accuracy": accuracy,
            "grammar_accuracy": grammar_accuracy 
        }
