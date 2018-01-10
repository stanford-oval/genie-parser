# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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

from util.loader import vectorize

class AbstractGrammar(object):
    '''
    Base class for a Grammar that defines the output of a Sequence to Sequence
    (or other X to Sequence) parser
    
    A Grammar defines the following attributes:
        - tokens: the list of string tokens in the grammar
        - dictionary: a mapping from token to its ID
        
    All Grammars must include a mapping for <<GO>> and <<EOS>>
    '''
    
    def __init__(self):
        self.tokens = []
        self.dictionary = dict()
        self.num_control_tokens = 2
        
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
    
    def reconstruct_program(self, sequence, ignore_errors=False):
        ret = []
        for x in sequence:
            if x == self.end:
                break
            ret.append(self.tokens[x])
        return ret
    
    def print_prediction(self, sequence):
        print(' '.join(self.tokens[x] for x in sequence))
    
    def vectorize_program(self, program, max_length):
        return vectorize(program, self.dictionary, max_length, add_eos=True)
    
    def get_embeddings(self, *args):
        return np.identity(self.output_size, np.float32)

    def compare(self, seq1, seq2):
        '''
        Compare two sequence, to check if they represent semantically equivalent outputs
        '''
        return seq1 == seq2

    def normalize_sequence(self, seq):
        pass
