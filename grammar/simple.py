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

from .abstract import AbstractGrammar

from util.loader import vectorize

class SimpleGrammar(AbstractGrammar):
    '''
    A simple implementation of AbstractGrammar, that reads the 
    sequence of tokens from a given file (one per line)
    
    The resulting grammar is:
    
    $ROOT -> $Token *
    
    where $Token is any grammar token
    '''
    
    def __init__(self, filename, flatten=True, **kw):
        super().__init__(**kw)
        
        if not flatten:
            raise ValueError('Cannot use a the extensible model with a simple grammar; use seq2seq model instead')
        
        self.tokens = ['</s>', '<s>']
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                self.tokens.append(line.strip())

        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i

    @property
    def primary_output(self):
        return 'tokens'

    @property
    def output_size(self):
        return {
            'tokens': len(self.tokens)
        }
        
    def vectorize_program(self, input_sentence, program, max_length):
        del input_sentence
        return {
            'tokens': vectorize(program, self.dictionary, max_length, add_eos=True)
        }
        
    def reconstruct_program(self, input_sentence, sequence, ignore_errors=False):
        if isinstance(sequence, dict):
            sequence = sequence['tokens']
        ret = []
        for x in sequence:
            if x == self.end:
                break
            ret.append(self.tokens[x])
        return ret