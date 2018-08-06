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
import os
import numpy as np
import tensorflow as tf

from .abstract import AbstractGrammar

from ..util.loader import vectorize


class SimpleGrammar(AbstractGrammar):
    '''
    A simple implementation of AbstractGrammar, that reads the 
    sequence of tokens from a given file (one per line)
    
    The resulting grammar is:
    
    $ROOT -> $Token *
    
    where $Token is any grammar token
    '''

    def __init__(self, filename, flatten=True, split_device=False, **kw):
        super().__init__(**kw)
        
        if not flatten:
            raise ValueError('Cannot use a the extensible model with a simple grammar; use seq2seq model instead')

        self._split_device = split_device
        
        self.tokens = ['<pad>', '</s>', '<s>']
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                self.tokens.append(line.strip())

        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i

        self.allfunctions = [x for x in self.tokens if x.startswith('@')]
        self.entities = set((x[len('GENERIC_ENTITY_'):-2], True) for x in self.tokens if x.startswith('GENERIC_ENTITY_'))
        self.entities = list(self.entities)
        self.entities.sort()

    @property
    def primary_output(self):
        return 'tokens'

    @property
    def output_size(self):
        return {
            'tokens': len(self.tokens)
        }
        
    def tokenize_to_vector(self, input_sentence, program):
        vector, vlen = vectorize(program, self.dictionary, max_length=None, add_eos=True, add_start=False)
        return vector[:vlen]

    def vectorize_program(self, input_sentence, program, direction='linear', max_length=60):
        if direction != 'linear':
            raise ValueError("Invalid " + direction + " direction for simple grammar")
        if isinstance(program, np.array):
            return { 'tokens': program }, len(program)

        del input_sentence
        vector, vlen = vectorize(program, self.dictionary, max_length, add_eos=True, add_start=False)
        return {
            'tokens': vector
        }, vlen
        
    def reconstruct_to_vector(self, sequences, direction, ignore_errors=False):
        if direction != 'linear':
            raise ValueError("Invalid " + direction + " direction for simple grammar")
        return sequences['tokens']

    def print_prediction(self, input_sentence, sequence):
        for token in sequence['tokens']:
            print(token, self.tokens[token])
    
    def prediction_to_string(self, sequence):
        return [self.tokens[x] for x in sequence[self.primary_output]]
