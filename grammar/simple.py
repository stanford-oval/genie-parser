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

from .abstract import AbstractGrammar
from .thingtalk import ThingTalkGrammar

class SimpleGrammar(AbstractGrammar):
    '''
    A simple implementation of AbstractGrammar, that reads the 
    sequence of tokens from a given file (one per line)
    
    The resulting grammar is:
    
    $ROOT -> $Token *
    
    where $Token is any grammar token
    '''
    
    def __init__(self, filename, split_device=False):
        super().__init__()
        
        self._split_device = split_device
        
        self.tokens = ['</s>', '<s>']
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
        print(self.entities)
        
        # HACK
        self._thingtalk = ThingTalkGrammar('./thingpedia.json')
        
    def vectorize_program(self, program, max_length):
        if not self._split_device:
            return super().vectorize_program(program, max_length=max_length)
        
        if isinstance(program, str):
            program = program.split(' ')
        def program_with_device():
            for tok in program:
                if tok.startswith('@'):
                    device = tok[:tok.rindex('.')]
                    yield '@' + device
                yield tok
        return super().vectorize_program(program_with_device(), max_length=max_length)
        
    def reconstruct_program(self, sequence, ignore_errors=False):
        program = super().reconstruct_program(sequence, ignore_errors=ignore_errors)
        if self._split_device:
            program = [x for x in program if not x.startswith('@@')]
        
        try:
            self._thingtalk.vectorize_program(program, 60)
            return program
        except:
            if ignore_errors:
                return []
            else:
                raise
        
