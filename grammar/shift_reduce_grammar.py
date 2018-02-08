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
Created on Dec 8, 2017

@author: gcampagn
'''

import numpy as np

from .abstract import AbstractGrammar
from .slr import SLRParserGenerator

class ShiftReduceGrammar(AbstractGrammar):
    def __init__(self):
        super().__init__()
        
        self.tokens = ['</s>', '<s>']
        
        self._parser = None

    def construct_parser(self, grammar):
        generator = SLRParserGenerator(grammar, '$input')
        self._parser = generator.build()
        
        print('num rules', self._parser.num_rules)
        print('num states', self._parser.num_states)
        return generator.terminals
    
    USE_SHIFT_REDUCE = True
    
    @property
    def output_size(self):
        if self.USE_SHIFT_REDUCE:
            # padding, eos, go, reduce 0 to n-1
            # the last rule is $ROOT -> $input <<EOF>>
            # which is a pseudo-rule needed for the SLR generator
            # we ignore it here
            return self.num_control_tokens + self._parser.num_rules - 1
        else:
            return len(self.tokens)

    def vectorize_program(self, program, max_length=60):
        if self.USE_SHIFT_REDUCE:
            if isinstance(program, str):
                program = program.split(' ')
            parsed = self._parser.parse(program)
            vector = np.zeros((max_length,), dtype=np.int32)
            i = 0
            for action, param in parsed:
                if action == 'shift':
                    continue
                if i >= max_length-1:
                    raise ValueError("Truncated parse of " + str(program) + " (needs " + str(len(parsed)) + " actions)")
                vector[i] = self.num_control_tokens + param
                assert vector[i] < self.num_control_tokens + self._parser.num_rules-1
                i += 1
            vector[i] = self.end # eos
            i += 1
            
            return vector, i
        else:
            vector, length = super().vectorize_program(program, max_length)
            self.normalize_sequence(vector)
            return vector, length

    def reconstruct_program(self, sequence, ignore_errors=False):
        if self.USE_SHIFT_REDUCE:
            try:
                def gen_action(x):
                    if x <= self.end:
                        return ('accept', None)
                    else:
                        return ('reduce', x - self.num_control_tokens)
                return self._parser.reconstruct((gen_action(x) for x in sequence))
            except (KeyError, TypeError, IndexError, ValueError):
                if ignore_errors:
                    # the NN generated something that does not conform to the grammar,
                    # ignore it 
                    return []
                else:
                    raise
        else:
            return super().reconstruct_program(sequence)

    def print_all_actions(self):
        print(0, 'accept')
        print(1, 'start')
        for i, (lhs, rhs) in enumerate(self._parser.rules):
            print(i+self.num_control_tokens, 'reduce', lhs, '->', ' '.join(rhs))

    def print_prediction(self, sequence):
        if not self.USE_SHIFT_REDUCE:
            super().print_prediction(sequence)
            return
    
        for action in sequence:
            if action == 0:
                print('accept')
            elif action == 1:
                print('start')
            else:
                lhs, rhs = self._parser.rules[action - self.num_control_tokens]
                print('reduce', action - self.num_control_tokens, ':', lhs, '->', ' '.join(rhs))

    def output_to_string(self, action):
        if action == 0:
            return 'accept'
        elif action == 1:
            return 'start'
        else:
            lhs, rhs = self._parser.rules[action - self.num_control_tokens]
            return (lhs + ' -> ' + (' '.join(rhs)))

    def prediction_to_string(self, sequence):
        def action_to_string(action):
            if action == 0:
                return 'A'
            elif action == 1:
                return 'S'
            else:
                return 'R' + str(action - self.num_control_tokens)
        return list(map(action_to_string, sequence))

    def parse(self, program):
        return self._parser.parse(program)

    def parse_all(self, fp):
        max_length = 0
        for line in fp.readlines():
            try:
                program = line.strip().split()
                parsed = self._parser.parse(program)
                reduces = [x for x in parsed if x[0] =='reduce']
                if len(reduces) > max_length:
                    max_length = len(reduces)
            except ValueError as e:
                print(' '.join(program))
                raise e
