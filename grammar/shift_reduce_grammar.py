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
import tensorflow as tf

from .abstract import AbstractGrammar
from .slr import SLRParserGenerator

class ShiftReduceGrammar(AbstractGrammar):
    def __init__(self, flatten=True):
        super().__init__()
        
        self.tokens = ['</s>', '<s>']
        
        self._parser = None
        self._extensible_terminals = []
        self._extensible_terminal_indices = dict()
        self._flatten = flatten

    @property
    def extensible_terminal_list(self):
        return self._extensible_terminals
    
    @property
    def extensible_terminals(self):
        return self._parser.extensible_terminals

    def construct_parser(self, grammar, extensible_terminals=dict()):
        if self._flatten:
            # turn each extensible terminal into a new grammar rule

            for nonterm in grammar:
                grammar[nonterm] = [
                    tuple(map(lambda x: ('$terminal_' + x if x in extensible_terminals else x), rule))
                    for rule in grammar[nonterm]]
            
            for ext_term, values in extensible_terminals.items():
                grammar['$terminal_' + ext_term] = [(v,) for v in values]
                
            extensible_terminals = dict()
            self._extensible_terminals = []
        else:
            self._extensible_terminals = list(extensible_terminals.keys())
            self._extensible_terminals.sort()
        
        generator = SLRParserGenerator(grammar, extensible_terminals, '$input')
        self._parser = generator.build()

        for i, term in enumerate(self._extensible_terminals):
            self._extensible_terminal_indices[term] = i
        
        print('num rules', self._parser.num_rules)
        print('num states', self._parser.num_states)
        print('num shifts', len(self._extensible_terminals))

        self._output_size = dict()
        self._output_size['actions'] = self.num_control_tokens + self._parser.num_rules + len(self._extensible_terminals)
        for term in self._extensible_terminals:
            self._output_size[term] = len(self._parser.extensible_terminals[term])
            
        return generator.terminals
    
    @property
    def primary_output(self):
        return 'actions'
    
    @property
    def output_size(self):
        return self._output_size
    
    def vectorize_program(self, program, max_length=60):
        if isinstance(program, str):
            program = program.split(' ')
        parsed = self._parser.parse(program)
        
        vectors = dict()
        vectors['actions'] = np.zeros((max_length,), dtype=np.int32)
        for term in self._extensible_terminals:
            vectors[term] = np.ones((max_length,), dtype=np.int32) * -1
        action_vector = vectors['actions']
        i = 0
        for action, param in parsed:
            if action == 'shift':
                term, tokenidx = param
                if not term in self._parser.extensible_terminals:
                    continue
                assert tokenidx < self._output_size[term]
                action_vector[i] = self.num_control_tokens + self._parser.num_rules + self._extensible_terminal_indices[term]
                vectors[term][i] = tokenidx
            else:
                action_vector[i] = self.num_control_tokens + param
            assert action_vector[i] < self.num_control_tokens + self._parser.num_rules + len(self._extensible_terminals)
            i += 1
            if i >= max_length-1:
                raise ValueError("Truncated parse of " + str(program) + " (needs " + str(len(parsed)) + " actions)")
        action_vector[i] = self.end # eos
        i += 1
        
        return vectors, i

    def reconstruct_program(self, sequences, ignore_errors=False):
        actions = sequences['actions']
        try:
            def gen_action(i):
                x = actions[i]
                if x <= self.end:
                    return ('accept', None)
                elif x < self.num_control_tokens + self._parser.num_rules:
                    return ('reduce', x - self.num_control_tokens)
                else:
                    term = self._extensible_terminals[x - self.num_control_tokens - self._parser.num_rules]
                    return ('shift', (term, sequences[term][i]))
            return self._parser.reconstruct((gen_action(i) for i in range(len(actions))))
        except (KeyError, TypeError, IndexError, ValueError):
            if ignore_errors:
                # the NN generated something that does not conform to the grammar,
                # ignore it 
                return []
            else:
                raise

    def print_all_actions(self):
        print(0, 'accept')
        print(1, 'start')
        for i, (lhs, rhs) in enumerate(self._parser.rules):
            print(i+self.num_control_tokens, 'reduce', lhs, '->', ' '.join(rhs))
        for i, term in enumerate(self._extensible_terminals):
            print(i+self.num_control_tokens+len(self._parser.rules), 'shift', term)

    def print_prediction(self, sequences):
        actions = sequences['actions']
        for action in actions:
            if action == 0:
                print('accept')
            elif action == 1:
                print('start')
            elif action - self.num_control_tokens < self._parser.num_rules:
                lhs, rhs = self._parser.rules[action - self.num_control_tokens]
                print('reduce', action - self.num_control_tokens, ':', lhs, '->', ' '.join(rhs))
            else:
                term = self._extensible_terminals[action - self.num_control_tokens - self._parser.num_rules]
                print('shift', term, self._parser.extensible_terminals[term][sequences[term]])

    def prediction_to_string(self, sequences):
        def action_to_string(action):
            if action == 0:
                return 'A'
            elif action == 1:
                return 'G'
            elif action - self.num_control_tokens < self._parser.num_rules:
                return 'R' + str(action - self.num_control_tokens)
            else:
                return 'S' + str(action - self.num_control_tokens - self._parser.num_rules)
        return list(map(action_to_string, sequences['actions']))