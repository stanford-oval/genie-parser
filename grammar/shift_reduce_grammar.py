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

    def __init__(self, reverse=False, flatten=True, max_input_length=60):
        super().__init__()
        
        self.tokens = ['</s>', '<s>']
        
        self._parser = None

        self._extensible_terminals = []
        self._extensible_terminal_indices = dict()
        self._copy_terminals = []
        self._copy_terminal_indices = dict()
        self._copy_tokens = []
        self._flatten = flatten
        self._reverse = reverse
        self._max_input_length = max_input_length

    @property
    def copy_tokens(self):
        return self._copy_tokens
    
    @property
    def copy_terminal_list(self):
        return ['COPY_' + x for x in self._copy_terminals]

    @property
    def extensible_terminal_list(self):
        return self._extensible_terminals
    
    @property
    def extensible_terminals(self):
        return self._parser.extensible_terminals

    def construct_parser(self, grammar, extensible_terminals=dict(), copy_terminals=dict()):
        if self._flatten:
            # turn each extensible terminal into a new grammar rule

            for nonterm in grammar:
                grammar[nonterm] = [
                    tuple(map(lambda x: ('$terminal_' + x if (x in extensible_terminals or x in copy_terminals) else x), rule))
                    for rule in grammar[nonterm]]
            
            for ext_term, values in extensible_terminals.items():
                grammar['$terminal_' + ext_term] = [(v,) for v in values]
            for copy_term, values in copy_terminals.items():
                grammar['$terminal_' + copy_term] = [(v,) for v in values]
                
            extensible_terminals = dict()
            copy_terminals = dict()
        else:
            self._extensible_terminals = list(extensible_terminals.keys())
            self._extensible_terminals.sort()
            self._copy_terminals = list(copy_terminals.keys())
            self._copy_terminals.sort()
            for copy_term, values in copy_terminals.items():
                for v in values:
                    self._copy_tokens.append(v)
            self._copy_tokens.sort()

        all_extensible_terminals = dict()
        all_extensible_terminals.update(extensible_terminals)
        all_extensible_terminals.update(copy_terminals)        
        generator = SLRParserGenerator(grammar, all_extensible_terminals, '$input')
        self._parser = generator.build()

        for i, term in enumerate(self._extensible_terminals):
            self._extensible_terminal_indices[term] = i
        for i, term in enumerate(self._copy_terminals):
            self._copy_terminal_indices[term] = i
        
        if self._reverse:
            print('using reversed grammar')
        print('num rules', self._parser.num_rules)
        print('num states', self._parser.num_states)
        print('num shifts', len(self._extensible_terminals) + 1)

        self._output_size = dict()
        self._output_size['actions'] = self.num_control_tokens + self._parser.num_rules + len(self._copy_terminals) + len(self._extensible_terminals)
        for term in self._extensible_terminals:
            self._output_size[term] = len(self._parser.extensible_terminals[term])
        for term in self._copy_terminals:
            self._output_size['COPY_' + term] = self._max_input_length
        
        self.input_to_copy_token_map = None
        self.copy_token_to_input_maps = dict()
        for term in self._copy_terminals:
            self.copy_token_to_input_maps['COPY_' + term] = np.zeros((1 + len(self._parser.extensible_terminals[term]),), dtype=np.int32)
        
        return generator.terminals
    
    @property
    def primary_output(self):
        return 'actions'
    
    def is_copy_type(self, output):
        return output.startswith('COPY_')
    
    @property
    def output_size(self):
        return self._output_size
    
    def vectorize_program(self, input_sentence, program, max_length=60):
        if isinstance(program, str):
            program = program.split(' ')
        if self._reverse:
            parsed = self._parser.parse_reverse(program)
        else:
            parsed = self._parser.parse(program)
        
        vectors = dict()
        vectors['actions'] = np.zeros((max_length,), dtype=np.int32)
        for term in self._extensible_terminals:
            vectors[term] = np.ones((max_length,), dtype=np.int32) * -1
        for term in self._copy_terminals:
            vectors['COPY_' + term] = np.zeros((max_length,), dtype=np.int32)
        action_vector = vectors['actions']
        i = 0
        for action, param in parsed:
            if action == 'shift':
                term, tokenidx = param
                if term in self._copy_terminal_indices:
                    assert tokenidx < self._output_size['COPY_' + term]
                    action_vector[i] = self.num_control_tokens + self._parser.num_rules + self._copy_terminal_indices[term]
                    token = self._parser.extensible_terminals[term][tokenidx]
                    input_idx = input_sentence.index(token)
                    vectors['COPY_' + term][i] = input_idx
                elif term in self._parser.extensible_terminals:
                    assert tokenidx < self._output_size[term]
                    action_vector[i] = self.num_control_tokens + self._parser.num_rules + len(self._copy_terminals) + self._extensible_terminal_indices[term]
                    vectors[term][i] = tokenidx
                else:
                    continue
            else:
                action_vector[i] = self.num_control_tokens + param
            assert action_vector[i] < self.num_control_tokens + self._parser.num_rules + len(self._copy_terminals) + len(self._extensible_terminals)
            i += 1
            if i >= max_length-1:
                raise ValueError("Truncated parse of " + str(program) + " (needs " + str(len(parsed)) + " actions)")
        action_vector[i] = self.end # eos
        i += 1
        
        return vectors, i

    def reconstruct_program(self, input_vector, sequences, ignore_errors=False):
        actions = sequences['actions']
        try:
            def gen_action(i):
                x = actions[i]
                if x <= self.end:
                    return ('accept', None)
                elif x < self.num_control_tokens + self._parser.num_rules:
                    return ('reduce', x - self.num_control_tokens)
                elif x < self.num_control_tokens + self._parser.num_rules + len(self._copy_terminals):
                    term = self._copy_terminals[x - self.num_control_tokens - self._parser.num_rules]
                    input_idx = sequences['COPY_' + term][i]
                    input_token_idx = input_vector[1+input_idx]
                    token_idx = self.input_to_copy_token_map[input_token_idx]
                    return ('shift', (term, token_idx))
                else:
                    term = self._extensible_terminals[x - self.num_control_tokens - len(self._copy_terminals) - self._parser.num_rules]
                    return ('shift', (term, sequences[term][i]))
            if self._reverse:
                return self._parser.reconstruct_reverse((gen_action(x) for x in range(len(actions))))
            else:
                return self._parser.reconstruct((gen_action(x) for x in range(len(actions))))
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
        for i, term in enumerate(self._copy_terminals):
            print(i+self.num_control_tokens+self._parser.num_rules, 'copy', term)
        for i, term in enumerate(self._extensible_terminals):
            print(i+self.num_control_tokens+len(self._copy_terminals)+self._parser.num_rules, 'shift', term)

    def _action_to_print_full(self, action):
        if action == 0:
            return ('accept',)
        elif action == 1:
            return ('start',)
        elif action - self.num_control_tokens < self._parser.num_rules:
            lhs, rhs = self._parser.rules[action - self.num_control_tokens]
            return ('reduce', ':', lhs, '->', ' '.join(rhs))
        elif action - self.num_control_tokens - self._parser.num_rules < len(self._copy_terminals):
            term = self._copy_terminals[action - self.num_control_tokens - self._parser.num_rules]
            return ('copy', term)
        else:
            term = self._extensible_terminals[action - self.num_control_tokens - len(self._copy_terminals) - self._parser.num_rules]
            return ('shift', term)

    def output_to_print_full(self, key, output):
        if key == 'actions':
            return self._action_to_print_full(output)
        elif key.startswith('COPY_'):
            if output <= 0:
                return ('null',)
            else:
                return (self._parser.extensible_terminals[key[5:]][output-1],)
        else:
            if output == -1:
                return ('null',)
            else:
                return (self._parser.extensible_terminals[key][output],)

    def print_prediction(self, input_vector, sequences):
        actions = sequences['actions']
        for i, action in enumerate(actions):
            if action == 0:

                print(action, 'accept')
                break
            elif action == 1:
                print(action, 'start')
            elif action - self.num_control_tokens < self._parser.num_rules:
                lhs, rhs = self._parser.rules[action - self.num_control_tokens]
                print(action, 'reduce', ':', lhs, '->', ' '.join(rhs))
            elif action - self.num_control_tokens - self._parser.num_rules < len(self._copy_terminals):
                term = self._copy_terminals[action - self.num_control_tokens - self._parser.num_rules]
                input_idx = sequences['COPY_' + term][i]
                input_token_idx = input_vector[1+input_idx]
                token_idx = self.input_to_copy_token_map[input_token_idx]
                print(action, 'copy', term, input_idx, self._parser.extensible_terminals[term][token_idx])
            else:
                term = self._extensible_terminals[action - self.num_control_tokens - len(self._copy_terminals) - self._parser.num_rules]
                print(action, 'shift', term, sequences[term][i], self._parser.extensible_terminals[term][sequences[term][i]])
    
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
