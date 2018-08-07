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
Created on Nov 30, 2017

@author: gcampagn
'''

from collections import defaultdict

from ..slr import EOF_ID, ACCEPT_CODE, REDUCE_CODE, SHIFT_CODE, INVALID_CODE


class ShiftReduceParser:
    '''
    A bottom-up parser for a deterministic CFG language, based on shift-reduce
    tables.
    
    The parser can transform a string in the language to a sequence of
    shifts and reduces, and can transform a valid sequence of reduces to
    a string in the language.
    '''
    
    def __init__(self, rules, rule_table, action_table, goto_table, terminals, dictionary, start_symbol):
        super().__init__()
        self.rules = rules
        self.rule_table = rule_table
        self._action_table = action_table
        self._goto_table = goto_table
        self.terminals = terminals
        self.dictionary = dictionary
        self._start_symbol = start_symbol

    @property
    def num_rules(self):
        # the last rule is $ROOT -> $input <<EOF>>
        # which is a pseudo-rule needed for the SLR generator
        # we ignore it here
        return len(self.rules)
    
    @property
    def num_states(self):
        return len(self._action_table)
    
    def parse_reverse(self, sequence):
        bottom_up_sequence = self.parse(sequence)
        lens = [None] * len(bottom_up_sequence)
        children = [None] * len(bottom_up_sequence)
        reduces = [None] * len(bottom_up_sequence)
        i = 0
        for action,param in bottom_up_sequence:
            if action == SHIFT_CODE:
                continue
            lhs, rhs = self.rules[param]
            current_child = i-1
            my_length = 1
            my_children = []
            for rhsitem in reversed(rhs):
                if rhsitem.startswith('$'):
                    my_children.append(current_child)
                    my_length += lens[current_child]
                    current_child -= lens[current_child]
            lens[i] = my_length
            reduces[i] = (action,param)
            children[i] = tuple(reversed(my_children))
            i += 1
        reversed_sequence = []
        def write_subsequence(node, start):
            reversed_sequence.append(reduces[node])
            for c in children[node]:
                write_subsequence(c, start)
                start += lens[c]
        write_subsequence(i-1, 0)
        return reversed_sequence

    def parse(self, sequence):
        stack = [0]
        state = 0
        result = []
        sequence_iter = iter(sequence)
        terminal_id, token = next(sequence_iter)
        while True:
            if self._action_table[state, terminal_id, 0] == INVALID_CODE:
                expected_token_ids,  = self._action_table[state, :, 0].nonzero()
                expected_tokens = [self.terminals[i] for i in expected_token_ids]
                
                raise ValueError(
                    "Parse error: unexpected token " + self.terminals[terminal_id] + " in state " + str(state) + ", expected " + str(
                        expected_tokens))
            action, param = self._action_table[state, terminal_id]
            if action == ACCEPT_CODE:
                return result
            #if action == 'shift':
            #    print('shift', param, token)
            #else:
            #    print('reduce', param, self.rules[param])
            if action == SHIFT_CODE:
                state = param
                result.append((SHIFT_CODE, (terminal_id, token)))
                stack.append(state)
                try:
                    terminal_id, token = next(sequence_iter)
                except StopIteration:
                    terminal_id = EOF_ID
            else:
                assert action == REDUCE_CODE
                rule_id = param
                result.append((REDUCE_CODE, rule_id))
                lhs_id, rhssize = self.rule_table[rule_id]
                for _ in range(rhssize):
                    stack.pop()
                state = stack[-1]
                state = self._goto_table[state, lhs_id]
                stack.append(state)
                
    def reconstruct_reverse(self, sequence):
        output_sequence = []
        if not isinstance(sequence, list):
            sequence = list(sequence)
    
        def recurse(start_at):
            action, param = sequence[start_at]
            if action != REDUCE_CODE:
                raise ValueError('invalid action')
            _, rhs = self.rules[param]
            length = 1
            for rhsitem in rhs:
                if rhsitem.startswith('$'):
                    length += recurse(start_at + length)
                else:
                    output_sequence.append(rhsitem)
            return length
    
        recurse(0)
        return output_sequence

    def reconstruct(self, sequence):
        # beware: this code is tricky
        # don't approach it without patience, pen and paper and
        # many test cases

        stack = []
        top_stack_id = None
        token_stacks = defaultdict(list)
        for action, param in sequence:
            if action == ACCEPT_CODE:
                break
            elif action == SHIFT_CODE:
                term_id, token = param
                token_stacks[term_id].append(token)
            else:
                assert action == REDUCE_CODE
                rule_id = param
                lhs, rhs = self.rules[rule_id]
                top_stack_id = self.rule_table[rule_id, 0]
                if len(rhs) == 1:
                    #print("fast path for", lhs, "->", rhs)
                    # fast path for unary rules
                    symbol = rhs[0]
                    if symbol[0] == '$':
                        # unary non-term to non-term, no stack manipulation
                        continue
                    # unary term to non-term, we push directly to the stack
                    # a list containing a single item, the terminal and its data
                    symbol_id = self.dictionary[symbol]
                    if symbol_id in token_stacks and token_stacks[symbol_id]:
                        token_stack = token_stacks[symbol_id]
                        stack.append([(symbol_id, token_stack.pop())])
                        if not token_stack:
                            del token_stacks[symbol_id]
                    else:
                        stack.append([(symbol_id, None)])
                    #print(stack)
                else:
                    #print("slow path for", lhs, "->", rhs)
                    new_prog = []
                    for symbol in reversed(rhs):
                        if symbol[0] == '$':
                            new_prog.extend(stack.pop())
                        else:
                            symbol_id = self.dictionary[symbol]
                            if symbol_id in token_stacks and token_stacks[symbol_id]:
                                token_stack = token_stacks[symbol_id]
                                new_prog.append((symbol_id, token_stack.pop()))
                                if not token_stack:
                                    del token_stacks[symbol_id]
                            else:
                                new_prog.append((symbol_id, None))
                    stack.append(new_prog)
                    #print(stack)
        #print("Stack", stack)
        if len(self.terminals) + top_stack_id != self._start_symbol or \
            len(stack) != 1:
            raise ValueError("Invalid sequence")
        
        assert len(stack) == 1
        # the program is constructed on the stack in reverse order
        # bring it back to the right order
        stack[0].reverse()
        return stack[0]


