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

import sys

EOF_TOKEN = '<<EOF>>'

class ItemSetInfo:
    def __init__(self):
        self.id = 0
        self.intransitions = set()
        self.outtransitions = set()


class ItemSet:
    def __init__(self, rules):
        self.rules = list(rules)
    def __hash__(self):
        h = 0
        for el in self.rules:
            h ^= hash(el)
        return h
    def __eq__(self, other):
        return self.rules == other.rules

DEBUG = False

ITEM_SET_SEP = ()


class SLRParserGenerator():
    '''
    Construct a shift-reduce parser given an SLR grammar.
    
    The grammar must be binarized beforehand.
    '''
    
    def __init__(self, grammar, extensible_terminals, start_symbol):
        # optimizations first
        self._start_symbol = start_symbol
        self._optimize_grammar(grammar)
        grammar['$ROOT'] = [(start_symbol, EOF_TOKEN)]
        
        self._extensible_terminals = extensible_terminals
        self._number_rules(grammar)
        self._extract_terminals_non_terminals()
        self._build_first_sets()
        self._build_follow_sets()
        self._generate_all_item_sets()
        self._build_state_transition_matrix()
        self._build_parse_tables()
        
        self._check_first_sets()
        self._check_follow_sets()
        
    def build(self):
        # the last rule is $ROOT -> $input <<EOF>>
        # which is a pseudo-rule needed for the SLR generator
        # we ignore it here
        return ShiftReduceParser(self.rules[:-1], self.action_table, self.goto_table, self._start_symbol,
                                 self._extensible_terminals)
    
    def _optimize_grammar(self, grammar):
        progress = True
        i = 0
        while progress:
            progress = False
            if DEBUG:
                print("Optimization pass", i+1)            
            progress = self._remove_empty_nonterminals(grammar) or progress
            progress = self._remove_unreachable_nonterminals(grammar) or progress
    
    def _remove_empty_nonterminals(self, grammar):
        progress = True
        any_change = False
        deleted = set()
        while progress:
            progress = False
            for lhs, rules in grammar.items():
                if len(rules) == 0:
                    if lhs not in deleted:
                        if DEBUG:
                            print("Non-terminal", lhs, "is empty, deleted")
                        progress = True
                        any_change = True
                    deleted.add(lhs)
                else:
                    new_rules = []
                    any_rules_deleted = False
                    for rule in rules:
                        rule_is_deleted = False
                        for rhs in rule:
                            if rhs in deleted:
                                rule_is_deleted = True
                                break
                        if not rule_is_deleted:
                            new_rules.append(rule)
                        else:
                            if DEBUG:
                                print("Rule", lhs, "->", rule, "deleted")
                            any_rules_deleted = True
                    if any_rules_deleted:
                        grammar[lhs] = new_rules
                        progress = True
                        any_change = True
        for lhs in deleted:
            del grammar[lhs]
        return any_change
    
    def _remove_unreachable_nonterminals(self, grammar):
        stack = [self._start_symbol]
        visited = set()
        while len(stack) > 0:
            nonterm = stack.pop()
            if nonterm in visited:
                continue
            visited.add(nonterm)
            for rhs in grammar[nonterm]:
                for rhs_token in rhs:
                    if rhs_token[0] == '$' and rhs_token not in visited:
                        stack.append(rhs_token)
                        if not rhs_token in grammar:
                            raise ValueError("Non-terminal " + str(rhs_token) + " does not exist, in rule " + nonterm + " -> " + str(rhs))
                        
        todelete = set()
        anychange = False
        for lhs in grammar:
            if lhs not in visited:
                if DEBUG:
                    print("Non-terminal " + lhs + " is not reachable, deleted")
                todelete.add(lhs)
                anychange = True
        for lhs in todelete:
            del grammar[lhs]
        return anychange
    
    def _check_first_sets(self):
        for lhs, first_set in self._first_sets.items():
            if len(first_set) == 0:
                print("WARNING: non-terminal " + lhs + " cannot start with any terminal")
    
    def _check_follow_sets(self):
        for lhs, follow_set in self._follow_sets.items():
            if lhs == '$ROOT':
                continue
            if len(follow_set) == 0:
                print("WARNING: non-terminal " + lhs + " cannot be followed by any terminal")
    
    def _extract_terminals_non_terminals(self):
        terminals = set()
        non_terminals = set()
        for lhs, rule in self.rules:
            non_terminals.add(lhs)
            for rhs in rule:
                assert isinstance(rhs, str)
                if rhs[0] != '$':
                    terminals.add(rhs)
                else:
                    non_terminals.add(rhs)
        
        self.terminals = list(terminals)
        self.terminals.sort()
        self.non_terminals = list(non_terminals)
        self.non_terminals.sort()

    def print_rules(self, fp=sys.stdout):
        for i, (lhs, rhs) in enumerate(self.rules):
            print(i, lhs, '->', ' '.join(rhs), file=fp)

    def _number_rules(self, grammar):
        self.rules = []
        self.grammar = dict()
        for lhs, rules in grammar.items():
            self.grammar[lhs] = []
            for rule in rules:
                assert isinstance(rule, tuple)
                #assert len(rule) == 1 or len(rule) == 2
                rule_id = len(self.rules)
                self.rules.append((lhs, rule))
                self.grammar[lhs].append(rule_id)
                if DEBUG:
                    print(rule_id, lhs, '->', rule)

    def _item_set_followers(self, item_set):
        for rule in item_set.rules:
            _, rhs = rule
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and rhs[i+1] != EOF_TOKEN:
                    yield rhs[i+1]

    def _advance(self, item_set, token):
        for rule in item_set.rules:
            rule_id, rhs = rule
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and rhs[i+1] == token:
                    yield rule_id, (rhs[:i] + (token, ITEM_SET_SEP) + rhs[i+2:])
                    break
        
    def _make_item_set(self, lhs):
        for rule_id in self.grammar[lhs]:
            lhs, rhs = self.rules[rule_id]
            yield rule_id, (ITEM_SET_SEP,) + rhs
    
    def _close(self, items):
        def _is_nonterminal(symbol):
            return symbol[0] == '$'
        
        item_set = set(items)
        stack = list(item_set)
        while len(stack) > 0:
            item = stack.pop()
            _, rhs = item
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and _is_nonterminal(rhs[i+1]):
                    for new_rule in self._make_item_set(rhs[i+1]):
                        if new_rule in item_set:
                            continue
                        item_set.add(new_rule)
                        stack.append(new_rule)
                    break
        item_set = list(item_set)
        #item_set.sort()
        return item_set
    
    def _generate_all_item_sets(self):
        item_sets = dict()
        i = 0
        item_set0 = ItemSet(self._close(self._make_item_set('$ROOT')))
        item_set0_info = ItemSetInfo()
        item_sets[item_set0] = item_set0_info
        i += 1
        queue = []
        queue.append(item_set0)
        while len(queue) > 0:
            item_set = queue.pop(0)
            my_info = item_sets[item_set]
            for next_token in self._item_set_followers(item_set):
                new_set = ItemSet(self._close(self._advance(item_set, next_token)))
                if new_set in item_sets:
                    info = item_sets[new_set]
                else:
                    info = ItemSetInfo()
                    info.id = i
                    i += 1
                    item_sets[new_set] = info
                    queue.append(new_set)
                info.intransitions.add((my_info.id, next_token))
                my_info.outtransitions.add((info.id, next_token))
        
        for item_set, info in item_sets.items():
            item_set.info = info
            if DEBUG:
                print("Item Set", item_set.info.id, item_set.info.intransitions)
                for rule in item_set.rules:
                    rule_id, rhs = rule
                    lhs, _ = self.rules[rule_id]
                    print(rule_id, lhs, '->', rhs)
                print()
            
        item_set_list = [None] * len(item_sets.keys())
        for item_set in item_sets.keys():
            item_set_list[item_set.info.id] = item_set
        self._item_sets = item_set_list
        self._n_states = len(self._item_sets)
    
    def _build_state_transition_matrix(self):
        self._state_transition_matrix = [dict() for  _ in range(self._n_states)]
        
        for item_set in self._item_sets:
            for next_id, next_token in item_set.info.outtransitions:
                if next_token in self._state_transition_matrix[item_set.info.id]:
                    raise ValueError("Ambiguous transition from", item_set.info.id, "through", next_token, "to", self._state_transition_matrix[item_set.info.id], "and", next_id)
                self._state_transition_matrix[item_set.info.id][next_token] = next_id
                
    def _build_first_sets(self):
        def _is_terminal(symbol):
            return symbol[0] != '$'
        
        first_sets = dict()
        for nonterm in self.non_terminals:
            first_sets[nonterm] = set()
        progress = True
        while progress:
            progress = False
            for lhs, rules in self.grammar.items():
                union = set()
                for rule_id in rules:
                    _, rule = self.rules[rule_id]
                    # Note: our grammar doesn't include rules of the form A -> epsilon
                    # because it's meant for an SLR parser not an LL parser, so this is
                    # simpler than what Wikipedia describes in the LL parser article
                    if _is_terminal(rule[0]):
                        first_set_rule = set([rule[0]])
                    else:
                        first_set_rule = first_sets.get(rule[0], set())
                    union |= first_set_rule
                if union != first_sets[lhs]:
                    first_sets[lhs] = union
                    progress = True
                    
        self._first_sets = first_sets
        
    def _build_follow_sets(self):
        follow_sets = dict()
        for nonterm in self.non_terminals:
            follow_sets[nonterm] = set()
        
        progress = True
        def _add_all(from_set, into_set):
            progress = False
            for v in from_set:
                if v not in into_set:
                    into_set.add(v)
                    progress = True
            return progress
        def _is_nonterminal(symbol):
            return symbol[0] == '$'
        
        while progress:
            progress = False
            for lhs, rule in self.rules:
                for i in range(len(rule)-1):
                    if _is_nonterminal(rule[i]):
                        if _is_nonterminal(rule[i+1]):
                            progress = _add_all(self._first_sets[rule[i+1]], follow_sets[rule[i]]) or progress
                        else:
                            if rule[i+1] not in follow_sets[rule[i]]:
                                follow_sets[rule[i]].add(rule[i+1])
                                progress = True
                if _is_nonterminal(rule[-1]):
                    progress = _add_all(follow_sets[lhs], follow_sets[rule[-1]]) or progress
                                    
        self._follow_sets = follow_sets
        if DEBUG:
            print()
            print("Follow sets")
            for nonterm, follow_set in follow_sets.items():
                print(nonterm, "->", follow_set)
            
    def _build_parse_tables(self):
        self.goto_table = [dict() for _ in range(self._n_states)]
        self.action_table = [dict() for _ in range(self._n_states)]
        
        for nonterm in self.non_terminals:
            for i in range(self._n_states):
                if nonterm in self._state_transition_matrix[i]:
                    self.goto_table[i][nonterm] = self._state_transition_matrix[i][nonterm]
        for term in self.terminals:
            for i in range(self._n_states):
                if term in self._state_transition_matrix[i]:
                    self.action_table[i][term] = ('shift', self._state_transition_matrix[i][term])
                    
        for item_set in self._item_sets:
            for item in item_set.rules:
                _, rhs = item
                for i in range(len(rhs)-1):
                    if rhs[i] == ITEM_SET_SEP and rhs[i+1] == EOF_TOKEN:
                        self.action_table[item_set.info.id][EOF_TOKEN] = ('accept', None)
        
        for item_set in self._item_sets:
            for item in item_set.rules:
                rule_id, rhs = item
                if rhs[-1] != ITEM_SET_SEP:
                    continue
                lhs, _ = self.rules[rule_id]
                for term in self.terminals:
                    if term in self._follow_sets.get(lhs, set()):
                        if term in self.action_table[item_set.info.id] and self.action_table[item_set.info.id][term] != ('reduce', rule_id):
                            print("Item Set", item_set.info.id, item_set.info.intransitions)
                            for rule in item_set.rules:
                                rule_id, rhs = rule
                                lhs, _ = self.rules[rule_id]
                                print(rule_id, lhs, '->', rhs)
                            print()
                            raise ValueError("Conflict for state", item_set.info.id, "terminal", term, "want", ("reduce", rule_id), "have", self.action_table[item_set.info.id][term])
                        self.action_table[item_set.info.id][term] = ('reduce', rule_id)

       
class ShiftReduceParser:
    '''
    A bottom-up parser for a deterministic CFG language, based on shift-reduce
    tables.
    
    The parser can transform a string in the language to a sequence of
    shifts and reduces, and can transform a valid sequence of reduces to
    a string in the language.
    '''
    
    def __init__(self, rules, action_table, goto_table, start_symbol, extensible_terminals):
        super().__init__()
        self.rules = rules
        self._action_table = action_table
        self._goto_table = goto_table
        self._start_symbol = start_symbol
        
        self._extensible_terminals = extensible_terminals
        # reverse the extensible terminal dictionary
        self._reverse_extensible_terminals = dict()
        for term, values in extensible_terminals.items():
            for idx, value in enumerate(values):
                if value in self._reverse_extensible_terminals:
                    raise ValueError('Ambiguous concrete token ' + value)
                self._reverse_extensible_terminals[value] = (term, idx)
        
    @property
    def num_rules(self):
        # the last rule is $ROOT -> $input <<EOF>>
        # which is a pseudo-rule needed for the SLR generator
        # we ignore it here
        return len(self.rules)-1
    
    @property
    def num_states(self):
        return len(self._action_table)
    
# <<<<<<< HEAD
#     def parse_reverse(self, sequence):
#         bottom_up_sequence = self.parse(sequence)
#         lens = [None] * len(bottom_up_sequence)
#         children = [None] * len(bottom_up_sequence)
#         reduces = [None] * len(bottom_up_sequence)
#         i = 0
#         for action,param in bottom_up_sequence:
#             if action == 'shift':
#                 continue
#             lhs, rhs = self.rules[param]
#             current_child = i-1
#             my_length = 1
#             my_children = []
#             for rhsitem in reversed(rhs):
#                 if rhsitem.startswith('$'):
#                     my_children.append(current_child)
#                     my_length += lens[current_child]
#                     current_child -= lens[current_child]
#             lens[i] = my_length
#             reduces[i] = (action,param)
#             children[i] = tuple(reversed(my_children))
#             i += 1
#         reversed_sequence = []
#         def write_subsequence(node, start):
#             reversed_sequence.append(reduces[node])
#             for c in children[node]:
#                 write_subsequence(c, start)
#                 start += lens[c]
#         write_subsequence(i-1, 0)
#         return reversed_sequence
# =======
    @property
    def extensible_terminals(self):
        return self._extensible_terminals
# >>>>>>> wip/extensible-aligner
        
    def parse(self, sequence):
        stack = [0]
        state = 0
        i = 0
        result = []
        
        sequence_iter = iter(sequence)
        sequence_done = False
        token = next(sequence_iter)
        while True:

            if i < len(sequence):
                token = sequence[i]
            else:
                token = EOF_TOKEN
            if token in self._reverse_extensible_terminals:
                terminal, tokenidx = self._reverse_extensible_terminals[token]
            else:
                terminal = token
                tokenidx = 0
                
            if terminal not in self._action_table[state]:
                raise ValueError("Parse error: unexpected token " + token + " in state " + str(state) + ", expected " + str(self._action_table[state].keys()))
            action, param = self._action_table[state][terminal]
            if action == 'accept':
                return result
            #if action == 'shift':
            #    print('shift', param, token)
            #else:
            #    print('reduce', param, self._rules[param])
            if action == 'shift':
                #print('shift', param, token)
                state = param
                result.append(('shift', (terminal, tokenidx)))
                stack.append(state)
                
                try:
                    token = next(sequence_iter)
                except StopIteration:
                    token = EOF_TOKEN
            else:
                rule_id = param
                result.append(('reduce', rule_id))
                lhs, rhs = self.rules[rule_id]
                #print('reduce', lhs, '->', rhs)
                for _ in rhs:
                    stack.pop()
                state = stack[-1]
                state = self._goto_table[state][lhs]
                stack.append(state)
                
    def reconstruct_reverse(self, sequence):
        output_sequence = []
        if not isinstance(sequence, list):
            sequence = list(sequence)
        
        def recurse(start_at):
            action, param = sequence[start_at]
            if action != 'reduce':
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
        stacks = dict()
        for action, param in sequence:
            if action == 'shift':
                term, tokenidx = param
                if term in self._extensible_terminals:
                    token = self._extensible_terminals[term][tokenidx]
                    if 'terminal_' + term not in stacks:
                        stacks['terminal_' + term] = [token]
                    else:
                        stacks['terminal_' + term].append(token)
            elif action == 'accept':
                break
            else:
                rule_id = param
                lhs, rhs = self.rules[rule_id]
                new_prog = []
                for symbol in reversed(rhs):
                    if symbol[0] == '$':
                        new_prog = stacks[symbol].pop() + new_prog
                    elif symbol in self._extensible_terminals:
                        new_prog.insert(0, stacks['terminal_' + symbol].pop())
                    else:
                        new_prog.insert(0, symbol)
                if lhs not in stacks:
                    stacks[lhs] = []
                stacks[lhs].append(new_prog)
        #print("Stacks")
        #for cat, progs in stacks.items():
        #    for prog in progs:
        #        print(cat, ":", prog)
        if len(stacks.get(self._start_symbol, ())) != 1:
            raise ValueError("Cannot reconstruct, ambiguous parse")
        return stacks[self._start_symbol][0]


TEST_GRAMMAR = {
'$prog':    [('$command',),
             ('$rule',)],

'$rule':    [('$stream', '$action'),
             ('$stream', 'notify')],
'$command': [('$table', 'notify'),
             ('$table', '$action')],
'$table':   [('$get',),
             ('$table', 'filter', '$filter')],
'$stream':  [('monitor', '$table')],
'$get':     [('$get', '$ip'),
             ('GET',)],
'$action':  [('$action', '$ip'),
             ('DO',)],
'$ip':      [('PARAM', '$number'),
             ('PARAM', '$string')],
'$number':  [('num0',),
             ('num1',)],
'$string':  [('qs0',),
             ('qs1',)],
'$filter':  [('PARAM', '==', '$number'),
             ('PARAM', '>', '$number'),
             ('PARAM', '<', '$number'),
             ('PARAM', '==', '$string'),
             ('PARAM', '=~', '$string')]
}
TEST_TERMINALS = {
    'PARAM': ['param:number', 'param:text'],
    'GET': ['xkcd.get_comic', 'thermostat.get_temp', 'twitter.search'],
    'DO': ['twitter.post']
}

# The grammar of nesting parenthesis
# this is context free but not regular
# (also not parsable with a Petri-net)
#
# The reduce sequence will be: the reduction of the inner most parenthesis pair,
# followed by the reduction for the next parenthesis, in order
#
# This has one important consequence: if the NN produces the sequence
# X Y Z* ...
# where X is reduction 4 or 5 (a or b), Y is reduction 0 or 1, and Z
# is 2 or 3, it will automatically produce a well-formed string in the language
# The NN is good at producing never ending sequences of the same thing (in
# fact, it tends to do that too much), so it should have no trouble with
# this language 
PARENTHESIS_GRAMMAR = {
'$S': [('(', '$V', ')'),
       ('[', '$V', ']'),
       ('(', '$S', ')'),
       ('[', '$S', ']')],
'$V': [('a',), ('b',)]
}


if __name__ == '__main__':
    if True:

        generator = SLRParserGenerator(TEST_GRAMMAR, TEST_TERMINALS, start_symbol='$prog')
        print("Action table:")
        for i, actions in enumerate(generator.action_table):
            print(i, ":", actions)

        #print()          
        #print("Goto table:")
        #for i, next_states in enumerate(generator.goto_table):
        #    print(i, ":", next_states)
        
        parser = generator.build()
        

        print(parser.parse(['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0']))
        
        TEST_VECTORS = [
            ['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'],
            ['monitor', 'thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify'],
            ['thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify']
        ]
        
        for expected in TEST_VECTORS:
            assert expected == parser.reconstruct(parser.parse(expected))
    else:
        generator = SLRParserGenerator(PARENTHESIS_GRAMMAR, extensible_terminals=dict(), start_symbol='$S')
        print("Action table:")
        for i, actions in enumerate(generator.action_table):
            print(i, ":", actions)
        
        print()          
        print("Goto table:")
        for i, next_states in enumerate(generator.goto_table):
            print(i, ":", next_states)
        
        parser = generator.build()
        
        print(parser.parse(['(', '(', '(', 'a', ')', ')', ')']))
        print(parser.parse(['[', '[', '[', 'a', ']', ']', ']']))
        print(parser.parse(['(', '[', '(', 'b', ')', ']', ')']))
