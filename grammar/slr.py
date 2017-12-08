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

class SLRParserGenerator():
    '''
    Construct a shift-reduce parser given an SLR grammar.
    
    The grammar must be binarized beforehand.
    '''
    
    def __init__(self, grammar, start_symbol):
        # optimizations first
        self._start_symbol = start_symbol
        self._optimize_grammar(grammar)
        grammar['$ROOT'] = [(start_symbol, EOF_TOKEN)]
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
        return ShiftReduceParser(self.rules, self.action_table, self.goto_table, self._start_symbol)
    
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
                if rhs[0] != '$':
                    terminals.add(rhs)
                else:
                    non_terminals.add(rhs)
        
        self.terminals = list(terminals)
        self.terminals.sort()
        self.non_terminals = list(non_terminals)
        self.non_terminals.sort()

    def print_rules(self, fp=sys.stdout):
        for lhs, rhs in self.rules:
            print(lhs, '->', ' '.join(rhs), file=fp)

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
                if rhs[i] == '*' and rhs[i+1] != EOF_TOKEN:
                    yield rhs[i+1]

    def _advance(self, item_set, token):
        for rule in item_set.rules:
            rule_id, rhs = rule
            for i in range(len(rhs)-1):
                if rhs[i] == '*' and rhs[i+1] == token:
                    yield rule_id, (rhs[:i] + (token, '*') + rhs[i+2:])
                    break
        
    def _make_item_set(self, lhs):
        for rule_id in self.grammar[lhs]:
            lhs, rhs = self.rules[rule_id]
            yield rule_id, ('*',) + rhs
    
    def _close(self, items):
        def _is_nonterminal(symbol):
            return symbol[0] == '$'
        
        item_set = set(items)
        stack = list(item_set)
        while len(stack) > 0:
            item = stack.pop()
            _, rhs = item
            for i in range(len(rhs)-1):
                if rhs[i] == '*' and _is_nonterminal(rhs[i+1]):
                    for new_rule in self._make_item_set(rhs[i+1]):
                        if new_rule in item_set:
                            continue
                        item_set.add(new_rule)
                        stack.append(new_rule)
                    break
        item_set = list(item_set)
        item_set.sort()
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
                    if rhs[i] == '*' and rhs[i+1] == EOF_TOKEN:
                        self.action_table[item_set.info.id][EOF_TOKEN] = ('accept', None)
        
        for item_set in self._item_sets:
            for item in item_set.rules:
                rule_id, rhs = item
                if rhs[-1] != '*':
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
    
    def __init__(self, rules, action_table, goto_table, start_symbol):
        super().__init__()
        self._rules = rules
        self._action_table = action_table
        self._goto_table = goto_table
        self._start_symbol = start_symbol
        
    @property
    def num_rules(self):
        return len(self._rules)
    
    @property
    def num_states(self):
        return len(self._action_table)
        
    def parse(self, sequence):
        stack = [0]
        state = 0
        i = 0
        result = []
        while True:
            if i < len(sequence):
                token = sequence[i]
            else:
                token = EOF_TOKEN
            if token not in self._action_table[state]:
                raise ValueError("Parse error: unexpected token " + token + " in state " + str(state) + ", expected " + str(self._action_table[state].keys()))
            action, param = self._action_table[state][token]
            if action == 'accept':
                return result
            result.append((action, param))
            #if action == 'shift':
            #    print('shift', param, token)
            #else:
            #    print('reduce', param, self._rules[param])
            if action == 'shift':
                state = param
                stack.append(state)
                i += 1
            else:
                rule_id = param
                lhs, rhs = self._rules[rule_id]
                for _ in rhs:
                    stack.pop()
                state = stack[-1]
                state = self._goto_table[state][lhs]
                stack.append(state)

    def reconstruct(self, sequence):
        stacks = dict()
        for action, param in sequence:
            if action == 'shift':
                pass
            elif action == 'accept':
                break
            else:
                rule_id = param
                lhs, rhs = self._rules[rule_id]
                new_prog = []
                for symbol in rhs:
                    new_prog += stacks[symbol].pop() if symbol[0] == '$' else (symbol,)
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
'$rule':    [('$stream', '$action')],
'$command': [('$table', 'notify'),
             ('$table', '$action')],
'$table':   [('$get',),
             ('$table', '$filter')],
'$stream':  [('monitor', '$table')],
'$get':     [('$get', '$ip'),
             ('xkcd.get_comic',),
             ('thermostat.get_temp',),
             ('twitter.search',)],
'$action':  [('$action', '$ip'),
             ('twitter.post',)],
'$ip':      [('param:number', '$number'),
             ('param:text', '$string')],
'$number':  [('num0',),
             ('num1',)],
'$string':  [('qs0',),
             ('qs1',)],
'$filter':  [('param:number', '==', '$number'),
             ('param:number', '>', '$number'),
             ('param:number', '<', '$number'),
             ('param:text', '==', '$string'),
             ('param:text', '=~', '$string')]
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
    if False:
        generator = SLRParserGenerator(TEST_GRAMMAR, '$prog')
        print("Action table:")
        for i, actions in enumerate(generator.action_table):
            print(i, ":", actions)
        
        print()          
        print("Goto table:")
        for i, next_states in enumerate(generator.goto_table):
            print(i, ":", next_states)
        
        parser = generator.build()
        
        print(parser.parse(['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0']))
        print(parser.reconstruct(parser.parse(['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'])))
    else:
        generator = SLRParserGenerator(PARENTHESIS_GRAMMAR, '$S')
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
