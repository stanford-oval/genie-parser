
import tensorflow as tf
import numpy as np

import sys

ENTITIES = ['USERNAME', 'HASHTAG',
            'QUOTED_STRING', 'NUMBER',
            'PHONE_NUMBER', 'EMAIL_ADDRESS', 'URL',
            'DATE', 'TIME', 'SET',
            'PERCENT', 'DURATION', 'MONEY', 'ORDINAL']

BEGIN_TOKENS = ['special', 'answer', 'command', 'rule', 'trigger', 'query', 'action']
SPECIAL_TOKENS = ['tt:root.special.yes', 'tt:root.special.no', 'tt:root.special.hello',
                  'tt:root.special.thankyou', 'tt:root.special.sorry', 'tt:root.special.cool',
                  'tt:root.special.nevermind', 'tt:root.special.debug', 'tt:root.special.failed']
IF = 'if'
THEN = 'then'
OPERATORS = ['is', 'contains', '>', '<', 'has']
VALUES = ENTITIES + ['ENTITY', 'true', 'false', 'absolute', 'rel_home', 'rel_work', 'rel_current_location']
TYPES = {
    'Entity': (['is'], ['ENTITY', 'QUOTED_STRING']),
    'Location': (['is'], ['absolute', 'rel_current_location', 'rel_work', 'rel_home']),
    'Boolean':  (['is'], ['true', 'false']),
    'Bool': (['is'], ['true', 'false']),
    'String': (['is', 'contains'], ['QUOTED_STRING']),
    'Date': (['is'], ['DATE']),
    'Time': (['is'], ['TIME']),
    'Username': (['is'], ['USERNAME', 'QUOTED_STRING']),
    'Hashtag': (['is'], ['HASHTAG', 'QUOTED_STRING']),
    'PhoneNumber': (['is'], ['PHONE_NUMBER', 'QUOTED_STRING']),
    'EmailAddress': (['is'], ['EMAIL_ADDRESS', 'QUOTED_STRING']),
    'URL': (['is'], ['URL']),
    'Number': (['is', '<', '>'], ['NUMBER']),
    'Picture': (['is'], [])
}

UNITS = dict(C=["C", "F"],
             ms=["ms", "s", "min", "h", "day", "week", "month", "year"],
             m=["m", "km", "mm", "cm", "mi", "in", "ft"],
             mps=["mps", "kmph", "mph"],
             kg=["kg", "g", "lb", "oz"],
             kcal=["kcal", "kJ"],
             bpm=["bpm"],
             byte=["byte", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB"])

COMMAND_TOKENS = ['list', 'help', 'generic', 'device', 'command', 'make', 'rule', 'configure', 'discover']

class ThingtalkGrammar(object):
    def __init__(self):
        triggers = dict()
        queries = dict()
        actions = dict()
        functions = dict(trigger=triggers, query=queries, action=actions)
        devices = []
        trigger_or_query_params = set()

        tokens = set()
        tokens.update(BEGIN_TOKENS)
        tokens.add(IF)
        tokens.add(THEN)
        tokens.update(OPERATORS)
        tokens.update(VALUES)
        tokens.update(COMMAND_TOKENS)
        tokens.update(SPECIAL_TOKENS)
        for unitlist in UNITS.itervalues():
            tokens.update(unitlist)
        tokens.add('tt:param.$event')
        trigger_or_query_params.add('tt:param.$event')
        
        enum_types = dict()
        
        with open('thingpedia.txt', 'r') as fp:
            for line in fp.readlines():
                line = line.strip().split()
                function_type = line[0]
                function = line[1]
                if function_type == 'device':
                    devices.append(function)
                    tokens.add(function)
                    continue
                
                parameters = line[2:]
                paramlist = []
                functions[function_type][function] = paramlist
                tokens.add(function)
                
                for i in xrange(len(parameters)/2):
                    param = parameters[2*i]
                    type = parameters[2*i+1]
                    
                    # lose the entity type information for now, to keep the mess manageable
                    if type.startswith('Entity('):
                        type = 'Entity'
                    if type.startswith('Array(Entity('):
                        type = 'Array(Entity)'
                    
                    paramlist.append((param, type))
                    tokens.add('tt:param.' + param)
                    if function_type != 'action':
                        trigger_or_query_params.add('tt:param.' + param)
                    
                    if type.startswith('Array('):
                        elementtype = type[len('Array('):-1]
                    else:
                        elementtype = type
                    if elementtype.startswith('Enum('):
                        enums = elementtype[len('Enum('):-1].split(',')
                        for enum in enums:
                            tokens.add(enum)
                        if not elementtype in enum_types:
                            enum_types[elementtype] = enums
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>'] + list(tokens)
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
            
        self.output_size = len(self.tokens)
        
        self.start = self.dictionary['<<GO>>']
        self.end = self.dictionary['<<EOS>>']
        
        # build a DFA that will parse the thingtalk-ish code

        states = []
        transitions = []
        state_names = []
        
        def to_ids(tokens, words):
            return list(map(lambda x: words[x], tokens))

        def add_allowed_tokens(state, tokens):
            state[to_ids(tokens, self.dictionary)] = 1
        
        def new_state(name):
            state = np.zeros((self.output_size,))
            states.append(state)
            state_names.append(name)
            return len(states)-1
        
        # start with one of the begin tokens
        self.start_state = new_state('start')
        
        # in the before end state we just wait for EOS
        self.before_end_state = new_state('before_end')
        
        # in the end state we are done
        self.end_state = new_state('end')
        transitions.append((self.before_end_state, self.end_state, '<<EOS>>'))
        transitions.append((self.end_state, self.end_state, '<<PAD>>'))
        
        # special
        special_id = new_state('special')
        transitions.append((self.start_state, special_id, 'special'))
        for t in SPECIAL_TOKENS:
            transitions.append((special_id, self.before_end_state, t))
            
        # command
        command_id = new_state('command')
        transitions.append((self.start_state, command_id, 'command'))
        # list command
        list_id = new_state('list')
        transitions.append((command_id, list_id, 'list'))
        for t in ['generic', 'device', 'command']:
            transitions.append((list_id, self.before_end_state, t))
        # help/configure/discover command
        help_id = new_state('help_configure_discover')
        for t in ['help', 'configure', 'discover']:
            transitions.append((command_id, help_id, t))
        transitions.append((help_id, self.before_end_state, 'generic'))
        for d in devices:
            transitions.append((help_id, self.before_end_state, d))
        # make rule
        make_id = new_state('make')
        transitions.append((command_id, make_id, 'make'))
        transitions.append((make_id, self.before_end_state, 'rule'))
        
        # answers
        answer_id = new_state('answer')
        transitions.append((self.start_state, answer_id, 'answer'))
        for v in VALUES:
            transitions.append((answer_id, self.before_end_state, v))
        
        # primitives
        actions_id = new_state('action')
        transitions.append((self.start_state, actions_id, 'action'))
        queries_id = new_state('query')
        transitions.append((self.start_state, queries_id, 'query'))
        triggers_id = new_state('trigger')
        transitions.append((self.start_state, triggers_id, 'trigger'))
        
        def do_invocation(invocation_name, params, for_action=False, can_have_scope=False):
            state_id = new_state(invocation_name)
            
            # go to each parameter
            for param_name, param_type in params:
                if param_type in ('Any'):
                    continue
                if param_type in ('Picture', 'Array(Picture)') and not can_have_scope:
                    continue
                elementtype = param_type
                is_array = False
                is_measure = False
                if param_type.startswith('Array('):
                    is_array = True
                    elementtype = param_type[len('Array('):-1]
                if elementtype.startswith('Measure('):
                    is_measure = True
                    operators = ['is', '<', '>']
                    values = UNITS[elementtype[len('Measure('):-1]]
                elif elementtype.startswith('Enum('):
                    operators = ['is']
                    values = enum_types[elementtype]
                else:
                    operators, values = TYPES[elementtype]
                if is_array:
                    if for_action:
                        continue
                    else:
                        operators = ['has']
                elif for_action:
                    operators = ['is']
                
                before_op = new_state(invocation_name + '_tt:param.' + param_name)
                transitions.append((state_id, before_op, 'tt:param.' + param_name))
                before_value = new_state(invocation_name + '_tt:param.' + param_name + '_value')

                for op in operators:
                    transitions.append((before_op, before_value, op))
                if is_measure:
                    before_unit = new_state(invocation_name + '_tt:param.' + param_name + '_unit')
                    transitions.append((before_value, before_unit, 'NUMBER'))
                    for unit in values:
                        transitions.append((before_unit, state_id, unit))
                else:
                    for v in values:
                        transitions.append((before_value, state_id, v))
                if can_have_scope:
                    for v in trigger_or_query_params:
                        transitions.append((before_value, state_id, v))
                    
            return state_id

        for action_name, params in actions.iteritems():
            state_id = do_invocation(action_name, params, for_action=True)
            transitions.append((actions_id, state_id, action_name))
            transitions.append((state_id, self.end_state, '<<EOS>>'))
        for query_name, params in queries.iteritems():
            state_id = do_invocation(query_name, params, for_action=False)
            transitions.append((queries_id, state_id, query_name))
            transitions.append((state_id, self.end_state, '<<EOS>>'))
        for trigger_name, params in triggers.iteritems():
            state_id = do_invocation(trigger_name, params, for_action=False)
            transitions.append((triggers_id, state_id, trigger_name))
            transitions.append((state_id, self.end_state, '<<EOS>>'))
        
        # rules
        rule_id = new_state('rule')
        transitions.append((self.start_state, rule_id, 'rule'))
        if_id = new_state('if')
        transitions.append((rule_id, if_id, 'if'))
        then_to_query_id = new_state('then_to_query_or_action')
        then_to_action_id = new_state('then_to_action')
        for trigger_name, params in triggers.iteritems():
            state_id = do_invocation(trigger_name, params, for_action=False)
            transitions.append((if_id, state_id, trigger_name))
            transitions.append((state_id, then_to_query_id, 'then'))
        for query_name, params in queries.iteritems():
            state_id = do_invocation(query_name, params, for_action=False)
            transitions.append((rule_id, state_id, query_name))
            transitions.append((state_id, then_to_action_id, 'then'))
            
            state_id = do_invocation(query_name, params, for_action=False, can_have_scope=True)
            transitions.append((then_to_query_id, state_id, query_name))
            transitions.append((state_id, then_to_action_id, 'then'))
            transitions.append((state_id, self.end_state, '<<EOS>>'))
        for action_name, params in actions.iteritems():
            state_id = do_invocation(action_name, params, for_action=True, can_have_scope=True)
            transitions.append((then_to_query_id, state_id, action_name))
            transitions.append((then_to_action_id, state_id, action_name))
            transitions.append((state_id, self.end_state, '<<EOS>>'))
        
        # now build the actual DFA
        num_states = len(states)
        self.num_states = num_states
        print "num states", num_states
        print "num tokens", self.output_size
        self.transition_matrix = np.zeros((num_states, self.output_size), dtype=np.int32)
        self.allowed_token_matrix = np.zeros((num_states, self.output_size), dtype=np.bool8)

        for from_state, to_state, token in transitions:
            token_id = self.dictionary[token]
            if self.transition_matrix[from_state, token_id] != 0 and \
                self.transition_matrix[from_state, token_id] != to_state:
                raise ValueError("Ambiguous transition around token " + token + " in state " + state_names[from_state])
            self.transition_matrix[from_state, token_id] = to_state
            self.allowed_token_matrix[from_state, token_id] = True
        self.state_names = state_names

    def dump_tokens(self):
        for token in self.tokens:
            print token

    def parse(self, program):
        curr_state = self.start_state
        for token in program:
            try:
                token_id = self.dictionary[token]
            except KeyError:
                raise ValueError("Unknown token " + token + " in " + (' '.join(program)) + " (in state " + self.state_names[curr_state] + ")")
            next = self.transition_matrix[curr_state, token_id]
            if next == 0:
                raise ValueError("Unexpected token " + token + " in " + (' '.join(program)) + " (in state " + self.state_names[curr_state] + ")")
            curr_state = next
            
        if curr_state != self.end_state:
            raise ValueError("Premature end of program in " + (' '.join(program)) + " (in state " + self.state_names[curr_state] + ")")

    def parse_all(self, fp):
        for line in fp.readlines():
            try:
                program = line.strip().split()
                program.append('<<EOS>>')
                self.parse(program)
            except ValueError, e:
                print e

    def constrain(self, logits, curr_state, batch_size, dtype=tf.int32):
        if curr_state is None:
            return tf.ones((batch_size,), dtype=dtype) * self.start, tf.ones((batch_size,), dtype=tf.int32) * self.start_state

        transitions = tf.gather(tf.constant(self.transition_matrix), curr_state)
        assert transitions.get_shape()[1:] == (self.output_size,)
        allowed_tokens = tf.gather(tf.constant(self.allowed_token_matrix), curr_state)
        assert allowed_tokens.get_shape()[1:] == (self.output_size,)
        
        constrained_logits = logits - tf.to_float(tf.logical_not(allowed_tokens)) * 100000
        choice = tf.cast(tf.argmax(constrained_logits, axis=1), dtype=dtype)
        indices = tf.stack((tf.range(0, batch_size), choice), axis=1)
            
        #print choice.get_shape()
        #print transitions.get_shape()
        next_state = tf.gather_nd(transitions, indices)
        return choice, next_state

    def decode_output(self, sequence):
        output = []
        curr_state = self.start_state
        for logits in sequence:
            assert logits.shape == (self.output_size,)
            allowed_tokens = self.allowed_token_matrix[curr_state]
            constrained_logits = logits - np.logical_not(allowed_tokens).astype(np.float32) * 100000
            word_idx = np.argmax(constrained_logits)
            if word_idx > 0:
                output.append(word_idx)
            curr_state = self.transition_matrix[curr_state, word_idx]
        return output

class SimpleGrammar():
    def __init__(self, filename):
        tokens = set()
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                tokens.add(line.strip().lower())
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>'] + list(tokens)
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
            
        self.output_size = len(self.tokens)
        
        self.start = self.dictionary['<<GO>>']
        self.end = self.dictionary['<<EOS>>']
        
    def constrain(self, logits, curr_state, batch_size, dtype=tf.int32):
        if curr_state is None:
            return tf.ones((batch_size,), dtype=dtype) * self.start, ()
        else:
            return tf.argmax(logits, axis=1), ()

    def decode_output(self, sequence):
        output = []
        for logits in sequence:
            assert logits.shape == (self.output_size,)
            word_idx = np.argmax(logits)
            if word_idx > 0:
                output.append(word_idx)
        return output

if __name__ == '__main__':
    grammar = ThingtalkGrammar()
    #grammar.dump_tokens()
    grammar.parse_all(sys.stdin)
    #for i, name in enumerate(grammar.state_names):
    #    print i, name
