
import tensorflow as tf
import numpy as np
import json
import os
import urllib.request
import ssl
import re
import sys
import itertools

from orderedset import OrderedSet
from collections import OrderedDict

from .abstract import AbstractGrammar

ENTITIES = ['USERNAME', 'HASHTAG',
            'QUOTED_STRING', 'NUMBER',
            'PHONE_NUMBER', 'EMAIL_ADDRESS', 'URL',
            'DATE', 'TIME', 'DURATION',
            'LOCATION']

BEGIN_TOKENS = ['bookkeeping', 'rule', 'policy', 'setup']
BOOKKEEPING_TOKENS = ['special', 'answer', 'command', 'help', 'generic']
SPECIAL_TOKENS = ['tt:root.special.yes', 'tt:root.special.no', 'tt:root.special.nevermind',
                  'tt:root.special.makerule', 'tt:root.special.failed']

OPERATORS = ['if', 'and', 'or', '=', 'contains', '>', '<', '>=', '<=', 'has', 'starts_with', 'ends_with']
VALUES = ['true', 'false', 'absolute', 'rel_home', 'rel_work', 'rel_current_location', '1', '0']
TYPES = {
    'Location': (['='], ['LOCATION', 'rel_current_location', 'rel_work', 'rel_home']),
    'Boolean':  (['='], ['true', 'false']),
    'Bool': (['='], ['true', 'false']),
    'String': (['=', 'contains', 'starts_with', 'ends_with'], ['QUOTED_STRING']),
    'Date': (['=', '>', '<'], ['DATE']),
    'Time': (['='], ['TIME']),
    'Number': (['=', '<', '>', '>=', '<='], ['NUMBER', '1', '0']),
    'Entity(tt:contact)': (['='], ['USERNAME', 'QUOTED_STRING']),
    'Entity(tt:username)': (['='], ['USERNAME', 'QUOTED_STRING']),
    'Entity(tt:hashtag)': (['='], ['HASHTAG', 'QUOTED_STRING']),
    'Entity(tt:phone_number)': (['='], ['USERNAME', 'PHONE_NUMBER', 'QUOTED_STRING']),
    'Entity(tt:email_address)': (['='], ['USERNAME', 'EMAIL_ADDRESS', 'QUOTED_STRING']),
    'Entity(tt:url)': (['='], ['URL', 'QUOTED_STRING']),
    'Entity(tt:picture)': (['='], [])
}
TYPE_RENAMES = {
    'Username': 'Entity(tt:username)',
    'Hashtag': 'Entity(tt:hashtag)',
    'PhoneNumber': 'Entity(tt:phone_number)',
    'EmailAddress': 'Entity(tt:email_address)',
    'URL': 'Entity(tt:url)',
    'Picture': 'Entity(tt:picture)'
}

UNITS = dict(C=["C", "F"],
             ms=["ms", "s", "min", "h", "day", "week", "month", "year"],
             m=["m", "km", "mm", "cm", "mi", "in", "ft"],
             mps=["mps", "kmph", "mph"],
             kg=["kg", "g", "lb", "oz"],
             kcal=["kcal", "kJ"],
             bpm=["bpm"],
             byte=["byte", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB"])

MAX_ARG_VALUES = 8

# first token is "special", "command" or "answer"
# "specials": yes, no, makerule...
# single token answers: "home" "work" "true" "false" etc.
# "help" + "generic"
# "help" + device
# two token answers: NUMBER_i + unit
# three tokens + EOS = 4
MAX_SPECIAL_LENGTH = 4
MAX_PRIMITIVE_LENGTH = 30

def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()

def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', name.lower())

class ThingtalkGrammar(AbstractGrammar):
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.init_from_file(filename)
        
    def reset(self):
        triggers = OrderedDict()
        queries = OrderedDict()
        actions = OrderedDict()
        functions = dict(triggers=triggers, queries=queries, actions=actions)
        self.functions = functions
        self.entities = OrderedSet()
        self.devices = []
        self._trigger_or_query_params = set()

        self._enum_types = OrderedDict()

        # Token order:
        # first the padding, go and end of sentence
        # then the begin tokens
        # then triggers - queries - actions
        # in this order
        # then parameters names
        # then operators
        # then values
        # then entity tokens
        #
        # This order is important as it affects the 3-part aligner
        # algorithm

        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>']
        self.num_control_tokens = 3
        self.tokens += BEGIN_TOKENS
        self.num_begin_tokens = len(BEGIN_TOKENS)
        
        self._token_canonicals = dict()
        
        # add the special functions
        functions['triggers']['tt:$builtin.now'] = []
        self._token_canonicals['tt:$builtin.now'] = 'now'
        functions['queries']['tt:$builtin.noop'] = []
        self._token_canonicals['tt:$builtin.noop'] = 'nothing'
        functions['actions']['tt:$builtin.notify'] = []
        self._token_canonicals['tt:$builtin.notify'] = 'notify'
        functions['actions']['tt:$builtin.return'] = []
        self._token_canonicals['tt:$builtin.return'] = 'return'
        
        self._param_tokens = OrderedSet()
        self._param_tokens.add('tt-param:$event')
        self._trigger_or_query_params.add('tt-param:$event')
        self._token_canonicals['tt-param:$event'] = 'the event'
    
    def _process_devices(self, devices):
        for device in devices:
            if device['kind_type'] == 'global':
                continue
            self.devices.append('tt-device:' + device['kind'])
            self._token_canonicals['tt-device:' + device['kind']] = device['kind_canonical']
            
            for function_type in ('triggers', 'queries', 'actions'):
                for name, function in device[function_type].items():
                    function_name = 'tt:' + device['kind'] + '.' + name
                    paramlist = []
                    self.functions[function_type][function_name] = paramlist
                    self._token_canonicals[function_name] = function['canonical']
                    for argname, argtype, is_input, argcanonical in zip(function['args'],
                                                                        function['schema'],
                                                                        function['is_input'],
                                                                        function['argcanonicals']):
                        direction = 'in' if is_input else 'out'                    
                        paramlist.append((argname, argtype, direction))
                        self._param_tokens.add('tt-param:' + argname)
                        self._token_canonicals['tt-param:' + argname] = argcanonical
                        if function_type != 'actions':
                            self._trigger_or_query_params.add('tt-param:' + argname)
                    
                        if argtype.startswith('Array('):
                            elementtype = argtype[len('Array('):-1]
                        else:
                            elementtype = argtype
                        if elementtype.startswith('Enum('):
                            enums = elementtype[len('Enum('):-1].split(',')
                            if not elementtype in self._enum_types:
                                self._enum_types[elementtype] = enums
    
    def _process_entities(self, entities):
        for entity in entities:
            if entity['is_well_known'] == 1:
                    continue
            self.entities.add(entity['type'])
            for j in range(MAX_ARG_VALUES):
                token = 'GENERIC_ENTITY_' + entity['type'] + "_" + str(j)
                self._token_canonicals[token] = ' '.join(tokenize(entity['name'])).strip()
    
    def init_from_file(self, filename):
        self.reset()

        with open(filename, 'r') as fp:
            thingpedia = json.load(fp)
        
        self._process_devices(thingpedia['devices'])
        self._process_entities(thingpedia['entities'])

        self.complete()

    def init_from_url(self, snapshot=-1, thingpedia_url=None):
        if thingpedia_url is None:
            thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')
        ssl_context = ssl.create_default_context()

        with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1', context=ssl_context) as res:
            self._process_devices(json.load(res)['data'])

        with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
            self._process_entities(json.load(res)['data'])
    
    def complete(self):
        for function_type in ('triggers', 'queries', 'actions'):
            for function in self.functions[function_type]:
                self.tokens.append(function)
        self.num_functions = len(self.tokens) - self.num_control_tokens - self.num_begin_tokens

        self.tokens += self._param_tokens
        self.num_params = len(self._param_tokens)

        self.tokens += OPERATORS
        self.tokens += VALUES
        self.tokens += SPECIAL_TOKENS
        self.tokens += BOOKKEEPING_TOKENS     
        self.tokens += self.devices
        
        self._enum_tokens = OrderedSet()
        for enum_type in self._enum_types.values():
            for enum in enum_type:
                if enum in self._enum_tokens:
                    continue
                self._enum_tokens.add(enum)
                self.tokens.append(enum)
                self._token_canonicals[enum] = clean(enum)
        
        for unitlist in UNITS.values():
            self.tokens += unitlist
        
        for i in range(MAX_ARG_VALUES):
            for entity in ENTITIES:
                self.tokens.append(entity + "_" + str(i))
                
        for generic_entity in self.entities:
            for i in range(MAX_ARG_VALUES):
                self.tokens.append('GENERIC_ENTITY_' + generic_entity + "_" + str(i))
        
        print('num functions', self.num_functions)
        print('num triggers', len(self.functions['triggers']))
        print('num queries', len(self.functions['queries']))
        print('num actions', len(self.functions['actions']))
        print('num params', self.num_params)
        first_value_token = self.num_functions + self.num_begin_tokens + self.num_control_tokens
        print('num value tokens', len(self.tokens) - first_value_token)
        
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
        
        # build a DFA that will parse the thingtalk-ish code

        states = []
        transitions = []
        state_names = []
        
        def to_ids(tokens, words):
            return list([words[x] for x in tokens])

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
        
        # bookkeeping
        bookkeeping_id = new_state('bookkeeping')
        transitions.append((self.start_state, bookkeeping_id, 'bookkeeping'))
        transitions.append((bookkeeping_id, self.end_state, '<<EOS>>'))
        self.bookeeping_state_id = bookkeeping_id
        
        # special
        special_id = new_state('special')
        transitions.append((bookkeeping_id, special_id, 'special'))
        for t in SPECIAL_TOKENS:
            transitions.append((special_id, self.before_end_state, t))
            
        # command
        command_id = new_state('command')
        transitions.append((bookkeeping_id, command_id, 'command'))
        # help/configure/discover command
        help_id = new_state('device_or_generic')
        transitions.append((command_id, help_id, 'help'))
        transitions.append((help_id, self.before_end_state, 'generic'))
        for d in self.devices:
            transitions.append((help_id, self.before_end_state, d))
        
        # answers
        answer_id = new_state('answer')
        transitions.append((bookkeeping_id, answer_id, 'answer'))
        for v in VALUES:
            if v != '0' and v != '1':
                transitions.append((answer_id, self.before_end_state, v))
        for v in ENTITIES:
            if v != 'NUMBER':
                for i in range(MAX_ARG_VALUES):
                    transitions.append((answer_id, self.before_end_state, v + '_' + str(i)))
        before_unit = new_state('answer_before_unit')
        for i in range(MAX_ARG_VALUES):
            transitions.append((answer_id, before_unit, 'NUMBER_' + str(i)))
        transitions.append((answer_id, before_unit, '0'))
        transitions.append((answer_id, before_unit, '1'))
        transitions.append((before_unit, self.end_state, '<<EOS>>'))
        for base_unit in UNITS:
            for unit in UNITS[base_unit]:
                transitions.append((before_unit, self.before_end_state, unit))
        
        def do_invocation(invocation_name, params, for_prim):
            state_id = new_state(invocation_name)
            
            # allow one USERNAME_ parameter to follow the invocation immediately
            #for i in range(MAX_ARG_VALUES):
            #    transitions.append((state_id, state_id, 'USERNAME_' + str(i)))
            
            # go to each "in" parameter
            for param_name, param_type, param_direction in params:
                if param_direction == 'out':
                    continue
                if param_type in ('Any',) or param_type.startswith('Array('):
                    continue
                elementtype = param_type
                is_measure = False
                if elementtype in TYPE_RENAMES:
                    elementtype = TYPE_RENAMES[elementtype]
                if elementtype.startswith('Measure('):
                    is_measure = True
                    base_unit = elementtype[len('Measure('):-1]
                    values = UNITS[base_unit]
                elif elementtype.startswith('Enum('):
                    values = self._enum_types[elementtype]
                elif elementtype == 'Entity(tt:device)':
                    values = self.devices
                elif elementtype in TYPES:
                    operators, values = TYPES[elementtype]
                elif elementtype.startswith('Entity('):
                    values = ['GENERIC_ENTITY_' + elementtype[len('Entity('):-1], 'QUOTED_STRING']
                else:
                    _, values = TYPES[elementtype]
                if len(values) == 0 and for_prim == 'trigger':
                    continue
                
                before_value = new_state(invocation_name + '_tt-param:' + param_name)
                transitions.append((state_id, before_value, 'tt-param:' + param_name))

                if is_measure:
                    before_unit = new_state(invocation_name + '_tt-param:' + param_name + ':unit')
                    for i in range(MAX_ARG_VALUES):
                        transitions.append((before_value, before_unit, '0'))
                        transitions.append((before_value, before_unit, '1'))
                        transitions.append((before_value, before_unit, 'NUMBER_' + str(i)))
                    for unit in values:
                        transitions.append((before_unit, state_id, unit))
                else:
                    for v in values:
                        if v[0].isupper():
                            for i in range(MAX_ARG_VALUES):
                                transitions.append((before_value, state_id, v + '_' + str(i)))
                        else:
                            transitions.append((before_value, state_id, v))
                if is_measure and base_unit == 'ms':
                    for i in range(MAX_ARG_VALUES):
                        transitions.append((before_value, state_id, 'DURATION_' + str(i)))
                if for_prim != 'triggers':
                    for v in self._trigger_or_query_params:
                        transitions.append((before_value, state_id, v))
            
            #if for_prim == 'action':
            #    return (state_id, -1)
            predicate_state = new_state(invocation_name + '_predicate')
            before_and_or = new_state(invocation_name + '_and_or')
            transitions.append((before_and_or, predicate_state, 'and'))
            transitions.append((before_and_or, predicate_state, 'or'))
            
            any_predicate = False
            for param_name, param_type, _ in params:
                if param_type in ('Any',):
                    continue
                
                elementtype = param_type
                is_array = False
                is_measure = False
                if param_type.startswith('Array('):
                    is_array = True
                    elementtype = param_type[len('Array('):-1]
                if elementtype in TYPE_RENAMES:
                    elementtype = TYPE_RENAMES[elementtype]
                if elementtype.startswith('Measure('):
                    is_measure = True
                    operators = ['=', '<', '>', '>=', '<=']
                    base_unit = elementtype[len('Measure('):-1]
                    values = UNITS[base_unit]
                elif elementtype.startswith('Enum('):
                    operators = ['=']
                    values = self._enum_types[elementtype]
                elif elementtype == 'Entity(tt:device)':
                    operators = ['=']
                    values = self.devices
                elif elementtype in TYPES:
                    operators, values = TYPES[elementtype]
                elif elementtype.startswith('Entity('):
                    operators = ['=']
                    values = ['GENERIC_ENTITY_' + elementtype[len('Entity('):-1], 'QUOTED_STRING']
                else:
                    operators, values = TYPES[elementtype]
                if is_array:
                    operators = ['has']
                if len(values) == 0 and for_prim == 'trigger':
                    continue
                
                before_op = new_state(invocation_name + '_pred_tt-param:' + param_name)
                transitions.append((predicate_state, before_op, 'tt-param:' + param_name))
                before_value = new_state(invocation_name + '_pred_tt-param:' + param_name + ':value')
                any_predicate = True

                for op in operators:
                    transitions.append((before_op, before_value, op))
                if is_measure:
                    before_unit = new_state(invocation_name + '_pred_tt-param:' + param_name + ':unit')
                    for i in range(MAX_ARG_VALUES):
                        transitions.append((before_value, before_unit, '0'))
                        transitions.append((before_value, before_unit, '1'))
                        transitions.append((before_value, before_unit, 'NUMBER_' + str(i)))
                    for unit in values:
                        transitions.append((before_unit, before_and_or, unit))
                else:
                    for v in values:
                        if v[0].isupper():
                            for i in range(MAX_ARG_VALUES):
                                transitions.append((before_value, before_and_or, v + '_' + str(i)))
                        else:
                            transitions.append((before_value, before_and_or, v))
                if is_measure and base_unit == 'ms':
                    for i in range(MAX_ARG_VALUES):
                        transitions.append((before_value, before_and_or, 'DURATION_' + str(i)))
                if for_prim != 'trigger':
                    for v in self._trigger_or_query_params:
                        transitions.append((before_value, before_and_or, v))
                    
            if any_predicate:
                transitions.append((state_id, predicate_state, 'if'))
                return (state_id, before_and_or)
            else:
                return (state_id, -1)
        
        # rules
        rule_id = new_state('rule')
        transitions.append((self.start_state, rule_id, 'rule'))
        policy_id = new_state('policy')
        transitions.append((self.start_state, policy_id, 'policy'))
        for i in range(MAX_ARG_VALUES):
            transitions.append((policy_id, rule_id, 'USERNAME_' + str(i)))
        setup_id = new_state('setup')
        transitions.append((self.start_state, setup_id, 'setup'))
        for i in range(MAX_ARG_VALUES):
            transitions.append((setup_id, rule_id, 'USERNAME_' + str(i)))
        
        trigger_ids = []
        query_ids = []
        
        for trigger_name, params in self.functions['triggers'].items():
            begin_state, end_state = do_invocation(trigger_name, params, 'triggers')
            transitions.append((rule_id, begin_state, trigger_name))
            transitions.append((policy_id, begin_state, trigger_name))
            trigger_ids.append(begin_state)
            if end_state >= 0:
                trigger_ids.append(end_state)
        for query_name, params in self.functions['queries'].items():
            begin_state, end_state = do_invocation(query_name, params, 'queries')
            for trigger_id in trigger_ids:
                transitions.append((trigger_id, begin_state, query_name))
            query_ids.append(begin_state)
            if end_state >= 0:
                query_ids.append(end_state)
        for action_name, params in self.functions['actions'].items():
            begin_state, end_state = do_invocation(action_name, params, 'actions')
            for query_id in query_ids:
                transitions.append((query_id, begin_state, action_name))
            transitions.append((begin_state, self.end_state, '<<EOS>>'))
            if end_state >= 0:
                transitions.append((end_state, self.end_state, '<<EOS>>'))
            
        # do a second copy of the transition matrix for split sequences
        self.function_states = np.zeros((self.num_functions,), dtype=np.int32)
        self.function_states.fill(-1)
        for part in ('triggers', 'queries', 'actions'):
            for function_name, params in self.functions[part].items():
                token = self.dictionary[function_name] - self.num_control_tokens - self.num_begin_tokens
                begin_state, end_state = do_invocation(function_name, params, part)
                transitions.append((begin_state, self.end_state, '<<EOS>>'))
                if end_state >= 0:
                    transitions.append((end_state, self.end_state, '<<EOS>>'))
                self.function_states[token] = begin_state
                

        # now build the actual DFA
        num_states = len(states)
        self.num_states = num_states
        print("num states", num_states)
        print("num tokens", self.output_size)
        self.transition_matrix = np.zeros((num_states, self.output_size), dtype=np.int32)
        self.transition_matrix.fill(-1)
        self.allowed_token_matrix = np.zeros((num_states, self.output_size), dtype=np.bool8)

        for from_state, to_state, token in transitions:
            token_id = self.dictionary[token]
            if self.transition_matrix[from_state, token_id] != -1 and \
                self.transition_matrix[from_state, token_id] != to_state:
                raise ValueError("Ambiguous transition around token " + token + " in state " + state_names[from_state])
            self.transition_matrix[from_state, token_id] = to_state
            self.allowed_token_matrix[from_state, token_id] = True

        if True:
            visited = set()
            def dfs(state):
                visited.add(state)
                any_out = False
                for next_state in self.transition_matrix[state]:
                    if next_state == -1:
                        continue
                    any_out = True
                    if next_state in visited:
                        continue
                    dfs(next_state)
                if not any_out:
                    raise ValueError('Reachable state %d (%s) has no outgoing states' % (state, state_names[state]))
            dfs(self.start_state)

        self.state_names = state_names

    def get_embeddings(self, input_words, input_embeddings):
        '''
        Create a feature matrix for each token in the TT program.
        This feature matrix is dot-producted with the output from the decoder
        LSTM (after a linear layer); whatever has the highest score is then
        selected as output.
        '''
        
        # Token class:
        # - 0: control
        # - 1: begin token
        # - 2; bookkeeping token
        # - 3: operator
        # - 4: unit
        # - 5: value
        # - 6: entity
        # - 7: enum value
        # - 8: device name
        # - 9: function name
        # - 10: parameter name
        token_classes = 10
        
        num_units = 0
        for unitlist in UNITS.values():
            num_units += len(unitlist)

        input_embed_size = input_embeddings.shape[-1]
        
        function_types = 3 # trigger, query or action
        depth = token_classes + self.num_control_tokens + len(BEGIN_TOKENS) + len(BOOKKEEPING_TOKENS) \
            + len(SPECIAL_TOKENS) + len(OPERATORS) + num_units + len(VALUES) \
            + len(ENTITIES) + MAX_ARG_VALUES + function_types + input_embed_size
        
        embedding = np.zeros((len(self.tokens), depth), dtype=np.float32)
        
        def embed_token(token):
            token_embedding = np.zeros((input_embed_size,), dtype=np.float32)
            canonical = self._token_canonicals[token]
            if not canonical:
                print("WARNING: token %s has no canonical" % (token,))
                return token_embedding
            for canonical_token in canonical.split(' '):
                if canonical_token in input_words:
                    token_embedding += input_embeddings[input_words[canonical_token]]
                else:
                    print("WARNING: missing word %s in canonical for output token %s" % (canonical_token, token))
                    token_embedding += input_embeddings[input_words['<<UNK>>']]
                return token_embedding
        
        off = token_classes
        for i in range(self.num_control_tokens):
            token_cls = 0
            embedding[i, token_cls] = 1
            embedding[i, off + i] = 1
        off += self.num_control_tokens
        for i, token in enumerate(BEGIN_TOKENS):
            token_cls = 1
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, off + i] = 1
        off += self.num_begin_tokens
        for i, token in enumerate(itertools.chain(BOOKKEEPING_TOKENS, SPECIAL_TOKENS)):
            token_cls = 2
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, off + i] = 1
        off += len(BOOKKEEPING_TOKENS) + len(SPECIAL_TOKENS)
        for i, token in enumerate(OPERATORS):
            token_cls = 3
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, off + i] = 1
        off += len(OPERATORS)
        for unitlist in UNITS.values():
            for i, token in enumerate(unitlist):
                token_cls = 4
                token_id = self.dictionary[token]
                embedding[token_id, token_cls] = 1
                embedding[token_id, off + i] = 1
            off += len(unitlist)
        for i, token in enumerate(VALUES):
            token_cls = 5
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, off + i] = 1
        off += len(VALUES)
        for i, token_prefix in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                token = token_prefix + '_' + str(j)
                token_cls = 5
                token_id = self.dictionary[token]
                embedding[token_id, token_cls] = 1
                embedding[token_id, off + i] = 1
                embedding[token_id, off + len(ENTITIES) + j] = 1
        off += len(ENTITIES)

        for token_infix in self.entities:
            for j in range(MAX_ARG_VALUES):
                token = 'GENERIC_ENTITY_' + token_infix + '_' + str(j)
                token_cls = 6
                token_id = self.dictionary[token]
                embedding[token_id, token_cls] = 1
                embedding[token_id, off + j] = 1
                embedding[token_id, -input_embed_size:] = embed_token(token)
        for token in self._enum_tokens:
            token_cls = 7
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, -input_embed_size:] = embed_token(token)
        for token in self.devices:
            token_cls = 8
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, -input_embed_size:] = embed_token(token)
        for function_type_feature, function_type in enumerate(('triggers', 'queries', 'actions')):
            for token in self.functions[function_type]:
                token_cls = 8
                token_id = self.dictionary[token]
                embedding[token_id, token_cls] = 1
                embedding[token_id, off + MAX_ARG_VALUES + function_type_feature] = 1
                embedding[token_id, -input_embed_size:] = embed_token(token)
        for token in self._param_tokens:
            token_cls = 9
            token_id = self.dictionary[token]
            embedding[token_id, token_cls] = 1
            embedding[token_id, -input_embed_size:] = embed_token(token)
        
        for i in range(len(embedding)):
            assert np.any(embedding[i] > 0)
        return embedding

    def dump_tokens(self):
        for token in self.tokens:
            print(token)

    def split_batch_in_parts(self, labels_batch):
        batch_size = len(labels_batch)
        top_batch = np.empty((batch_size,), dtype=np.int32)
        special_label_batch = np.zeros((batch_size, MAX_SPECIAL_LENGTH), dtype=np.int32)
        part_function_batches = dict()
        part_sequence_batches = dict()
        part_sequence_length_batches = dict()
        for part in ('trigger', 'query', 'action'):
            part_function_batches[part] = np.zeros((batch_size,), dtype=np.int32)
            part_sequence_batches[part] = np.zeros((batch_size, MAX_PRIMITIVE_LENGTH), dtype=np.int32)
            part_sequence_length_batches[part] = np.zeros((batch_size,), dtype=np.int32)
            
        rule_token = self.dictionary['rule']
        first_value_token = self.num_functions + self.num_begin_tokens + self.num_control_tokens
        for i, label in enumerate(labels_batch):
            top_batch[i] = label[0]
            if top_batch[i] != rule_token:
                special_label_batch[i] = label[1:1+MAX_SPECIAL_LENGTH]
                for part in ('trigger', 'query', 'action'):
                    if part == 'trigger':
                        function_offset = self.num_begin_tokens + self.num_control_tokens
                    elif part == 'query':
                        function_offset = self.num_begin_tokens + self.num_control_tokens + len(self.functions['trigger'])
                    else:
                        function_offset = self.num_begin_tokens + self.num_control_tokens + len(self.functions['trigger']) + len(self.functions['query'])
                    # add dummy values to the sequences to preserve the ability to compute
                    # losses and grammar constraints
                    part_function_batches[part][i] = function_offset
                    part_sequence_batches[part][i,0] = self.end
            else:
                special_label_batch[i,0] = self.end
                j = 1
                for part in ('trigger', 'query', 'action'):
                    if part == 'trigger':
                        function_offset = self.num_begin_tokens + self.num_control_tokens
                    elif part == 'query':
                        function_offset = self.num_begin_tokens + self.num_control_tokens + len(self.functions['trigger'])
                    else:
                        function_offset = self.num_begin_tokens + self.num_control_tokens + len(self.functions['trigger']) + len(self.functions['query'])
                    function_max = len(self.functions[part])
                    assert function_offset <= label[j] < function_offset+function_max, (function_offset, function_max, label[j], self.tokens[label[j]])
                    
                    part_function_batches[part][i] = label[j]
                    j += 1
                    start = j
                    while label[j] >= first_value_token:
                        j+= 1
                    end = j
                    assert end-start+1 < MAX_PRIMITIVE_LENGTH
                    part_sequence_batches[part][i,0:end-start] = label[start:end]
                    part_sequence_batches[part][i,end-start] = self.end
                    part_sequence_length_batches[part][i] = end-start+1
        return top_batch, special_label_batch, part_function_batches, part_sequence_batches, part_sequence_length_batches

    def vectorize_program(self, program, max_length=60):
        vector, length = super().vectorize_program(program, max_length)
        self.normalize_sequence(vector)
        return vector, length

    def parse(self, program):
        curr_state = self.start_state
        for token_id in program:
            next = self.transition_matrix[curr_state, token_id]
            if next == -1:
                raise ValueError("Unexpected token " + self.tokens[token_id] + " in " + (' '.join(self.tokens[x] for x in program)) + " (in state " + self.state_names[curr_state] + ")")
            #print("transition", self.state_names[curr_state], "->", self.state_names[next])
            curr_state = next

        if curr_state != self.end_state:
            raise ValueError("Premature end of program in " + (' '.join(self.tokens[x] for x in program)) + " (in state " + self.state_names[curr_state] + ")")
        #print(*(self.tokens[x] for x in program))

    def parse_all(self, fp):
        vectors = []
        for line in fp.readlines():
            try:
                program = line.strip().split()
                vector = self.vectorize_program(program)[0]
                self.parse(vector)
                vectors.append(vector)
            except ValueError as e:
                print(e)
        return np.array(vectors, dtype=np.int32)

    def get_function_init_state(self, function_tokens):
        next_state = tf.gather(self.function_states, function_tokens - (self.num_begin_tokens + self.num_control_tokens))
        assert2 = tf.Assert(tf.reduce_all(next_state >= 0), [function_tokens])
        with tf.control_dependencies([assert2]):
            return tf.identity(next_state)

    def get_init_state(self, batch_size):
        return tf.ones((batch_size,), dtype=tf.int32) * self.start_state

    def constrain_value_logits(self, logits, curr_state):
        first_value_token = self.num_functions + self.num_begin_tokens + self.num_control_tokens
        num_value_tokens = self.output_size - first_value_token
        value_allowed_token_matrix = np.concatenate((self.allowed_token_matrix[:,:self.num_control_tokens], self.allowed_token_matrix[:,first_value_token:]), axis=1)
        
        with tf.name_scope('constrain_logits'):
            allowed_tokens = tf.gather(tf.constant(value_allowed_token_matrix), curr_state)
            assert allowed_tokens.get_shape()[1:] == (self.num_control_tokens + num_value_tokens,)

            constrained_logits = logits - tf.to_float(tf.logical_not(allowed_tokens)) * 1e+10
        return constrained_logits

    def constrain_logits(self, logits, curr_state):
        with tf.name_scope('constrain_logits'):
            allowed_tokens = tf.gather(tf.constant(self.allowed_token_matrix), curr_state)
            assert allowed_tokens.get_shape()[1:] == (self.output_size,)

            constrained_logits = tf.where(allowed_tokens, logits, tf.fill(tf.shape(allowed_tokens), -1e+10))
        return constrained_logits

    def value_transition(self, curr_state, next_symbols, batch_size):
        first_value_token = self.num_functions + self.num_begin_tokens + self.num_control_tokens
        num_value_tokens = self.output_size - first_value_token
        with tf.name_scope('grammar_transition'):
            adjusted_next_symbols = tf.where(next_symbols >= self.num_control_tokens, next_symbols + (first_value_token - self.num_control_tokens), next_symbols)
            
            assert1 = tf.Assert(tf.reduce_all(tf.logical_and(next_symbols < num_value_tokens, next_symbols >= 0)), [curr_state, next_symbols])
            with tf.control_dependencies([assert1]):
                transitions = tf.gather(tf.constant(self.transition_matrix), curr_state)
            assert transitions.get_shape()[1:] == (self.output_size,)
            
            indices = tf.stack((tf.range(0, batch_size), adjusted_next_symbols), axis=1)
            next_state = tf.gather_nd(transitions, indices)
            
            assert2 = tf.Assert(tf.reduce_all(next_state >= 0), [curr_state, adjusted_next_symbols, next_state])
            with tf.control_dependencies([assert2]):
                return tf.identity(next_state)

    def transition(self, curr_state, next_symbols, batch_size):
        with tf.name_scope('grammar_transition'):
            transitions = tf.gather(tf.constant(self.transition_matrix), curr_state)
            assert transitions.get_shape()[1:] == (self.output_size,)

            indices = tf.stack((tf.range(0, batch_size), next_symbols), axis=1)
            next_state = tf.gather_nd(transitions, indices)
            return next_state
    
    def _normalize_invocation(self, seq, start):
        assert self.tokens[seq[start]].startswith('tt:')
        if self.tokens[seq[start]].startswith('USERNAME_'):
            start += 1
        end = start
        
        params = []
        while end < len(seq) and seq[end] != self.end and self.tokens[seq[end]].startswith('tt-param:'):
            param_id = seq[end]
            end += 1
            if end >= len(seq) or seq[end] == self.end:
                # truncated output
                return end
            param_value = [seq[end]]
            end += 1
            while end < len(seq) and seq[end] != self.end and not self.tokens[seq[end]].startswith('tt:') and self.tokens[seq[end]] != 'if':
                param_value.append(seq[end])
                end += 1
            params.append((param_id, param_value))
        params.sort(key=lambda x: x[0])
        assert end <= len(seq)

        i = start
        for param_id, operator, param_value in params:
            seq[i] = param_id
            seq[i+1] = operator
            seq[i+2:i+2+len(param_value)] = param_value
            i += 2 + len(param_value)
            assert i <= end
        
        return end
    
    def normalize_sequence(self, seq):
        i = 0
        if seq[0] == self.dictionary['rule']:
            i += 1
            i = self._normalize_invocation(seq, i)
            if i < len(seq) and seq[i] != self.end:
                i = self._normalize_invocation(seq, i)
            if i < len(seq) and seq[i] != self.end:
                i = self._normalize_invocation(seq, i)
    
    def compare(self, gold, decoded):
        decoded = list(decoded)
        #self._normalize_sequence(decoded)
        return gold == decoded
        

if __name__ == '__main__':
    grammar = ThingtalkGrammar(sys.argv[1])
    #grammar.dump_tokens()
    #grammar.normalize_all(sys.stdin)
    matrix = grammar.parse_all(sys.stdin)
    print('Parsed', matrix.shape)
    np.save('programs.npy', matrix)
    #for i, name in enumerate(grammar.state_names):
    #    print i, name
