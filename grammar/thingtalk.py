# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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

from .shift_reduce_grammar import ShiftReduceGrammar

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
    'String': (['=', 'contains', 'starts_with', 'ends_with'], ['QUOTED_STRING']),
    'Date': (['=', '>', '<'], ['DATE']),
    'Time': (['='], ['TIME']),
    'Number': (['=', '<', '>', '>=', '<='], ['NUMBER', '1', '0']),
    'Entity(tt:username)': (['='], ['USERNAME']),
    'Entity(tt:hashtag)': (['='], ['HASHTAG']),
    'Entity(tt:phone_number)': (['='], ['PHONE_NUMBER']),
    'Entity(tt:email_address)': (['='], ['EMAIL_ADDRESS']),
    'Entity(tt:url)': (['='], ['URL']),
    'Entity(tt:picture)': (['='], [])
}
TYPE_RENAMES = {
    'Username': 'Entity(tt:username)',
    'Hashtag': 'Entity(tt:hashtag)',
    'PhoneNumber': 'Entity(tt:phone_number)',
    'EmailAddress': 'Entity(tt:email_address)',
    'URL': 'Entity(tt:url)',
    'Picture': 'Entity(tt:picture)',
    'Bool': 'Boolean'
}

UNITS = dict(C=["C", "F"],
             ms=["ms", "s", "min", "h", "day", "week", "month", "year"],
             m=["m", "km", "mm", "cm", "mi", "in", "ft"],
             mps=["mps", "kmph", "mph"],
             kg=["kg", "g", "lb", "oz"],
             kcal=["kcal", "kJ"],
             bpm=["bpm"],
             byte=["byte", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB"])

MAX_ARG_VALUES = 5

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

class ThingtalkGrammar(ShiftReduceGrammar):
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
        # first the control tokens (padding, go, eos)
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
        print('num tokens', len(self.tokens))
        
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
        
        GRAMMAR = OrderedDict({
            '$input': [('rule', '$program'),
                       ('setup', '$constant_Entity(tt:username)', '$program'),
                       ('policy', '$policy'),
                       ('bookkeeping', '$bookkeeping')],
            '$policy': [('$constant_Entity(tt:username)', '$policy_program'),
                        ('$policy_program',)],
            '$bookkeeping': [('special', '$special'),
                             ('command', '$command')],
            '$special': [(x,) for x in SPECIAL_TOKENS],
            '$command': [('help', 'generic'),
                         ('help', '$constant_Entity(tt:device)')],
            '$program': [('$trigger', '$query', '$action')],
            '$policy_program': [('$trigger', '$query', '$policy_action')],
            '$trigger': [('$trigger_function',),
                         ('$trigger_function', 'if', '$filter'),
                         ('tt:$builtin.now',)],
            '$trigger_function': [('$triggers_function_name',),
                                  ('$triggers_function_name', '$constant_Entity(tt:username)'),
                                  ('$trigger_function', '$in_param')],
            '$triggers_function_name': [],
            '$query': [('$query_function',),
                       ('$query_function', 'if', '$filter'),
                       ('tt:$builtin.noop',)],
            '$query_function': [('$queries_function_name',),
                                ('$queries_function_name', '$constant_Entity(tt:username)'),
                                ('$query_function', '$in_param')],
            '$queries_function_name': [],
            '$action': [('$action_function',),
                        ('tt:$builtin.notify',),
                        ('tt:$builtin.return',)],
            '$policy_action': [('$action_function',),
                               ('$action_function', 'if', '$filter'),
                               ('tt:$builtin.notify',)],
            '$action_function': [('$actions_function_name',),
                                 ('$actions_function_name', '$constant_Entity(tt:username)'),
                                 ('$action_function', '$in_param')],
            '$actions_function_name': [],
            '$filter': [('$atom_filter', 'and', '$filter'),
                        ('$atom_filter', 'or', '$filter'),
                        ('$atom_filter',)],
            '$value_filter': OrderedSet(), 
            '$atom_filter': OrderedSet(),
            '$in_param': OrderedSet(),
            '$out_param': OrderedSet([('tt-param:$event',)])
        })
        
        def add_type(type, value_rules, operators):
            operator_rules = []
            assert all(isinstance(x, tuple) for x in value_rules)
            GRAMMAR['$constant_' + type] = value_rules
            GRAMMAR['$bookkeeping'].append(('answer', '$constant_' + type))
            for op in operators:
                GRAMMAR['$value_filter'].add((op, '$constant_' + type))
                GRAMMAR['$value_filter'].add((op, '$out_param'))
            GRAMMAR['$value_filter'].add(('has', '$constant_' + type))
        
        # base types
        for type, (operators, values) in TYPES.items():
            value_rules = []
            for v in values:
                if v[0].isupper():
                    for i in range(MAX_ARG_VALUES):
                        value_rules.append((v + '_' + str(i), ))
                else:
                    value_rules.append((v,))
            add_type(type, value_rules, operators)
        for base_unit, units in UNITS.items():
            value_rules = [('$constant_Number', unit) for unit in units]
            operators, _ = TYPES['Number']
            add_type('Measure(' + base_unit + ')', value_rules, operators)
        for i in range(MAX_ARG_VALUES):
            GRAMMAR['$constant_Measure(ms)'].append(('DURATION_' + str(i),))

        # well known entities
        add_type('Entity(tt:device)', [(device,) for device in self.devices], ['='])
            
        # other entities
        for generic_entity in self.entities:
            value_rules = [('GENERIC_ENTITY_' + generic_entity + "_" + str(i), ) for i in range(MAX_ARG_VALUES)]
            add_type('Entity(' + generic_entity + ')', value_rules, ['='])
            
        # maps a parameter to the list of types it can possibly have
        # over the whole Thingpedia
        param_types = OrderedDict()
        
        for function_type in ('triggers', 'queries', 'actions'):
            for function_name, params in self.functions[function_type].items():
                if function_name.startswith('tt:$'):
                    continue
                for param_name, param_type, param_direction in params:
                    if param_type in TYPE_RENAMES:
                        param_type = TYPE_RENAMES[param_type]
                    if param_type.startswith('Array('):
                        element_type = param_type[len('Array('):-1]
                        if element_type in TYPE_RENAMES:
                            param_type = 'Array(' + TYPE_RENAMES[element_type] + ')'
                    if param_name not in param_types:
                        param_types[param_name] = OrderedSet()
                    param_types[param_name].add((param_type, param_direction))
                GRAMMAR['$' + function_type + '_function_name'].append((function_name,))

        for param_name, options in param_types.items():
            for (param_type, param_direction) in options:
                if param_type == 'Any':
                    continue
                if param_direction == 'out':
                    GRAMMAR['$out_param'].add(('tt-param:' + param_name,))
                GRAMMAR['$atom_filter'].add(('tt-param:' + param_name, '$value_filter'))
                
                if param_type.startswith('Enum('):
                    enum_type = self._enum_types[param_type]
                    for enum in enum_type:
                        GRAMMAR['$atom_filter'].add(('tt-param:' + param_name, '=', enum))
                        if param_direction == 'in':
                            GRAMMAR['$in_param'].add(('tt-param:' + param_name, enum))
                else:
                    if param_direction == 'in':
                        GRAMMAR['$in_param'].add(('tt-param:' + param_name, '$out_param'))
                        GRAMMAR['$in_param'].add(('tt-param:' + param_name, '$constant_' + param_type))
                        if param_type.startswith('Entity('):
                            GRAMMAR['$in_param'].add(('tt-param:' + param_name, '$constant_String'))
                        if param_type in ('Entity(tt:phone_number)', 'Entity(tt:email_address)'):
                            GRAMMAR['$in_param'].add(('tt-param:' + param_name, '$constant_Entity(tt:username)'))

        self.construct_parser(GRAMMAR)

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
    grammar.parse_all(sys.stdin)
    #for i, name in enumerate(grammar.state_names):
    #    print i, name
