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
'''
Created on Dec 8, 2017

@author: gcampagn
'''

import json
import os
import urllib.request
import ssl
import re
import sys
import itertools
import numpy as np
from .shift_reduce_grammar import ShiftReduceGrammar

from collections import OrderedDict
from orderedset import OrderedSet

SPECIAL_TOKENS = ['special:yes', 'special:no', 'special:nevermind',
                  'special:makerule', 'special:failed']
TYPES = {
    'Location': (['=='], ['LOCATION', 'location:current_location', 'location:work', 'location:home']),
    'Boolean':  ([], []), # booleans are handled per-parameter, like enums
    'String': (['==', '=~', '~=', 'starts_with', 'ends_with', 'prefix_of', 'suffix_of'], ['""', 'QUOTED_STRING']),
    'Date': (['==', '>', '<', '>=', '<='], [
        'DATE',
        'now',
        ('start_of', 'unit:h'),
        ('start_of', 'unit:day'),
        ('start_of', 'unit:week'),
        ('start_of', 'unit:mon'),
        ('start_of', 'unit:year'),
        ('end_of', 'unit:h'),
        ('end_of', 'unit:day'),
        ('end_of', 'unit:week'),
        ('end_of', 'unit:mon'),
        ('end_of', 'unit:year'),
        ('$constant_Date', '+', '$constant_Measure(ms)'),
        ('$constant_Date', '-', '$constant_Measure(ms)')
        ]),
    'Time': (['=='], ['TIME']),
    'Number': (['==', '<', '>', '>=', '<='], ['NUMBER', '1', '0']),
    'Entity(tt:username)': (['=='], ['USERNAME']),
    'Entity(tt:hashtag)': (['=='], ['HASHTAG']),
    'Entity(tt:phone_number)': (['=='], ['PHONE_NUMBER']),
    'Entity(tt:email_address)': (['=='], ['EMAIL_ADDRESS']),
    'Entity(tt:url)': (['=='], ['URL']),
    'Entity(tt:picture)': (['=='], [])
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
             ms=["ms", "s", "min", "h", "day", "week", "mon", "year"],
             m=["m", "km", "mm", "cm", "mi", "in", "ft"],
             mps=["mps", "kmph", "mph"],
             kg=["kg", "g", "lb", "oz"],
             kcal=["kcal", "kJ"],
             bpm=["bpm"],
             byte=["byte", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB"])

MAX_ARG_VALUES = 4
MAX_STRING_ARG_VALUES = 5

def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()

def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', name.lower())

class NewThingTalkGrammar(ShiftReduceGrammar):
    '''
    The grammar of New thingtalk
    '''
    
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.init_from_file(filename)
        
    def reset(self):
        queries = OrderedDict()
        actions = OrderedDict()
        functions = dict(queries=queries, actions=actions)
        self.functions = functions
        self.entities = []
        self._enum_types = OrderedDict()
        self.devices = []
    
    def _process_devices(self, devices):
        for device in devices:
            if device['kind_type'] == 'global':
                continue
            self.devices.append('device:' + device['kind'])
            
            for function_type in ('queries', 'actions'):
                for name, function in device[function_type].items():
                    function_name = '@' + device['kind'] + '.' + name
                    paramlist = []
                    self.functions[function_type][function_name] = paramlist
                    for argname, argtype, is_input in zip(function['args'],
                                                          function['schema'],
                                                          function['is_input']):
                        direction = 'in' if is_input else 'out'                    
                        paramlist.append((argname, argtype, direction))
                    
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
            self.entities.append((entity['type'], entity['has_ner_support']))
    
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
        self.num_functions = len(self.functions['queries']) + len(self.functions['actions'])
        
        GRAMMAR = OrderedDict({
            #'$input': [('$program',),
            #           ('bookkeeping', '$bookkeeping')],
            #'$bookkeeping': [('special', '$special'),
            #                 ('command', '$command'),
            #                 ('answer', '$constant_Any')],
            #'$special': [(x,) for x in SPECIAL_TOKENS],
            #'$command': [('help', 'generic'),
            #             ('help', '$constant_Entity(tt:device)')],
            #'$program': [('$rule',),
            #             ('$constant_Entity(tt:username)', ':', '$rule')],
            '$input': [('$stream', '=>', '$thingpedia_actions'),
                       ('$stream', '=>', 'notify'),
                       ('now', '=>', '$table', '=>', '$thingpedia_actions'),
                       ('now', '=>', '$table', '=>', 'notify'),
                       ('now', '=>', '$thingpedia_actions'),
                       ('$input', 'on', '$param_passing')],
            '$table': [('$thingpedia_queries',),
                       ('(', '$table', ')', 'filter', '$filter'),
                       #('aggregate', 'min', '$out_param', 'of', '(', '$table', ')'),
                       #('aggregate', 'max', '$out_param', 'of', '(', '$table', ')'),
                       #('aggregate', 'sum', '$out_param', 'of', '(', '$table', ')'),
                       #('aggregate', 'avg', '$out_param', 'of', '(', '$table', ')'),
                       #('aggregate', 'count', 'of', '(', '$table', ')'),
                       #('aggregate', 'argmin', '$out_param', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       #('aggregate', 'argmax', '$out_param', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('$table_join',),
                       #('window', '$constant_Number', ',', '$constant_Number', 'of', '(', '$stream', ')'),
                       #('timeseries', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$stream', ')'),
                       #('sequence', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       #('history', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$table', ')')
                       ],
            '$table_join': [('(', '$table', ')', 'join', '(', '$table', ')'),
                            ('$table_join', 'on', '$param_passing')],
            '$stream': [('timer', 'base', '=', '$constant_Date', ',', 'interval', '=', '$constant_Measure(ms)'),
                        ('attimer', 'time', '=', '$constant_Time',),
                        ('monitor', '(', '$table', ')'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '$out_param'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '[', '$out_param_list', ']'),
                        ('edge', '(', '$stream', ')', 'on', '$filter'),
                        #('edge', '(', '$stream', ')', 'on', 'true'),
                        ('$stream_join',)],
            '$stream_join': [('(', '$stream', ')', 'join', '(', '$table', ')'),
                             ('$stream_join', 'on', '$param_passing')],
            '$thingpedia_queries': [('$thingpedia_queries', '$const_param')],
            '$thingpedia_actions': [('$thingpedia_actions', '$const_param')],
            '$param_passing': OrderedSet(),
            '$const_param': OrderedSet(),
            '$out_param': OrderedSet(),
            '$out_param_list': [('$out_param',),
                                ('$out_param_list', ',', '$out_param')],

            '$filter': [('$or_filter',),
                        ('$filter', 'and', '$or_filter',)],
            '$or_filter': [('$atom_filter',),
                           ('not', '$atom_filter',),
                           ('$or_filter', 'or', '$atom_filter')],
            '$atom_filter': OrderedSet([('$out_param', 'in_array', '$constant_Array')]),
            
            '$constant_Array': [('[', '$constant_array_values', ']',)],
            '$constant_array_values': [('$constant_Any',),
                                       ('$constant_array_values', ',', '$constant_Any')],
            '$constant_Any': OrderedSet(),
        })
        
        def add_type(type, value_rules, operators):
            assert all(isinstance(x, tuple) for x in value_rules)
            GRAMMAR['$constant_' + type] = value_rules
            GRAMMAR['$constant_Any'].add(('$constant_' + type,))
            for op in operators:
                GRAMMAR['$atom_filter'].add(('$out_param', op, '$constant_' + type))
                # FIXME reenable some day
                #GRAMMAR['$atom_filter'].add(('$out_param', op, '$out_param'))
            GRAMMAR['$atom_filter'].add(('$out_param', 'contains', '$constant_' + type))

        # base types
        for type, (operators, values) in TYPES.items():
            value_rules = []
            for v in values:
                if isinstance(v, tuple):
                    value_rules.append(v) 
                elif v == 'QUOTED_STRING':
                    for i in range(MAX_STRING_ARG_VALUES):
                        value_rules.append((v + '_' + str(i), ))
                elif v[0].isupper():
                    for i in range(MAX_ARG_VALUES):
                        value_rules.append((v + '_' + str(i), ))
                else:
                    value_rules.append((v,))
            add_type(type, value_rules, operators)
        for base_unit, units in UNITS.items():
            value_rules = [('$constant_Number', 'unit:' + unit) for unit in units]
            # FIXME reenable someday when we want to handle 6ft 3in
            #value_rules += [('$constant_Measure(' + base_unit + ')', '$constant_Number', 'unit:' + unit) for unit in units]
            operators, _ = TYPES['Number']
            add_type('Measure(' + base_unit + ')', value_rules, operators)
        for i in range(MAX_ARG_VALUES):
            GRAMMAR['$constant_Measure(ms)'].append(('DURATION_' + str(i),))

        # well known entities
        #add_type('Entity(tt:device)', [(device,) for device in self.devices], ['='])
        add_type('Entity(tt:device)', [], ['='])

        # other entities
        for generic_entity, has_ner in self.entities:
            if has_ner:
                value_rules = [('GENERIC_ENTITY_' + generic_entity + "_" + str(i), ) for i in range(MAX_ARG_VALUES)]
            else:
                value_rules = []
            add_type('Entity(' + generic_entity + ')', value_rules, ['=='])
            
        # maps a parameter to the list of types it can possibly have
        # over the whole Thingpedia
        param_types = OrderedDict()
        
        for function_type in ('queries', 'actions'):
            for function_name, params in self.functions[function_type].items():
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
                GRAMMAR['$thingpedia_' + function_type].append((function_name,))

        for param_name, options in param_types.items():
            for (param_type, param_direction) in options:
                if param_type.startswith('Enum('):
                    enum_type = self._enum_types[param_type]
                    for enum in enum_type:
                        #GRAMMAR['$atom_filter'].add(('$out_param', '==', 'enum:' + enum))
                        if param_direction == 'in':
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', 'enum:' + enum))
                        else:
                            # NOTE: enum filters don't follow the usual convention for filters
                            # this is because, linguistically, it does not make much sense to go
                            # through $out_param: enum parameters are often implicit
                            # one does not say "if the mode of my hvac is off", one says "if my hvac is off"
                            # (same, and worse, with booleans)
                            GRAMMAR['$atom_filter'].add(('param:' + param_name, '==', 'enum:' + enum))
                else:
                    if param_direction == 'out':
                        if param_type != 'Boolean':
                            GRAMMAR['$out_param'].add(('param:' + param_name,))
                        else:
                            GRAMMAR['$atom_filter'].add(('param:' + param_name, '==', 'true'))
                            GRAMMAR['$atom_filter'].add(('param:' + param_name, '==', 'false'))
                    if param_direction == 'in':
                        GRAMMAR['$param_passing'].add(('param:' + param_name, '=', '$out_param'))
                        if param_type == 'String':
                            GRAMMAR['$param_passing'].add(('param:' + param_name, '=', 'event'))
                    if param_direction == 'in' and param_type != 'Any':
                        if param_type != 'Boolean':
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', '$constant_' + param_type))
                        else:
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', 'true'))
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', 'false'))
                        if param_type.startswith('Entity('):
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', '$constant_String'))
                        if param_type in ('Entity(tt:phone_number)', 'Entity(tt:email_address)'):
                            GRAMMAR['$const_param'].add(('param:' + param_name, '=', '$constant_Entity(tt:username)'))

        self.tokens += self.construct_parser(GRAMMAR)
        print('num functions', self.num_functions)
        print('num queries', len(self.functions['queries']))
        print('num actions', len(self.functions['actions']))
        print('num other', len(self.tokens) - self.num_functions - self.num_control_tokens)
        
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i

if __name__ == '__main__':
    grammar = NewThingTalkGrammar(sys.argv[1])
    #grammar.dump_tokens()
    #grammar.normalize_all(sys.stdin)
    vectors = []
    for line in sys.stdin:
        try:
            vectors.append(grammar.vectorize_program(line.strip())[0])
        except:
            print(line.strip())
            raise
    np.save('programs.npy', np.array(vectors), allow_pickle=False)
    #for i, name in enumerate(grammar.state_names):
    #    print i, name
