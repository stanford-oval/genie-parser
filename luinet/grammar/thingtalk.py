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

import json
import os
import urllib.request
import ssl
import re
import sys

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from orderedset import OrderedSet

from .shift_reduce_grammar import ShiftReduceGrammar
from ..util.loader import load_dictionary
from ..util.metrics import make_pyfunc_metric_fn, accuracy, grammar_accuracy, \
    adjust_predictions_labels


SPECIAL_TOKENS = ['special:yes', 'special:no', 'special:nevermind',
                  'special:makerule', 'special:failed', 'special:help',
                  'special:thank_you', 'special:hello',
                  'special:sorry', 'special:cool', 'special:wakeup']
TYPES = {
    'Location': (['=='], ['LOCATION', 'location:current_location', 'location:work', 'location:home']),
    'Boolean':  ([], []), # booleans are handled per-parameter, like enums
    'String': (['==', '=~', '~=', 'starts_with', 'ends_with'], ['""', ('"', '$word_list', '"'), 'QUOTED_STRING']),
    'Date': (['==', '>=', '<='], [
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
        ('$constant_Date', '-', '$constant_Measure(ms)'),
        ('$constant_Time',),
        ]),
    'Time': (['=='], ['TIME']),
    'Currency': (['==', '>=', '<='], ['CURRENCY']),
    'Number': (['==', '>=', '<='], ['NUMBER', '1', '0']),
    'Entity(tt:username)': (['=='], ['USERNAME', ('"', '$word_list', '"', '^^tt:username'), ('$constant_String',) ]),
    'Entity(tt:contact)': (['=='], [('$constant_Entity(tt:username)',) ]),
    'Entity(tt:hashtag)': (['=='], ['HASHTAG', ('"', '$word_list', '"', '^^tt:hashtag'), ('$constant_String',) ]),
    'Entity(tt:phone_number)': (['=='], ['PHONE_NUMBER', 'USERNAME', ('"', '$word_list', '"', '^^tt:username'), ('$constant_String',) ]),
    'Entity(tt:email_address)': (['=='], ['EMAIL_ADDRESS', 'USERNAME', ('"', '$word_list', '"', '^^tt:username'), ('$constant_String',) ]),
    'Entity(tt:url)': (['=='], ['URL', ('$constant_String',) ]),
    'Entity(tt:path_name)': (['=='], ['PATH_NAME', ('$constant_String',) ]),
    'Entity(tt:picture)': (['=='], []),
    'Entity(tt:program)': (['=='], [])
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
             byte=["byte", "KB", "MB", "GB", "TB"])

MAX_ARG_VALUES = 4
MAX_STRING_ARG_VALUES = 5


def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()


def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', name.lower())


class ThingTalkGrammar(ShiftReduceGrammar):
    '''
    The grammar of ThingTalk
    '''
    
    def __init__(self, filename=None, split_device=False, **kw):
        super().__init__(**kw)
        self._input_dictionary = None
        self._split_device = split_device
        if filename is not None:
            self.init_from_file(filename)
        
    def reset(self):
        queries = OrderedDict()
        actions = OrderedDict()
        functions = dict(queries=queries, actions=actions)
        self.functions = functions
        self.allfunctions = []
        self.entities = []
        self._enum_types = OrderedDict()
        self.devices = []
        self._grammar = None
    
    def _process_devices(self, devices):
        for device in devices:
            if device['kind_type'] in ('global', 'discovery', 'category'):
                continue
            self.devices.append('device:' + device['kind'])
            if device['kind'] == 'org.thingpedia.builtin.test':
                continue
            
            for function_type in ('queries', 'actions'):
                for name, function in device[function_type].items():
                    function_name = '@' + device['kind'] + '.' + name
                    paramlist = []
                    self.functions[function_type][function_name] = paramlist
                    self.allfunctions.append(function_name)
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

        with tf.gfile.Open(filename, 'r') as fp:
            thingpedia = json.load(fp)
        
        self._devices = thingpedia['devices']
        self._process_devices(thingpedia['devices'])
        self._process_entities(thingpedia['entities'])

        self.complete()

    def init_from_url(self, snapshot=-1, thingpedia_url=None):
        self.reset()

        if thingpedia_url is None:
            thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')
        ssl_context = ssl.create_default_context()

        with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1', context=ssl_context) as res:
            self._devices = res
            self._process_devices(json.load(res)['data'])

        with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
            self._process_entities(json.load(res)['data'])

        self.complete()
    
    def complete(self):
        self.num_functions = len(self.functions['queries']) + len(self.functions['actions'])
        
        GRAMMAR = OrderedDict({
            '$input': [('$rule',),
                       ('executor', '=', '$constant_Entity(tt:username)', ':', '$rule'),
                       ('policy', '$policy'),
                       ('bookkeeping', '$bookkeeping')],
            '$bookkeeping': [('special', '$special'),
                             ('answer', '$constant_Any')],
            '$special': [(x,) for x in SPECIAL_TOKENS],
            '$rule':  [('$stream', '=>', '$action'),
                       ('$stream_join', '=>', '$action'),
                       ('now', '=>', '$table', '=>', '$action'),
                       ('now', '=>', '$action'),
                       ('$rule', 'on', '$param_passing')],
            '$policy': [('true', ':', '$policy_body'),
                        ('$filter', ':', '$policy_body')],
            '$policy_body': [('now', '=>', '$policy_action'),
                             ('$policy_query', '=>', 'notify'),
                             ('$policy_query', '=>', '$policy_action')],
            '$policy_query': [('*',),
                              #('$thingpedia_device_star'),
                              ('$thingpedia_queries',),
                              ('$thingpedia_queries', 'filter', '$filter')],
            '$policy_action': [('*',),
                               #('$thingpedia_device_star'),
                               ('$thingpedia_actions',),
                               ('$thingpedia_actions', 'filter', '$filter')],
            '$table': [('$thingpedia_queries',),
                       ('(', '$table', ')', 'filter', '$filter'),
                       ('aggregate', 'min', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'max', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'sum', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'avg', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'count', 'of', '(', '$table', ')'),
                       ('aggregate', 'argmin', '$out_param_Any', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('aggregate', 'argmax', '$out_param_Any', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('$table_join',),
                       ('window', '$constant_Number', ',', '$constant_Number', 'of', '(', '$stream', ')'),
                       ('timeseries', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$stream', ')'),
                       ('sequence', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('history', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$table', ')')
                       ],
            '$table_join': [('(', '$table', ')', 'join', '(', '$table', ')'),
                            ('$table_join', 'on', '$param_passing')],
            '$stream': [('timer', 'base', '=', '$constant_Date', ',', 'interval', '=', '$constant_Measure(ms)'),
                        ('attimer', 'time', '=', '$constant_Time',),
                        ('monitor', '(', '$table', ')'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '$out_param_Any'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '[', '$out_param_list', ']'),
                        ('edge', '(', '$stream', ')', 'on', '$filter'),
                        ('edge', '(', '$stream', ')', 'on', 'true'),
                        #('$stream_join',)
                        ],
            '$stream_join': [('(', '$stream', ')', 'join', '(', '$table', ')'),
                             ('$stream_join', 'on', '$param_passing')],
            '$action': [('notify',),
                        ('return',),
                        ('$thingpedia_actions',)],
            '$thingpedia_queries': [('$thingpedia_queries', '$const_param')],
            '$thingpedia_actions': [('$thingpedia_actions', '$const_param')],
            '$param_passing': [],
            '$const_param': [],
            '$out_param_Any': [],
            '$out_param_Array(Any)': [],
            '$out_param_list': [('$out_param_Any',),
                                ('$out_param_list', ',', '$out_param_Any')],

            '$filter': [('$or_filter',),
                        ('$filter', 'and', '$or_filter',)],
            '$or_filter': [('$atom_filter',),
                           ('not', '$atom_filter',),
                           ('$or_filter', 'or', '$atom_filter')
                           ],
            '$atom_filter': [('$thingpedia_queries', '{', 'true', '}'),
                             ('$thingpedia_queries', '{', '$filter', '}')],

            '$constant_Array': [('[', '$constant_array_values', ']',)],
            '$constant_array_values': [('$constant_Any',),
                                       ('$constant_array_values', ',', '$constant_Any')],
            '$constant_Any': OrderedSet(),

            '$word_list': [('SPAN',),],
                           #('WORD',),
                           #('$word_list', 'WORD',)]
        })
        
        def add_type(type, value_rules, operators):
            assert all(isinstance(x, tuple) for x in value_rules)
            GRAMMAR['$constant_' + type] = value_rules
            if not type.startswith('Entity(') and type != 'Time':
                GRAMMAR['$constant_Any'].add(('$constant_' + type,))
            for op in operators:
                GRAMMAR['$atom_filter'].append(('$out_param_' + type, op, '$constant_' + type))
                # FIXME reenable some day
                #GRAMMAR['$atom_filter'].add(('$out_param', op, '$out_param'))
            GRAMMAR['$atom_filter'].append(('$out_param_' + type, 'in_array', '[', '$constant_' + type, ',', '$constant_' + type, ']'))
            GRAMMAR['$atom_filter'].append(('$out_param_Array(' + type + ')', 'contains', '$constant_' + type))
            GRAMMAR['$out_param_' + type] = []
            GRAMMAR['$out_param_Array(' + type + ')'] = []
            GRAMMAR['$out_param_Any'].append(('$out_param_' + type,))
            GRAMMAR['$out_param_Any'].append(('$out_param_Array(' + type + ')',))

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
            value_rules += [('$constant_Measure(' + base_unit + ')', '$constant_Number', 'unit:' + unit) for unit in units]
            operators, _ = TYPES['Number']
            add_type('Measure(' + base_unit + ')', value_rules, operators)
        for i in range(MAX_ARG_VALUES):
            GRAMMAR['$constant_Measure(ms)'].append(('DURATION_' + str(i),))

        # well known entities
        add_type('Entity(tt:device)', [(device,) for device in self.devices], ['=='])
        #add_type('Entity(tt:device)', [], ['='])

        # other entities
        for generic_entity, has_ner in self.entities:
            if has_ner:
                value_rules = [('GENERIC_ENTITY_' + generic_entity + "_" + str(i), ) for i in range(MAX_ARG_VALUES)]
                value_rules.append(('$constant_String',))
                value_rules.append(('"', '$word_list', '"', '^^' + generic_entity,))
            else:
                value_rules = []
            add_type('Entity(' + generic_entity + ')', value_rules, ['=='])
            
        # maps a parameter to the list of types it can possibly have
        # over the whole Thingpedia
        param_types = OrderedDict()
        # add a parameter over the source
        param_types['source'] = OrderedSet()
        param_types['source'].add(('Entity(tt:contact)', 'out'))
        
        if self._split_device:
            GRAMMAR['$thingpedia_device_name'] = []
            for function_type in ('queries', 'actions'):
                GRAMMAR['$thingpedia_' + function_type].append(('$thingpedia_device_name', '$thingpedia_' + function_type + '_name',))
                GRAMMAR['$thingpedia_' + function_type + '_name'] = []
            for device in self._devices:
                if device['kind_type'] in ('global', 'discovery', 'category'):
                    continue
                kind = device['kind']
                if kind == 'org.thingpedia.builtin.test':
                    continue
                
                GRAMMAR['$thingpedia_device_name'].append(('@@' + kind,))
                for function_type in ('queries', 'actions'):
                    for name, function in device[function_type].items():
                        function_name = '@' + kind + '.' + name
                        GRAMMAR['$thingpedia_' + function_type + '_name'].append((function_name,))
        else:
            for function_type in ('queries', 'actions'):
                for function_name, params in self.functions[function_type].items():
                    GRAMMAR['$thingpedia_' + function_type].append((function_name,))

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
                    if param_direction == 'in':
                        # add the corresponding in out direction too, so we can handle
                        # filters on it for policies
                        param_types[param_name].add((param_type, 'out'))

        for param_name, options in param_types.items():
            for (param_type, param_direction) in options:
                if param_type.startswith('Enum('):
                    enum_type = self._enum_types[param_type]
                    for enum in enum_type:
                        #GRAMMAR['$atom_filter'].add(('$out_param', '==', 'enum:' + enum))
                        if param_direction == 'in':
                            GRAMMAR['$const_param'].append(('param:' + param_name + ':' + param_type, '=', 'enum:' + enum))
                        else:
                            # NOTE: enum filters don't follow the usual convention for filters
                            # this is because, linguistically, it does not make much sense to go
                            # through $out_param: enum parameters are often implicit
                            # one does not say "if the mode of my hvac is off", one says "if my hvac is off"
                            # (same, and worse, with booleans)
                            GRAMMAR['$atom_filter'].append(('param:' + param_name + ':' + param_type, '==', 'enum:' + enum))
                else:
                    if param_direction == 'out':
                        if param_type != 'Boolean':
                            GRAMMAR['$out_param_' + param_type].append(('param:' + param_name + ':' + param_type,))
                        else:
                            GRAMMAR['$atom_filter'].append(('param:' + param_name + ':' + param_type, '==', 'true'))
                            GRAMMAR['$atom_filter'].append(('param:' + param_name + ':' + param_type, '==', 'false'))
                    else:
                        if param_type == 'String':
                            GRAMMAR['$param_passing'].append(('param:' + param_name + ':' + param_type, '=', '$out_param_Any'))
                            GRAMMAR['$param_passing'].append(('param:' + param_name + ':' + param_type, '=', 'event'))
                        elif param_type.startswith('Entity('):
                            GRAMMAR['$param_passing'].append(('param:' + param_name + ':' + param_type, '=', '$out_param_' + param_type))
                            GRAMMAR['$param_passing'].append(('param:' + param_name + ':' + param_type, '=', '$out_param_String'))
                        else:
                            GRAMMAR['$param_passing'].append(('param:' + param_name + ':' + param_type, '=', '$out_param_' + param_type))
                    if param_direction == 'in':
                        if param_type == 'Any':
                            GRAMMAR['$const_param'].append(('param:' + param_name + ':' + param_type, '=', '$constant_String'))
                        elif param_type != 'Boolean':
                            GRAMMAR['$const_param'].append(('param:' + param_name + ':' + param_type, '=', '$constant_' + param_type))
                        else:
                            GRAMMAR['$const_param'].append(('param:' + param_name + ':' + param_type, '=', 'true'))
                            GRAMMAR['$const_param'].append(('param:' + param_name + ':' + param_type, '=', 'false'))

        self._grammar = GRAMMAR

    def tokenize_program(self, program):
        if isinstance(program, str):
            program = program.split(' ')

        in_string = False
        string_begin = None
        string_end = None
        for i, token in enumerate(program):
            if self._flatten:
                yield self.dictionary[token], None
                continue

            if token == '"':
                in_string = not in_string
                if in_string:
                    string_begin = i+1
                else:
                    string_end = i
                    yield self._span_id, program[string_begin:string_end]
                    string_begin = None
                    string_end = None
                yield self.dictionary[token], None
            elif in_string:
                continue
            elif token not in self.dictionary:
                raise ValueError("Invalid token " + token)
            else:
                yield self.dictionary[token], None

    def decode_program(self, input_sentence, tokenized_program):
        program = []
        tokenized_program = np.reshape(tokenized_program, (-1, 3))
        
        for i in range(len(tokenized_program)):
            token = tokenized_program[i, 0]
            if token == self._span_id:
                begin_position = tokenized_program[i, 1]
                end_position = tokenized_program[i, 2]
                input_span = self._input_dictionary.decode_list(input_sentence[begin_position:end_position+1])
                program.extend(input_span)
            else:
                program.append(self.tokens[token])
        return program

    def set_input_dictionary(self, input_dictionary):
        #non_entity_words = [x for x in input_dictionary if not x[0].isupper() and x != '$']
        self._input_dictionary = input_dictionary
        
        self.construct_parser(self._grammar, copy_terminals={
            'SPAN': []
        })
        self._span_id = self.dictionary['SPAN']

        if not self._quiet:
            print('num functions', self.num_functions)
            print('num queries', len(self.functions['queries']))
            print('num actions', len(self.functions['actions']))
            print('num other', len(self.tokens) - self.num_functions - self.num_control_tokens)

    def eval_metrics(self):
        def get_functions(program, what=None):
            return [x for x in program[:, 0] if self.tokens[x].startswith('@')]
        
        def accuracy_without_parameters(predictions, labels, features):
            batch_size, predictions, labels = adjust_predictions_labels(predictions, labels,
                                                                        num_elements_per_time=3)
            weights = tf.ones((batch_size,), dtype=tf.float32)
            ok = tf.to_float(tf.reduce_all(tf.equal(predictions[:,:,0], labels[:,:,0]), axis=1))
            return ok, weights
        
        return {
            "accuracy": accuracy,
            "grammar_accuracy": grammar_accuracy,
            "function_accuracy": make_pyfunc_metric_fn(
                lambda pred, label: get_functions(pred, 'p') == get_functions(label, 'l')),
            "accuracy_without_parameters": accuracy_without_parameters
        }


if __name__ == '__main__':
    grammar = ThingTalkGrammar(sys.argv[1], flatten=False)
    dictionary, _ = load_dictionary(sys.argv[2], use_types=True, grammar=grammar)
    grammar.set_input_dictionary(dictionary)
    #grammar.dump_tokens()
    #grammar.normalize_all(sys.stdin)
    for line in sys.stdin:
        try:
            sentence, program = line.strip().split('\t')[1:3]
            sentence = sentence.split(' ')
            
            tokenized, length = grammar.tokenize_to_vector(sentence, program)
            vector, length = grammar.vectorize_program(sentence, tokenized)
            reconstructed = grammar.reconstruct_program(sentence, vector)
            assert program == ' '.join(reconstructed)
            #print()
            #print(program)
            #grammar.print_prediction(sentence_vector, vector)
        except:
            print(line.strip())
            grammar.print_prediction(sentence, vector)
            raise