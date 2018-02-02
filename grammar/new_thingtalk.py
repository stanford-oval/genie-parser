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
import itertools
import numpy as np
from .shift_reduce_grammar import ShiftReduceGrammar

from collections import OrderedDict

SPECIAL_TOKENS = ['special:yes', 'special:no', 'special:nevermind',
                  'special:makerule', 'special:failed']
TYPES = {
    'Location': (['=='], ['LOCATION', 'location:current_location', 'location:work', 'location:home']),
    'Boolean':  (['=='], ['true', 'false']),
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
               #('aggregate', 'min', 'PARAM', 'of', '(', '$table', ')'),
               #('aggregate', 'max', 'PARAM', 'of', '(', '$table', ')'),
               #('aggregate', 'sum', 'PARAM', 'of', '(', '$table', ')'),
               #('aggregate', 'avg', 'PARAM', 'of', '(', '$table', ')'),
               #('aggregate', 'count', 'of', '(', '$table', ')'),
               #('aggregate', 'argmin', 'PARAM', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
               #('aggregate', 'argmax', 'PARAM', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
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
                ('monitor', '(', '$table', ')', 'on', 'new', 'PARAM'),
                ('monitor', '(', '$table', ')', 'on', 'new', '[', '$out_param_list', ']'),
                ('edge', '(', '$stream', ')', 'on', '$filter'),
                #('edge', '(', '$stream', ')', 'on', 'true'),
                ('$stream_join',)],
    '$stream_join': [('(', '$stream', ')', 'join', '(', '$table', ')'),
                     ('$stream_join', 'on', '$param_passing')],
    '$thingpedia_queries': [('$thingpedia_queries', '$const_param'),
                            ('THINGPEDIA_QUERIES',)],
    '$thingpedia_actions': [('$thingpedia_actions', '$const_param'),
                            ('THINGPEDIA_ACTIONS',)],
    '$param_passing': [('PARAM', '=', 'PARAM'),
                       ('PARAM', '=', 'event')],
    '$const_param': [],
    '$out_param_list': [('PARAM',),
                        ('$out_param_list', ',', 'PARAM')],

    '$filter': [('$or_filter',),
                ('$filter', 'and', '$or_filter',)],
    '$or_filter': [('$atom_filter',),
                   ('not', '$atom_filter',),
                   ('$or_filter', 'or', '$atom_filter')],
    '$atom_filter': [('PARAM', 'in_array', '$constant_Array')],
    
    '$constant_Array': [('[', '$constant_array_values', ']',)],
    '$constant_array_values': [('$constant_Any',),
                               ('$constant_array_values', ',', '$constant_Any')],
    '$constant_Any': [],
})

MAX_ARG_VALUES = 4
MAX_STRING_ARG_VALUES = 5

def load_grammar():
    def add_type(type, value_rules, operators):
            assert all(isinstance(x, tuple) for x in value_rules)
            GRAMMAR['$const_param'].append(('PARAM', '=', '$constant_' + type))
            GRAMMAR['$constant_' + type] = value_rules
            GRAMMAR['$constant_Any'].append(('$constant_' + type,))
            for op in operators:
                GRAMMAR['$atom_filter'].append(('PARAM', op, '$constant_' + type))
                # FIXME reenable some day
                #GRAMMAR['$atom_filter'].add(('$out_param', op, '$out_param'))
            GRAMMAR['$atom_filter'].append(('PARAM', 'contains', '$constant_' + type))
            
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
    add_type('Entity(tt:device)', [('DEVICE',)], ['=='])
    
    # other entities
    add_type('Entity(*)', [('GENERIC_ENTITY',)], ['=='])
    
    # enums
    add_type('Enum', [('ENUM',)], ['=='])
load_grammar()


def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()


def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?\_]', re.sub('[()]', '', name.lower()))


class NewThingTalkGrammar(ShiftReduceGrammar):
    '''
    The grammar of New thingtalk
    '''
    
    def __init__(self, filename=None, flatten=True):
        super().__init__(flatten=flatten)
        if filename is not None:
            self.init_from_file(filename)
        
    def reset(self):
        self.entities = []
        self.devices = []
        self._token_canonicals = dict()
    
    def _process_devices(self, devices, extensible_terminals):
        params = set()
        enums = set()

        for device in devices:
            if device['kind_type'] in ('global', 'category', 'discovery'):
                continue
            kind = device['kind']
            self.devices.append('device:' + kind)
            self._token_canonicals['device:' + kind] = device.get('kind_canonical', None)
            
            for function_type in ('queries', 'actions'):
                for name, function in device[function_type].items():
                    function_name = '@' + device['kind'] + '.' + name
                    extensible_terminals['THINGPEDIA_' + function_type.upper()].append(function_name)
                    self._token_canonicals[function_name] = function['canonical']

                    for argname, argtype, argcanonical in zip(function['args'], function['schema'], function['argcanonicals']):
                        if argtype in TYPE_RENAMES:
                            argtype = TYPE_RENAMES[argtype]
                        params.add('param:' + argname)
                        self._token_canonicals['param:' + argname] = argcanonical
                                    
                        if argtype.startswith('Array('):
                            elementtype = argtype[len('Array('):-1]
                            if elementtype in TYPE_RENAMES:
                                argtype = 'Array(' + TYPE_RENAMES[elementtype] + ')'
                        else:
                            elementtype = argtype
                        if elementtype.startswith('Enum('):
                            enum_variants = elementtype[len('Enum('):-1].split(',')
                            for enum in enum_variants:
                                enums.add('enum:' + enum)
                                self._token_canonicals['enum:' + enum] = clean(enum)

        extensible_terminals['ENUM'] = list(enums)
        extensible_terminals['ENUM'].sort()
        extensible_terminals['PARAM'] = list(params)
        extensible_terminals['PARAM'].sort()

    def _process_entities(self, entities, extensible_terminals):
        for entity in entities:
            if entity['is_well_known'] == 1:
                continue
            if entity['has_ner_support']:
                for i in range(MAX_ARG_VALUES):
                    token = 'GENERIC_ENTITY_' + entity['type'] + "_" + str(i)
                    extensible_terminals['GENERIC_ENTITY'].append(token)
                    self._token_canonicals[token] = ' '.join(tokenize(entity['name'])).strip() + ' ' + str(i)
                self.entities.append(entity['type'])
    
    def init_from_file(self, filename):
        self.reset()
        extensible_terminals = {
            'DEVICE': self.devices,
            'THINGPEDIA_QUERIES': [],
            'THINGPEDIA_ACTIONS': [],
            'PARAM': [],
            'ENUM': [],
            'GENERIC_ENTITY': []
        }

        with open(filename, 'r') as fp:
            thingpedia = json.load(fp)
            
        self._process_devices(thingpedia['devices'], extensible_terminals)
        self._process_entities(thingpedia['entities'], extensible_terminals)

        self.complete(extensible_terminals)

    def init_from_url(self, snapshot=-1, thingpedia_url=None):
        if thingpedia_url is None:
            thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')
        ssl_context = ssl.create_default_context()
        extensible_terminals = {
            'DEVICE': self.devices,
            'THINGPEDIA_QUERIES': list(self.functions['queries'].keys()),
            'THINGPEDIA_ACTIONS': list(self.functions['actions'].keys()),
            'PARAM': [],
            'ENUM': [],
            'GENERIC_ENTITY': []
        }

        with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1', context=ssl_context) as res:
            self._process_devices(json.load(res)['data'], extensible_terminals)

        with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
            self._process_entities(json.load(res)['data'], extensible_terminals)
            
        self.complete(extensible_terminals)
    
    def complete(self, extensible_terminals):
        num_queries = len(extensible_terminals['THINGPEDIA_QUERIES'])
        num_actions = len(extensible_terminals['THINGPEDIA_ACTIONS'])
        self.num_functions = num_queries + num_actions
        
        print('num functions', self.num_functions)
        print('num queries', num_queries)
        print('num actions', num_actions)
        self.tokens += self.construct_parser(GRAMMAR, extensible_terminals)
        
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
            
    def _embed_token(self, token, input_words, input_embeddings):
        input_embed_size = input_embeddings.shape[-1]
        token_embedding = np.zeros((input_embed_size,), dtype=np.float32)
        canonical = self._token_canonicals[token]
        if not canonical:
            print("WARNING: token %s has no canonical" % (token,))
            return token_embedding
        for canonical_token in canonical.split(' '):
            if canonical_token in ('in', 'on', 'of'):
                continue
            if canonical_token in input_words:
                token_embedding += input_embeddings[input_words[canonical_token]]
            else:
                print("WARNING: missing word %s in canonical for output token %s" % (canonical_token, token))
                token_embedding += input_embeddings[input_words['<unk>']]
        if np.any(np.isnan(token_embedding)):
            raise ValueError('Embedding for ' + token + ' is NaN')
        return token_embedding

    def get_embeddings(self, input_words, input_embeddings):
        all_embeddings = {
            'actions': np.identity(self.output_size['actions'], dtype=np.float32)
        }
        
        depth = input_embeddings.shape[-1]
        for key in self.extensible_terminal_list:
            size = self.output_size[key]
            embedding_matrix = np.zeros((size, depth), dtype=np.float32)
            for i, token in enumerate(self.extensible_terminals[key]):
                embedding_matrix[i] = self._embed_token(token, input_words, input_embeddings)
            assert not np.any(np.isnan(embedding_matrix))
            all_embeddings[key] = embedding_matrix
        
        return all_embeddings


if __name__ == '__main__':
    grammar = NewThingTalkGrammar(sys.argv[1], flatten=False)
    #grammar.dump_tokens()
    #grammar.normalize_all(sys.stdin)
    vectors = dict()
    for key in grammar.output_size:
        vectors[key] = []
    for line in sys.stdin:
        line = line.strip()
        try:
            vector, length = grammar.vectorize_program(line)
            assert ' '.join(grammar.reconstruct_program(vector, ignore_errors=False)) == line
            for key, vec in vector.items():
                vectors[key].append(vec)
        except:
            print(line)
            raise
        
    for key in grammar.output_size:
        vectors[key] = np.array(vectors[key])
    np.savez('programs.npz', **vectors)
    #for i, name in enumerate(grammar.state_names):
    #    print i, name
