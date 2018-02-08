#!/usr/bin/python3

import sys
import random
import itertools
import os

def is_function(token):
    if token.startswith('@') and token != '@org.thingpedia.builtin.thingengine.builtin.say':
        return True
    if token in ('timer', 'attimer'):
        return True
    return False

def readfile(filename):
    with open(filename) as fp:
        return [x.strip().split('\t') for x in fp]

def writefile(filename, data):
    print('writing to', filename, '...')
    with open(filename, 'w') as fp:
        for sentence in data:
            print(*sentence, sep='\t', file=fp)

input = readfile(sys.argv[1])

dirname = os.path.dirname(sys.argv[1])
filename = os.path.basename(sys.argv[1])
filename = filename[:filename.rindex('.')]

complexity = {
'prim': [],
'prim-filter': [],
'prim-filter-newtt': [],
'compound': [],
'compound-1pp': [],
'compound-multipp': [],
'compound-filter': [],
'compound-filter-newtt': []
}

NEWTT_TOKENS = set(['new', 'edge', 'in_array', 'not', 'or'])

for id, sentence, code in input:
    program = code.split(' ')
    
    num_pp = 0
    num_functions = 0
    has_filter = False
    is_newtt = False
    for i, token in enumerate(program):
        if token in NEWTT_TOKENS:
            is_newtt = True
        if token.startswith('param:') and i < len(program)-2 \
            and program[i+1] == '=' and program[i+2].startswith('param:'):
            num_pp += 1
            if token == 'param:message:Any':
                is_newtt = True
        elif token == 'filter':
            has_filter = True
        elif is_function(token):
            num_functions += 1

    is_compound = num_functions > 1

    if is_compound:
        if is_newtt:
            complexity['compound-filter-newtt'].append((id,sentence,code))
        elif has_filter:
            complexity['compound-filter'].append((id,sentence,code))
        elif num_pp > 1:
            complexity['compound-multipp'].append((id,sentence,code))
        elif num_pp > 0:
            complexity['compound-1pp'].append((id,sentence,code))
        else:
            complexity['compound'].append((id,sentence,code))
    else:
        if is_newtt:
            complexity['prim-filter-newtt'].append((id,sentence,code))
        elif has_filter:
            complexity['prim-filter'].append((id,sentence,code))
        else:
            complexity['prim'].append((id,sentence,code))


for key, values in complexity.items():
    print(key, len(values))
    writefile(os.path.join(dirname, filename + '-' + key + '.tsv'), values)
