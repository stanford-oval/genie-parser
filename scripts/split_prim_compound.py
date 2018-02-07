#!/usr/bin/python3

import sys
import random
import itertools
import os

def get_functions(prog):
    return [x for x in prog.split(' ') if x.startswith('@') and x != '@org.thingpedia.builtin.thingengine.builtin.say']

def is_compound(prog):
    return len(get_functions(prog)) >= 2

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

compound = [x for x in input if is_compound(x[2])]
print('compound', len(compound))

prim = [x for x in input if not is_compound(x[2])]
print('prim', len(prim))

writefile(os.path.join(dirname, filename + '-prim.tsv'), prim)
writefile(os.path.join(dirname, filename + '-compound.tsv'), compound)


