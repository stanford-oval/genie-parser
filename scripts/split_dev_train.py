#!/usr/bin/python3
#
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

import random
import itertools
import sys
import os

def is_function(x):
    return x.startswith('@') or x in ('timer', 'attimer', 'now', 'notify')

def get_functions(prog):
    return tuple(x for x in prog.split(' ') if is_function(x))

def is_compound(prog):
    return len(get_functions(prog)) >= 2

def readfile(filename):
    with open(filename) as fp:
        return [x.strip().split('\t') for x in fp]

def writefile(filename, data):
    with open(filename, 'w') as fp:
        for sentence in data:
            print(*sentence, sep='\t', file=fp)

def is_remote(prog):
    prog = prog.split(' ')
    for i in range(len(prog)-1):
        if (prog[i].startswith('tt:') or prog[i].startswith('@')) and prog[i+1].startswith('USERNAME_'):
            return True
    return False

traindevfiles = set()
generatedfiles = set()

with os.scandir(sys.argv[1]) as iter:
    for entry in iter:
        if not entry.is_file() or not entry.name.endswith('.tsv'):
            continue
        if entry.name.endswith('-train+dev.tsv'):
            print('Found train+dev set ' + entry.name[:-len('-train+dev.tsv')])
            traindevfiles.add(entry.path)
        elif entry.name.endswith('-train.tsv') or entry.name.endswith('-dev.tsv') or entry.name in ('train.tsv', 'train-nosynthetic.tsv', 'dev.tsv'):
            continue
        elif entry.name.startswith('generated'):
            generatedfiles.add(entry.path)
            print('Found generated set ' + entry.name[:-4])
        else:
            print('Ignored set ' + entry.name[:-4])

traindevsets = dict()
traindevall = []
traindevprogs = set()
traindevsigs = set()

for filename in traindevfiles:
    prefix = os.path.basename(filename)[:-len('-train+dev.tsv')]
    sentences = readfile(filename)
    print('%d sentences in %s' % (len(sentences), prefix))
    traindevsets[prefix] = sentences
    traindevall += sentences
    progs = set(x[2] for x in sentences)
    print('= %d programs' % len(progs))
    traindevprogs |= progs
    sigs = set(map(get_functions, progs))
    print('= %d signatures' % len(sigs))
    traindevsigs |= sigs

print('Total train+dev: %d sentences, %d programs, %d signatures' % (len(traindevall), len(traindevprogs), len(traindevsigs)))

dev_sigs = set(random.sample(traindevsigs, len(traindevsigs)//10))
#dev_progs = set(x for x in dev_progs if is_compound(x))

print('%d dev signatures' % len(dev_sigs))
devall = [x for x in traindevall if get_functions(x[2]) in dev_sigs]
print('%d dev sentences' % len(devall))

trainsets = dict()
devsets = dict()
trainall = []
for prefix,dataset in traindevsets.items():
    train = [x for x in dataset if get_functions(x[2]) not in dev_sigs]
    print('%d %s train sentences' % (len(train), prefix))
    trainsets[prefix] = train
    trainall += train
    dev = [x for x in dataset if get_functions(x[2]) in dev_sigs]
    print('%d %s dev sentences' % (len(dev), prefix))
    devsets[prefix] = dev

generated = []
for filename in generatedfiles:
    prefix = os.path.basename(filename)[:-4]
    other = readfile(filename)
    print('%d %s train sentences' % (len(other), prefix))
    other = [x for x in other if get_functions(x[2]) not in dev_sigs]
    print('= %d after filtering' % len(other))
    generated += other

writefile(os.path.join(sys.argv[1], 'dev.tsv'), devall)
for prefix,dataset in trainsets.items():
    writefile(os.path.join(sys.argv[1], prefix + '-train.tsv'), dataset)
for prefix,dataset in devsets.items():
    writefile(os.path.join(sys.argv[1], prefix + '-dev.tsv'), dataset)
if len(generated) > 0:
    writefile(os.path.join(sys.argv[1], 'filtered-generated.tsv'), generated)
    writefile(os.path.join(sys.argv[1], 'train-nosynthetic.tsv'), trainall)
    writefile(os.path.join(sys.argv[1], 'train.tsv'), itertools.chain(trainall, generated))
else:
    writefile(os.path.join(sys.argv[1], 'train.tsv'), trainall)
