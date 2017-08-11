#!/usr/bin/python3
#
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

import random
import itertools

def get_functions(prog):
    return [x for x in prog.split(' ') if x.startswith('tt:') and not x.startswith('tt:$builtin.')]

def is_compound(prog):
    return len(get_functions(prog)) >= 2

def readfile(filename):
    with open(filename) as fp:
        return [x.strip().split('\t') for x in fp]

def writefile(filename, data):
    with open(filename, 'w') as fp:
        for sentence in data:
            print(*sentence, sep='\t', file=fp)

paraall = readfile('./paraphrasing-train+dev.tsv')
print('%d sentences in paraphrasing-train+dev' % len(paraall))
paraall_progs = set(x[2] for x in paraall)
print('= %d programs' % len(paraall_progs))

dev_progs = random.sample(paraall_progs, len(paraall_progs)//10)
#dev = readfile('./dev.tsv')
dev_progs = set(x for x in dev_progs if is_compound(x))
#dev_progs = set(x[1] for x in dev)
print('%d dev programs' % len(dev_progs))
dev = [x for x in paraall if x[2] in dev_progs]
print('%d dev sentences' % len(dev))
paratrain = [x for x in paraall if x[2] not in dev_progs]
print('%d para train sentences' % len(paratrain))

base_author = readfile('./base-author.tsv')
print('%d base author sentences' % len(base_author))
base_author = [x for x in base_author if x[2] not in dev_progs]
print('= %d after filtering' % len(base_author))

other = sum((readfile(x) for x in ('./generated.tsv', './generated-cheatsheet.tsv')), [])
print('%d other train sentences' % len(other))
other = [x for x in other if x[2] not in dev_progs]
print('= %d after filtering' % len(other))

writefile('./dev.tsv', dev)
writefile('./paraphrasing-train.tsv', paratrain)
writefile('./filtered-base-author.tsv', base_author)
writefile('./filtered-generated.tsv', other)
