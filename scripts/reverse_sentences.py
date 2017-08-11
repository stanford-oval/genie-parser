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

import sys
import itertools

def parse_tree(tree_seq, i):
    if tree_seq[i] == '(':
       tree0, i0 = parse_tree(tree_seq, i+1)
       tree1, i1 = parse_tree(tree_seq, i0)
       assert tree_seq[i1] == ')'
       return (tree0, tree1), i1+1
    else:
       assert tree_seq[i] != ')'
       return tree_seq[i], i+1

def reverse_tree(tree):
    if isinstance(tree, tuple):
       return (reverse_tree(tree[1]), reverse_tree(tree[0]))
    else:
       return tree

def write_tree(tree):
    if isinstance(tree, tuple):
       return itertools.chain(['('], write_tree(tree[0]), write_tree(tree[1]), [')'])
    else:
       return [tree]

for line in sys.stdin:
    id, sentence, label, parse = line.strip().split('\t')
    parse, _ = parse_tree(parse.split(' '), 0)
    parse = ' '.join(write_tree(reverse_tree(parse)))
    print(id, ' '.join(reversed(sentence.split(' '))), label, parse, sep='\t')
