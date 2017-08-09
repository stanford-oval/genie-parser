#!/usr/bin/python3

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
