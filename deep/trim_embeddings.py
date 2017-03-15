#!/usr/bin/python

import sys

words = set()
with open(sys.argv[1], 'r') as fp:
    for line in fp.readlines():
        words.add(line.strip())

for line in sys.stdin.readlines():
    sp = line.strip().split()
    if sp[0] in words:
        print line.strip()
