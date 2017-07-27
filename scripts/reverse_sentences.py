#!/usr/bin/python3

import sys

for line in sys.stdin:
    sentence, label = line.strip().split('\t')
    print(' '.join(reversed(sentence.split(' '))), label, sep='\t')
