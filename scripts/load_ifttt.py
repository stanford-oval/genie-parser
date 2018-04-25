#!/usr/bin/python2

'''
Created on Apr 24, 2018

@author: gcampagn
'''

from __future__ import print_function

import cPickle as pickle
import sys

def run():
    with open(sys.argv[1], 'rb') as fp:
        data = pickle.load(fp)
    
    channels = set()
    triggers = set()
    actions = set()
    for what in ('train', 'dev', 'test'):
        with open('ifttt-' + what + '.tsv', 'w') as fout:
            for i, sentence in enumerate(data[what]):
                channels.add(sentence['label_names'][0])
                triggers.add(sentence['label_names'][1])
                channels.add(sentence['label_names'][2])
                actions.add(sentence['label_names'][3])
                print(what + str(i), ' '.join(sentence['words']), ' '.join(sentence['label_names']), sep='\t', file=fout)

    with open('ifttt-output-words.txt', 'w') as fout:
        for c in sorted(channels):
            print('channel', c, file=fout)
        for t in sorted(triggers):
            print('trigger', t, file=fout)
        for a in sorted(actions):
            print('action', a, file=fout)

if __name__ == '__main__':
    run()