#!/usr/bin/python2

'''
Created on Apr 24, 2018

@author: gcampagn
'''

from __future__ import print_function

import cPickle as pickle
import urllib
import sys

channels = set()
triggers = set()
actions = set()
param_names = set()
param_values = set()

def do_one(fout, data, what, tag = None):
    for i, sentence in enumerate(data):
        if tag is not None:
            if 'tags' not in sentence or tag not in sentence['tags']:
                continue 
        
        ch1 = sentence['label_names'][0]
        channels.add(ch1)
        trigger = sentence['label_names'][1]
        triggers.add(trigger)
        ch2 = sentence['label_names'][2]
        channels.add(ch2)
        action = sentence['label_names'][3]
        actions.add(action)
        
        #trig_params, act_params = sentence['params']
        #trig_params.sort(key=lambda x:x[0])
        #act_params.sort(key=lambda x:x[0])
        
        prog = '@@' + ch1 + ' @' + trigger
        #for name, value in trig_params:
        #    if ' ' in name:
        #        # preprocessor fail, ignore
        #        continue
        #    param_names.add('param:' + name)
        #    value = urllib.quote_plus(value)
        #    param_values.add(value)
        #    prog += ' param:' + name + ' ' + '\"' + value + '\"'
        prog += ' => ' + '@@' + ch2 + ' @' + action
        #for name, value in act_params:
        #    if ' ' in name:
        #        # preprocessor fail, ignore
        #        continue
        #    param_names.add('param:' + name)
        #    value = urllib.quote_plus(value)
        #    param_values.add(value)
        #    prog += ' param:' + name + ' ' + '\"' + value + '\"'
        #if what == 'test':
        #    print(sentence)
        
        print(what + str(i), ' '.join(sentence['words']), prog, sep='\t', file=fout)

def run():
    with open(sys.argv[1], 'rb') as fp:
        data = pickle.load(fp)
    
    for what in ('train', 'dev', 'test'):
        with open('ifttt-' + what + '.tsv', 'w') as fout:
            do_one(fout, data[what], what)
    for tag in ('english', 'intelligible', 'gold'):
        with open('ifttt-test-' + tag + '.tsv', 'w') as fout:
            do_one(fout, data['test'], 'test', tag)

    with open('ifttt-output-words.txt', 'w') as fout:
        for c in sorted(channels):
            print('channel', '@@' + c, file=fout)
        for t in sorted(triggers):
            print('trigger', '@' + t, file=fout)
        for a in sorted(actions):
            print('action', '@' + a, file=fout)
        for p in sorted(param_names):
            print('param', p, file=fout)
        for v in sorted(param_values):
            print('value', v, file=fout)

if __name__ == '__main__':
    run()
