#!/usr/bin/python3
'''
Created on Dec 20, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import itertools
import tensorflow as tf

from models import Config
from util.loader import load_data, unknown_tokens

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

def run():
    if len(sys.argv) < 3:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Train Data>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])

    # load programs
    all_programs = np.load(sys.argv[2], allow_pickle=False)

    # concatenate all programs into one vector
    all_programs = np.reshape(all_programs, (-1,))
    
    counts = np.bincount(all_programs, minlength=config.output_size)
    #config.grammar.print_all_actions()
    
    # ignore the control tokens
    begin = int(sys.argv[3])
    end = int(sys.argv[4])

    for i in range(begin, end):
        lhs, rhs = config.grammar._parser.rules[i - config.grammar.num_control_tokens]
        print(i, counts[i], (lhs + ' -> ' + (' '.join(rhs))), sep='\t')

if __name__ == '__main__':
    run()
