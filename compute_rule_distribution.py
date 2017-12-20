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
    train_data = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
    print("unknown", unknown_tokens)
    
    # concatenate all programs into one vector
    all_programs = np.reshape(train_data[1], (-1,))
    
    counts = np.bincount(all_programs, minlength=config.output_size)
    
    # ignore the control tokens
    counts = counts[config.grammar.num_control_tokens:]
    
    X = np.arange(config.grammar.num_control_tokens, config.output_size)
    
    plt.figure()
    plt.bar(X, counts, width=1)
    
    plt.show()

if __name__ == '__main__':
    run()