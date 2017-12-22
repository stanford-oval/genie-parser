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
'''
Created on Dec 22, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np

from models import Config
from util.loader import load_data, unknown_tokens
from util.general_utils import logged_loop, Progbar

def top_k(array, k=5):
    top_values = np.zeros((k,), dtype=array.dtype)
    top_indices = np.zeros((k,), dtype=array.dtype)
    top_set = 0
    for i in range(len(array)):
        added = False
        for j in range(top_set):
            if array[i] > top_values[j]:
                if top_set < k:
                    for j2 in range(top_set-1,j-1,-1):
                        top_values[j2+1] = top_values[j2]
                        top_indices[j2+1] = top_indices[j2]
                    top_set += 1
                else:
                    for j2 in range(k-2,j-1,-1):
                        top_values[j2+1] = top_values[j2]
                        top_indices[j2+1] = top_indices[j2]
                top_values[j] = array[i]
                top_indices[j] = i
                added = True
                break
        if not added and top_set < k:
            top_values[top_set] = array[i]
            top_indices[top_set] = i
            top_set += 1
    return top_values[:top_set], top_indices[:top_set]

def run():
    if len(sys.argv) < 3:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Train Data>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    
    if False:
        sentences, _, _, labels, _, = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
        print("unknown", unknown_tokens)
        
        correlation_matrix = np.zeros((config.dictionary_size, config.grammar.output_size), dtype=np.int32)
        for sentence,label in logged_loop(zip(sentences, labels), n=len(sentences)):
            correlation_matrix[np.ix_(sentence,label)] += 1
        
        np.save('correlations.npy', correlation_matrix)
    else:
        correlation_matrix = np.load('correlations.npy')
    
    with open('correlations.txt', 'w') as fp:
        for word in range(config.dictionary_size):
            if np.sum(correlation_matrix[word]) == 0:
                print(config.reverse_dictionary[word], ': missing', file=fp)
                continue

            row = correlation_matrix[word]
            _,indices = top_k(row, k=10)
            for action in indices:
                # ignore start/accept actions
                if action >= config.grammar.num_control_tokens:
                    lhs, rhs = config.grammar._parser.rules[action-config.grammar.num_control_tokens]
                    print(config.reverse_dictionary[word], ':', action, lhs, '->', ' '.join(rhs), '=', correlation_matrix[word, action], file=fp)
    
if __name__ == '__main__':
    run()