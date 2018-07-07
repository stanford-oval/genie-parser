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
Created on Mar 29, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

from models import Config, create_model
from util.loader import unknown_tokens, load_data, load_embeddings
from util.general_utils import get_minibatches

def show_pca(X, sentences):
    plt.figure()
    plt.plot(X[:,0], X[:,1], 'x')
    
    for x, sentence in zip(X, sentences):
        plt.text(x[0]-0.01, x[1]-0.01, sentence, horizontalalignment='center', verticalalignment='top', color=('black' if sentence not in ('raise','lower') else 'red'))
    
    plt.show()

def run():
    if len(sys.argv) < 2:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>>")
        sys.exit(1)

    np.random.seed(int(sys.argv[2]))

    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])

    embedding = config.input_embedding_matrix

    mean = np.mean(embedding, axis=0)
    centered_X = embedding - mean
    U, S, V = np.linalg.svd(centered_X)

    # take only the top 2 components
    V = V[:2]

    words = set(sys.argv[3:])
    for idx in np.random.choice(np.arange(len(embedding)), 10):
        words.add(config.reverse_dictionary[idx])

    words = list(words)
    X = np.zeros((len(words), embedding.shape[1]))
    for i,w in enumerate(words):
        X[i] = embedding[config.dictionary[w]]

    for i in range(len(sys.argv[3:])-1):
        w1 = sys.argv[3+i]
        w2 = sys.argv[3+i+1]

        print()
        print('pair', w1, w2)
        x1 = embedding[config.dictionary[w1]]
        x2 = embedding[config.dictionary[w2]]
        print('L2 distance', np.linalg.norm(x1-x2))
        print('Cosine similarity', np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2)))

    X_pca = np.matmul(X - mean, V.T)

    #show_pca(X_pca, words)

if __name__ == '__main__':
    run()
