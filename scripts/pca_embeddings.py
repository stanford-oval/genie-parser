#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
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

import numpy as np
import tensorflow as tf

import sys

data = []
words = []
with open(sys.argv[1], 'r') as fp:
    for i, line in enumerate(fp):
        if i % 100000 == 0:
            print(i)
        sp = line.strip().split(' ')
        words.append(sp[0])
        data.append([float(x) for x in sp[1:]])
words = np.array(words, dtype=str)
data = np.array(data)
print('Loaded', len(words), data.shape)

X = tf.constant(data)
mean = tf.reduce_mean(X, axis=0)
X -= mean

S, U, V = tf.svd(X)

U = U[:,:150]
U *= S[:150]

with tf.Session() as sess:
    data = U.eval(session=sess)
print('Transformed', data.shape)
with open(sys.argv[2], 'w') as fp:
    for i, (word, vec) in enumerate(zip(words, data)):
        if i % 100000 == 0:
            print(i)
        print(word, *vec, file=fp)
