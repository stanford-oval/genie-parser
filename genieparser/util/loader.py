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
'''
Created on Mar 16, 2017

@author: gcampagn
'''

import re
import numpy as np


def vectorize(sentence, words, max_length=None, add_eos=False, add_start=False):
    if isinstance(sentence, str):
        sentence = sentence.split(' ')
    if max_length is None:
        max_length = len(sentence)
        if add_start:
            max_length += 1
        if add_eos:
            max_length += 1
    vector = np.zeros((max_length,), dtype=np.int32)
    if add_start:
        vector[0] = words['<s>']
        i = 1
    else:
        i = 0
    for word in sentence:
        word = word.strip()
        if len(word) == 0:
            raise ValueError("empty token in " + str(sentence))
        if word[0].isupper() and '*' in word:
            word, _ = word.rsplit('*', maxsplit=1)
        if word in words:
            vector[i] = words[word]
        elif '<unk>' in words:
            #print("sentence: ", sentence, "; word: ", word)
            vector[i] = words['<unk>']
        else:
            raise ValueError('Unknown token ' + word)
        i += 1
        if i == max_length:
            break
    length = i
    if add_eos:
        if length < max_length:
            vector[length] = words['</s>']
            length += 1
        else:
            print("unterminated sentence", sentence)
    else:
        if length == max_length and length < len(sentence):
            print("truncated sentence", sentence)
    return (vector, length)