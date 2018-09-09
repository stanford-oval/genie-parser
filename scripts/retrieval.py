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
'''
Created on Apr 12, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from util.trainer import Trainer

from util.loader import load_dictionary, load_embeddings
from util.loader import unknown_tokens, load_data
from thingtalk.grammar import ThingtalkGrammar

def run():
    if len(sys.argv) < 5:
        print("** Usage: python " + sys.argv[0] + " <<Input Vocab>> <<Word Embeddings>> <<Train Set> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)
    
    words, reverse = load_dictionary(sys.argv[1], 'tt')
    print("%d words in dictionary" % (len(words),))
    embeddings_matrix = load_embeddings(sys.argv[2], words, embed_size=300)
    max_length = 60
    
    grammar = ThingtalkGrammar()
    
    train_data = load_data(sys.argv[3], words, grammar.dictionary,
                           reverse, grammar.tokens,
                           max_length)
    test_data = load_data(sys.argv[4], words, grammar.dictionary,
                             reverse, grammar.tokens,
                             max_length)
    print("unknown", unknown_tokens)

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            input_embed_matrix = tf.constant(embeddings_matrix)
            train_inputs = tf.nn.embedding_lookup([input_embed_matrix], np.array(train_data[0]))
            train_encoded = tf.reduce_sum(train_inputs, axis=1)
            #print train_encoded.eval()
            train_norm = tf.sqrt(tf.reduce_sum(train_encoded * train_encoded, axis=1))

            test_inputs = tf.nn.embedding_lookup([input_embed_matrix], np.array(test_data[0]))
            test_encoded = tf.reduce_sum(test_inputs, axis=1)
            test_norm = tf.sqrt(tf.reduce_sum(test_encoded * test_encoded, axis=1))
            #print test_encoded.eval()

            #print (train_encoded - test_encoded).eval()
        
            distances = tf.matmul(test_encoded, tf.transpose(train_encoded))
            distances /= tf.reshape(train_norm, (1, -1))
            distances /= tf.reshape(test_norm, (-1, 1))
            #print distances.eval()
            indices = tf.argmax(distances, axis=1)
            #print indices.eval()

            ok_0 = 0
            ok_ch = 0
            ok_fn = 0
            ok_full = 0
            correct_programs = set()
            gold_programs = set()
            for gold in test_data[2]:
                try:
                    gold = gold[:list(gold).index(grammar.end)]
                except ValueError:
                    pass
                gold_programs.add(tuple(gold))

            indices = indices.eval(session=sess)
            print(indices.shape)
            
            for test_i, train_i in enumerate(indices):
                gold = list(test_data[2][test_i])
                decoded = list(train_data[2][train_i])
                try:
                    decoded = decoded[:decoded.index(grammar.end)]
                except ValueError:
                    pass
                decoded_tuple = tuple(decoded)
                
                try:
                    gold = gold[:gold.index(grammar.end)]
                except ValueError:
                    pass
                
                #print "GOLD:", ' '.join(grammar.tokens[l] for l in gold)
                #print "DECODED:", ' '.join(grammar.tokens[l] for l in decoded)

                if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]:
                    ok_0 += 1

                def get_functions(seq):
                    return set([x for x in [grammar.tokens[x] for x in seq] if x.startswith('tt:') and not x.startswith('tt:param.')])
                gold_functions = get_functions(gold)
                decoded_functions = get_functions(decoded)
                gold_channels = set([x[x.index('.')+1:] for x in gold_functions])
                decoded_channels = set([x[x.index('.')+1:] for x in decoded_functions])
                if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0] and gold_functions == decoded_functions:
                    ok_fn += 1
                if gold_channels == decoded_channels:
                    ok_ch += 1
                if grammar.compare(gold, decoded):
                    correct_programs.add(decoded_tuple)
                    ok_full += 1
        
        print("ok 0:", float(ok_0)/len(test_data[0]))
        print("ok channel:", float(ok_ch)/len(test_data[0]))
        print("ok function:", float(ok_fn)/len(test_data[0]))
        print("ok full:", float(ok_full)/len(test_data[0]))
        print("recall:", float(len(correct_programs))/len(gold_programs))



if __name__ == "__main__":
    run()
