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
import tensorflow as tf

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

from tensorflow.python.util import nest

from models import Config, create_model
from util.loader import unknown_tokens, load_data
from util.general_utils import get_minibatches

def show_pca(X, sentences):
    plt.figure()
    plt.plot(X[:,0], X[:,1], 'x')
    
    for x, sentence in zip(X, sentences):
        plt.text(x[0]-0.01, x[1]-0.01, sentence, horizontalalignment='center', verticalalignment='top')
    
    plt.show()

def reconstruct_sentences(inputs, input_lengths, reverse):
    sentences = [None]*len(inputs)
    
    for i, input in enumerate(inputs):
        input = list(input)
        try:
            input = input[:input_lengths[i]]
        except ValueError:
            pass
        sentence = [reverse[x] for x in input if reverse[x].startswith('tt:')]
        sentences[i] = ' '.join(sentence)
        #sentences[i] = ' '.join([reverse[x] for x in input])
        
        #if len(sentences[i]) > 50:
        #    sentences[i] = sentences[i][:50] + '...'
    
    return sentences

def run():
    if len(sys.argv) < 4:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Train Set>> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)
    train_data = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
    pca_data = load_data(sys.argv[3], config.dictionary, config.grammar, config.max_length)
    print("unknown", unknown_tokens)

    with tf.Graph().as_default():
        model.build()
        loader = tf.train.Saver()

        with tf.Session() as sess:
            loader.restore(sess, os.path.join(model_dir, 'best'))
                
            inputs, input_lengths, parses, _, _ = train_data
            
            final_encoder_state = tf.concat(nest.flatten(model.final_encoder_state), axis=1)
            final_encoder_size = final_encoder_state.get_shape()[1]
            
            final_states_arrays = []
            # capture all the final encoder states
            for input_batch, input_length_batch, parse_batch in get_minibatches([inputs, input_lengths, parses],
                                                                                config.batch_size):
                feed_dict = model.create_feed_dict(input_batch, input_length_batch, parse_batch)
                state_array = sess.run(final_encoder_state, feed_dict=feed_dict)
                #print state_array.shape
                final_states_arrays.append(state_array)

            X = np.concatenate(final_states_arrays, axis=0)
            assert X.shape == (len(inputs), final_encoder_size)
            X = tf.constant(X)

            mean = tf.reduce_mean(X, axis=0)
            centered_X = X - mean
            S, U, V = tf.svd(centered_X)
                
            # take only the top 2 components
            V = V[:2]
            V_array, mean_array = sess.run([V, mean])
                
            inputs, input_lengths, parses, labels, label_lengths = pca_data
            
            X = final_encoder_state
            centered_X = X - tf.constant(mean_array)
            transformed_X = tf.matmul(centered_X, tf.constant(V_array.T))
            
            feed_dict = model.create_feed_dict(inputs, input_lengths, parses)
            X_pca = sess.run(transformed_X, feed_dict=feed_dict)
            
            if False:
                sentences = reconstruct_sentences(inputs, input_lengths, config.reverse_dictionary)
            else:
                sentences = reconstruct_sentences(labels, label_lengths, config.grammar.tokens)
            show_pca(X_pca, sentences)
            
if __name__ == '__main__':
    run()
