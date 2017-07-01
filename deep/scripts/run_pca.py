#!/usr/bin/python2
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

from util.seq2seq import Seq2SeqEvaluator
from util.loader import unknown_tokens, load_data
from util.general_utils import get_minibatches
from model import initialize

def show_pca(X, sentences):
    plt.figure()
    plt.plot(X[:,0], X[:,1], 'x')
    
    for x, sentence in zip(X, sentences):
        plt.text(x[0]+0.01, x[1]-0.01, sentence, horizontalalignment='left', verticalalignment='top')
    
    plt.show()

def reconstruct_sentences(inputs, end_of_string, reverse):
    sentences = [None]*len(inputs)
    
    for i, input in enumerate(inputs):
        input = list(input)
        try:
            input = input[:input.index(end_of_string)]
        except ValueError:
            pass
        sentences[i] = ' '.join(map(lambda x: reverse[x], input))
        
        if len(sentences[i]) > 50:
            sentences[i] = sentences[i][:50] + '...'
    
    return sentences

def run():
    if len(sys.argv) < 6:
        print "** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Model: bagofwords/seq2seq>> <<Input Vocab>> <<Word Embeddings>> <<Model Directory>> <<Train Set>> <<PCA Set>>"
        sys.exit(1)

    np.random.seed(42)
    benchmark = sys.argv[1]
    config, words, reverse, model = initialize(benchmark=benchmark, model_type=sys.argv[2], input_words=sys.argv[3], embedding_file=sys.argv[4]);
    model_dir = sys.argv[5]

    train_data = load_data(sys.argv[6], words, config.grammar.dictionary,
                           reverse, config.grammar.tokens,
                           config.max_length)
    pca_data = load_data(sys.argv[7], words, config.grammar.dictionary,
                         reverse, config.grammar.tokens,
                         config.max_length)
    config.apply_cmdline(sys.argv[8:])
    
    print "unknown", unknown_tokens

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model.capture_final_encoder_state = True
        model.build()
        loader = tf.train.Saver()

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            loader.restore(sess, os.path.join(model_dir, 'best'))
                
            inputs, input_lengths, _, _ = train_data
            
            final_encoder_state = None
            final_encoder_size = None
            if config.rnn_cell_type == 'lstm':
                final_encoder_state = tf.concat([model.final_encoder_state[-1].c, model.final_encoder_state[-1].h], 1)
                final_encoder_size = 2 * config.hidden_size
            else:
                final_encoder_state = model.final_encoder_state[-1]
                final_encoder_size = config.hidden_size
            
            final_states_arrays = []
            # capture all the final encoder states
            for input_batch, input_length_batch in get_minibatches([inputs, input_lengths],
                                                                   config.batch_size):
                feed_dict = model.create_feed_dict(input_batch, input_length_batch)
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
                
            inputs, input_lengths, _, _ = pca_data
            
            X = final_encoder_state
            centered_X = X - tf.constant(mean_array)
            transformed_X = tf.matmul(centered_X, tf.constant(V_array.T))
            
            feed_dict = model.create_feed_dict(inputs, input_lengths)
            X_pca = sess.run(transformed_X, feed_dict=feed_dict)
            
            sentences = reconstruct_sentences(inputs, words['<<EOS>>'], reverse)
            show_pca(X_pca, sentences)
            
if __name__ == '__main__':
    run()
