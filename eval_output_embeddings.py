#!/usr/bin/python3
'''
Created on Mar 29, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf
import csv

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

from models import Config, create_model
from util.loader import unknown_tokens, load_data
from util.general_utils import get_minibatches

def show_pca(X, programs):
    plt.figure()
    plt.plot(X[:,0], X[:,1], 'x')
    
    for x, program in zip(X, programs):
        plt.text(x[0]-0.01, x[1]-0.01, program, horizontalalignment='center', verticalalignment='top')
    
    plt.show()

def reconstruct_programs(inputs, input_lengths, reverse):
    programs = [None]*len(inputs)
    
    for i, input in enumerate(inputs):
        input = list(input)
        try:
            input = input[:input_lengths[i]]
        except ValueError:
            pass
        program = [reverse[x] for x in input if reverse[x].startswith('tt:')]
        programs[i] = ' '.join(program)
        #sentences[i] = ' '.join([reverse[x] for x in input])
        
        #if len(sentences[i]) > 50:
        #    sentences[i] = sentences[i][:50] + '...'
    
    return programs

def bag_of_tokens(config, labels, label_lengths):
    if config.train_output_embeddings:
        with tf.variable_scope('embed', reuse=True):
            output_embeddings = tf.get_variable('output_embedding')
    else:
        output_embeddings = tf.constant(config.output_embedding_matrix)

    #everything_label_placeholder = tf.placeholder(shape=(None, config.max_length,), dtype=tf.int32)
    #everything_label_length_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

    labels = tf.constant(np.array(labels))
    embedded_output = tf.gather(output_embeddings, labels)
    print('embedded_output before', embedded_output)
    #mask = tf.sequence_mask(label_lengths, maxlen=config.max_length, dtype=tf.float32)
    # note: this multiplication will broadcast the mask along all elements of the depth dimension
    # (which is why we run the expand_dims to choose how to broadcast)
    #embedded_output = embedded_output * tf.expand_dims(mask, axis=2)
    #print('embedded_output after', embedded_output)

    return tf.reduce_sum(embedded_output, axis=1)

def pca_fit(X, n_components):
    mean = tf.reduce_mean(X, axis=0)
    centered_X = X - mean
    S, U, V = tf.svd(centered_X)
                
    return V[:n_components], mean

def pca_transform(X, V, mean):
    centered_X = X - mean
    return tf.matmul(centered_X, V, transpose_b=True)

def load_one_program(config, prog):
    if True:
        return [config.grammar.dictionary[x] for x in prog if x.startswith('tt:')], 3
    else:
        return config.grammar.vectorize_program(program, config.max_length)

def load_programs(config, filename):
    with open(filename) as fp:
        return zip(*(load_one_program(config, program) for program in map(lambda x: x.strip().split(' '), fp) if program[0] == 'rule'))

def sort_uniq(*args):
    pairs = list(zip(*args))
    pairs.sort()
    def uniq():
        prev = None
        for x in pairs:
            if prev == None or x != prev:
                yield x
                prev = x
    return zip(*uniq())

def sample(grammar, data, length, N=10):
    data, length = sort_uniq(data, length)

    # pick 2 triggers, 2 queries, 2 actions
    # then pick everything with them
    function_offset = grammar.num_control_tokens + grammar.num_begin_tokens
    num_triggers = len(grammar.functions['trigger'])
    num_queries = len(grammar.functions['query'])
    num_actions = len(grammar.functions['action'])
    triggers = np.random.choice(num_triggers, size=2, replace=False) + function_offset
    queries = np.random.choice(num_queries, size=2, replace=False) + function_offset + num_triggers
    actions = np.random.choice(num_actions, size=2, replace=False) + function_offset + num_triggers + num_actions

    indices = [i for i in range(len(data)) if data[i][0] in triggers or data[i][1] in queries or data[i][2] in actions]
    data = [data[i] for i in indices]
    length = [length[i] for i in indices]

    if len(data) > N:
        indices = np.random.choice(len(data), size=N, replace=False)
        return [data[i] for i in indices], [length[i] for i in indices]
    else:
        return data, length

def run():
    if len(sys.argv) < 4:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Everything Set>> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)

    everything_labels, everything_label_lengths = load_programs(config, sys.argv[2])
    test_labels, test_label_lengths = load_programs(config, sys.argv[3])
    #test_labels, test_label_lengths = sample(config.grammar, test_labels, test_label_lengths)
    print("unknown", unknown_tokens)

    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        model.build()
        loader = tf.train.Saver()

        train_bag_of_tokens = bag_of_tokens(config, everything_labels, everything_label_lengths)
        V, mean = pca_fit(train_bag_of_tokens, n_components=2)

        eval_bag_of_tokens = bag_of_tokens(config, test_labels, test_label_lengths)
        transformed = pca_transform(eval_bag_of_tokens, V, mean)

        with tf.Session() as sess:
            loader.restore(sess, os.path.join(model_dir, 'best'))
            transformed = transformed.eval(session=sess)
        
        programs = reconstruct_programs(test_labels, test_label_lengths, config.grammar.tokens)
        show_pca(transformed, programs)

if __name__ == '__main__':
    run()
