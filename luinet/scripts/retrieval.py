#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
#         Giovanni Campagna <gcampagn@cs.stanford.edu>
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
Created on Oct 18, 2018

@author: mehrad, gcampagn
'''
import os
import sys

# workaround to import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import argparse
import pickle
import math
import tensorflow as tf
from luinet.scripts.utils.loader import load_dictionary, load_embeddings, load_data
from luinet.grammar.thingtalk import ThingTalkGrammar



def run(args):
    max_length = 60

    if 'noquote' in args.problem:
        flatten = False
    else:
        flatten = True
    cached_grammar = args.cached_grammar
    load_grammar = args.load_grammar

    cached_embedding = args.cached_embeddings
    load_embedding = args.load_embeddings
    if load_grammar:
        if not os.path.exists(cached_grammar):
            print('** Grammar file doesn\'t exist **')
            load_grammar = False
        else:
            print('Loading grammar from file...')
            with open(cached_grammar, 'rb') as fr:
                grammar = pickle.load(fr)
            words, reverse = load_dictionary(args.input_vocab, 'tt', grammar)

    if not load_grammar:
        print('Building grammar file...')
        grammar = ThingTalkGrammar(args.thingpedia_snapshot, flatten=flatten)
        words, reverse = load_dictionary(args.input_vocab, 'tt', grammar)
        grammar.set_input_dictionary(words)
        with open(cached_grammar, 'wb') as fw:
            pickle.dump(grammar, fw, pickle.HIGHEST_PROTOCOL)

    print("%d words in dictionary" % (len(words),))
    glove = os.getenv('GLOVE', args.word_embedding)
    if load_embedding:
        if not os.path.exists(cached_embedding):
            print('** Embedding file doesn\'t exist **')
            load_embedding = False
        else:
            print('Loading embeddings from file...')
            with open(cached_embedding, "rb") as fr:
                embeddings_matrix = np.load(fr)
    if not load_embedding:
        print('Building embedding matrix...')
        embeddings_matrix, embed_size = load_embeddings(glove, words, embed_size=300)
        with open(cached_embedding, "wb") as fw:
            np.save(fw, embeddings_matrix)


    train_data = load_data(args.train_set, words, grammar, max_length)
    test_data = load_data(args.test_set, words, grammar, max_length)
    N_train = train_data[1].shape[0]
    N_test = test_data[1].shape[0]
    train_batch_size = args.train_batch_size
    train_n_batches = math.ceil(N_train / train_batch_size)

    with tf.Graph().as_default():
        # define placeholders
        train_data_placeholder_sentences = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='train_s')
        train_data_placeholder_programs = tf.placeholder(dtype=tf.int32, shape=[None, max_length, None], name='train_p')
        test_data_placeholder_sentences = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='test_s')
        test_data_placeholder_programs = tf.placeholder(dtype=tf.int32, shape=[None, max_length, None], name='test_p')

        # Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data_placeholder_sentences, train_data_placeholder_programs))
        train_batched_dataset = train_dataset.batch(train_batch_size)
        train_iterator = train_batched_dataset.make_initializable_iterator()
        train_sentences, train_programs = train_iterator.get_next()

        input_embed_matrix = tf.constant(embeddings_matrix)

        train_inputs = tf.nn.embedding_lookup([input_embed_matrix], train_sentences)
        train_encoded = tf.reduce_sum(train_inputs, axis=1)
        train_norm = tf.sqrt(tf.reduce_sum(train_encoded * train_encoded, axis=1))

        test_inputs = tf.nn.embedding_lookup([input_embed_matrix], test_data_placeholder_sentences)
        test_encoded = tf.reduce_sum(test_inputs, axis=1)
        test_norm = tf.sqrt(tf.reduce_sum(test_encoded * test_encoded, axis=1))

        distances = tf.matmul(test_encoded, tf.transpose(train_encoded))
        distances /= tf.reshape(train_norm, (1, -1))
        distances /= tf.reshape(test_norm, (-1, 1))

        indices = tf.argmax(distances, axis=1)

        gold = test_data_placeholder_programs
        gold = tf.reshape(gold, [tf.shape(gold)[0], -1])
        gold = tf.expand_dims(tf.expand_dims(gold, axis=-1), axis=-1)

        decoded = train_programs
        decoded = tf.gather(decoded, indices, axis=0)
        decoded = tf.reshape(decoded, [tf.shape(decoded)[0], -1])

        metrics = grammar.eval_metrics()
        eval_metrics = {}

        for metric_key, metric_fn in metrics.items():
            metric_name = "metrics-{}".format(metric_key)
            first, second = metric_fn(decoded, gold, None)
            scores, weights = first, second
            eval_metrics[metric_name] = tf.contrib.metrics.streaming_concat(tf.reshape(scores * weights, [N_test, 1]), axis=1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([train_iterator.initializer], feed_dict={train_data_placeholder_sentences: np.array(train_data[1]),
                                                             train_data_placeholder_programs: np.stack(list(train_data[4].values()), axis=-1),
                                                             test_data_placeholder_sentences: np.array(test_data[1]),
                                                             test_data_placeholder_programs: np.stack(list(test_data[4].values()), axis=-1)})
            sess.run(tf.local_variables_initializer())

            for i in range(1, train_n_batches+1):
                sess.run([metric_val[1] for metric_key, metric_val in eval_metrics.items()], feed_dict={test_data_placeholder_sentences: np.array(test_data[1]),
                                                             test_data_placeholder_programs: np.stack(list(test_data[4].values()), axis=-1)})
                print("iteration- {} / {}".format(i, train_n_batches))
                if not i%20:
                    metric_val = sess.run([metric_val[0] for metric_key, metric_val in eval_metrics.items()])
                    for k, (metric_key, _) in enumerate(eval_metrics.items()):
                        print("value of " + metric_key + " for iteration-{}".format(i), ":", np.mean(np.max(metric_val[k], axis=1)))

            metric_val = sess.run([metric_val[0] for metric_key, metric_val in eval_metrics.items()])
            for k, (metric_key, _) in enumerate(eval_metrics.items()):
                print(metric_key, ":", np.mean(np.max(metric_val[k], axis=1)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vocab', type=str)
    parser.add_argument('--word_embedding', type=str)
    parser.add_argument('--thingpedia_snapshot', type=str)
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--test_set', type=str)
    parser.add_argument('--load_grammar', default=False, type=bool)
    parser.add_argument('--cached_grammar', type=str)
    parser.add_argument('--load_embeddings', default=False, type=bool)
    parser.add_argument('--cached_embeddings', type=str)
    parser.add_argument('--train_batch_size', default=100, type=int)
    parser.add_argument('--problem', default='semparse_thingtalk_noquote', type=str)

    args = parser.parse_args()

    run(args)
