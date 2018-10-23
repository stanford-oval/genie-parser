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
from luinet.scripts.utils.loader import load_dictionary, load_embeddings, load_data, unknown_tokens
from luinet.grammar.thingtalk import ThingTalkGrammar



def run(args):
    cached_grammar = args.cached_grammar

    if args.load_grammar:
        with tf.gfile.Open(cached_grammar, 'rb') as fr:
            grammar = pickle.load(fr)
        words, reverse = load_dictionary(args.input_vocab, 'tt', grammar)
    else:
        grammar = ThingTalkGrammar(args.thingpedia_snapshot, flatten=False)
        words, reverse = load_dictionary(args.input_vocab, 'tt', grammar)
        grammar.set_input_dictionary(words)
        with tf.gfile.Open(cached_grammar, 'wb') as fw:
            pickle.dump(grammar, fw, pickle.HIGHEST_PROTOCOL)

    print("%d words in dictionary" % (len(words),))
    glove = os.getenv('GLOVE', args.word_embedding)
    if args.load_embeddings:
        with tf.gfile.Open(args.cached_embeddings, "rb") as fr:
            embeddings_matrix = np.load(fr)
    else:
        embeddings_matrix, embed_size = load_embeddings(glove, words, embed_size=300)
        with tf.gfile.Open(args.cached_embeddings, "wb") as fw:
            np.save(fw, embeddings_matrix)
    max_length = 60

    train_data = load_data(args.train_set, words, grammar, max_length)
    test_data = load_data(args.test_set, words, grammar, max_length)
    N = test_data[1].shape[0]
    batch_size = args.batch_size
    n_batches = math.ceil(N // batch_size)

    with tf.Graph().as_default():
        # define placeholders
        test_data_placeholder_sentences = tf.placeholder(dtype=tf.int32, shape=[None, max_length])
        test_data_placeholder_programs = tf.placeholder(dtype=tf.int32, shape=[None, max_length, 3])

        # Dataset
        dataset = tf.data.Dataset.from_tensor_slices((test_data_placeholder_sentences, test_data_placeholder_programs))
        batched_dataset = dataset.batch(batch_size)

        iterator = batched_dataset.make_initializable_iterator()
        sentences, programs = iterator.get_next()

        input_embed_matrix = tf.constant(embeddings_matrix)

        train_inputs = tf.nn.embedding_lookup([input_embed_matrix], np.array(train_data[1]))
        train_encoded = tf.reduce_sum(train_inputs, axis=1)
        train_norm = tf.sqrt(tf.reduce_sum(train_encoded * train_encoded, axis=1))

        test_inputs = tf.nn.embedding_lookup([input_embed_matrix], sentences)
        test_encoded = tf.reduce_sum(test_inputs, axis=1)
        test_norm = tf.sqrt(tf.reduce_sum(test_encoded * test_encoded, axis=1))

        distances = tf.matmul(test_encoded, tf.transpose(train_encoded))
        distances /= tf.reshape(train_norm, (1, -1))
        distances /= tf.reshape(test_norm, (-1, 1))

        indices = tf.argmax(distances, axis=1)

        gold = programs
        gold = tf.reshape(gold, [tf.shape(gold)[0], -1])
        gold = tf.expand_dims(tf.expand_dims(gold, axis=-1), axis=-1)

        decoded = tf.stack(list(train_data[5].values()), axis=-1)
        decoded = tf.gather(decoded, indices, axis=0)
        decoded = tf.reshape(decoded, [tf.shape(decoded)[0], -1])

        metrics = grammar.eval_metrics()
        eval_metrics = {}

        for metric_key, metric_fn in metrics.items():
            metric_name = "metrics-{}".format(metric_key)
            first, second = metric_fn(decoded, gold, None)

            scores, weights = first, second
            eval_metrics[metric_name] = tf.metrics.mean(scores, weights, name=metric_name)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={test_data_placeholder_sentences: np.array(test_data[1]),
                                                      test_data_placeholder_programs: np.stack(list(test_data[5].values()), axis=-1)})
            sess.run(tf.local_variables_initializer())

            for i in range(n_batches):
                sess.run([metric_val[1] for metric_key, metric_val in eval_metrics.items()])

            metric_val = sess.run([metric_val[0] for metric_key, metric_val in eval_metrics.items()])
            for j, (metric_key, _) in enumerate(eval_metrics.items()):
                print(metric_key, ":", metric_val[j])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vocab', default='./workdir/t2t_copy_data_generated/input_words.txt', type=str)
    parser.add_argument('--word_embedding', default='./workdir/t2t_copy_data_generated/glove.42B.300d.txt', type=str)
    parser.add_argument('--thingpedia_snapshot', default='./workdir/t2t_copy_data_generated/thingpedia.json', type=str)
    parser.add_argument('--train_set', default='./dataset/t2t_copy_data/train.tsv', type=str)
    parser.add_argument('--test_set', default='./dataset/t2t_copy_data/test.tsv', type=str)
    parser.add_argument('--load_grammar', default=False, type=bool)
    parser.add_argument('--cached_grammar', default='./workdir/cached_grammar.pkl', type=str)
    parser.add_argument('--load_embeddings', default=False, type=bool)
    parser.add_argument('--cached_embeddings', default='./workdir/input_embeddings.npy', type=str)
    parser.add_argument('--batch_size', default=100, type=int)

    args = parser.parse_args()

    run(args)
