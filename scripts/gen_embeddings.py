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
Takes a pre-trained matrix of embeddings, a train file and a vocabulary, does
one pass of training and then writes out the trained embeddings

Created on Jul 17, 2017

@author: gcampagn
'''

import tensorflow as tf
import numpy as np

import sys
import json
from math import ceil

from util.loader import load_dictionary, vectorize, load_embeddings, ENTITIES, MAX_ARG_VALUES
from util.general_utils import get_minibatches

def make_skipgram_softmax_loss(embeddings_matrix, vocabulary_size, vector_size):
    vectors = tf.get_variable('vectors', (vocabulary_size, vector_size), dtype=tf.float32, initializer=tf.constant_initializer(embeddings_matrix))
    minibatch = tf.placeholder(shape=(None, 2), dtype=tf.int32)
    
    center_word_vector = tf.nn.embedding_lookup(vectors, minibatch[:,0])
    yhat = tf.matmul(center_word_vector, vectors, transpose_b=True)
    
    predict_word = minibatch[:,1]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=predict_word, logits=yhat)
    loss = tf.reduce_mean(loss)
    return vectors, minibatch, loss

N_EPOCHS = 3

def cosine_sim(v1, v2):
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    if m1*m2 < 1e-7:
        return 1
    return np.dot(v1, v2) / (m1*m2)

def run():
    if len(sys.argv) < 6:
        print("** Usage: python3 " + sys.argv[0] + " <<Input Vocab>> <<Word Embeddings>> <<Output File>> <<Train Set>> <<vector size>>")
        sys.exit(1)

    np.random.seed(42)
    vector_size = int(sys.argv[5])
    win_size = 2
    
    words, reverse = load_dictionary(sys.argv[1], 'tt')
    initial_embeddings = load_embeddings(sys.argv[2], words, embed_size=vector_size)
    
    inputs = []
    with open(sys.argv[4], 'r') as data:
        for line in data:
            sentence, _ = line.split('\t')
            input = list(map(lambda x: words[x], sentence.split(' '))) + [words['<<EOS>>']]
            
            for i, center_word in enumerate(input):
                for j in range(-win_size, win_size+1):
                    if i+j < 0 or i+j >= len(input):
                        continue
                    if j == 0:
                        continue
                    predict_word = input[i+j]
                    inputs.append((center_word, predict_word))

    losses = []
    with tf.Graph().as_default():
        vectors, minibatch_placeholder, loss = make_skipgram_softmax_loss(initial_embeddings, len(words), vector_size)
        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for _ in range(N_EPOCHS):
                n_batches = ceil(len(inputs)/8192)
                for i, minibatch in enumerate(get_minibatches(inputs, minibatch_size=8192)):
                    _, current_loss = sess.run((optimize, loss), feed_dict={ minibatch_placeholder: minibatch })
                    print("Batch %d/%d; Loss: %f (%s)" % (i+1, n_batches, current_loss, str(type(current_loss))))
                    losses.append(float(current_loss))
            final_vectors = vectors.eval(session=sess)
    
    with open('train-stats.json', 'w') as fp:
        json.dump(losses, fp)
    
    with open(sys.argv[3], 'w') as fp:
        for i, word in enumerate(reverse):
            if word == '<<PAD>>':
                continue
            print(word, *final_vectors[i], file=fp)

    sum_cosine_sim = 0
    for i in range(len(reverse)):
        sum_cosine_sim += cosine_sim(initial_embeddings[i], final_vectors[i])
    print('Avg cosine similarity:', sum_cosine_sim/len(reverse))
    sum_cosine_sim = 0
    for w in ENTITIES:
        for i in range(MAX_ARG_VALUES):
            entity_word = w + '_' + str(i)
            entity_word_id = words[entity_word]
            entity_cosine_sim = cosine_sim(initial_embeddings[entity_word_id], final_vectors[entity_word_id])
            print('Entity', entity_word, entity_cosine_sim)
            sum_cosine_sim += entity_cosine_sim
    print('Avg cosine similarity (entities):', sum_cosine_sim/(len(ENTITIES)*MAX_ARG_VALUES))

if __name__ == '__main__':
    run()
