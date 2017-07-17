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

from util.loader import load_dictionary, vectorize, load_embeddings
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

N_EPOCHS = 2

def run():
    if len(sys.argv) < 5:
        print("** Usage: python3 " + sys.argv[0] + " <<Input Vocab>> <<Word Embeddings>> <<Train Set>> <<vector size>>")
        sys.exit(1)
        
    vector_size = int(sys.argv[4])
    win_size = 5
    
    words, reverse = load_dictionary(sys.argv[1], 'tt')
    embeddings_matrix = load_embeddings(sys.argv[2], words, embed_size=vector_size)
    
    inputs = []
    with open(sys.argv[3], 'r') as data:
        for line in data:
            sentence, _ = line.split('\t')
            input, _ = vectorize(sentence, words, max_length=60)
            
            for i, center_word in enumerate(input):
                for j in range(-win_size, win_size):
                    if i+j < 0 or i+j >= len(input):
                        continue
                    if j == 0:
                        continue
                    predict_word = input[i+j]
                    inputs.append((center_word, predict_word))

    losses = []
    with tf.Graph().as_default():
        vectors, minibatch_placeholder, loss = make_skipgram_softmax_loss(embeddings_matrix, len(words), vector_size)
        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for _ in range(N_EPOCHS):
                for i, minibatch in enumerate(get_minibatches(inputs, minibatch_size=256)):
                    _, current_loss = sess.run((optimize, loss), feed_dict={ minibatch_placeholder: minibatch })
                    #print("Batch %d/%d; Loss: %f" % (i+1, n_batches, current_loss))
                    losses.append(current_loss)
            final_vectors = vectors.eval(session=sess)
    
    with open('train-stats.json', 'w') as fp:
        json.dump(losses, fp)
    
    for i, word in enumerate(words):
        print(word, *final_vectors[i])

if __name__ == '__main__':
    run()