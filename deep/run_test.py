'''
Created on Mar 16, 2017

@author: gcampagn
'''

import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from util.loader import unknown_tokens, load_data
from model import LSTMAligner, initialize

def run():
    if len(sys.argv) < 5:
        print "** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Input Vocab>> <<Word Embeddings>> <<Test Set>>"
        sys.exit(1)

    np.random.seed(42)
    config, words, reverse, embeddings_matrix = initialize(benchmark=sys.argv[1], input_words=sys.argv[2], embedding_file=sys.argv[3]);
    
    test_data = load_data(sys.argv[4], words, config.grammar.dictionary,
                          reverse, config.grammar.tokens,
                          config.max_length)
    print "unknown", unknown_tokens
    
    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = LSTMAligner(config, embeddings_matrix)
        init = tf.global_variables_initializer()
        
        test_eval = Seq2SeqEvaluator(model, config.grammar, test_data, 'test', batch_size=config.batch_size)

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            sess.run(init)

            test_eval.eval(sess, save_to_file=True)
            
if __name__ == '__main__':
    run()