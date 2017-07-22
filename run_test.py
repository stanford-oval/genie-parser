#/!usr/bin/python3
'''
Created on Mar 16, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from models import Config, create_model
from util.loader import unknown_tokens, load_data

def run():
    if len(sys.argv) < 3:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)
    test_data = load_data(sys.argv[2], config.dictionary, config.grammar.dictionary, config.max_length)
    print("unknown", unknown_tokens)

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # Build the model and add the variable initializer Op
            model.build()
        
            test_eval = Seq2SeqEvaluator(model, config.grammar, test_data, 'test', beam_size=config.beam_size, batch_size=config.batch_size)
            loader = tf.train.Saver()

            # Create a session for running Ops in the Graph
            with tf.Session() as sess:
                loader.restore(sess, os.path.join(model_dir, 'best'))
                test_eval.eval(sess, save_to_file=True)
            
if __name__ == '__main__':
    run()
