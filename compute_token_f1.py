#!/usr/bin/python3

import os
import sys
import numpy as np
import itertools
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from models import Config, create_model
from util.loader import load_data, unknown_tokens

def run():
    if len(sys.argv) < 3:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Test Data>>")
        sys.exit(1)

    np.random.seed(42)
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)
    test_data = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
    print("unknown", unknown_tokens)
    
    config.grammar.print_all_actions()
    output_size = config.grammar.output_size

    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        with tf.device('/cpu:0'):
            model.build()
            
            test_eval = Seq2SeqEvaluator(model, config.grammar, test_data, 'test', config.reverse_dictionary,
                                         beam_size=config.beam_size,
                                         batch_size=64)
            loader = tf.train.Saver()

            with tf.Session() as sess:
                loader.restore(sess, os.path.join(model_dir, 'best'))

                # predicted vs gold
                confusion_matrix = test_eval.compute_confusion_matrix(sess)

    # precision: sum over columns (% of the sentences where this token was predicted
    # in which it was actually meant to be there)
    # recall: sum over rows (% of the sentences where this token was meant
    # to be there in which it was actually predicted)
    #
    # see "A systematic analysis of performance measures for classification tasks"
    # MarinaSokolova, GuyLapalme, Information Processing & Management, 2009
    confusion_matrix = np.ma.asarray(confusion_matrix)
    
    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    f1 = 2 * (precision * recall) / (precision + recall)

    with open('./f1-parse-actions.tsv', 'w') as out:
        for i in range(output_size):
            print(i, precision[i], recall[i], f1[i], sep='\t', file=out)
    
    precision = np.ma.masked_invalid(precision)
    recall = np.ma.masked_invalid(recall)
    
    overall_precision = np.power(np.prod(precision, dtype=np.float64), 1/len(precision))
    overall_recall = np.power(np.prod(recall, dtype=np.float64), 1/len(recall))
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    print(overall_precision, overall_recall, overall_f1)

if __name__ == '__main__':
    run()
