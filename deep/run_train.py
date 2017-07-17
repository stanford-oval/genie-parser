#/!usr/bin/python3

import os
import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from util.trainer import Trainer

from util.loader import unknown_tokens, load_data
from model import initialize

def run():
    if len(sys.argv) < 6:
        print("** Usage: python3 " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Model: bagofwords/seq2seq>> <<Input Vocab>>" \
                                                  " <<Word Embeddings>> <<Model Directory>> <<Train Set> [<<Dev Set>>]")
        sys.exit(1)

    np.random.seed(42)
    benchmark = sys.argv[1]
    config, words, reverse, model = initialize(benchmark=benchmark, model_type=sys.argv[2], input_words=sys.argv[3], embedding_file=sys.argv[4]);
    model_dir = sys.argv[5]
    train_data = load_data(sys.argv[6], words, config.grammar.dictionary,
                           reverse, config.grammar.tokens,
                           config.max_length)
    if len(sys.argv) > 7:
        dev_data = load_data(sys.argv[7], words, config.grammar.dictionary,
                             reverse, config.grammar.tokens,
                             config.max_length)
    else:
        dev_data = None
    if len(sys.argv) > 8:
        config.apply_cmdline(sys.argv[8:])
    print("unknown", unknown_tokens)
    try:
        os.mkdir(model_dir)
    except OSError:
        pass

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model.build()
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver(max_to_keep=config.n_epochs)
        
        train_eval = Seq2SeqEvaluator(model, config.grammar, train_data, 'train', beam_size=config.beam_size, batch_size=config.batch_size)
        dev_eval = Seq2SeqEvaluator(model, config.grammar, dev_data, 'dev', beam_size=config.beam_size, batch_size=config.batch_size)
        trainer = Trainer(model, train_data, train_eval, dev_eval, saver,
                          model_dir=model_dir,
                          max_length=config.max_length,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          dropout=config.dropout)

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            
            #for var in tf.trainable_variables():
            #    print var.name
            #    print var.get_shape()
            #sys.exit(0)
            
            # Fit the model
            best_dev, best_train = trainer.fit(sess)
            
            #print "Final result"
            #train_eval.eval(sess, save_to_file=True)
            #if dev_data is not None:
            #    dev_eval.eval(sess, save_to_file=True)
            print("best train", best_train)
            print("best dev", best_dev)

if __name__ == "__main__":
    run()
