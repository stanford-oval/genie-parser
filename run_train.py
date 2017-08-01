#!/usr/bin/python3

import os
import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from util.trainer import Trainer

from models import Config, create_model
from util.loader import unknown_tokens, load_data

from tensorflow.python import debug as tf_debug

def run():
    if len(sys.argv) < 3:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Train Set>> [<<Dev Set>>]")
        sys.exit(1)

    np.random.seed(42)
    
    model_dir = sys.argv[1]
    model_conf = os.path.join(model_dir, 'model.conf')
    config = Config.load(['./default.conf', model_conf])
    model = create_model(config)
    train_data = load_data(sys.argv[2], config.dictionary, config.grammar.dictionary, config.max_length)
    if len(sys.argv) > 3:
        dev_data = load_data(sys.argv[3], config.dictionary, config.grammar.dictionary, config.max_length)
    else:
        dev_data = None
    print("unknown", unknown_tokens)
    try:
        os.mkdir(model_dir)
    except OSError:
        pass
    if not os.path.exists(model_conf):
        config.save(model_conf)

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
                          output_dropout=config.output_dropout,
                          state_dropout=config.state_dropout)

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # Fit the model
            best_dev, best_train = trainer.fit(sess)

            print("best train", best_train)
            print("best dev", best_dev)

if __name__ == "__main__":
    run()
