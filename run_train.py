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

import os
import sys
import numpy as np
import tensorflow as tf

from util.eval import Seq2SeqEvaluator
from util.trainer import Trainer

from models import Config, create_model
from util.loader import unknown_tokens, load_data

from tensorflow.python import debug as tf_debug

def run():
    if len(sys.argv) < 4:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> [--continue] <<Train Set>> <<Dev Set>>")
        sys.exit(1)

    np.random.seed(42)
    
    model_dir = sys.argv[1]
    model_conf = os.path.join(model_dir, 'model.conf')
    config = Config.load(['./default.conf', model_conf])
    model = create_model(config)

    off = 2
    load_existing = False
    if sys.argv[2] == '--continue':
        load_existing = True
        off = 3

    print('Loading', sys.argv[off], 'as training')
    train_data = load_data(sys.argv[off], config.dictionary, config.grammar, config.max_length)
    dev_data = dict()
    for filename in sys.argv[off+1:]:
        print('Loading', filename, 'as dev')
        data = load_data(filename, config.dictionary, config.grammar, config.max_length)
        key = os.path.basename(filename)
        key = key[:key.rindex('.')]
        dev_data[key] = data
    print("unknown", unknown_tokens)
    try:
        os.mkdir(model_dir)
    except OSError:
        pass
    if not os.path.exists(model_conf):
        config.save(model_conf)

    np.save('train-weights.npy', train_data[-1])

    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        model.build()
        if not load_existing:
            init = tf.global_variables_initializer()
        else:
            init = None
        
        saver = tf.train.Saver(max_to_keep=config.n_epochs)
        
        train_eval = Seq2SeqEvaluator(model, config.grammar, train_data, 'train', config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size)
        dev_evals = []
        for key, data in dev_data.items():
            dev_evals.append(Seq2SeqEvaluator(model, config.grammar, data, key, config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size))
        trainer = Trainer(model, train_data, train_eval, dev_evals, saver,
                          opt_eval_metric='accuracy',
                          model_dir=model_dir,
                          max_length=config.max_length,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          shuffle_data=config.shuffle_training_data,
                          dropout=config.dropout)

        tfconfig = tf.ConfigProto()
        tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=tfconfig) as sess:
            # Run the Op to initialize the variables.
            if not load_existing:
                sess.run(init)
            else:
                saver.restore(sess, os.path.join(model_dir, 'best'))
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # Fit the model
            trainer.fit(sess)

if __name__ == "__main__":
    run()
