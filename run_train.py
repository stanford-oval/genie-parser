#!/usr/bin/python3
#
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
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Train Set>> <<Dev Set>>")
        sys.exit(1)

    np.random.seed(42)
    
    model_dir = sys.argv[1]
    model_conf = os.path.join(model_dir, 'model.conf')
    config = Config.load(['./default.conf', model_conf])
    model = create_model(config)
    train_data = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
    dev_data = load_data(sys.argv[3], config.dictionary, config.grammar, config.max_length)
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
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver(max_to_keep=config.n_epochs)
        
        train_eval = Seq2SeqEvaluator(model, config.grammar, train_data, 'train', config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size)
        dev_eval = Seq2SeqEvaluator(model, config.grammar, dev_data, 'dev', config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size)
        trainer = Trainer(model, train_data, train_eval, dev_eval, saver,
                          opt_eval_metric='parse_action_f1',
                          model_dir=model_dir,
                          max_length=config.max_length,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          dropout=config.dropout)

        tfconfig = tf.ConfigProto()
        tfconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=tfconfig) as sess:
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
