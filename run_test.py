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
Created on Mar 16, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from util.eval import Seq2SeqEvaluator
from models import Config, create_model
from util.loader import unknown_tokens, load_data

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def run():
    if len(sys.argv) < 4:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)

    load_grammar = sys.argv.pop()
    
    model_dir = sys.argv[1]
    cached_grammar = os.path.join(model_dir, 'grammar.pkl')
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')], load_grammar=True, cached_grammar=cached_grammar)
    model = create_model(config)

    test_data = dict()
    for filename in sys.argv[2:]:
        print('Loading', filename)
        data = load_data(filename, config.dictionary, config.grammar, config.max_length)
        if len(data[0]) == 0:
            continue
        key = os.path.basename(filename)
        key = key[:key.rindex('.')]
        test_data[key] = data
    print("unknown", unknown_tokens)

    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        with tf.device('/cpu:0'):
            model.build()

            test_evals = dict()
            for key, data in test_data.items():
                test_evals[key] = Seq2SeqEvaluator(model, config.grammar, data, key, config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size)
            loader = tf.train.Saver()

            with tf.Session() as sess:
                loader.restore(sess, os.path.join(model_dir, 'best'))

                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

                for test_eval in test_evals.values():
                    print()
                    test_eval.eval(sess, save_to_file=True)
            
if __name__ == '__main__':
    run()
