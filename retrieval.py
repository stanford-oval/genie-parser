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
Created on Apr 12, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf

from util.eval import Seq2SeqEvaluator
from util.trainer import Trainer

from models import Config
from models.base_model import BaseModel
from util.loader import unknown_tokens, load_data

class RetrievalModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='input_sequence')
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='input_length')
        
    def build(self, train_data):
        self.add_placeholders()
        
        input_embed_matrix = tf.constant(self.config.input_embedding_matrix)
        train_inputs = tf.nn.embedding_lookup([input_embed_matrix], train_data[0])
        train_encoded = tf.reduce_sum(train_inputs, axis=1)
        #print train_encoded.eval()
        train_norm = tf.sqrt(tf.reduce_sum(train_encoded * train_encoded, axis=1))
        
        test_inputs = tf.nn.embedding_lookup([input_embed_matrix], self.input_placeholder)
        test_encoded = tf.reduce_sum(test_inputs, axis=1)
        test_norm = tf.sqrt(tf.reduce_sum(test_encoded * test_encoded, axis=1))
        
        distances = tf.matmul(test_encoded, tf.transpose(train_encoded))
        distances /= tf.reshape(train_norm, (1, -1))
        distances /= tf.reshape(test_norm, (-1, 1))
        indices = tf.argmax(distances, axis=1)
        
        self.pred = tf.gather(train_data[4][self.config.grammar.primary_output], indices)
        print(self.pred)
        # add a dimension of one
        self.preds = {
            self.config.grammar.primary_output: tf.expand_dims(self.pred, axis=1)
        }
        self.eval_loss = tf.reduce_sum(tf.reduce_max(distances, axis=1), axis=0)
        
    def create_feed_dict(self, inputs_batch, *args, **kw):
        return { self.input_placeholder: inputs_batch }


def run():
    if len(sys.argv) < 4:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Dir>> <<Train Set> <<Test Set>>")
        sys.exit(1)

    np.random.seed(42)

    model_dir = sys.argv[1]
    model_conf = os.path.join(model_dir, 'model.conf')
    config = Config.load(['./default.conf', model_conf])
    model = RetrievalModel(config)
    
    train_data = load_data(sys.argv[2], config.dictionary, config.grammar, config.max_length)
    test_data = dict()
    for filename in sys.argv[3:]:
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
            model.build(train_data)
        
            test_evals = dict()
            for key, data in test_data.items():
                test_evals[key] = Seq2SeqEvaluator(model, config.grammar, data, key, config.reverse_dictionary, beam_size=config.beam_size, batch_size=config.batch_size)

            with tf.Session() as sess:
                for test_eval in test_evals.values():
                    print()
                    test_eval.eval(sess, save_to_file=True)

if __name__ == "__main__":
    run()
