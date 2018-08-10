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
Created on Jul 20, 2017

@author: gcampagn
'''

import re
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

from util.loader import Dataset

from .base_model import BaseModel
from .encoders import RNNEncoder, BiRNNEncoder, BagOfWordsEncoder
from .tree_encoder import TreeEncoder

from .config import Config

class BaseAligner(BaseModel):
    '''
    The base class for encoder-decoder based models for semantic parsing.
    One such model is Seq2Seq. Another model is Beam Search with Beam Training.
    '''
    
    def build(self):
        self.add_placeholders()
        
        xavier = tf.contrib.layers.xavier_initializer()
        inputs = self.add_input_op(xavier)
        
        # the encoder
        with tf.variable_scope('encoder', initializer=xavier):
            enc_hidden_states, enc_final_state = self.add_encoder_op(inputs=inputs)
        self.final_encoder_state = enc_final_state
        
        # the training decoder
        with tf.name_scope('train_decoder'):
            with tf.variable_scope('decoder', initializer=xavier):
                train_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, training=True)

        with tf.name_scope('training_sequence_loss'):
            training_sequence_loss = self.add_loss_op(train_preds)
        
        with tf.name_scope('training_loss'):
            self.loss = training_sequence_loss + self.add_regularization_loss()
        self.train_op = self.add_training_op(self.loss)
        
        # the inference decoder
        with tf.name_scope('inference_decoder'):
            with tf.variable_scope('decoder', initializer=xavier, reuse=True):
                eval_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, training=False)
        self.raw_preds = eval_preds
        self.preds = self.finalize_predictions(eval_preds)
        if not isinstance(self.preds, dict):
            self.preds = {
                self.config.grammar.primary_output: self.preds
            }
        
        with tf.name_scope('eval_sequence_loss'):
            eval_sequence_loss = self.add_loss_op(eval_preds)
        with tf.name_scope('eval_loss'):
            self.eval_loss = eval_sequence_loss

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        size = 0
        def get_size(w):
            shape = w.get_shape()
            if shape.ndims == 2:
                return int(shape[0])*int(shape[1])
            else:
                assert shape.ndims == 1
                return int(shape[0])
        for w in weights:
            sz = get_size(w)
            print('weight', w, sz)
            size += sz
        print('total model size', size)

    def add_placeholders(self):
        # batch size x number of words in the sentence
        self.add_input_placeholders()
        self.add_output_placeholders()
        self.add_extra_placeholders()
        
    def add_input_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='input_sequence')
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='input_length')
        self.constituency_parse_placeholder = tf.placeholder(tf.bool, shape=(None, 2*self.config.max_length-1), name='input_constituency_parse')
        
    def add_output_placeholders(self):
        self.output_placeholders = dict()
        for key in self.config.grammar.output_size:
            self.output_placeholders[key] = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='output_sequence')
        self.primary_output_placeholder = self.output_placeholders[self.config.grammar.primary_output]
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='output_length')

    def add_extra_placeholders(self):
        self.batch_number_placeholder = tf.placeholder(tf.int32, shape=(), name='batch_number')
        self.epoch_placeholder = tf.placeholder(tf.int32, shape=(), name='epoch_number')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_probability')

    def create_feed_dict(self, data : Dataset, dropout=1, batch_number=0, epoch=0):
        feed_dict = dict()
        feed_dict[self.input_placeholder] = data.input_vectors
        feed_dict[self.input_length_placeholder] = data.input_lengths
        feed_dict[self.constituency_parse_placeholder] = data.constituency_parse
        if data.label_vectors is not None:
            for key, batch in data.label_vectors.items():
                feed_dict[self.output_placeholders[key]] = batch
            feed_dict[self.output_length_placeholder] = data.label_lengths
        feed_dict[self.dropout_placeholder] = dropout
        feed_dict[self.batch_number_placeholder] = batch_number
        feed_dict[self.epoch_placeholder] = epoch
        return feed_dict

    def add_encoder_op(self, inputs):
        if self.config.encoder_type == "rnn":
            encoder = RNNEncoder(cell_type=self.config.rnn_cell_type,
                                 output_size=self.config.encoder_hidden_size,
                                 dropout=self.dropout_placeholder,
                                 num_layers=self.config.rnn_layers)
        elif self.config.encoder_type == 'birnn':
            encoder = BiRNNEncoder(cell_type=self.config.rnn_cell_type,
                                   output_size=self.config.encoder_hidden_size,
                                   dropout=self.dropout_placeholder,
                                   num_layers=self.config.rnn_layers)
        elif self.config.encoder_type == "bagofwords":
            encoder = BagOfWordsEncoder(cell_type=self.config.rnn_cell_type,
                                        output_size=self.config.encoder_hidden_size,
                                        dropout=self.dropout_placeholder)
        elif self.config.encoder_type == "tree":
            encoder = TreeEncoder(cell_type=self.config.rnn_cell_type,
                                  output_size=self.config.encoder_hidden_size,
                                  dropout=self.dropout_placeholder,
                                  num_layers=self.config.rnn_layers,
                                  max_time=self.config.max_length)
        else:
            raise ValueError("Invalid encoder type")
        return encoder.encode(inputs, self.input_length_placeholder, self.constituency_parse_placeholder)

    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]

    def add_input_op(self, xavier):
        with tf.variable_scope('embed', initializer=xavier):
            # first the embed the input
            with tf.variable_scope('input'):
                if self.config.train_input_embeddings:
                    if self.config.input_embedding_matrix is not None:
                        initializer = tf.constant_initializer(self.config.input_embedding_matrix)
                    else:
                        initializer = None
                    self.input_embed_matrix = tf.get_variable('embedding',
                                                              shape=(self.config.dictionary_size, self.config.input_embed_size),
                                                              initializer=initializer)
                else:
                    self.input_embed_matrix = tf.constant(self.config.input_embedding_matrix)
    
                # dictionary size x embed_size
                assert self.input_embed_matrix.get_shape() == (self.config.dictionary_size, self.config.input_embed_size)

            # now embed the output
            with tf.variable_scope('output'):
                self.output_embed_matrices = dict()
                
                pretrained_output_embed_matrices = self.config.output_embedding_matrix
                for key, size in self.config.grammar.output_size.items():
                    if self.config.grammar.is_copy_type(key):
                        continue
                    if key == self.config.grammar.primary_output and self.config.train_output_embeddings:
                        self.output_embed_matrices[key] = tf.get_variable('embedding_' + key,
                                                                          shape=(size, self.config.output_embed_size))
                    else:
                        self.output_embed_matrices[key] = tf.constant(pretrained_output_embed_matrices[key], name='embedding_' + key)
                    
        inputs = tf.nn.embedding_lookup([self.input_embed_matrix], self.input_placeholder)
        # batch size x max length x embed_size
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.input_embed_size)
        
        # now project the input down to a small size
        input_projection = tf.layers.Dense(units=self.config.input_projection,
                                           name='input_projection')
        
        return input_projection(inputs)
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, training):
        raise NotImplementedError()

    def _add_l2_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search(where, w.name)]
        regularizer = tf.contrib.layers.l2_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def _add_l1_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search(where, w.name)]
        if not weights:
            return 0
        regularizer = tf.contrib.layers.l1_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def add_regularization_loss(self):

        return self._add_l2_helper('/kernel:0$', self.config.l2_regularization) + \
            self._add_l2_helper('/embedding(_[a-z]+)?:0$', self.config.embedding_l2_regularization) + \
            self._add_l1_helper('/kernel:0$', self.config.l1_regularization)

    def finalize_predictions(self, preds):
        raise NotImplementedError()

    def add_loss_op(self, preds, training = True):
        raise NotImplementedError()

    def add_training_op(self, loss):
        optclass = getattr(tf.train, self.config.optimizer + 'Optimizer')
        assert issubclass(optclass, tf.train.Optimizer)

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step,
                                                   1000, self.config.learning_rate_decay)
        optimizer = optclass(learning_rate)

        gradient_var_pairs = optimizer.compute_gradients(loss)
        vars = [x[1] for x in gradient_var_pairs]
        gradients = [x[0] for x in gradient_var_pairs]
        if self.config.gradient_clip > 0:
            clipped, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
        else:
            clipped = gradients

        self.grad_norm = tf.global_norm(clipped)
        train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=global_step)
        return train_op

    def __init__(self, config : Config):
        self.config = config
