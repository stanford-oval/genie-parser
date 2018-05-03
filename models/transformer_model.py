# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: sawyerb <sawyerb@cs.stanford.edu>
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
Created on May 2, 2017

@author: sawyerb
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest

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
        
        if self.config.decoder_action_count_loss > 0:
            # only consider the first 23 values (excluding any query, action, parameter or filter)
            #count_layer = tf.layers.Dense(self.config.grammar.output_size, name='action_count_layer')
            count_layer = tf.layers.Dense(23, name='action_count_layer')
            action_count_logits = count_layer(tf.concat(nest.flatten(enc_final_state), axis=1))
            #self.action_counts = action_count_logits > 0
            self.action_counts = None
        else:
            self.action_counts = None

        # the training decoder
        with tf.variable_scope('decoder', initializer=xavier):
            train_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, training=True)

        if self.config.decoder_action_count_loss > 0:
            with tf.name_scope('action_count_loss'):
                binarized_label = tf.cast(self.output_action_counts[:,:23] >= 1, dtype=tf.float32)
                #binarized_predictions = tf.cast(self.action_counts >= 0.5, dtype=tf.float32)
                #action_count_loss = tf.nn.l2_loss(tf.cast(self.output_action_counts, dtype=tf.float32) - self.action_counts)
                
                action_count_loss = tf.losses.hinge_loss(labels=binarized_label, logits=action_count_logits,
                                                         reduction=tf.losses.Reduction.NONE)
                action_count_loss = tf.reduce_sum(action_count_loss, axis=1)
                action_count_loss = tf.reduce_mean(action_count_loss)
        else:
            action_count_loss = 0
        
        if self.config.decoder_sequence_loss > 0:
            with tf.name_scope('training_sequence_loss'):
                training_sequence_loss = self.add_loss_op(train_preds)
        else:
            training_sequence_loss = 0
        
        with tf.name_scope('training_loss'):
            self.loss = self.config.decoder_action_count_loss * action_count_loss + \
                self.config.decoder_sequence_loss * training_sequence_loss + \
                self.add_regularization_loss()
        self.train_op = self.add_training_op(self.loss)
        
        # the inference decoder
        with tf.variable_scope('decoder', initializer=xavier, reuse=True):
            eval_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, training=False)
        self.pred = self.finalize_predictions(eval_preds)
        
        if self.config.decoder_sequence_loss > 0:
            with tf.name_scope('eval_sequence_loss'):
                eval_sequence_loss = self.add_loss_op(eval_preds)
        else:
            eval_sequence_loss = 0
        with tf.name_scope('eval_loss'):
            self.eval_loss = self.config.decoder_action_count_loss * action_count_loss + \
                self.config.decoder_sequence_loss * eval_sequence_loss

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
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name='output_sequence')
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,), name='output_length')
        self.output_action_counts = tf.placeholder(dtype=tf.int32, shape=(None, self.config.grammar.output_size), name='output_action_counts')
        
    def add_extra_placeholders(self):
        self.batch_number_placeholder = tf.placeholder(tf.int32, shape=(), name='batch_number')
        self.epoch_placeholder = tf.placeholder(tf.int32, shape=(), name='epoch_number')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_probability')

    def create_feed_dict(self, inputs_batch, input_length_batch, parses_batch,
                         labels_batch=None, label_length_batch=None,
                         dropout=1, batch_number=0, epoch=0):
        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.input_length_placeholder] = input_length_batch
        feed_dict[self.constituency_parse_placeholder] = parses_batch
        if labels_batch is not None:
            feed_dict[self.output_placeholder] = labels_batch
        if label_length_batch is not None:
            feed_dict[self.output_length_placeholder] = label_length_batch
        feed_dict[self.dropout_placeholder] = dropout
        feed_dict[self.batch_number_placeholder] = batch_number
        feed_dict[self.epoch_placeholder] = epoch
        
        if self.config.decoder_action_count_loss > 0 and labels_batch is not None:
            action_count_batch = np.zeros((len(labels_batch), self.config.grammar.output_size), dtype=np.int32)
            for i in range(len(labels_batch)):
                action_count_batch[i] = np.bincount(labels_batch[i][:label_length_batch[i]],
                                                    minlength=self.config.grammar.output_size)
            feed_dict[self.output_action_counts] = action_count_batch
        return feed_dict
        
        return feed_dict

    def add_encoder_op(self, inputs):
        if self.config.encoder_type == "rnn":
            encoder = RNNEncoder(cell_type=self.config.rnn_cell_type,
                                 input_size=self.config.input_projection,
                                 output_size=self.config.encoder_hidden_size,
                                 dropout=self.dropout_placeholder,
                                 num_layers=self.config.rnn_layers)
        elif self.config.encoder_type == 'birnn':
            encoder = BiRNNEncoder(cell_type=self.config.rnn_cell_type,
                                   input_size=self.config.input_projection,
                                   output_size=self.config.encoder_hidden_size,
                                   dropout=self.dropout_placeholder,
                                   num_layers=self.config.rnn_layers)
        elif self.config.encoder_type == "bagofwords":
            encoder = BagOfWordsEncoder(cell_type=self.config.rnn_cell_type,
                                        output_size=self.config.encoder_hidden_size,
                                        dropout=self.dropout_placeholder)
        elif self.config.encoder_type == "tree":
            encoder = TreeEncoder(cell_type=self.config.rnn_cell_type,
                                  input_size=self.config.input_projection,
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
                if self.config.train_output_embeddings:
                    self.output_embed_matrix = tf.get_variable('embedding',
                                                               shape=(self.config.output_size, self.config.output_embed_size))
                else:
                    self.output_embed_matrix = tf.constant(self.config.output_embedding_matrix)
    
                assert self.output_embed_matrix.get_shape() == (self.config.output_size, self.config.output_embed_size)

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
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith(where)]
        regularizer = tf.contrib.layers.l2_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def _add_l1_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith(where)]
        regularizer = tf.contrib.layers.l1_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def add_regularization_loss(self):
        return self._add_l2_helper('/kernel:0', self.config.l2_regularization) + \
            self._add_l2_helper('/embedding:0', self.config.embedding_l2_regularization) + \
            self._add_l1_helper('/kernel:0', self.config.l1_regularization) + \
            self._add_l2_helper('/decoder/dense/kernel:0', self.config.embedding_l2_regularization)

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
