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
'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

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
        inputs, output_embed_matrix = self.add_input_op(xavier)
        
        # the encoder
        with tf.variable_scope('RNNEnc', initializer=xavier):
            enc_hidden_states, enc_final_state = self.add_encoder_op(inputs=inputs)
        self.final_encoder_state = enc_final_state

        # the training decoder
        with tf.variable_scope('RNNDec', initializer=xavier):
            train_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, output_embed_matrix=output_embed_matrix, training=True)
        self.loss = self.add_loss_op(train_preds) + self.add_regularization_loss()
        self.train_op = self.add_training_op(self.loss)
        
        # the inference decoder
        with tf.variable_scope('RNNDec', initializer=xavier, reuse=True):
            eval_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, output_embed_matrix=output_embed_matrix, training=False)
        self.pred = self.finalize_predictions(eval_preds)
        self.eval_loss = self.add_loss_op(eval_preds)

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
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.constituency_parse_placeholder = tf.placeholder(tf.bool, shape=(None, 2*self.config.max_length-1))
        
    def add_output_placeholders(self):
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        
    def add_extra_placeholders(self):
        self.batch_number_placeholder = tf.placeholder(tf.int32, shape=())
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, input_length_batch, parses_batch, labels_batch=None, label_length_batch=None, dropout=1, batch_number=0):
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
                    if self.config.input_embedding_matrix:
                        initializer = tf.constant_initializer(self.config.input_embedding_matrix)
                    else:
                        initializer = None
                    input_embed_matrix = tf.get_variable('embedding',
                                                         shape=(self.config.dictionary_size, self.config.input_embed_size),
                                                         initializer=initializer)
                else:
                    input_embed_matrix = tf.constant(self.config.input_embedding_matrix)
    
                # dictionary size x embed_size
                assert input_embed_matrix.get_shape() == (self.config.dictionary_size, self.config.input_embed_size)

            # now embed the output
            with tf.variable_scope('output'):
                if self.config.train_output_embeddings:
                    output_embed_matrix = tf.get_variable('embedding',
                                                          shape=(self.config.output_size, self.config.output_embed_size))
                else:
                    output_embed_matrix = tf.constant(self.config.output_embedding_matrix)
    
                assert output_embed_matrix.get_shape() == (self.config.output_size, self.config.output_embed_size)

        inputs = tf.nn.embedding_lookup([input_embed_matrix], self.input_placeholder)
        # batch size x max length x embed_size
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.input_embed_size)
        
        # now project the input down to a small size
        input_projection = tf.layers.Dense(units=self.config.input_projection,
                                           activation=tf.nn.relu,
                                           name='input_projection')
        
        return input_projection(inputs), output_embed_matrix
    
    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        raise NotImplementedError()

    def _add_l2_helper(self, where, amount):
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith(where)]
        regularizer = tf.contrib.layers.l2_regularizer(amount)
        return tf.contrib.layers.apply_regularization(regularizer, weights)

    def add_regularization_loss(self):
        return self._add_l2_helper('/kernel:0', self.config.l2_regularization) + \
            self._add_l2_helper('/embedding:0', self.config.embedding_l2_regularization)

    def finalize_predictions(self, preds):
        raise NotImplementedError()

    def add_loss_op(self, preds, training = True):
        raise NotImplementedError()

    def add_training_op(self, loss):
        optclass = getattr(tf.train, self.config.optimizer + 'Optimizer')
        assert issubclass(optclass, tf.train.Optimizer)
        optimizer = optclass(self.config.learning_rate)

        gradient_var_pairs = optimizer.compute_gradients(loss)
        vars = [x[1] for x in gradient_var_pairs]
        gradients = [x[0] for x in gradient_var_pairs]
        if self.config.gradient_clip > 0:
            clipped, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip)
        else:
            clipped = gradients

        self.grad_norm = tf.global_norm(clipped)
        train_op = optimizer.apply_gradients(zip(clipped, vars))
        return train_op

    def __init__(self, config : Config):
        self.config = config
