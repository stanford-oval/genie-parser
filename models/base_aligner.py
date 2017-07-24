'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_model import BaseModel
from .seq2seq_helpers import Seq2SeqDecoder, AttentionSeq2SeqDecoder

from .config import Config

class BaseAligner(BaseModel):
    def build(self):
        self.add_placeholders()
        
        
        xavier = tf.contrib.layers.xavier_initializer(seed=1234)
        inputs, output_embed_matrix = self.add_input_op(xavier)
        
        # the encoder
        with tf.variable_scope('RNNEnc', initializer=xavier) as scope:
            enc_hidden_states, enc_final_state = self.add_encoder_op(inputs=inputs, scope=scope)
            
        # the training decoder
        with tf.variable_scope('RNNDec', initializer=xavier) as scope:
            train_preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, output_embed_matrix=output_embed_matrix, training=True, scope=scope)
        self.loss = self.add_loss_op(train_preds)
        self.train_op = self.add_training_op(self.loss)
        
        # the inference decoder
        with tf.variable_scope('RNNDec', initializer=xavier, reuse=True) as scope:
            self.pred = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, output_embed_matrix=output_embed_matrix, training=False, scope=scope)
    
    def add_placeholders(self):
        # batch size x number of words in the sentence
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.output_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, input_length_batch, labels_batch=None, label_length_batch=None, dropout=1):
        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.input_length_placeholder] = input_length_batch
        if labels_batch is not None:
            feed_dict[self.output_placeholder] = labels_batch
        if label_length_batch is not None:
            feed_dict[self.output_length_placeholder] = label_length_batch
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict
    
    def make_rnn_cell(self, id):
        if self.config.rnn_cell_type == "lstm":
            cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
        elif self.config.rnn_cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        elif self.config.rnn_cell_type == "basic-tanh":
            cell = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size)
        else:
            raise ValueError("Invalid RNN Cell type")
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_placeholder, seed=8 + 33 * id)
        return cell

    def add_encoder_op(self, inputs):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return tf.shape(self.input_placeholder)[0]

    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training, scope=None):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])
        if self.config.apply_attention:
            decoder = AttentionSeq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                              self.output_placeholder, self.output_length_placeholder)
        else:
            decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                     self.output_placeholder, self.output_length_placeholder)
        return decoder.decode(cell_dec, enc_hidden_states, enc_final_state, output_embed_matrix, training)

    def add_input_op(self, initializer):
        with tf.variable_scope('embed'):
            # first the embed the input
            if self.config.train_input_embeddings:
                input_embed_matrix = tf.get_variable('input_embedding',
                                                     shape=(self.config.dictionary_size, self.config.embed_size),
                                                     initializer=tf.constant_initializer(self.pretrained_embeddings))    
            else:
                input_embed_matrix = tf.constant(self.pretrained_embeddings)

            # dictionary size x embed_size
            assert input_embed_matrix.get_shape() == (self.config.dictionary_size, self.config.embed_size)

            # now embed the output
            if self.config.train_output_embeddings:
                output_embed_matrix = tf.get_variable('output_embedding',
                                                      shape=(self.config.output_size, self.config.output_embed_size),
                                                      initializer=initializer)
            else:
                output_embed_matrix = tf.eye(self.config.output_size)
                
            assert output_embed_matrix.get_shape() == (self.config.output_size, self.config.output_embed_size)

        inputs = tf.nn.embedding_lookup([input_embed_matrix], self.input_placeholder)
        # batch size x max length x embed_size
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.embed_size)
        return inputs, output_embed_matrix

    def add_loss_op(self, preds):
        length_diff = tf.reshape(self.config.max_length - tf.shape(preds)[1], shape=(1,))
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0, 0]], axis=0), shape=(3, 2))
        preds = tf.pad(preds, padding, mode='constant')
        mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(preds, self.output_placeholder, mask)

        return loss

    def add_training_op(self, loss):
        #optimizer = tf.train.AdamOptimizer(self.config.lr)
        #optimizer = tf.train.AdagradOptimizer(self.config.lr)
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=0.95)
        train_op = optimizer.minimize(loss)
        return train_op

    def __init__(self, config : Config):
        self.config = config
        self.pretrained_embeddings = config.input_embedding_matrix
