'''
Created on Mar 16, 2017

@author: gcampagn, rakeshr
'''

import tensorflow as tf
from tensorflow.python.layers import core as tf_core_layers

from util.model import Model
# FIXME restore grammar constraints
#from util.seq2seq import grammar_decoder_fn_inference
from util.loader import load_dictionary, load_embeddings

from tensorflow.contrib.seq2seq import BasicDecoder, \
    TrainingHelper, GreedyEmbeddingHelper, LuongAttention, AttentionWrapper, AttentionWrapperState

from util.seq2seq import SimpleGrammar
from thingtalk.grammar import ThingtalkGrammar

class Config(object):
    max_length = 60
    dropout = 0.5
    #dropout = 1
    embed_size = 300
    hidden_size = 175
    batch_size = 256
    #beam_size = 10
    beam_size = -1 # no beam decoding
    n_epochs = 40
    lr = 0.001
    train_input_embeddings = False
    train_output_embeddings = False
    output_embed_size = 50
    rnn_cell_type = "lstm"
    rnn_layers = 1
    apply_attention = True
    grammar = None
    
    def apply_cmdline(self, cmdline):
        self.dropout = float(cmdline[0])
        self.hidden_size = int(cmdline[1])
        self.rnn_cell_type = cmdline[2]
        self.rnn_layers = int(cmdline[3])
        self.apply_attention = (cmdline[4] == "yes")
        return 6

class BaseAligner(Model):
    def add_placeholders(self):
        # batch size x number of words in the sentence
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.output_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length,))
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

    def add_encoder_op(self, inputs, training):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return tf.shape(self.output_placeholder)[0]

    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training, scope=None):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])

        if self.config.apply_attention:
            attention = LuongAttention(self.config.hidden_size, enc_hidden_states, self.input_length_placeholder,
                                       probability_fn=tf.nn.softmax)
            cell_dec = AttentionWrapper(cell_dec, attention,
                                        cell_input_fn=lambda inputs, _: inputs,
                                        attention_layer_size=self.config.hidden_size,
                                        initial_cell_state=enc_final_state)
            enc_final_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        linear_layer = tf_core_layers.Dense(self.config.output_size)

        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.sos
        if training:
            output_ids_with_go = tf.concat([tf.expand_dims(go_vector, axis=1), self.output_placeholder], axis=1)
            outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
            helper = TrainingHelper(outputs, self.output_length_placeholder+1)
        else:
            helper = GreedyEmbeddingHelper(output_embed_matrix, go_vector, self.config.eos)
        decoder = BasicDecoder(cell_dec, helper, enc_final_state, output_layer = linear_layer)

        dec_final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=self.config.max_length)            
        return dec_final_outputs.rnn_output

    def add_prediction_op(self, training):
        xavier = tf.contrib.layers.xavier_initializer(seed=1234)

        with tf.variable_scope('embed', reuse=not training):
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
                                                      initializer=xavier)
            else:
                output_embed_matrix = tf.eye(self.config.output_size)
                
            assert output_embed_matrix.get_shape() == (self.config.output_size, self.config.output_embed_size)

        inputs = tf.nn.embedding_lookup([input_embed_matrix], self.input_placeholder)
        # batch size x max length x embed_size
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.embed_size)
        
        # the encoder
        with tf.variable_scope('RNNEnc', initializer=xavier, reuse=not training) as scope:
            enc_hidden_states, enc_final_state = self.add_encoder_op(inputs=inputs, training=training, scope=scope)
            if not training:
                if self.capture_final_encoder_state:
                    self.final_encoder_state = enc_final_state
                else:
                    self.final_encoder_state = None

        # the decoder
        with tf.variable_scope('RNNDec', initializer=xavier, reuse=not training) as scope:
            preds = self.add_decoder_op(enc_final_state=enc_final_state, enc_hidden_states=enc_hidden_states, output_embed_matrix=output_embed_matrix, training=training, scope=scope)

        return preds

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
        optimizer = tf.train.RMSPropOptimizer(self.config.lr, decay=0.95)
        train_op = optimizer.minimize(loss)
        return train_op

    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.pretrained_embeddings = pretrained_embeddings
        self.capture_attention = False
        self.capture_final_encoder_state = False


class LSTMAligner(BaseAligner):
    def add_encoder_op(self, inputs, training, scope=None):
        cell_enc = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(id) for id in range(self.config.rnn_layers)])
        #cell_enc = tf.contrib.rnn.AttentionCellWrapper(cell_enc, 5, state_is_tuple=True)

        return tf.nn.dynamic_rnn(cell_enc, inputs, sequence_length=self.input_length_placeholder,
                                 dtype=tf.float32, scope=scope)
        # assert enc_preds.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
        # if self.config.input_cell == "lstm":
        #     assert enc_final_state[0][0].get_shape()[1:] == (self.config.hidden_size,)
        #     assert enc_final_state[0][1].get_shape()[1:] == (self.config.hidden_size,)
        # else:
        #     assert enc_final_state.get_shape()[1:] == (self.config.hidden_size,)


class BagOfWordsAligner(BaseAligner):
    def add_encoder_op(self, inputs, training, scope=None):
        W = tf.get_variable('W', (self.config.embed_size, self.config.hidden_size))
        b = tf.get_variable('b', shape=(self.config.hidden_size,), initializer=tf.constant_initializer(0, tf.float32))

        enc_hidden_states = tf.tanh(tf.tensordot(inputs, W, [[2], [0]]) + b)
        enc_hidden_states.set_shape((None, self.config.max_length, self.config.hidden_size))
        enc_final_state = tf.reduce_sum(enc_hidden_states, axis=1)

        #assert enc_hidden_states.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
        
        if self.config.rnn_cell_type == 'lstm':
            enc_final_state = (tf.contrib.rnn.LSTMStateTuple(enc_final_state, enc_final_state),)
        
        return enc_hidden_states, enc_final_state


def initialize(benchmark, model_type, input_words, embedding_file):
    config, words, reverse, embeddings_matrix = load(benchmark, input_words, embedding_file)
    model = create_model(config, model_type, embeddings_matrix)

    return config, words, reverse, model


def create_model(config, model_type, embeddings_matrix):
    if model_type == 'bagofwords':
        model = BagOfWordsAligner(config, embeddings_matrix)
    elif model_type == 'seq2seq':
        model = LSTMAligner(config, embeddings_matrix)
    else:
        raise ValueError("Invalid model type %s" % (model_type,))
    
    return model


def load(benchmark, input_words, embedding_file):
    config = Config()

    if benchmark == "tt":
        print("Loading ThingTalk Grammar")
        config.grammar = ThingtalkGrammar()

        #Uncomment this for separate channel
        #config.grammar = SimpleGrammar("/srv/data/deep-sempre/workdir.sepchannel/output_tokens.txt")
    elif benchmark == "geo":
        print("Loading Geoqueries Grammar")
        config.grammar = SimpleGrammar("geoqueries/output_tokens.txt")
    else:
        raise ValueError("Invalid benchmark %s" % (benchmark,))

    words, reverse = load_dictionary(input_words, benchmark)
    config.dictionary_size = len(words)
    print("%d words in dictionary" % (config.dictionary_size,))
    embeddings_matrix = load_embeddings(embedding_file, words, embed_size=config.embed_size)

    config.output_size = config.grammar.output_size
    if not config.train_output_embeddings:
        config.output_embed_size = config.output_size
    print("%d output tokens" % (config.output_size,))
    config.sos = config.grammar.start
    config.eos = config.grammar.end

    return config, words, reverse, embeddings_matrix