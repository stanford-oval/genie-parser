import time

import time
import os
import logging
import sys
import numpy as np
import tensorflow as tf

from model import Model
from general_utils import get_minibatches

class Config(object):
    max_length = 30
    dropout = 0.7
    #dropout = 1
    embed_size = 300
    hidden_size = 150
    batch_size = 128
    #batch_size = 1
    n_epochs = 40
    lr = 0.6


class LSTMAligner(Model):
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

    def add_prediction_op(self, training):
        xavier = tf.contrib.layers.xavier_initializer(seed=42)

        with tf.variable_scope('embed', reuse=not training):
            # first the embed the input
            input_embed_matrix = tf.constant(self.pretrained_embeddings)
            #input_embed_matrix = tf.get_variable('input_embedding', shape=(self.config.dictionary_size, self.config.embed_size), initializer=tf.constant_initializer(self.pretrained_embeddings))    
            # dictionary size x embed_size
            assert input_embed_matrix.get_shape() == (self.config.dictionary_size, self.config.embed_size)

            # now embed the output
            #output_embed_matrix = tf.get_variable('output_embedding',
            #                                      shape=(self.config.output_size, self.config.embed_size),
            #                                      initializer=xavier)
            #assert output_embed_matrix.get_shape() == (self.config.output_size, self.config.embed_size)
            output_embed_matrix = tf.eye(self.config.output_size)

        inputs = tf.nn.embedding_lookup([input_embed_matrix], self.input_placeholder)
        # batch size x max length x embed_size
        assert inputs.get_shape()[1:] == (self.config.max_length, self.config.embed_size)
        
        # the encoder
        with tf.variable_scope('RNNEnc', initializer=xavier, reuse=not training) as scope:
            lstm_enc = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            lstm_enc = tf.contrib.rnn.DropoutWrapper(lstm_enc, input_keep_prob=self.dropout_placeholder, seed=7)
            enc_preds, enc_final_state = tf.nn.dynamic_rnn(lstm_enc, inputs, sequence_length=self.input_length_placeholder,
                                                           dtype=tf.float32, scope=scope)
            assert enc_preds.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
            assert enc_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
            assert enc_final_state[1].get_shape()[1:] == (self.config.hidden_size,)

        # the decoder
        with tf.variable_scope('RNNDec', initializer=xavier, reuse=not training) as scope:
            lstm_dec = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            lstm_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, output_keep_prob=self.dropout_placeholder, seed=8)
            
            U = tf.get_variable('U', shape=(self.config.hidden_size, self.config.output_size), initializer=xavier)
            b_y = tf.get_variable('b_y', shape=(self.config.output_size,), initializer=tf.constant_initializer(0, tf.float32))
            
            if training and False:
                go_vector = tf.ones((tf.shape(self.output_placeholder)[0], 1), dtype=tf.int32) * self.config.sos
                output_ids_with_go = tf.concat([go_vector, self.output_placeholder], axis=1)
                outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
                #assert outputs.get_shape()[1:] == (self.config.max_length+1, self.config.output_size)

                decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_final_state)
                dec_preds, dec_final_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(lstm_dec, decoder_fn,
                    inputs=outputs, sequence_length=self.output_length_placeholder, scope=scope)

                assert dec_preds.get_shape()[2:] == (self.config.hidden_size,)
                assert dec_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
                assert dec_final_state[1].get_shape()[1:] == (self.config.hidden_size,)
                preds = tf.tensordot(dec_preds, U, [[2], [0]]) + b_y
            else:
                def output_fn(cell_output):
                    assert cell_output.get_shape()[1:] == (self.config.hidden_size,)
                    result = tf.matmul(cell_output, U) + b_y
                    assert result.get_shape()[1:] == (self.config.output_size,)
                    return result

                decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, enc_final_state,
                    output_embed_matrix, self.config.sos, self.config.eos, self.config.max_length-1, self.config.output_size)
                dec_preds, dec_final_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(lstm_dec, decoder_fn, scope=scope)

                assert dec_preds.get_shape()[2:] == (self.config.output_size,)
                assert dec_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
                assert dec_final_state[1].get_shape()[1:] == (self.config.hidden_size,)
                preds = dec_preds
            #print preds.get_shape()
            #assert preds.get_shape()[2:] == (self.config.output_size,)

        return preds

    def add_loss_op(self, preds):
        length_diff = tf.reshape(self.config.max_length - tf.shape(preds)[1], shape=(1,))
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0, 0]], axis=0), shape=(3, 2))
        preds = tf.pad(preds, padding, mode='constant')
        #labels = tf.slice(self.output_placeholder, [0, 0], [-1, self.output_length_placeholder])
        #labels = self.output_placeholder[:,:self.output_length_placeholder]
        labels = self.output_placeholder
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        assert loss.get_shape()[1:] == (self.config.max_length,)
        output_mask = tf.sequence_mask(self.output_length_placeholder, self.config.max_length)
        loss = tf.boolean_mask(loss, output_mask)
        asserts = [tf.Assert(tf.reduce_any(loss > 0), [loss], name='loss_gt_0'),
                   #tf.Assert(tf.shape(preds)[1:] == [self.config.max_length, self.config.output_size], [preds, tf.shape(preds)[1:]], name='shape_of_preds'),
                   tf.Assert(tf.reduce_any(output_mask != False), [output_mask], name='output_mask'),
                   tf.Assert(tf.reduce_all(tf.argmax(preds[:,0,:], axis=1) != self.config.eos), [preds[:,0,:]], name='assert_not_empty')]
        with tf.control_dependencies(asserts):
            loss = tf.reduce_sum(loss)
            assert loss.get_shape() == ()
            return loss

    def add_training_op(self, loss):
        #optimizer = tf.train.AdamOptimizer(self.config.lr)
        optimizer = tf.train.AdagradOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(self, sess, inputs, input_lengths,
                  labels, label_lengths, **kw):
        n_minibatches, total_loss = 0, 0
        for data_batch in get_minibatches([inputs, input_lengths, labels, label_lengths], self.config.batch_size):
            n_minibatches += 1
            for x in data_batch:
                assert len(x) == len(data_batch[0])
            assert len(data_batch[0]) <= self.config.batch_size
            total_loss += self.train_on_batch(sess, *data_batch, **kw)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, input_lengths, labels, label_lengths):
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, input_lengths,
                                          labels, label_lengths,
                                          dropout=self.config.dropout)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.pretrained_embeddings = pretrained_embeddings
        self.build()

unknown_tokens = set()

def vectorize(sentence, words, max_length):
    vector = np.zeros((max_length,), dtype=np.int32)
    assert words['<<PAD>>'] == 0
    #vector[0] = words['<<GO>>']
    for i, word in enumerate(sentence.split(' ')):
        if i+1 == max_length:
            break
        word = word.strip()
        if word in words:
            vector[i] = words[word]
        else:
            unknown_tokens.add(word)
            vector[i] = words['<<UNK>>']
    vector[i] = words['<<EOS>>']
    return (vector, i+1)

ENTITIES = ['USERNAME', 'HASHTAG',
            'QUOTED_STRING', 'NUMBER',
            'PHONE_NUMBER', 'EMAIL_ADDRESS', 'URL',
            'DATE', 'TIME', 'SET',
            'PERCENT', 'DURATION', 'MONEY', 'ORDINAL']

def load_dictionary(file):
    print "Loading dictionary from %s..." % (file,)
    words = dict()

    # special tokens
    words['<<PAD>>'] = len(words)
    words['<<EOS>>'] = len(words)
    words['<<GO>>'] = len(words)
    words['<<UNK>>'] = len(words)
    reverse = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>']

    for entity in ENTITIES:
        words[entity] = len(words)
        reverse.append(entity)

    with open(file, 'r') as word_file:
        for word in word_file:
            word = word.strip()
            if word not in words:
                words[word] = len(words)
                reverse.append(word)
    for id in xrange(len(reverse)):
        if words[reverse[id]] != id:
            print "found problem at", id
            print "word: ", reverse[id]
            print "expected: ", words[reverse[id]]
            raise AssertionError
    return words, reverse

def load_embeddings(words, config):
    print "Loading pretrained embeddings...",
    start = time.time()
    word_vectors = {}
    for line in open("embeddings.txt").readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    n_tokens = len(words)
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (n_tokens, config.embed_size)), dtype='float32')
    for token, id in words.iteritems():
        if token in word_vectors:
            embeddings_matrix[id] = word_vectors[token]
    print "took {:.2f} seconds".format(time.time() - start)
    return embeddings_matrix

def load_data(input_words, output_words, input_reverse, output_reverse, max_length):
    inputs = []
    input_lengths = []
    labels = []
    label_lengths = []
    with open(sys.argv[1], 'r') as data:
        for line in data:
            sentence, canonical = line.split('\t')
            input, in_len = vectorize(sentence, input_words, max_length)
            inputs.append(input)
            input_lengths.append(in_len)
            label, label_len = vectorize(canonical, output_words, max_length)
            labels.append(label)
            label_lengths.append(label_len)
            #print "input", ' '.join(map(lambda x: input_reverse[x], inputs[-1]))
            #print "label", map(lambda x: output_reverse[x], labels[-1])
    return inputs, input_lengths, labels, label_lengths

def softmax(x):
    max_x = np.max(x)
    x -= max_x
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    x = exp_x / sum_exp_x
    return x

def decode_output(sequence, config):
    output = []
    for word_prob in sequence:
        assert word_prob.shape == (config.output_size,)
        word_idx = np.argmax(word_prob)
        if word_idx > 0:
            output.append(word_idx)
    return output

def run():
    config = Config()
    
    words, reverse = load_dictionary('words.txt')
    config.dictionary_size = len(words)
    print "%d words in dictionary" % (config.dictionary_size,)
    embeddings_matrix = load_embeddings(words, config)
    canonical_words, canonical_reverse = load_dictionary('canonical_tokens.txt')
    config.output_size = len(canonical_words)
    print "%d output tokens" % (config.output_size,)
    config.sos = canonical_words['<<GO>>']
    config.eos = canonical_words['<<EOS>>']
    inputs, input_lengths, labels, label_lengths = load_data(words, canonical_words,
                                                             reverse, canonical_reverse,
                                                             config.max_length)
    print "unknown", unknown_tokens

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = LSTMAligner(config, embeddings_matrix)
        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            losses = model.fit(sess, inputs, input_lengths, labels, label_lengths)
            
            sequences = model.predict_on_batch(sess, inputs, input_lengths)
            
            ok_0 = 0
            ok_3 = 0
            for i, seq in enumerate(sequences):
                decoded = decode_output(seq, config)
                print len(decoded), ' '.join(map(lambda x: canonical_reverse[x], decoded))
                if decoded[0] == labels[i][0]:
                    ok_0 += 1
                if np.all(decoded[:3] == labels[i][:3]):
                    ok_3 += 1
            print "Ok 0:", float(ok_0)/len(labels)
            print "Ok 3:", float(ok_3)/len(labels)

if __name__ == "__main__":
    run()
