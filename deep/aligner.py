import time

import time
import os
import logging
import sys
import numpy as np
import tensorflow as tf

from model import Model
from general_utils import get_minibatches

import grammar

class Config(object):
    max_length = 40
    dropout = 1
    #dropout = 1
    embed_size = 300
    hidden_size = 150
    batch_size = 64
    #batch_size = 5
    n_epochs = 40
    lr = 0.5
    train_input_embeddings = False
    train_output_embeddings = False
    output_embed_size = 50
    rnn_cell_type = "lstm"
    rnn_layers = 2
    grammar = None

def grammar_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                 maximum_length, grammar,
                                 dtype=tf.int32, name=None):
    """ A version of tf.contrib.seq2seq.simple_decoder_fn_inference
        that applies grammar constraints to the output """
    with tf.name_scope(name, "grammar_decoder_fn_inference",
                       [output_fn, encoder_state, embeddings,
                        maximum_length, dtype]):
        end_of_sequence_id = tf.convert_to_tensor(grammar.end, dtype)
        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = tf.convert_to_tensor(grammar.output_size, dtype)
        encoder_info = encoder_state
        while isinstance(encoder_info, tuple):
            encoder_info = encoder_info[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with tf.name_scope(name, "grammar_decoder_fn_inference",
                           [time, cell_state, cell_input, cell_output,
                            context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            
            if cell_output is None:
                # invariant that this is time == 0
                cell_state = encoder_state
                cell_output = tf.zeros((num_decoder_symbols,),
                                        dtype=tf.float32)
            else:
                cell_output = output_fn(cell_output, cell_state)
            next_input_id, next_state = grammar.constrain(cell_output, context_state, batch_size, dtype=dtype)
            next_input = tf.gather(embeddings, next_input_id)
            done = tf.equal(next_input_id, end_of_sequence_id)

            # if time > maxlen, return all true vector
            done = tf.cond(tf.greater(time, maximum_length),
                           lambda: tf.ones((batch_size,), dtype=tf.bool),
                           lambda: done)
            return (done, cell_state, next_input, cell_output, next_state)
    return decoder_fn

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
            if self.config.train_input_embeddings:
                input_embed_matrix = tf.get_variable('input_embedding',
                                                     shape=(self.config.dictionary_size, self.config.embed_size),
                                                     initializer=tf.constant_initializer(self.pretrained_embeddings))    
            else:
                #input_variable_embeds = tf.get_variable('special_token_embedding',
                #                                        shape=(self.config.variable_embeddings, self.config.embed_size),
                #                                        initializer=xavier)
                #input_constant_embeds = tf.constant(self.pretrained_embeddings[self.config.variable_embeddings:])
                #input_embed_matrix = tf.concat((input_variable_embeds, input_constant_embeds), axis=0)
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
        
        def make_rnn_cell(id):
            with tf.variable_scope('Layer_' + str(id)):
                if self.config.rnn_cell_type == "lstm":
                    cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
                elif self.config.input_cell == "gru":
                    cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
                else:
                    raise ValueError("Invalid RNN Cell type")
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_placeholder, seed=8 + 33 * id)
            return cell
        
        # the encoder
        with tf.variable_scope('RNNEnc', initializer=xavier, reuse=not training) as scope:

            cell_enc = tf.contrib.rnn.MultiRNNCell([make_rnn_cell(id) for id in xrange(self.config.rnn_layers)])
            cell_enc = tf.contrib.rnn.AttentionCellWrapper(cell_enc, 5, state_is_tuple=(self.config.input_cell == "lstm"))

            #cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, input_keep_prob=self.dropout_placeholder, seed=7)
            enc_preds, enc_final_state = tf.nn.dynamic_rnn(cell_enc, inputs, sequence_length=self.input_length_placeholder,
                                                           dtype=tf.float32, scope=scope)
            # assert enc_preds.get_shape()[1:] == (self.config.max_length, self.config.hidden_size)
            # if self.config.input_cell == "lstm":
            #     assert enc_final_state[0][0].get_shape()[1:] == (self.config.hidden_size,)
            #     assert enc_final_state[0][1].get_shape()[1:] == (self.config.hidden_size,)
            # else:
            #     assert enc_final_state.get_shape()[1:] == (self.config.hidden_size,)


        # the decoder
        with tf.variable_scope('RNNDec', initializer=xavier, reuse=not training) as scope:
            cell_dec = tf.contrib.rnn.MultiRNNCell([make_rnn_cell(id) for id in xrange(self.config.rnn_layers)])
            
            U = tf.get_variable('U', shape=(self.config.hidden_size, self.config.output_size), initializer=xavier)
            #V = tf.get_variable('V', shape=(self.config.hidden_size, self.config.output_size), initializer=xavier)
            b_y = tf.get_variable('b_y', shape=(self.config.output_size,), initializer=tf.constant_initializer(0, tf.float32))
            
            if training:
                go_vector = tf.ones((tf.shape(self.output_placeholder)[0], 1), dtype=tf.int32) * self.config.sos
                output_ids_with_go = tf.concat([go_vector, self.output_placeholder], axis=1)
                outputs = tf.nn.embedding_lookup([output_embed_matrix], output_ids_with_go)
                #assert outputs.get_shape()[1:] == (self.config.max_length+1, self.config.output_size)

                decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_final_state[0])
                dec_preds, dec_final_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn,
                    inputs=outputs, sequence_length=self.output_length_placeholder, scope=scope)

                assert dec_preds.get_shape()[2:] == (self.config.hidden_size,)

                # hidden_dec_final_state = dec_final_state
                # if self.config.output_cell == "lstm":
                #     assert dec_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
                #     assert dec_final_state[1].get_shape()[1:] == (self.config.hidden_size,)
                #     hidden_dec_final_state = dec_final_state[1]
                # else:
                #     assert dec_final_state.get_shape()[1:] == (self.config.hidden_size,)
                #
                # hidden_enc_final_state = enc_final_state
                # if self.config.input_cell == "lstm":
                #     hidden_enc_final_state = enc_final_state[1]

                # Attention mechanism
                #print hidden_dec_final_state.get_shape()
                #raw_att_score = tf.matmul(hidden_dec_final_state, hidden_enc_final_state, transpose_b = True)
                #norm_att_score = tf.nn.softmax(raw_att_score)
                #att_context = tf.matmul(norm_att_score, hidden_enc_final_state)

                #preds = tf.tensordot(dec_preds, U, [[2], [0]]) + tf.tensordot(att_context, V, [[2], [0]]) + b_y

                preds = tf.tensordot(dec_preds, U, [[2], [0]]) + b_y
            else:
                def output_fn(cell_output, enc_final_state):
                    assert cell_output.get_shape()[1:] == (self.config.hidden_size,)
                    #hidden_final_state = enc_final_state
                    #if self.config.input_cell == "lstm":
                    #    assert enc_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
                    #    assert enc_final_state[1].get_shape()[1:] == (self.config.hidden_size,)
                    #    hidden_final_state = enc_final_state[1]
                    #else:
                    #    assert enc_final_state.get_shape()[1:] == (self.config.hidden_size,)

                    ## Attention mechanism
                    #raw_att_score = tf.matmul(cell_output, hidden_final_state, transpose_b = True)
                    #print raw_att_score.get_shape()
                    #norm_att_score = tf.nn.softmax(raw_att_score)
                    #att_context = tf.matmul(norm_att_score, hidden_final_state)

                    #result = tf.matmul(cell_output, U) + tf.matmul(att_context, V) + b_y
                    result = tf.matmul(cell_output, U) + b_y

                    assert result.get_shape()[1:] == (self.config.output_size,)
                    return result

                #decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, enc_final_state,
                #    output_embed_matrix, self.config.sos, self.config.eos, self.config.max_length-1, self.config.output_size)
                decoder_fn = grammar_decoder_fn_inference(output_fn, enc_final_state[0], output_embed_matrix,
                                                          self.config.max_length-1, self.config.grammar)
                dec_preds, dec_final_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn, scope=scope)

                assert dec_preds.get_shape()[2:] == (self.config.output_size,)
                #if self.config.rnn_cell_type == "lstm":
                #    assert dec_final_state[0].get_shape()[1:] == (self.config.hidden_size,)
                #    assert dec_final_state[1].get_shape()[1:] == (self.config.hidden_size,)
                #else:
                #    assert dec_final_state.get_shape()[1:] == (self.config.hidden_size,)
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

    def fit(self, sess, train_data, dev_data):
        inputs, input_lengths, labels, label_lengths = train_data
        inputs = np.array(inputs)
        input_lengths = np.reshape(np.array(input_lengths, dtype=np.int32), (len(inputs), 1))
        labels = np.array(labels)
        label_lengths = np.reshape(np.array(label_lengths, dtype=np.int32), (len(inputs), 1))
        stacked_train_data = np.concatenate((inputs, input_lengths, labels, label_lengths), axis=1)
        assert stacked_train_data.shape == (len(train_data[0]), self.config.max_length + 1 + self.config.max_length + 1)
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            shuffled = np.array(stacked_train_data, copy=True)
            np.random.shuffle(shuffled)
            inputs = shuffled[:,:self.config.max_length]
            input_lengths = shuffled[:,self.config.max_length]
            labels = shuffled[:,self.config.max_length + 1:-1]
            label_lengths = shuffled[:,-1]
            
            average_loss = self.run_epoch(sess, inputs, input_lengths,
                                          labels, label_lengths,
                                          dropout=self.config.dropout)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
            print_stats(sess, self, self.config, train_data, 'train', do_print=False)
            if dev_data is not None:
                print_stats(sess, self, self.config, dev_data, 'dev', do_print=False)
            print
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
        word = word.strip()
        if word in words:
            vector[i] = words[word]
        else:
            unknown_tokens.add(word)
            print "sentence: ", sentence, "; word: ", word
            vector[i] = words['<<UNK>>']
        if i+1 == max_length:
            break
    length = i+1
    if length < max_length:
        vector[length] = words['<<EOS>>']
        length += 1
    else:
        print "truncated sentence", sentence
    return (vector, length)

ENTITIES = ['USERNAME', 'HASHTAG',
            'QUOTED_STRING', 'NUMBER',
            'PHONE_NUMBER', 'EMAIL_ADDRESS', 'URL',
            'DATE', 'TIME', 'SET',
            'PERCENT', 'DURATION', 'MONEY', 'ORDINAL']

def load_dictionary(file, benchmark):
    print "Loading dictionary from %s..." % (file,)
    words = dict()

    # special tokens
    words['<<PAD>>'] = len(words)
    words['<<EOS>>'] = len(words)
    words['<<GO>>'] = len(words)
    words['<<UNK>>'] = len(words)
    reverse = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>']

    if benchmark == "tt":
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

def load_embeddings(from_file, words, config):
    print "Loading pretrained embeddings...",
    start = time.time()
    word_vectors = {}
    for line in open(from_file).readlines():
        sp = line.strip().split()
        if sp[0] in words:
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    n_tokens = len(words)
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (n_tokens, config.embed_size)), dtype='float32')
    for token, id in words.iteritems():
        if token in word_vectors:
            embeddings_matrix[id] = word_vectors[token]
    print "took {:.2f} seconds".format(time.time() - start)
    return embeddings_matrix

def load_data(from_file, input_words, output_words, input_reverse, output_reverse, max_length):
    inputs = []
    input_lengths = []
    labels = []
    label_lengths = []
    with open(from_file, 'r') as data:
        for line in data:
            sentence, canonical = line.split('\t')
            input, in_len = vectorize(sentence, input_words, max_length)
            inputs.append(input)
            input_lengths.append(in_len)
            label, label_len = vectorize(canonical, output_words, max_length)
            labels.append(label)
            label_lengths.append(label_len)
            #print "input", in_len, ' '.join(map(lambda x: input_reverse[x], inputs[-1]))
            #print "label", label_len, ' '.join(map(lambda x: output_reverse[x], labels[-1]))
    return inputs, input_lengths, labels, label_lengths

def softmax(x):
    max_x = np.max(x)
    x -= max_x
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    x = exp_x / sum_exp_x
    return x

def print_stats(sess, model, config, data, tag, do_print=True):
    inputs, input_lengths, labels, _ = data
    sequences = []
    
    dict_reverse = config.grammar.tokens

    ok_0 = 0
    ok_1 = 0
    ok_full = 0
    with open("stats_" + tag + ".txt", "w") as fp:
        if do_print:
            print "Writing decoded values to ", fp.name

        for input_batch, input_length_batch, label_batch in get_minibatches([inputs, input_lengths, labels], config.batch_size):
            sequences = list(model.predict_on_batch(sess, input_batch, input_length_batch))

            for i, seq in enumerate(sequences):
                decoded = list(config.grammar.decode_output(seq))
                try:
                    decoded = decoded[:decoded.index(config.eos)]
                except ValueError:
                    pass
            
                gold = list(label_batch[i])
                try:
                    gold = gold[:gold.index(config.eos)]
                except ValueError:
                    pass

                if do_print:
                    gold_str = ' '.join(dict_reverse[l] for l in gold)
                    decoded_str = ' '.join(dict_reverse[l] for l in decoded)
                    print >>fp, gold_str,  '\t',  decoded_str, '\t', (gold_str == decoded_str)

                if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]:
                    ok_0 += 1            
                if len(decoded) > 1 and len(gold) > 1 and decoded[0:2] == gold[0:2]:
                    ok_1 += 1
                if decoded == gold:
                    ok_full += 1
    print tag, "ok 0:", float(ok_0)/len(labels)
    print tag, "ok 1:", float(ok_1)/len(labels)
    print tag, "ok full:", float(ok_full)/len(labels)

def print_embed_matrix(matrix):
    print matrix
    avg = np.average(matrix, axis=1)
    avg_x2 = np.average(matrix * matrix, axis=1)
    stddev = avg_x2 - avg * avg
    print "avg", avg
    print "stddev", stddev
    
    avg = np.average(matrix)
    avg_x2 = np.average(matrix * matrix)
    stddev = avg_x2 - avg * avg
    print "avg", avg
    print "stddev", stddev

def run():
    if len(sys.argv) < 5:
        print "** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Input Vocab>> <<Word Embeddings>> <<Train Set> [<<Dev Set>>]"
        sys.exit(1)

    np.random.seed(42)
    config = Config()

    benchmark = sys.argv[1]
    if benchmark == "tt":
        print "Loading ThingTalk Grammar"
        config.grammar = grammar.ThingtalkGrammar()
    elif benchmark == "geo":
        print "Loading Geoqueries Grammar"
        config.grammar = grammar.SimpleGrammar("geoqueries/output_tokens.txt")
    else:
        print "Invalid benchmark", benchmark
        sys.exit(1)

    words, reverse = load_dictionary(sys.argv[2], benchmark)
    if benchmark == "tt":
        config.variable_embeddings = 4 + len(ENTITIES)
    else:
        config.variable_embeddings = 4
    config.dictionary_size = len(words)
    print "%d words in dictionary" % (config.dictionary_size,)
    embeddings_matrix = load_embeddings(sys.argv[3], words, config)

    config.output_size = config.grammar.output_size
    if not config.train_output_embeddings:
        config.output_embed_size = config.output_size
    print "%d output tokens" % (config.output_size,)
    config.sos = config.grammar.start
    config.eos = config.grammar.end
    train_data = load_data(sys.argv[4], words, config.grammar.dictionary,
                           reverse, config.grammar.tokens,
                           config.max_length)
    if len(sys.argv) > 5:
        dev_data = load_data(sys.argv[5], words, config.grammar.dictionary,
                             reverse, config.grammar.tokens,
                             config.max_length)
    else:
        dev_data = None
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
            losses = model.fit(sess, train_data, dev_data)
            
            print "Final result"
            print_stats(sess, model, config, train_data, "train")
            if dev_data is not None:
                print_stats(sess, model, config, dev_data, "dev")
            
            #for var in tf.global_variables():
            #    print var.name,
            #    print var.value().eval(session=sess)

if __name__ == "__main__":
    run()
