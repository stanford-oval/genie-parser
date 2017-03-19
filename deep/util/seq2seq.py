'''
Created on Mar 16, 2017

@author: gcampagn
'''

import tensorflow as tf
import numpy as np

from general_utils import get_minibatches


class AbstractGrammar(object):
    def __init__(self):
        self.tokens = []
        self.dictionary = dict()
        self.output_size = 0
        self.start = 0
        self.end = 0
    
    def constrain(self, logits, curr_state, batch_size, dtype=tf.int32):
        if curr_state is None:
            return tf.ones((batch_size,), dtype=dtype) * self.start, ()
        else:
            return tf.cast(tf.argmax(logits, axis=1), dtype=dtype), ()

    def decode_output(self, sequence):
        output = []
        for logits in sequence:
            assert logits.shape == (self.output_size,)
            word_idx = np.argmax(logits)
            if word_idx > 0:
                output.append(word_idx)
        return output
    
    def compare(self, seq1, seq2):
        return seq1 == seq2


class SimpleGrammar(AbstractGrammar):
    def __init__(self, filename):
        tokens = set()
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                tokens.add(line.strip().lower())
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>'] + list(tokens)
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
            
        self.output_size = len(self.tokens)
        
        self.start = self.dictionary['<<GO>>']
        self.end = self.dictionary['<<EOS>>']


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
                cell_output = output_fn(cell_output, cell_state, batch_size)
            next_input_id, next_state = grammar.constrain(cell_output, context_state, batch_size, dtype=dtype)
            next_input = tf.gather(embeddings, next_input_id)
            done = tf.equal(next_input_id, end_of_sequence_id)

            # if time > maxlen, return all true vector
            done = tf.cond(tf.greater(time, maximum_length),
                           lambda: tf.ones((batch_size,), dtype=tf.bool),
                           lambda: done)
            return (done, cell_state, next_input, cell_output, next_state)
    return decoder_fn


class Seq2SeqEvaluator(object):
    '''
    Evaluate a sequence to sequence model on some data against some gold data
    '''

    def __init__(self, model, grammar, data, tag, batch_size=256):
        self.model = model
        self.grammar = grammar
        self.data = data
        self.tag = tag
        self._batch_size = batch_size
        
    def eval(self, session, save_to_file=False):
        inputs, input_lengths, labels, _ = self.data
        sequences = []
    
        dict_reverse = self.grammar.tokens

        ok_0 = 0
        ok_fn = 0
        ok_ch = 0
        ok_full = 0
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print "Writing decoded values to ", fp.name

        try:
            for input_batch, input_length_batch, label_batch in get_minibatches([inputs, input_lengths, labels], self._batch_size):
                sequences = list(self.model.predict_on_batch(session, input_batch, input_length_batch))

                for i, seq in enumerate(sequences):
                    decoded = list(self.grammar.decode_output(seq))
                    try:
                        decoded = decoded[:decoded.index(self.grammar.end)]
                    except ValueError:
                        pass
            
                    gold = list(label_batch[i])
                    try:
                        gold = gold[:gold.index(self.grammar.end)]
                    except ValueError:
                        pass

                    if save_to_file:
                        gold_str = ' '.join(dict_reverse[l] for l in gold)
                        decoded_str = ' '.join(dict_reverse[l] for l in decoded)
                        print >>fp, gold_str,  '\t',  decoded_str, '\t', (gold_str == decoded_str)

                    if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]:
                        ok_0 += 1

                    def get_functions(seq):
                        return set(filter(lambda x:x.startswith('tt:') and not x.startswith('tt:param.'), map(lambda x: self.grammar.tokens[x], seq)))
                    gold_functions = get_functions(gold)
                    decoded_functions = get_functions(decoded)
                    gold_channels = set(map(lambda x:x[x.index('.')+1:], gold_functions))
                    decoded_channels = set(map(lambda x:x[x.index('.')+1:], decoded_functions))
                    if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0] and gold_functions == decoded_functions:
                        ok_fn += 1
                    if gold_channels == decoded_channels:
                        ok_ch += 1
                    if self.grammar.compare(gold, decoded):
                        ok_full += 1
            print self.tag, "ok 0:", float(ok_0)/len(labels)
            print self.tag, "ok channel:", float(ok_ch)/len(labels)
            print self.tag, "ok function:", float(ok_fn)/len(labels)
            print self.tag, "ok full:", float(ok_full)/len(labels)
        finally:
            if fp is not None:
                fp.close()
        
        return float(ok_full)/len(labels)
        
