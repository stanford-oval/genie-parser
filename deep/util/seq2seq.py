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
                tokens.add(line.strip())
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>'] + list(tokens)
        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
            
        self.output_size = len(self.tokens)
        
        self.start = self.dictionary['<<GO>>']
        self.end = self.dictionary['<<EOS>>']


def grammar_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                 maximum_length, grammar,
                                 dtype=tf.int32, name=None, first_output_state=None):
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
        if first_output_state is None:
            first_output_state = tf.zeros((1,))

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
                grammar_state = None
                next_output_state = first_output_state
            else:
                grammar_state, output_state = context_state
                cell_output, next_output_state = output_fn(time, cell_output, cell_state, batch_size, output_state)

            next_input_id, next_grammar_state = grammar.constrain(cell_output, grammar_state, batch_size, dtype=dtype)
            next_input = tf.gather(embeddings, next_input_id)
            done = tf.equal(next_input_id, end_of_sequence_id)

            # if time > maxlen, return all true vector
            done = tf.cond(tf.greater(time, maximum_length),
                           lambda: tf.ones((batch_size,), dtype=tf.bool),
                           lambda: done)
            return (done, cell_state, next_input, cell_output, (next_grammar_state, next_output_state))
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
        self._beam_size = 10
        self._batch_size = batch_size
        
    def eval(self, session, save_to_file=False):
        inputs, input_lengths, labels, _ = self.data
        sequences = []
        gold_programs = set()
        correct_programs = [set() for _ in xrange(self._beam_size)]
        for gold in labels:
            try:
                gold = gold[:list(gold).index(self.grammar.end)]
            except ValueError:
                pass
            gold_programs.add(tuple(gold))
    
        dict_reverse = self.grammar.tokens

        ok_0 = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn = 0
        ok_ch = 0
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print "Writing decoded values to ", fp.name

        def get_functions(seq):
            return set(filter(lambda x: x.startswith('tt:') and not x.startswith('tt:param.'),
                              map(lambda x: self.grammar.tokens[x], seq)))

        try:
            for input_batch, input_length_batch, label_batch in get_minibatches([inputs, input_lengths, labels], self._batch_size):
                sequences = self.model.predict_on_batch(session, input_batch, input_length_batch)
                #print sequences.shape
                #print sequences

                for i, seq in enumerate(sequences):
                    print
                    gold = list(label_batch[i])
                    try:
                        gold = gold[:gold.index(self.grammar.end)]
                    except ValueError:
                        pass
                    print "GOLD:", ' '.join(dict_reverse[l] for l in gold)
                    gold_functions = get_functions(gold)
                    gold_channels = set(map(lambda x: x[x.index('.') + 1:], gold_functions))

                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        decoded = list(self.grammar.decode_output(beam))
                        try:
                            decoded = decoded[:decoded.index(self.grammar.end)]
                        except ValueError:
                            pass
                        print "TOP%d:"%(beam_pos), ' '.join(dict_reverse[l] for l in decoded)

                        decoded_tuple = tuple(decoded)

                        if beam_pos == 0 and save_to_file:
                            gold_str = ' '.join(dict_reverse[l] for l in gold)
                            decoded_str = ' '.join(dict_reverse[l] for l in decoded)
                            print >>fp, gold_str,  '\t',  decoded_str, '\t', (gold_str == decoded_str)

                        if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]:
                            ok_0[beam_pos] += 1

                        if beam_pos == 0:
                            decoded_functions = get_functions(decoded)
                            decoded_channels = set(map(lambda x:x[x.index('.')+1:], decoded_functions))
                            if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0] and gold_functions == decoded_functions:
                                ok_fn += 1
                            if gold_channels == decoded_channels:
                                ok_ch += 1
                        if self.grammar.compare(gold, decoded):
                            correct_programs[beam_pos].add(decoded_tuple)
                            ok_full[beam_pos] += 1
            print self.tag, "ok 0:", ok_0.astype(np.float32)/len(labels)
            print self.tag, "ok channel:", float(ok_ch)/len(labels)
            print self.tag, "ok function:", float(ok_fn)/len(labels)
            print self.tag, "ok full:", ok_full.astype(np.float32)/len(labels)
            print self.tag, "recall:", [float(len(p))/len(gold_programs) for p in correct_programs]
        finally:
            if fp is not None:
                fp.close()
        
        return ok_full.astype(np.float32)[0]/len(labels)
        
