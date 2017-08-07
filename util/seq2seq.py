'''
Created on Mar 16, 2017

@author: gcampagn
'''

import tensorflow as tf
import numpy as np

from .general_utils import get_minibatches


class Seq2SeqEvaluator(object):
    '''
    Evaluate a sequence to sequence model on some data against some gold data
    '''

    def __init__(self, model, grammar, data, tag, reverse_dictionary=None, beam_size=10, batch_size=256):
        self.model = model
        self.grammar = grammar
        self.data = data
        self.tag = tag
        self._reverse_dictionary = reverse_dictionary
        self._beam_size = beam_size
        self._batch_size = batch_size
        
    def eval(self, session, save_to_file=False):
        inputs, input_lengths, parses, labels, label_length = self.data
        sequences = []
        sum_eval_loss = 0
        gold_programs = set()
        correct_programs = [set() for _ in range(self._beam_size)]
        for gold in labels:
            try:
                gold = gold[:list(gold).index(self.grammar.end)]
            except ValueError:
                pass
            gold_programs.add(tuple(gold))
    
        dict_reverse = self.grammar.tokens

        ok_0 = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn = 0
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_functions(seq):
            return set(x for x in (self.grammar.tokens[x] for x in seq) if x.startswith('tt:') and not x.startswith('tt:param.'))

        n_minibatches = 0
        try:
            for input_batch, input_length_batch, parse_batch, label_batch, label_length_batch in get_minibatches([inputs, input_lengths, parses, labels, label_length], self._batch_size):
                sequences, eval_loss = self.model.eval_on_batch(session, input_batch, input_length_batch, parse_batch, label_batch, label_length_batch, batch_number=n_minibatches)
                sum_eval_loss += eval_loss
                n_minibatches += 1
                #print sequences.shape
                #print sequences

                for i, seq in enumerate(sequences):
                    #print
                    gold = list(label_batch[i])
                    try:
                        gold = gold[:gold.index(self.grammar.end)]
                    except ValueError:
                        pass
                    #print "GOLD:", ' '.join(dict_reverse[l] for l in gold)
                    gold_functions = get_functions(gold)

                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        decoded = list(filter(lambda x: x != 0, beam))
                        try:
                            decoded = decoded[:decoded.index(self.grammar.end)]
                        except ValueError:
                            pass

                        decoded_tuple = tuple(decoded)

                        if beam_pos == 0 and save_to_file:
                            sentence = ' '.join(self._reverse_dictionary[x] for x in input_batch[i][:input_length_batch[i]])
                            gold_str = ' '.join(dict_reverse[l] for l in gold)
                            decoded_str = ' '.join(dict_reverse[l] for l in decoded)
                            print(sentence, gold_str, decoded_str, (gold_str == decoded_str), sep='\t', file=fp)

                        if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]:
                            ok_0[beam_pos] += 1

                        if beam_pos == 0:
                            decoded_functions = get_functions(decoded)
                            if len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0] and gold_functions == decoded_functions:
                                ok_fn += 1
                        if self.grammar.compare(gold, decoded):
                            correct_programs[beam_pos].add(decoded_tuple)
                            ok_full[beam_pos] += 1
            
            acc_0 = ok_0.astype(np.float32)/len(labels)
            acc_fn = float(ok_fn)/len(labels)
            acc_full = ok_full.astype(np.float32)/len(labels)
            recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            print(self.tag, "ok 0:", acc_0)
            print(self.tag, "ok function:", acc_fn)
            print(self.tag, "ok full:", acc_full)
            print(self.tag, "recall:", recall)
        finally:
            if fp is not None:
                fp.close()
        
        return acc_full[0], (sum_eval_loss / n_minibatches), acc_fn, recall[0] 
        
