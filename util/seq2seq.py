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
        ok_fn = np.zeros((self._beam_size,), dtype=np.int32)
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_functions(seq):
            return [x for x in (self.grammar.tokens[x] for x in seq) if x.startswith('tt:')]

        n_minibatches = 0
        try:
            for data_batch in get_minibatches(self.data, self._batch_size):
                input_batch, input_length_batch, _, label_batch, _ = data_batch
                sequences, eval_loss = self.model.eval_on_batch(session, *data_batch, batch_number=n_minibatches)
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

                    is_ok_0 = False
                    is_ok_fn = False
                    is_ok_full = False
                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        decoded = list(filter(lambda x: x != 0, beam))
                        try:
                            decoded = decoded[:decoded.index(self.grammar.end)]
                        except ValueError:
                            pass
                        #self.grammar.normalize_sequence(decoded)

                        decoded_tuple = tuple(decoded)

                        if is_ok_0 or (len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]):
                            ok_0[beam_pos] += 1
                            is_ok_0 = True

                        decoded_functions = get_functions(decoded)
                        if is_ok_fn or (is_ok_0 and gold_functions == decoded_functions):
                            ok_fn[beam_pos] += 1
                            is_ok_fn = True

                        if beam_pos == 0 and save_to_file:
                            sentence = ' '.join(self._reverse_dictionary[x] for x in input_batch[i][:input_length_batch[i]])
                            gold_str = ' '.join(dict_reverse[l] for l in gold)
                            decoded_str = ' '.join(dict_reverse[l] for l in decoded)
                            #gold_str = ' '.join(gold_functions)
                            #decoded_str = ' '.join(decoded_functions)
                            print(sentence, gold_str, decoded_str, (gold_str == decoded_str), sep='\t', file=fp)

                        if is_ok_full or self.grammar.compare(gold, decoded):
                            correct_programs[beam_pos].add(decoded_tuple)
                            ok_full[beam_pos] += 1
                            is_ok_full = True
            
            acc_0 = ok_0.astype(np.float32)/len(labels)
            acc_fn = ok_full.astype(np.float32)/len(labels)
            acc_full = ok_full.astype(np.float32)/len(labels)
            recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            print(self.tag, "ok 0:", acc_0)
            print(self.tag, "ok function:", acc_fn)
            print(self.tag, "ok full:", acc_full)
            print(self.tag, "recall:", recall)
        finally:
            if fp is not None:
                fp.close()
        
        return acc_full[0], (sum_eval_loss / n_minibatches), acc_fn[0], recall[0] 
        
