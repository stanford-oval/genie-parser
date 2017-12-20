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
from sklearn.metrics.classification import confusion_matrix
'''
Created on Mar 16, 2017

@author: gcampagn
'''

import numpy as np

from grammar.abstract import AbstractGrammar
from .general_utils import get_minibatches, Progbar


class Seq2SeqEvaluator(object):
    '''
    Evaluate a sequence to sequence model on some data against some gold data
    '''

    def __init__(self, model, grammar : AbstractGrammar, data, tag : str, reverse_dictionary=None, beam_size=10, batch_size=256):
        self.model = model
        self.grammar = grammar
        self.data = data
        self.tag = tag
        self._reverse_dictionary = reverse_dictionary
        self._beam_size = beam_size
        self._batch_size = batch_size
        
    def compute_confusion_matrix(self, session):
        output_size = self.grammar.output_size
        print('output_size', output_size)
        confusion_matrix = np.zeros((output_size, output_size), dtype=np.int32)
        for i in range(output_size):
             confusion_matrix[i,i] += 1

        n_minibatches = 0
        total_n_minibatches = (len(self.data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        
        for data_batch in get_minibatches(self.data, self._batch_size):
            input_batch, input_length_batch, _, label_batch, _ = data_batch
            sequences, _ = self.model.eval_on_batch(session, *data_batch, batch_number=n_minibatches)
            n_minibatches += 1
    
            for i, beam in enumerate(sequences):
                gold = label_batch[i]
                prediction = beam[0] # top of the beam
                
                for i in range(len(gold)):
                    pred_action = prediction[i] if i < len(prediction) else 0 # pad
                    confusion_matrix[pred_action,gold[i]] += 1
            progbar.update(n_minibatches)
        return confusion_matrix
        
    def eval(self, session, save_to_file=False):
        inputs, input_lengths, parses, labels, label_length = self.data
        sequences = []
        sum_eval_loss = 0
        if save_to_file:
            gold_programs = set()
            correct_programs = [set() for _ in range(self._beam_size)]
            for gold in labels:
                gold = self.grammar.reconstruct_program(gold, ignore_errors=False)
                gold_programs.add(tuple(gold))
        else:
            gold_programs = set()
            correct_programs = None
    
        ok_0 = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn = np.zeros((self._beam_size,), dtype=np.int32)
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_functions(seq):
            return [x for x in seq if (x.startswith('tt:') or x.startswith('@'))]

        n_minibatches = 0
        total_n_minibatches = (len(self.data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
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
                    gold = self.grammar.reconstruct_program(label_batch[i], ignore_errors=False)
                    #print "GOLD:", ' '.join(gold)
                    gold_functions = get_functions(gold)

                    is_ok_0 = False
                    is_ok_fn = False
                    is_ok_full = False
                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        #self.grammar.normalize_sequence(decoded)
                        decoded = self.grammar.reconstruct_program(beam, ignore_errors=True)

                        if save_to_file:
                            decoded_tuple = tuple(decoded)
                        else:
                            decoded_tuple = None

                        if is_ok_0 or (len(decoded) > 0 and len(gold) > 0 and decoded[0] == gold[0]):
                            ok_0[beam_pos] += 1
                            is_ok_0 = True

                        decoded_functions = get_functions(decoded)
                        if is_ok_fn or (is_ok_0 and gold_functions == decoded_functions):
                            ok_fn[beam_pos] += 1
                            is_ok_fn = True

                        if beam_pos == 0 and save_to_file:
                            sentence = ' '.join(self._reverse_dictionary[x] for x in input_batch[i][:input_length_batch[i]])
                            gold_str = ' '.join(gold)
                            decoded_str = ' '.join(decoded)
                            #gold_str = ' '.join(gold_functions)
                            #decoded_str = ' '.join(decoded_functions)
                            print(sentence, gold_str, decoded_str, (gold_str == decoded_str), sep='\t', file=fp)

                        if is_ok_full or self.grammar.compare(gold, decoded):
                            if save_to_file:
                                correct_programs[beam_pos].add(decoded_tuple)
                            ok_full[beam_pos] += 1
                            is_ok_full = True
                progbar.update(n_minibatches)
            
            acc_0 = ok_0.astype(np.float32)/len(labels)
            acc_fn = ok_full.astype(np.float32)/len(labels)
            acc_full = ok_full.astype(np.float32)/len(labels)
            if save_to_file:
                recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            else:
                recall = [0]
            print(self.tag, "ok 0:", acc_0)
            print(self.tag, "ok function:", acc_fn)
            print(self.tag, "ok full:", acc_full)
            print(self.tag, "recall:", recall)
        finally:
            if fp is not None:
                fp.close()
        
        return acc_full[0], (sum_eval_loss / n_minibatches), acc_fn[0], recall[0] 
        
