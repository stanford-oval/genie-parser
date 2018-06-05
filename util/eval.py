# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
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

import numpy as np
from collections import Counter

from util.loader import Dataset
from grammar.abstract import AbstractGrammar
from .general_utils import get_minibatches, Progbar


class Seq2SeqEvaluator(object):
    '''
    Evaluate a sequence to sequence model on some data against some gold data
    '''

    def __init__(self, model, grammar : AbstractGrammar, data : Dataset, tag : str, reverse_dictionary=None, beam_size=10, batch_size=256):
        self.model = model
        self.grammar = grammar
        self.data = data
        self.tag = tag
        self._reverse_dictionary = reverse_dictionary
        self._beam_size = beam_size
        self._batch_size = batch_size
        
    def eval(self, session, save_to_file=False):
        sum_eval_loss = 0

        if save_to_file:
            label_sequences = self.data.label_sequences
            gold_programs = set()
            correct_programs = [set() for _ in range(self._beam_size)]
            for label in label_sequences:
                gold_programs.add(tuple(label))
        else:
            gold_programs = set()
            correct_programs = None
    
        ok_grammar = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn_count = np.zeros((self._beam_size,), dtype=np.int32)
        ok_device = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn = np.zeros((self._beam_size,), dtype=np.int32)
        ok_signature = np.zeros((self._beam_size,), dtype=np.int32)
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_devices(seq):
            return tuple(x for x in seq if x.startswith('@@'))
        def get_functions(seq):
            return tuple(x for x in seq if (x.startswith('@') and not x.startswith('@@')))
        def get_signature(seq):
            return [x for x in seq if (x.startswith('@') and not x.startswith('@@')) or x in ('now', 'monitor', 'timer', 'attimer', 'notify')]

        output_size = self.grammar.output_size[self.grammar.primary_output]
        confusion_matrix = np.zeros((output_size, output_size), dtype=np.int32)
        function_tp = Counter()
        function_fp = Counter()
        #function_tn = Counter()
        function_fn = Counter()
        gold_functions_counter = Counter()
        #all_functions = set(self.grammar.allfunctions)
        
        n_minibatches = 0
        total_n_minibatches = (len(self.data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        try:
            for data_batch in get_minibatches(self.data, self._batch_size, shuffle=False):
                predicted_sequences, eval_loss = self.model.eval_on_batch(session, data_batch, batch_number=n_minibatches)
                sum_eval_loss += eval_loss
                
                primary_sequences = predicted_sequences[self.grammar.primary_output]
                primary_label_batch = data_batch.label_vectors[self.grammar.primary_output]

                for i, seq in enumerate(primary_sequences):
                    gold = data_batch.label_sequences[i]
                    gold_devices = get_devices(gold)
                    gold_functions = get_functions(gold)
                    gold_function_set = set(gold_functions)
                    gold_functions_counter.update(gold_functions)
                    gold_signature = get_signature(gold)

                    is_ok_grammar = False
                    is_ok_fn_count = False
                    is_ok_device = False
                    is_ok_fn = False
                    is_ok_signature = False
                    is_ok_full = False
                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        
                        decoded_vectors = dict()
                        for key in self.grammar.output_size:
                            decoded_vectors[key] = predicted_sequences[key][i,beam_pos]
                        decoded = self.grammar.reconstruct_program(data_batch.input_sequences[i], decoded_vectors, ignore_errors=True)

                        if save_to_file:
                            decoded_tuple = tuple(decoded)
                        else:
                            decoded_tuple = None
                        
                        if is_ok_grammar or len(decoded) > 0:
                            ok_grammar[beam_pos] += 1
                            is_ok_grammar = True

                        decoded_devices = get_devices(decoded)
                        decoded_functions = get_functions(decoded)
                        decoded_function_set = set(decoded_functions)
                        if save_to_file:
                            function_tp.update(gold_function_set & decoded_function_set)
                            function_fp.update(decoded_function_set - gold_function_set)
                            function_fn.update(gold_function_set - decoded_function_set)
                            #function_tn.update(all_functions - (gold_functions | decoded_functions))
                        decoded_signature = get_signature(decoded)

                        if is_ok_fn_count or (is_ok_grammar and len(gold_functions) == len(decoded_functions)):
                            ok_fn_count[beam_pos] += 1
                            is_ok_fn_count = True
                        
                        if is_ok_device or (is_ok_grammar and gold_devices == decoded_devices):
                            ok_device[beam_pos] += 1
                            is_ok_device = True

                        if is_ok_fn or (is_ok_grammar and gold_functions == decoded_functions):
                            ok_fn[beam_pos] += 1
                            is_ok_fn = True

                        if is_ok_signature or (is_ok_grammar and gold_signature == decoded_signature):
                            ok_signature[beam_pos] += 1
                            is_ok_signature = True

                        if is_ok_full or (is_ok_grammar and self.grammar.compare(gold, decoded)):
                            if save_to_file:
                                correct_programs[beam_pos].add(decoded_tuple)
                            ok_full[beam_pos] += 1
                            is_ok_full = True
                        
                        if beam_pos == 0:
                            length_diff = len(primary_label_batch[i]) - len(beam)
                            if length_diff > 0:
                                padded_pred = np.concatenate((beam, np.zeros((length_diff,), np.int32)), axis=0)
                            else:
                                padded_pred = beam
                            confusion_matrix[padded_pred,primary_label_batch[i]] += 1

                        if beam_pos == 0 and save_to_file:
                            sentence = ' '.join(data_batch.input_sequences[i])
                            gold_str = ' '.join(gold)
                            decoded_str = ' '.join(decoded)
                            print(sentence, gold_str, decoded_str, is_ok_full,
                                  'CorrectGrammar' if is_ok_grammar else 'IncorrectGrammar',
                                  'CorrectFunction' if is_ok_fn else 'IncorrectFunction',
                                  'CorrectNumFunction' if is_ok_fn_count else 'IncorrectNumFunction',
                                  'CorrectSignature' if is_ok_signature else 'IncorrectSignature',
                                  sep='\t', file=fp)
                
                n_minibatches += 1
                example_counter = n_minibatches * self._batch_size
                progbar.update(n_minibatches, values=[('loss', eval_loss)],
                               exact=[('accuracy', ok_full[0]/example_counter),
                                      ('function accuracy', ok_fn[0]/example_counter)])
            
            # precision: sum over columns (% of the sentences where this token was predicted
            # in which it was actually meant to be there)
            # recall: sum over rows (% of the sentences where this token was meant
            # to be there in which it was actually predicted)
            #
            # see "A systematic analysis of performance measures for classification tasks"
            # MarinaSokolova, GuyLapalme, Information Processing & Management, 2009
            confusion_matrix = np.ma.asarray(confusion_matrix)
            
            parse_action_precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
            parse_action_recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        
            if save_to_file:
                parse_action_f1 = 2 * (parse_action_precision * parse_action_recall) / (parse_action_precision + parse_action_recall)
                with open(self.tag + '-f1.tsv', 'w') as out:
                    for i in range(output_size):
                        print(i, parse_action_precision[i], parse_action_recall[i], parse_action_f1[i], sep='\t', file=out)
                        
                with open(self.tag + '-function-f1.tsv', 'w') as out:
                    all_f1s = []
                    for f in self.grammar.allfunctions:
                        if gold_functions_counter[f] == 0:
                            continue
                        function_precision = function_tp[f] / (function_tp[f] + function_fp[f] + 1e-5)
                        function_recall = function_tp[f] / (function_tp[f] + function_fn[f])
                        function_f1 = 2* (function_precision * function_recall) / (function_precision + function_recall + 1e-5)
                        all_f1s.append(function_f1)
                        print(f, gold_functions_counter[f], function_precision, function_recall, function_f1, sep='\t', file=out)
                    all_f1s = np.array(all_f1s)
                    print(self.tag, 'function f1', np.average(all_f1s))#np.power(np.prod(all_f1s, dtype=np.float64), 1/len(all_f1s)))
            
            parse_action_precision = np.ma.masked_invalid(parse_action_precision)
            parse_action_recall = np.ma.masked_invalid(parse_action_recall)
            
            overall_parse_action_precision = np.mean(parse_action_precision, dtype=np.float64)
            overall_parse_action_recall = np.mean(parse_action_recall, dtype=np.float64)
            
            # avoid division by 0
            if np.abs(overall_parse_action_precision + overall_parse_action_recall) < 1e-6:
                overall_parse_action_f1 = 0
            else:
                overall_parse_action_f1 = 2 * (overall_parse_action_precision * overall_parse_action_recall) / \
                    (overall_parse_action_precision + overall_parse_action_recall)
            
            acc_grammar = ok_grammar.astype(np.float32)/len(self.data[0])
            acc_fn_count = ok_fn_count.astype(np.float32)/len(self.data[0])
            acc_device = ok_device.astype(np.float32)/len(self.data[0])
            acc_fn = ok_fn.astype(np.float32)/len(self.data[0])
            acc_signature = ok_signature.astype(np.float32)/len(self.data[0])
            acc_full = ok_full.astype(np.float32)/len(self.data[0])
            if save_to_file:
                recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            else:
                recall = [0]
                
            print(self.tag, "ok grammar:", acc_grammar)
            print(self.tag, "ok function count:", acc_fn_count)
            print(self.tag, "ok device:", acc_device)
            print(self.tag, "ok function:", acc_fn)
            print(self.tag, "ok signature:", acc_signature)
            print(self.tag, "ok full:", acc_full)
            print(self.tag, "program recall:", recall)
            print(self.tag, "parse-action avg precision:", overall_parse_action_precision, "over %d actions" % parse_action_precision.count())
            print(self.tag, "parse-action avg recall:", overall_parse_action_recall, "over %d actions" % parse_action_recall.count())
            print(self.tag, "parse-action min precision:", np.min(parse_action_precision))
            print(self.tag, "parse-action min recall:", np.min(parse_action_recall))
            print(self.tag, "parse-action F1:", overall_parse_action_f1)
                
            metrics = {
                'eval_loss': (sum_eval_loss / n_minibatches),
                'grammar_accuracy': float(acc_grammar[0]),
                'accuracy': float(acc_full[0]),
                'function_count_accuracy': float(acc_fn_count[0]),
                'device_accuracy': float(acc_device[0]),
                'function_accuracy': float(acc_fn[0]),
                'signature_accuracy': float(acc_signature[0]),
                'program_recall': float(recall[0]),
                'parse_action_precision': overall_parse_action_precision,
                'parse_action_recall': overall_parse_action_recall,
                'parse_action_f1': overall_parse_action_f1,
            }
            return metrics
        finally:
            if fp is not None:
                fp.close()
