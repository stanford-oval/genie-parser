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
        
    def eval(self, session, save_to_file=False):
        sequences = []
        sum_eval_loss = 0
        _, _, _, label_sequences, _, _ = self.data
        
        if save_to_file:
            gold_programs = set()
            correct_programs = [set() for _ in range(self._beam_size)]
            for label in label_sequences:
                gold_programs.add(tuple(label))
        else:
            gold_programs = set()
            correct_programs = None
    
        ok_grammar = np.zeros((self._beam_size,), dtype=np.int32)
        ok_fn = np.zeros((self._beam_size,), dtype=np.int32)
        ok_full = np.zeros((self._beam_size,), dtype=np.int32)
        fp = None
        if save_to_file:
            fp = open("stats_" + self.tag + ".txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_functions(seq):
            return [x for x in seq if (x.startswith('tt:') or x.startswith('@'))]

        output_size = self.grammar.output_size[self.grammar.primary_output]
        confusion_matrix = np.zeros((output_size, output_size), dtype=np.int32)
        action_count_tp = np.zeros((output_size,), dtype=np.int32)
        action_count_fp = np.zeros((output_size,), dtype=np.int32)
        action_count_tn = np.zeros((output_size,), dtype=np.int32)
        action_count_fn = np.zeros((output_size,), dtype=np.int32)
        
        n_minibatches = 0
        total_n_minibatches = (len(self.data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        try:
            for data_batch in get_minibatches(self.data, self._batch_size, shuffle=False):
                input_batch, input_length_batch, _, _, label_batch, label_length_batch = data_batch
                
                if self.model.action_counts is not None:
                    feed = self.model.create_feed_dict(*data_batch, batch_number=n_minibatches)
                    sequences, action_counts, eval_loss = session.run([self.model.pred, self.model.action_counts, self.model.eval_loss], feed_dict=feed)
                    
                    label_action_counts = np.zeros((len(label_batch), output_size), dtype=np.int32)
                    for i in range(len(label_batch[self.grammar.primary_output])):
                        label_action_counts[i] = np.bincount(label_batch[self.grammar.primary_output][i, :label_length_batch[i]],
                                                             minlength=output_size)
                else:
                    sequences, eval_loss = self.model.eval_on_batch(session, *data_batch, batch_number=n_minibatches)
                    action_counts = None
                    label_action_counts = None
                sum_eval_loss += eval_loss
                
                if action_counts is not None:
                    binarized_action_counts = action_counts >= 0.5
                    label_action_counts = label_action_counts >= 1
                    true_positives = np.sum(np.logical_and(binarized_action_counts, label_action_counts), axis=0)
                    false_positives = np.sum(np.logical_and(binarized_action_counts, np.logical_not(label_action_counts)), axis=0)
                    true_negatives = np.sum(np.logical_and(np.logical_not(binarized_action_counts), np.logical_not(label_action_counts)), axis=0)
                    false_negatives = np.sum(np.logical_and(np.logical_not(binarized_action_counts), label_action_counts), axis=0)
                    action_count_tp += true_positives
                    action_count_fp += false_positives
                    action_count_tn += true_negatives
                    action_count_fn += false_negatives

                primary_sequences = sequences[self.grammar.primary_output]
                primary_label_batch = label_batch[self.grammar.primary_output]

                for i, seq in enumerate(primary_sequences):
                    gold = label_sequences[n_minibatches * self._batch_size + i]
                    gold_functions = get_functions(gold)

                    is_ok_grammar = False
                    is_ok_fn = False
                    is_ok_full = False
                    for beam_pos, beam in enumerate(seq):
                        if beam_pos >= self._beam_size:
                            break
                        
                        decoded_vectors = dict()
                        for key in self.grammar.output_size:
                            decoded_vectors[key] = sequences[key][i,beam_pos]
                        decoded = self.grammar.reconstruct_program(input_batch[i], decoded_vectors, ignore_errors=True)

                        if save_to_file:
                            decoded_tuple = tuple(decoded)
                        else:
                            decoded_tuple = None
                        
                        if is_ok_grammar or len(decoded) > 0:
                            ok_grammar[beam_pos] += 1
                            is_ok_grammar = True

                        decoded_functions = get_functions(decoded)
                        if is_ok_fn or (is_ok_grammar and gold_functions == decoded_functions):
                            ok_fn[beam_pos] += 1
                            is_ok_fn = True
                            
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
                            sentence = ' '.join(self._reverse_dictionary[x] for x in input_batch[i][:input_length_batch[i]])
                            gold_str = ' '.join(gold)
                            decoded_str = ' '.join(decoded)
                            print(sentence, gold_str, decoded_str, is_ok_full,
                                  'CorrectGrammar' if is_ok_grammar else 'IncorrectGrammar',
                                  'CorrectFunction' if is_ok_fn else 'IncorrectFunction',
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
            
            parse_action_precision = np.ma.masked_invalid(parse_action_precision)
            parse_action_recall = np.ma.masked_invalid(parse_action_recall)
            
            overall_parse_action_precision = np.power(np.prod(parse_action_precision, dtype=np.float64), 1/parse_action_precision.count())
            overall_parse_action_recall = np.power(np.prod(parse_action_recall, dtype=np.float64), 1/parse_action_recall.count())
            
            # avoid division by 0
            if np.abs(overall_parse_action_precision + overall_parse_action_recall) < 1e-6:
                overall_parse_action_f1 = 0
            else:
                overall_parse_action_f1 = 2 * (overall_parse_action_precision * overall_parse_action_recall) / \
                    (overall_parse_action_precision + overall_parse_action_recall)
            
            acc_grammar = ok_grammar.astype(np.float32)/len(self.data[0])
            acc_fn = ok_fn.astype(np.float32)/len(self.data[0])
            acc_full = ok_full.astype(np.float32)/len(self.data[0])
            if save_to_file:
                recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            else:
                recall = [0]
                
            print(self.tag, "ok grammar:", acc_grammar)
            print(self.tag, "ok function:", acc_fn)
            print(self.tag, "ok full:", acc_full)
            print(self.tag, "program recall:", recall)
            print(self.tag, "parse-action avg precision:", overall_parse_action_precision, "over %d actions" % parse_action_precision.count())
            print(self.tag, "parse-action avg recall:", overall_parse_action_recall, "over %d actions" % parse_action_recall.count())
            print(self.tag, "parse-action min precision:", np.min(parse_action_precision))
            print(self.tag, "parse-action min recall:", np.min(parse_action_recall))
            print(self.tag, "parse-action F1:", overall_parse_action_f1)
            if self.model.action_counts is not None:
                action_count_tp = np.ma.asarray(action_count_tp)
                action_count_tn = np.ma.asarray(action_count_tn)
                action_count_fp = np.ma.asarray(action_count_fp)
                action_count_fn = np.ma.asarray(action_count_fn)
                
                action_count_precision = action_count_tp / (action_count_tp + action_count_fp)
                action_count_recall = action_count_tp / (action_count_tp + action_count_fn)
                if save_to_file:
                    action_count_f1 = 2 * (action_count_precision * action_count_recall) / (action_count_precision + action_count_recall)
                    with open(self.tag + '-action-count-f1.tsv', 'w') as out:
                        for i in range(output_size):
                            print(i, action_count_precision[i], action_count_recall[i], action_count_f1[i], sep='\t', file=out)
                
                action_count_precision = np.ma.masked_invalid(action_count_precision)
                action_count_avg_precision = np.power(np.prod(action_count_precision, dtype=np.float64), 1/action_count_precision.count())
                action_count_recall = np.ma.masked_invalid(action_count_recall)
                action_count_avg_recall = np.power(np.prod(action_count_recall, dtype=np.float64), 1/action_count_recall.count())
                print(self.tag, "action-count avg precision:", action_count_avg_precision, "over %d actions" % action_count_precision.count())
                print(self.tag, "action-count avg recall:", action_count_avg_recall, "over %d actions" % action_count_recall.count())
                print(self.tag, "action-count min precision:", np.min(action_count_precision))
                print(self.tag, "action-count min recall:", np.min(action_count_recall))
                
                # avoid division by 0
                if np.abs(action_count_avg_precision + action_count_avg_recall) < 1e-6:
                    action_count_f1 = 0
                else:
                    action_count_f1 = 2 * (action_count_avg_precision * action_count_avg_recall) / \
                        (action_count_avg_precision + action_count_avg_recall)
                print(self.tag, "action-count F1:", action_count_f1)
            else:
                action_count_avg_precision = 0
                action_count_avg_recall = 0
                action_count_f1 = 0
                
            metrics = {
                'eval_loss': (sum_eval_loss / n_minibatches),
                'grammar_accuracy': acc_grammar[0],
                'accuracy': acc_full[0],
                'function_accuracy': acc_fn[0],
                'program_recall': recall[0],
                'parse_action_precision': overall_parse_action_precision,
                'parse_action_recall': overall_parse_action_recall,
                'parse_action_f1': overall_parse_action_f1,
                'action_count_precision': action_count_avg_precision,
                'action_count_recall': action_count_avg_recall,
                'action_count_f1': action_count_f1
            }
            return metrics
        finally:
            if fp is not None:
                fp.close()
