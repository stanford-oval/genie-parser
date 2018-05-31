# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
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

import numpy as np
from collections import Counter

from grammar.abstract import AbstractGrammar
from models import Config
import os
import sys

########################################################################
# This needs to be run in a directory with thingpedia.json
# This needs to be run after t2t-datagen and t2t_decode.py (on t2t_test_x)
##########################################################################

class T2T_Evaluator(object):
    '''
    Evaluate a sequence to sequence model on some data against some gold data
    '''

    def __init__(self, grammar : AbstractGrammar, reverse_dictionary=None):
        self.grammar = grammar
        self._reverse_dictionary = reverse_dictionary
        self._beam_size = 1
        with open('/home/gcampagn/almond-nnparser/translation.tt', 'r') as seq:
            lines = seq.readlines()
            self.sequences = [line.strip() for line in lines]
        with open('/home/gcampagn/dataset/t2t_dir/t2t_test_x', 'r') as inputs:
            lines = inputs.readlines()
            self.input_sequences = [line.strip() for line in lines]
        with open('/home/gcampagn/dataset/t2t_dir/t2t_test_y', 'r') as labels:
            lines = labels.readlines()
            self.label_sequences = [line.strip() for line in lines]
        
    def eval(self, save_to_file=False):
        sequences = self.sequences
        label_sequences = self.label_sequences
        beam_pos = 0

        if save_to_file:
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
            fp = open("t2t_stats.txt", "w")
            print("Writing decoded values to ", fp.name)

        def get_devices(seq):
            return tuple(x for x in seq if x.startswith('@@'))
        def get_functions(seq):
            return tuple(x for x in seq if (x.startswith('@') and not x.startswith('@@')))
        def get_signature(seq):
            return [x for x in seq if (x.startswith('@') and not x.startswith('@@')) or x in ('now', 'monitor', 'timer', 'attimer', 'notify')]

        output_size = self.grammar.output_size[self.grammar.primary_output]
        gold_functions_counter = Counter()
        #all_functions = set(self.grammar.allfunctions)
        
        try:
            for i, seq in enumerate(sequences):
                gold = label_sequences[i]
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

                decoded = seq

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
                
                if beam_pos == 0 and save_to_file:
                    sentence = self.input_sequences[i]
                    gold_str = gold
                    decoded_str = decoded
                    print(sentence, gold_str, decoded_str, is_ok_full,
                          'CorrectGrammar' if is_ok_grammar else 'IncorrectGrammar',
                          'CorrectFunction' if is_ok_fn else 'IncorrectFunction',
                          'CorrectNumFunction' if is_ok_fn_count else 'IncorrectNumFunction',
                          'CorrectSignature' if is_ok_signature else 'IncorrectSignature',
                          sep='\t', file=fp)
                
            acc_grammar = ok_grammar.astype(np.float32)/len(sequences)
            acc_fn_count = ok_fn_count.astype(np.float32)/len(sequences)
            acc_device = ok_device.astype(np.float32)/len(sequences)
            acc_fn = ok_fn.astype(np.float32)/len(sequences)
            acc_signature = ok_signature.astype(np.float32)/len(sequences)
            acc_full = ok_full.astype(np.float32)/len(sequences)
            if save_to_file:
                recall = [float(len(p))/len(gold_programs) for p in correct_programs]
            else:
                recall = [0]
                
            print("ok grammar:", acc_grammar)
            print("ok function count:", acc_fn_count)
            print("ok device:", acc_device)
            print("ok function:", acc_fn)
            print("ok signature:", acc_signature)
            print("ok full:", acc_full)
            print("program recall:", recall)
                
            metrics = {
                'grammar_accuracy': float(acc_grammar[0]),
                'accuracy': float(acc_full[0]),
                'function_count_accuracy': float(acc_fn_count[0]),
                'device_accuracy': float(acc_device[0]),
                'function_accuracy': float(acc_fn[0]),
                'signature_accuracy': float(acc_signature[0]),
                'program_recall': float(recall[0]),
            }
            return metrics
        finally:
            if fp is not None:
                fp.close()


config = Config.load([os.path.join(sys.argv[1], 'model.conf')])
evaluator = T2T_Evaluator(config.grammar, config.reverse_dictionary)
print(evaluator.eval(True))
