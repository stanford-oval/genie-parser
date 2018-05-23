#!/usr/bin/python3
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
Created on Apr 30, 2018

@author: gcampagn
'''

import sys
from grammar.thingtalk import ThingTalkGrammar
import csv

FILTERS = {'==', '>=', '<=', '=~', '~=', 'starts_with', 'ends_with'}
OTHER_VALUES = {'start_of', 'end_of', 'NUMBER_0', 'NUMBER_1', 'NUMBER_2', 'NUMBER_3', 'NUMBER_4'}

def get_devices(x):
    return tuple(t.rsplit('.',maxsplit=1)[0] for t in x if t.startswith('@'))
def get_non_op_values(sequence):
    return tuple(t for i,t in enumerate(sequence) if t not in FILTERS and t != 'not' and sequence[i-1] != '=' and sequence[i-1] not in FILTERS and (i== 0 or sequence[i-1] not in OTHER_VALUES))
def get_non_values(sequence):
    return tuple(t for i,t in enumerate(sequence) if sequence[i-1] != '=' and sequence[i-1] not in FILTERS and (i==0 or sequence[i-1] not in OTHER_VALUES))

def get_parameters_or_filters(sequence):
    for i,token in enumerate(sequence):
        if token.startswith('param:'):
            is_filter = i+1 < len(sequence) and sequence[i+1] in ('==', '>=', '<=', '=~', '~=', 'starts_with', 'ends_with')
            is_param = sequence[i-1] != '='
            if is_filter or is_param:
                yield token

def is_subset(one, two):
    for x in one:
        if not x in two:
            return False
    return True

def find_error_token(gold, predicted):
    if len(gold) != len(predicted):
        return None
    error_token = None
    for gt, pt in zip(gold, predicted):
        if gt != pt:
            if error_token is None:
                error_token = gt
            else:
                return None
    return error_token

def main():
    writer = csv.DictWriter(sys.stdout, ('sentence_length', 'gold_length', 'gold_num_prod',
                                         'num_total_params', 'num_total_filters', 'num_params', 'num_filters',
                                         'ok', 'ok_grammar', 'ok_device', 'ok_function', 'ok_fn_count', 'ok_signature',
                                         'ok_non_op_value', 'ok_non_value', 'ok_param_subset', 'param_strict_subset', 'ok_rearrange',
                                         'single_token_error', 'single_token_unit', 'single_token_constant', 'single_token_enum', 'single_token_pp'))
    writer.writeheader()
    grammar = ThingTalkGrammar(sys.argv[1], quiet=True)
    device = '@' + sys.argv[2] if len(sys.argv) >= 3 else None
    
    for line in sys.stdin:
        sentence, gold, predicted, ok, ok_grammar, ok_function, ok_fn_count, ok_signature = line.strip().split('\t')
        assert ok in ('True', 'False')
        assert ok_grammar in ('CorrectGrammar', 'IncorrectGrammar')
        assert ok_function in ('CorrectFunction', 'IncorrectFunction')
        assert ok_fn_count in ('CorrectNumFunction', 'IncorrectNumFunction')
        assert ok_signature in ('CorrectSignature', 'IncorrectSignature')

        gold = gold.split(' ')
        predicted = predicted.split(' ')
        ok_device = ok_grammar == 'CorrectGrammar' and get_devices(gold) == get_devices(predicted)
        ok_non_op_value = ok_signature == 'CorrectSignature' and get_non_op_values(gold) == get_non_op_values(predicted)
        ok_non_value = ok_signature == 'CorrectSignature' and get_non_values(gold) == get_non_values(predicted)

        p1, p2 = set(get_parameters_or_filters(predicted)), set(get_parameters_or_filters(gold))
        ok_param_subset = ok_signature == 'CorrectSignature' and is_subset(p1, p2)
        param_strict_subset = ok_param_subset and len(p1) < len(p2)

        ok_rearrange = ok_signature == 'CorrectSignature' and list(sorted(gold)) == list(sorted(predicted))

        #assert ok == 'False' or ok_rearrange
        #assert not ok_rearrange or ok_non_value, (' '.join(gold), ' '.join(predicted), ok, ok_signature)
        #assert not ok_non_value or ok_non_op_value
        #assert not ok_non_op_value or ok_param_subset

        single_token_error = False
        single_token_unit = False
        single_token_constant = False
        single_token_pp = False
        single_token_enum = False
        if ok_non_value and ok == 'False':
            error_token = find_error_token(gold, predicted)
            if error_token is not None:
                #print(' '.join(gold), ' '.join(predicted), sep='\t', file=sys.stderr)
                single_token_error = True
                single_token_enum = error_token.startswith('enum:')
                single_token_unit = error_token.startswith('unit:')
                single_token_pp = error_token.startswith('param:')
                single_token_constant = error_token[0].isupper()

        num_total_params = 0
        num_total_filters = 0
        num_params = 0
        num_filters = 0
        in_device = device is None
        in_filter = False

        param_comb = []
        for i,token in enumerate(gold):
            if device is not None:
                if token.startswith(device):
                    in_device = True
                elif token.startswith('@'):
                    in_device = False
            if token.startswith('param:'):
                if in_device:
                    param_comb.append(token)

                is_filter = i+1 < len(gold) and gold[i+1] in ('==', '>=', '<=', '=~', '~=', 'starts_with', 'ends_with')
                is_param = gold[i-1] != '='
                if is_filter:
                    num_total_filters += 1
                    if in_device:
                        num_filters += 1
                elif is_param:
                    num_total_params += 1
                    if in_device:
                        num_params += 1

        #if not ok_non_op_value and ok == 'False' and ok_rearrange: #and not param_strict_subset:
        #    print(' '.join(gold), ' '.join(predicted), sep='\t', file=sys.stderr)
        #if ok_signature == 'CorrectSignature' and not ok_param_subset:
        #    print(sentence, ' '.join(gold), ' '.join(predicted), sep='\t', file=sys.stderr)
        #if ok_param_subset and not ok_non_op_value:
        #if ok_non_op_value and ok == 'False':
        #    print(sentence, ' '.join(gold), ' '.join(predicted), sep='\t', file=sys.stderr)

        sentence = sentence.split(' ')[1:-1] # remove <s> and </s>
        vector, length = grammar.vectorize_program(sentence, gold, max_length=60)
        
        writer.writerow({
            'sentence_length': len(sentence),
            'gold_length': len(gold),
            'gold_num_prod': length,
            'num_total_params': num_total_params,
            'num_total_filters': num_total_filters,
            'num_params': num_params,
            'num_filters': num_filters,
            'ok': 1 if ok == 'True' else 0,
            'ok_grammar': 1 if ok_grammar == 'CorrectGrammar' else 0,
            'ok_device': 1 if ok_device else 0,
            'ok_function': 1 if ok_function == 'CorrectFunction' else 0,
            'ok_fn_count': 1 if ok_fn_count == 'CorrectNumFunction' else 0,
            'ok_signature': 1 if ok_signature == 'CorrectSignature' else 0,
            'ok_param_subset': 1 if ok_param_subset else 0,
            'ok_non_op_value': 1 if ok_non_op_value else 0,
            'ok_non_value': 1 if ok_non_value else 0,
            'ok_rearrange': 1 if ok_rearrange else 0,
            'param_strict_subset': 1 if param_strict_subset else 0,
            'single_token_error': 1 if single_token_error else 0,
            'single_token_unit': 1 if single_token_unit else 0,
            'single_token_enum': 1 if single_token_enum else 0,
            'single_token_constant': 1 if single_token_constant else 0,
            'single_token_pp': 1 if single_token_pp else 0,
        })

if __name__ == '__main__':
    main()
