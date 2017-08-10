#!/usr/bin/python3

import sys
import numpy as np
import itertools

from grammar.thingtalk import ThingtalkGrammar

grammar = ThingtalkGrammar('./thingpedia.txt')

only_functions= False
if len(sys.argv) > 3 and sys.argv[3] == 'functions':
    only_functions = True
    output_size = grammar.num_functions
    function_offset = grammar.num_control_tokens + grammar.num_begin_tokens
else:
    output_size = grammar.output_size

# predicted vs gold
confusion_matrix = np.zeros((output_size, output_size), dtype=np.int32)

with open(sys.argv[2]) as fp:
    for line in fp:
        sentence, gold, predicted, _ = line.strip().split('\t')
        gold = (gold + ' <<EOS>>').split(' ')
        predicted = (predicted + ' <<EOS>>').split(' ')
        if only_functions:
            gold = (x for x in gold if x.startswith('tt:') and not x.startswith('tt:root.'))
            predicted = (x for x in predicted if x.startswith('tt:') and not x.startswith('tt:root.'))
        for gold_token, predicted_token in itertools.zip_longest(gold, predicted, fillvalue='<<PAD>>'):
            gold_id = grammar.dictionary[gold_token]
            predicted_id = grammar.dictionary[predicted_token]
            if only_functions:
                gold_id -= function_offset
                predicted_id -= function_offset
            try:
                confusion_matrix[predicted_id, gold_id] += 1
            except:
                raise ValueError(gold_token + ' ' + predicted_token)

# precision: sum over columns (% of the sentences where this token was predicted
# in which it was actually meant to be there)
# recall: sum over rows (% of the sentences where this token was meant
# to be there in which it was actually predicted)
#
# see "A systematic analysis of performance measures for classification tasks"
# MarinaSokolova, GuyLapalme, Information Processing & Management, 2009
precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)
f1 = 2 * (precision * recall) / (precision + recall)

with open('./f1-functions.tsv' if only_functions else './f1-tokens.tsv', 'w') as out:
    for i in range(output_size):
        if only_functions:
            token_id = i + function_offset
        else:
            token_id = i
        print(grammar.tokens[token_id], precision[i], recall[i], f1[i], sep='\t', file=out)

overall_precision = np.nanmean(precision)
overall_recall = np.nanmean(recall)
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
print(overall_precision, overall_recall, overall_f1)
