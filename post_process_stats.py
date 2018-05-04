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

def main():
    writer = csv.DictWriter(sys.stdout, ('sentence_length', 'gold_length', 'gold_num_prod', 'ok',
                                         'ok_grammar', 'ok_function', 'ok_fn_count'))
    grammar = ThingTalkGrammar(sys.argv[1])
    
    for line in sys.stdin:
        sentence, gold, predicted, ok, ok_grammar, ok_function, ok_fn_count = line.strip().split('\t')
        gold = gold.split(' ')
        vector, length = grammar.vectorize_program(gold, max_length=60)
        
        writer.writerow({
            'sentence_length': len(sentence.split(' ')),
            'gold_length': len(gold),
            'gold_num_prod': length,
            'ok': 1 if ok == 'True' else 0,
            'ok_grammar': 1 if ok_grammar == 'CorrectGrammar' else 0,
            'ok_function': 1 if ok_function == 'CorrectFunction' else 0,
            'ok_fn_count': 1 if ok_fn_count == 'CorrectNumFunction' else 0,
        })

if __name__ == '__main__':
    main()