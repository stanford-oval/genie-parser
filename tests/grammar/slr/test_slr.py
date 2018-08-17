# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#           2018 Google LLC
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
Created on Nov 30, 2017

@author: gcampagn
'''

import pytest

from luinet.grammar.slr.generator import SLRParserGenerator
from luinet.grammar.slr import SHIFT_CODE, REDUCE_CODE


TEST_GRAMMAR = {
'$prog':    [('$command',),
             ('$rule',)],

'$rule':    [('$stream', '$action'),
             ('$stream', 'notify')],
'$command': [('$table', 'notify'),
             ('$table', '$action')],
'$table':   [('$get',),
             ('$table', 'filter', '$filter')],
'$stream':  [('monitor', '$table')],
'$get':     [('$get', '$ip'),
             ('GET',)],
'$action':  [('$action', '$ip'),
             ('DO',)],
'$ip':      [('PARAM', '$number'),
             ('PARAM', '$string')],
'$number':  [('num0',),
             ('num1',)],
'$string':  [('qs0',),
             ('qs1',)],
'$filter':  [('PARAM', '==', '$number'),
             ('PARAM', '>', '$number'),
             ('PARAM', '<', '$number'),
             ('PARAM', '==', '$string'),
             ('PARAM', '=~', '$string')]
}
TEST_TERMINALS = {
    'PARAM': { 'param:number', 'param:text' },
    'GET': { 'xkcd.get_comic', 'thermostat.get_temp', 'twitter.search' },
    'DO': { 'twitter.post' }
}

# The grammar of nesting parenthesis
# this is context free but not regular
# (also not parsable with a Petri-net)
#
# The reduce sequence will be: the reduction of the inner most parenthesis pair,
# followed by the reduction for the next parenthesis, in order
#
# This has one important consequence: if the NN produces the sequence
# X Y Z* ...
# where X is reduction 4 or 5 (a or b), Y is reduction 0 or 1, and Z
# is 2 or 3, it will automatically produce a well-formed string in the language
# The NN is good at producing never ending sequences of the same thing (in
# fact, it tends to do that too much), so it should have no trouble with
# this language 
PARENTHESIS_GRAMMAR = {
'$S': [('(', '$V', ')'),
       ('[', '$V', ']'),
       ('(', '$S', ')'),
       ('[', '$S', ']')],
'$V': [('a',), ('b',)]
}


def tokenize(seq, generator, terminals=None):
    for token in seq:
        found = False
        if terminals is not None:
            for test_term, values in terminals.items():
                if token in values:
                    yield generator.dictionary[test_term], token
                    found = True
                    break
        if not found:
            yield generator.dictionary[token], token


def reconstruct(seq, generator):
    for token_id, token in seq:
        if token is not None:
            yield token
        else:
            yield generator.terminals[token_id]


def remove_shifts(sequence, generator, terminals=None):
    for action, param in sequence:
        if action != SHIFT_CODE:
            yield action, param
            continue
        
        token_id, payload = param
        is_necessary = False
        if terminals is not None:
            # this is a "necessary" shift, because the payload
            # is meaningful
            # we leave it there (or we won't reconstruct the
            # program)
            is_necessary = generator.terminals[token_id] in terminals
        # only gen the action if necessary
        if is_necessary:
            yield action, param


def do_test_with_grammar(grammar, start_symbol, test_vectors, terminals=None):
    generator = SLRParserGenerator(grammar, start_symbol)
    parser = generator.build()

    for expected in test_vectors:
        tokenized = list(tokenize(expected, generator, terminals))
        parsed = list(parser.parse(tokenized))
        reconstructed = list(reconstruct(parser.reconstruct(parsed), generator))
        assert expected == reconstructed
        
        parsed_td = list(parser.parse_reverse(tokenized))
        reconstructed_td = list(reconstruct(parser.reconstruct_reverse(parsed_td), generator))
        assert expected == reconstructed_td
        
        assert len(parsed) == len(parsed_td)

        parsed_without_shifts = remove_shifts(parsed, generator, terminals)
        assert expected == list(reconstruct(parser.reconstruct(parsed_without_shifts), generator))
        
        parsed_td_without_shifts = remove_shifts(parsed_td, generator, terminals)
        assert expected == list(reconstruct(parser.reconstruct_reverse(parsed_td_without_shifts), generator))


def do_test_manual(grammar, start_symbol, test_vectors, parses, terminals=None):
    generator = SLRParserGenerator(grammar, start_symbol)
    parser = generator.build()

    for program, expected_parse in zip(test_vectors, parses):
        tokenized = list(tokenize(program, generator, terminals))
        parsed = list(parser.parse(tokenized))
        assert parsed == expected_parse


def do_test_invalid(grammar, start_symbol, test_vectors, terminals=None):
    generator = SLRParserGenerator(grammar, start_symbol)
    parser = generator.build()
    
    for program in test_vectors:
        tokenized = tokenize(program, generator, terminals)
        with pytest.raises(ValueError):
            parser.parse(tokenized)


def do_test_invalid_tokenize(grammar, start_symbol, test_vectors, terminals=None):
    generator = SLRParserGenerator(grammar, start_symbol)
    
    for program in test_vectors:
        with pytest.raises(KeyError):
            # force running the generator all the way to the end
            list(tokenize(program, generator, terminals))


def test_tiny_thingtalk():
    TEST_VECTORS = [
        ['monitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'],
        ['monitor', 'thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify'],
        ['thermostat.get_temp', 'filter', 'param:number', '>', 'num0', 'notify']
    ]
    do_test_with_grammar(TEST_GRAMMAR, '$prog', TEST_VECTORS, TEST_TERMINALS)


def test_invalid_tiny_thingtalk():
    TEST_VECTORS = [
        ['monitor', 'twitter.post', 'param:text', 'qs0'],
        ['thermostat.get_temp', 'filter', 'notify']
    ]
    do_test_invalid(TEST_GRAMMAR, '$prog', TEST_VECTORS, TEST_TERMINALS)


def test_invalid_tokenize():
    TEST_VECTORS = [
        ['manitor', 'thermostat.get_temp', 'twitter.post', 'param:text', 'qs0'],
        ['thermostat.get_temp', 'filter', 'param:namber', '>', 'num0', 'notify']
    ]
    do_test_invalid_tokenize(TEST_GRAMMAR, '$prog', TEST_VECTORS, TEST_TERMINALS)


def test_parenthesis():
    TEST_VECTORS = [
        ['(', '(', '(', 'a', ')', ')', ')'],
        ['[', '[', '[', 'a', ']', ']', ']'],
        ['(', '[', '(', 'b', ')', ']', ')']
    ]
    do_test_with_grammar(PARENTHESIS_GRAMMAR, '$S', TEST_VECTORS, terminals=None)

def test_parenthesis_manual():
    TEST_VECTORS = [
        ['(', '(', '(', 'a', ')', ')', ')'],
    ]
    TEST_EXPECTED = [
        [(SHIFT_CODE, (3, '(')),
         (SHIFT_CODE, (3, '(')),
         (SHIFT_CODE, (3, '(')),
         (SHIFT_CODE, (7, 'a')),
         (REDUCE_CODE, 4),
         (SHIFT_CODE, (4, ')')),
         (REDUCE_CODE, 0),
         (SHIFT_CODE, (4, ')')),
         (REDUCE_CODE, 2),
         (SHIFT_CODE, (4, ')')),
         (REDUCE_CODE, 2)]
    ]
    do_test_manual(PARENTHESIS_GRAMMAR, '$S', TEST_VECTORS, TEST_EXPECTED, terminals=None)
    


def test_invalid_parenthesis():
    TEST_VECTORS = [
        ['[', '[', '[', 'a', ')', ']', ']'],
        ['(', '[', '(', 'b', ']', ')']
    ]
    do_test_invalid(PARENTHESIS_GRAMMAR, '$S', TEST_VECTORS, terminals=None)