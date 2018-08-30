# Copyright 2018 Google LLC
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
Created on Aug 14, 2018

@author: gcampagn
'''

import os
import numpy as np

import pytest

from luinet.grammar.thingtalk import ThingTalkGrammar

class IdentityTextEncoder():
    def decode_list(self, x):
        return x


def test_uninitialized():
    grammar = ThingTalkGrammar(quiet=True)
    with pytest.raises(AttributeError):
        grammar.complete()


def test_initialize_from_file():
    filename = os.path.join(os.path.dirname(__file__), '../data/thingpedia.json')
    grammar = ThingTalkGrammar(filename, quiet=True)
    grammar.set_input_dictionary(IdentityTextEncoder())


def test_initialize_from_url():
    grammar = ThingTalkGrammar()
    grammar.init_from_url(-1, 'https://thingpedia.stanford.edu/thingpedia')
    grammar.set_input_dictionary(IdentityTextEncoder())


@pytest.fixture
def thingtalk_grammar():
    filename = os.path.join(os.path.dirname(__file__), '../data/thingpedia.json')
    grammar = ThingTalkGrammar(filename, flatten=True, quiet=True)
    grammar.set_input_dictionary(IdentityTextEncoder())
    return grammar


@pytest.fixture
def noquotes_thingtalk_grammar():
    filename = os.path.join(os.path.dirname(__file__), '../data/thingpedia.json')
    grammar = ThingTalkGrammar(filename, flatten=False, quiet=True)
    grammar.set_input_dictionary(IdentityTextEncoder())
    return grammar


def test_direct_string_to_string(thingtalk_grammar):
    test_vector_file = os.path.join(os.path.dirname(__file__), '../data/programs-withquotes.txt')
    with open(test_vector_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            parsed, _ = thingtalk_grammar.vectorize_program([], line, direction='bottomup', max_length=None)
            assert line == ' '.join(thingtalk_grammar.reconstruct_program([], parsed, direction='bottomup', ignore_errors=False))


def test_noquotes_direct_string_to_string(noquotes_thingtalk_grammar):
    test_vector_file = os.path.join(os.path.dirname(__file__), '../dataset/semparse_thingtalk_noquote/train.tsv')
    with open(test_vector_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            sentence, program = line.split('\t')[1:3]
            sentence = sentence.split(' ')
            parsed, _ = noquotes_thingtalk_grammar.vectorize_program(sentence, program, direction='bottomup', max_length=None)
            assert program == ' '.join(noquotes_thingtalk_grammar.reconstruct_program(sentence, parsed, direction='bottomup', ignore_errors=False))


def test_tokenize_and_parse(thingtalk_grammar):
    test_vector_file = os.path.join(os.path.dirname(__file__), '../data/programs-withquotes.txt')
    with open(test_vector_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            program = line.split(' ')
            tokenized = thingtalk_grammar.tokenize_to_vector([], program)
            assert program == thingtalk_grammar.decode_program([], tokenized)
            assert tokenized.shape == (len(program)*3,)
            
            for direction in ('bottomup', 'topdown', 'linear'):
                parsed, _ = thingtalk_grammar.vectorize_program(None, tokenized, direction=direction, max_length=None)
                if direction == 'linear':
                    assert np.all(np.equal(tokenized[::3], parsed['actions'][:-1]))
                reconstructed = thingtalk_grammar.reconstruct_to_vector(parsed, direction=direction, ignore_errors=False)
                assert np.all(np.equal(tokenized, reconstructed))


def test_noquotes_tokenized_and_parse(noquotes_thingtalk_grammar):
    test_vector_file = os.path.join(os.path.dirname(__file__), '../dataset/semparse_thingtalk_noquote/train.tsv')
    with open(test_vector_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            sentence, program = line.split('\t')[1:3]
            sentence = sentence.split(' ')
            program = program.split(' ')
            tokenized = noquotes_thingtalk_grammar.tokenize_to_vector(sentence, program)
            assert program == noquotes_thingtalk_grammar.decode_program(sentence, tokenized)
            assert tokenized.shape == (len(program)*3,)
            
            for direction in ('bottomup', 'topdown', 'linear'):
                parsed, _ = noquotes_thingtalk_grammar.vectorize_program(None, tokenized, direction=direction, max_length=None)
                if direction == 'linear':
                    assert np.all(np.equal(tokenized[::3], parsed['actions'][:-1]))
                reconstructed = noquotes_thingtalk_grammar.reconstruct_to_vector(parsed, direction=direction, ignore_errors=False)
                assert np.all(np.equal(tokenized, reconstructed))