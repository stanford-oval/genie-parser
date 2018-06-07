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
import time

from collections import namedtuple
from .general_utils import logged_loop

Dataset = namedtuple('Dataset', ('input_sequences', 'input_vectors', 'input_lengths',
    'constituency_parse', 'label_sequences', 'label_vectors', 'label_lengths'))

unknown_tokens = set()

def vectorize_constituency_parse(parse, max_length, expect_length):
    vector = np.zeros((2*max_length-1,), dtype=np.bool)
    
    # convention: false = shift, true = reduce
    if isinstance(parse, str):
        parse = parse.split(' ')
    i = 0
    for token in parse:
        if i == 2*max_length-1:
            break
        if token == '(':
            continue
        elif token == ')': # reduce
            vector[i] = True
            i += 1
        else: # shift
            vector[i] = False
            i += 1
    if i <= 2*max_length-1:
        assert i == 2*expect_length-1, (i, expect_length, parse)
    else:
        # we don't have space to shift/reduce the last token
        # this means that we truncated the sentence before
        raise ValueError('truncated constituency parse ' + str(parse))
    return vector

def vectorize(sentence, words, max_length, add_eos=False, add_start=False):
    vector = np.zeros((max_length,), dtype=np.int32)
    assert words['</s>'] == 0
    if isinstance(sentence, str):
        sentence = sentence.split(' ')
    if add_start:
        vector[0] = words['<s>']
        i = 1
    else:
        i = 0
    for word in sentence:
        word = word.strip()
        if len(word) == 0:
            raise ValueError("empty token in " + str(sentence))
        if word[0].isupper() and '*' in word:
            word, _ = word.rsplit('*', maxsplit=1)
        if word in words:
            vector[i] = words[word]
        elif '<unk>' in words:
            unknown_tokens.add(word)
            vector[i] = words['<unk>']
        else:
            raise ValueError('Unknown token ' + word)
        i += 1
        if i == max_length:
            break
    length = i
    if add_eos:
        if length < max_length:
            vector[length] = words['</s>']
            length += 1
        else:
            print("unterminated sentence", sentence)
    else:
        if length == max_length and length < len(sentence):
            print("truncated sentence", sentence)
    return (vector, length)

ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME', 'PATH_NAME', 'CURRENCY']
MAX_ARG_VALUES = 5

def load_dictionary(file, use_types=False, grammar=None):
    print("Loading dictionary from %s..." % (file,))
    words = dict()

    # special tokens
    words['</s>'] = len(words)
    words['<s>'] = len(words)
    words['<unk>'] = len(words)
    reverse = ['</s>', '<s>', '<unk>']
    def add_word(word):
        if word not in words:
            words[word] = len(words)
            reverse.append(word)
    
    if use_types:
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                add_word(entity + '_' + str(j))
        for i, (entity, has_ner) in enumerate(grammar.entities):
            if not has_ner:
                continue
            for j in range(MAX_ARG_VALUES):
                add_word('GENERIC_ENTITY_' + entity + '_' + str(j))

    with open(file, 'r') as word_file:
        for word in word_file:
            word = word.strip()
            if use_types and word[0].isupper():
                continue
            add_word(word)
    return words, reverse

def load_embeddings(from_file, words, use_types=False, grammar=None, embed_size=300):
    print("Loading pretrained embeddings...", end=' ')
    start = time.time()
    word_vectors = {}
    with open(from_file, 'r') as fp:
        for line in fp:
            sp = line.strip().split()
            if sp[0] in words:
                word_vectors[sp[0]] = [float(x) for x in sp[1:]]
                if len(word_vectors[sp[0]]) > embed_size:
                    raise ValueError("Truncated word vector for " + sp[0])
    n_tokens = len(words)
    
    original_embed_size = embed_size
    if use_types:
        num_entities = len(ENTITIES) + len(grammar.entities)
        embed_size += num_entities + MAX_ARG_VALUES + 2
    else:
        embed_size += 2

    # we give <unk> tokens the fully 0 vector (which means they have no
    # effect on the sentence)
    # we reserve the last two features in the embedding for <s> and </s>
    # </s> thus is a one-hot vector

    embeddings_matrix = np.zeros((n_tokens, embed_size), dtype='float32')
    embeddings_matrix[words['</s>'], embed_size-1] = 1.
    embeddings_matrix[words['<s>'], embed_size-2] = 1.

    for token, token_id in words.items():
        if token in ('<unk>', '</s>', '<s>'):
            continue
        if use_types and token[0].isupper():
            continue
        if token in word_vectors:
            vec = word_vectors[token]
            embeddings_matrix[token_id, 0:len(vec)] = vec
        else:
            raise ValueError("missing vector for", token)
    if use_types:
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                token_id = words[entity + '_' + str(j)]
                embeddings_matrix[token_id, original_embed_size + i] = 1.
                embeddings_matrix[token_id, original_embed_size + num_entities + j] = 1.
        for i, (entity, has_ner) in enumerate(grammar.entities):
            if not has_ner:
                continue
            for j in range(MAX_ARG_VALUES):
                token_id = words['GENERIC_ENTITY_' + entity + '_' + str(j)]
                embeddings_matrix[token_id, original_embed_size + len(ENTITIES) + i] = 1.
                embeddings_matrix[token_id, original_embed_size + num_entities + j] = 1.
    
    print("took {:.2f} seconds".format(time.time() - start))
    return embeddings_matrix, embed_size

# A custom list-like object that can be also indexed by a numpy array
# This object exists for compatibility with get_minibatches, for cases
# where the full numpy array would be too heavy
class CustomList(object):
    __slots__ = ['_data']

    def __init__(self, iterable):
        self._data = list(iterable)
    def __len__(self):
        return len(self._data)
    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            return CustomList(self._data[x] for x in key)
        else:
            return self._data[key]

def load_data(from_file, input_words, grammar, max_length):
    input_sequences = []
    inputs = []
    input_lengths = []
    parses = []
    labels = dict()
    for key in grammar.output_size:
        labels[key] = []
    label_lengths = []
    label_sequences = []
    total_label_len = 0

    N = 0
    with open(from_file, 'r') as data:
        for line in data:
            N += 1

        data.seek(0)
        for line in logged_loop(data, N):
            #print(line)
            split = line.strip().split('\t')
            if len(split) == 4:
                _, sentence, label, parse = split
            else:
                _, sentence, label = split
                parse = None

            input_vector, in_len = vectorize(sentence, input_words, max_length, add_eos=True, add_start=True)
            
            sentence = sentence.split(' ')
            input_sequences.append(sentence)

            inputs.append(input_vector)
            input_lengths.append(in_len)
            label_sequence = label.split(' ')
            label_vector, label_len = grammar.vectorize_program(sentence, label_sequence, max_length)

            label_sequences.append(label_sequence)
            total_label_len += label_len
            for key in grammar.output_size:
                labels[key].append(label_vector[key])
            label_lengths.append(label_len)
            if parse is not None:
                parses.append(vectorize_constituency_parse(parse, max_length, in_len-2))
            else:
                parses.append(np.zeros((2*max_length-1,), dtype=np.bool))
    print('avg label productions', total_label_len/len(inputs))

    input_sequences = CustomList(input_sequences)
    label_sequences = CustomList(label_sequences)
    inputs = np.array(inputs)
    input_lengths = np.array(input_lengths)
    parses = np.array(parses)
    for key in grammar.output_size:
        labels[key] = np.array(labels[key])
    label_lengths = np.array(label_lengths)

    return Dataset(input_sequences=input_sequences,
                   input_vectors=inputs,
                   input_lengths=input_lengths,
                   constituency_parse=parses,
                   label_sequences=label_sequences,
                   label_vectors=labels,
                   label_lengths=label_lengths)
