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

import re
import sys
import time
from collections import namedtuple, defaultdict

import tensorflow as tf
import numpy as np

Dataset = namedtuple('Dataset', ('input_sequences', 'input_vectors', 'input_lengths',
                                 'constituency_parse', 'label_sequences', 'label_vectors', 'label_lengths'))

unknown_tokens = set()

ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME', 'PATH_NAME', 'CURRENCY']
MAX_ARG_VALUES = 5

HACK_REPLACEMENT = {
    'onedrive': 'skydrive',
    'imgflip': 'imgur'
}


def clean(name):
    """Normalize argument names into English words.

    Removes the "v_" prefix, converts camelCase to multiple words, and underscores
    to spaces.
    """
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()


def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', re.sub('[()]', '', name.lower()))


def vectorize(sentence, words, max_length=None, add_eos=False, add_start=False):
    if isinstance(sentence, str):
        sentence = sentence.split(' ')
    if max_length is None:
        max_length = len(sentence)
        if add_start:
            max_length += 1
        if add_eos:
            max_length += 1
    vector = np.zeros((max_length,), dtype=np.int32)
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


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self._isatty = sys.stderr.isatty()

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        if not self._isatty:
            return

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stderr.write("\b" * prev_total_width)
            sys.stderr.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stderr.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stderr.write(info)
            sys.stderr.flush()

            if current >= self.target:
                sys.stderr.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stderr.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def logged_loop(iterable, n=None):
    if n is None:
        n = len(iterable)
    step = max(1, n // 1000)
    prog = Progbar(n)
    for i, elem in enumerate(iterable):
        if i % step == 0 or i == n - 1:
            prog.update(i + 1)
        yield elem


def vectorize_constituency_parse(parse, max_length, expect_length):
    vector = np.zeros((2 * max_length - 1,), dtype=np.bool)

    # convention: false = shift, true = reduce
    if isinstance(parse, str):
        parse = parse.split(' ')
    i = 0
    for token in parse:
        if i == 2 * max_length - 1:
            break
        if token == '(':
            continue
        elif token == ')':  # reduce
            vector[i] = True
            i += 1
        else:  # shift
            vector[i] = False
            i += 1
    if i <= 2 * max_length - 1:
        assert i == 2 * expect_length - 1, (i, expect_length, parse)
    else:
        # we don't have space to shift/reduce the last token
        # this means that we truncated the sentence before
        raise ValueError('truncated constituency parse ' + str(parse))
    return vector


def load_dictionary(file, use_types=False, grammar=None):
    print("Loading dictionary from %s..." % (file,))
    words = dict()

    # special tokens
    words['<pad>'] = len(words)
    words['</s>'] = len(words)
    words['<s>'] = len(words)
    words['<unk>'] = len(words)
    reverse = ['<pad>', '</s>', '<s>', '<unk>']

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


def load_embeddings(glove, dictionary, use_types=False, grammar=None, embed_size=300):
    print("Loading pretrained embeddings...", end=' ')
    start = time.time()
    original_embed_size = embed_size
    if use_types:
        num_entities = len(ENTITIES) + len(grammar.entities)
        embed_size += num_entities + MAX_ARG_VALUES + 2
    else:
        embed_size += 2

    # there are 4 reserved tokens: pad, eos, start and unk
    embedding_matrix = np.zeros((len(dictionary), embed_size),
                                dtype=np.float32)
    # careful here! various places in t2t assume that padding
    # will have a all-zero embedding
    # and nothing else will have all-zero
    # (eg. common_attention.embedding_to_padding, which
    # is called by transformer_prepare_encoder)

    # we also use all-one for <unk>
    embedding_matrix[dictionary['</s>'], embed_size - 1] = 1.
    embedding_matrix[dictionary['<s>'], embed_size - 2] = 1.
    embedding_matrix[dictionary['<unk>'], :original_embed_size] = np.ones((original_embed_size,))

    trimmed_glove = dict()
    hack_values = HACK_REPLACEMENT.values()
    with tf.gfile.Open(glove, "r") as fp:
        for line in fp:
            line = line.strip()
            vector = line.split(' ')
            word, vector = vector[0], vector[1:]
            if word not in dictionary and word not in hack_values:
                continue
            vector = np.array(list(map(float, vector)))
            trimmed_glove[word] = vector

    BLANK = re.compile('^_+$')
    for word, word_id in dictionary.items():
        if word in ['<pad>', '</s>', '<s>', '<unk>']:
            continue
        assert isinstance(word, str), (word, word_id)
        if use_types and word[0].isupper():
            continue
        if word in trimmed_glove:
            embedding_matrix[word_id, :original_embed_size] = trimmed_glove[word]
            continue

        if not word or re.match('\s+', word):
            raise ValueError('Invalid word "%s"' % (word,))
        vector = None
        if BLANK.match(word):
            # normalize blanks
            vector = trimmed_glove['____']
        elif word.endswith('s') and word[:-1] in trimmed_glove:
            vector = trimmed_glove[word[:-1]]
        elif (word.endswith('ing') or word.endswith('api')) and word[:-3] in trimmed_glove:
            vector = trimmed_glove[word[:-3]]
        elif word in HACK_REPLACEMENT:
            vector = trimmed_glove[HACK_REPLACEMENT[word]]
        elif '-' in word:
            vector = np.zeros(shape=(original_embed_size,), dtype=np.float64)
            for w in word.split('-'):
                if w in trimmed_glove:
                    vector += trimmed_glove[w]
                else:
                    vector = None
                    break
        if vector is not None:
            embedding_matrix[word_id, :original_embed_size] = vector
        else:
            tf.logging.warn("missing word from GloVe: %s", word)
            embedding_matrix[word_id, :original_embed_size] = np.random.normal(0, 0.9, (original_embed_size,))
    del trimmed_glove

    if use_types:
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                token_id = dictionary[entity + '_' + str(j)]
                embedding_matrix[token_id, original_embed_size + i] = 1.
                embedding_matrix[token_id, original_embed_size + num_entities + j] = 1.
        for i, (entity, has_ner) in enumerate(grammar.entities):
            if not has_ner:
                continue
            for j in range(MAX_ARG_VALUES):
                token_id = dictionary['GENERIC_ENTITY_' + entity + '_' + str(j)]
                embedding_matrix[token_id, original_embed_size + len(ENTITIES) + i] = 1.
                embedding_matrix[token_id, original_embed_size + num_entities + j] = 1.

    print("took {:.2f} seconds".format(time.time() - start))
    return embedding_matrix, embed_size


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

def vectorize_program(program, grammar, direction='linear', max_length=60):
    if direction != 'linear':
        raise ValueError("Invalid " + direction + " direction for simple grammar")
    if isinstance(program, np.ndarray):
        return {'tokens': program}, len(program)

    vector, vlen = vectorize(program, grammar.dictionary, max_length, add_eos=True, add_start=False)
    return {
               'tokens': vector
           }, vlen


def load_data(from_file, input_words, grammar, max_length):
    input_sequences = []
    inputs = []
    input_lengths = []
    parses = []
    labels = defaultdict(list)
    label_lengths = []
    label_sequences = []
    total_label_len = 0

    N = 0
    with open(from_file, 'r') as data:
        for line in data:
            N += 1

        data.seek(0)
        for line in logged_loop(data, N):
            # print(line)
            split = line.strip().split('\t')
            if len(split) == 4:
                _, sentence, label, parse = split
            else:
                _, sentence, label = split
                parse = None

            input_vector, in_len = vectorize(sentence, input_words, max_length, add_eos=False, add_start=False)

            sentence = sentence.split(' ')
            input_sequences.append(sentence)

            inputs.append(input_vector)
            input_lengths.append(in_len)
            label_sequence = label.split(' ')
            output_vector = grammar.tokenize_to_vector(sentence, label_sequence)
            output_vector = np.reshape(output_vector, [-1, 3])
            label_len = output_vector.shape[0]
            label_vector = np.zeros((max_length,3), dtype=np.int32)
            for i in range(label_len):
                label_vector[i, :] = output_vector[i, :]

            label_sequences.append(label_sequence)
            total_label_len += label_len
            for i, key in enumerate(grammar.output_size):
                labels[key].append(label_vector[:, i])
            label_lengths.append(label_len)
            if parse is not None:
                parses.append(vectorize_constituency_parse(parse, max_length, in_len - 2))
            else:
                parses.append(np.zeros((2 * max_length - 1,), dtype=np.bool))
    print('avg label productions', total_label_len / len(inputs))

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
