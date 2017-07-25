'''
Created on Mar 16, 2017

@author: gcampagn
'''

import numpy as np
import time

unknown_tokens = set()

def vectorize(sentence, words, max_length, add_eos=False):
    vector = np.zeros((max_length,), dtype=np.int32)
    assert words['<<PAD>>'] == 0
    #vector[0] = words['<<GO>>']
    if isinstance(sentence, str):
        sentence = sentence.split(' ')
    for i, word in enumerate(sentence):
        word = word.strip()
        if len(word) == 0:
            raise ValueError("empty token in " + str(sentence))
        if word in words:
            vector[i] = words[word]
        elif '<<UNK>>' in words:
            unknown_tokens.add(word)
            #print("sentence: ", sentence, "; word: ", word)
            vector[i] = words['<<UNK>>']
        else:
            raise ValueError('Unknown token ' + word)
        if i+1 == max_length:
            break
    length = i+1
    if add_eos:
        if length < max_length:
            vector[length] = words['<<EOS>>']
            length += 1
        else:
            print("unterminated sentence", sentence)
    else:
        if length == max_length and length < len(sentence):
            print("truncated sentence", sentence)
    return (vector, length)

ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME']
MAX_ARG_VALUES = 5

def load_dictionary(file, use_types=False, grammar=None):
    print("Loading dictionary from %s..." % (file,))
    words = dict()

    # special tokens
    words['<<PAD>>'] = len(words)
    words['<<UNK>>'] = len(words)
    reverse = ['<<PAD>>', '<<UNK>>']
    def add_word(word):
        if word not in words:
            words[word] = len(words)
            reverse.append(word)
    
    if use_types:
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                add_word(entity + '_' + str(j))
        for i, entity in enumerate(grammar.entities):
            for j in range(MAX_ARG_VALUES):
                add_word('GENERIC_ENTITY_' + entity + '_' + str(j))

    with open(file, 'r') as word_file:
        for word in word_file:
            word = word.strip()
            if use_types and word[0].isupper():
                continue
            add_word(word)
    return words, reverse

ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME']
MAX_ARG_VALUES = 10

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
        embed_size += num_entities + MAX_ARG_VALUES
    
    embeddings_matrix = np.zeros((n_tokens, embed_size), dtype='float32')
    for token, id in words.items():
        if token in ('<<PAD>>', '<<UNK>>'):
            continue
        if use_types and token[0].isupper():
            continue
        if token in word_vectors:
            vec = word_vectors[token]
            embeddings_matrix[id, 0:len(vec)] = vec
        else:
            raise ValueError("missing vector for", token)
    if use_types:
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                token_id = words[entity + '_' + str(j)]
                embeddings_matrix[token_id, original_embed_size + i] = 1.
                embeddings_matrix[token_id, original_embed_size + num_entities + j] = 1.
        for i, entity in enumerate(grammar.entities):
            for j in range(MAX_ARG_VALUES):
                token_id = words['GENERIC_ENTITY_' + entity + '_' + str(j)]
                embeddings_matrix[token_id, original_embed_size + len(ENTITIES) + i] = 1.
                embeddings_matrix[token_id, original_embed_size + num_entities + j] = 1.
    
    print("took {:.2f} seconds".format(time.time() - start))
    return embeddings_matrix, embed_size

def load_data(from_file, input_words, output_words, max_length):
    inputs = []
    input_lengths = []
    labels = []
    label_lengths = []
    with open(from_file, 'r') as data:
        for line in data:
            sentence, canonical = line.split('\t')
            input, in_len = vectorize(sentence, input_words, max_length, add_eos=False)
            inputs.append(input)
            input_lengths.append(in_len)
            label, label_len = vectorize(canonical, output_words, max_length, add_eos=True)
            labels.append(label)
            label_lengths.append(label_len)
    return inputs, input_lengths, labels, label_lengths
