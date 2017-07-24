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

def load_dictionary(file):
    print("Loading dictionary from %s..." % (file,))
    words = dict()

    # special tokens
    words['<<PAD>>'] = len(words)
    words['<<UNK>>'] = len(words)
    reverse = ['<<PAD>>', '<<UNK>>']

    with open(file, 'r') as word_file:
        for word in word_file:
            word = word.strip()
            if word not in words:
                words[word] = len(words)
                reverse.append(word)
    return words, reverse

def load_embeddings(from_file, words, embed_size=300):
    print("Loading pretrained embeddings...", end=' ')
    start = time.time()
    word_vectors = {}
    for line in open(from_file).readlines():
        sp = line.strip().split()
        if sp[0] in words:
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    word_vectors['<<PAD>>'] = np.zeros((embed_size,))
    word_vectors['<<UNK>>'] = np.zeros((embed_size,))
    n_tokens = len(words)
    embeddings_matrix = np.empty((n_tokens, embed_size), dtype='float32')
    for token, id in words.items():
        if token in word_vectors:
            embeddings_matrix[id] = word_vectors[token]
        else:
            raise ValueError("missing vector for", token)
    print("took {:.2f} seconds".format(time.time() - start))
    return embeddings_matrix

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
