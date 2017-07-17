'''
Created on Mar 16, 2017

@author: gcampagn
'''

import numpy as np
import time

unknown_tokens = set()

def vectorize(sentence, words, max_length):
    vector = np.zeros((max_length,), dtype=np.int32)
    assert words['<<PAD>>'] == 0
    #vector[0] = words['<<GO>>']
    if isinstance(sentence, str):
        sentence = sentence.split(' ')
    for i, word in enumerate(sentence):
        word = word.strip()
        if word in words:
            vector[i] = words[word]
        else:
            unknown_tokens.add(word)
            #print("sentence: ", sentence, "; word: ", word)
            vector[i] = words['<<UNK>>']
        if i+1 == max_length:
            break
    length = i+1
    if length < max_length:
        vector[length] = words['<<EOS>>']
        length += 1
    else:
        print("truncated sentence", sentence)
    return (vector, length)

ENTITIES = ['USERNAME', 'HASHTAG',
            'QUOTED_STRING', 'NUMBER',
            'PHONE_NUMBER', 'EMAIL_ADDRESS', 'URL',
            'DATE', 'TIME', 'SET',
            'PERCENT', 'DURATION', 'MONEY', 'ORDINAL']

def load_dictionary(file, benchmark):
    print("Loading dictionary from %s..." % (file,))
    words = dict()

    # special tokens
    words['<<PAD>>'] = len(words)
    words['<<EOS>>'] = len(words)
    words['<<GO>>'] = len(words)
    words['<<UNK>>'] = len(words)
    reverse = ['<<PAD>>', '<<EOS>>', '<<GO>>', '<<UNK>>']

    if benchmark == "tt":
        for entity in ENTITIES:
            words[entity] = len(words)
            reverse.append(entity)

    with open(file, 'r') as word_file:
        for word in word_file:
            word = word.strip()
            if word not in words:
                words[word] = len(words)
                reverse.append(word)
    for id in range(len(reverse)):
        if words[reverse[id]] != id:
            print("found problem at", id)
            print("word: ", reverse[id])
            print("expected: ", words[reverse[id]])
            raise AssertionError
    return words, reverse

def load_embeddings(from_file, words, embed_size=300):
    print("Loading pretrained embeddings...", end=' ')
    start = time.time()
    word_vectors = {}
    for line in open(from_file).readlines():
        sp = line.strip().split()
        if sp[0] in words:
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    n_tokens = len(words)
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (n_tokens, embed_size)), dtype='float32')
    for token, id in words.items():
        if token in word_vectors:
            embeddings_matrix[id] = word_vectors[token]
    print("took {:.2f} seconds".format(time.time() - start))
    return embeddings_matrix

def load_data(from_file, input_words, output_words, input_reverse, output_reverse, max_length):
    inputs = []
    input_lengths = []
    labels = []
    label_lengths = []
    with open(from_file, 'r') as data:
        for line in data:
            sentence, canonical = line.split('\t')
            input, in_len = vectorize(sentence, input_words, max_length)
            inputs.append(input)
            input_lengths.append(in_len)
            label, label_len = vectorize(canonical, output_words, max_length)
            labels.append(label)
            label_lengths.append(label_len)
            #print "input", in_len, ' '.join(map(lambda x: input_reverse[x], inputs[-1]))
            #print "label", label_len, ' '.join(map(lambda x: output_reverse[x], labels[-1]))
    return inputs, input_lengths, labels, label_lengths