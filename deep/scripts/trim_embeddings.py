#!/usr/bin/python3

import sys
import orderedset

import numpy as np

words = orderedset.OrderedSet()
with open(sys.argv[1], 'r') as fp:
    for line in fp.readlines():
        word = line.strip()
        words.add(word)

# HACK: glove is missing a word for xkcds (plural), ubering, weatherapi and a few others
# we don't want to stem or lemmatize everything yet,
# but we need a good vector for it
# so we take a closely related vector instead
# the complication comes from not being able to keep the whole matrix in memory
HACK = {
    'xkcd': None,
    'uber': None,
    'weather': None
}
for line in sys.stdin.readlines():
    stripped = line.strip()
    sp = stripped.split()
    if sp[0] in HACK:
        HACK[sp[0]] = sp[1:]
    if sp[0] in words:
        print(stripped)
        words.remove(sp[0])

# add small predictable random values for the words missing
EMBED_SIZE = 300
np.random.seed(1234)

for word in words:
    vector = None
    if word.endswith('s') and word[:-1] in HACK:
        vector = HACK[word[:-1]]
    elif (word.endswith('ing') or word.endswith('api')) and word[:-3] in HACK:
        vector = HACK[word[:-3]]
    if vector:
        print(word, *vector)
    else:
        if not word[0].isupper():
            print("warning: missing word", word, file=sys.stderr)
        print(word, *np.random.normal(0, 0.9, (EMBED_SIZE,)))