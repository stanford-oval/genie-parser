#!/usr/bin/python3
#
# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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

import sys
import orderedset
import re

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
    'weather': None,
    'skydrive': None,
    'imgur': None,
    '____': None
}
HACK_REPLACEMENT = {
    # onedrive is the new name of skydrive
    'onedrive': 'skydrive',

    # imgflip is kind of the same as imgur (or 9gag)
    # until we have either in thingpedia, it's fine to reuse the word vector
    'imgflip': 'imgur'
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
if len(sys.argv) > 2:
    EMBED_SIZE = int(sys.argv[2])
else:
    EMBED_SIZE = 300
np.random.seed(1234)

blank = re.compile('^_+$')

for word in words:
    vector = None
    if blank.match(word):
        # normalize blanks
        vector = HACK['____']
    elif word.endswith('s') and word[:-1] in HACK:
        vector = HACK[word[:-1]]
    elif (word.endswith('ing') or word.endswith('api')) and word[:-3] in HACK:
        vector = HACK[word[:-3]]
    elif word in HACK_REPLACEMENT:
        vector = HACK[HACK_REPLACEMENT[word]]
    if vector:
        print(word, *vector)
    else:
        if not word[0].isupper():
            print("warning: missing word", word, file=sys.stderr)
        print(word, *np.random.normal(0, 0.9, (EMBED_SIZE,)))
