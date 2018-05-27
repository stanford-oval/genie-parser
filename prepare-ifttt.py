#!/usr/bin/python3
#
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
Created on Nov 6, 2017

@author: gcampagn
'''

import sys
import os
import urllib.request
import ssl
import zipfile
import re
import tempfile
import shutil
import numpy as np

ssl_context = ssl.create_default_context()

def add_words(input_words, canonical):
    if isinstance(canonical, str):
        sequence = canonical.split(' ')
    else:
        sequence = canonical
    for word in sequence:
        if not word:
            continue
        input_words.add(word)

def download_glove(glove):
    if os.path.exists(glove):
        return

    print('Downloading glove...')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen('http://nlp.stanford.edu/data/glove.42B.300d.zip') as res:
            shutil.copyfileobj(res, tmp)
        with zipfile.ZipFile(tmp, 'r') as glove_zip:
            glove_zip.extract('glove.42B.300d.txt', path=os.path.dirname(glove))
    print('Done')

def create_dictionary(input_words, dataset):
    for filename in os.listdir(dataset):
        if not filename.endswith('.tsv'):
            continue

        with open(os.path.join(dataset, filename), 'r') as fp:
            for line in fp:
                sentence = line.strip().split('\t')[1]
                add_words(input_words, sentence)

    if len(sys.argv) > 4:
       extra_word_file = sys.argv[4]
       print('Adding extra dictionary from', extra_word_file)
       with open(extra_word_file, 'r') as fp:
           for line in fp:
               input_words.add(line.strip())


def save_dictionary(input_words, workdir):
    with open(os.path.join(workdir, 'input_words.txt'), 'w') as fp:
        for word in sorted(input_words):
            print(word, file=fp)

def trim_embeddings(input_words, workdir, embed_size, glove):
    HACK = {
        'xkcd': None,
        'uber': None,
        'weather': None,
        'skydrive': None,
        'imgur': None,
        '____': None,
        'github': None,
        'related': None,
        'med.': None,
        'sized': None,
        'primary': None,
        'category': None,
        'alphabetically': None,
        'ordered': None,
        'recently': None,
        'changed': None,
        'reverse': None,
        'abc': None,
        'newly': None,
        'reported': None,
        'newest': None,
        'updated': None,
        'home': None,
        'taken': None,
        'direct': None,
        'messages': None
    }
    HACK_REPLACEMENT = {
        # onedrive is the new name of skydrive
        'onedrive': 'skydrive',
    
        # imgflip is kind of the same as imgur (or 9gag)
        # until we have either in thingpedia, it's fine to reuse the word vector
        'imgflip': 'imgur'
    }
    blank = re.compile('^_+$')
    
    output_embedding_file = os.path.join(workdir, 'embeddings-' + str(embed_size) + '.txt')
    with open(output_embedding_file, 'w') as outfp:
        with open(glove, 'r') as fp:
            for line in fp:
                stripped = line.strip()
                sp = stripped.split()
                if sp[0] in HACK:
                    HACK[sp[0]] = sp[1:]
                if sp[0] in input_words:
                    print(stripped, file=outfp)
                    input_words.remove(sp[0])

        for word in input_words:
            if not word or re.match('\s+', word):
                raise ValueError('Invalid word "%s"' % (word,))
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
            elif '-' in word:
                vector = np.zeros(shape=(len(HACK['____']),), dtype=np.float64)
                for w in word.split('-'):
                    if w in HACK:
                        vector += np.array(HACK[w], dtype=np.float64)
                    else:
                        vector = None
                        break
            if vector is not None:
                print(word, *vector, file=outfp)
            else:
                if not word[0].isupper():
                    print("WARNING: missing word", word)
                print(word, *np.random.normal(0, 0.9, (embed_size,)), file=outfp)

def main():
    np.random.seed(1234)

    workdir = sys.argv[1]
    if len(sys.argv) > 2:
        embed_size = int(sys.argv[2])
    else:
        embed_size = 300

    dataset = os.getenv('DATASET', workdir)
    glove = os.getenv('GLOVE', os.path.join(workdir, 'glove.42B.300d.txt'))
    
    download_glove(glove)
    
    input_words = set()

    create_dictionary(input_words, dataset)
    save_dictionary(input_words, workdir)
    trim_embeddings(input_words, workdir, embed_size, glove)

if __name__ == '__main__':
    main()
