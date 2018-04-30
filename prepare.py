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
import json
import os
import urllib.request
import ssl
import zipfile
import re
import tempfile
import shutil
import numpy as np

ssl_context = ssl.create_default_context()

def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()

def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?\_]', re.sub('[()]', '', name.lower()))

def add_words(input_words, canonical):
    if isinstance(canonical, str):
        sequence = canonical.split(' ')
    else:
        sequence = canonical
    for word in sequence:
        if not word:
            raise ValueError('Invalid canonical "%s"' % (canonical,))
        input_words.add(word)

def get_thingpedia(input_words, workdir, snapshot):
    thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')

    output = dict()
    with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1', context=ssl_context) as res:
        output['devices'] = json.load(res)['data']
        for device in output['devices']:
            if device['kind_type'] in ('global', 'category', 'discovery'):
                continue
            if device.get('kind_canonical', None):
                add_words(input_words, device['kind_canonical'])
            else:
                print('WARNING: missing canonical for device:%s' % (device['kind'],))
            for function_type in ('triggers', 'queries', 'actions'):
                for function_name, function in device[function_type].items():
                    if not function['canonical']:
                        print('WARNING: missing canonical for @%s.%s' % (device['kind'], function_name))
                    else:
                        add_words(input_words, function['canonical'])
                    for argname, argcanonical in zip(function['args'], function['argcanonicals']):
                        if argcanonical:
                            add_words(input_words, argcanonical)
                        else:
                            add_words(input_words, clean(argname))
                    for argtype in function['schema']:
                        if not argtype.startswith('Enum('):
                            continue
                        enum_entries = argtype[len('Enum('):-1].split(',')
                        for enum_value in enum_entries:
                            add_words(input_words, clean(enum_value))

    with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=ssl_context) as res:
        output['entities'] = json.load(res)['data']
        for entity in output['entities']:
            if entity['is_well_known'] == 1:
                continue
            add_words(input_words, tokenize(entity['name']))
    
    with open(os.path.join(workdir, 'thingpedia.json'), 'w') as fp:
        json.dump(output, fp, indent=2)

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
        '____': None,
    }
    HACK_REPLACEMENT = {
        # onedrive is the new name of skydrive
        'onedrive': 'skydrive',

        'phdcomic': 'phdcomics',
        'yahoofinance': 'yahoo!finance',

        # imgflip is kind of the same as imgur (or 9gag)
        # until we have either in thingpedia, it's fine to reuse the word vector
        'imgflip': 'imgur',

        'thecatapi': 'cat',
        'thedogapi': 'dog'
    }
    for v in HACK_REPLACEMENT.values():
        HACK[v] = None
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
        snapshot = int(sys.argv[2])
    else:
        snapshot = -1
    if len(sys.argv) > 3:
        embed_size = int(sys.argv[3])
    else:
        embed_size = 300
        
    dataset = os.getenv('DATASET', workdir)
    glove = os.getenv('GLOVE', os.path.join(workdir, 'glove.42B.300d.txt'))
    
    download_glove(glove)
    
    input_words = set()

    # and a few canonical words that are useful
    add_words(input_words, 'now nothing notify return the event 0 1 2 3 4 5 6 7 8 9 10')
    
    create_dictionary(input_words, dataset)
    get_thingpedia(input_words, workdir, snapshot)
    save_dictionary(input_words, workdir)
    trim_embeddings(input_words, workdir, embed_size, glove)

if __name__ == '__main__':
    main()
