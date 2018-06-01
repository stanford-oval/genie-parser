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
import configparser

from grammar import thingtalk
from util.loader import vectorize

ssl_context = ssl.create_default_context()

def clean(name):
    if name.startswith('v_'):
        name = name[len('v_'):]
    return re.sub('([^A-Z])([A-Z])', '$1 $2', re.sub('_', ' ', name)).lower()

def tokenize(name):
    return re.split(r'\s+|[,\.\"\'!\?]', re.sub('[()]', '', name.lower()))

def add_words(input_words, canonical):
    if isinstance(canonical, str):
        sequence = canonical.split(' ')
    else:
        sequence = canonical
    for word in sequence:
        if not word or word != '$' and word.startswith('$'):
            print('Invalid word "%s" in phrase "%s"' % (word, canonical,))
            continue
        if word[0].isupper():
            continue
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
                        add_words(input_words, function['canonical'].lower())
                    for argname, argcanonical in zip(function['args'], function['argcanonicals']):
                        if argcanonical:
                            add_words(input_words, argcanonical.lower())
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

def download_glove(glove, embed_size):
    if os.path.exists(glove):
        return

    print('Downloading glove...')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen('https://nlp.stanford.edu/data/glove.42B.' + str(embed_size) + 'd.zip') as res:
            shutil.copyfileobj(res, tmp)
        with zipfile.ZipFile(tmp, 'r') as glove_zip:
            glove_zip.extract('glove.42B.' + str(embed_size) + 'd.txt', path=os.path.dirname(glove))
    print('Done')

def load_dataset(input_words, dataset):
    for filename in os.listdir(dataset):
        if not filename.endswith('.tsv'):
            continue

        with open(os.path.join(dataset, filename), 'r') as fp:
            for line in fp:
                sentence = line.strip().split('\t')[1]
                sentence = sentence.split(' ')
                add_words(input_words, sentence)

def verify_dataset(input_words, dataset, grammar):
    for filename in os.listdir(dataset):
        if not filename.endswith('.tsv'):
            continue

        with open(os.path.join(dataset, filename), 'r') as fp:
            for line in fp:
                sentence, program = line.strip().split('\t')[1:3]
                sentence = sentence.split(' ')
                sentence_vector, _ = vectorize(sentence, input_words, max_length=60, add_start=True, add_eos=True)
                grammar.vectorize_program(sentence_vector, program)

def add_extra_words(input_words, extra_word_file):
    print('Adding extra dictionary from', extra_word_file)
    with open(extra_word_file, 'r') as fp:
        for line in fp:
            input_words.add(line.strip())


def save_dictionary(input_words, workdir):
    dictionary = dict()
    dictionary['</s>'] = 0
    dictionary['<s>'] = 1
    dictionary['<unk>'] = 2

    with open(os.path.join(workdir, 'input_words.txt'), 'w') as fp:
        for i, word in enumerate(sorted(input_words)):
            dictionary[word] = 3+i
            print(word, file=fp)
    return dictionary

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

def create_model_conf(workdir, embed_size):
    os.makedirs(os.path.join(workdir, 'model'), exist_ok=True)
    
    model_config = configparser.ConfigParser()
    model_config['input'] = {
        'input_words': os.path.join(workdir, 'input_words.txt'),
        'input_embeddings': os.path.join(workdir, 'embeddings-' + str(embed_size) + '.txt'),
        'input_embed_size': embed_size
    }
    model_config['output'] = {
        'grammar': 'thingtalk.ThingTalkGrammar',
        'grammar_input_file': os.path.join(workdir, 'thingpedia.json')
    }
    
    with open(os.path.join(workdir, 'model', 'model.conf'), 'w') as fp:
        model_config.write(fp)

def main():
    np.random.seed(1234)

    workdir = sys.argv[1]
    os.makedirs(workdir, exist_ok=True)
    dataset = sys.argv[2]
    if len(sys.argv) > 3:
        snapshot = int(sys.argv[3])
    else:
        snapshot = -1
    if len(sys.argv) > 4:
        embed_size = int(sys.argv[4])
    else:
        embed_size = 300
    if len(sys.argv) > 5:
        extra_word_file = sys.argv[5]
    else:
        extra_word_file = None
        
    glove = os.getenv('GLOVE', os.path.join(workdir, 'glove.42B.' + str(embed_size) + 'd.txt'))    
    download_glove(glove, embed_size)
    
    input_words = set()
    # add the canonical words for the builtin functions
    add_words(input_words, 'now nothing notify return the event')

    get_thingpedia(input_words, workdir, snapshot)
    grammar = thingtalk.ThingTalkGrammar(os.path.join(workdir, 'thingpedia.json'), flatten=False)
    
    load_dataset(input_words, dataset)
    if extra_word_file:
        add_extra_words(input_words, extra_word_file)
    dictionary = save_dictionary(input_words, workdir)
    grammar.set_input_dictionary(dictionary)
    verify_dataset(dictionary, dataset, grammar)

    trim_embeddings(input_words, workdir, embed_size, glove)
    create_model_conf(workdir, embed_size)

if __name__ == '__main__':
    main()
