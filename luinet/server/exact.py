# Copyright 2017-2018 The Board of Trustees of the Leland Stanford Junior University
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

import tensorflow as tf


class TrieNode:
    def __init__(self):
        self.value = None
        self.children = dict()

    def set_value(self, value):
        self.value = value

    def add_child(self, key):
        child = TrieNode()
        self.children[key] = child
        return child
    
    def get_child(self, key):
        return self.children.get(key, None)


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, sequence, value):
        node = self.root
        for key in sequence:
            child = node.get_child(key)
            if child is None:
                child = node.add_child(key)
            node = child
        node.set_value(value)
    
    def search(self, sequence):
        node = self.root
        for key in sequence:
            child = node.get_child(key)
            if child is None:
                return None
            node = child
        return node.value


class ExactMatcher():
    def __init__(self, database, language, model_tag):
        self._database = database
        self._language = language
        self._model_tag = model_tag
        
        self._trie = Trie()
    
    def load(self):
        if self._model_tag is not None:
            # FIXME
            tf.logging.info('Skipping exact matcher for non-default model @%s', self._model_tag)
            return
        
        n = 0
        for row in self._database.execute("""
select preprocessed,target_code from example_utterances
where language =  %(language)s and type in ('online', 'online-bookkeeping', 'commandpedia')
and preprocessed <> ''""",
                                          language=self._language):
            self.add(row['preprocessed'], row['target_code'])
            n += 1
        tf.logging.info('Loaded %d exact matches for language %s', n, self._language)
            
    def add(self, utterance, target_json):
        self._trie.insert(utterance.split(' '), target_json.split(' '))
        
    def get(self, utterance):
        return self._trie.search(utterance.split(' '))
