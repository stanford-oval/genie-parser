# Copyright 2018 The Board of Trustees of the Leland Stanford Junior University
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
Created on Oct 23, 2018

@author: gcampagn
'''

WILDCARD = object()


class TrieNode:
    def __init__(self):
        self.value = None
        self.children = dict()

    def add_value(self, value, limit):
        if self.value is None:
            self.value = []
        self.value.insert(0, value)
        self.value = self.value[:limit]

    def add_child(self, key):
        child = TrieNode()
        self.children[key] = child
        return child
    
    def get_child(self, key, allow_wildcard=False):
        child = self.children.get(key, None)
        if allow_wildcard and child is None:
            child = self.children.get(WILDCARD, None)
        return child

class Trie:
    '''A simple Trie-based key-value store.
    '''
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, sequence, value, limit):
        node = self.root
        for key in sequence:
            child = node.get_child(key)
            if child is None:
                child = node.add_child(key)
            node = child
        node.add_value(value, limit)
    
    def search(self, sequence):
        node = self.root
        for key in sequence:
            child = node.get_child(key, allow_wildcard=True)
            if child is None:
                return None
            node = child
        return node.value