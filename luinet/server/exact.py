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
import code
'''
Created on Nov 6, 2017

@author: gcampagn
'''

import tensorflow as tf

from ..util.strings import find_span
from ..util.trie import Trie, WILDCARD

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
select preprocessed,target_code from example_utterances use index (language_flags)
where language =  %(language)s and find_in_set('exact', flags) and not is_base and preprocessed <> ''
order by type asc, id desc""",
                                          language=self._language):
            self.add(row['preprocessed'], row['target_code'])
            n += 1
        tf.logging.info('Loaded %d exact matches for language %s', n, self._language)
    
    def add(self, utterance, target_code):
        utterance = utterance.split(' ') 
        target_code = target_code.split(' ')

        in_string = False
        span_begin = None
        for i, token in enumerate(target_code):
            if token != '"':
                continue
            in_string = not in_string
            if in_string:
                span_begin = i+1
            else:
                span_end = i
                span = target_code[span_begin:span_end]
                begin_index, end_index = find_span(utterance, span)

                # find_span returns inclusive indices (because the NN likes those better)
                for j in range(begin_index, end_index + 1):
                    utterance[j] = WILDCARD
                for j in range(span_begin, span_end):
                    target_code[j] = begin_index + j - span_begin
        
        self._trie.insert(utterance, target_code)
        
    def get(self, utterance):
        utterance = utterance.split(' ')
        
        code = self._trie.search(utterance)
        if code is None:
            return code
        code = list(code)
        for i, token in enumerate(code):
            if isinstance(token, int):
                code[i] = utterance[token]
        return code