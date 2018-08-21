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

class ExactMatcher():
    def __init__(self, database, language):
        self._database = database
        self._language = language
        
        self._dict = dict() 
    
    def load(self):
        n = 0
        for row in self._database.execute("""
select preprocessed,target_code from example_utterances
where language =  %(language)s and type in ('online', 'online-bookkeeping', 'commandpedia')
and preprocessed <> ''""",
                                          language=self._language):
            self._dict[row['preprocessed']] = row['target_code'].split(' ')
            n += 1
        print('Loaded %d exact matches for language %s' % (n, self._language))
            
    def add(self, utterance, target_json):
        self._dict[utterance] = target_json.split(' ')
        
    def get(self, utterance):
        return self._dict.get(utterance, None)
