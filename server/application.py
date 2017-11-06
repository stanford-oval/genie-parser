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
'''
Created on Jul 1, 2017

@author: gcampagn
'''

import re
import tornado.web
import sqlalchemy

from .query_handler import QueryHandler
from .learn_handler import LearnHandler

class LanguageContext(object):
    def __init__(self, tag, tokenizer, session, config, model):
        self.tag = tag
        self.tokenizer = tokenizer
        self.session = session
        self.config = config
        self.model = model

class Application(tornado.web.Application):
    def __init__(self, config, thread_pool):
        super().__init__([(r"/query", QueryHandler), (r"/learn", LearnHandler)])
    
        if config.db_url:
            self.database = sqlalchemy.create_engine(config.db_url, pool_recycle=3600)
        else:
            self.database = None
        self.config = config
        self._languages = dict()
        self.thread_pool = thread_pool
        
    def add_language(self, tag, language):
        self._languages[tag] = language
    
    def get_language(self, locale):
        '''
        Convert a locale tag into a preloaded language
        '''
        
        split_tag = re.split("[_\\.\\-]", locale)
        # try with language and country
        language = None
        if len(split_tag) >= 2:
            language = self._languages.get(split_tag[0] + "-" + split_tag[1], None)
        if language is None and len(split_tag) >= 1:
            language = self._languages.get(split_tag[0], None)

        # fallback to english if the language is not recognized or
        # locale was not specified
        if language:
            return language
        else:
            return language['en']
