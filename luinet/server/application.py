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
Created on Jul 1, 2017

@author: gcampagn
'''

import re
import tornado.web
import sqlalchemy
import os
import tensorflow as tf

from models import Config, create_model

from .query_handlers import QueryHandler, TokenizeHandler
from .learn_handler import LearnHandler
from .admin_handlers import ReloadHandler
from .exact import ExactMatcher
from .tokenizer import Tokenizer

class LanguageContext(object):
    def __init__(self, tag, tokenizer, session, config, model):
        self.tag = tag
        self.tokenizer = tokenizer
        self.session = session
        self.config = config
        self.model = model

class Application(tornado.web.Application):
    def __init__(self, config, thread_pool, tokenizer_service):
        super().__init__([
            (r"/query", QueryHandler),
            (r"/learn", LearnHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/tokenize", TokenizeHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/query", QueryHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/learn", LearnHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/admin/reload", ReloadHandler)
        ])
    
        if config.db_url:
            self.database = sqlalchemy.create_engine(config.db_url, pool_recycle=600)
        else:
            self.database = None
        self.config = config
        self._languages = dict()
        self.thread_pool = thread_pool
        self._tokenizer = tokenizer_service
        
    def _load_language(self, tag, model_dir):
        config = Config.load(['./default.conf', './default.' + tag + '.conf', os.path.join(model_dir, 'model.conf')])
        model = create_model(config)
        
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with graph.as_default():
            tf.set_random_seed(1234)
            
            # Force everything to run on CPU, we run on single inputs so there is not much point
            # on going through the GPU
            with tf.device('/cpu:0'):
                model.build()
                loader = tf.train.Saver()
    
            with session.as_default():
                loader.restore(session, os.path.join(model_dir, 'best'))

        tokenizer = Tokenizer(self._tokenizer, tag)
        language = LanguageContext(tag, tokenizer, session, config, model)
        self._languages[tag] = language
        if self.database:
            language.exact = ExactMatcher(self.database, tag)
            language.exact.load()
        else:
            language.exact = None
        print('Loaded language ' + tag)
            
    def load_all_languages(self):
        for tag in self.config.languages:
            self._load_language(tag, self.config.get_model_directory(tag))
    
    def reload_language(self, tag):
        print('Reloading language ' + tag)
        self._load_language(tag, self.config.get_model_directory(tag))
    
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
            return self._languages[self.config.default_language]
