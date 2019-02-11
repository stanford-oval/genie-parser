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
import json

import tensorflow as tf

from .query_handlers import QueryHandler, TokenizeHandler
from .learn_handler import LearnHandler
from .admin_handlers import ReloadHandler, ExactMatcherReload
from .exact import ExactMatcher
from .tokenizer import Tokenizer
from .predictor import Predictor


class LanguageContext(object):
    def __init__(self, tag, language_tag, model_tag, tokenizer, predictor):
        self.tag = tag
        self.language_tag = language_tag
        self.model_tag = model_tag
        self.tokenizer = tokenizer
        self.predictor = predictor


class Application(tornado.web.Application):
    def __init__(self, config, thread_pool, tokenizer_service):
        super().__init__([
            (r"/query", QueryHandler),
            (r"/learn", LearnHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/tokenize", TokenizeHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/query", QueryHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/learn", LearnHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/admin/reload", ReloadHandler),
            (r"/(?P<locale>[a-zA-Z-]+)/admin/exact/reload", ExactMatcherReload),
            (r"/@(?P<model_tag>[a-zA-Z0-9_\.-]+)/(?P<locale>[a-zA-Z-]+)/tokenize", TokenizeHandler),
            (r"/@(?P<model_tag>[a-zA-Z0-9_\.-]+)/(?P<locale>[a-zA-Z-]+)/query", QueryHandler),
            (r"/@(?P<model_tag>[a-zA-Z0-9_\.-]+)/(?P<locale>[a-zA-Z-]+)/learn", LearnHandler),
            (r"/@(?P<model_tag>[a-zA-Z0-9_\.-]+)/(?P<locale>[a-zA-Z-]+)/admin/reload", ReloadHandler),
            (r"/@(?P<model_tag>[a-zA-Z0-9_\.-]+)/(?P<locale>[a-zA-Z-]+)/admin/exact/reload", ExactMatcherReload),
        ])
    
        if config.db_url:
            self.database = sqlalchemy.create_engine(config.db_url, pool_recycle=600)
        else:
            self.database = None
        self.config = config
        self._languages = dict()
        self.thread_pool = thread_pool
        self._tokenizer = tokenizer_service
        
    def _load_language(self, language_tag, model_tag, model_dir):
        with tf.gfile.Open(os.path.join(model_dir, "model.json")) as fp:
            config = json.load(fp)

        tokenizer = Tokenizer(self._tokenizer, language_tag)
        predictor = Predictor(model_dir, config)

        if model_tag is not None:
            tag = '@%s/%s' % (model_tag, language_tag)
        else:
            tag = language_tag
        
        language = LanguageContext(tag, language_tag, model_tag, tokenizer, predictor)
        self._languages[tag] = language
        if self.database:
            language.exact = ExactMatcher(self.database, language_tag, model_tag)
            language.exact.load()
        else:
            language.exact = None
        if model_tag is not None:
            tf.logging.info('Loaded model @%s/%s', model_tag, language_tag)
        else:
            tf.logging.info('Loaded model @default/%s', language_tag)
            
    def load_all_languages(self):
        for tag in self.config.languages:
            if tag.startswith('@'):
                model_tag, language_tag = tag.split('/')
                model_tag = model_tag[1:]
            else:
                language_tag = tag
                model_tag = None
            self._load_language(language_tag, model_tag, self.config.get_model_directory(tag))
    
    def reload_language(self, language_tag, model_tag=None):
        if model_tag is not None:
            tag = '@%s/%s' % (model_tag, language_tag)
            tf.logging.info('Reloading model @%s/%s', model_tag, language_tag)
        else:
            tag = language_tag
            tf.logging.info('Reloading model @default/%s', language_tag) 
        self._load_language(language_tag, model_tag, self.config.get_model_directory(tag))
    
    def get_language(self, locale, model_tag=None):
        '''
        Convert a locale tag into a preloaded language
        '''

        if model_tag == 'default':
            model_tag = None
        
        split_tag = re.split("[_\\.\\-]", locale)
        
        # try with language and country
        language = None
        if len(split_tag) >= 2:
            key = split_tag[0] + "-" + split_tag[1]
            if model_tag is not None:
                key = '@' + model_tag + '/' + key
            language = self._languages.get(key, None)
        if language is None and len(split_tag) >= 1:
            key = split_tag[0]
            if model_tag is not None:
                key = '@' + model_tag + '/' + key
            language = self._languages.get(key, None)

        # fallback to english if the language is not recognized or
        # locale was not specified
        if language:
            return language
        else:
            if model_tag is not None:
                key = '@' + model_tag + '/' + self.config.default_language
                language = self._languages.get(key, None)
                if language is not None:
                    return language
                else:
                    tf.logging.warning("Ignored model tag " + model_tag)
            
            return self._languages[self.config.default_language]
