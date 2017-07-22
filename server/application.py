'''
Created on Jul 1, 2017

@author: gcampagn
'''

import re
import tornado.web

from .request_handlers import QueryHandler

class LanguageContext(object):
    def __init__(self, tag, tokenizer, session, config, input_words, model):
        self.tag = tag
        self.tokenizer = tokenizer
        self.session = session
        self.config = config
        self.input_words = input_words
        self.model = model

class Application(tornado.web.Application):
    def __init__(self, thread_pool):
        super().__init__([(r"/query", QueryHandler)])
    
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
