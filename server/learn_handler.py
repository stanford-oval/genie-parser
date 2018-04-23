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
Created on Aug 2, 2017

@author: gcampagn
'''

import tornado.web
import re

entity_re = re.compile('^[A-Z_]+_[0-9]')
def check_program_entities(program, entities):
    for token in program:
        if entity_re.match(token) or token.startswith('GENERIC_ENTITY_'):
            if not token in entities:
                return False
    return True

class LearnHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self, locale='en-US', **kw):
        self.set_header('Access-Control-Allow-Origin', '*')

        query = self.get_argument("q")
        language = self.application.get_language(locale)
        target_code = self.get_argument("target")
        store = self.get_argument("store", "automatic")
        #print('POST /%s/learn' % locale, target_code)
        
        grammar = language.config.grammar
        sequence = target_code.split(' ')
        try:
            program_vector, program_length = grammar.vectorize_program(sequence, max_length=language.config.max_length)
        except ValueError:
            raise tornado.web.HTTPError(400, reason="Invalid ThingTalk")
        
        tokenized = yield language.tokenizer.tokenize(query)
        if not check_program_entities(sequence, tokenized.values):
            raise tornado.web.HTTPError(400, reason="Missing entities")
        preprocessed = ' '.join(tokenized.tokens)
        
        if store == 'no':
            # do nothing, successfully
            self.finish(dict(result="Learnt successfully"))
            return
        
        if not store in ('automatic', 'online'):
            raise tornado.web.HTTPError(400, reason="Invalid store parameter")
        if store == 'online' and sequence[0] == 'bookkeeping':
            store = 'online-bookkeeping'
        
        if not self.application.database:
            raise tornado.web.HTTPError(500, "Server not configured for online learning")
        self.application.database.execute("insert into example_utterances (is_base, language, type, utterance, preprocessed, target_json, target_code, click_count) " +
                                          "values (0, %(language)s, %(type)s, %(utterance)s, %(preprocessed)s, '', %(target_code)s, -1)",
                                          language=language.tag,
                                          utterance=query,
                                          preprocessed=preprocessed,
                                          type=store,
                                          target_code=target_code)
        if language.exact and store in ('online', 'online-bookkeeping'):
            language.exact.add(preprocessed, target_code)
        self.write(dict(result="Learnt successfully"))
        self.finish()
