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

from .constants import LATEST_THINGTALK_VERSION, DEFAULT_THINGTALK_VERSION

entity_re = re.compile('^[A-Z_]+_[0-9]')
def check_program_entities(program, entities):
    for token in program:
        if entity_re.match(token) or token.startswith('GENERIC_ENTITY_'):
            if not token in entities:
                return False
    return True

class LearnHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self, locale='en-US', model_tag=None, **kw):
        self.set_header('Access-Control-Allow-Origin', '*')

        query = self.get_argument("q")
        language = self.application.get_language(locale, model_tag)
        target_code = self.get_argument("target")
        store = self.get_argument("store", "automatic")
        owner = self.get_argument("owner", None) or None
        thingtalk_version = self.get_argument("thingtalk_version",
                                              DEFAULT_THINGTALK_VERSION)
        #print('POST /%s/learn' % locale, target_code)
        
        grammar = language.predictor.problem.grammar
        
        tokenized = yield language.tokenizer.tokenize(query)
        if len(tokenized.tokens) == 0:
            raise tornado.web.HTTPError(400, reason="Refusing to learn an empty sentence")

        # if the client is out of date, don't even try to parse the code
        # (as it might have changed meaning in the newer version of ThingTalk
        # anyway)
        if thingtalk_version != LATEST_THINGTALK_VERSION:
            self.write(dict(result="Ignored request from older ThingTalk"))
            self.finish()
            return

        sequence = target_code.split(' ')
        try:
            vectorized = grammar.tokenize_to_vector(tokenized.tokens, sequence)
            grammar.verify_program(vectorized)
        except ValueError as e:
            print(e)
            raise tornado.web.HTTPError(400, reason="Invalid ThingTalk")

        if not check_program_entities(sequence, tokenized.values):
            raise tornado.web.HTTPError(400, reason="Missing entities")
        preprocessed = ' '.join(tokenized.tokens)
        
        if store == 'no':
            # do nothing, successfully
            self.finish(dict(result="Learnt successfully"))
            return
        
        if not store in ('automatic', 'online', 'commandpedia'):
            raise tornado.web.HTTPError(400, reason="Invalid store parameter")
        if store == 'online' and sequence[0] == 'bookkeeping':
            store = 'online-bookkeeping'
        
        training_flag = store in ('online', 'online-bookkeeping', 'commandpedia')
        
        if not self.application.database:
            raise tornado.web.HTTPError(500, "Server not configured for online learning")
        self.application.database.execute("insert into example_utterances (is_base, language, type, flags, utterance, preprocessed, target_json, target_code, click_count, owner) " +
                                          "values (0, %(language)s, %(type)s, %(flags)s, %(utterance)s, %(preprocessed)s, '', %(target_code)s, 10, %(owner)s)",
                                          language=language.tag,
                                          utterance=query,
                                          preprocessed=preprocessed,
                                          type=store,
                                          flags=('training,exact' if training_flag else ''),
                                          target_code=target_code,
                                          owner=owner)
        if training_flag:
            # insert a second copy of the sentence with the "replaced" flag
            self.application.database.execute("insert into example_utterances (is_base, language, type, flags, utterance, preprocessed, target_json, target_code, click_count, owner) " +
                                              "values (0, %(language)s, %(type)s, %(flags)s, %(utterance)s, %(preprocessed)s, '', %(target_code)s, 0, %(owner)s)",
                                              language=language.tag,
                                              utterance=query,
                                              preprocessed=preprocessed,
                                              type=store,
                                              flags='training,replaced',
                                              target_code=target_code,
                                              owner=owner)

        if language.exact and training_flag:
            language.exact.add(preprocessed, target_code)
        self.write(dict(result="Learnt successfully"))
        self.finish()
