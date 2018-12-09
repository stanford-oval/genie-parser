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

TRAINABLE_TYPES = ('online', 'online-bookkeeping', 'commandpedia')

class LearnHandler(tornado.web.RequestHandler):
    @tornado.concurrent.run_on_executor
    def _save_to_db(self, languageTag, utterance, preprocessed, target_code, store, owner):
        training_flag = store in TRAINABLE_TYPES
        
        with self.application.database.begin() as conn:
            if len(owner) == 8:
                lookup_result = conn.execute("select id from users where cloud_id = %(cloud_id)s", cloud_id=owner).first()
                if lookup_result is None:
                    raise tornado.web.HTTPError(400, reason="Invalid command owner")
                owner_id = lookup_result['id']
            else:
                try:
                    owner_id = int(owner)
                except:
                    raise tornado.web.HTTPError(400, reason="Invalid command owner")
            
            result = conn.execute("insert into example_utterances (is_base, language, type, flags, utterance, preprocessed, target_json, target_code, click_count, owner, like_count) " +
                                  "values (0, %(language)s, %(type)s, %(flags)s, %(utterance)s, %(preprocessed)s, '', %(target_code)s, 1, %(owner)d, %(like_count)d)",
                                  language=languageTag,
                                  utterance=utterance,
                                  preprocessed=preprocessed,
                                  type=store,
                                  flags=('training,exact' if training_flag else ''),
                                  target_code=target_code,
                                  owner=owner_id,
                                  like_count=1 if store == 'commandpedia' else 0)
            example_id = result.lastrowid
            if training_flag:
                # insert a second copy of the sentence with the "replaced" flag
                conn.execute("insert into replaced_example_utterances (language, type, flags, preprocessed, target_code) " +
                             "values (%(language)s, %(type)s, %(flags)s, %(preprocessed)s, %(target_code)s)",
                             language=languageTag,
                             preprocessed=preprocessed,
                             type=store,
                             flags='training,exact',
                             target_code=target_code)
            if store == 'commandpedia':
                conn.execute("insert into example_likes(example_id, user_id) values (%(example_id)d, %(user_id)d)",
                             example_id=example_id,
                             user_id=owner_id)
            return example_id
    
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
        if store == 'commandpedia' and owner is None:
            raise tornado.web.HTTPError(400, reason="Missing owner for commandpedia command")
        
        training_flag = store in TRAINABLE_TYPES
        
        if not self.application.database:
            raise tornado.web.HTTPError(500, "Server not configured for online learning")
        
        example_id = yield self._save_to_db(language.tag, query, preprocessed, target_code, store, owner)

        if language.exact and training_flag:
            language.exact.add(preprocessed, target_code)
        self.write(dict(result="Learnt successfully", example_id=example_id))
        self.finish()
