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
Created on Jul 1, 2017

@author: gcampagn
'''

import numpy as np
import tornado.web
import tornado.gen
import tornado.concurrent
import sys
import datetime

from util.loader import vectorize, vectorize_constituency_parse

class TokenizeHandler(tornado.web.RequestHandler):
    '''
    Handle /tokenize
    '''

    @tornado.gen.coroutine
    def get(self, locale='en-US', **kw):
        query = self.get_query_argument("q")
        language = self.application.get_language(locale)
        
        print('GET /%s/tokenize' % locale, query)
        tokenized = yield language.tokenizer.tokenize(query)
        print("Tokenized", tokenized.tokens, tokenized.values)
        
        sys.stdout.flush()
        cache_time = 3600
        self.set_header("Expires", datetime.datetime.utcnow() + datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "public,max-age=" + str(cache_time))
        self.write(dict(tokens=tokenized.tokens, entities=tokenized.values))
        self.finish()


class QueryHandler(tornado.web.RequestHandler):
    '''
    Handle /query
    '''
    def __init__(self, app, request):
        super().__init__(app, request)
        
        self.executor = app.thread_pool
    
    @tornado.concurrent.run_on_executor
    def _do_run_query(self, language, tokenized, limit):
        tokens = tokenized.tokens
        parse = tokenized.constituency_parse

        results = []
        config = language.config
        grammar = config.grammar
        with language.session.as_default():
            with language.session.graph.as_default():
                input, input_len = vectorize(tokens, config.dictionary, config.max_length, add_eos=True, add_start=True)
                print('Vectorized', input, input_len)
                if parse:
                    parse_vector = vectorize_constituency_parse(parse, config.max_length, input_len)
                else:
                    parse_vector = np.zeros((2*config.max_length-1,), dtype=np.bool)
                input_batch, input_length_batch, parse_batch = [input], [input_len], [parse_vector]
                sequences = language.model.predict_on_batch(language.session, input_batch, input_length_batch, parse_batch)
                assert len(sequences) == 1
                
                for i, beam in enumerate(sequences[0]):
                    decoded = grammar.reconstruct_program(beam, ignore_errors=True)
                    print("Beam", i+1, decoded if decoded else 'failed to predict')
                    if not decoded:
                        continue
                    json_rep = dict(code=decoded, score=1)
                    results.append(json_rep)
                    if limit >= 0 and len(results) >= limit:
                        break
        return results

    @tornado.gen.coroutine
    def get(self, **kw):
        query = self.get_query_argument("q")
        locale = kw.get('locale', None) or self.get_query_argument("locale", default="en-US")
        language = self.application.get_language(locale)
        try:
            limit = int(self.get_query_argument("limit", default=5))
        except ValueError:
            raise tornado.web.HTTPError(400, reason='Invalid limit argument')
        expect = self.get_query_argument('expect', default=None)
        print('GET /%s/query' % locale, query)

        tokenized = yield language.tokenizer.tokenize(query)
        print("Tokenized", tokenized.tokens, tokenized.values)
        
        result = None
        tokens = tokenized.tokens
        if len(tokens) == 1 and tokens[0].isupper():
            # if the whole input is just an entity, return that as an answer
            result = [dict(code=['bookkeeping', 'answer', tokens[0]], score='Infinity')]
        if result is None and language.exact:
            exact = language.exact.get(' '.join(tokens))
            if exact:
                result = [dict(code=exact, score='Infinity')]
                
        if result is None:
            result = yield self._do_run_query(language, tokenized, limit)
        
        if len(result) > 0 and self.application.database:
            self.application.database.execute("insert into example_utterances (is_base, language, type, utterance, preprocessed, target_json, target_code, click_count) " +
                                              "values (0, %(language)s, 'log', %(utterance)s, %(preprocessed)s, '', %(target_code)s, -1)",
                                              language=language.tag,
                                              utterance=query,
                                              preprocessed=' '.join(tokens),
                                              target_code=' '.join(result[0]['code']))
        
        sys.stdout.flush()
        cache_time = 3600
        self.set_header("Expires", datetime.datetime.utcnow() + datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "public,max-age=" + str(cache_time))
        self.write(dict(candidates=result, tokens=tokens, entities=tokenized.values))
        self.finish()
