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

class QueryHandler(tornado.web.RequestHandler):
    '''
    Handle /query
    '''
    def __init__(self, app, request):
        super().__init__(app, request)
        
        self.executor = app.thread_pool
    
    @tornado.concurrent.run_on_executor
    def _do_run_query(self, language, tokenized, limit):
        tokens, values, parse = tokenized

        results = []
        config = language.config
        grammar = config.grammar
        with language.session.as_default():
            with language.session.graph.as_default():
                input, input_len = vectorize(tokens, config.dictionary, config.max_length)
                if parse:
                    parse_vector = vectorize_constituency_parse(parse, config.max_length, input_len)
                else:
                    parse_vector = np.zeros((2*config.max_length-1,), dtype=np.bool)
                input_batch, input_length_batch, parse_batch = [input], [input_len], [parse_vector]
                sequences = language.model.predict_on_batch(language.session, input_batch, input_length_batch, parse_batch)
                assert len(sequences) == 1
                
                for i, beam in enumerate(sequences[0]):
                    if i >= limit:
                        break
                    decoded = grammar.reconstruct_program(beam, ignore_errors=True)
                    if not decoded:
                        continue
                    print("Beam", i+1, decoded)
                    json_rep = dict(code=decoded, score=1)
                    results.append(json_rep)
        return results

    @tornado.gen.coroutine
    def get(self, **kw):
        query = self.get_query_argument("q")
        locale = kw.get('locale', None) or self.get_query_argument("locale", default="en-US")
        language = self.application.get_language(locale)
        limit = int(self.get_query_argument("limit", default=5))
        expect = self.get_query_argument('expect', default=None)
        print('GET /%s/query' % locale, query)

        tokenized = yield language.tokenizer.tokenize(query)
        tokens, values, _ = tokenized
        print("Input", tokens, values)
        
        result = None
        if tokens[0].isupper() and len(tokens[0]) == 1:
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
        self.write(dict(candidates=result, tokens=tokens, entities=values))
        self.finish()
