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
import semver

from .constants import LATEST_THINGTALK_VERSION, DEFAULT_THINGTALK_VERSION

class TokenizeHandler(tornado.web.RequestHandler):
    '''
    Handle /tokenize
    '''

    @tornado.gen.coroutine
    def get(self, locale='en-US', model_tag=None, **kw):
        self.set_header('Access-Control-Allow-Origin', '*')

        query = self.get_query_argument("q")
        language = self.application.get_language(locale, model_tag)
        
        #print('GET /%s/tokenize' % locale, query)
        tokenized = yield language.tokenizer.tokenize(query)
        #print("Tokenized", tokenized.tokens, tokenized.values)
        
        sys.stdout.flush()
        cache_time = 3600
        self.set_header("Expires", datetime.datetime.utcnow() + datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "public,max-age=" + str(cache_time))
        self.write(dict(tokens=tokenized.tokens, entities=tokenized.values))
        self.finish()


def clean_tokens(tokens):
    for t in tokens:
        if t[0].isupper() and '*' in t:
            yield t.rsplit('*', maxsplit=1)[0]
        else:
            yield t


def _pad_to_batch(batch):
    max_len = max(len(tokens) for tokens in batch)
    batch_size = len(batch)

    matrix = np.empty((batch_size, max_len), dtype=np.object)
    for i in range(batch_size):
        item_len = len(batch[i])
        matrix[i, :item_len] = batch[i]
        matrix[i, item_len:] = np.zeros((max_len - item_len,),
                                        dtype=np.str)
    return matrix


class QueryHandler(tornado.web.RequestHandler):
    '''
    Handle /query
    '''
    def __init__(self, app, request):
        super().__init__(app, request)
        
        self.executor = app.thread_pool
    
    @tornado.concurrent.run_on_executor
    def _do_run_query(self, language, tokenized, limit):
        tokens = list(clean_tokens(tokenized.tokens))
        
        # ignore the constituency parse
        # parse = tokenized.constituency_parse

        predicted = language.predictor.predict({
            # wrap into a batch of 1
            "inputs/string": [tokens]
        })
        outputs = predicted["outputs"][0]
        
        if len(outputs.shape) == 1:
            # add beam dimension if we're using greedy decoding
            outputs = np.expand_dims(outputs, axis=0)
            scores = [1]
        else:
            scores = predicted["scores"][0]
        
        results = []
        for decoded, score in zip(outputs, scores):
            decoded = [x.decode('utf-8') for x in decoded if x != b'']
            if len(decoded) == 0:
                # grammar error, skip
                continue
            json_rep = dict(code=decoded, score=float(score))
            results.append(json_rep)
            if limit >= 0 and len(results) >= limit:
                break
        return results
    
    @tornado.concurrent.run_on_executor
    def _run_retrieval_query(self, language, tokens, choices, limit):
        choice_list = list(choices.keys())
        
        predicted = language.predictor.predict({
            # wrap into a single batch both input and choices
            "inputs/string": _pad_to_batch([tokens] +
                                           [choices[c_id].tokens for c_id in choice_list])
        }, signature_key="encoded_inputs")
        encoded = predicted["encoded_inputs"]

        input_encoded = encoded[0]
        input_norm = np.linalg.norm(input_encoded, ord=2)
        
        def try_one_choice(i):
            choice_id = choice_list[i]
            choice_encoded = encoded[i+1]
            choice_norm = np.linalg.norm(choice_encoded, ord=2)
            
            choice_score = np.dot(input_encoded, choice_encoded)
            choice_score /= input_norm
            choice_score /= choice_norm
            
            return dict(code=['bookkeeping', 'choice', str(choice_id)], score=float(choice_score))
        
        results = [try_one_choice(i) for i in range(len(choice_list))]
        results.sort(key=lambda x: -x['score'])
        results = results[:limit]
        return results

    def _apply_compatibility(self, results, thingtalk_version):
        if semver.match(thingtalk_version, "<1.3.0"):
            # convert stream-join "=>" to "join" 
            
            for result in results:
                code = result['code']
                if len(code) == 0 or code[0] == 'policy':
                    continue
                
                start = 0
                if code[0] == 'executor':
                    while code[start] != ':':
                        start += 1
                    start += 1
                
                has_now = code[start] == 'now'
                if has_now:
                    start += 2 # "now" & "=>"

                last_arrow = None
                for i in range(len(code)-1, start, -1):
                    if code[i] == '=>':
                        last_arrow = i
                        break
                if last_arrow is None:
                    continue
                
                for i in range(start, last_arrow):
                    if code[i] == '=>':
                        code[i] = 'join'

    @tornado.gen.coroutine
    def get(self, model_tag=None, **kw):
        self.set_header('Access-Control-Allow-Origin', '*')

        query = self.get_query_argument("q")
        locale = kw.get('locale', None) or self.get_query_argument("locale", default="en-US")
        store = self.get_query_argument("store", "yes")
        language = self.application.get_language(locale, model_tag)
        thingtalk_version = self.get_argument("thingtalk_version",
                                              DEFAULT_THINGTALK_VERSION)
        try:
            limit = int(self.get_query_argument("limit", default=5))
        except ValueError:
            raise tornado.web.HTTPError(400, reason='Invalid limit argument')
        if store not in ('yes','no'):
            raise tornado.web.HTTPError(400, reason='Invalid store argument')
        expect = self.get_query_argument('expect', default=None)
        #print('GET /%s/query' % locale, query)

        tokenized = yield language.tokenizer.tokenize(query, expect)
        #print("Tokenized", tokenized.tokens, tokenized.values)
        
        result = None
        exact = None
        tokens = tokenized.tokens
        if len(tokens) == 0:
            result = [dict(code=['bookkeeping', 'special', 'special:failed'], score='Infinity')]
        elif len(tokens) == 1 and (tokens[0][0].isupper() or tokens[0] in ('1', '0')):
            # if the whole input is just an entity, return that as an answer
            result = [dict(code=['bookkeeping', 'answer', tokens[0]], score='Infinity')]
        elif expect == 'MultipleChoice':
            choices = dict()
            for arg in self.request.query_arguments:
                if arg == 'choices[]':
                    for choice in self.get_query_arguments(arg):
                        choices[len(choices)] = yield language.tokenizer.tokenize(choice, expect)
                elif arg.startswith('choices['):
                    choices[arg[len('choices['):-1]] = yield language.tokenizer.tokenize(self.get_query_argument(arg), expect)
            
            result = yield self._run_retrieval_query(language, tokens, choices, limit)
        elif result is None and language.exact:
            exact = language.exact.get(' '.join(tokens))
                
        if result is None:
            result = yield self._do_run_query(language, tokenized, limit)
        
        if self.application.database and store != 'no' and expect != 'MultipleChoice' and len(tokens) > 0:
            self.application.database.execute("insert into utterance_log (language, preprocessed, target_code) " +
                                              "values (%(language)s, %(preprocessed)s, %(target_code)s)",
                                              language=language.tag,
                                              preprocessed=' '.join(tokens),
                                              target_code=' '.join(result[0]['code'] if len(result) > 0 else []))

        if exact is not None:
            result = [dict(code=x, score='Infinity') for x in exact] + result
        
        self._apply_compatibility(result, thingtalk_version)
        
        sys.stdout.flush()
        #cache_time = 3600
        #self.set_header("Expires", datetime.datetime.utcnow() + datetime.timedelta(seconds=cache_time))
        #self.set_header("Cache-Control", "public,max-age=" + str(cache_time))
        self.set_header("Cache-Control", "no-store,must-revalidate")
        self.write(dict(candidates=result, tokens=tokens, entities=tokenized.values))
        self.finish()
