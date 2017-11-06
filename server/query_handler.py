'''
Created on Jul 1, 2017

@author: gcampagn
'''

import numpy as np
import tornado.web
import tornado.gen
import tornado.concurrent
import json
import sys
import traceback

from util.loader import vectorize, vectorize_constituency_parse

from . import json_syntax

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
        print("Input", tokens, values)

        results = []
        config = language.config
        grammar = config.grammar
        with language.session.as_default():
            with language.session.graph.as_default():
                input, input_len = vectorize(tokens, config.dictionary, config.max_length)
                parse_vector = vectorize_constituency_parse(parse, config.max_length, input_len)
                input_batch, input_length_batch, parse_batch = [input], [input_len], [parse_vector]
                sequences = language.model.predict_on_batch(language.session, input_batch, input_length_batch, parse_batch)
                assert len(sequences) == 1
                
                for i, decoded in enumerate(sequences[0]):
                    if i >= limit:
                        break
                    decoded = list(decoded)
                    try:
                        decoded = decoded[:decoded.index(grammar.end)]
                    except ValueError:
                        pass
                    decoded = [grammar.tokens[x] for x in decoded]
                    print("Beam", i+1, decoded)
                    try:
                        json_rep = dict(answer=json.dumps(json_syntax.to_json(decoded, grammar, values)), prob=1./len(sequences[0]), confidence=1)
                    except Exception as e:
                        print("Failed to represent " + str(decoded) + " as json", e)
                        traceback.print_exc(file=sys.stdout)
                        continue
                    results.append(json_rep)
        return results

    @tornado.gen.coroutine
    def get(self):
        query = self.get_query_argument("q")
        locale = self.get_query_argument("locale", default="en-US")
        language = self.application.get_language(locale)
        limit = int(self.get_query_argument("limit", default=5))
        print('GET /query', query)

        tokenized = yield language.tokenizer.tokenize(query)
        result = yield self._do_run_query(language, tokenized, limit)
        
        if len(result) > 0 and self.application.database:
            self.application.database.execute("insert into example_utterances (is_base, language, type, utterance, target_json, click_count) " +
                                              "values (0, %(language)s, 'log', %(utterance)s, %(target_json)s, -1)",
                                              language=language.tag,
                                              utterance=query,
                                              target_json=result[0]['answer'])
        sys.stdout.flush()
        self.write(dict(candidates=result, sessionId='X'))
