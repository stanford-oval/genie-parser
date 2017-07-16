'''
Created on Jul 1, 2017

@author: gcampagn
'''

import itertools
import tornado.web
import tornado.gen
import tornado.concurrent
import json

from util.loader import vectorize

from thingtalk.grammar import UNITS
ALL_UNITS = set(itertools.chain(*UNITS.values()))

def _read_value(decoded, off, values):
    token = decoded[off]
    value = dict()
    consumed = 1
    if token == 'true':
        value['type'] = 'Bool'
        value['value'] = dict(value=True)
    elif token == 'false':
        value['type'] = 'Bool'
        value['value'] = dict(value=False)
    elif token == 'rel_home' or token == 'rel_work' or token == 'rel_current_location':
        value['type'] = 'Location'
        value['value'] = dict(relativeTag=token, latitude=-1., longitude=-1.)
    elif token.starts_with('LOCATION_'):
        value['type'] = 'Location'
        value['value'] = values[token]
    elif token.starts_with('tt:param.'):
        value['type'] = 'VarRef'
        value['value'] = dict(id=token)
    elif token.starts_with('QUOTED_STRING_'):
        value['type'] = 'String'
        value['value'] = values[token]
    elif token.starts_with('DATE_'):
        value['type'] = 'Date'
        value['value'] = values[token]
    elif token.starts_with('TIME_'):
        value['type'] = 'Time'
        value['value'] = values[token]
    elif token.starts_with('USERNAME_'):
        value['type'] = 'Entity(tt:username)'
        value['value'] = values[token]
    elif token.starts_with('HASHTAG_'):
        value['type'] = 'Entity(tt:hashtag)'
        value['value'] = values[token]
    elif token.starts_with('PHONE_NUMBER_'):
        value['type'] = 'Entity(tt:phone_number)'
        value['value'] = values[token]
    elif token.starts_with('EMAIL_ADDRESS_'):
        value['type'] = 'Entity(tt:email_address)'
        value['value'] = values[token]
    elif token.starts_with('URL_'):
        value['type'] = 'Entity(tt:url)'
        value['value'] = values[token]
    elif token.starts_with('DURATION_') or token.starts_with('SET_'):
        value['type'] = 'Measure(ms)'
        value['value'] = values[token]
    elif token.starts_with('NUMBER_'):
        if len(decoded) > off + 1 and decoded[off+1] in ALL_UNITS:
            value['type'] = 'Measure(' + decoded[off+1] + ')'
            consumed = 2
        else:
            value['type'] = 'Number'
        value['value'] = values[token]
    elif token.starts_with('GENERIC_ENTITY_'):
        entity_type = token[len('GENERIC_ENTITY_'):]
        value['type'] = 'Entity(' + entity_type + ')'
        value['value'] = values[token]
    else:
        # assume an enum, and hope for the best
        value['type'] = 'Enum'
        value['value'] = dict(value=token)

    return value, consumed

def _read_prim(decoded, off, values):
    fn = decoded[off]
    prim = dict(name=dict(id=fn), args=[])
    args = prim['args']
    consumed = 1
    while off + consumed < len(decoded) and decoded[off+consumed].starts_with('tt:param.'):
        pname = decoded[off+consumed]
        op = decoded[off+consumed+1]
        value, consumed_arg = _read_value(decoded, off+consumed, values)
        value['name'] = dict(id=pname)
        value['operator'] = op
        args.append(value)
        consumed += consumed_arg
    return prim, consumed

def _to_json(decoded, grammar, values):
    type = decoded[0]
    if type == 'special':
        return dict(special=dict(id=decoded[1]))
    elif type == 'answer':
        value, consumed = _read_value(decoded, 1, values)
        return dict(answer=value)
    elif type == 'command':
        if decoded[1] != 'type':
            raise ValueError('Invalid command type ' + decoded[1])
        if decoded[2] == 'generic':
            return dict(command=dict(type='help', value=dict(value='generic')))
        else:
            return dict(command=dict(type='help', value=dict(value=values[decoded[2]])))
    elif type in ('trigger', 'query', 'action'):
        # trigger, query, action
        rule = dict()
        prim, consumed = _read_prim(decoded, 1, values)
        rule[type] = prim
        return rule
    else:
        # rule
        rule = dict()
        off = 1
        trigger, consumed = _read_prim(decoded, off, values)
        rule['trigger'] = trigger
        off += consumed
        prim2, consumed = _read_prim(decoded, off, values)
        off += consumed
        if prim2['name']['id'] in grammar.functions['query']:
            rule['query'] = prim2
            if off < len(decoded):
                action, consumed = _read_prim(decoded, off, values)
            rule['action'] = action
        else:
            rule['action'] = prim2
        return dict(rule=rule)

class QueryHandler(tornado.web.RequestHandler):
    '''
    Handle /query
    '''
    def __init__(self, app, request):
        super().__init__(app, request)
        
        self.executor = self.app.thread_pool
    
    @tornado.concurrent.run_on_executor
    def _do_run_query(self, language, tokenized, limit):
        tokens = tokenized.tokens
        values = tokenized.values

        results = []
        with language.session.as_default():
            with language.session.graph.as_default():
                input, input_len = vectorize(tokens, language.input_words, language.config.max_length)
                input_batch, input_length_batch = [input], [input_len]
                sequences = language.model.predict_on_batch(language.session, input_batch, input_length_batch)
                assert len(sequences) == 1
                
                for beam in sequences[0]:
                    if len(results) >= limit:
                        break
                    decoded = language.config.grammar.decode_output(beam)
                    try:
                        decoded = decoded[:decoded.index(language.config.grammar.end)]
                    except ValueError:
                        pass
                    try:
                        results.append(dict(answer=json.dumps(_to_json(decoded, values)), prob=1./len(results), confidence=0))
                    except Exception as e:
                        print("Failed to represent " + str(decoded) + " as json", e)
        return results

    @tornado.gen.coroutine
    def get(self):
        query = self.get_query_argument("q")
        locale = self.get_query_argument("locale", default="en-US")
        language = self.application.get_language(locale)
        limit = int(self.get_query_argument("limit", default=5))

        tokenized = yield language.tokenizer.tokenize(query)
        result = yield self._do_run_query(language, tokenized)
        self.write(dict(candidates=result))
