'''
Created on Jul 16, 2017

@author: gcampagn
'''

import socket
import os
import sys

import json
import tornado.gen
import tornado.ioloop
from tornado.iostream import IOStream, StreamClosedError
from tornado.concurrent import Future

PORT = 8888

class TokenizerService(object):
    '''
    Wraps the IPC to the Java TokenizerService (which runs tokenization and named
    entity extraction through CoreNLP)
    '''

    def __init__(self):
        self._socket = IOStream(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        self._requests = dict()
        self._next_id = 0
        
    @tornado.gen.coroutine
    def run(self):
        yield self._socket.connect(('127.0.0.1', PORT))
        
        while True:
            try:
                response = yield self._socket.read_until(b'\n')
            except StreamClosedError:
                response = None
            if not response:
                return
            response = json.loads(response)
            
            id = int(response['req'])
            self._requests[id].set_result((response['tokens'], response['values'], response['constituencyParse']))
            del self._requests[id]
        
    def tokenize(self, language_tag, query):
        id = self._next_id
        self._next_id += 1
        
        req = dict(req=id, utterance=query, languageTag=language_tag)
        outer = Future()
        self._requests[id] = outer
        
        def then(future):
            if future.exception():
                outer.set_exception(future.exception())
                del self._requests[id]
        
        future = self._socket.write(json.dumps(req).encode())
        future.add_done_callback(then)
        return outer
        
    
class Tokenizer(object):
    def __init__(self, service, language_tag):
        self._service = service
        self._lang = language_tag
        
    def tokenize(self, query):
        return self._service.tokenize(self._lang, query)