#!/usr/bin/python3
'''
Created on Jul 15, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf
import tornado.ioloop
import configparser
from concurrent.futures import ThreadPoolExecutor

from models import Config, create_model

from server.application import Application, LanguageContext
from server.tokenizer import Tokenizer, TokenizerService
from server.config import ServerConfig

def load_language(app, tokenizer_service, tag, model_dir):
    config = Config.load(['./default.conf', './default.' + tag + '.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)
    
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        # Force everything to run on CPU, we run on single inputs so there is not much point
        # on going through the GPU
        with tf.device('/cpu:0'):
            model.build()
            loader = tf.train.Saver()

        with session.as_default():
            loader.restore(session, os.path.join(model_dir, 'best'))
    tokenizer = Tokenizer(tokenizer_service, tag)
    app.add_language(tag, LanguageContext(tag, tokenizer, session, config, model))
    print('Loaded language ' + tag)

def run():
    if len(sys.argv) < 2:
        print("** Usage: python3 " + sys.argv[0] + " <<Language:Model Dir>>")
        sys.exit(1)
    
    np.random.seed(42)
    config = ServerConfig.load(('./server.conf',))
    
    if sys.version_info[2] >= 6:
        thread_pool = ThreadPoolExecutor(thread_name_prefix='query-thread-')
    else:
        thread_pool = ThreadPoolExecutor()
    app = Application(config, thread_pool)
    app.listen(config.port)
    tokenizer_service = TokenizerService()
    tokenizer_service.run()
    
    for language, model_directory in map(lambda x : x.split(':'), sys.argv[1:]):
        load_language(app, tokenizer_service, language, model_directory)

    tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    run()
