#!/usr/bin/python3
#
# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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
Created on Jul 15, 2017

@author: gcampagn
''' 

import os
import sys
import numpy as np
import tensorflow as tf
import tornado.ioloop
import configparser
import ssl
import pwd, grp
try:
    from systemd import daemon as sd
except ImportError:
    sd = None

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
        thread_pool = ThreadPoolExecutor(max_workers=32)
    app = Application(config, thread_pool)

    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(config.ssl_chain, config.ssl_key)
    app.listen(config.port, ssl_options=ssl_ctx)
    os.setgid(grp.getgrnam(config.user)[2])
    os.setuid(pwd.getpwnam(config.user)[2])

    if sd:
        sd.notify('READY=1')

    tokenizer_service = TokenizerService()
    tokenizer_service.run()
    
    for language, model_directory in map(lambda x : x.split(':'), sys.argv[1:]):
        load_language(app, tokenizer_service, language, model_directory)

    sys.stdout.flush()
    tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    run()
