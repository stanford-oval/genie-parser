#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
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
Created on Jul 15, 2017

@author: gcampagn
''' 

import os
import sys
import tensorflow as tf
import numpy as np
import tornado.ioloop

import ssl
import pwd, grp
try:
    from systemd import daemon as sd
except ImportError:
    sd = None

from concurrent.futures import ThreadPoolExecutor

from .application import Application
from .tokenizer import TokenizerService
from .config import ServerConfig

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(42)

    config = ServerConfig.load(('./server.conf',))
    
    if sys.version_info[2] >= 6:
        thread_pool = ThreadPoolExecutor(thread_name_prefix='query-thread-')
    else:
        thread_pool = ThreadPoolExecutor(max_workers=32)
    tokenizer_service = TokenizerService()
    app = Application(config, thread_pool, tokenizer_service)

    if config.ssl_key:
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(config.ssl_chain, config.ssl_key)
        app.listen(config.port, ssl_options=ssl_ctx)
    else:
        app.listen(config.port)
    
    if config.user:
        os.setgid(grp.getgrnam(config.user)[2])
        os.setuid(pwd.getpwnam(config.user)[2])

    if sd:
        sd.notify('READY=1')

    tokenizer_service.run()
    app.load_all_languages()

    sys.stdout.flush()
    tornado.ioloop.IOLoop.current().start()
