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
Created on Aug 2, 2017

@author: gcampagn
'''

import configparser

class ServerConfig():
    def __init__(self):
        self._config = configparser.ConfigParser()
        
        self._config['server'] = {
            'port': '8400',
            'user': '',
            'default_language': 'en',
            'admin_token': ''
        }
        
        self._config['db'] = {
            'url': '',
        }

        self._config['ssl'] = {
            'chain': '',
            'key': ''
        }
        
        self._config['models'] = {
            'en': './en/model'
        }
        
    @property
    def port(self):
        return int(self._config['server']['port'])

    @property
    def user(self):
        return self._config['server']['user']

    @property
    def default_language(self):
        return self._config['server']['default_language']
    
    @property
    def admin_token(self):
        return self._config['server']['admin_token']

    @property
    def db_url(self):
        return self._config['db']['url']

    @property
    def ssl_chain(self):
        return self._config['ssl']['chain']

    @property
    def ssl_key(self):
        return self._config['ssl']['key']

    @property
    def languages(self):
        return self._config['models'].keys()

    def get_model_directory(self, language):
        if language in self._config['models']:
            return self._config['models'][language]
        else:
            return './' + language + '/model'

    @staticmethod
    def load(filenames):
        self = ServerConfig()
        print('Loading server configuration from', filenames)
        self._config.read(filenames)
        return self
