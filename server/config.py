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
Created on Aug 2, 2017

@author: gcampagn
'''

import configparser

class ServerConfig():
    def __init__(self):
        self._config = configparser.ConfigParser()
        
        self._config['server'] = {
            'port': '8400',
        }
        
        self._config['db'] = {
            'url': 'mysql://sempre:sempre@thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine?charset=utf8mb4',
        }
        
    @property
    def port(self):
        return int(self._config['server']['port'])
    
    @property
    def db_url(self):
        return self._config['db']['url']
        
    @staticmethod
    def load(filenames):
        self = ServerConfig()
        print('Loading server configuration from', filenames)
        self._config.read(filenames)
        return self