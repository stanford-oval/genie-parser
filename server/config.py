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