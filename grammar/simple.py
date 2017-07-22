'''
Created on Jul 20, 2017

@author: gcampagn
'''

from .abstract import AbstractGrammar

class SimpleGrammar(AbstractGrammar):
    '''
    A simple implementation of AbstractGrammar, that reads the 
    sequence of tokens from a given file (one per line)
    
    The resulting grammar is:
    
    $ROOT -> $Token *
    
    where $Token is any grammar token
    '''
    
    def __init__(self, filename):
        super().__init__()
        
        self.tokens = ['<<PAD>>', '<<EOS>>', '<<GO>>']
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                self.tokens.append(line.strip())

        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i