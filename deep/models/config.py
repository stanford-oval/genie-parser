'''
Created on Jul 20, 2017

@author: gcampagn
'''

import configparser
import grammar

from util.loader import load_dictionary, load_embeddings

class Config(object):
    def __init__(self):
        self._config = configparser.ConfigParser()
        
        self._config['model'] = {
            'model_type': 'seq2seq',
            'hidden_size': 300,
            'rnn_cell_type': 'lstm',
            'rnn_layers': 1,
            'apply_attention': 'true'
        }
        self._config['training'] = {
            'batch_size': 256,
            'n_epochs': 20,
            'learning_rate': 0.001,
            'dropout': 0.5
        }
        self._config['input'] = {
            'input_words': './input_words.txt',
            'input_embeddings': './embeddings.txt',
            'input_embed_size': 300,
            'max_length': 60,
            'train_input_embeddings': 'false'
        }
        self._config['output'] = {
            'grammar': 'tt',
            'grammar_input_file': './thingpedia.txt',
            'train_output_embeddings': 'false',
            'output_embed_size': 50,
            'use_grammar_constraints': 'false',
            'use_beam_decode': 'false',
            'beam_width': 10
        }
        
        self._grammar = None
        self._words = None
        self._reverse = None
        self._embeddings_matrix = None
            
    @property
    def model_type(self):
        return self._config['model']['model_type']
    
    @property
    def max_length(self):
        return int(self._config['input']['max_length'])
    
    @property
    def dropout(self):
        return float(self._config['training']['dropout'])
    
    @property
    def embed_size(self):
        return int(self._config['input']['input_embed_size'])
    
    @property
    def output_embed_size(self):
        if self._config['output'].getboolean('train_output_embeddings'):
            return int(self._config['input']['output_embed_size'])
        else:
            return self._grammar.output_size
        
    @property
    def output_size(self):
        return self._grammar.output_size
        
    @property
    def hidden_size(self):
        return int(self._config['model']['hidden_size'])
    
    @property
    def batch_size(self):
        return int(self._config['training']['batch_size'])
    
    @property
    def beam_size(self):
        if self._config['output'].getboolean('use_beam_decode'):
            return int(self._config['output']['beam_width'])
        else:
            return -1
        
    @property
    def n_epochs(self):
        return int(self._config['training']['n_epochs'])
    
    @property
    def learning_rate(self):
        return float(self._config['training']['learning_rate'])

    @property
    def train_input_embeddings(self):
        return self._config['input'].getboolean('train_input_embeddings')
    
    @property
    def train_output_embeddings(self):
        return self._config['input'].getboolean('train_output_embeddings')
    
    @property
    def rnn_cell_type(self):
        return self._config['model']['rnn_cell_type']
    
    @property
    def rnn_layers(self):
        return int(self._config['model']['rnn_layers'])
    
    @property
    def apply_attention(self):
        return self._config['model'].getboolean('apply_attention')
    
    @property
    def grammar(self):
        return self._grammar
    
    @property
    def dictionary_size(self):
        return len(self._words)
    
    @property
    def dictionary(self):
        return self._words
    
    @property
    def input_embedding_matrix(self):
        return self._embeddings_matrix
        
    def save(self, filename):
        with open(filename, 'w') as fp:
            self._config.write(fp)
        
    @staticmethod
    def load(filenames):
        self = Config()
        print('Loading configuration from', filenames)
        self._config.read(filenames)
        
        words, reverse = load_dictionary(self._config['input']['input_words'])
        self._words = words
        self._reverse = reverse
        print("%d words in dictionary" % (self.dictionary_size,))
    
        self._embeddings_matrix = load_embeddings(self._config['input']['input_embeddings'], words, embed_size=self.embed_size)

        self._grammar = grammar.create_grammar(self._config['output']['grammar'], self._config['output']['grammar_input_file'])
        print("%d output tokens" % (self.output_size,))
        
        return self
