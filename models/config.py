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
            'encoder_type': 'rnn',
            'hidden_size': 175,
            'rnn_cell_type': 'lstm',
            'rnn_layers': 1,
            'apply_attention': 'true'
        }
        self._config['training'] = {
            'batch_size': 256,
            'n_epochs': 20,
            'learning_rate': 0.01,
            'dropout': 0.5
        }
        self._config['input'] = {
            'input_words': './input_words.txt',
            'input_embeddings': './embeddings.txt',
            'input_embed_size': 300,
            'max_length': 60,
            'train_input_embeddings': 'false',
            'use_typed_embeddings': 'false'
        }
        self._config['output'] = {
            'grammar': 'tt',
            'grammar_input_file': './thingpedia.txt',
            'train_output_embeddings': 'false',
            'use_grammar_constraints': 'false',
            'use_typed_embeddings': 'false',
            'beam_width': 10
        }
        
        self._grammar = None
        self._words = None
        self._reverse = None
        self._embeddings_matrix = None
        self._output_embeddings_matrix = None
        self._embed_size = int(self._config['input']['input_embed_size'])
            
    @property
    def model_type(self):
        return self._config['model']['model_type']
    
    @property
    def encoder_type(self):
        return self._config['model']['encoder_type']
    
    @property
    def max_length(self):
        return int(self._config['input']['max_length'])
    
    @property
    def dropout(self):
        return float(self._config['training']['dropout'])
    
    @property
    def embed_size(self):
        return self._embed_size
    
    @property
    def output_embed_size(self):
        return self._output_embeddings_matrix.shape[1]
        
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
        if self._config['model']['model_type'] == 'beamsearch':
            return int(self._config['output']['beam_width'])
        else:
            return 1
        
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
    def typed_input_embeddings(self):
        return self._config['input'].getboolean('use_typed_embeddings')
    
    @property
    def typed_output_embeddings(self):
        return self._config['output'].getboolean('use_typed_embeddings')
    
    @property
    def train_output_embeddings(self):
        return self._config['output'].getboolean('train_output_embeddings')
    
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
    def use_grammar_constraints(self):
        return self._config['output'].getboolean('use_grammar_constraints')
    
    @property
    def dictionary_size(self):
        return len(self._words)
    
    @property
    def dictionary(self):
        return self._words

    @property
    def reverse_dictionary(self):
        return self._reverse

    @property
    def input_embedding_matrix(self):
        return self._embeddings_matrix
    
    @property
    def output_embedding_matrix(self):
        return self._output_embeddings_matrix
        
    def save(self, filename):
        with open(filename, 'w') as fp:
            self._config.write(fp)
        
    @staticmethod
    def load(filenames):
        self = Config()
        print('Loading configuration from', filenames)
        self._config.read(filenames)
        self._embed_size = int(self._config['input']['input_embed_size'])
        
        self._grammar = grammar.create_grammar(self._config['output']['grammar'], self._config['output']['grammar_input_file'])
        print("%d output tokens" % (self.output_size,))
        self._output_embeddings_matrix = self._grammar.get_embeddings(use_types=self.typed_output_embeddings)
        print("Output embed size", self._output_embeddings_matrix.shape[1])
        
        words, reverse = load_dictionary(self._config['input']['input_words'],
                                         use_types=self.typed_input_embeddings,
                                         grammar=self._grammar)
        self._words = words
        self._reverse = reverse
        print("%d words in dictionary" % (self.dictionary_size,))
        
        self._embeddings_matrix, self._embed_size = load_embeddings(self._config['input']['input_embeddings'], words,
                                                                    use_types=self.typed_input_embeddings,
                                                                    grammar=self._grammar,
                                                                    embed_size=self.embed_size)
        print("Input embed size", self._embed_size)
        
        return self
