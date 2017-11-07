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
Created on Jul 20, 2017

@author: gcampagn
'''

import configparser
import grammar

from util.loader import load_dictionary, load_embeddings
from collections import OrderedDict

class Config(object):
    def __init__(self):
        self._config = configparser.ConfigParser()
        
        self._config['model'] = OrderedDict(
            model_type='seq2seq',
            encoder_type='birnn',
            encoder_hidden_size=35,
            decoder_hidden_size=70,
            function_hidden_size=100,
            function_nonlinearity='tanh',
            first_token_hidden_size=25,
            rnn_cell_type='lstm',
            rnn_layers=1,
            apply_attention='true',
        )
        self._config['training'] = OrderedDict(
            batch_size=256,
            n_epochs=25,
            learning_rate=0.01,
            dropout=0.5,
            gradient_clip=0.5,
            l2_regularization=0.0,
            optimizer='RMSProp'
        )
        self._config['input'] = OrderedDict(
            input_words='./input_words.txt',
            input_embeddings='./embeddings-300.txt',
            input_embed_size=300,
            max_length=60,
            train_input_embeddings='false',
            use_typed_embeddings='true'
        )
        self._config['output'] = OrderedDict(
            grammar='tt',
            grammar_input_file='./thingpedia.json',
            train_output_embeddings='true',
            output_embed_size=15,
            use_grammar_constraints='true',
            beam_width=10,
            training_beam_width=10,
            use_dot_product_output='false',
            connect_output_decoder='true'
        )
        
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
        if self.train_output_embeddings:
            return int(self._config['output']['output_embed_size'])
        else:
            return self._output_embeddings_matrix.shape[1]
        
    @property
    def output_size(self):
        return self._grammar.output_size
        
    @property
    def encoder_hidden_size(self):
        model_conf = self._config['model']
        if 'hidden_size' in model_conf:
            return int(model_conf['hidden_size'])
        else:
            return int(model_conf['encoder_hidden_size'])
        
    @property
    def decoder_hidden_size(self):
        model_conf = self._config['model']
        if 'hidden_size' in model_conf:
            return int(model_conf['hidden_size'])
        else:
            return int(model_conf['decoder_hidden_size'])
    
    @property
    def function_hidden_size(self):
        return int(self._config['model']['function_hidden_size'])
    
    @property
    def function_nonlinearity(self):
        return self._config['model']['function_nonlinearity']

    @property
    def first_token_hidden_size(self):
        return int(self._config['model']['first_token_hidden_size'])
    
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
    def training_beam_size(self):
        if self._config['model']['model_type'] == 'beamsearch':
            return int(self._config['output']['training_beam_width'])
        else:
            return 1
        
    @property
    def n_epochs(self):
        return int(self._config['training']['n_epochs'])
    
    @property
    def learning_rate(self):
        return float(self._config['training']['learning_rate'])

    @property
    def gradient_clip(self):
        return float(self._config['training']['gradient_clip'])
    
    @property
    def l2_regularization(self):
        return float(self._config['training']['l2_regularization'])

    @property
    def optimizer(self):
        return self._config['training']['optimizer']
    
    @property
    def train_input_embeddings(self):
        return self._config['input'].getboolean('train_input_embeddings')
    
    @property
    def typed_input_embeddings(self):
        return self._config['input'].getboolean('use_typed_embeddings')
    
    @property
    def use_dot_product_output(self):
        return self._config['output'].getboolean('use_dot_product_output')
    
    @property
    def connect_output_decoder(self):
        return self._config['output'].getboolean('connect_output_decoder')
    
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
        
        words, reverse = load_dictionary(self._config['input']['input_words'],
                                         use_types=self.typed_input_embeddings,
                                         grammar=self._grammar)
        self._words = words
        self._reverse = reverse
        print("%d words in dictionary" % (self.dictionary_size,))
        print("%d output tokens" % (self.output_size,))

        if self._config['input']['input_embeddings'] == 'xavier':
            assert self.train_input_embeddings
            assert not self.typed_input_embeddings
            self._embeddings_matrix = None
        else:
            self._embeddings_matrix, self._embed_size = load_embeddings(self._config['input']['input_embeddings'], words,
                                                                        use_types=self.typed_input_embeddings,
                                                                        grammar=self._grammar,
                                                                        embed_size=self.embed_size)
        print("Input embed size", self._embed_size)
        
        if self._config['output'].get('output_embeddings', None):
            self._output_embeddings_matrix, _ = load_embeddings(self._config['output']['output_embeddings'],
                                                                self._grammar.dictionary,
                                                                use_types=False, grammar=None,
                                                                embed_size=int(self._config['output']['output_embed_size']))
        else:
            self._output_embeddings_matrix = self._grammar.get_embeddings(words, self._embeddings_matrix)
        print("Output embed size", self._output_embeddings_matrix.shape[1])
        
        return self
