'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from .seq2seq_aligner import Seq2SeqAligner
from .config import Config
    
def create_model(config):
    if config.model_type == 'seq2seq':
        model = Seq2SeqAligner(config)
    else:
        raise ValueError("Invalid model type %s" % (config.model_type,))
    
    return model