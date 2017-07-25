'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from .seq2seq_aligner import Seq2SeqAligner
from .beam_aligner import BeamAligner

from .config import Config
    
def create_model(config):
    if config.model_type == 'seq2seq':
        model = Seq2SeqAligner(config)
    elif config.model_type == 'beamsearch':
        model = BeamAligner(config)
    else:
        raise ValueError("Invalid model type %s" % (config.model_type,))
    
    return model