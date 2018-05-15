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
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from .seq2seq_aligner import Seq2SeqAligner
from .beam_aligner import BeamAligner
from .rpag_aligner import RPAGAligner
from .beamdecode_aligner import BeamDecodeAligner

import importlib 

from .config import Config
    
def create_model(config):
    pkg = None
    class_name = None

    # for compat with existing configuration files
    if config.model_type == 'seq2seq':
        pkg = 'seq2seq_aligner'
        class_name = 'Seq2SeqAligner'
    elif config.model_type == 'beamdecode':
        pkg = 'beamdecode_aligner'
        class_name = 'BeamDecodeAligner'
    elif config.model_type == 'beamsearch':
        pkg = 'beamsearch'
        class_name = 'BeamAligner'
    elif config.model_type == 'rpag':
        pkg = 'rpag_aligner'
        class_name = 'RPAGAligner'
    elif config.model_type == 'extensible':
        pkg = 'extensible_aligner'
        class_name = 'ExtensibleGrammarAligner'
    else:
        raise ValueError("Invalid model type %s" % (config.model_type,))
    
    module = importlib.import_module('models.' + pkg)
    return getattr(module, class_name)(config)
