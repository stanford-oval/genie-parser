# Copyright 2018 The Board of Trustees of the Leland Stanford Junior University
#                Google LLC
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
Created on Jul 24, 2018

@author: gcampagn
'''

import os
import sys
import json
import os
import urllib.request
import ssl
import zipfile
import re
import tempfile
import shutil
import configparser
from collections import Counter

from tensor2tensor.utils import registry
import numpy as np
import tensorflow as tf

from .semantic_parsing import SemanticParsingProblem
from ..grammar.thingtalk import ThingTalkGrammar

from ..util.loader import clean, tokenize

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("thingpedia_snapshot", -1, "Thingpedia snapshot to use")
tf.flags.DEFINE_string("thingpedia_subset", "", "Subset Thingpedia to the given devices (leave empty for the full Thingpedia)")

_ssl_context = None
def get_ssl_context():
    global _ssl_context
    if _ssl_context is None:
        _ssl_context = ssl.create_default_context()
    return _ssl_context

class GrammarDirection(object):
    BOTTOM_UP = "bottomup"
    TOP_DOWN = "topdown"
    LINEAR = "linear"


class AbstractThingTalkProblem(SemanticParsingProblem):
    def _get_thingpedia(self, workdir):
        thingpedia_url = os.getenv('THINGPEDIA_URL', 'https://thingpedia.stanford.edu/thingpedia')
        snapshot = FLAGS.thingpedia_snapshot
        subset = FLAGS.thingpedia_subset or None
        tf.logging.info("Downloading Thingpedia snapshot %d from %s", snapshot, thingpedia_url)
    
        output = dict()
        with urllib.request.urlopen(thingpedia_url + '/api/snapshot/' + str(snapshot) + '?meta=1',
                                    context=get_ssl_context()) as res:
            all_devices = json.load(res)['data']
            output['devices'] = []
            for device in all_devices:
                if device['kind_type'] in ('global', 'category', 'discovery'):
                    continue
                if subset is not None and device['kind'] not in subset:
                    continue
                output['devices'].append(device)
                if device.get('kind_canonical', None):
                    self._add_words_to_dictionary(device['kind_canonical'])
                else:
                    print('WARNING: missing canonical for device:%s' % (device['kind'],))
                for function_type in ('triggers', 'queries', 'actions'):
                    for function_name, function in device[function_type].items():
                        if not function['canonical']:
                            print('WARNING: missing canonical for @%s.%s' % (device['kind'], function_name))
                        else:
                            self._add_words_to_dictionary(function['canonical'].lower())
                        for argname, argcanonical in zip(function['args'], function['argcanonicals']):
                            if argcanonical:
                                self._add_words_to_dictionary(argcanonical.lower())
                            else:
                                self._add_words_to_dictionary(clean(argname))
                        for argtype in function['schema']:
                            if not argtype.startswith('Enum('):
                                continue
                            enum_entries = argtype[len('Enum('):-1].split(',')
                            for enum_value in enum_entries:
                                self._add_words_to_dictionary(clean(enum_value))
    
        with urllib.request.urlopen(thingpedia_url + '/api/entities?snapshot=' + str(snapshot), context=get_ssl_context()) as res:
            output['entities'] = json.load(res)['data']
            for entity in output['entities']:
                if entity['is_well_known'] == 1:
                    continue
                self._add_words_to_dictionary(tokenize(entity['name']))
        
        with tf.gfile.Open(os.path.join(workdir, 'thingpedia.json'), 'w') as fp:
            json.dump(output, fp, indent=2)

    def begin_data_generation(self, data_dir):
        self._get_thingpedia(data_dir)

    @property
    def use_typed_embeddings(self):
        return True

    @property
    def export_assets(self):
        assets = super().export_assets
        assets['thingpedia.json'] = os.path.join(self._data_dir, 'thingpedia.json')
        return assets


@registry.register_problem("semparse_thingtalk")
class ThingTalkProblem(AbstractThingTalkProblem):
    def __init__(self, was_reversed, was_copy):
        super().__init__(was_reversed=was_reversed,
                         was_copy=was_copy)

    def grammar_factory(self, out_dir, **kw):
        return ThingTalkGrammar(os.path.join(out_dir, 'thingpedia.json'),
                                flatten=True)            


@registry.register_problem("semparse_thingtalk_noquote")
class QuoteFreeThingTalkProblem(AbstractThingTalkProblem):
    def __init__(self, was_reversed, was_copy):
        super().__init__(flatten_grammar=False,
                         was_reversed=was_reversed,
                         was_copy=was_copy)
        
    def grammar_factory(self, out_dir, **kw):
        return ThingTalkGrammar(os.path.join(out_dir, 'thingpedia.json'),
                                flatten=False)
