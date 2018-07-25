# Copyright 2018 The Board of Trustees of the Leland Stanford Junior University
#                Google LLC
#
# Author: Mehrad Moradshahi <mehrad@stanford.edu>
#         Giovanni Campagna <gcampagn@cs.stanford.edu>
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

@author: mehrad, gcampagn
'''

import os

import tensorflow as tf

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder

from ..grammar.abstract import AbstractGrammar

FLAGS = tf.flags.FLAGS

class IdentityEncoder(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def encode(self, x):
        return x
    
    def decode(self, x):
        return x


class SemanticParsingProblem(text_problems.Text2TextProblem):
    """Tensor2Tensor problem for Grammar-Based semantic parsing."""
    
    grammar_direction = None
  
    def __init__(self,
                 flatten_grammar=True,
                 was_reversed=False,
                 was_copy=False):
        super().__init__(was_reversed=was_reversed,
                         was_copy=was_copy)
        self._grammar = None
        self._flatten_grammar = flatten_grammar
        self._building_dictionary = None

    def grammar_factory(self, out_dir, **kw):
        raise NotImplementedError()

    @property
    def has_inputs(self):
        return True
    
    def feature_encoders(self, data_dir):
        grammar = self.get_grammar(data_dir)
        
        encoders = {
            "inputs": self.get_or_create_vocab(data_dir, None)
        }
        grammar.set_input_dictionary(encoders["inputs"])
        encoders["targets"] = IdentityEncoder(len(grammar.tokens))
        # note: this mostly does not matter, because we override
        # preprocess_example later
        return encoders
  
    def hparams(self, hparams, model_hparams):
        # do not chain up! we have different features and modalities
        # than Text2TextProblem
        hp = hparams
        
        hp.stop_at_eos = True
        hp.add_hparam("grammar_direction", "bottomup")
        
        source_vocab_size = self._encoders["inputs"].vocab_size
        hp.input_modality = {
            "inputs": ("symbol:default", source_vocab_size)
        }
        
        data_dir = (model_hparams and hasattr(model_hparams, "data_dir") and
                    model_hparams.data_dir) or None
        
        grammar = self.get_grammar(data_dir)
        if self._flatten_grammar or hp.grammar_direction == 'linear':
            if hp.grammar_direction == "linear":
                tgt_vocab_size = len(grammar.tokens)
            else:
                tgt_vocab_size = grammar.output_size[grammar.primary_output]
            hp.target_modality = ("symbol:default", tgt_vocab_size)
        else:
            hp.target_modality = {}
            for key, size in grammar.output_size.items():
                if key == grammar.primary_output:
                    hp.target_modality[key] = ("symbol:default", size)
                elif grammar.is_copy_type(key):
                    hp.target_modality[key] = ("symbol:copy", size)
                else:
                    hp.target_modality[key] = ("symbol:identity", size)
  
    @property
    def is_generate_per_split(self):
        return True
    
    @property
    def vocab_filename(self):
        return 'input_words.txt'
    
    @property
    def oov_token(self):
        return '<unk>'
    
    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN
    
    def get_grammar(self, out_dir) -> AbstractGrammar:
        if self._grammar:
            return self._grammar
        
        self._grammar = self.grammar_factory(out_dir, flatten=self._flatten_grammar)
        return self._grammar
    
    def _parse_program(self, program, model_hparams, grammar):
        our_hparams = self.get_hparams(model_hparams)

        def parse_program_pyfunc(program_nparray):
            # we don't need to pass the input sentence, the program
            # was tokenized already
            vectors, length = grammar.vectorize_program([], program_nparray,
                                                        direction=our_hparams.grammar_direction,
                                                        # pass a large max length, we'll
                                                        # truncate if necessary
                                                        max_length=100)
            for key in vectors:
                vectors[key] = vectors[key][:length]
            return vectors
        
        output_dtypes = dict()
        output_shapes = dict()
        if our_hparams.grammar_direction == "linear":
            output_dtypes["targets"] = tf.int32
            output_shapes["targets"] = tf.TensorShape([None])
        else:
            for key in grammar.output_size:
                output_shapes[key] = tf.TensorShape([None])
                output_dtypes[key] = tf.int32
        
        return tf.contrib.framework.py_func(parse_program_pyfunc, [program],
                                            output_shapes=output_shapes,
                                            output_types=output_dtypes,
                                            stateful=False)
    
    def preprocess_example(self, example, mode, model_hparams):
        output_example = {
            "inputs": example["inputs"]
        }
        
        grammar = self.get_grammar(model_hparams.data_dir)
        parsed = self._parse_program(example["targets"], model_hparams, grammar)
        for key, value in parsed.items():
            processed_key = "targets" if key == grammar.primary_output else \
                    "targets_" + key
            output_example[processed_key] = value
        return output_example
    
    def begin_data_generation(self, data_dir):
        # nothing to do
        pass
    
    def _add_words_to_dictionary(self, canonical):
        assert self._building_dictionary is not None, \
            "_add_words_to_dictionary can only be called during data generation"
        if isinstance(canonical, str):
            sequence = canonical.split(' ')
        else:
            sequence = canonical
        for word in sequence:
            if not word or word != '$' and word.startswith('$'):
                print('Invalid word "%s" in phrase "%s"' % (word, canonical,))
                continue
            if word[0].isupper():
                continue
            self._building_dictionary.add(word)
    
    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        # override to call begin_data_generation and build the dictionary
        
        self._building_dictionary = set()
        
        # download any subclass specific data
        self.begin_data_generation(data_dir)
        # create and save the input dictionary
        self._create_input_vocab(data_dir)
        
        super().generate_data(data_dir, tmp_dir, task_id=task_id)
    
    def _load_words_from_files(self, data_dir):
        filepattern = os.path.join(data_dir, '*.tsv')
        for filename in tf.contrib.slim.parallel_reader.get_data_files(filepattern):
            with tf.gfile.Open(filename, 'r') as fp:
                for line in fp:
                    sentence = line.strip().split('\t')[1]
                    self._add_words_to_dictionary(sentence)
    
    def _create_input_vocab(self, data_dir):
        with tf.gfile.Open(os.path.join(data_dir, self.vocab_filename), 'w') as fp:
            print(text_encoder.PAD, file=fp)
            print(text_encoder.EOS, file=fp)
            print(self.oov_token, file=fp)
            for i, word in enumerate(sorted(self._building_dictionary)):
                print(word, file=fp)
        
        self._building_dictionary = None
    
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        grammar = self.get_grammar(data_dir)
        input_vocabulary = self.get_feature_encoders(data_dir)["inputs"]
        
        src_data_dir = FLAGS.src_data_dir or data_dir
        with tf.gfile.Open(os.path.join(src_data_dir, dataset_split + ".tsv"), "r") as fp:
            for line in fp:
                # forget about constituency parses, they were a bad idea
                _id, sentence, program = line.strip().split('\t')[:3]
                
                vectorized, length = grammar.vectorize_program(sentence.split(' '),
                                                               program,
                                                               direction='tokenizeonly')
                grammar.verify_program(vectorized)
                
                sample = {
                    "inputs": input_vocabulary.encode(sentence),
                    
                    # t2t explicitly wants a list of python integers, just to convert
                    # it back to a packed representation immediately after
                    # because
                    "targets": list(map(int, vectorized[:length]))
                }
                sample["inputs"].append(text_encoder.EOS_ID)
                yield sample
    