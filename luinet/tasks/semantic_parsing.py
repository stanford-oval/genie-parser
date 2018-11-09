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
import sys
import json
import urllib.request
import ssl
import zipfile
import re
import tempfile
import shutil
import numpy as np
import configparser

import numpy as np
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import data_reader
from tensor2tensor.layers import common_layers

from ..layers.modalities import PretrainedEmbeddingModality, PointerModality
from ..grammar.abstract import AbstractGrammar
from ..tasks import base_problem


FLAGS = tf.flags.FLAGS

START_TOKEN = '<s>'
START_ID = text_encoder.EOS_ID + 1
UNK_TOKEN = '<unk>'
UNK_ID = text_encoder.EOS_ID + 2


class IdentityEncoder(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def encode(self, x):
        return x
    
    def decode(self, x):
        return ' '.join(filter(lambda y: y not in ['<s>', ' 0PAD'], (map(lambda x: x.decode('utf-8'), x))))


def _make_pointer_modality(name, vocab_size):
    return lambda model_hparams: \
        PointerModality(name, model_hparams, vocab_size=vocab_size)


ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME', 'PATH_NAME', 'CURRENCY']
MAX_ARG_VALUES = 5


HACK_REPLACEMENT = {
    # onedrive is the new name of skydrive
    'onedrive': 'skydrive',

    # imgflip is kind of the same as imgur (or 9gag)
    # until we have either in thingpedia, it's fine to reuse the word vector
    'imgflip': 'imgur'
}


class SemanticParsingProblem(text_problems.Text2TextProblem,
                             base_problem.LUINetProblem):
    """Tensor2Tensor problem for Grammar-Based semantic parsing."""
  
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
    def grammar(self):
        """The grammar associated with this SemanticParsingProblem.
        
        Returns the grammar, if initialized, or None.
        Use get_grammar() to initialize the grammar on demand instead.
        """
        return self._grammar

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

    def example_reading_spec(self):
        data_fields = {
            "type": tf.VarLenFeature(tf.int64),
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def hparams(self, hparams, model_hparams):
        # do not chain up! we have different features and modalities
        # than Text2TextProblem
        hp = hparams
        hp.stop_at_eos = True
        
        modality_name_prefix = self.name + "_"
        
        source_vocab_size = self._encoders["inputs"].vocab_size
        
        with tf.gfile.Open(os.path.join(model_hparams.data_dir,
                                        "input_embeddings.npy"), "rb") as fp:
            pretrained_input_embeddings = np.load(fp)
        pretrained_modality = lambda model_hparams: \
            PretrainedEmbeddingModality("inputs", pretrained_input_embeddings,
                                        model_hparams)
        hp.input_modality = {
            "inputs": pretrained_modality
        }
        
        data_dir = (model_hparams and hasattr(model_hparams, "data_dir") and
                    model_hparams.data_dir) or None
        self._data_dir = data_dir
        
        grammar = self.get_grammar(data_dir)
        if model_hparams.grammar_direction == "linear":
            tgt_vocab_size = len(grammar.tokens)
        else:
            tgt_vocab_size = grammar.output_size[grammar.primary_output]

        hp.target_modality = {}
        hp.add_hparam("primary_target_modality", "targets_" + grammar.primary_output)
        for key, size in grammar.output_size.items():
            if key == grammar.primary_output:
                if model_hparams.use_margin_loss:
                    hp.target_modality["targets_" + key] = ("symbol:max_margin", tgt_vocab_size)
                else:
                    hp.target_modality["targets_" + key] = ("symbol:softmax", tgt_vocab_size)
                # else:
                #     hp.target_modality["targets_" + key] = ("symbol:default", tgt_vocab_size)
            elif grammar.is_copy_type(key):
                hp.target_modality["targets_" + key] = ("symbol:copy", size)
            else:
                modality_name = modality_name_prefix + "pointer_" + key
                hp.target_modality["targets_" + key] = _make_pointer_modality(modality_name, size)

    @property
    def is_generate_per_split(self):
        return True
    
    @property
    def vocab_filename(self):
        return 'input_words.txt'
    
    @property
    def export_assets(self):
        return {'input_words.txt': os.path.join(self._data_dir, 'input_words.txt'),
                
                # technically, this would be exported as a Const operation in the saved model
                # but we can't use the SavedModel directly (due to pyfuncs) so we need
                # export this and load it later
                'input_embeddings.npy': os.path.join(self._data_dir, 'input_embeddings.npy')}
    
    @property
    def oov_token(self):
        return UNK_TOKEN
    
    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN
    
    def get_grammar(self, out_dir=None) -> AbstractGrammar:
        if self._grammar:
            return self._grammar
        
        self._grammar = self.grammar_factory(out_dir, flatten=self._flatten_grammar)
        return self._grammar
    
    def _parse_program(self, program, model_hparams, grammar):
        def parse_program_pyfunc(program_nparray):
            # we don't need to pass the input sentence, the program
            # was tokenized already
            vectors, length = grammar.vectorize_program(None, program_nparray,
                                                        direction=model_hparams.grammar_direction,
                                                        max_length=None)
            for key in vectors:
                vectors[key] = vectors[key][:length]
            return vectors
        
        output_dtypes = dict()
        output_shapes = dict()
        for key in grammar.output_size:
            output_shapes[key] = tf.TensorShape([None])
            output_dtypes[key] = tf.int32
        
        return tf.contrib.framework.py_func(parse_program_pyfunc, [program],
                                            output_shapes=output_shapes,
                                            output_types=output_dtypes,
                                            stateful=False)
    
    def preprocess_example(self, example, mode, model_hparams):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            global_step = tf.train.get_or_create_global_step()
        schedule = tf.exp(tf.cast(-global_step, dtype=tf.float32) / float(FLAGS.train_steps))
        _type = tf.reshape(example["type"], ())
        zeros = tf.zeros_like(_type, dtype=tf.float32)
        synth_weight = tf.where(tf.equal(_type, 0), 0.1 + 0.9 * schedule, zeros)
        para_weight = tf.where(tf.equal(_type, 1), 0.6 + (1 - schedule), zeros)
        aug_weight = tf.where(tf.equal(_type, 2), 0.3 + (1 - schedule), zeros)

        output_example = {
            "inputs": example["inputs"],
            "weight": tf.expand_dims(synth_weight + para_weight + aug_weight, axis=0)
        }
        
        if "targets" in example:
            output_example["targets"] = example["targets"]
        
            grammar = self.get_grammar(model_hparams.data_dir)
        
            parsed = self._parse_program(example["targets"], model_hparams, grammar)
            for key, value in parsed.items():
                output_example["targets_" + key] = value
        
        return output_example
    
    def _encode_string_input(self, features, tf_dictionary):
        string_input = features["inputs/string"]
        int64_input = tf_dictionary.lookup(string_input)
        
        return {
            "inputs": tf.concat(([START_ID], int64_input, [text_encoder.EOS_ID]),
                                axis=0)
        }
    
    def direct_serving_input_fn(self, hparams):
        """Input fn for serving export, starting from appropriate placeholders."""
        mode = tf.estimator.ModeKeys.PREDICT
        
        placeholders = {
            "inputs/string": tf.placeholder(shape=tf.TensorShape([None, None]),
                                            dtype=tf.string)
        }
        batch_size = tf.shape(placeholders["inputs/string"], out_type=tf.int64)[0]
        
        data_dir = hparams.data_dir        
        vocab_file = os.path.join(data_dir, self.vocab_filename)
        tf_dictionary = tf.contrib.lookup.index_table_from_file(vocab_file,
                                                                default_value=UNK_ID)
        
        dataset = tf.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.map(lambda ex: self._encode_string_input(ex, tf_dictionary))
        dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
        dataset = dataset.map(self.maybe_reverse_and_copy)
        dataset = dataset.map(data_reader.cast_ints_to_int32)
        dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
        dataset = dataset.map(problem.standardize_shapes)
        features = tf.contrib.data.get_single_element(dataset)
        if self.has_inputs:
            features.pop("targets", None)
    
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=placeholders)
    
    def eval_metrics(self):
        grammar = self.get_grammar()
        return grammar.eval_metrics()
    
    def decode_targets(self, targets, features, model_hparams=None):
        grammar = self.get_grammar(model_hparams.data_dir)
        targets = tf.squeeze(targets, axis=[2, 3])
        input_sentence = tf.squeeze(features["inputs"], axis=[2, 3])
        
        def decode_pyfunc(program_batch, input_sentence_batch):
            batch_size = len(program_batch)
            output_len = None
            outputs = []
            for i in range(batch_size):
                vector = grammar.decode_program(input_sentence_batch[i], program_batch[i])
                outputs.append(vector)
                if output_len is None or len(vector) > output_len:
                    output_len = len(vector)
            output_matrix = np.empty((batch_size, output_len), dtype=object)
            for i in range(batch_size):
                item_len = len(outputs[i])
                output_matrix[i, :item_len] = outputs[i]
                output_matrix[i, item_len:] = np.zeros((output_len - item_len,),
                                                       dtype=np.str)
            return output_matrix
        
        return tf.contrib.framework.py_func(decode_pyfunc,
                                            [targets, input_sentence],
                                            output_shapes=tf.TensorShape([None, None]),
                                            output_types=tf.string,
                                            stateful=False)
    
    def compute_predictions(self, outputs, features, model_hparams=None,
                            decode=False):
        grammar = self.get_grammar(model_hparams.data_dir)

        sample_ids = dict()
        for key in outputs:
            assert key.startswith("targets_")
            grammar_key = key[len("targets_"):]
            sample_ids[grammar_key] = outputs[key]
        input_sentence = tf.squeeze(features["inputs"], axis=[2, 3])
        
        beam_size = None
        batch_size = None
        time = None
        if len(sample_ids[grammar.primary_output].shape) > 2:
            # beam search is active, merge the beam and batch dimensions
            # and tile the input sentence appropriately
            batch_size, beam_size, time = common_layers.shape_list(sample_ids[grammar.primary_output])
            input_sentence = tf.tile(tf.expand_dims(input_sentence, axis=1), [1, beam_size, 1])
            input_sentence = tf.reshape(input_sentence, [batch_size * beam_size, tf.shape(input_sentence)[2]])
            for key in grammar.output_size:
                sample_ids[key] = tf.reshape(sample_ids[key], [batch_size * beam_size, time])
        
        def compute_prediction_pyfunc(sample_ids, input_sentence_batch):
            batch_size = len(sample_ids[grammar.primary_output])
            
            # each value in sample_ids should be [batch, time] or
            # [batch * beam, time]
            assert len(sample_ids[grammar.primary_output].shape) == 2
            for key in grammar.output_size:
                assert sample_ids[key].shape == sample_ids["actions"].shape, \
                    (key, sample_ids[key].shape)
            assert len(input_sentence_batch) == batch_size

            outputs = []
            output_len = None
            for i in range(batch_size):
                decoded_vectors = dict()
                for key in grammar.output_size:
                    decoded_vectors[key] = sample_ids[key][i]
                    
                #grammar.print_prediction([], decoded_vectors)
                vector = grammar.reconstruct_to_vector(decoded_vectors,
                                                       direction=model_hparams.grammar_direction,
                                                       ignore_errors=True)
                if decode:
                    vector = grammar.decode_program(input_sentence_batch[i], vector)
                    #print(input_sentence_batch[i], vector)
                outputs.append(vector)
                if output_len is None or len(vector) > output_len:
                    output_len = len(vector)
                    
            output_matrix = np.empty((batch_size, output_len),
                                     dtype=object if decode else np.int32)
            for i in range(batch_size):
                item_len = len(outputs[i])
                output_matrix[i, :item_len] = outputs[i]
                output_matrix[i, item_len:] = np.zeros((output_len - item_len,),
                                                       dtype=np.str if decode else np.int32)
                
            return output_matrix
        
        predictions = tf.contrib.framework.py_func(compute_prediction_pyfunc,
                                                   [sample_ids, input_sentence],
                                                   output_shapes=tf.TensorShape([None, None]),
                                                   output_types=(tf.string if decode else tf.int32),
                                                   stateful=False)
        if beam_size is not None:
            # beam search is active, resplit the batch and beam dimensions
            predictions = tf.reshape(predictions, [batch_size, beam_size, tf.shape(predictions)[1]])
        return predictions
    
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
                tf.logging.warn('Invalid word "%s" in phrase "%s"' % (word, canonical,))
                continue
            if word[0].isupper():
                continue
            self._building_dictionary.add(word)

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        # override to call begin_data_generation and build the dictionary
        
        self._building_dictionary = set()
        
        # download any subclass specific data
        self.begin_data_generation(data_dir)
        
        # load the dataset once to build the dictionary
        src_data_dir = FLAGS.src_data_dir or data_dir
        self._load_words_from_files(src_data_dir)
        
        # create and save the input dictionary
        self._create_input_vocab(data_dir)
        
        super().generate_data(data_dir, tmp_dir, task_id=task_id)

    def _load_words_from_files(self, src_data_dir):
        filepattern = os.path.join(src_data_dir, '*.tsv')
        for filename in tf.contrib.slim.parallel_reader.get_data_files(filepattern):
            with tf.gfile.Open(filename, 'r') as fp:
                for line in fp:
                    sentence = line.strip().split('\t')[1]
                    self._add_words_to_dictionary(sentence)
    
    def _download_glove(self, glove, embed_size):
        if tf.gfile.Exists(glove):
            return
    
        tf.logging.info('Downloading pretrained GloVe vectors into %s', glove)
        with tempfile.TemporaryFile() as tmp:
            with urllib.request.urlopen('https://nlp.stanford.edu/data/glove.42B.' + str(embed_size) + 'd.zip') as res:
                shutil.copyfileobj(res, tmp)
            with zipfile.ZipFile(tmp, 'r') as glove_zip:
                glove_zip.extract('glove.42B.' + str(embed_size) + 'd.txt', path=os.path.dirname(glove))
        tf.logging.info('Done downloading GloVe')
    
    @property
    def use_typed_embeddings(self):
        return False
    
    def _convert_glove_to_numpy(self, glove, data_dir,
                                dictionary, grammar, embed_size):
        original_embed_size = embed_size
        if self.use_typed_embeddings:
            num_entities = len(ENTITIES) + len(grammar.entities)
            embed_size += num_entities + MAX_ARG_VALUES + 2
        else:
            embed_size += 2
        
        # there are 4 reserved tokens: pad, eos, start and unk
        embedding_matrix = np.zeros((4 + len(dictionary), embed_size),
                                    dtype=np.float32)
        # careful here! various places in t2t assume that padding
        # will have a all-zero embedding
        # and nothing else will have all-zero
        # (eg. common_attention.embedding_to_padding, which
        # is called by transformer_prepare_encoder)
        
        # we also use all-one for <unk>
        embedding_matrix[text_encoder.EOS_ID, embed_size-1] = 1.
        embedding_matrix[START_ID, embed_size-2] = 1.
        embedding_matrix[3, :original_embed_size] = np.ones((original_embed_size,))

        trimmed_glove = dict()
        hack_values = HACK_REPLACEMENT.values()
        with tf.gfile.Open(glove, "r") as fp:
            for line in fp:
                line = line.strip()
                vector = line.split(' ')
                word, vector = vector[0], vector[1:]
                if not word in dictionary and word not in hack_values:
                    continue
                vector = np.array(list(map(float, vector)))
                trimmed_glove[word] = vector
        
        BLANK = re.compile('^_+$')
        for word, word_id in dictionary.items():
            assert isinstance(word, str), (word, word_id)
            if self.use_typed_embeddings and word[0].isupper():
                continue
            if word in trimmed_glove:
                embedding_matrix[word_id, :original_embed_size] = trimmed_glove[word]
                continue
            
            if not word or re.match('\s+', word):
                raise ValueError('Invalid word "%s"' % (word,))
            vector = None
            if BLANK.match(word):
                # normalize blanks
                vector = trimmed_glove['____']
            elif word.endswith('s') and word[:-1] in trimmed_glove:
                vector = trimmed_glove[word[:-1]]
            elif (word.endswith('ing') or word.endswith('api')) and word[:-3] in trimmed_glove:
                vector = trimmed_glove[word[:-3]]
            elif word in HACK_REPLACEMENT:
                vector = trimmed_glove[HACK_REPLACEMENT[word]]
            elif '-' in word:
                vector = np.zeros(shape=(original_embed_size,), dtype=np.float64)
                for w in word.split('-'):
                    if w in trimmed_glove:
                        vector += trimmed_glove[w]
                    else:
                        vector = None
                        break
            if vector is not None:
                embedding_matrix[word_id, :original_embed_size] = vector
            else:
                tf.logging.warn("missing word from GloVe: %s", word)
                embedding_matrix[word_id, :original_embed_size] = np.random.normal(0, 0.9, (original_embed_size,))
        del trimmed_glove
        
        if self.use_typed_embeddings:
            for i, entity in enumerate(ENTITIES):
                for j in range(MAX_ARG_VALUES):
                    token_id = dictionary[entity + '_' + str(j)]
                    embedding_matrix[token_id, original_embed_size + i] = 1.
                    embedding_matrix[token_id, original_embed_size + num_entities + j] = 1.
            for i, (entity, has_ner) in enumerate(grammar.entities):
                if not has_ner:
                    continue
                for j in range(MAX_ARG_VALUES):
                    token_id = dictionary['GENERIC_ENTITY_' + entity + '_' + str(j)]
                    embedding_matrix[token_id, original_embed_size + len(ENTITIES) + i] = 1.
                    embedding_matrix[token_id, original_embed_size + num_entities + j] = 1.
    
        with tf.gfile.Open(os.path.join(data_dir, "input_embeddings.npy"), "wb") as fp:
            np.save(fp, embedding_matrix)
    
    def _create_input_vocab(self, data_dir):
        dictionary = dict()
        grammar = self.get_grammar(data_dir)
        
        with tf.gfile.Open(os.path.join(data_dir, self.vocab_filename), 'w') as fp:
            print(text_encoder.PAD, file=fp)
            print(text_encoder.EOS, file=fp)
            print(START_TOKEN, file=fp)
            print(self.oov_token, file=fp)
            
            token_id = 4
            if self.use_typed_embeddings:
                for i, entity in enumerate(ENTITIES):
                    for j in range(MAX_ARG_VALUES):
                        token = entity + '_' + str(j)
                        dictionary[token] = token_id
                        print(token, file=fp)
                        token_id += 1
                for i, (entity, has_ner) in enumerate(grammar.entities):
                    if not has_ner:
                        continue
                    for j in range(MAX_ARG_VALUES):
                        token = 'GENERIC_ENTITY_' + entity + '_' + str(j)
                        dictionary[token] = token_id
                        print(token, file=fp)
                        token_id += 1
            
            for word in sorted(self._building_dictionary):
                if word in dictionary:
                    continue
                dictionary[word] = token_id
                print(word, file=fp)
                token_id += 1
                
        glove = os.getenv('GLOVE', os.path.join(data_dir, 'glove.42B.300d.txt'))    
        self._download_glove(glove, embed_size=300)
        self._convert_glove_to_numpy(glove, data_dir,
                                     dictionary, grammar, embed_size=300)
        
        self._building_dictionary = None
    
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        grammar = self.get_grammar(data_dir)
        input_vocabulary = self.get_feature_encoders(data_dir)["inputs"]
        
        src_data_dir = FLAGS.src_data_dir or data_dir
        with tf.gfile.Open(os.path.join(src_data_dir, dataset_split + ".tsv"), "r") as fp:
            for line in fp:
                # forget about constituency parses, they were a bad idea
                _id, sentence, program = line.strip().split('\t')[:3]

                _type = 0
                if 'S' in _id:
                    _type = 0
                elif 'P' in _id:
                    _type = 1
                else:
                    _type = 2
                
                sentence = sentence.split(' ')
                sentence.insert(0, START_TOKEN)

                vectorized = grammar.tokenize_to_vector(sentence, program)
                grammar.verify_program(vectorized)

                encoded_input: list = input_vocabulary.encode(' '.join(sentence))
                assert text_encoder.PAD_ID not in encoded_input
                encoded_input.append(text_encoder.EOS_ID)
                
                yield {
                    "type": [_type],

                    "inputs": encoded_input,
                    
                    # t2t explicitly wants a list of python integers, just to convert
                    # it back to a packed representation immediately after
                    # because
                    "targets": list(map(int, vectorized))
                }
    
