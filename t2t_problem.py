# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports


from tensor2tensor.layers import common_hparams
from tensor2tensor.models.transformer import transformer_base, transformer_tiny
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities

import tensorflow as tf


@registry.register_hparams
def almond_params():
    # Start with the base set
    hp = transformer_tiny()
    # Modify existing hparams
    hp.target_modality = 'symbol:almond'
    #hp.num_hidden_layers = 2
    # Add new hparams
    #hp.add_hparam("filter_size", 2048)
    return hp

@registry.register_symbol_modality("almond")
class AlmondSymbolModality(modalities.SymbolModality):

    def loss(self, top_out, targets):
        #  we assume targets are , [batch, length, 1, 1] here.
        #  we assume logits are , [batch, length, 1, 1, num_classes] here.
        print("we're inside loss function")
        hp = self._model_hparams
        logits = top_out


        targets = tf.squeeze(targets, axis=[2, 3])
        logits = tf.squeeze(logits, axis=[2, 3])

        # batch_size = hp.batch_size
        # max_length = hp.max_length
        batch_size = tf.shape(logits)[0]
        max_length = tf.shape(logits)[1]
        num_classes = tf.shape(logits)[2]

        print("batch_size:", batch_size)
        print("max_length", max_length)
        print("num_classes", num_classes)


        targets_shape = targets.get_shape().as_list()
        logits_shape = logits.get_shape().as_list()
        print("logits_shape: ", logits_shape, "\ntargets_shape: ", targets_shape)



        with tf.name_scope("max_margin_loss", values=[logits, targets]):

            targets_mask = tf.subtract(1.0, tf.to_float(tf.equal(targets, 0)))
            print('target_mask isssss : ', targets_mask)

            flat_mask = tf.reshape(targets_mask, (batch_size * max_length,))

            flat_preds = tf.reshape(logits, (batch_size * max_length, num_classes))
            flat_gold = tf.reshape(targets, (batch_size * max_length,))

            flat_indices = tf.range(batch_size * max_length, dtype=tf.int32)
            flat_gold_indices = tf.stack((flat_indices, flat_gold), axis=1)

            one_hot_gold = tf.one_hot(targets, depth=num_classes, dtype=tf.float32)
            marginal_scores = logits - one_hot_gold + 1

            marginal_scores = tf.reshape(marginal_scores, (batch_size * max_length, num_classes))
            max_margin = tf.reduce_max(marginal_scores, axis=1)

            gold_score = tf.gather_nd(flat_preds, flat_gold_indices)
            margin = max_margin - gold_score

            margin = margin * flat_mask

            margin = tf.reshape(margin, (batch_size, max_length))
            output_length = tf.reduce_sum(targets_mask, axis=1)

            weights = self.targets_weights_fn(targets)

            return tf.reduce_mean(tf.div(tf.reduce_sum(margin, axis=1), output_length), axis=0), \
                    tf.reduce_sum(weights)


@registry.register_problem
class ParseAlmond(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def approx_vocab_size(self):
    return 4604

  @property
  def vocab_filename(self):
    print("reading vocab_file: all_words.txt")
    return "all_words.txt"
    # return vocab.bpe.%d" % self.approx_vocab_size

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_filename) and force_get:
      raise ValueError("Vocab %s not found" % vocab_filename)

    return text_encoder.TokenTextEncoder(vocab_filename, replace_oov='UNK')


  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the WMT en->de task, training set."""
    train = dataset_split == problem.DatasetSplit.TRAIN
    print(train)
    if train:
        train_path = "../dataset/t2t_data/t2t_train"
    else:
        train_path = "../dataset/t2t_data/t2t_dev"

    # Vocab
    token_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(token_path):
      token_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
      tf.gfile.Copy(token_tmp_path, token_path)
      with tf.gfile.GFile(token_path, mode="r") as f:
        vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
      with tf.gfile.GFile(token_path, mode="w") as f:
        f.write(vocab_data)

    return text_problems.text2text_txt_iterator(train_path + "_x",
                                                train_path + "_y")
