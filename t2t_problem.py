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
  hp = common_hparams.basic_params1()
  # Modify existing hparams
  hp.target_modality = 'symbol:almond'
  #hp.num_hidden_layers = 2
  # Add new hparams
  #hp.add_hparam("filter_size", 2048)
  return hp

@registry.register_symbol_modality("almond")
class AlmondSymbolModality(modalities.SymbolModality):

    def loss(self, top_out, targets):
        logits = top_out
        cas
        with tf.name_scope("max_margin_loss", values=[logits, targets]):
            # For CTC we assume targets are 1d, [batch, length, 1, 1] here.
            targets_shape = targets.get_shape().as_list()
            logits_shape = logits.get_shape().as_list()
            print("logits_shape: ", logits_shape, "\n targets_shape: ", targets_shape)

            targets = tf.squeeze(targets, axis=[2, 3])
            logits = tf.squeeze(logits, axis=[2, 3])
            targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
            targets_lengths = tf.reduce_sum(targets_mask, axis=1)
            sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
                targets, targets_lengths)
            xent = tf.nn.ctc_loss(
                sparse_targets,
                logits,
                targets_lengths,
                time_major=False,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=False)
            weights = self.targets_weights_fn(targets)  # pylint: disable=not-callable
            return tf.reduce_sum(xent), tf.reduce_sum(weights)




@registry.register_problem
class ParseAlmondCommands(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def approx_vocab_size(self):
    return 4604

  @property
  def vocab_filename(self):
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

    if train:
        train_path = "../dataset/t2t_data/t2t_train"
    else:
        train_path = "../dataset/t2t_data/t2t_dev"
    #train_path = _get_wmt_ende_bpe_dataset(tmp_dir, dataset_path)

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
