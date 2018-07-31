# Copyright 2018 Google LLC
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
Created on Jul 26, 2018

@author: gcampagn
'''

# Part of this code is derived from Tensor2Tensor, which is:
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

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import Transformer as OriginalTransformer
from tensor2tensor.models.transformer import features_to_nonpadding, \
    fast_decode, transformer_prepare_decoder
from tensor2tensor.layers import common_layers, common_attention

from .base_model import LUINetModel

@registry.register_model("luinet_transformer")
class Transformer(OriginalTransformer, LUINetModel):
    '''
    The standard TransformerModel, but also inherits
    from LUINetModel so it's compatible with the rest
    of the LUINet library, and it also overrides some
    of the features to make the job of CopyTransformer
    a little easier.
    '''
    
    def eval_autoregressive(self, features=None, decode_length=50):
        """Autoregressive eval.
    
        Quadratic time in decode_length.
    
        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
    
        Returns:
          logits: `Tensor`
          losses: a dictionary: {loss-name (string): floating point `Scalar`}.
              Contains a single key "training".
        """
        results = self._greedy_infer(features, decode_length=decode_length)
        return results["logits"], results["losses"]
    
    def _fast_decode_tpu(self,
                         features,
                         decode_length,
                         beam_size=1):
        raise NotImplementedError("TPU decoding is not implemented")

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        """Fast decoding.
        
        Overrides tensor2tensor.models.transformer.Transformer._fast_decode
        to let symbols_to_logits_fn be a method of this class.
    
        Implements both greedy and beam search decoding, uses beam search iff
        beam_size > 1, otherwise beam search related arguments are ignored.
    
        Args:
          features: a map of string to model  features.
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.
    
        Returns:
          A dict of decoding results {
              "body_output": tensor of size
                  [batch_size, <= decode_length, hidden_size]
                  (or [batch_size, top_beams, <= decode_length, hidden_size])
                  giving the raw output of the Transformer decoder corresponding
                  to the predicted sequences
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
    
        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.target_modality
        if isinstance(target_modality, dict):
            target_modality = target_modality["targets"]
    
        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                    common_layers.shape_list(inputs)[1] + features.get(
                        "decode_length", decode_length))
    
            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.input_modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None
    
            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]
    
        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                hparams.max_length, "targets_positional_embedding", None)
        else:
            positional_encoding = None
    
        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.
    
              This includes:
                - Embedding the ids.
                - Flattening to 3D tensor.
                - Optionally adding timing signals.
    
              Args:
                targets: inputs ids to the decoder. [batch_size, 1]
                i: scalar, Step number of the decoding loop.
    
              Returns:
                Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)
    
            # At step 0, targets will have 0 size, and instead we want to
            # create an embedding of all-zero, corresponding to the start symbol
            # this matches what transformer_prepare_decoder does to the target
            # outputs during training
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)
    
            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets
    
        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)
    
        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]
        
                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
                        -1e9)
        
                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache
    
        ret = fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        
        if beam_size > 1 and top_beams > 1:
            output_shape_list = common_layers.shape_list(ret["outputs"])
            decoder_input = tf.reshape(ret["outputs"],
                                       [-1] + output_shape_list[2:])
        else:
            decoder_input = ret["outputs"]
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            decoder_input, hparams, features=features)
            
        decoder_output = self.decode(decoder_input, encoder_output,
                                     encoder_decoder_attention_bias,
                                     decoder_self_attention_bias,
                                     hparams,
                                     nonpadding=features_to_nonpadding(features, "targets"))
        if beam_size > 1 and top_beams > 1:
            decoder_output = tf.reshape(decoder_output,
                                        output_shape_list + decoder_output.shape[-1])
        
        ret["body_output"] = decoder_output
        ret["encoder_output"] = encoder_output
        ret["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        return ret
