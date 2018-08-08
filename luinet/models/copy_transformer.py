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

from collections import OrderedDict
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import features_to_nonpadding
from tensor2tensor.models.transformer import transformer_prepare_decoder
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

from ..layers.modalities import CopyModality
from ..layers.common import AttentivePointerLayer,\
    DecayingAttentivePointerLayer, EmbeddingPointerLayer

from .transformer import Transformer


@registry.register_model("luinet_copy_transformer")
class CopyTransformer(Transformer):
    '''
    A Transformer subclass that supports copying from the inputs.
    '''
    
    def _symbols_to_logits_fn(self,
                              targets,
                              features,
                              bias,
                              cache):
        with tf.variable_scope("body"):
            decoder_outputs = self._data_parallelism(
                self.decode,
                targets,
                cache.get("encoder_output"),
                cache.get("encoder_decoder_attention_bias"),
                bias,
                self.hparams,
                cache,
                nonpadding=features_to_nonpadding(features, "targets"))

        logits = dict()
        target_modality = self._problem_hparams.target_modality
            
        def copy_sharded(decoder_output):
            output_layer = DecayingAttentivePointerLayer(cache.get("encoder_output"))
            scores = output_layer(decoder_output)
            scores += cache.get("encoder_decoder_attention_bias")
            return scores

        for key, modality in target_modality.items():
            if isinstance(modality, CopyModality):
                with tf.variable_scope("body"):
                    with tf.variable_scope("copy_layer/" + key):
                        body_outputs = self._data_parallelism(copy_sharded,
                                                              decoder_outputs)
            else:
                body_outputs = decoder_outputs

            with tf.variable_scope(key):
                with tf.variable_scope(modality.name):
                    logits[key] = modality.top_sharded(body_outputs, None,
                                                       self._data_parallelism)[0]
            
            cache["logits"][key] = tf.concat((cache["logits"][key], logits[key]), axis=1)
            if key != self._problem_hparams.primary_target_modality:
                squeezed_logits = tf.squeeze(logits[key], axis=[1,2,3])
                current_sample = tf.argmax(squeezed_logits, axis=-1)
                current_sample = tf.expand_dims(current_sample, axis=1)
                current_sample = tf.expand_dims(current_sample, axis=2)
                
                cache["outputs_" + key] = tf.concat((cache["outputs_" + key], current_sample),
                                                    axis=1)
        
        return logits[self._problem_hparams.primary_target_modality]
    
    def _prepare_decoder_cache(self,
                               batch_size,
                               beam_size,
                               cache):
        cache["logits"] = dict()
        target_modality = self._problem_hparams.target_modality
        
        for key, modality in target_modality.items():
            cache["logits"][key] = tf.zeros((batch_size * beam_size, 0, 1, 1, modality.top_dimensionality))
            # the last dimension of all cache tensors must be defined and fixed in the loop
            cache["outputs_" + key] = tf.zeros((batch_size * beam_size, 0, 1), dtype=tf.int64)

    def _fast_decode(self, 
        features, 
        decode_length, 
        beam_size=1, 
        top_beams=1, 
        alpha=1.0):
        ret = super()._fast_decode(features, decode_length,
                                   beam_size=beam_size,
                                   top_beams=top_beams,
                                   alpha=alpha)
        
        new_outputs = dict()
        target_modality = self._problem_hparams.target_modality
        for key in target_modality:
            if key == self._problem_hparams.primary_target_modality:
                new_outputs[key] = ret["outputs"]
                del ret["outputs"]
            else:
                # remove the extra dimension that was added to appease the shape
                # invariants
                new_outputs[key] = tf.squeeze(ret["outputs_" + key], axis=2)
                del ret["outputs_" + key]
                
        print(new_outputs)
        new_outputs[self._problem_hparams.primary_target_modality] = \
            tf.Print(new_outputs[self._problem_hparams.primary_target_modality],
                     data=(tf.shape(new_outputs[self._problem_hparams.primary_target_modality]),
                           tf.shape(new_outputs['targets_COPY_SPAN_begin'])))
        ret["outputs"] = new_outputs
        return ret

    def body(self, features):
        """CopyTransformer main model_fn.
    
        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
              "targets": Target decoder outputs.
                  [batch_size, decoder_length, hidden_dim]
              "targets_*": Additional decoder outputs to generate, for copying
                  and pointing; [batch_size, decoder_length]
              "target_space_id": A scalar int from data_generators.problem.SpaceID.
    
        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        hparams = self._hparams
    
        losses = []
    
        inputs = features["inputs"]
        
        target_space = features["target_space_id"]
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, target_space, hparams, features=features, losses=losses)
    
        if "targets_actions" in features:
            targets = features["targets_actions"]
        else:
            tf.logging.warn("CopyTransformer must be used with a SemanticParsing problem with a ShiftReduceGrammar; bad things will happen otherwise")
            targets = features["targets"]
        
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
    
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            targets, hparams, features=features)
    
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, "targets"),
            losses=losses)
    
        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}
    
        decoder_output = tf.reshape(decoder_output, targets_shape)
        
        body_output = dict()
        target_modality = self._problem_hparams.target_modality \
            if self._problem_hparams else {"targets": None} 
        for key, modality in target_modality.items():
            if isinstance(modality, CopyModality):
                with tf.variable_scope("copy_layer/" + key):
                    output_layer = DecayingAttentivePointerLayer(encoder_output)
                    scores = output_layer(decoder_output)
                    scores += encoder_decoder_attention_bias
                    body_output[key] = scores
            else:
                body_output[key] = decoder_output
        
        if losses:
            return body_output, {"extra_loss": tf.add_n(losses)}
        else:
            return body_output