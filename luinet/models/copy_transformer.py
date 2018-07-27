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
from tensor2tensor.models.transformer import Transformer
from tensor2tensor.models.transformer import features_to_nonpadding
from tensor2tensor.models.transformer import transformer_prepare_decoder
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

from ..layers.modalities import CopyModality
from ..layers.common import AttentivePointerLayer,\
    DecayingAttentivePointerLayer, EmbeddingPointerLayer

from .base_model import LUINetModel

@registry.register_model("luinet_copy_transformer")
class CopyTransformer(Transformer, LUINetModel):
    '''
    A Transformer subclass that supports copying from the inputs.
    '''
             
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
        inputs_3d = common_layers.flatten4d3d(inputs)
        
        target_space = features["target_space_id"]
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, target_space, hparams, features=features, losses=losses)
    
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
            if key == "targets":
                body_output[key] = decoder_output
            elif isinstance(modality, CopyModality):
                with tf.variable_scope("copy_layer/" + key):
                    output_layer = AttentivePointerLayer(encoder_output)
                    scores = output_layer(decoder_output)
                    scores += encoder_decoder_attention_bias
                    body_output[key] = scores
            else:
                body_output[key] = decoder_output
        
        if losses:
          return body_output, {"extra_loss": tf.add_n(losses)}
        else:
          return body_output