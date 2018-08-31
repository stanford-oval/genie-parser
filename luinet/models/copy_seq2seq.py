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
Created on Jan 31, 2018

@author: gcampagn
'''

import copy
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.models.lstm import lstm_bid_encoder, lstm_attention_decoder
from tensor2tensor.utils import beam_search
from tensor2tensor.layers import common_layers, common_attention

from .base_model import LUINetModel
from ..layers.common import AttentivePointerLayer, DecayingAttentivePointerLayer
from ..layers import common
from ..layers.modalities import CopyModality


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


def dropout_lstm_cell(hparams):
    return tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.LSTMCell(hparams.hidden_size),
        
        # note that dropout is set to 0 if the model
        # is not training (T2TModel does it)
        # also note that we need the DropoutWrapper
        # regardless to make sure that variable names
        # stay consistent
        input_keep_prob=1.0 - hparams.dropout)
  

def construct_decoder_cell(hparams, encoder_outputs, encoder_output_length):
    layers = [dropout_lstm_cell(hparams)
              for _ in range(hparams.num_hidden_layers)]
    if hparams.attention_mechanism == "luong":
        attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
    elif hparams.attention_mechanism == "bahdanau":
        attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
    else:
        raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                         "luong or bahdanau." % hparams.attention_mechanism)
    attention_mechanism = attention_mechanism_class(
        hparams.hidden_size, encoder_outputs,
        memory_sequence_length=encoder_output_length)
    
    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism]*hparams.num_heads,
        attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
        output_attention=(hparams.output_attention == 1))
    
    return cell


@registry.register_model("luinet_copy_seq2seq")
class CopySeq2SeqModel(LUINetModel):
    '''
    A model that implements Seq2Seq with an extensible (pointer-based) grammar:
    that is, it uses a sequence loss during training, and a greedy decoder during inference 
    '''
    
    def encode(self, inputs, hparams, features=None):
        train = hparams.mode == tf.estimator.ModeKeys.TRAIN
        inputs_length = common_layers.length_from_embedding(inputs)
        
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)
        
        encoder_padding = common_attention.embedding_to_padding(inputs)
        encoder_decoder_attention_bias = common_attention.attention_bias_ignore_padding(
            encoder_padding)

        # LSTM encoder.
        encoder_outputs, final_encoder_state = lstm_bid_encoder(
            inputs, inputs_length, self._hparams, train, "encoder")
        
        return encoder_outputs, final_encoder_state, encoder_decoder_attention_bias, inputs_length
    
    def body(self, features):
        inputs = features["inputs"]
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        
        encoder_outputs, final_encoder_state, encoder_decoder_attention_bias, inputs_length = \
            self.encode(inputs, self._hparams)
        
        if "targets_actions" in features:
            targets = features["targets_actions"]
        else:
            tf.logging.warn(
                "CopySeq2Seq must be used with a SemanticParsing problem with a ShiftReduceGrammar; bad things will happen otherwise")
            targets = features["targets"]
        
        # LSTM decoder with attention
        shifted_targets = common_layers.shift_right(targets)

        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1
        shifted_targets = common_layers.flatten4d3d(shifted_targets)
        
        hparams_decoder = copy.copy(self._hparams)
        hparams_decoder.hidden_size = 2 * self._hparams.hidden_size
        
        decoder_output = lstm_attention_decoder(shifted_targets, hparams_decoder, train,
            "decoder", final_encoder_state, encoder_outputs,
            inputs_length, targets_length)
        decoder_output = tf.expand_dims(decoder_output, axis=2)
        
        body_output = dict()
        target_modality = self._problem_hparams.target_modality \
            if self._problem_hparams else {"targets": None}

        assert self._hparams.pointer_layer in ("attentive", "decaying_attentive")

        for key, modality in target_modality.items():
            if isinstance(modality, CopyModality):
                with tf.variable_scope("copy_layer/" + key):
                    if self._hparams.pointer_layer == "decaying_attentive":
                        output_layer = DecayingAttentivePointerLayer(encoder_outputs)
                    else:
                        output_layer = AttentivePointerLayer(encoder_outputs)
                    scores = output_layer(decoder_output)
                    scores += encoder_decoder_attention_bias
                    body_output[key] = scores
            else:
                body_output[key] = decoder_output
        
        return body_output
    
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
        tf.logging.info("Using autoregressive decoder for evaluation")
        self._fill_problem_hparams_features(features)
        results = self._greedy_infer(features, decode_length=decode_length)
        
        logits = results["logits"]
        losses = self.loss(logits, features)
        
        return logits, losses
    
    def _greedy_infer(self, features, decode_length, use_tpu=False):
        """Fast version of greedy decoding.
    
        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          use_tpu: A bool. Whether to build the inference graph for TPU.
    
        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
    
        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        assert not use_tpu
        with tf.variable_scope(self.name):
            return self._fast_decode(features, decode_length)

    def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
        """Beam search decoding.
    
        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.
    
        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
        """
        with tf.variable_scope(self.name):
            return self._fast_decode(features, decode_length, beam_size, top_beams,
                                     alpha)

    def _prepare_decoder_cache(self,
                               batch_size,
                               beam_size,
                               features,
                               cache):
        cache["logits"] = dict()
        target_modality = self._problem_hparams.target_modality

        for key, modality in target_modality.items():
            if isinstance(modality, CopyModality):
                cache["logits"][key] = tf.zeros((batch_size * beam_size, 0, 1, 1, tf.shape(features["inputs"])[1]))
            else:
                cache["logits"][key] = tf.zeros((batch_size * beam_size, 0, 1, 1, modality.top_dimensionality))
            # the last dimension of all cache tensors must be fixed in the loop
            cache["outputs_" + key] = tf.zeros((batch_size * beam_size, 0, 1), dtype=tf.int64)

    def _symbols_to_logits_fn(self,
                              targets,
                              features,
                              cell,
                              cache):
        with tf.variable_scope("body/decoder/rnn"):
            
            # targets has shape [batch, 1, depth]
            # but RNN does not like the extra 1
            def dp_step(cell, targets, cell_state):
                targets = tf.squeeze(targets, axis=1)
                output, state = cell(targets, cell_state)
                # restore the time dimension we just squeezed
                output = tf.expand_dims(output, axis=1)
                # expand to 4d so we can feed to the modality
                output = tf.expand_dims(output, axis=2)
                return output, state
            
            decoder_output, decoder_state = self._data_parallelism(
                dp_step, cell, targets, cache['cell_state'])
        cache['cell_state'] = decoder_state

        logits = dict()
        target_modality = self._problem_hparams.target_modality

        assert self.hparams.pointer_layer in ("attentive", "decaying_attentive")

        def copy_sharded(decoder_output):
            encoder_output = cache.get("encoder_output")
            if self.hparams.pointer_layer == "decaying_attentive":
                output_layer = DecayingAttentivePointerLayer(encoder_output)
                scores = output_layer(decoder_output)
            else:
                output_layer = AttentivePointerLayer(encoder_output)
                scores = output_layer(decoder_output)
            scores += cache.get("encoder_decoder_attention_bias")
            return scores

        for key, modality in target_modality.items():
            if isinstance(modality, CopyModality):
                with tf.variable_scope("body"):
                    with tf.variable_scope("copy_layer/" + key):
                        body_outputs = self._data_parallelism(copy_sharded,
                                                              decoder_output)
            else:
                body_outputs = decoder_output

            with tf.variable_scope(key):
                with tf.variable_scope(modality.name):
                    logits[key] = modality.top_sharded(body_outputs, None,
                                                       self._data_parallelism)[0]

            cache["logits"][key] = tf.concat((cache["logits"][key], logits[key]), axis=1)
            if key != self._problem_hparams.primary_target_modality:
                squeezed_logits = tf.squeeze(logits[key], axis=[1, 2, 3])
                current_sample = tf.argmax(squeezed_logits, axis=-1)
                current_sample = tf.expand_dims(current_sample, axis=1)
                current_sample = tf.expand_dims(current_sample, axis=2)

                cache["outputs_" + key] = tf.concat((cache["outputs_" + key], current_sample),
                                                    axis=1)

        return logits[self._problem_hparams.primary_target_modality]

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        """Fast decoding.
    
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
            primary_target_feature = self._problem_hparams.primary_target_modality
            primary_target_modality = target_modality[primary_target_feature]
            bottom_variable_scope = "%s/%s" % (primary_target_modality.name, primary_target_feature)
        else:
            primary_target_feature = "targets"
            primary_target_modality = target_modality
            bottom_variable_scope = target_modality.name
            
        inputs = features["inputs"]
        s = common_layers.shape_list(inputs)
        batch_size = s[0]
        
        # _shard_features called to ensure that the variable names match
        inputs = self._shard_features({"inputs": inputs})["inputs"]
        input_modality = self._problem_hparams.input_modality["inputs"]
        with tf.variable_scope(input_modality.name):
            inputs = input_modality.bottom_sharded(inputs, dp)
        with tf.variable_scope("body"):
            encoder_output, final_encoder_state, encoder_decoder_attention_bias, input_length = dp(
                self.encode,
                inputs,
                hparams,
                features=features)
            
            # undo the data parallelism
            encoder_output = encoder_output[0]
            final_encoder_state = final_encoder_state[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            input_length = input_length[0]
    
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
            targets = self._shard_features({primary_target_feature: targets})[primary_target_feature]
            with tf.variable_scope(bottom_variable_scope):
                targets = primary_target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)
    
            # At step 0, targets will have 0 size, and instead we want to
            # create an embedding of all-zero, corresponding to the start symbol
            # this matches the common_layers.shift_right() that we do at training time
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)
            return targets
    
        hparams_decoder = copy.copy(self._hparams)
        hparams_decoder.hidden_size = 2 * self._hparams.hidden_size
        
        def dp_initial_state(encoder_output, input_length, final_encoder_state):
            decoder_cell = construct_decoder_cell(hparams_decoder, encoder_output, input_length)
            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=final_encoder_state)
            return decoder_cell, initial_state
        
        with tf.variable_scope("body"):
            decoder_cell, initial_state = self._data_parallelism(
                dp_initial_state, encoder_output, input_length, final_encoder_state)
    
        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            logits = self._symbols_to_logits_fn(targets, features, decoder_cell, cache)
            logits = tf.squeeze(logits, axis=[1, 2, 3])
            return logits, cache
    
        cache = dict()
        infer_out = dict()
        if encoder_output is not None:
            padding_mask = 1. - common_attention.attention_bias_to_padding(encoder_decoder_attention_bias)
            masked_encoded_output = encoder_output * tf.expand_dims(padding_mask, axis=2)

            infer_out["encoded_inputs"] = tf.reduce_sum(masked_encoded_output, axis=1)

        self._prepare_decoder_cache(batch_size, beam_size, features, cache)
        cache['cell_state'] = initial_state
        cache['encoder_output'] = encoder_output
        cache['encoder_decoder_attention_bias'] = encoder_decoder_attention_bias

        ret = common.fast_decode(
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=primary_target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length,
            cache=cache)
        infer_out.update(ret)
        
        new_outputs = dict()
        target_modality = self._problem_hparams.target_modality
        for key in target_modality:
            if key == self._problem_hparams.primary_target_modality:
                new_outputs[key] = infer_out["outputs"]
                del infer_out["outputs"]
            else:
                # remove the extra dimension that was added to appease the shape
                # invariants
                new_outputs[key] = tf.squeeze(infer_out["outputs_" + key], axis=2)
                del infer_out["outputs_" + key]

        infer_out["outputs"] = new_outputs
        return infer_out
