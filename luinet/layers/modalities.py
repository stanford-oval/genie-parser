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
from tensor2tensor.layers import common_layers
'''
Created on Jul 24, 2018

@author: gcampagn
'''

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.layers.modalities import IdentitySymbolModality,\
    SymbolModality

from .common import EmbeddingPointerLayer


def pad_to_length(logits, targets):
    length_diff = tf.shape(targets)[1] - tf.shape(logits)[1]
    padding = tf.convert_to_tensor([[0, 0], [0, length_diff], [0, 0]], name='padding')
    
    orig_shape = logits.shape
    logits = tf.pad(logits, padding, mode='constant')
    logits.set_shape(tf.TensorShape([orig_shape[0], targets.shape[1], orig_shape[2]]))
    return logits

@registry.register_symbol_modality("max_margin")
class MaxMarginModality(SymbolModality):
    @property
    def name(self):
        return super().name + "_maxmargin"
    
    def loss(self, logits, targets):
        # top_out is [batch, time, width, height, depth]
        # (to support videos)
        # remove width and height
        logits = tf.squeeze(logits, axis=[2, 3])

        logits = pad_to_length(logits, targets)
        batch_size, max_length, num_classes = common_layers.shape_list(logits)

        targets_mask = tf.to_float(tf.not_equal(targets, 0))

        with tf.name_scope("max_margin_loss", values=[logits, targets]):
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

            return tf.reduce_sum(margin * flat_mask), tf.reduce_sum(flat_mask)


@registry.register_symbol_modality("copy")
class CopyModality(IdentitySymbolModality):
    '''
    Almost same as IdentitySymbolModality, but it's a subclass so
    CopyTransformer can learn to recognize it.
    '''

    @property
    def name(self):
        return "symbol_copy"

    @property
    def top_is_pointwise(self):
        # top_is_pointwise means it's OK to call this function
        # with just the last decoding step (for speed)
        # IdentitySymbolModality sets top_is_pointwise to False
        # for generality, but CopyModality fits the bill
        return True
    
    def top(self, body_output, _):
        # expand the dimensions to be compatible with top_is_pointwise
        return tf.expand_dims(body_output, 3)

    def loss(self, logits, targets):
        # top_out is [batch, time, width, height, depth]
        # (to support videos)
        # remove width and height
        logits = tf.squeeze(logits, axis=[2, 3])
        length_diff = tf.shape(targets)[1] - tf.shape(logits)[1]
        
        def if_targets_longer(logits, targets):
            padding = tf.convert_to_tensor([[0, 0], [0, length_diff], [0, 0]], name='padding')
            orig_shape = logits.shape
            logits = tf.pad(logits, padding, mode='constant')
            logits.set_shape(tf.TensorShape([orig_shape[0], targets.shape[1], orig_shape[2]]))
            return logits, targets
        def if_logits_longer(logits, targets):
            padding = tf.convert_to_tensor([[0, 0], [0, -length_diff]], name='padding')
            orig_shape = targets.shape
            targets = tf.pad(targets, padding, mode='constant')
            targets.set_shape(tf.TensorShape([orig_shape[0], logits.shape[1]]))
            return logits, targets
        logits, targets = tf.cond(length_diff > 0,
                                  lambda: if_targets_longer(logits, targets),
                                  lambda: if_logits_longer(logits, targets))
        
        mask = tf.to_float(tf.not_equal(targets, 0))
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        
        # add a small amount to the denominator to avoid 0/0 division
        # if there is nothing to copy
        return tf.reduce_sum(xent * mask), tf.reduce_sum(mask) + 1e-8


# do not register this one, it is registered on demand by
# tasks.SemanticParsing; this is so multiple inputs/outputs
# can use the same class, but have different embedding matrices
class PretrainedEmbeddingModality(SymbolModality):
    def __init__(self, name, pretrained_embeddings, model_hparams,
                 vocab_size=None):
        super().__init__(model_hparams, vocab_size)
        assert vocab_size is None or pretrained_embeddings.shape[0] == vocab_size
        self._pretrained = pretrained_embeddings
        self._name = name
        self._trainable = model_hparams.train_input_embeddings
    
    @property
    def name(self):
        return "symbol_pretrained_" + self._name
    
    def _get_weights(self, hidden_dim=None):
        current_scope = tf.get_variable_scope()
        if current_scope.name.endswith("/softmax"):
            # for softmax, we never return the pretrained embeddings
            return super()._get_weights(hidden_dim)
        
        if self._trainable:
            return tf.get_variable("weights", self._pretrained.shape,
                                   initializer=tf.constant_initializer(self._pretrained))
        else:
            return tf.constant(self._pretrained)
        
    def bottom_simple(self, x, name, reuse):
        y = super().bottom_simple(x, name, reuse)
        if name != "softmax":
            # the pretrained matrix will have arbitrary, and often large, size
            # so we linearly project it to the correct expected size
            # bad things happen if we don't
            y = tf.layers.dense(y, self._model_hparams.hidden_size,
                                use_bias=False,
                                reuse=reuse,
                                name=name + "_projection")
        return y


# do not register this one, it is registered on demand by
# tasks.SemanticParsing; this is so multiple inputs/outputs
# can use the same class, but have different parameter vectors
class PointerModality(SymbolModality):
    def __init__(self, name, hparams, vocab_size=None):
        super().__init__(hparams, vocab_size)
        self._name = name
        
    @property
    def name(self):
        return "symbol_pointer_" + self._name + ("_%d_%d" % (self._vocab_size, self._body_input_depth))
    
    def bottom(self, x):
        return self.bottom_simple(x, "shared")
    
    def targets_bottom(self, x):
        return self.bottom_simple(x, "shared")
    
    def top(self, body_output, _):    
        body_output_shape = common_layers.shape_list(body_output)
        with tf.variable_scope("shared", reuse=True):
            embeddings = self._get_weights(self._body_input_depth)
        
        hidden_size = self._model_hparams.hidden_size
        dropout = self._model_hparams.dropout
        with tf.variable_scope("pointer_modality"):
            kernel2 = tf.get_variable('kernel', shape=(body_output_shape[-1], hidden_size))
            bias = tf.get_variable('bias', shape=(hidden_size,),)
            output_projection = tf.get_variable('output_projection', shape=(hidden_size,))
            
            matmul2 = tf.tensordot(body_output, kernel2, [[len(body_output_shape)-1], [0]])
            
            for _ in range(len(body_output_shape)-1):
                embeddings = tf.expand_dims(embeddings, axis=0)
            
            matmul2 = tf.expand_dims(matmul2, axis=len(body_output_shape)-1)
            
            # this is a broadcasting add
            # the result has shape body_output_shape + [embedding_size, hidden_size]
            # this is fairly expensive, so it should be used with care
            # dot_product_attention might be a better choice if speed is an issue
            # (they are conceptually similar) 
            neuron_input = embeddings + matmul2
            neuron_input = tf.nn.bias_add(neuron_input, bias)
            activation = tf.nn.relu(neuron_input)
            activation = tf.nn.dropout(activation, keep_prob=1 - dropout)
            
            scores = tf.tensordot(activation, output_projection, [[len(body_output_shape)], [0]])
            
            # top should return [batch, width, height, ?, logits]
            # this is somewhat weird but I guess it's useful to deal with images and videos?
            return tf.expand_dims(scores, axis=3)