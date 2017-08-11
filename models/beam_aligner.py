# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
'''
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from tensorflow.python.layers import core as tf_core_layers
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper, BeamSearchDecoder

from collections import namedtuple

from .base_aligner import BaseAligner

BeamSearchOptimizationDecoderOutput = namedtuple('BeamSearchOptimizationDecoderOutput', ('scores', 'predicted_ids', 'parent_ids', 'loss'))
BeamSearchOptimizationDecoderState = namedtuple('BeamSearchOptimizationDecoderState', ('cell_state', 'gold_cell_state', 'previous_scores', 'previous_gold_scores', 'finished'))
FinalBeamSearchOptimizationDecoderOutput = namedtuple('FinalBeamSearchOptimizationDecoderOutput', ('beam_search_decoder_output', 'predicted_ids', 'total_loss'))

# Some of the code here was copied from Tensorflow contrib/seq2seq/python/ops/beam_search_decoder.py
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

class BeamSearchOptimizationDecoder():
    def __init__(self, training, cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer=None, gold_sequence=None, gold_sequence_length=None):
        self._training = training
        self._cell = cell
        self._output_layer = output_layer
        self._embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)

        self._batch_size = tf.size(start_tokens)
        self._initial_cell_state = nest.map_structure(
            self._maybe_split_batch_beams,
            initial_state)
        self._start_tokens = tf.tile(tf.expand_dims(start_tokens, axis=1), [1, self._beam_width])
        self._end_token = end_token
        self._beam_width = beam_width

        self._gold_sequence = gold_sequence
        self._gold_sequence_length = gold_sequence_length
        if training:
            assert self._gold_sequence is not None
            assert self._gold_sequence_length is not None
            # transpose gold sequence to be time major
            self._gold_sequence = tf.transpose(gold_sequence, [1, 0])
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def output_size(self):
        # Return the cell output and the id
        return BeamSearchOptimizationDecoderOutput(
            scores=tf.TensorShape([self._beam_width]),
            predicted_ids=tf.TensorShape([self._beam_width]),
            parent_ids=tf.TensorShape([self._beam_width]),
            loss=tf.TensorShape(()))
        
    @property
    def output_dtype(self):
        return BeamSearchOptimizationDecoderOutput(
            scores=tf.float32,
            predicted_ids=tf.int32,
            parent_ids=tf.int32,
            loss=tf.float32)
    
    def _merge_batch_beams(self, t):
        """Merges the tensor from a batch of beams into a batch by beams.
        More exactly, t is a tensor of dimension [batch_size, beam_width, s]. We
        reshape this into [batch_size*beam_width, s]
        Args:
          t: Tensor of dimension [batch_size, beam_width, s]
        Returns:
          A reshaped version of t with dimension [batch_size * beam_width, s].
        """
        t_shape = tf.shape(t)
        return tf.reshape(t, tf.concat(([self._batch_size * self._beam_width], t_shape[2:]), axis=0))

    def _split_batch_beams(self, t):
        """Splits the tensor from a batch by beams into a batch of beams.
        More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
        reshape this into [batch_size, beam_width, s]
        Args:
          t: Tensor of dimension [batch_size*beam_width, s].
          s: (Possibly known) depth shape.
        Returns:
          A reshaped version of t with dimension [batch_size, beam_width, s].
        Raises:
          ValueError: If, after reshaping, the new tensor is not shaped
            `[batch_size, beam_width, s]` (assuming batch_size and beam_width
            are known statically).
        """
        t_shape = tf.shape(t)
        return tf.reshape(t, tf.concat(([self._batch_size, self._beam_width], t_shape[1:]), axis=0))

    def _maybe_split_batch_beams(self, t):
        """Maybe splits the tensor from a batch by beams into a batch of beams.
        We do this so that we can use nest and not run into problems with shapes.
        Args:
          t: Tensor of dimension [batch_size*beam_width, s]
          s: Tensor, Python int, or TensorShape.
        Returns:
          Either a reshaped version of t with dimension
          [batch_size, beam_width, s] if t's first dimension is of size
          batch_size*beam_width or t if not.
        Raises:
          TypeError: If t is an instance of TensorArray.
          ValueError: If the rank of t is not statically known.
        """
        return self._split_batch_beams(t) if t.shape.ndims >= 1 else t 

    def _maybe_merge_batch_beams(self, t):
        """Splits the tensor from a batch by beams into a batch of beams.
        More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
        reshape this into [batch_size, beam_width, s]
        Args:
          t: Tensor of dimension [batch_size*beam_width, s]
          s: Tensor, Python int, or TensorShape.
        Returns:
          A reshaped version of t with dimension [batch_size, beam_width, s].
        Raises:
          TypeError: If t is an instance of TensorArray.
          ValueError:  If the rank of t is not statically known.
        """
        return self._merge_batch_beams(t) if t.shape.ndims >= 2 else t
    
    def initialize(self):
        """Initialize the decoder.
        Args:
          name: Name scope for any created operations.
        Returns:
          `(finished, start_inputs, initial_state)`.
        """
        start_inputs = self._embedding_fn(self._start_tokens)
        finished = tf.zeros((self._batch_size, self._beam_width), dtype=tf.bool)

        initial_state = BeamSearchOptimizationDecoderState(
            cell_state=self._initial_cell_state,
            gold_cell_state=self._initial_cell_state,
            previous_scores=tf.zeros([self._batch_size, self._beam_width], dtype=tf.float32),
            previous_gold_scores=tf.zeros([self._batch_size, self._beam_width], dtype=tf.float32),
            finished=finished)
        
        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state : BeamSearchOptimizationDecoderState , name=None):
        """Perform a decoding step.
        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.
        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token

        with tf.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(self._merge_batch_beams, inputs)
            cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(self._split_batch_beams, cell_outputs)
            next_cell_state = nest.map_structure(self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)
            
            if self._training:
                gold_cell_state = state.gold_cell_state
                gold_cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state)
                gold_cell_outputs, next_gold_cell_state = self._cell(inputs, gold_cell_state)
                gold_cell_outputs = nest.map_structure(self._split_batch_beams, cell_outputs)
                next_gold_cell_state = nest.map_structure(self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)
            else:
                next_gold_cell_state = state.gold_cell_state

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                training=self._training,
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                gold_cell_outputs=gold_cell_outputs,
                next_gold_cell_state=next_gold_cell_state,
                gold_sequence=self._gold_sequence,
                gold_sequence_length=self._gold_sequence_length)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = tf.cond(tf.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            return (beam_search_output, beam_search_state, next_inputs, finished)
        
    def finalize(self, outputs : BeamSearchOptimizationDecoderOutput, final_state : BeamSearchOptimizationDecoderState, sequence_lengths):
        # all output fields are [max_time, batch_size, ...]
        predicted_ids = tf.contrib.seq2seq.gather_tree(
            outputs.predicted_ids, outputs.parent_ids,
            sequence_length=sequence_lengths)
        total_loss = tf.reduce_sum(outputs.loss, axis=0)
        return FinalBeamSearchOptimizationDecoderOutput(beam_search_decoder_output=outputs, predicted_ids=predicted_ids, total_loss=total_loss) 

def _beam_search_step(training, time, logits, next_cell_state, beam_state : BeamSearchOptimizationDecoderState, batch_size,
                      beam_width, end_token, gold_cell_outputs,
                      next_gold_cell_state, gold_sequence, gold_sequence_length):
    """Performs a single step of Beam Search Decoding.
    Args:
      time: Beam search time step, should start at 0. At time 0 we assume
        that all beams are equal and consider only the first beam for
        continuations.
      logits: Logits at the current time step. A tensor of shape
        `[batch_size, beam_width, vocab_size]`
      next_cell_state: The next state from the cell, e.g. an instance of
        AttentionWrapperState if the cell is attentional.
      beam_state: Current state of the beam search.
        An instance of `BeamSearchDecoderState`.
      batch_size: The batch size for this input.
      beam_width: Python int.  The size of the beams.
      end_token: The int32 end token.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    Returns:
      A new beam state.
    """
    previously_finished = beam_state.finished
    
    # Calculate the scores for each beam
    #
    # Following Wiseman and Rush, we use the unnormalized logits of the current token
    # as the scores, without softmax
    # and WITHOUT SUMMING PREVIOUS TIME STEPS (this is different from other implementations
    # of beam search out there)
    scores = tf.where(previously_finished, beam_state.previous_scores, logits)
    print('scores', scores)
    
    # if we want to apply grammar constraints, this is the place to do it
    scores = tf.identity(scores)
    
    vocab_size = logits.shape[-1].value
    time = tf.convert_to_tensor(time, name="time")
    
    # During the first time step we only consider the initial beam
    scores_shape = tf.shape(scores)
    scores_flat = tf.cond(
        time > 0,
        lambda: tf.reshape(scores, [batch_size, -1]),
        lambda: scores[:, 0])
    num_available_beam = tf.cond(
        time > 0,
        lambda: tf.reduce_prod(scores_shape[1:]),
        lambda: tf.reduce_prod(scores_shape[2:]))
    

    # Pick the next beams according to the specified successors function
    next_beam_size = tf.minimum(tf.convert_to_tensor(beam_width, dtype=tf.int32, name="beam_width"), num_available_beam)
    next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=next_beam_size)
    next_beam_scores.set_shape([None, beam_width])
    word_indices.set_shape([None, beam_width])
    
    # Pick out the beam_ids, and states according to the chosen predictions
    next_word_ids = tf.to_int32(word_indices % vocab_size)
    next_beam_ids = tf.to_int32(word_indices / vocab_size)
    
    # Pick out the cell_states according to the next_beam_ids. We use a
    # different gather_shape here because the cell_state tensors, i.e.
    # the tensors that would be gathered from, all have dimension
    # greater than two and we need to preserve those dimensions.
    next_cell_state = nest.map_structure(
        lambda gather_from: _maybe_tensor_gather_helper(
            gather_indices=next_beam_ids,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=beam_width,
            gather_shape=[batch_size * beam_width, -1]),
        next_cell_state)
    
    # At training time, check for margin violations, and if so reset the beam
    if training:
        gold_finished = time >= gold_sequence_length
        gold_scores = tf.where(gold_finished, beam_state.previous_gold_scores, gold_cell_outputs)
        
        # the score of the last element of the beam
        beam_bottom_indices = tf.stack((tf.range(batch_size), tf.fill((batch_size,), beam_width-1)), axis=1)
        beam_bottom_score = tf.gather_nd(next_beam_scores, beam_bottom_indices)
        
        beam_violation = gold_scores < beam_bottom_score + 1
        
        loss = tf.where(beam_violation, 1 - gold_scores + beam_bottom_score, tf.zeros((batch_size,)))
        
        reset_token = gold_sequence[time+1]
        next_word_ids = tf.where(beam_violation, tf.tile(reset_token, [1, beam_width]), next_word_ids)
        next_beam_scores = tf.where(beam_violation, tf.tile(gold_scores, [1, beam_width], next_beam_scores))
    
        # Note: next_beam_ids is used only to reconstruct predicted_ids, so we leave it as is
        # in practice, it means that we're not fully resetting the beam, rather we're building a bastardized
        # beam that has the previous sequences
        # this is ok because predicted_ids is only used at inference time (where none of this resetting
        # business occurs)
        next_cell_state = nest.map_structure(lambda gold, predicted: tf.where(beam_violation, tf.tile(gold, [1, beam_width]), predicted),
                                             next_gold_cell_state,
                                             next_cell_state)
    else:
        loss = tf.zeros((batch_size,), dtype=tf.float32)

    previously_finished = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=previously_finished,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1])
    next_finished = tf.logical_or(previously_finished, tf.equal(next_word_ids, end_token))

    next_state = BeamSearchOptimizationDecoderState(
        cell_state=next_cell_state,
        gold_cell_state=next_gold_cell_state,
        previous_scores=scores,
        previous_gold_scores=gold_scores,
        finished=next_finished)
    
    output = BeamSearchOptimizationDecoderOutput(
        scores=next_beam_scores,
        predicted_ids=next_word_ids,
        parent_ids=next_beam_ids,
        loss=loss)
    
    return output, next_state

def _maybe_tensor_gather_helper(gather_indices, gather_from, batch_size,
                                range_size, gather_shape):
    """Maybe applies _tensor_gather_helper.
    This applies _tensor_gather_helper when the gather_from dims is at least as
    big as the length of gather_shape. This is used in conjunction with nest so
    that we don't apply _tensor_gather_helper to inapplicable values like scalars.
    Args:
      gather_indices: The tensor indices that we use to gather.
      gather_from: The tensor that we are gathering from.
      batch_size: The batch size.
      range_size: The number of values in each range. Likely equal to beam_width.
      gather_shape: What we should reshape gather_from to in order to preserve the
        correct values. An example is when gather_from is the attention from an
        AttentionWrapperState with shape [batch_size, beam_width, attention_size].
        There, we want to preserve the attention_size elements, so gather_shape is
        [batch_size * beam_width, -1]. Then, upon reshape, we still have the
        attention_size as desired.
    Returns:
      output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
        or the original tensor if its dimensions are too small.
    """
    if gather_from.shape.ndims >= len(gather_shape):
        return _tensor_gather_helper(
            gather_indices=gather_indices,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=range_size,
            gather_shape=gather_shape)
    else:
        return gather_from


def _tensor_gather_helper(gather_indices, gather_from, batch_size,
                          range_size, gather_shape):
    """Helper for gathering the right indices from the tensor.
    This works by reshaping gather_from to gather_shape (e.g. [-1]) and then
    gathering from that according to the gather_indices, which are offset by
    the right amounts in order to preserve the batch order.
    Args:
      gather_indices: The tensor indices that we use to gather.
      gather_from: The tensor that we are gathering from.
      batch_size: The input batch size.
      range_size: The number of values in each range. Likely equal to beam_width.
      gather_shape: What we should reshape gather_from to in order to preserve the
        correct values. An example is when gather_from is the attention from an
        AttentionWrapperState with shape [batch_size, beam_width, attention_size].
        There, we want to preserve the attention_size elements, so gather_shape is
        [batch_size * beam_width, -1]. Then, upon reshape, we still have the
        attention_size as desired.
    Returns:
      output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
    """
    range_ = tf.expand_dims(tf.range(batch_size) * range_size, 1)
    gather_indices = tf.reshape(gather_indices + range_, [-1])
    output = tf.gather(tf.reshape(gather_from, gather_shape), gather_indices)
    final_shape = tf.shape(gather_from)[:1 + len(gather_shape)]
    final_static_shape = (tf.TensorShape([None]).concatenate(gather_from.shape[1:1 + len(gather_shape)]))
    output = tf.reshape(output, final_shape)
    output.set_shape(final_static_shape)
    return output
    
class BeamAligner(BaseAligner):
    '''
    A Beam Search based semantic parser, using beam search for
    both training and inference
    '''
    
    def __init__(self, config):
        super().__init__(config)
        
        if config.beam_size <= 1:
            raise ValueError("Must specify a beam size of more than 1 with seq2seq model")

    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i) for i in range(self.config.rnn_layers)])
        if self.config.apply_attention:
            attention = LuongAttention(self.config.hidden_size, enc_hidden_states, self.input_length_placeholder,
                                       probability_fn=tf.nn.softmax)
            cell_dec = AttentionWrapper(cell_dec, attention,
                                        cell_input_fn=lambda inputs, _: inputs,
                                        attention_layer_size=self.config.hidden_size,
                                        initial_cell_state=enc_final_state)
            enc_final_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
        
        linear_layer = tf_core_layers.Dense(self.config.output_size)
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        
        print(enc_final_state)
        decoder = BeamSearchOptimizationDecoder(training, cell_dec, output_embed_matrix, go_vector, self.config.grammar.end,
                                                tf.contrib.seq2seq.tile_batch(enc_final_state, self.config.beam_size),
                                                beam_width=self.config.beam_size, output_layer=linear_layer,
                                                gold_sequence=self.input_placeholder if training else None,
                                                gold_sequence_length=self.input_length_placeholder if training else None)
        
        if self.config.use_grammar_constraints:
            raise NotImplementedError("Grammar constraints are not implemented for the beam search yet")
        
        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.config.max_length)
        return final_outputs
        
    def finalize_predictions(self, preds : FinalBeamSearchOptimizationDecoderOutput):
        # predicted_ids is [batch_size, max_time, beam_width] because that's how gather_tree produces it
        # transpose it to be [batch_size, beam_width, max_time] which is what we expect
        return tf.transpose(preds.predicted_ids, [0, 2, 1])
    
    def add_loss_op(self, preds : FinalBeamSearchOptimizationDecoderOutput):
        # For beam search, the loss is computed as we go along, so we just sum it here
        return preds.total_loss
