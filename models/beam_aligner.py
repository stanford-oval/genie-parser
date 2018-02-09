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
Created on Jul 25, 2017

@author: gcampagn
'''

import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper

from collections import namedtuple

from .base_aligner import BaseAligner
from . import common

BeamSearchOptimizationDecoderOutput = namedtuple('BeamSearchOptimizationDecoderOutput',
                                                 ('scores', 'gold_score', 'predicted_ids', 'parent_ids', 'loss'))
BeamSearchOptimizationDecoderState = namedtuple('BeamSearchOptimizationDecoderState',
                                                ('cell_state', 'previous_score', 'previous_logits', 'num_available_beams', 'gold_beam_id', 'finished'))
FinalBeamSearchOptimizationDecoderOutput = namedtuple('FinalBeamSearchOptimizationDecoderOutput',
                                                      ('beam_search_decoder_output', 'predicted_ids', 'scores', 'gold_score', 'gold_beam_id', 'num_available_beams', 'total_violation_loss'))

# Some of the code here was copied from Tensorflow contrib/seq2seq/python/ops/beam_search_decoder.py
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

class BeamSearchOptimizationDecoder(tf.contrib.seq2seq.Decoder):
    def __init__(self, training, cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer=None, gold_sequence=None, gold_sequence_length=None):
        self._training = training
        self._cell = cell
        self._output_layer = output_layer
        self._embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)

        self._output_size = output_layer.units if output_layer is not None else self._output.output_size
        self._batch_size = tf.size(start_tokens)
        self._beam_width = beam_width
        self._tiled_initial_cell_state = nest.map_structure(self._maybe_split_batch_beams, initial_state, self._cell.state_size)
        self._start_tokens = start_tokens
        self._tiled_start_tokens = self._maybe_tile_batch(start_tokens)
        self._end_token = end_token

        self._original_gold_sequence = gold_sequence
        self._gold_sequence = gold_sequence
        self._gold_sequence_length = gold_sequence_length
        if training:
            assert self._gold_sequence is not None
            assert self._gold_sequence_length is not None
            self._max_time = int(self._gold_sequence.shape[1])
            # transpose gold sequence to be time major and make it into a TensorArray
            self._gold_sequence = tf.TensorArray(dtype=tf.int32, size=self._max_time)
            self._gold_sequence = self._gold_sequence.unstack(tf.transpose(gold_sequence, [1, 0]))
    
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
            gold_score=tf.TensorShape(()),
            loss=tf.TensorShape(()))
        
    @property
    def output_dtype(self):
        return BeamSearchOptimizationDecoderOutput(
            scores=tf.float32,
            gold_score=tf.float32,
            predicted_ids=tf.int32,
            parent_ids=tf.int32,
            loss=tf.float32)

    def _tile_batch(self, t):
        if t.shape.ndims is None or t.shape.ndims < 1:
            raise ValueError("t must have statically known rank")
        tiling = [1] * (t.shape.ndims + 1)
        tiling[1] = self._beam_width
        tiled = tf.tile(tf.expand_dims(t, 1), tiling)
        return tiled

    def _maybe_tile_batch(self, t):
        return self._tile_batch(t) if t.shape.ndims >= 1 else t
    
    def _merge_batch_beams(self, t, s):
        """Merges the tensor from a batch of beams into a batch by beams.
        More exactly, t is a tensor of dimension [batch_size, beam_width, s]. We
        reshape this into [batch_size*beam_width, s]
        Args:
          t: Tensor of dimension [batch_size, beam_width, s]
        Returns:
          A reshaped version of t with dimension [batch_size * beam_width, s].
        """
        t_shape = tf.shape(t)
        reshaped = tf.reshape(t, tf.concat(([self._batch_size * self._beam_width], t_shape[2:]), axis=0))
        reshaped.set_shape(tf.TensorShape([None]).concatenate(s))
        return reshaped

    def _split_batch_beams(self, t, s):
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
        reshaped = tf.reshape(t, tf.concat(([self._batch_size, self._beam_width], t_shape[1:]), axis=0))
        reshaped.set_shape(tf.TensorShape([None, self._beam_width]).concatenate(t.shape[1:]))
        expected_reshaped_shape = tf.TensorShape([None, self._beam_width]).concatenate(s)
        if not reshaped.shape.is_compatible_with(expected_reshaped_shape):
            raise ValueError("Unexpected behavior when reshaping between beam width "
                             "and batch size.  The reshaped tensor has shape: %s.  "
                             "We expected it to have shape "
                             "(batch_size, beam_width, depth) == %s.  Perhaps you "
                             "forgot to create a zero_state with "
                             "batch_size=encoder_batch_size * beam_width?"
                             % (reshaped.shape, expected_reshaped_shape))
        return reshaped

    def _maybe_split_batch_beams(self, t, s):
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
        return self._split_batch_beams(t, s) if t.shape.ndims >= 1 else t 

    def _maybe_merge_batch_beams(self, t, s):
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
        return self._merge_batch_beams(t, s) if t.shape.ndims >= 2 else t
    
    def _make_beam_mask(self, num_available_beams):
        mask = tf.sequence_mask(num_available_beams, self._beam_width)
        return tf.tile(tf.expand_dims(mask, axis=2), multiples=[1, 1, self._output_size])
    
    def initialize(self):
        """Initialize the decoder.
        Args:
          name: Name scope for any created operations.
        Returns:
          `(finished, start_inputs, initial_state)`.
        """
        start_inputs = self._embedding_fn(self._tiled_start_tokens)
        print('start_inputs', start_inputs)
        finished = tf.zeros((self.batch_size, self._beam_width), dtype=tf.bool)
        
        self._initial_num_available_beams = tf.ones((self._batch_size,), dtype=tf.int32)
        self._full_num_available_beams = tf.fill((self._batch_size,), self._beam_width)
        
        with tf.name_scope('first_beam_mask'):
            self._first_beam_mask = self._make_beam_mask(self._initial_num_available_beams)
        with tf.name_scope('full_beam_mask'):
            self._full_beam_mask = self._make_beam_mask(self._full_num_available_beams)
        with tf.name_scope('minus_inifinity_scores'):
            self._minus_inifinity_scores = tf.fill((self.batch_size, self._beam_width, self._output_size), -1e+8)

        self._batch_size_range = tf.range(self.batch_size)
        initial_state = BeamSearchOptimizationDecoderState(
            cell_state=self._tiled_initial_cell_state,
            previous_logits=tf.zeros([self.batch_size, self._beam_width, self._output_size], dtype=tf.float32),
            previous_score=tf.zeros([self.batch_size, self._beam_width], dtype=tf.float32),
            # During the first time step we only consider the initial beam
            num_available_beams=self._initial_num_available_beams,
            gold_beam_id=tf.zeros([self.batch_size], dtype=tf.int32),
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
        with tf.name_scope(name, "BeamSearchOptimizationDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            with tf.name_scope('merge_cell_input'):
                inputs = nest.map_structure(lambda x: self._merge_batch_beams(x, s=x.shape[2:]), inputs)
            print('inputs', inputs)
            with tf.name_scope('merge_cell_state'):
                cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            with tf.name_scope('split_cell_outputs'):
                cell_outputs = nest.map_structure(self._split_batch_beams, cell_outputs, self._output_size)
            with tf.name_scope('split_cell_state'):
                next_cell_state = nest.map_structure(self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)
            
            beam_search_output, beam_search_state = self._beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = self._embedding_fn(sample_ids)
            return (beam_search_output, beam_search_state, next_inputs, finished)
        
    def finalize(self, outputs : BeamSearchOptimizationDecoderOutput, final_state : BeamSearchOptimizationDecoderState, sequence_lengths):
        # all output fields are [max_time, batch_size, ...]
        predicted_ids = tf.contrib.seq2seq.gather_tree(
            outputs.predicted_ids, outputs.parent_ids,
            sequence_length=sequence_lengths, name='predicted_ids')
        total_loss = tf.reduce_sum(outputs.loss, axis=0, name='violation_loss')

        predicted_time = tf.shape(predicted_ids)[0]
        last_score = predicted_time-1
        with tf.name_scope('gold_score'):
            gold_score = outputs.gold_score[last_score]
        with tf.name_scope('sequence_scores'):
            sequence_scores = outputs.scores[last_score]
        
        return FinalBeamSearchOptimizationDecoderOutput(beam_search_decoder_output=outputs,
                                                        predicted_ids=predicted_ids,
                                                        scores=sequence_scores,
                                                        gold_score=gold_score,
                                                        gold_beam_id=final_state.gold_beam_id,
                                                        num_available_beams=final_state.num_available_beams,
                                                        total_violation_loss=total_loss), final_state

    def _beam_where(self, cond, x, y):
        assert x.shape.is_compatible_with(y.shape)
        original_static_shape = x.shape
        cond = tf.reshape(cond, [self.batch_size * self._beam_width])
        x = self._merge_batch_beams(x, original_static_shape[2:])
        y = self._merge_batch_beams(y, original_static_shape[2:])
        return self._split_batch_beams(tf.where(cond, x, y), original_static_shape[2:])

    def _beam_search_step(self, time, logits, next_cell_state, beam_state : BeamSearchOptimizationDecoderState):
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
          beam_width: Python int.  The size of the beams.
        Returns:
          A new beam state.
        """
        
        # FIXME is not quite right because we'll pick top-k again
        #previously_finished = beam_state.finished
        #logits = self._beam_where(previously_finished, beam_state.previous_logits, logits)
        
        # if we want to apply grammar constraints, this is the place to do it
        #logits = tf.identity(logits)
        print('logits', logits)

        # Calculate the scores for each beam
        #
        # Following Wiseman and Rush, we use the unnormalized logits of the current token
        # as the scores, without softmax
        # and WITHOUT SUMMING PREVIOUS TIME STEPS (this is different from other implementations
        # of beam search out there)
        #
        # scores therefore is [batch_size, beam_width, output_size] (aka [batch, beam, transitions])
        # after top_k, we get back [batch, beam], as it picks the highest transitions overall
        scores = tf.identity(logits, name='beam_scores')
        
        vocab_size = self._output_size
        time = tf.convert_to_tensor(time, name="time")
        
        # Optimization compared to Tensorflow code: if vocab_size >= beam_width, by time == 1
        # we already have all the beams, and next beam size will be full
        # If it's not, at time == 0 the next beam size is the number of available beams (== vocab_size)
        # at time == 1, it's vocab_size^2 and so on
        #
        # With this optimization, for each batch element we either have one beam (after a reset)
        # or full beams, and we don't need to compute masks dynamically
        assert vocab_size >= self._beam_width
        
        # Make sure to consider only beams that are available
        def if_time_gt_0():
            assert0 = tf.Assert(tf.reduce_all(beam_state.num_available_beams >= 1), (time, beam_state.num_available_beams, self._first_beam_mask, self._full_beam_mask))
            with tf.control_dependencies([assert0]):
                valid_beam_mask = tf.where(tf.equal(beam_state.num_available_beams, 1), self._first_beam_mask, self._full_beam_mask, name='valid_beam_mask')
            scores_flat = tf.where(valid_beam_mask, scores, self._minus_inifinity_scores)
            scores_flat = tf.reshape(scores_flat, [self.batch_size, -1])
            return scores_flat
        
        with tf.name_scope('beam_scores_flat'):
            scores_flat = tf.cond(time > 0, if_time_gt_0, lambda: scores[:, 0])
        
        # Pick the next beams according to the specified successors function
        next_beam_size = self._beam_width
        num_available_beams = self._full_num_available_beams

        next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=next_beam_size)
        print('next_beam_scores', next_beam_scores)
        next_beam_scores.set_shape([None, self._beam_width])
        word_indices.set_shape([None, self._beam_width])
        
        # Pick out the beam_ids, and states according to the chosen predictions
        next_word_ids = tf.to_int32(word_indices % vocab_size)
        next_beam_ids = tf.to_int32(word_indices // vocab_size)
        # stop back propagation across the word decision here, or tensorflow will crap itself as usual
        next_word_ids = tf.stop_gradient(next_word_ids)
        next_beam_ids = tf.stop_gradient(next_beam_ids)
        
        # Pick out the cell_states according to the next_beam_ids. We use a
        # different gather_shape here because the cell_state tensors, i.e.
        # the tensors that would be gathered from, all have dimension
        # greater than two and we need to preserve those dimensions.
        full_cell_state = next_cell_state
        print('full_cell_state', full_cell_state)
        next_cell_state = nest.map_structure(
            lambda gather_from: _maybe_tensor_gather_helper(
                gather_indices=next_beam_ids,
                gather_from=gather_from,
                batch_size=self.batch_size,
                range_size=self._beam_width,
                gather_shape=[self.batch_size * self._beam_width, -1]),
            next_cell_state)
        
        # At training time, check for margin violations, and if so reset the beam
        if self._training:
            gold_finished = time >= self._gold_sequence_length
            with tf.name_scope('gold_token'):
                gold_token = tf.cond(time >= self._max_time,
                        lambda: tf.fill((self.batch_size,), self._end_token),
                        lambda: self._gold_sequence.read(time))

            assert1 = tf.Assert(tf.reduce_all(gold_token >= 0), (time, self._original_gold_sequence, gold_token))
            
            with tf.control_dependencies([assert1]):
                current_gold_beam_indices = tf.stack((self._batch_size_range, beam_state.gold_beam_id), axis=1, name='current_gold_beam_indices')
                gold_score_indices = tf.stack((self._batch_size_range, beam_state.gold_beam_id, gold_token), axis=1, name='gold_score_indices')
            gold_score = tf.where(gold_finished,
                                  tf.gather_nd(beam_state.previous_score, current_gold_beam_indices),
                                  tf.gather_nd(logits, gold_score_indices), name='gold_score')
            
            with tf.name_scope('next_gold_cell_state'):
                def maybe_gather_nd(state):
                    if len(state.shape) > 1:
                        return tf.gather_nd(state, current_gold_beam_indices)
                    else:
                        # time, and other scalar things that advance equally across all batches beams and stuff
                        # used by AttentionWrapper mainly
                        return state
                next_gold_cell_state = nest.map_structure(maybe_gather_nd, next_cell_state)
            
            # compute the highest beam that contains the gold token
            is_gold_beam = tf.logical_and(tf.equal(next_beam_ids, tf.expand_dims(beam_state.gold_beam_id, axis=1)),
                                          tf.equal(next_word_ids, tf.expand_dims(gold_token, axis=1)), name='is_gold_beam')
            # Note: if there is exactly one gold beam, argmax will find it and set next_gold_beam_id correctly
            # if there is no gold beam, argmax will pick some unspecified beam, but then we will definitely have a beam
            # violation (because gold_score < bram_bottom_scores necessarily, so margin >= 1 > 0) and we'll reset the beam
            # there cannot be multiple gold beams, because the gold is a single sequence of actions (tokens)
            next_gold_beam_id = tf.argmax(tf.to_int32(is_gold_beam), axis=1, output_type=tf.int32, name='next_gold_beam_id')
            print('next_gold_beam_id', next_gold_beam_id)
            
            # the score of the last element of the beam
            beam_bottom_indices = tf.stack((tf.range(self.batch_size), tf.fill((self.batch_size,), self._beam_width-1)), axis=1, name='beam_bottom_indices')
            print('beam_bottom_indices', beam_bottom_indices)
            beam_bottom_scores = tf.gather_nd(next_beam_scores, beam_bottom_indices, name='beam_bottom_scores')
            print('beam_bottom_scores', beam_bottom_scores)
            
            margin = tf.add(1 - gold_score, beam_bottom_scores, name='beam_bottom_margin')
            print('margin', margin)
            beam_violation = tf.greater(margin, 0, name='beam_violation')
            
            # The two formulations are identical, but the second one uses broadcasting and avoids creating the zero tensor 
            #loss = tf.where(beam_violation, margin, tf.zeros((self.batch_size,)), name='beam_violation_loss')
            loss = tf.maximum(margin, 0, name='beam_violation_loss')
            print('loss', loss)
            
            reset_token = gold_token
            with tf.name_scope('beam_reset'):
                next_word_ids = tf.where(beam_violation, self._maybe_tile_batch(reset_token), next_word_ids, name='next_word_ids')
                next_beam_ids = tf.where(beam_violation, self._maybe_tile_batch(beam_state.gold_beam_id), next_beam_ids, name='next_beam_ids')
                next_beam_scores = tf.where(beam_violation, self._maybe_tile_batch(gold_score), next_beam_scores, name='next_beam_scores')
                num_available_beams = tf.where(beam_violation, self._initial_num_available_beams, num_available_beams, name='num_available_beams')
                next_gold_beam_id = tf.where(beam_violation, tf.zeros((self.batch_size,), dtype=tf.int32), next_gold_beam_id, name='next_gold_beam_id')
                
                tiled_gold_state = nest.map_structure(self._maybe_tile_batch, next_gold_cell_state)
    
                # note the shape trickery: if .shape.ndims is 0 (a scalar) then we reset to gold for all examples
                # in the batch
                # this is technically incorrect, but it's ok because scalar shape is used for things like "time"
                # in AttentionWrapper, which should advance the same way for the gold and the real cell
                with tf.name_scope('next_cell_state'):
                    next_cell_state = nest.map_structure(lambda gold, predicted: tf.where(beam_violation, gold, predicted) if gold.shape.ndims >= 1 else gold,
                                                         tiled_gold_state,
                                                         next_cell_state)

        else:
            gold_token = self._start_tokens
            gold_score = tf.zeros((self.batch_size,), dtype=tf.float32)
            next_gold_beam_id = tf.zeros((self.batch_size,), dtype=tf.int32)
            loss = tf.zeros((self.batch_size,), dtype=tf.float32)

        previously_finished = _tensor_gather_helper(
            gather_indices=next_beam_ids,
            gather_from=beam_state.finished,
            batch_size=self.batch_size,
            range_size=self._beam_width,
            gather_shape=[-1])
        next_finished = tf.logical_or(previously_finished, tf.equal(next_word_ids, self._end_token))

        next_state = BeamSearchOptimizationDecoderState(
            cell_state=next_cell_state,
            # note: nothing in the code cares about previous score
            previous_score=next_beam_scores,
            previous_logits=logits,
            num_available_beams=num_available_beams,
            gold_beam_id=next_gold_beam_id,
            finished=next_finished)
        
        output = BeamSearchOptimizationDecoderOutput(
            scores=next_beam_scores,
            gold_score=gold_score,
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

    def add_decoder_op(self, enc_final_state, enc_hidden_states, training):
        cell_dec = common.make_multi_rnn_cell(self.config.rnn_layers, self.config.rnn_cell_type,
                                              self.config.output_embed_size,
                                              self.config.decoder_hidden_size,
                                              self.dropout_placeholder)
        enc_hidden_states, enc_final_state = common.unify_encoder_decoder(cell_dec,
                                                                          enc_hidden_states,
                                                                          enc_final_state)

        beam_width = self.config.training_beam_size if training else self.config.beam_size

        #cell_dec = ParentFeedingCellWrapper(cell_dec, tf.contrib.seq2seq.tile_batch(enc_final_state, beam_width))
        if self.config.apply_attention:
            tiled_enc_hidden_states = tf.contrib.seq2seq.tile_batch(enc_hidden_states, beam_width)
            tiled_input_length = tf.contrib.seq2seq.tile_batch(self.input_length_placeholder, beam_width)
            tiled_enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, beam_width)
            
            cell_dec, enc_final_state = common.apply_attention(cell_dec,
                                                               tiled_enc_hidden_states,
                                                               tiled_enc_final_state,
                                                               tiled_input_length,
                                                               self.batch_size * beam_width,
                                                               self.config.attention_probability_fn,
                                                               alignment_history=False)
        else:
            enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, beam_width)
        
        print('enc_final_state', enc_final_state)
        
        output_layer = tf.layers.Dense(self.config.grammar.output_size, use_bias=False)
        go_vector = tf.ones((self.batch_size,), dtype=tf.int32) * self.config.grammar.start
        decoder = BeamSearchOptimizationDecoder(training, cell_dec, self.output_embed_matrix, go_vector, self.config.grammar.end,
                                                enc_final_state,
                                                beam_width=beam_width,
                                                output_layer=output_layer,
                                                gold_sequence=self.output_placeholder if training else None,
                                                gold_sequence_length=(self.output_length_placeholder+1) if training else None)

        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                output_time_major=True,
                                                                maximum_iterations=self.config.max_length)
        return final_outputs
        
    def finalize_predictions(self, preds : FinalBeamSearchOptimizationDecoderOutput):
        # predicted_ids is [max_time, batch_size, beam_width] because that's how gather_tree produces it
        # transpose it to be [batch_size, beam_width, max_time] which is what we expect
        return tf.transpose(preds.predicted_ids, [1, 2, 0])
    
    def add_loss_op(self, preds : FinalBeamSearchOptimizationDecoderOutput):
        # For beam search, the loss is computed as we go along, so we just average it along 
        # the beam here
        print('violation_loss', preds.total_violation_loss)
        
        # Scale violation loss by 10 to offset the non-convexity that arises when
        # fewer violations means we compare the correctness more often because we don't
        # reset the beam as much
        violation_loss = 10 * tf.reduce_mean(preds.total_violation_loss, axis=0)

        # the loss so far is the cost of falling off the beam
        # now we add a second term that checks that the highest scored prediction is
        # correct
        
        # To simplify the problem (which reduces training time), we assume that the gold is correct, and nothing else
        # is correct
        #
        # We know what position in the beam the gold is, and we know it is in the beam because
        # we would reset the beam otherwise
        # If the gold is at position 0, the prediction is correct; the margin is between the gold and the second prediction
        # If the gold is not at position 0, the prediction is inccorrect; the margin is between the gold and the first prediction
        #
        # The trickery here is that if the beam was reset at the last step, gold_beam_id will be 0 but the second prediction
        # will also be the same as gold (num_available_beams == 1)
        # In that case we don't want to have loss (cause we would have margin of a prediction against itself and the gradients
        # would be very confused) so we compute a margin of 0 (hence a loss of 0); this is ok because we already
        # paid the cost of resetting the beam; as the cost of resetting the beam goes down the correctness loss
        # will go up (because now we start comparing against predictions we have) and then down again.
        # This makes the whole thing non-convex and tricky, unfortunately. Sad!
        # 
        with tf.name_scope('correctness_loss'):
            is_incorrect = tf.not_equal(preds.gold_beam_id, 0)
            beam_indices = tf.range(self.batch_size, name='beam_indices')
            indices_0 = tf.stack((beam_indices, tf.zeros((self.batch_size,), dtype=tf.int32)), axis=1, name='indices_beam_0')
            indices_1 = tf.stack((beam_indices, tf.ones((self.batch_size,), dtype=tf.int32)), axis=1, name='indices_beam_1')
            beam_0_score = tf.gather_nd(preds.scores, indices_0, name='beam_0_score')
            beam_1_score = tf.gather_nd(preds.scores, indices_1, name='beam_1_score')
            
            if_correct_margin = tf.where(tf.greater(preds.num_available_beams, 1), 1 - preds.gold_score + beam_1_score, tf.zeros((self.batch_size,)), name='if_correct_margin')
            # if the gold is at position 1, then necessarily there are at least 2 beams (ie, the beam was not reset at the last step)
            # so we're good
            margin = tf.where(is_incorrect, 1 - preds.gold_score + beam_0_score, if_correct_margin, name='margin')
            correctness_loss = tf.maximum(margin, 0, name='loss')
        
        print('correctness_loss', correctness_loss)
        correctness_loss = tf.reduce_mean(correctness_loss, axis=0)
        return violation_loss + correctness_loss
