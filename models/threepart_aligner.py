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
'''
Created on Aug 2, 2017

@author: gcampagn
'''

import tensorflow as tf

from .base_aligner import BaseAligner
from .seq2seq_helpers import Seq2SeqDecoder
from .grammar_decoder import GrammarHelper

from grammar.thingtalk import ThingtalkGrammar, MAX_SPECIAL_LENGTH, MAX_PRIMITIVE_LENGTH

from collections import namedtuple
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper
from tensorflow.python.util import nest

def pad_up_to(vector, size, rank):
    length_diff = tf.reshape(size - tf.shape(vector)[1], shape=(1,))
    with tf.control_dependencies([tf.assert_non_negative(length_diff, data=(vector, size, tf.shape(vector)))]):
        padding = tf.reshape(tf.concat([[0, 0, 0], length_diff, [0,0]*(rank-1)], axis=0), shape=((rank+1), 2))
        return tf.pad(vector, padding, mode='constant')

ThreePartAlignerResult=namedtuple('ThreePartAlignerResult', ('top_logits', 'part_logit_preds', 'part_logit_sequence_preds', 'logit_special_sequence', 'sequence'))

class PrimitiveSequenceGrammarHelper(GrammarHelper):
    def __init__(self, grammar : ThingtalkGrammar, adjusted_function_token):
        super().__init__(grammar)
        self._adjusted_function_token = adjusted_function_token
    
    def get_init_state(self, _batch_size):
        return self.grammar.get_function_init_state(self._adjusted_function_token)
    
    def constrain_logits(self, logits, curr_state):
        return self.grammar.constrain_value_logits(logits, curr_state)
    
    def transition(self, curr_state, next_symbols, batch_size):
        return self.grammar.value_transition(curr_state, next_symbols, batch_size)
    
class SpecialSequenceGrammarHelper(GrammarHelper):
    def get_init_state(self, batch_size):
        return tf.ones((batch_size,), dtype=tf.int32) * self.grammar.bookeeping_state_id
    
    def constrain_logits(self, logits, curr_state):
        return self.grammar.constrain_value_logits(logits, curr_state)
    
    def transition(self, curr_state, next_symbols, batch_size):
        return self.grammar.value_transition(curr_state, next_symbols, batch_size)

class ThreePartAligner(BaseAligner):
    '''
    An aligner that uses three separate decoders for when/get/do
    '''
    
    def __init__(self, config):
        super().__init__(config)
        if not isinstance(self.config.grammar, ThingtalkGrammar):
            raise TypeError("ThreepartAligner can only be used with the ThingtalkGrammar")
    
    def add_output_placeholders(self):
        self.top_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.special_label_placeholder = tf.placeholder(tf.int32, shape=(None, MAX_SPECIAL_LENGTH))
        self.part_function_placeholders = dict()
        self.part_sequence_placeholders = dict()
        self.part_sequence_length_placeholders = dict()
        for part in ('trigger', 'query', 'action'):
            self.part_function_placeholders[part] = tf.placeholder(tf.int32, shape=(None,))
            self.part_sequence_placeholders[part] = tf.placeholder(tf.int32, shape=(None, MAX_PRIMITIVE_LENGTH))
            self.part_sequence_length_placeholders[part] = tf.placeholder(tf.int32, shape=(None,))
    
    def create_feed_dict(self, inputs_batch, input_length_batch, parses_batch, labels_batch=None, label_length_batch=None, **kw):
        feed_dict = BaseAligner.create_feed_dict(self, inputs_batch, input_length_batch, parses_batch, labels_batch=None, label_length_batch=None, **kw)
        if labels_batch is None or label_length_batch is None:
            return feed_dict

        top_batch, special_label_batch, part_function_batches, part_sequence_batches, part_sequence_length_batches = self.config.grammar.split_batch_in_parts(labels_batch)
        feed_dict[self.top_placeholder] = top_batch
        feed_dict[self.special_label_placeholder] = special_label_batch
        for part in ('trigger', 'query', 'action'):
            feed_dict[self.part_function_placeholders[part]] = part_function_batches[part]
            feed_dict[self.part_sequence_placeholders[part]] = part_sequence_batches[part]
            feed_dict[self.part_sequence_length_placeholders[part]] = part_sequence_length_batches[part]
        return feed_dict

    def add_decoder_op(self, enc_final_state, enc_hidden_states, output_embed_matrix, training):
        original_enc_final_state = enc_final_state
        flat_enc_final_state = nest.flatten(enc_final_state)
        enc_final_state = tf.concat(flat_enc_final_state, axis=1)
        enc_final_size = int(enc_final_state.get_shape()[1])

        part_logit_preds = dict()
        part_token_preds = dict()
        part_logit_sequence_preds = dict()
        part_token_sequence_preds = dict()
        part_layers = []
        grammar = self.config.grammar
        for i, part in enumerate(('trigger', 'query', 'action')):
            with tf.variable_scope('decode_function_' + part):
                activation = getattr(tf.nn, self.config.function_nonlinearity) if hasattr(tf.nn, self.config.function_nonlinearity) else getattr(tf, self.config.function_nonlinearity)
                layer = tf.contrib.layers.fully_connected(enc_final_state, self.config.function_hidden_size, activation_fn=activation)
                part_layers.append(layer)
                layer_with_dropout = tf.nn.dropout(layer, keep_prob=self.dropout_placeholder, seed=443 * i)
                part_logit_preds[part] = tf.layers.dense(layer_with_dropout, len(grammar.functions[part]))
                part_token_preds[part] = tf.cast(tf.argmax(part_logit_preds[part], axis=1), dtype=tf.int32)
        
        first_value_token = grammar.num_functions + grammar.num_begin_tokens + grammar.num_control_tokens
        num_value_tokens = grammar.output_size - first_value_token
        output_embed_matrix = tf.concat((output_embed_matrix[0:grammar.num_control_tokens], output_embed_matrix[first_value_token:]), axis=0)
        
        adjusted_trigger = part_token_preds['trigger'] + (grammar.num_control_tokens + grammar.num_begin_tokens)
        adjusted_query = part_token_preds['query'] + (grammar.num_control_tokens + grammar.num_begin_tokens + len(grammar.functions['trigger']))
        adjusted_action = part_token_preds['action'] + (grammar.num_control_tokens + grammar.num_begin_tokens + len(grammar.functions['trigger']) + len(grammar.functions['query']))
        
        layer_concat = tf.concat(part_layers, axis=1)
        for i, part in enumerate(('trigger', 'query', 'action')):
            with tf.variable_scope('decode_sequence_' + part):
                def one_decoder_input(i, like):
                    with tf.variable_scope(str(i)):
                        return tf.layers.dense(layer_concat, like.get_shape()[1])
                flat_decoder_initial_state = [one_decoder_input(i, like) for i, like in enumerate(flat_enc_final_state)]
                decoder_initial_state = nest.pack_sequence_as(original_enc_final_state, flat_decoder_initial_state)
                cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i, True) for i in range(self.config.rnn_layers)])
                
                # uncompress function tokens (to look them up in the grammar)
                if training:
                    adjusted_function_token = self.part_function_placeholders[part]
                else:
                    if part == 'trigger':   
                        adjusted_function_token = adjusted_trigger
                    elif part == 'query':
                        adjusted_function_token = adjusted_query
                    elif part == 'action':
                        adjusted_function_token = adjusted_action
                
                # adjust the sequence to "skip" function tokens
                output_size = grammar.num_control_tokens + num_value_tokens
                output = self.part_sequence_placeholders[part]
                adjusted_output = tf.where(output >= grammar.num_control_tokens, output - (first_value_token - grammar.num_control_tokens), output)
                
                if self.config.apply_attention:
                    attention = LuongAttention(self.config.decoder_hidden_size, enc_hidden_states, self.input_length_placeholder,
                                               probability_fn=tf.nn.softmax)
                    cell_dec = AttentionWrapper(cell_dec, attention,
                                                cell_input_fn=lambda inputs, _: inputs,
                                                attention_layer_size=self.config.decoder_hidden_size,
                                                initial_cell_state=decoder_initial_state)
                    decoder_initial_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
                decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                         adjusted_output, self.part_sequence_length_placeholders[part], self.batch_number_placeholder,
                                         max_length=MAX_PRIMITIVE_LENGTH)
                rnn_output, sample_ids = decoder.decode(cell_dec, decoder_initial_state, output_size, output_embed_matrix,
                                                        training, grammar_helper=PrimitiveSequenceGrammarHelper(grammar, adjusted_function_token))
                part_logit_sequence_preds[part] = rnn_output
                part_token_sequence_preds[part] = tf.cast(sample_ids, dtype=tf.int32)
   
        with tf.variable_scope('top_classifier'):
            top_hidden = tf.contrib.layers.fully_connected(enc_final_state, self.config.first_token_hidden_size, activation_fn=tf.tanh)
            top_hidden_with_dropout = tf.nn.dropout(top_hidden, keep_prob=self.dropout_placeholder, seed=127)
            top_logits = tf.layers.dense(top_hidden_with_dropout, grammar.num_begin_tokens)
            top_token = tf.cast(tf.argmax(top_logits, axis=1), dtype=tf.int32)
   
        with tf.variable_scope('decode_special'):
            output_size = grammar.num_control_tokens + num_value_tokens
            output = self.special_label_placeholder
            adjusted_output = tf.where(output >= grammar.num_control_tokens, output - (first_value_token - grammar.num_control_tokens), output)
            cell_dec = tf.contrib.rnn.MultiRNNCell([self.make_rnn_cell(i, True) for i in range(self.config.rnn_layers)])
            
            sequence_length = tf.ones((self.batch_size,), dtype=tf.int32) * MAX_SPECIAL_LENGTH
            decoder_initial_state = original_enc_final_state
            if self.config.apply_attention:
                attention = LuongAttention(self.config.decoder_hidden_size, enc_hidden_states, self.input_length_placeholder,
                                               probability_fn=tf.nn.softmax)
                cell_dec = AttentionWrapper(cell_dec, attention,
                                            cell_input_fn=lambda inputs, _: inputs,
                                            attention_layer_size=self.config.decoder_hidden_size,
                                            initial_cell_state=original_enc_final_state)
                decoder_initial_state = cell_dec.zero_state(self.batch_size, dtype=tf.float32)
            decoder = Seq2SeqDecoder(self.config, self.input_placeholder, self.input_length_placeholder,
                                     adjusted_output, sequence_length, self.batch_number_placeholder, max_length=MAX_SPECIAL_LENGTH)
            rnn_output, sample_ids = decoder.decode(cell_dec, decoder_initial_state, output_size, output_embed_matrix, training,
                                                    grammar_helper=SpecialSequenceGrammarHelper(grammar))
            logit_special_sequence = rnn_output
            token_special_sequence = tf.cast(sample_ids, dtype=tf.int32)
   
        # adjust tokens back to their output code
        adjusted_top = tf.expand_dims(top_token + grammar.num_control_tokens, axis=1)
        
        adjusted_special_sequence = tf.where(token_special_sequence >= grammar.num_control_tokens, token_special_sequence + (first_value_token - grammar.num_control_tokens), token_special_sequence)
        
        adjusted_token_sequences = dict()
        for part in ('trigger', 'query', 'action'):
            token_sequence = part_token_sequence_preds[part]
            adjusted_token_sequence = tf.where(token_sequence >= grammar.num_control_tokens, token_sequence + (first_value_token - grammar.num_control_tokens), token_sequence)
            adjusted_token_sequences[part] = adjusted_token_sequence
        # remove EOS from the middle of the sentence
        adjusted_token_sequences['trigger'] = tf.where(tf.equal(adjusted_token_sequences['trigger'], grammar.end), tf.zeros_like(adjusted_token_sequences['trigger']), adjusted_token_sequences['trigger'])
        adjusted_token_sequences['query'] = tf.where(tf.equal(adjusted_token_sequences['query'], grammar.end), tf.zeros_like(adjusted_token_sequences['query']), adjusted_token_sequences['query'])

        adjusted_trigger = tf.expand_dims(adjusted_trigger, axis=1)
        adjusted_query = tf.expand_dims(adjusted_query, axis=1)
        adjusted_action = tf.expand_dims(adjusted_action, axis=1)
        
        program_sequence = tf.concat((adjusted_top, adjusted_trigger, adjusted_token_sequences['trigger'], adjusted_query, adjusted_token_sequences['query'],
                                      adjusted_action, adjusted_token_sequences['action']), axis=1)
        full_special_sequence = tf.concat((adjusted_top, adjusted_special_sequence), axis=1)
        # full special sequence is smaller than program sequence, so we need to pad it all the way to the same shape
        full_special_sequence = pad_up_to(full_special_sequence, tf.shape(program_sequence)[1], rank=1)
        
        rule_token = grammar.dictionary['rule'] - grammar.num_control_tokens
        full_sequence = tf.where(tf.equal(top_token, rule_token), program_sequence, full_special_sequence)
            
        return ThreePartAlignerResult(top_logits, part_logit_preds, part_logit_sequence_preds, logit_special_sequence, full_sequence)
    
    def finalize_predictions(self, preds):
        # add a dimension of 1 between the batch size and the sequence length to emulate a beam width of 1 
        return tf.expand_dims(preds.sequence, axis=1)
    
    def add_loss_op(self, preds, _training=True):
        grammar = self.config.grammar
        first_value_token = grammar.num_functions + grammar.num_begin_tokens + grammar.num_control_tokens

        top_logits, part_logit_preds, part_logit_sequence_preds, logit_special_sequence, _ = preds
        gold_top = self.top_placeholder - grammar.num_control_tokens 
        top_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_top, logits=top_logits)
        
        program_loss = 0
        for part in ('trigger', 'query', 'action'):
            if part == 'trigger':
                function_offset = grammar.num_begin_tokens + grammar.num_control_tokens
            elif part == 'query':
                function_offset = grammar.num_begin_tokens + grammar.num_control_tokens + len(grammar.functions['trigger'])
            elif part == 'action':
                function_offset = grammar.num_begin_tokens + grammar.num_control_tokens + len(grammar.functions['trigger']) + len(grammar.functions['query'])
            
            with tf.control_dependencies([tf.Assert(tf.reduce_all(self.part_function_placeholders[part] >= function_offset),
                                         (part, self.part_function_placeholders[part], function_offset))]):
                gold_function = self.part_function_placeholders[part] - function_offset
                function_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_function, logits=part_logit_preds[part])
            
            padded_logits = pad_up_to(part_logit_sequence_preds[part], MAX_PRIMITIVE_LENGTH, rank=2)
            mask = tf.sequence_mask(self.part_sequence_length_placeholders[part], MAX_PRIMITIVE_LENGTH, dtype=tf.float32)
            
            gold_sequence = self.part_sequence_placeholders[part]
            gold_sequence = tf.where(gold_sequence >= grammar.num_control_tokens, gold_sequence - (first_value_token - grammar.num_control_tokens), gold_sequence)
            
            function_sequence_loss = tf.contrib.seq2seq.sequence_loss(targets=gold_sequence, logits=padded_logits, weights=mask,
                                                                      average_across_batch=False)
            program_loss += function_loss + function_sequence_loss
        
        padded_logits = pad_up_to(logit_special_sequence, MAX_SPECIAL_LENGTH, rank=2)
        gold_special_sequence = self.special_label_placeholder
        gold_special_sequence = tf.where(gold_special_sequence >= grammar.num_control_tokens, gold_special_sequence - (first_value_token - grammar.num_control_tokens), gold_special_sequence)
        special_loss = tf.contrib.seq2seq.sequence_loss(targets=gold_special_sequence, logits=padded_logits, weights=tf.ones((self.batch_size, MAX_SPECIAL_LENGTH)),
                                                        average_across_batch=False)
        
        rule_token = self.config.grammar.dictionary['rule']
        element_loss = top_loss + tf.where(tf.equal(self.top_placeholder, rule_token), program_loss, special_loss)
        
        batch_loss = tf.reduce_mean(element_loss, axis=0)
        return batch_loss
        



