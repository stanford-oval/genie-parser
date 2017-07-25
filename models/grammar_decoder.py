'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

from tensorflow.contrib.seq2seq import BasicDecoder, BasicDecoderOutput

class GrammarBasicDecoder(BasicDecoder):
    def __init__(self, grammar, *args, training_output=None, **kw):
        super().__init__(*args, **kw)
        self._grammar = grammar
        self._fixed_outputs = training_output
        if training_output is not None:
            self._fixed_outputs = tf.TensorArray(dtype=tf.int32, size=training_output.get_shape()[1])
            self._fixed_outputs = self._fixed_outputs.unstack(tf.transpose(training_output, [1, 0]))
        
    def initialize(self, name=None):
        # wrap the state to add the grammar state
        finished, first_inputs, initial_state = BasicDecoder.initialize(self, name=name)
        return finished, first_inputs, (initial_state, self._grammar.get_init_state(self.batch_size))
        
    def step(self, time, inputs, state, name=None):
        with tf.name_scope(name, "GrammarDecodingStep", (time, inputs, state)):
            decoder_state, grammar_state = state
            cell_outputs, cell_state = self._cell(inputs, decoder_state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            grammar_cell_outputs = self._grammar.constrain_logits(cell_outputs, grammar_state)
            cell_outputs = grammar_cell_outputs
            sample_ids = self._helper.sample(time=time, outputs=grammar_cell_outputs, state=cell_state)
            (finished, next_inputs, next_decoder_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
            if self._fixed_outputs is not None:
                next_grammar_state = self._grammar.transition(grammar_state, self._fixed_outputs.read(time), self.batch_size)
            else:
                next_grammar_state = self._grammar.transition(grammar_state, sample_ids, self.batch_size)
            next_state = (next_decoder_state, next_grammar_state)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
