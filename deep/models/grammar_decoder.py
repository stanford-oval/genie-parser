'''
Created on Jul 20, 2017

@author: gcampagn
'''

import tensorflow as tf

# FIXME: this code is completely broken, it needs to be rewritten to
# use the new Tensorflow API 
def grammar_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                 maximum_length, grammar,
                                 dtype=tf.int32, name=None, first_output_state=None):
    """ A version of tf.contrib.seq2seq.simple_decoder_fn_inference
        that applies grammar constraints to the output """
    with tf.name_scope(name, "grammar_decoder_fn_inference",
                       [output_fn, encoder_state, embeddings,
                        maximum_length, dtype]):
        end_of_sequence_id = tf.convert_to_tensor(grammar.end, dtype)
        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = tf.convert_to_tensor(grammar.output_size, dtype)
        encoder_info = encoder_state
        while isinstance(encoder_info, tuple):
            encoder_info = encoder_info[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]
        if first_output_state is None:
            first_output_state = tf.zeros((1,))

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with tf.name_scope(name, "grammar_decoder_fn_inference",
                           [time, cell_state, cell_input, cell_output,
                            context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)

            if cell_output is None:
                # invariant that this is time == 0
                cell_state = encoder_state
                cell_output = tf.zeros((num_decoder_symbols,),
                                        dtype=tf.float32)
                grammar_state = None
                next_output_state = first_output_state
            else:
                grammar_state, output_state = context_state
                cell_output, next_output_state = output_fn(time, cell_output, cell_state, batch_size, output_state)

            next_input_id, next_grammar_state = grammar.constrain(cell_output, grammar_state, batch_size, dtype=dtype)
            next_input = tf.gather(embeddings, next_input_id)
            done = tf.equal(next_input_id, end_of_sequence_id)

            # if time > maxlen, return all true vector
            done = tf.cond(tf.greater(time, maximum_length),
                           lambda: tf.ones((batch_size,), dtype=tf.bool),
                           lambda: done)
            return (done, cell_state, next_input, cell_output, (next_grammar_state, next_output_state))
    return decoder_fn