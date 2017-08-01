'''
Created on Jul 27, 2017

@author: gcampagn
'''

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from .base_encoder import BaseEncoder
from collections import namedtuple


class TreeDropoutWrapper(object):
    '''
    A dropout wrapper for TreeRNN cells
    '''
    def __init__(self, cell, state_keep_prob, output_keep_prob, seed):
        self._cell = cell
        self._state_keep_prob = state_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    def zero_state(self, batch_size, dtype=tf.float32):
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, left_state, right_state, extra_input=None):
        outputs, new_state = self._cell(left_state, right_state, extra_input=extra_input)
        return tf.nn.dropout(outputs, keep_prob=self._output_keep_prob, seed=self._seed * 7 + 1), \
            LSTMStateTuple(tf.nn.dropout(new_state.c, keep_prob=self._state_keep_prob, seed=self._seed * 7 + 1),
                           tf.nn.dropout(new_state.h, keep_prob=self._state_keep_prob, seed=self._seed * 7 + 1))


class TreeLSTM(object):
    '''
    As described in Tai et al. 2015
    
    Basically a traditional LSTM, except that it takes two states
    and two memories (h1, c1) and (h2, c2), then composes them
    in the natural way for an LSTM
    '''
    
    def __init__(self, num_cells):
        self._num_cells = num_cells
        
    def zero_state(self, batch_size, dtype=tf.float32):
        zeros = tf.zeros((batch_size, self._num_cells), dtype=dtype)
        return LSTMStateTuple(zeros, zeros)
        
    def __call__(self, left_state, right_state, extra_input=None):
        with tf.variable_scope('TreeLSTM'):
            c1, h1 = left_state
            c2, h2 = right_state

            if extra_input is not None:
                input_concat = tf.concat((extra_input, h1, h2), axis=1)
            else:
                input_concat = tf.concat((h1, h2), axis=1)
            concat = tf.layers.dense(input_concat, 5 * self._num_cells)
            i, f1, f2, o, g = tf.split(concat, 5, axis=1)
            i = tf.sigmoid(i)
            f1 = tf.sigmoid(f1)
            f2 = tf.sigmoid(f2)
            o = tf.sigmoid(o)
            g = tf.tanh(g)

            cnew = f1 * c1 + f2 * c2 + i * g
            hnew = o * cnew

            newstate = LSTMStateTuple(c=cnew, h=hnew)
            return hnew, newstate


class TreeEncoder(BaseEncoder):
    '''
    An encoder that uses constituency parsing and TreeLSTM
    to build a syntactic tree of the input sentence.
    
    This is based on the SPINN (Bowman, Gauthier et al. 2016)
    model for sentence understanding.
    
    It also implements the Thin Stack optimization, using
    tensorflow's TensorArrays and back pointers.
    '''
    
    def __init__(self, cell_type, num_layers, max_time, state_dropout, *args, train_syntactic_parser=False, use_tracking_rnn=True, **kw):
        super().__init__(*args, **kw)
        self._num_layers = num_layers
        self._max_time = max_time
        self._train_syntactic_parser = train_syntactic_parser
        self._use_tracking_rnn = use_tracking_rnn
        self._state_dropout = state_dropout
        if self._num_layers > 1:
            raise NotImplementedError("multi-layer TreeRNN is not implemented yet (and i'm not sure how it'd work)")
        self._cell_type = cell_type

    def _make_rnn_cell(self, i):
        if self._cell_type == "lstm":
            cell = tf.contrib.rnn.LSTMCell(self.output_size)
        elif self._cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell(self.output_size)
        elif self._cell_type == "basic-tanh":
            cell = tf.contrib.rnn.BasicRNNCell(self.output_size)
        else:
            raise ValueError("Invalid RNN Cell type")
        cell = tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=self._state_dropout, output_keep_prob=self._output_dropout, seed=88 + 33 * i)
        return cell
    
    def _make_tree_cell(self, i):
        if self._cell_type == "lstm":
            cell = TreeLSTM(self.output_size)
        elif self._cell_type in ("gru", "basic-tanh"):
            raise NotImplementedError("GRU/basic-tanh tree cells not implemented yet")
        else:
            raise ValueError("Invalid RNN Cell type")
        cell = TreeDropoutWrapper(cell, state_keep_prob=self._state_dropout, output_keep_prob=self._output_dropout, seed=99 + 33 * i)
        return cell

    def encode(self, inputs: tf.Tensor, input_length: tf.Tensor, parses : tf.Tensor):
        with tf.variable_scope('treeenc'):
            tree_cell = self._make_tree_cell(0)
            rnn_cell = self._make_rnn_cell(0)
        
            # make the inputs time major first
            batch_size = tf.shape(inputs)[0]
            inputs = tf.transpose(inputs, [1, 0, 2])
            parses = tf.transpose(parses, [1, 0])
            
            # convert the inputs to be the buffer
            # (which is shaped like an LSTM cell)
            # with a linear transformation
            with tf.variable_scope('buffer'):
                buffer = tf.layers.dense(inputs, 2*self._output_size)
                buffer = tf.reshape(buffer, shape=(self._max_time, batch_size, 2*self._output_size))

            # create tensor arrays that we'll use for temporary and output storage
            # we walk through 2*max_time-1 shift/reduce ops, and we start by pushing
            # twice the initial TreeLSTM state on the stack, so we need 2*max_time+1 space in the
            # stack
            # (we need to have at least two elements in the stack to drive the tracking RNN)
            # FIXME support RNNs other than LSTM, using tuples instead of 2 explicit TAs
            #
            # but we don't necessarily write all of the stack, because if the whole minibatch
            # terminates early we stop the loop
            # Tensorflow complains loudly if you try to .stack() a TensorArray that was only
            # written partially, so we set dynamic_size instead
            # (this is what dynamic_rnn does)
            initial_stack_hta = tf.TensorArray(dtype=tf.float32, size=2, dynamic_size=True, clear_after_read=False, name="stack_hta")
            initial_stack_cta = tf.TensorArray(dtype=tf.float32, size=2, dynamic_size=True, clear_after_read=False, name="stack_cta")
            initial_output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, name="output_ta")

            # the back pointer TA is the tricky bit of the thin stack optimization (which
            # we need to get automatic gradients and batching from Tensorflow, otherwise
            # it just explodes)
            #
            # the goal is to construct a virtual stack, using TensorArrays, which are
            # write once, in concert with stack_hta/stack_cta
            # at any time, some elements in the TensorArray are in the virtual stack
            # and some don't - we cannot overwrite them, but we have to ignore them
            # at any time, the top of the stack (the current last element of the stack TAs)
            # is an element of the virtual stack too
            # by following the back pointers from the top of the stack, you can walk
            # all elements that are still in the virtual stack, skipping elements that
            # have been "popped"
            # the back_pointer_ta of the first element is 0, which means that if you
            # keep popping you end up in a loop - don't do that
            initial_back_pointer_ta = tf.TensorArray(dtype=tf.int32, size=2, dynamic_size=True, clear_after_read=False, name="back_pointer_ta")

            def cond(finished, time, buffer_ptr, stack_hta, stack_cta, back_pointer_ta, output_ta, rnn_state):
                return tf.logical_and(time < 2*self._max_time-1, tf.logical_not(tf.reduce_all(finished)))

            def body(finished, time, buffer_ptr, stack_hta, stack_cta, back_pointer_ta, output_ta, rnn_state):
                # time: our progression through the shift-reduce operation
                time = time
                # we push two elements to the stack before the loop, so the top of the stack
                # is element 1 when we start
                stack_top = time+1

                # buffer top : the current element at the top of the buffer (the next
                # word to read)
                # note: time is less than 2*max_length-1, so buffer_ptr is less than
                # max_length, which means this read is safe
                # it could be equal to input_length though, in which case we would read a padding
                # token; this is ok because we will never shift in that case and we ignore the
                # tracking RNN output, so we won't look at this value
                batch_indices = tf.range(0, batch_size, dtype=tf.int32)
                diag_indices = tf.stack((batch_indices, batch_indices), axis=1)
                buffer_top_indices = tf.stack((buffer_ptr, batch_indices), axis=1, name='buffer_top_indices')
                buffer_top = tf.gather_nd(buffer, buffer_top_indices, name='buffer_top')
                # stack_htop : the h vector at the top of the stack
                stack_htop = tf.check_numerics(stack_hta.read(stack_top, name='stack_htop'), 'invalid stack htop')
                # ptr_stack_prev_to_top : the pointer (index) to the second element from the top
                # in the stack
                ptr_stack_prev_to_top = back_pointer_ta.read(stack_top, name='ptr_stack_prev_to_top')
                # stack_h_prev_to_top : the h vector of the second element from the top
                # in the stack
                # NOTE: gather will return a tensor whose i-th element is the ptr_stack_prev_to_top[i]-th
                # element of the TA (eg. if ptr_stack_prev_to_top[0] is 3,
                # stack_h_prev_to_top[0] = stack_hta.read(3))
                # in other words, stack_h_prev_to_top[i] contains the full minibatch
                # at the right time for element i in the minibatch; we only care
                # about the element i in said minibatch (everything else is irrelevant
                # for minibatch element i), hence we index elements [0,0], [1,1], [2,2] etc.
                with tf.control_dependencies([tf.Assert(tf.reduce_all(ptr_stack_prev_to_top < stack_top), [ptr_stack_prev_to_top, stack_top])]):
                    stack_h_prev_to_top = stack_hta.gather(ptr_stack_prev_to_top, name='stack_h_prev_to_top')
                    stack_h_prev_to_top = tf.check_numerics(tf.gather_nd(stack_h_prev_to_top, diag_indices), 'invalid stack h prev to top')

                # apply the rnn
                rnn_inputs = tf.concat((buffer_top, stack_htop, stack_h_prev_to_top), axis=1, name='rnn_inputs')
                next_rnn_output, next_rnn_state = rnn_cell(rnn_inputs, rnn_state)

                # choose the next parser transition (or read it from the input)
                if self._train_syntactic_parser:
                    with tf.variable_scope('parser'):
                        next_op_logits = tf.nn.softmax(tf.layers.dense(next_rnn_output, 2), axis=1)
                        next_op_is_reduce = tf.cast(tf.argmax(next_op_logits, axis=1, dtype=tf.int16), tf.bool)
                else:
                    next_op_is_reduce = parses[time]
                # if we're out of buffer, we cannot shift
                next_op_is_reduce = tf.where(buffer_ptr + 1 >= input_length, tf.ones((batch_size,), dtype=tf.bool), next_op_is_reduce)

                # discard the tracking RNN output if it used a padding token as buffer_top
                zero_output = tf.zeros((batch_size, self._output_size), dtype=tf.float32)
                next_rnn_output = tf.where(buffer_ptr >= input_length, zero_output, next_rnn_output)
                # (we could also discard the RNN state here and copy over the previous state, but it does not
                # matter and I don't want to mess with LSTMStateTuple more than I need)

                # now the tricky part: conditionally execute both shift and
                # reduce

                # if shift, the top of the stack will contain the current top of the
                # buffer (split between c and h)
                # (NOTE: the buffer's c and h come from a linear transform, so it does not
                # matter that much which is c and which is h - the learned linear transform
                # will be the same up to a permutation of the columns if you swap them
                # for consistency with LSTMStateTuple, the first part is c and the second h)
                if_shift_stack_ctop = buffer_top[:, :self._output_size]
                if_shift_stack_htop = buffer_top[:, self._output_size:]
                # the top of the back pointer stack (which we're about to write to back_pointer_ta)
                # will point to the current stack top
                if_shift_back_pointer_top = tf.ones((batch_size,), dtype=tf.int32) * stack_top
                # the buffer will be advanced
                if_shift_buffer_ptr = buffer_ptr+1
                # to the output we write the stack top
                if_shift_output = if_shift_stack_htop

                # if reduce, we call the tree cell
                # stack_ctop and stack_c_prev_to_top are the same as their h versions
                stack_ctop = stack_cta.read(stack_top)
                stack_c_prev_to_top = stack_cta.gather(ptr_stack_prev_to_top)
                stack_c_prev_to_top = tf.gather_nd(stack_c_prev_to_top, diag_indices)
                # note that it doesn't really matter what is left and what is right,
                # the TreeLSTM is symmetric; we only need to pair (h,c) for the correct
                # elements
                left_child = LSTMStateTuple(stack_c_prev_to_top, stack_h_prev_to_top)
                right_child = LSTMStateTuple(stack_ctop, stack_htop)

                next_tree_output, next_tree_state = tree_cell(left_child, right_child,
                                                              extra_input=next_rnn_output if self._use_tracking_rnn else None)
                # the top the stack will contain the tree cell result
                if_reduce_stack_htop = next_tree_state.h
                if_reduce_stack_ctop = next_tree_state.c
                # the top of the back pointer stack will contain the back pointer of 
                # the second to last element in the stack
                # this will means that the next time we read the stack, we will skip
                # both the current top of the stack and the element immediately before
                # the current top of the stack
                if_reduce_back_pointer_top = back_pointer_ta.gather(ptr_stack_prev_to_top)
                if_reduce_back_pointer_top = tf.gather_nd(if_reduce_back_pointer_top, diag_indices, name='if_reduce_back_pointer_top')
                with tf.control_dependencies([tf.Assert(tf.reduce_all(if_reduce_back_pointer_top < stack_top), [if_reduce_back_pointer_top, ptr_stack_prev_to_top, stack_top])]):
                    if_reduce_back_pointer_top = tf.identity(if_reduce_back_pointer_top)
                if_reduce_buffer_ptr = buffer_ptr
                if_reduce_output = next_tree_output

                new_stack_htop = tf.where(next_op_is_reduce, if_reduce_stack_htop, if_shift_stack_htop, name='new_stack_htop')
                new_stack_ctop = tf.where(next_op_is_reduce, if_reduce_stack_ctop, if_shift_stack_ctop, name='new_stack_ctop')
                new_back_pointer_top = tf.where(next_op_is_reduce, if_reduce_back_pointer_top, if_shift_back_pointer_top, name='new_back_pointer_top')
                new_buffer_ptr = tf.where(next_op_is_reduce, if_reduce_buffer_ptr, if_shift_buffer_ptr, name='new_buffer_ptr')
                new_output = tf.where(next_op_is_reduce, if_reduce_output, if_shift_output, name='new_output')

                # now check if we had finished actually
                # if we had finished, we copy over the top of the stack instead of whatever shift/reduce tells
                # us to write (because the value we just read of shift/reduce was bogus)
                # because we copy, stack_hta/stack.cta[time] always contains the result of the
                # last reduce, and we don't need a vector for stack_top (which saves a few
                # expensive .gather() + gather_nd() ops)
                # we also save the right back pointer for this copy of the stack top, in case
                # we ever want to walk the stack back
                finish_time = 2*input_length-1
                finished = time >= finish_time

                new_buffer_ptr = tf.where(finished, buffer_ptr, new_buffer_ptr, name='new_buffer_ptr_finished')
                new_stack_htop = tf.where(finished, stack_htop, new_stack_htop, name='new_stack_htop_finished')
                new_stack_hta = stack_hta.write(stack_top+1, tf.check_numerics(new_stack_htop, 'invalid new stack htop'))
                new_stack_ctop = tf.where(finished, stack_ctop, new_stack_ctop, name='new_stack_ctop_finished')
                new_stack_cta = stack_cta.write(stack_top+1, tf.check_numerics(new_stack_ctop, 'invalid new stack ctop'))
                new_back_pointer_top = tf.where(finished, ptr_stack_prev_to_top, new_back_pointer_top, name='new_back_pointer_top_finished')
                new_back_pointer_ta = back_pointer_ta.write(stack_top+1, new_back_pointer_top)
                new_output = tf.where(finished, tf.zeros((batch_size, self._output_size)), new_output)
                new_output_ta = output_ta.write(time, new_output)

                new_time = tf.add(time, 1, name='new_time')

                with tf.control_dependencies([tf.group(finished, new_time, new_stack_hta.flow, new_stack_cta.flow,
                                                       new_back_pointer_ta.flow, new_output_ta.flow)]):
                    return (finished, new_time, new_buffer_ptr, new_stack_hta, new_stack_cta, new_back_pointer_ta, new_output_ta, next_rnn_state)

            initial_tree_state = tree_cell.zero_state(batch_size)
            initial_stack_hta = initial_stack_hta.write(0, initial_tree_state.h).write(1, initial_tree_state.h)
            initial_stack_cta = initial_stack_cta.write(0, initial_tree_state.c).write(1, initial_tree_state.c)
            initial_back_pointer_ta = initial_back_pointer_ta.write(0, tf.zeros((batch_size,), dtype=tf.int32)).write(1, tf.zeros((batch_size,), dtype=tf.int32))

            initial_rnn_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            initial_finished = tf.zeros((batch_size,), dtype=tf.bool, name='initial_finished')
            initial_time = 0
            initial_buffer_ptr = tf.zeros((batch_size,), dtype=tf.int32, name='initial_buffer_ptr')

            initial_loop_state = (initial_finished, initial_time, initial_buffer_ptr,
                                  initial_stack_hta, initial_stack_cta, initial_back_pointer_ta,
                                  initial_output_ta, initial_rnn_state)
            _, final_time, _, final_stack_hta, final_stack_cta, final_back_pointer_ta, final_output_ta, _ = tf.while_loop(cond, body, initial_loop_state, parallel_iterations=1, swap_memory=True)

            final_stack_htop = final_stack_hta.read(final_time+1)
            final_stack_ctop = final_stack_cta.read(final_time+1)

            # transpose outputs back to be batch major
            outputs = final_output_ta.stack()
            outputs = tf.transpose(outputs, (1, 0, 2))

            # final state is a tuple of LSTMStateTuple (one LSTMStateTuple for each RNN layer)
            final_state = (LSTMStateTuple(final_stack_ctop, final_stack_htop),)
            return outputs, final_state
