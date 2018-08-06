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
Utilities for creating Tensorflow metric functions.

Created on Jul 30, 2018

@author: gcampagn
'''

import numpy as np
import tensorflow as tf


def adjust_predictions_labels(predictions, labels, num_elements_per_time=3):
    batch_size = tf.shape(predictions)[0]
    
    predictions = tf.reshape(predictions, (batch_size, -1, num_elements_per_time))
    
    _assert1 = tf.Assert(tf.equal(tf.shape(labels)[0], batch_size),
                         data=(batch_size, tf.shape(labels), tf.shape(predictions), labels))
    with tf.control_dependencies([_assert1]):
        labels = tf.reshape(labels, (batch_size, -1, num_elements_per_time))
    
    prediction_time = tf.shape(predictions)[1]
    label_time = tf.shape(labels)[1]
    max_time = tf.maximum(prediction_time, label_time)
    
    predictions = tf.pad(predictions, [[0, 0], [0, max_time - prediction_time], [0, 0]])
    predictions.set_shape((None, None, num_elements_per_time))
    labels = tf.pad(labels, [[0, 0], [0, max_time - label_time], [0, 0]])
    labels.set_shape((None, None, num_elements_per_time))
    return batch_size, predictions, labels

def accuracy(predictions, labels, features, num_elements_per_time=3):
    batch_size, predictions, labels = adjust_predictions_labels(predictions, labels,
                                                                num_elements_per_time)
    weights = tf.ones((batch_size,), dtype=tf.float32)
    ok = tf.to_float(tf.reduce_all(tf.equal(predictions, labels), axis=[1, 2]))
    return ok, weights
        
        
def grammar_accuracy(predictions, labels, features, num_elements_per_time=3):
    batch_size, predictions, labels = adjust_predictions_labels(predictions, labels,
                                                                num_elements_per_time)
    weights = tf.ones((batch_size,), dtype=tf.float32)
    return tf.cond(tf.shape(predictions)[1] > 0,
                   lambda: tf.to_float(predictions[:,0,0] > 0),
                   lambda: tf.zeros_like(weights)), weights


def make_pyfunc_metric_fn(per_element_pyfunc, num_elements_per_time=3):
    def batch_pyfunc(predictions, labels):
        assert len(predictions) == len(labels)
        output = np.empty((len(predictions),), dtype=np.float32)
        for i in range(len(predictions)):
            output[i] = per_element_pyfunc(predictions[i], labels[i])
        return output
    
    def metric_fn(predictions, labels, features):
        batch_size, predictions, labels = adjust_predictions_labels(predictions, labels,
                                                                    num_elements_per_time)
        weights = tf.ones((batch_size,), dtype=tf.float32)
        
        ok = tf.contrib.framework.py_func(batch_pyfunc,
                                          [predictions, labels],
                                          output_shapes=tf.TensorShape([None]),
                                          output_types=tf.float32)
        return ok, weights
    return metric_fn