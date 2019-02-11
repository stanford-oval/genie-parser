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
Created on Aug 9, 2018

@author: gcampagn
'''

import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.utils import data_reader


class LUINetProblem(problem.Problem):
    """Base class for genie problems.
    
    All problems in genie should inherit from this class
    rather than (or in addition to) classes in t2t.data_generators.
    """
    
    def decode_targets(self, targets, features, model_hparams=None):
        return targets
    
    def compute_predictions(self, outputs, features, model_hparams=None,
                            decode=False):
        return outputs
    
    def direct_example_reading_spec(self):
        """Return a map from feature name to tf.Feature for direct (in-process) serving.
        
        By default, this uses the same features as loading from a dataset."""
        return self.example_reading_spec()
    
    def direct_serving_input_fn(self, hparams):
        """Input fn for serving export, starting from appropriate placeholders."""
        mode = tf.estimator.ModeKeys.PREDICT
        
        placeholders = dict()
        batch_size = None
        data_fields, _ = self.direct_example_reading_spec()
        for key, feature in data_fields:
            if isinstance(feature, tf.FixedLenFeature):
                placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                                   shape=tf.TensorShape([None]).concatenate(feature.shape),
                                                   name="placeholder_" + key)
            elif isinstance(feature, tf.VarLenFeature):
                placeholders[key] = tf.placeholder(dtype=feature.dtype,
                                                   shape=tf.TensorShape([None, None]),
                                                   name="placeholder_" + key)
            else:
                raise TypeError("Invalid feature type " + str(type(feature)))
            if batch_size is None:
                batch_size = tf.shape(placeholders[key], out_type=tf.int64)[0]

        dataset = tf.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
        dataset = dataset.map(self.maybe_reverse_and_copy)
        dataset = dataset.map(data_reader.cast_ints_to_int32)
        dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
        dataset = dataset.map(problem.standardize_shapes)
        features = tf.contrib.data.get_single_element(dataset)
    
        if self.has_inputs:
            features.pop("targets", None)
    
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=placeholders)