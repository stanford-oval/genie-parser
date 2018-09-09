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
from backcall._signatures import signature
from tensorflow import estimator

'''
Created on Aug 9, 2018

@author: gcampagn
'''

import os

import tensorflow as tf

from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import decoding
from tensor2tensor.utils import t2t_model

class Signature(object):
    def __init__(self, name, placeholders, predictions):
        self._name = name
        self._placeholders = placeholders
        self._predictions = predictions

    @property
    def name(self):
        return self._name

    def __call__(self, session, inputs):
        return session.run(self._predictions, feed_dict={
            self._placeholders[k]: v for k, v in inputs.items()
        })

class Predictor(object):
    def __init__(self, model_dir, config):
        self._signatures = dict()
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(1234)

            # initialize the hparams, problem and model
            self._hparams = trainer_lib.create_hparams(config['hparams_set'],
                                                       config.get('hparams_overrides', ''),
                                                       os.path.join(model_dir, 'assets.extra'),
                                                       config['problem'])
            problem = self._hparams.problem
            
            decode_hp = decoding.decode_hparams(config.get('decode_hparams', ''))
            
            run_config = trainer_lib.create_run_config(self._hparams,
                                                       model_dir=model_dir,
                                                       schedule="decode")
            
            model_fn = t2t_model.T2TModel.make_estimator_model_fn(
                config['model'], self._hparams, decode_hparams=decode_hp)
            
            # create the orediction signatures (input/output ops)
            serving_receiver = problem.direct_serving_input_fn(self._hparams)
            estimator_spec = model_fn(serving_receiver.features, None,
                                      mode=tf.estimator.ModeKeys.PREDICT,
                                      params=None,
                                      config=run_config)
            
            for key, sig_spec in estimator_spec.export_outputs.items():
                # only PredictOutputs are supported, ClassificationOutput
                # and RegressionOutputs are weird artifacts of Google shipping
                # almost unmodified Tensorflow graphs through their Cloud ML
                # platform
                assert isinstance(sig_spec, tf.estimator.export.PredictOutput)
                
                sig = Signature(key,
                                serving_receiver.receiver_tensors,
                                sig_spec.outputs)
                self._signatures[key] = sig
            
            # load the model & init the session
            
            scaffold = tf.train.Scaffold()
            checkpoint_filename = os.path.join(model_dir,
                                               tf.saved_model.constants.VARIABLES_DIRECTORY,
                                               tf.saved_model.constants.VARIABLES_FILENAME)
            session_creator = tf.train.ChiefSessionCreator(scaffold,
                                                           config=run_config.session_config,
                                                           checkpoint_filename_with_path=checkpoint_filename)
            self._session = tf.train.MonitoredSession(session_creator=session_creator)

    @property
    def problem(self):
        return self._hparams.problem

    @property
    def signatures(self):
        return self._signatures.keys()
         
    def predict(self, inputs, signature_key=None):
        if signature_key is None:
            signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        
        return self._signatures[signature_key](self._session, inputs)