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
Created on Jul 26, 2018

@author: gcampagn
'''

# Part of this code is derived from Tensor2Tensor, which is:
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tensor2tensor.utils import metrics
from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.layers import common_layers
from tensor2tensor.utils.t2t_model import _remove_summaries


class LUINetModel(T2TModel):
    '''
    A base model, similar to T2TModel, that overrides some of the
    methods to make it more suitable to our task.
    
    All models in LUINet should inherit from LUINetModel
    '''
    
    def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
        """Construct EstimatorSpec for EVAL mode."""
        del losses_dict
        hparams = self.hparams
    
        problem = hparams.problem
        if common_layers.is_on_tpu():
            raise NotImplementedError("TPU usage is not supported")
      
        if hasattr(problem, "compute_predictions"):
            predictions = problem.compute_predictions(logits, features, hparams)
        else:
            predictions = tf.contrib.framework.nest.map_structure(lambda x: tf.argmax(x, axis=-1),
                                                                  logits)
        
        problem_metrics = problem.eval_metrics()
        if isinstance(problem_metrics, list):
            eval_metrics = metrics.create_evaluation_metrics([problem], hparams)
            
            for metric_name, metric_fn in eval_metrics.items():
                eval_metrics[metric_name] = metric_fn(logits, features,
                                                      features["targets"])
        else:
            eval_metrics = {}
            
            for metric_key, metric_fn in problem_metrics.items():
                metric_name = "metrics-%s/%s" % (problem.name, metric_key)
                first, second = metric_fn(predictions, labels, features)
                
                if isinstance(second, tf.Tensor):
                    scores, weights = first, second
                    eval_metrics[metric_name] = tf.metrics.mean(scores, weights)
                else:
                    value, update_op = first, second
                    eval_metrics[metric_name] = (value, update_op)
          
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                eval_metric_ops=eval_metrics,
                loss=loss)

    def estimator_spec_predict(self, features, use_tpu=False):
        """Construct EstimatorSpec for PREDICT mode."""
        
        if use_tpu:
            raise NotImplementedError("TPU usage is not supported")
        
        decode_hparams = self._decode_hparams
        infer_out = self.infer(
            features,
            beam_size=decode_hparams.beam_size,
            top_beams=(decode_hparams.beam_size
                       if decode_hparams.return_beams else 1),
            alpha=decode_hparams.alpha,
            decode_length=decode_hparams.extra_length,
            use_tpu=use_tpu)
        
        problem = decode_hparams.problem
        if isinstance(infer_out, dict):
            if "outputs" in infer_out:
                outputs = infer_out["outputs"]
            elif hasattr(problem, "compute_predictions"):
                outputs = problem.compute_predictions(infer_out["logits"], features,
                                                      model_hparams=self._hparams)
            else:
                outputs = tf.contrib.framework.nest.map_structure(lambda x: tf.argmax(x, axis=-1),
                                                                  infer_out["logits"])
            scores = infer_out["scores"]
        else:
            outputs = infer_out
            scores = None
    
        inputs = features.get("inputs")
        if inputs is None:
            inputs = features["targets"]
    
        predictions = {
            "outputs": outputs,
            "scores": scores,
            "inputs": inputs,
            "targets": features.get("infer_targets"),
            "batch_prediction_key": features.get("batch_prediction_key"),
        }
        for k in list(predictions.keys()):
            if predictions[k] is None:
                del predictions[k]
    
        export_out = {"outputs": predictions["outputs"]}
        if "scores" in predictions:
            export_out["scores"] = predictions["scores"]
    
        # Necessary to rejoin examples in the correct order with the Cloud ML Engine
        # batch prediction API.
        if "batch_prediction_key" in predictions:
            export_out["batch_prediction_key"] = predictions["batch_prediction_key"]
    
        _remove_summaries()
    
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(export_out)
        }
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs)