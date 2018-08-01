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

import copy
import tensorflow as tf

from tensor2tensor.utils import metrics
from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.layers import common_layers
from tensor2tensor.utils.t2t_model import _remove_summaries, log_info
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils.optimize import log_variable_sizes, \
    weight_decay_and_noise, ConditionalOptimizer


class LUINetModel(T2TModel):
    '''
    A base model, similar to T2TModel, that overrides some of the
    methods to make it more suitable to our task.
    
    All models in LUINet should inherit from LUINetModel
    '''
    
    def model_fn(self, features):
        with tf.variable_scope(tf.get_variable_scope(), use_resource=True):
            transformed_features = self.bottom(features)
    
            if self.hparams.activation_dtype == "bfloat16":
                for k, v in transformed_features.items():
                    if v.dtype == tf.float32:
                        transformed_features[k] = tf.cast(v, tf.bfloat16)
    
            with tf.variable_scope("body"):
                log_info("Building model body")
                body_out = self.body(transformed_features)
            output, losses = self._normalize_body_output(body_out)
    
            if "training" in losses:
                log_info("Skipping T2TModel top and loss because training loss "
                         "returned from body")
                logits = output
            else:
                logits = self.top(output, features)
                losses["training"] = 0.0
                if self._hparams.mode != tf.estimator.ModeKeys.PREDICT:
                    training_loss = self.loss(logits, features)
                    if isinstance(training_loss, dict):
                        assert "training" in training_loss
                        losses.update(training_loss)
                    else:
                        losses["training"] = training_loss
    
            return logits, losses
        
    def loss(self, logits, features):
        if isinstance(logits, dict):
            if self._problem_hparams:
                target_modality = self._problem_hparams.target_modality
            else:
                target_modality = {k: None for k in logits.keys()}
            for k in logits.keys():
                assert k in target_modality.keys(), (
                    "The key %s of model_body's returned logits dict must be in "
                    "problem_hparams.target_modality's dict." % k)
            losses = {}
            for k, v in logits.items():
                n, d = self._loss_single(v, target_modality[k], features[k])
                losses[k] = n / d
            losses["training"] = tf.add_n(list(losses.values()))
            return losses
        else:
            if self._problem_hparams:
                target_modality = self._problem_hparams.target_modality
            else:
                target_modality = None
            if isinstance(target_modality, dict):
                assert "targets" in target_modality, (
                    "model_body returned single logits so 'targets' must be a key "
                    "since problem_hparams.target_modality is a dict.")
            target_modality = target_modality["targets"]
            return self._loss_single(logits, target_modality, features["targets"])

    def optimize(self, loss, num_async_replicas=1):
        """Return a training op minimizing loss."""
        hparams = self.hparams
        
        lr = learning_rate.learning_rate_schedule(hparams)
        if num_async_replicas > 1:
            log_info("Dividing learning rate by num_async_replicas: %d",
                     num_async_replicas)
        lr /= tf.sqrt(float(num_async_replicas))
        
        loss = weight_decay_and_noise(loss, hparams, lr)
        loss = tf.identity(loss, name="total_loss")
        log_variable_sizes(verbose=hparams.summarize_vars)
        opt = ConditionalOptimizer(hparams.optimizer, lr, hparams)

        opt_summaries = ["loss", "learning_rate", "global_gradient_norm"]
        
        if hparams.clip_grad_norm:
            tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
        if hparams.grad_noise_scale:
            tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                            hparams.grad_noise_scale)
        
        return tf.contrib.layers.optimize_loss(
              name="training",
              loss=loss,
              global_step=tf.train.get_or_create_global_step(),
              learning_rate=lr,
              clip_gradients=hparams.clip_grad_norm or None,
              gradient_noise_scale=hparams.grad_noise_scale or None,
              optimizer=opt,
              summaries=opt_summaries,
              colocate_gradients_with_ops=True)

    @classmethod
    def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):
        """Model fn for Estimator.
    
        Args:
          hparams: HParams, model hyperparameters
          features: dict<str name, Tensor feature>
          labels: Tensor
          mode: tf.estimator.ModeKeys
          config: RunConfig, possibly with data_parallelism attribute
          params: dict, may include batch_size
          decode_hparams: HParams, used when mode == PREDICT.
          use_tpu: bool, whether using TPU
    
        Returns:
          TPUEstimatorSpec if use tpu else EstimatorSpec
        """
        hparams = copy.deepcopy(hparams)

        # Instantiate model
        data_parallelism = None
        if not use_tpu and config:
            data_parallelism = config.data_parallelism
        model = cls(
            hparams,
            mode,
            data_parallelism=data_parallelism,
            decode_hparams=decode_hparams)

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return model.estimator_spec_predict(features, use_tpu=use_tpu)

        # TRAIN and EVAL modes
        if hparams.eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
            logits, losses_dict = model.eval_autoregressive(features)
        else:
            logits, losses_dict = model(features)  # pylint: disable=not-callable

        assert "training" in losses_dict
        loss = losses_dict["training"]

        # Summarize losses
        with tf.name_scope("losses"):
            for loss_name, loss_val in sorted(losses_dict.items()):
                tf.summary.scalar(loss_name, loss_val)

        # EVAL mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return model.estimator_spec_eval(features, logits, labels, loss,
                                             losses_dict)

        # TRAIN mode
        assert mode == tf.estimator.ModeKeys.TRAIN
        num_async_replicas = (1 if (use_tpu or not config) else
                              config.t2t_device_info["num_async_replicas"])
        return model.estimator_spec_train(
            loss, num_async_replicas=num_async_replicas)
    
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