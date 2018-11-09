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
from tensor2tensor.utils import registry
from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils.optimize import log_variable_sizes, \
    weight_decay_and_noise, ConditionalOptimizer

from tensor2tensor.utils.t2t_model import _remove_summaries, \
    log_info, log_warn, set_custom_getter_compose

FLAGS = tf.flags.FLAGS

class LUINetModel(T2TModel):
    '''
    A base model, similar to T2TModel, that overrides some of the
    methods to make it more suitable to our task.
    
    All models in LUINet should inherit from LUINetModel
    '''
    
    @property
    def _target_modality_is_real(self):
        """Whether the target modality is real-valued."""
        target_modality = self._problem_hparams.target_modality
        return not isinstance(target_modality, dict) and \
             target_modality.name.startswith("real_")

    def _create_modalities(self, problem_hparams, hparams):
        """Construct modalities in problem_hparams."""

        input_modality_overrides = {}
        for override_str in hparams.input_modalities.split(";"):
           if override_str != "default":
               parts = override_str.split(":")
               feature_name = parts[0]
               modality_name = ":".join(parts[1:])
               input_modality_overrides[feature_name] = modality_name

        target_modality_name = None
        if hparams.target_modality and hparams.target_modality != "default":
            target_modality_name = hparams.target_modality

        input_modality = {}
        for f, modality_spec in problem_hparams.input_modality.items():
            if f in input_modality_overrides:
                _warn_changed_modality_type(input_modality_overrides[f],
                                            modality_spec[0], f)
                modality_spec = (input_modality_overrides[f], modality_spec[1])
            if isinstance(modality_spec, tuple):
                input_modality[f] = registry.create_modality(modality_spec, hparams)
            else:
                input_modality[f] = modality_spec(hparams)
        problem_hparams.input_modality = input_modality

        if isinstance(problem_hparams.target_modality, dict):
            target_modality = {}
            for f, modality_spec in problem_hparams.target_modality.items():
                # TODO(lukaszkaiser): allow overriding other target modalities.
                if target_modality_name and f == "targets":
                    _warn_changed_modality_type(target_modality_name, modality_spec[0],
                                                "target_modality/%s" % f)
                    modality_spec = (target_modality_name, modality_spec[1])
                if isinstance(modality_spec, tuple):
                    target_modality[f] = registry.create_modality(modality_spec, hparams)
                else:
                    target_modality[f] = modality_spec
        else:
            target_modality_spec = problem_hparams.target_modality
            if target_modality_name:
                _warn_changed_modality_type(target_modality_name,
                                            target_modality_spec[0], "target")
                target_modality_spec = (target_modality_name, target_modality_spec[1])

            if isinstance(modality_spec, tuple):
                target_modality = registry.create_modality(target_modality_spec, hparams)
            else:
                target_modality = modality_spec(hparams)
        problem_hparams.target_modality = target_modality
    
    def infer(self, 
        features=None, 
        decode_length=50, 
        beam_size=1, 
        top_beams=1, 
        alpha=0.0, 
        use_tpu=False):
        """A inference method.

        Quadratic time in decode_length.
    
        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.
          use_tpu: bool, whether to build the inference graph for TPU.
    
        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
          if slow greedy decoding is used then the dict will also contain {
              "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
              "losses": a dictionary: {loss-name (string): floating point `Scalar`
          }
        """
        set_custom_getter_compose(self._custom_getter)
        with self._eager_var_store.as_default():
            self.prepare_features_for_infer(features)
            if not self.has_input and beam_size > 1:
                log_warn("Beam searching for a model with no inputs.")
            if not self.has_input and self.hparams.sampling_method != "random":
                log_warn("Non-random sampling for a model with no inputs.")
            self._fill_problem_hparams_features(features)
    
            if self._problem_hparams:
                target_modality = self._problem_hparams.target_modality
                if not isinstance(target_modality, dict) and target_modality.is_class_modality:
                    beam_size = 1  # No use to run beam-search for a single class.
            if beam_size == 1:
                log_info("Greedy Decoding")
                results = self._greedy_infer(features, decode_length, use_tpu)
            else:
                log_info("Beam Decoding with beam size %d" % beam_size)
                results = self._beam_decode(features, decode_length, beam_size,
                                            top_beams, alpha)
    
            return results

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
                n, d = target_modality[k].loss(v, targets=features[k], features=features, curriculum=FLAGS.curriculum)
                n *= self._problem_hparams.loss_multiplier
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
            return self._loss_single(logits, target_modality, features["targets"]) * features["weight"]

    def optimize(self, loss, num_async_replicas=1, use_tpu=False):
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
        
        tf.summary.scalar("training/learning_rate", lr)
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
        if common_layers.is_xla_compiled():
            raise NotImplementedError("TPU usage is not supported")
      
        outputs = tf.contrib.framework.nest.map_structure(lambda x: tf.squeeze(tf.argmax(x, axis=-1), axis=[2, 3]),
                                                          logits)
      
        if hasattr(problem, "compute_predictions"):
            predictions = problem.compute_predictions(outputs, features, hparams,
                                                      decode=False)
        else:
            predictions = outputs

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
        
        problem = self.hparams.problem
        if isinstance(infer_out, dict):
            outputs = infer_out["outputs"]
            scores = infer_out["scores"]
        else:
            outputs = infer_out
            scores = None
        
        if hasattr(problem, "compute_predictions"):
            outputs = problem.compute_predictions(outputs, features,
                                                  model_hparams=self._hparams,
                                                  decode=True)    
    
        inputs = features.get("inputs")
        if inputs is None:
            inputs = features["targets"]
            
        infer_targets = features.get("infer_targets")
        if infer_targets is not None and \
            hasattr(problem, "decode_targets"):
            infer_targets = problem.decode_targets(infer_targets, features,
                                                   model_hparams=self._hparams)
    
        predictions = {
            "outputs": outputs,
            "scores": scores,
            "inputs": inputs,
            "targets": infer_targets,
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
        
        # if the model is encoder+decoder, offer the ability to run only the
        # encoder (this is used by the server in "retrieval mode", when
        # dealing with MultipleChoice queries) 
        if isinstance(infer_out, dict) and "encoded_inputs" in infer_out:
            export_outputs["encoded_inputs"] = tf.estimator.export.PredictOutput({
                "encoded_inputs": infer_out["encoded_inputs"]
            })
        
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs)
