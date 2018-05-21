# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
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
Created on Mar 16, 2017

@author: gcampagn
'''

import sys
import json
import os
import tensorflow as tf
import numpy as np
import time

from .general_utils import get_minibatches, Progbar

class Trainer(object):
    '''
    Train a model on data
    '''

    def __init__(self, model, train_sets, train_data, train_evals, dev_evals, saver,
                 opt_eval_metric='accuracy',
                 model_dir='./model',
                 max_length=40,
                 batch_size=256,
                 n_epochs=40,
                 shuffle_data=True,
                 load_existing=False,
                 curriculum_max_prob=0.7,
                 curriculum_schedule=0.05,
                 **kw):
        '''
        Constructor
        '''
        self.model = model
        self.train_sets = train_sets
        self.train_data = train_data

        self.train_evals = train_evals
        self.dev_evals = dev_evals
        self.saver = saver

        self._opt_eval_metric = opt_eval_metric
        self._model_dir = model_dir
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._shuffle_data = shuffle_data
        self._load_existing = load_existing
        
        if len(self.train_sets) == 1:
            curriculum_schedule = 0
            curriculum_max_prob = 0
        if len(self.train_sets) != 2 and curriculum_schedule != 0.0:
            raise ValueError('Must have exactly two training sets for curriculum learning')
        self._curriculum_schedule = curriculum_schedule
        self._curriculum_max_prob = curriculum_max_prob
        
        self._extra_kw = kw
        
        self._eval_metrics = dict()
        self._losses = []
        self._grad_norms = []
        
        self._best = None
        
        if self._load_existing:
            with open(os.path.join(self._model_dir, 'train-stats.json'), 'r') as fp:
                existing = json.load(fp)
            self._losses = existing['loss']
            self._grad_norms = existing['grad']
            for key in existing:
                if key in ('loss', 'grad'):
                    continue
                self._eval_metrics[key] = existing[key]
            if len(self._eval_metrics[self._opt_eval_metric]) > 0:
                self._best = max(x[1] for x in self._eval_metrics[self._opt_eval_metric])

    def run_epoch(self, sess, train_data, epoch):
        n_minibatches, total_loss = 0, 0
        total_n_minibatches = (len(train_data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        for data_batch in get_minibatches(train_data, self._batch_size, shuffle=(self._shuffle_data or epoch >= 3)):
            loss, grad_norm = self.model.train_on_batch(sess, *data_batch, batch_number=n_minibatches, epoch=epoch, **self._extra_kw)
            total_loss += loss
            self._losses.append(float(loss))
            self._grad_norms.append(float(grad_norm))
            n_minibatches += 1
            progbar.update(n_minibatches, values=[('loss', loss)])
        return total_loss / n_minibatches

    def _save_stats(self):
        with open(os.path.join(self._model_dir, 'train-stats.json'), 'w') as fp:
            output = dict(loss=self._losses, grad=self._grad_norms)
            output.update(self._eval_metrics)
            json.dump(output, fp)

    def fit(self, sess):
        if self._curriculum_schedule != 0.0:
            self._fit_curriculum(sess)
        else:
            for i, key in enumerate(self.train_sets):
                self._fit_set(sess, key, self.train_data[key], i)
    
    def _mix_curriculum(self, epoch):
        easy_set = self.train_data[self.train_sets[0]]
        hard_set = self.train_data[self.train_sets[1]]
        
        easy_set_size = len(easy_set[0])
        hard_set_size = len(hard_set[0])
        
        hard_set_fraction = min(self._curriculum_max_prob, self._curriculum_schedule * epoch)
        
        easy_set_target = int((1 - hard_set_fraction) * (easy_set_size + hard_set_size))
        hard_set_target = int(hard_set_fraction * (easy_set_size + hard_set_size))
        if hard_set_target > hard_set_size:
            hard_set_target = hard_set_size
            easy_set_target = int(hard_set_size * (1 - hard_set_fraction) / hard_set_fraction)
        elif easy_set_target > easy_set_size:
            easy_set_target = easy_set_size
            hard_set_target = int(easy_set_size * hard_set_fraction / (1 - hard_set_fraction))
        
        print('Epoch %d: easy set = %d, hard set = %d' % (epoch, easy_set_target, hard_set_target))
        easy_set_indices = np.random.choice(np.arange(easy_set_size), size=easy_set_target)
        hard_set_indices = np.random.choice(np.arange(hard_set_size), size=hard_set_target)

        def mix_one(easy, hard):
            if isinstance(easy, dict):
                mixed = dict()
                for key in easy:
                    mixed[key] = np.concatenate((easy[key][easy_set_indices], hard[key][hard_set_indices]), axis=0)
                return mixed
            elif isinstance(easy, list):
                return [easy[i] for i in easy_set_indices] + [hard[i] for i in hard_set_indices]
            else:
                return np.concatenate((easy[easy_set_indices], hard[hard_set_indices]), axis=0)
        
        return tuple(mix_one(easy, hard) for easy, hard in zip(easy_set, hard_set))
    
    def _fit_curriculum(self, sess):
        # flush stdout so we show the output before the first progress bar
        sys.stdout.flush()
        try:
            for epoch in range(self._n_epochs):
                train_data = self._mix_curriculum(epoch)
                average_loss = self.run_epoch(sess, train_data, epoch)
                print('Epoch {:}: loss = {:.4f}'.format(epoch, average_loss))

                epoch_metrics = dict()

                # eval train accuracy only on the hard set
                train_metrics = self.train_evals[1].eval(sess, save_to_file=False)
                for metric, train_value in train_metrics.items():
                    epoch_metrics[metric] = [train_value]

                for dev_eval in self.dev_evals:
                    dev_metrics = dev_eval.eval(sess, save_to_file=False)
                    for metric, dev_value in dev_metrics.items():
                        epoch_metrics[metric].append(float(dev_value))
                for metric, values in epoch_metrics.items():
                    if metric not in self._eval_metrics:
                        self._eval_metrics[metric] = []
                    self._eval_metrics[metric].append(values)

                comparison_metric = epoch_metrics[self._opt_eval_metric][1]
                
                if self._best is None or comparison_metric >= self._best:
                    print('Found new model with best ' + self._opt_eval_metric + ' (' + str(comparison_metric) + ')')
                    self.saver.save(sess, os.path.join(self._model_dir, 'best'))
                    self._best = comparison_metric
                print()
                sys.stdout.flush()
                self._save_stats()
        finally:
            self._save_stats()

    def _fit_set(self, sess, train_key, train_data, i):
        start_epoch = i * self._n_epochs
        # flush stdout so we show the output before the first progress bar
        sys.stdout.flush()
        try:
            for epoch in range(start_epoch, start_epoch + self._n_epochs):
                average_loss = self.run_epoch(sess, train_data, epoch)
                print('Epoch {:}: loss = {:.4f}'.format(epoch, average_loss))

                epoch_metrics = dict()

                train_metrics = self.train_evals[i].eval(sess, save_to_file=False)
                for metric, train_value in train_metrics.items():
                    epoch_metrics[metric] = [train_value]

                for dev_eval in self.dev_evals:
                    dev_metrics = dev_eval.eval(sess, save_to_file=False)
                    for metric, dev_value in dev_metrics.items():
                        epoch_metrics[metric].append(float(dev_value))
                for metric, values in epoch_metrics.items():
                    if metric not in self._eval_metrics:
                        self._eval_metrics[metric] = []
                    self._eval_metrics[metric].append(values)

                comparison_metric = epoch_metrics[self._opt_eval_metric][1]
                
                if self._best is None or comparison_metric >= self._best:
                    print('Found new model with best ' + self._opt_eval_metric + ' (' + str(comparison_metric) + ')')
                    self.saver.save(sess, os.path.join(self._model_dir, 'best'))
                    self._best = comparison_metric
                print()
                sys.stdout.flush()
                self._save_stats()
        finally:
            self._save_stats()
