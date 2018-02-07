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

    def __init__(self, model, train_data, train_eval, dev_eval, saver,
                 opt_eval_metric='accuracy',
                 model_dir='./model',
                 max_length=40,
                 batch_size=256,
                 n_epochs=40,
                 **kw):
        '''
        Constructor
        '''
        self.model = model
        self.train_data = train_data

        self.train_eval = train_eval
        self.dev_eval = dev_eval
        self.saver = saver

        self._opt_eval_metric = opt_eval_metric
        self._model_dir = model_dir
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._extra_kw = kw

    def run_epoch(self, sess, losses, grad_norms, epoch):
        n_minibatches, total_loss = 0, 0
        total_n_minibatches = (len(self.train_data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        for data_batch in get_minibatches(self.train_data, self._batch_size):
            loss, grad_norm = self.model.train_on_batch(sess, *data_batch, batch_number=n_minibatches, epoch=epoch, **self._extra_kw)
            total_loss += loss
            losses.append(float(loss))
            grad_norms.append(float(grad_norm))
            n_minibatches += 1
            progbar.update(n_minibatches, values=[('loss', loss)])
        return total_loss / n_minibatches

    def fit(self, sess):
        best = None
        best_train = None
        losses = []
        grad_norms = []
        eval_metrics = dict()
        # flush stdout so we show the output before the first progress bar
        sys.stdout.flush()
        try:
            for epoch in range(self._n_epochs):
                average_loss = self.run_epoch(sess, losses, grad_norms, epoch)
                print('Epoch {:}: loss = {:.4f}'.format(epoch, average_loss))

                train_metrics = self.train_eval.eval(sess, save_to_file=False)
                dev_metrics = self.dev_eval.eval(sess, save_to_file=False)
                for metric, dev_value in dev_metrics.items():
                    if metric not in eval_metrics:
                        eval_metrics[metric] = []
                    eval_metrics[metric].append((float(train_metrics[metric]), float(dev_value)))
                comparison_metric = dev_metrics[self._opt_eval_metric]
                
                if best is None or comparison_metric >= best:
                    print('Found new model with best ' + self._opt_eval_metric)
                    self.saver.save(sess, os.path.join(self._model_dir, 'best'))
                    best = comparison_metric
                    best_train = train_metrics[self._opt_eval_metric]
                print()
                sys.stdout.flush()
                with open(os.path.join(self._model_dir, 'train-stats.json'), 'w') as fp:
                    output = dict(loss=losses, grad=grad_norms)
                    output.update(eval_metrics)
                    json.dump(output, fp)
        finally:
            with open(os.path.join(self._model_dir, 'train-stats.json'), 'w') as fp:
                output = dict(loss=losses, grad=grad_norms)
                output.update(eval_metrics)
                json.dump(output, fp)
        return best, best_train
