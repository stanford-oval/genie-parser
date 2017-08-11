# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
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

    def __init__(self, model, train_data, train_eval, dev_eval, saver, model_dir='./model', max_length=40, batch_size=256, n_epochs=40, **kw):
        '''
        Constructor
        '''
        self.model = model
        self.train_data = train_data

        self.train_eval = train_eval
        self.dev_eval = dev_eval
        self.saver = saver

        self._model_dir = model_dir
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._extra_kw = kw

    def run_epoch(self, sess, losses, grad_norms, batch_number):
        n_minibatches, total_loss = 0, 0
        total_n_minibatches = (len(self.train_data[0])+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        for data_batch in get_minibatches(self.train_data, self._batch_size):
            loss, grad_norm = self.model.train_on_batch(sess, *data_batch, batch_number=batch_number, **self._extra_kw)
            total_loss += loss
            losses.append(float(loss))
            grad_norms.append(float(grad_norm))
            n_minibatches += 1
            progbar.update(n_minibatches)
        return total_loss / n_minibatches

    def fit(self, sess):
        best = None
        best_train = None
        accuracy_stats = []
        function_accuracy_stats = []
        eval_losses = []
        recall = []
        losses = []
        grad_norms = []
        # flush stdout so we show the output before the first progress bar
        sys.stdout.flush()
        try:
            for epoch in range(self._n_epochs):
                start_time = time.time()

                average_loss = self.run_epoch(sess, losses, grad_norms, batch_number=epoch)
                duration = time.time() - start_time
                print('Epoch {:}: loss = {:.4f} ({:.3f} sec)'.format(epoch, average_loss, duration))
                #self.saver.save(sess, os.path.join(self._model_dir, 'epoch'), global_step=epoch)

                train_acc, train_eval_loss, train_acc_fn, train_recall = self.train_eval.eval(sess, save_to_file=False)
                if self.dev_eval is not None:
                    dev_acc, dev_eval_loss, dev_acc_fn, dev_recall = self.dev_eval.eval(sess, save_to_file=False)
                    if best is None or dev_acc > best:
                        print('Found new best model')
                        self.saver.save(sess, os.path.join(self._model_dir, 'best'))
                        if dev_acc > 0:
                            best = dev_acc
                            best_train = train_acc
                    accuracy_stats.append((float(train_acc), float(dev_acc)))
                    function_accuracy_stats.append((float(train_acc_fn), float(dev_acc_fn)))
                    eval_losses.append((float(train_eval_loss), float(dev_eval_loss)))
                    recall.append((float(train_recall), float(dev_recall)))
                else:
                    accuracy_stats.append((float(train_acc),))
                    function_accuracy_stats.append((float(train_acc_fn),))
                    eval_losses.append((float(train_eval_loss),))
                    recall.append((float(train_recall),))
                print()
                sys.stdout.flush()
        finally:
            with open(os.path.join(self._model_dir, 'train-stats.json'), 'w') as fp:
                json.dump(dict(accuracy=accuracy_stats, eval_loss=eval_losses, function_accuracy=function_accuracy_stats, recall=recall,
                               loss=losses, grad=grad_norms), fp)
        return best, best_train
