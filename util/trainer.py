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
        
        inputs, input_lengths, parses, labels, label_lengths = train_data
        inputs = np.array(inputs)
        parses = np.array(parses)
        input_lengths = np.reshape(np.array(input_lengths, dtype=np.int32), (len(inputs), 1))
        labels = np.array(labels)
        label_lengths = np.reshape(np.array(label_lengths, dtype=np.int32), (len(inputs), 1))
        stacked_train_data = np.concatenate((inputs, input_lengths, parses, labels, label_lengths), axis=1)
        assert stacked_train_data.shape == (len(train_data[0]), max_length + 1 + 2*max_length-1 + max_length + 1)
        self.train_data = stacked_train_data

        self.train_eval = train_eval
        self.dev_eval = dev_eval
        self.saver = saver

        self._model_dir = model_dir
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._extra_kw = kw

    def run_epoch(self, sess, inputs, input_lengths, parses,
                  labels, label_lengths, losses, grad_norms):
        n_minibatches, total_loss = 0, 0
        total_n_minibatches = (len(inputs)+self._batch_size-1)//self._batch_size
        progbar = Progbar(total_n_minibatches)
        for data_batch in get_minibatches([inputs, input_lengths, parses, labels, label_lengths], self._batch_size):
            loss, grad_norm = self.model.train_on_batch(sess, *data_batch, **self._extra_kw)
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
        losses = []
        grad_norms = []
        try:
            for epoch in range(self._n_epochs):
                start_time = time.time()
                shuffled = np.array(self.train_data, copy=True)
                np.random.shuffle(shuffled)
                inputs = shuffled[:,:self._max_length]
                input_lengths = shuffled[:,self._max_length]
                parses = shuffled[:,self._max_length + 1:self._max_length + 1 + 2*self._max_length-1]
                labels = shuffled[:,3*self._max_length:-1]
                label_lengths = shuffled[:,-1]

                average_loss = self.run_epoch(sess, inputs, input_lengths, parses,
                                              labels, label_lengths, losses, grad_norms)
                duration = time.time() - start_time
                print('Epoch {:}: loss = {:.4f} ({:.3f} sec)'.format(epoch, average_loss, duration))
                #self.saver.save(sess, os.path.join(self._model_dir, 'epoch'), global_step=epoch)

                train_acc = self.train_eval.eval(sess, save_to_file=False)
                if self.dev_eval is not None:
                    dev_acc = self.dev_eval.eval(sess, save_to_file=False)
                    if best is None or dev_acc > best:
                        print('Found new best model')
                        self.saver.save(sess, os.path.join(self._model_dir, 'best'))
                        best = dev_acc
                        best_train = train_acc
                    accuracy_stats.append((train_acc, dev_acc))
                else:
                    accuracy_stats.append((train_acc,))
                print()
                sys.stdout.flush()
        finally:
            with open(os.path.join(self._model_dir, 'train-stats.json'), 'w') as fp:
                json.dump(dict(accuracy=accuracy_stats, loss=losses, grad=grad_norms), fp)
        return best, best_train
