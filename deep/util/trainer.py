'''
Created on Mar 16, 2017

@author: gcampagn
'''

import numpy as np
import time

from general_utils import get_minibatches

class Trainer(object):
    '''
    Train a model on data
    '''

    def __init__(self, model, train_data, train_eval, dev_eval, max_length=40, batch_size=256, n_epochs=40, dropout=1):
        '''
        Constructor
        '''
        self.model = model
        
        inputs, input_lengths, labels, label_lengths = train_data
        inputs = np.array(inputs)
        input_lengths = np.reshape(np.array(input_lengths, dtype=np.int32), (len(inputs), 1))
        labels = np.array(labels)
        label_lengths = np.reshape(np.array(label_lengths, dtype=np.int32), (len(inputs), 1))
        stacked_train_data = np.concatenate((inputs, input_lengths, labels, label_lengths), axis=1)
        assert stacked_train_data.shape == (len(train_data[0]), max_length + 1 + max_length + 1)
        self.train_data = stacked_train_data
        
        self.train_eval = train_eval
        self.dev_eval = dev_eval
        
        self._max_length = max_length
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._dropout = dropout

    def run_epoch(self, sess, inputs, input_lengths,
                  labels, label_lengths, **kw):
        n_minibatches, total_loss = 0, 0
        for data_batch in get_minibatches([inputs, input_lengths, labels, label_lengths], self._batch_size):
            n_minibatches += 1
            for x in data_batch:
                assert len(x) == len(data_batch[0])
            assert len(data_batch[0]) <= self._batch_size
            total_loss += self.model.train_on_batch(sess, *data_batch, **kw)
        return total_loss / n_minibatches

    def fit(self, sess):
        
        losses = []
        for epoch in range(self._n_epochs):
            start_time = time.time()
            shuffled = np.array(self.train_data, copy=True)
            np.random.shuffle(shuffled)
            inputs = shuffled[:,:self._max_length]
            input_lengths = shuffled[:,self._max_length]
            labels = shuffled[:,self._max_length + 1:-1]
            label_lengths = shuffled[:,-1]
            
            average_loss = self.run_epoch(sess, inputs, input_lengths,
                                          labels, label_lengths,
                                          dropout=self._dropout)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
            self.train_eval.eval(sess, save_to_file=False)
            if self.dev_eval is not None:
                self.dev_eval.eval(sess, save_to_file=False)
            print
        return losses