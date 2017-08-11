#!/usr/bin/python3
#
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

import numpy as np

import sys
import matplotlib
matplotlib.use('Gtk3Cairo')
import matplotlib.pyplot as plt

import json

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def learning():
    with open(sys.argv[1], 'r') as fp:
        data = json.load(fp)
    loss = np.array(data['loss'])[200:]
    accuracy = np.array(data['accuracy'])
    train_acc = 100*accuracy[:,0]
    dev_acc = 100*accuracy[:,1]
    grad_norm = np.array(data['grad'])[200:]
    #print('grad norm mean', np.average(grad_norm[100:]))
    #print('grad norm std', np.std(grad_norm[100:]))
    function_accuracy = np.array(data['function_accuracy'])
    train_fnacc = 100*function_accuracy[:,0]
    dev_fnacc = 100*function_accuracy[:,1]
    recall = np.array(data['recall'])
    train_recall = 100*recall[:,0]
    dev_recall = 100*recall[:,1]
    eval_loss = np.array(data['eval_loss'])
    train_eval_loss = eval_loss[:,0]
    dev_eval_loss = eval_loss[:,1]

    plt.suptitle(sys.argv[1])

    plt.subplot(3, 2, 1)
    plt.title('Training Loss')
    X = 200+np.arange(len(loss))
    plt.xlabel('Minibatch #')
    plt.plot(X, loss)
    plt.plot(X[5:-5], movingaverage(loss,11), 'y-')
    plt.legend(["Loss"], loc="upper right")

    plt.subplot(3, 2, 3)
    plt.title('Gradient')
    plt.xlabel('Minibatch #')
    plt.plot(X, grad_norm)
    plt.legend(["Gradient"], loc='upper right')

    train_eval_loss = train_eval_loss[3:]
    dev_eval_loss = dev_eval_loss[3:]
    X = 3+np.arange(len(train_eval_loss))
    plt.subplot(3, 2, 5)
    plt.title('Evaluation Loss')
    plt.xlim(3+0, 3+len(train_eval_loss)+1)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_eval_loss)
    plt.plot(X, dev_eval_loss)
    plt.plot(X[2:-2], movingaverage(dev_eval_loss, 5), '--')
    plt.legend(['Train Loss', 'Dev Loss'], loc='upper right')

    X = np.arange(len(train_acc))
    plt.subplot(3, 2, 2)
    plt.title('Function Accuracy')
    plt.xlim(0, len(train_acc)+1)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_fnacc)
    plt.plot(X, dev_fnacc)
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")

    plt.subplot(3, 2, 4)
    plt.title('Program Accuracy')
    plt.xlim(0, len(train_acc)+1)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_acc)
    plt.plot(X, dev_acc)
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")

    plt.subplot(3, 2, 6)
    plt.title('Recall')
    plt.xlim(0, len(train_acc)+1)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_recall)
    plt.plot(X, dev_recall)
    plt.ylim(0, 100)
    plt.legend(["Train Recall", "Dev Recall"], loc="lower right")

    plt.tight_layout()
    plt.show()

learning()
