#!/usr/bin/python3
#
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
    grammar_accuracy = np.array(data.get('grammar_accuracy', None) or data['correct grammar'])
    train_gracc = 100*grammar_accuracy[:,0]
    dev_gracc = 100*grammar_accuracy[:,1]
    #recall = np.array(data['recall'])
    #train_recall = 100*recall[:,0]
    #dev_recall = 100*recall[:,1]
    precision = np.array(data['parse_action_precision'])
    train_precision = 100*precision[:,0]
    dev_precision = 100*precision[:,1]
    recall = np.array(data['parse_action_recall'])
    train_recall = 100*recall[:,0]
    dev_recall = 100*recall[:,1]
    f1 = np.array(data['parse_action_f1'])
    train_f1 = 100*f1[:,0]
    dev_f1 = 100*f1[:,1]
    eval_loss = np.array(data['eval_loss'])
    train_eval_loss = eval_loss[:,0]
    dev_eval_loss = eval_loss[:,1]

    plt.suptitle(sys.argv[1])

    plt.subplot(3, 3, 1)
    plt.title('Training Loss')
    X = 200+np.arange(len(loss))
    plt.xlabel('Minibatch #')
    plt.plot(X, loss)
    plt.plot(X[5:-5], movingaverage(loss,11), 'y-')
    #plt.legend(["Loss"], loc="upper right")

    plt.subplot(3, 3, 4)
    plt.title('Gradient')
    plt.xlabel('Minibatch #')
    plt.plot(X, grad_norm)
    #plt.legend(["Gradient"], loc='upper right')

    X = 1 + np.arange(len(train_eval_loss))
    plt.subplot(3, 3, 7)
    plt.title('Evaluation Loss')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_eval_loss)
    plt.plot(X, dev_eval_loss)
    #plt.plot(X[2:-2], movingaverage(dev_eval_loss, 5), '--')
    plt.legend(['Train Loss', 'Dev Loss'], loc='upper right')

    plt.subplot(3, 3, 2)
    plt.title('Function Accuracy')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_fnacc)
    plt.plot(X, dev_fnacc)
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")

    plt.subplot(3, 3, 5)
    plt.title('Program Accuracy')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_acc)
    plt.plot(X, dev_acc)
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")
    
    plt.subplot(3, 3, 8)
    plt.title('Grammar Accuracy')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_gracc)
    plt.plot(X, dev_gracc)
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")

    plt.subplot(3, 3, 3)
    plt.title('Parse-Action Precision')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_precision)
    plt.plot(X, dev_precision)
    plt.ylim(0, 100)
    plt.legend(["Train Precision", "Dev Precision"], loc="lower right")

    plt.subplot(3, 3, 6)
    plt.title('Parse-Action Recall')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_recall)
    plt.plot(X, dev_recall)
    plt.ylim(0, 100)
    plt.legend(["Train Recall", "Dev Recall"], loc="lower right")

    plt.subplot(3, 3, 9)
    plt.title('Parse-Action F1 Score')
    plt.xlim(1, len(train_eval_loss)+0.5)
    plt.xticks(X)
    plt.xlabel('Train Epoch')
    plt.plot(X, train_f1)
    plt.plot(X, dev_f1)
    plt.ylim(0, 100)
    plt.legend(["Train F1", "Dev F1"], loc="lower right")

#     plt.subplot(3, 2, 6)
#     plt.title('Recall')
#     plt.xlim(0, len(train_acc)+1)
#     plt.xlabel('Train Epoch')
#     plt.plot(X, train_recall)
#     plt.plot(X, dev_recall)
#     plt.ylim(0, 100)
#     plt.legend(["Train Recall", "Dev Recall"], loc="lower right")

    plt.tight_layout()
    plt.show()
    #plt.savefig('plot.png')

learning()
