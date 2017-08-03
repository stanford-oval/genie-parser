#!/usr/bin/python3

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
    if isinstance(data, list):
        # old style
        data = np.array(data)
        loss = data[:,0]
        train_acc = 100*data[:,1]
        dev_acc = 100*data[:,2]
        grad_norm = None
    else:
        loss = np.array(data['loss'])
        accuracy = np.array(data['accuracy'])
        train_acc = 100*accuracy[:,0]
        dev_acc = 100*accuracy[:,1]
        grad_norm = np.array(data['grad'])

    dev_mov_avg = movingaverage(dev_acc, 3)

    plt.title(sys.argv[1])
    plt.subplot(3, 1, 1)
    X = 1 + np.arange(len(train_acc))
    plt.xlim(0, len(train_acc)+1)

    plt.xlabel('Train Epoch')

    plt.plot(X, train_acc)
    plt.plot(X, dev_acc)
    if len(dev_mov_avg) > 2:
        plt.plot(X[1:-1], dev_mov_avg, '--')
    plt.ylim(0, 100)
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")

    plt.subplot(3, 1, 2)
    X = np.arange(len(loss))
    plt.xlabel('Minibatch #')
    plt.plot(X, loss)
    plt.legend(["Loss"], loc="upper right")

    if grad_norm is not None:
        plt.subplot(3, 1, 3)
        X = np.arange(len(loss))
        plt.xlabel('Minibatch #')
        plt.plot(X, grad_norm)
        plt.legend(["Gradient"], loc='upper right')

    plt.show()

learning()
