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
        data = np.array(json.load(fp), dtype=np.float32)
    
    loss = data[:,0]
    train_acc = 100*data[:,1]
    dev_acc = 100*data[:,2]

    dev_mov_avg = movingaverage(dev_acc, 3)
    
    X = 1 + np.arange(len(data))
    plt.xlim(0, len(data)+1)
    
    #plt.plot(X, loss)
    #plt.ylabel('Loss')
    plt.xlabel('Training epoch', fontsize=20)
    
    #plt.gca().twinx()
    plt.plot(X, train_acc)
    plt.plot(X, dev_acc)
    plt.plot(X[1:-1], dev_mov_avg, '--')
    #plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.legend(["Train Accuracy", "Dev Accuracy"], loc="lower right")
    plt.show()

learning()
