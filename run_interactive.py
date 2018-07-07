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
'''
Created on Mar 22, 2017

@author: gcampagn
'''

import os
import sys
import numpy as np
import tensorflow as tf

import readline, atexit

import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

from models import Config, create_model
from util.loader import vectorize

from tensorflow.python import debug as tf_debug

#plt.rc('font', size=20)

def show_heatmap(x, y, attention):
    #print attention[:len(y),:len(x)]
    #print attention[:len(y),:len(x)].shape
    #data = np.transpose(attention[:len(y),:len(x)])
    data = attention[:len(y),:len(x)].T

    #ax = plt.axes(aspect=0.4)
    ax = plt.axes()
    heatmap = plt.pcolor(data, cmap=plt.cm.Blues)

    xticks = np.arange(len(y)) + 0.5
    xlabels = y
    yticks = np.arange(len(x)) + 0.5
    ylabels = x
    plt.xticks(xticks, xlabels)#, rotation='vertical')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    
    # make it look less like a scatter plot and more like a colored table
    ax.tick_params(axis='both', length=0)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    plt.colorbar(heatmap)
    
    #plt.show()
    plt.savefig('./attention-out.pdf')

def run():
    if len(sys.argv) < 2:
        print("** Usage: python3 " + sys.argv[0] + " <<Model Directory>>")
        sys.exit(1)

    np.random.seed(42)
    tf.set_random_seed(1234)
    
    model_dir = sys.argv[1]
    config = Config.load(['./default.conf', os.path.join(model_dir, 'model.conf')])
    model = create_model(config)
    
    histfile = ".almondnn_history"
    try:
        readline.read_history_file(histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except IOError:
        pass
    atexit.register(readline.write_history_file, histfile)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            model.build()
            loader = tf.train.Saver()

            # Create a session for running Ops in the Graph
            with tf.Session() as sess:
                loader.restore(sess, os.path.join(model_dir, 'best'))

                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                
                try:
                    while True:
                        line = input('> ').strip()
                        if not line:
                            continue
                        
                        sentence, sentence_length = vectorize(line, config.dictionary, config.max_length, add_eos=True, add_start=True)
                        print('Vectorized', sentence, sentence_length)
                        fake_input, fake_length = vectorize('ig to fb', config.dictionary, config.max_length, add_eos=True, add_start=True)
                        fake_parse = np.zeros((2*config.max_length-1,))
                        
                        feed = model.create_feed_dict([sentence, fake_input],
                                                      [sentence_length, fake_length],
                                                      [fake_parse, fake_parse])
                        predictions, attention_scores = sess.run((model.preds, model.attention_scores), feed_dict=feed)


                        prediction = dict()
                        for key in predictions:
                            prediction[key] = predictions[key][0,0]
                        primary_prediction = prediction[config.grammar.primary_output]
                        index, = np.where(primary_prediction == config.grammar.end)
                        if len(index):
                            for key in predictions:
                                prediction[key] = prediction[key][:index[0]+1]
                            primary_prediction = primary_prediction[:index[0]+1]
                        for key in prediction:
                            print(key, '=>', prediction[key])
                        config.grammar.print_prediction(sentence, prediction)
                        if len(predictions[config.grammar.primary_output][0]) == 1:
                            try:
                                print('predicted', ' '.join(config.grammar.reconstruct_program(sentence, prediction)))
                            except (KeyError, TypeError, IndexError, ValueError):
                                print('failed to predict')
                        else:
                            for i, beam in enumerate(predictions[config.grammar.primary_output][0]):
                                beam_prediction = dict()
                                for key in predictions:
                                    beam_prediction[key] = predictions[key][0,i]
                                try:
                                    print('beam', i, ' '.join(config.grammar.reconstruct_program(sentence, beam_prediction)))
                                except (KeyError, TypeError, IndexError, ValueError):
                                    print('beam', i, 'failed to predict')

                        sentence = list(config.reverse_dictionary[x] for x in sentence[:sentence_length])
                        # show_heatmap(sentence, config.grammar.prediction_to_string(prediction), attention_scores[0])
                except EOFError:
                    pass
            
if __name__ == '__main__':
    run()
