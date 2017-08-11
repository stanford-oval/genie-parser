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

from util.seq2seq import Seq2SeqEvaluator
from util.loader import vectorize, unknown_tokens, load_data
from model import initialize

plt.rc('font', size=20)

def show_heatmap(x, y, attention):
    #print attention[:len(y),:len(x)]
    #print attention[:len(y),:len(x)].shape
    #data = np.transpose(attention[:len(y),:len(x)])
    data = attention[:len(y),:len(x)]
    x, y = y, x

    #ax = plt.axes(aspect=0.4)
    ax = plt.axes()
    heatmap = plt.pcolor(data, cmap=plt.cm.Blues)

    xticks = np.arange(len(y)) + 0.5
    xlabels = y
    yticks = np.arange(len(x)) + 0.5
    ylabels = x
    plt.xticks(xticks, xlabels, rotation='vertical')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    
    # make it look less like a scatter plot and more like a colored table
    ax.tick_params(axis='both', length=0)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    plt.colorbar(heatmap)
    
    plt.show()
    #plt.savefig('./attention-out.pdf')

def run():
    if len(sys.argv) < 6:
        print("** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Model: bagofwords/seq2seq>> <<Input Vocab>> <<Word Embeddings>> <<Model Directory>>")
        sys.exit(1)

    np.random.seed(42)
    benchmark = sys.argv[1]
    config, words, reverse, model = initialize(benchmark=benchmark, model_type=sys.argv[2], input_words=sys.argv[3], embedding_file=sys.argv[4]);
    model_dir = sys.argv[5]

    config.apply_cmdline(sys.argv[6:])
    
    print("unknown", unknown_tokens)

    histfile = ".history"
    try:
        readline.read_history_file(histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except IOError:
        pass
    atexit.register(readline.write_history_file, histfile)

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # Build the model and add the variable initializer Op
            model.capture_attention = True
            model.build()
            loader = tf.train.Saver()

            # Create a session for running Ops in the Graph
            with tf.Session() as sess:
                loader.restore(sess, os.path.join(model_dir, 'best'))
                
                try:
                    while True:
                        line = input('> ').strip()
                        if not line:
                            continue
                        
                        input, input_length = vectorize(line, words, config.max_length)
                        fake_input, fake_length = vectorize('ig to fb', words, config.max_length)
                        
                        feed = model.create_feed_dict([input, fake_input], [input_length, fake_length])
                        predictions, attention_scores = sess.run((model.pred, model.attention_scores), feed_dict=feed)
                        
                        assert len(predictions) == 2
                        assert len(attention_scores) == 2
                        
                        decoded = list(config.grammar.decode_output(predictions[0,0]))
                        try:
                            decoded = decoded[:decoded.index(config.grammar.end)]
                        except ValueError:
                            pass
                        output = [config.grammar.tokens[x] for x in decoded]
                        
                        print(' '.join(output))
                        
                        input = [reverse[x] for x in input[:input_length]]
                        
                        show_heatmap(input, output, attention_scores[0])
                except EOFError:
                    pass
            
if __name__ == '__main__':
    run()
