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

def show_heatmap(x, y, attention):
    #print attention[:len(y),:len(x)]
    #print attention[:len(y),:len(x)].shape
    data = np.transpose(attention[:len(y),:len(x)])
    
    #ax = plt.axes(aspect=0.4)
    ax = plt.axes()
    heatmap = plt.pcolor(data, cmap=plt.cm.Blues)

    xticks = np.arange(len(y)) + 0.5
    xlabels = y
    yticks = np.arange(len(x)) + 0.5
    ylabels = x
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    
    # make it look less like a scatter plot and more like a colored table
    ax.tick_params(axis='both', length=0)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    plt.colorbar(heatmap)
    
    plt.show()

def run():
    if len(sys.argv) < 6:
        print "** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Model: bagofwords/seq2seq>> <<Input Vocab>> <<Word Embeddings>> <<Model Directory>>"
        sys.exit(1)

    np.random.seed(42)
    benchmark = sys.argv[1]
    config, words, reverse, model = initialize(benchmark=benchmark, model_type=sys.argv[2], input_words=sys.argv[3], embedding_file=sys.argv[4]);
    model_dir = sys.argv[5]

    config.apply_cmdline(sys.argv[6:])
    
    print "unknown", unknown_tokens

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
                        line = raw_input('> ').strip()
                        if not line:
                            continue
                        
                        input, input_length = vectorize(line, words, config.max_length)
                        fake_input, fake_length = vectorize('ig to fb', words, config.max_length)
                        
                        feed = model.create_feed_dict([input, fake_input], [input_length, fake_length])
                        predictions, attention_scores = sess.run((model.pred, model.attention_scores), feed_dict=feed)
                        
                        assert len(predictions) == 2
                        assert len(attention_scores) == 2
                        
                        decoded = list(config.grammar.decode_output(predictions[0]))
                        try:
                            decoded = decoded[:decoded.index(config.grammar.end)]
                        except ValueError:
                            pass
                        output = map(lambda x:config.grammar.tokens[x], decoded)
                        
                        print ' '.join(output)
                        
                        input = map(lambda x:reverse[x],input[:input_length])
                        
                        show_heatmap(input, output, attention_scores[0])
                except EOFError:
                    pass
            
if __name__ == '__main__':
    run()
