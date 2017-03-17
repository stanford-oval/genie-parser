
import sys
import numpy as np
import tensorflow as tf

from util.seq2seq import Seq2SeqEvaluator
from util.trainer import Trainer

from util.loader import unknown_tokens, load_data
from model import LSTMAligner, initialize

def run():
    if len(sys.argv) < 5:
        print "** Usage: python " + sys.argv[0] + " <<Benchmark: tt/geo>> <<Input Vocab>> <<Word Embeddings>> <<Train Set> [<<Dev Set>>]"
        sys.exit(1)

    np.random.seed(42)
    config, words, reverse, embeddings_matrix = initialize(benchmark=sys.argv[1], input_words=sys.argv[2], embedding_file=sys.argv[3]);
    
    train_data = load_data(sys.argv[4], words, config.grammar.dictionary,
                           reverse, config.grammar.tokens,
                           config.max_length)
    if len(sys.argv) > 5:
        dev_data = load_data(sys.argv[5], words, config.grammar.dictionary,
                             reverse, config.grammar.tokens,
                             config.max_length)
    else:
        dev_data = None
    print "unknown", unknown_tokens
    

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = LSTMAligner(config, embeddings_matrix)
        init = tf.global_variables_initializer()
        
        train_eval = Seq2SeqEvaluator(model, config.grammar, train_data, 'train', batch_size=config.batch_size)
        dev_eval = Seq2SeqEvaluator(model, config.grammar, dev_data, 'dev', batch_size=config.batch_size)
        trainer = Trainer(model, train_data, train_eval, dev_eval,
                          max_length=config.max_length,
                          batch_size=config.batch_size,
                          n_epochs=config.n_epochs,
                          dropout=config.dropout)

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            trainer.fit(sess)
            
            print "Final result"
            train_eval.eval(sess, save_to_file=True)
            if dev_data is not None:
                dev_eval.eval(sess, save_to_file=True)

if __name__ == "__main__":
    run()
