import re
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train-tsv', default='train.tsv')
parser.add_argument('--test-tsv', default='test.tsv')
parser.add_argument('--dev-tsv', default='dev.tsv')
parser.add_argument('--workdir', default='./../workdir')
parser.add_argument('--dataset', default='./../dataset')
parser.add_argument('--grammar', default='./../workdir/thingpedia.json')
args = parser.parse_args()

WORKDIR = args.workdir
DATASET = args.dataset
DIR = os.path.join(DATASET, 't2t_data')

if not os.path.exists(DIR):
    os.makedirs(DIR)

grammar = ThingTalkGrammar(args.grammar, reverse=False)

split_input_and_labels(args.dev_tsv, 't2t_train_x', 't2t_train_y')
split_input_and_labels(args.dev_tsv, 't2t_test_x', 't2t_test_y')
split_input_and_labels(args.dev_tsv, 't2t_dev_x', 't2t_dev_y')

with open(os.path.join(DIR, 'all_words.txt'), 'w') as f:
    with open(os.path.join(WORKDIR, 'input_words.txt'), 'r') as read:
        lines = read.readlines()
        f.writelines(lines)
    with open(os.path.join(WORKDIR, 'output_words.txt'), 'r') as write:
        lines = write.readlines()
        f.writelines(lines)
    f.writelines(['UNK\n'])     # Need to manually add in UNK apparently.


def split_input_and_labels(filename, input_file, label_file):
    with open(os.path.join(DIR, input_file), 'w') as f1:
        with open(os.path.join(DIR, label_file), 'w') as f2:
            with open(os.path.join(DATASET, filename), 'r') as f:
                for line in f.readlines():
                    parts = re.split(r'\t+', line)
                    sentence = parts[1].split(' ')
                    program = parts[2]
                    vector, length = grammar.vectorize_program(sentence, program)
                    f1.write(' '.join(grammar.prediction_to_string(vector)) + '\n')
                    f2.write(parts[2] + '\n')
