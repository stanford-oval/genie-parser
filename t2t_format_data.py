import re
import argparse
import os
from grammar.thingtalk import ThingTalkGrammar

HOME = os.path.expanduser('~')

parser = argparse.ArgumentParser()
parser.add_argument('--train-tsv', default='train.tsv')
parser.add_argument('--test-tsv', default='test.tsv')
parser.add_argument('--dev-tsv', default='dev.tsv')
parser.add_argument('--workdir', default=os.path.join(HOME, 'workdir'))
parser.add_argument('--dataset', default=os.path.join(HOME, 'dataset'))
parser.add_argument('--grammar', default=os.path.join(HOME, 'workdir/en/thingpedia.json'))
args = parser.parse_args()

WORKDIR = args.workdir
DATASET = args.dataset
DIR = os.path.join(DATASET, 't2t_data')

def split_input_and_labels(filename, input_file, label_file):
    with open(os.path.join(DIR, input_file), 'w') as f1:
        with open(os.path.join(DIR, label_file), 'w') as f2:
            with open(os.path.join(DATASET, filename), 'r') as f:
                for line in f.readlines():
                    parts = re.split(r'\t+', line)
                    sentence = parts[1].split(' ')
                    program = parts[2].strip()
                    prog_vector, length = grammar.vectorize_program(sentence, program)
                    prog_vector['actions'] = prog_vector['actions'][:length-1]
                    prog_vector = grammar.prediction_to_string(prog_vector)
                    f1.write(parts[1] + '\n')
                    f2.write(' '.join(prog_vector) + '\n')

    print('done splitting {}'.format(filename))

if not os.path.exists(DIR):
    os.makedirs(DIR)

grammar = ThingTalkGrammar(args.grammar, reverse=False)

split_input_and_labels(args.train_tsv, 't2t_train_x', 't2t_train_y')
split_input_and_labels(args.test_tsv, 't2t_test_x', 't2t_test_y')
split_input_and_labels(args.dev_tsv, 't2t_dev_x', 't2t_dev_y')

with open(os.path.join(DIR, 'all_words.txt'), 'w') as f:
    f.write('<pad>\n<EOS>\n')   # does this work?

    with open(os.path.join(WORKDIR, '/en/input_words.txt'), 'r') as read:
        lines = read.readlines()
        f.writelines(lines)
    # with open(os.path.join(WORKDIR, 'output_words.txt'), 'r') as write:
    #     lines = write.readlines()
    #     f.writelines(lines)
    num_actions = grammar.num_control_tokens + grammar._parser.num_rules
    num_actions += len(grammar._copy_terminals) + len(grammar._extensible_terminals)

    actions = list(range(grammar.num_control_tokens, num_actions))
    grammarized_actions = grammar.prediction_to_string({'actions': actions})

    for token in grammarized_actions:
        f.write(token + '\n')
    f.write('UNK\n') # Need to manually add in UNK apparently.
