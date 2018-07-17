import re
import argparse
import os
from grammar.simple import SimpleGrammar
from models import Config


HOME = os.path.expanduser('~')

parser = argparse.ArgumentParser()
parser.add_argument('--train-tsv', action='append')
parser.add_argument('--test-tsv', default='test.tsv')
parser.add_argument('--dev-tsv', default='dev.tsv')
parser.add_argument('--curriculum', action='store_true')
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

model_conf = os.path.join(WORKDIR, 'en/model/model.conf')
config = Config.load(['./default.conf', model_conf])

grammar = config.grammar

train_file_lists = args.train_tsv
curriculum = args.curriculum

if curriculum and len(train_data_lists) < 2:
    raise ValueError('Must have exactly two training sets for curriculum learning')

for file in train_file_lists:
    key = os.path.basename(file)
    key = key[:key.rindex('.')]
    if not curriculum:
        key = ''
    split_input_and_labels(file, 't2t_train' + key + '_x', 't2t_train' + key + '_y')

split_input_and_labels(args.test_tsv, 't2t_test_x', 't2t_test_y')
split_input_and_labels(args.dev_tsv, 't2t_dev_x', 't2t_dev_y')

with open(os.path.join(DIR, 'all_words.txt'), 'w') as f:
    f.write('<pad>\n<EOS>\n')   # does this work?

    with open(os.path.join(WORKDIR, 'en/input_words.txt'), 'r') as read:
        lines = read.readlines()
        f.writelines(lines)
    # with open(os.path.join(WORKDIR, 'output_words.txt'), 'r') as write:
    #     lines = write.readlines()
    #     f.writelines(lines)
    if hasattr(grammar, '_parser'):
        num_actions = grammar.num_control_tokens + grammar._parser.num_rules
    else:
        num_actions = grammar.output_size[grammar.primary_output]
    if hasattr(grammar, '_copy_terminals') and hasattr(grammar, '_extensible_terminals'):
        num_actions += len(grammar._copy_terminals) + len(grammar._extensible_terminals)

    actions = list(range(grammar.num_control_tokens, num_actions))
    grammarized_actions = grammar.prediction_to_string({grammar.primary_output: actions})

    for token in grammarized_actions:
        f.write(token + '\n')
    f.write('<unk>\n') # Need to manually add in UNK apparently.
    f.write('UNK\n')  # Need to manually add in UNK apparently.
