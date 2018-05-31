import re
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train-tsv', default='train.tsv')
parser.add_argument('--test-tsv', default='test.tsv')
parser.add_argument('--dev-tsv', default='dev.tsv')
args = parser.parse_args()

WORKDIR = './../workdir'
DATASET = './../dataset'
DIR = os.path.join(DATASET, 't2t_dir')

if not os.path.exists(DIR):
    os.makedirs(DIR)

with open(os.path.join(DIR, 't2t_train_x'), 'w') as f1:
    with open(os.path.join(DIR, 't2t_train_y'), 'w') as f2:
        with open(os.path.join(DATASET, args.train_tsv), 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])

with open(os.path.join(DIR, 't2t_test_x'), 'w') as f1:
    with open(os.path.join(DIR, 't2t_test_y'), 'w') as f2:
        with open(os.path.join(DATASET, args.test_tsv), 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])

with open(os.path.join(DIR, 't2t_dev_x'), 'w') as f1:
    with open(os.path.join(DIR, 't2t_dev_y'), 'w') as f2:
        with open(os.path.join(DATASET, args.dev_tsv), 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])

if not os.path.exists(os.path.join(DATASET, 't2t_data')):
    os.makedirs(os.path.join(DATASET, 't2t_data'))

with open(os.path.join(DATASET, 't2t_data/all_words.txt'), 'w') as f:
    with open(os.path.join(WORKDIR, 'input_words.txt'), 'r') as read:
        lines = read.readlines()
        f.writelines(lines)
    with open(os.path.join(WORKDIR, 'output_words.txt'), 'r') as write:
        lines = write.readlines()
        f.writelines(lines)
    f.writelines(['UNK\n'])
