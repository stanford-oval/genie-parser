import re
import os

directory = './../dataset/t2t_dir'
if not os.path.exists(directory):
    os.makedirs(directory)

with open('./../dataset/t2t_dir/t2t_train_x', 'w') as f1:
    with open('./../dataset/t2t_dir/t2t_train_y', 'w') as f2:
        with open('./../dataset/train.tsv', 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])

with open('../dataset/t2t_dir/t2t_test_x', 'w') as f1:
    with open('../dataset/t2t_dir/t2t_test_y', 'w') as f2:
        with open('../dataset/test.tsv', 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])

with open('../dataset/t2t_dir/t2t_dev_x', 'w') as f1:
    with open('../dataset/t2t_dir/t2t_dev_y', 'w') as f2:
        with open('../dataset/t2t-dev.tsv', 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])
