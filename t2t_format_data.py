import re

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
        with open('../dataset/dev2.tsv', 'r') as f:
            for line in f.readlines():
                parts = re.split(r'\t+', line)
                f1.write(parts[1] + '\n')
                f2.write(parts[2])


'''
with open('./../workdir/all_words.txt', 'r') as f1:
    with open('./../workdir/all_words_final.txt', 'w') as f2:
        for line in f1.readlines():
            parts = re.split(' ', line)
            f2.write(parts[-1])
'''
