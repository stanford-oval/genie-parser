import re
import csv
import os

labels = ['step', 'loss', 'accuracy', 'sequence accuracy', 'top 5 accuracy', 'approximate bleu score', 'neg log perplexity', 'rouge 2 fscore', 'rouge L fscore']

with open('transformerOutput.txt', 'r') as f:
    output = f.read()

pattern  = re.compile("INFO:tensorflow:Saving dict for global step.+?(?=INFO)", re.DOTALL)
best_acc = 0
best_step = -1

with open('transformer_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(labels)
    for match in re.findall(pattern, output):
        step = re.search("(?<=global_step = ).+?(?=,)", match, re.DOTALL).group(0)
        loss = re.search("(?<=loss = ).+?(?=,)", match, re.DOTALL).group(0)
        acc = re.search("(?<=accuracy = ).+?(?=,)", match, re.DOTALL).group(0)
        if float(acc) > best_acc:
            best_acc = float(acc)
            best_step = step
        seq_acc = re.search("(?<=accuracy_per_sequence = ).+?(?=,)", match, re.DOTALL).group(0)
        acc_t5 = re.search("(?<=accuracy_top5 = ).+?(?=,)", match, re.DOTALL).group(0)
        bleu = re.search("(?<=bleu_score = ).+?(?=,)", match, re.DOTALL).group(0)
        perp = re.search("(?<=perplexity = ).+?(?=,)", match, re.DOTALL).group(0)
        rouge_f2 = re.search("(?<=rouge_2_fscore = ).+?(?=,)", match, re.DOTALL).group(0)
        rouge_fL = re.search("(?<=rouge_L_fscore = ).*", match, re.DOTALL).group(0)
        row = [int(step), float(loss), float(acc), float(seq_acc), float(acc_t5), float(bleu), float(perp), float(rouge_f2), float(rouge_fL)]
        print(row)
        writer.writerow(row)

if best_step != -1:
    pattern = 'model.ckpt-' + best_step + '\.'
    DIR = '/home/gcampagn/t2t_train/parse_almond_test/transformer-transformer_base_single_gpu'
    for f in os.listdir(DIR):
        if not re.search(pattern, f):
            file_path = os.path.join(DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print('Removing ', file_path)
