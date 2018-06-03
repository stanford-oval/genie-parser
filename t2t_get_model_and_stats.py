import re
import csv
from shutil import copyfile
import os
import argparse

LABELS = ['step', 'loss', 'accuracy', 'sequence accuracy', 'top 5 accuracy', 'approximate bleu score', 'neg log perplexity', 'rouge 2 fscore', 'rouge L fscore']

parser = argparse.ArgumentParser()
parser.add_argument("--clean", action='store_true')
parser.add_argument("--print-only", action='store_true')
parser.add_argument("--rank-seq-acc", action='store_true')
parser.add_argument("--model-dir", default='/home/gcampagn/workdir/t2t_train/parse_almond/transformer-transformer_base_single_gpu')
parser.add_argument("--file", default='transformer_output.txt')
parser.add_argument("--outdir", default='/home/gcampagn/workdir/model_and_stats/')
parser.add_argument("--outfile", default='transformer_output.csv')
args = parser.parse_args()

if not os.path.exists(args.outdir) and not args.print_only:
    os.makedirs(args.outdir)

with open(args.file, 'r') as f:
    output = f.read()
    if not args.print_only:
        if args.clean:
            os.rename(args.file, os.path.join(args.outdir, args.file))
        else:
            copyfile(args.file, os.path.join(args.outdir, args.file))

pattern  = re.compile("INFO:tensorflow:Saving dict for global step.+?(?=INFO)", re.DOTALL)
best_acc = 0
best_step = -1
best_stats = None

if not args.print_only:
    f = open(os.path.join(args.outdir, args.outfile), 'w')
    writer = csv.writer(f)
    writer.writerow(LABELS)

print('\t'.join(LABELS))
for match in re.findall(pattern, output):
    step = re.search("(?<=global_step = ).+?(?=,)", match, re.DOTALL).group(0)
    loss = re.search("(?<=loss = ).+?(?=,)", match, re.DOTALL).group(0)
    acc = re.search("(?<=accuracy = ).+?(?=,)", match, re.DOTALL).group(0)
    seq_acc = re.search("(?<=accuracy_per_sequence = ).+?(?=,)", match, re.DOTALL).group(0)
    if args.rank_seq_acc:
        if float(seq_acc) > best_acc:
            best_acc = float(seq_acc)
            best_step = step
    else:
        if float(acc) > best_acc:
            best_acc = float(acc)
            best_step = step

    acc_t5 = re.search("(?<=accuracy_top5 = ).+?(?=,)", match, re.DOTALL).group(0)
    bleu = re.search("(?<=bleu_score = ).+?(?=,)", match, re.DOTALL).group(0)
    perp = re.search("(?<=perplexity = ).+?(?=,)", match, re.DOTALL).group(0)
    rouge_f2 = re.search("(?<=rouge_2_fscore = ).+?(?=,)", match, re.DOTALL).group(0)
    rouge_fL = re.search("(?<=rouge_L_fscore = ).*", match, re.DOTALL).group(0)
    row = [int(step), float(loss), float(acc), float(seq_acc), float(acc_t5), float(bleu), float(perp), float(rouge_f2), float(rouge_fL)]
    row = [str(num) for num in row]
    print('\t'.join(row).strip())
    if not args.print_only:
        writer.writerow(row)

if not args.print_only:
    f.close()

if best_step != -1:
    print('Best step was {} with accuracy {}'.format(best_step, best_acc))
    if args.clean and not args.print_only:
        model_pattern = 'model.ckpt-'
        pattern = model_pattern + best_step + '\.'
        DIR = args.model_dir
        for f in os.listdir(DIR):
            if not re.search(pattern, f) and re.search(model_pattern, f):
                file_path = os.path.join(DIR, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            else:
                os.rename(os.path.join(DIR, f), os.path.join(args.outdir, f))

        print('In out directory, ', args.outdir)
        for f in os.listdir(args.outdir):
            print(f)
