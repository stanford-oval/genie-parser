import re
from tqdm import tqdm
import csv
from subprocess import call

labels = ['model', 'step', 'loss', 'accuracy', 'sequence accuracy', 'top 5 accuracy', 'approximate bleu score', 'neg log perplexity', 'rouge 2 fscore', 'rouge L fscore']

with open('transformerOutput.txt', 'r') as f:
	output = f.read()

pattern  = re.compile("INFO:tensorflow:Saving dict for global step.+?(?=INFO)", re.DOTALL)
best_acc = 0
best_step = -1

with open('transformer_data.csv', 'ab') as f:
	writer = csv.writer(f)
	for match in re.findall(pattern, output):
		step = re.search("(?<=global_step = ).+?(?=,)", match, re.DOTALL).group(0)
		loss = re.search("(?<=loss = ).+?(?=,)", match, re.DOTALL).group(0)
		acc = re.search("(?<=accuracy = ).+?(?=,)", match, re.DOTALL).group(0)
		if(acc > best_acc):
			best_acc = acc
			best_step = step
		seq_acc = re.search("(?<=accuracy_per_sequence = ).+?(?=,)", match, re.DOTALL).group(0)
		acc_t5 = re.search("(?<=accuracy_top5 = ).+?(?=,)", match, re.DOTALL).group(0)
		bleu = re.search("(?<=bleu_score = ).+?(?=,)", match, re.DOTALL).group(0)
		perp = re.search("(?<=perplexity = ).+?(?=,)", match, re.DOTALL).group(0)
		rouge_f2 = 	re.search("(?<=rouge_2_fscore = ).+?(?=,)", match, re.DOTALL).group(0)
		rouge_fL = re.search("(?<=rouge_L_fscore = ).*", match, re.DOTALL).group(0)
		writer.writerow(["test", step, loss, acc, seq_acc, acc_t5, bleu, perp, rouge_f2, rouge_fL])



call('rm ~/t2t_data/parse_almond/transformer-transformer_base_single_gpu/ model.ckpt-[!' + best_step + ']*')

