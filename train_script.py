import os
from subprocess import call
from subprocess import check_output
from os.path import join

HOME = os.path.expanduser('~')

call("rm -rf workdir/t2t_results", shell=True)

os.chdir("almond-nnparser")

print("formatting data")
## format almond data
call('python3 t2t_format_data.py --train-tsv train.tsv --dev-tsv paraphrasing-dev.tsv --test-tsv paraphrasing-test.tsv', shell=True)

## generate t2t data
print("generating t2t data")
call('python3 t2t_transformer.py --datagen', shell=True)

## train model
print("training t2t model")
call("python3 t2t_transformer.py --hparams-set transformer_tiny --hparams 'batch_size=2048' --train-steps 60000", shell=True) 

## get best checkpoint
print("finding best checkpoint")
best_ckpt = check_output("python3 t2t_get_model_and_stats.py --clean --model-dir transformer_tiny --get_best_ckpt", shell=True)
best_ckpt = int(str(best_ckpt, 'utf-8'))

## decode test file
print("decoding test file")
call("python3 t2t_decode.py --file ../dataset/t2t_data/t2t_test_x --hparams-set transformer_tiny --ckpt_path ../workdir/t2t_results/model.ckpt-" + str(best_ckpt), shell=True)

## eval
print("evaluating")
#call("cd ../workdir", shell=True)
os.chdir(HOME + "/workdir")
call("python3 ../almond-nnparser/t2t_eval.py --pre-t2t-data ../dataset/paraphrasing-test.tsv", shell=True)

