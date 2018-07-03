import argparse
import os
from subprocess import call
from os.path import join

HOME = os.path.expanduser('~')
DEFAULT_PROBLEM = 'parse_almond'
DEFAULT_MODEL = 'transformer'
DEFAULT_HPARAMS = 'transformer_base_single_gpu'

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default=DEFAULT_PROBLEM)
parser.add_argument('--model', default=DEFAULT_MODEL)
parser.add_argument('--hparams-set', default=DEFAULT_HPARAMS)
parser.add_argument('--hparams', default='batch_size=512')
parser.add_argument('--problem-dir', default=join(HOME, 'almond-nnparser-private'))
parser.add_argument('--data-dir', default=join(HOME, 'dataset/t2t_data'))
parser.add_argument('--tmp-dir', default='/tmp/t2t_datagen')
parser.add_argument('--train-steps', default=10000)
parser.add_argument('--eval-steps', default=200)
parser.add_argument('--train-dir')
parser.add_argument('--datagen', action='store_true')
args = parser.parse_args()

if args.train_dir is None:
    train_dir = 'workdir/t2t_train/{}/{}-{}'.format(args.problem, args.model, args.hparams_set)
    args.train_dir = join(HOME, train_dir)

call("mkdir -p {} {} {}".format(args.data_dir, args.tmp_dir, args.train_dir), shell=True)

if args.datagen:
    # Generate data
    cmd = "t2t-datagen --t2t_usr_dir={} --data_dir={} --tmp_dir={} --problem={}"
    cmd = cmd.format(args.problem_dir, args.data_dir, args.tmp_dir, args.problem)
    call(cmd, shell=True)
else:
    # Train
    cmd = ' '.join((
        "t2t-trainer --t2t_usr_dir={} --data_dir={} --model={}",
        "--train_steps={} --eval_steps={} --hparams_set={}",
        "--output_dir={} --problem={} --hparams={}",
        "&> transformer_output.txt",
    ))

    cmd = cmd.format(args.problem_dir, args.data_dir, args.model,
            args.train_steps, args.eval_steps, args.hparams_set, args.train_dir,
            args.problem, args.hparams)
    print(cmd)
    call(cmd, shell=True)
