from subprocess import call
import os
import argparse

HOME = os.path.expanduser('~')
DEFAULT_PROBLEM = 'parse_almond'
DEFAULT_MODEL = 'transformer'
DEFAULT_HPARAMS = 'transformer_base_single_gpu'
DEFAULT_TRAIN_DIR = 'workdir/t2t_train/' + DEFAULT_PROBLEM + '/' + \
        DEFAULT_MODEL + '-' + DEFAULT_HPARAMS
DEFAULT_DECODE_HPARAM = 'beam_size=10,alpha=0.6'

parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--problem', default=DEFAULT_PROBLEM)
parser.add_argument('--model', default=DEFAULT_MODEL)
parser.add_argument('--hparams-set', default=DEFAULT_HPARAMS)
parser.add_argument('--decode-hparams', default=DEFAULT_DECODE_HPARAM)
parser.add_argument('--problem-dir', default=os.path.join(HOME, 'almond-nnparser'))
parser.add_argument('--data-dir', default=os.path.join(HOME, 'dataset/t2t_data'))
parser.add_argument('--train-dir', default=os.path.join(HOME, DEFAULT_TRAIN_DIR))
args = parser.parse_args()

def exec_decode(decode_file):
    cmd = ' '.join(('t2t-decoder --t2t_usr_dir={} --data_dir={}',
                       '--problem={} --model={} --hparams_set={}',
                       '--output_dir={} --decode_hparams={}',
                       '--decode_from_file={}',
                       '--decode_to_file=translation.tt'))
    cmd = cmd.format(args.problem_dir, args.data_dir, args.problem,
            args.model, args.hparams_set, args.train_dir,
            args.decode_hparams, decode_file)
    call(cmd, shell=True)

if args.file is None:
    decode_file = args.data_dir + '/decode_this.txt'

    while(True):
        s = input("Input sentance to translate:  ")
        if s == 'done':
            break

        call('echo "{}" > {}'.format(s, decode_file), shell=True)
        exec_decode(decode_file)

        print('\n\n\n\n Input sentence: \t' + s)
        call('cat translation.tt', shell=True)

else:
    decode_file = args.file
    exec_decode(decode_file)

