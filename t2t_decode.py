from subprocess import call
import os
from grammar.thingtalk import ThingTalkGrammar
import argparse

HOME = os.path.expanduser('~')
DEFAULT_PROBLEM = 'parse_almond'
DEFAULT_MODEL = 'transformer'
DEFAULT_HPARAMS = 'transformer_base_single_gpu'
DEFAULT_DECODE_HPARAM = 'beam_size=10,alpha=0.6'

parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--problem', default=DEFAULT_PROBLEM)
parser.add_argument('--model', default=DEFAULT_MODEL)
parser.add_argument('--hparams-set', default=DEFAULT_HPARAMS)
parser.add_argument('--decode-hparams', default=DEFAULT_DECODE_HPARAM)
parser.add_argument('--problem-dir', default=os.path.join(HOME, 'almond-nnparser'))
parser.add_argument('--data-dir', default=os.path.join(HOME, 'dataset/t2t_data'))
parser.add_argument('--grammar', default=os.path.join(HOME, 'workdir/thingpedia.json'))
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--outfile', default=os.path.join(HOME, 'workdir/t2t_results/translation.tt'))
args = parser.parse_args()

grammar = ThingTalkGrammar(args.grammar, reverse=False)

def exec_decode(decode_file):
    cmd = ' '.join(('t2t-decoder --t2t_usr_dir={} --data_dir={}',
                       '--problem={} --model={} --hparams_set={}',
                       '--checkpoint_path={} --decode_hparams={}',
                       '--decode_from_file={} --decode_to_file={}'))
    cmd = cmd.format(args.problem_dir, args.data_dir, args.problem,
            args.model, args.hparams_set, args.ckpt_path,
            args.decode_hparams, decode_file, args.outfile + '.tmp')
    call(cmd, shell=True)

    # Need to convert grammarized output to actual words
    with open(args.outfile, 'w') as out, open(decode_file, 'r') as inputs:
        with open(args.outfile + '.tmp', 'r') as grammarized:
            for inp, grammarized in zip(inputs, grammarized):
                inp = inp.strip().split()
                grammarized = grammarized.strip().split()

                actions = grammar.string_to_prediction(grammarized)  # get back actions vector
                reconstructed = grammar.reconstruct_program(inp, {'actions': actions}, ignore_errors=True)
                out.write(' '.join(reconstructed) + '\n')

    os.remove(args.outfile + '.tmp')

print('WARNING: does not support copy or extensible terminals. Will write out a blank prediction.')

if args.file is None:
    decode_file = '/tmp/t2t_decode.txt'

    while(True):
        s = input("Input sentence to translate:  ")
        if s == 'done':
            break

        call('echo "{}" > {}'.format(s, decode_file), shell=True)
        exec_decode(decode_file)

        print('\n\n\n Input sentence: \t' + s)
        call('cat {}'.format(args.outfile), shell=True)
        print('\n')

else:
    exec_decode(args.file)
