from subprocess import call
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file')
args = parser.parse_args()

problem = 'parse_almond'
model = 'transformer'
hparams = 'transformer_base_single_gpu'
problem_dir = '~/almond-nnparser/'
data_dir = '../dataset/t2t_data'
tmp_dir = '/tmp/t2t_datagen'
train_dir = '~/workdir/t2t_train/' + problem +'/' + model + '-' + hparams

beam_size = '10'
alpha = '0.6'

if args.file is None:
    decode_file = data_dir + '/decode_this.txt'

    while(True):
        s = input("Input sentance to translate:  ")
        if(s == 'done'):
            break

        call('echo \"' + s + '\" > ' + decode_file, shell=True)
        call('t2t-decoder --t2t_usr_dir=' + problem_dir + ' --data_dir=' + data_dir + \
                ' --problem=' + problem + ' --model=' + model + ' --hparams_set=' + \
                hparams + ' --output_dir=' + train_dir + ' --decode_hparams=\"beam_size=' \
                + beam_size + ',alpha=' + alpha +'\" --decode_from_file=' + decode_file + \
                ' --decode_to_file=translation.tt', shell=True)

        print('\n\n\n\n Input sentence: \t' + s)
        call('cat translation.tt', shell=True)

else:
    decode_file = args.file
    call('t2t-decoder --t2t_usr_dir=' + problem_dir + ' --data_dir=' + data_dir + \
            ' --problem=' + problem + ' --model=' + model + ' --hparams_set=' + \
            hparams + ' --output_dir=' + train_dir + ' --decode_hparams=\"beam_size=' \
            + beam_size + ',alpha=' + alpha +'\" --decode_from_file=' + decode_file + \
            ' --decode_to_file=translation.tt', shell=True)
