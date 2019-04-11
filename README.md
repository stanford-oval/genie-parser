**This repository is obsolete**

It was used in the submission, *only the submission*, of the paper: _Genie: A Generator of Natural Language Semantic Parsers for Virtual Assistants Commands_.
The model used in the final version lives instead in [genie-toolkit](https://github.com/stanford-oval/genie-toolkit) and [decaNLP](https://github.com/stanford-oval/genie-toolkit).

The original README follows.

# Genie-parser

A neural-network based semantic parser, designed to be used in conjuction with
[genie-toolkit](https://github.com/Stanford-Mobisocial-IoT-Lab/genie-toolkit),
a set of tools to generate large scale semantic parsing datasets quickly.

Genie was described in the paper:

_Genie: A Generator of Natural Language Parsers for Compositional Virtual Assistants_  
Giovanni Campagna (\*), Silei Xu (\*), Mehrad Moradshahi, and Monica S. Lam  
Conditionally accepted to _Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation_ (PLDI 2019), Phoenix, AZ, June 2019.

Genie-parser is part of Almond, a research project of the Mobile and Social
Computing Lab at Stanford University.  You can find more
information at <http://almond.stanford.edu/>.

## Installation

genie-parser depends on numpy, Tensorflow 1.12 for Python 3, and a number of
other python modules. To install the dependencies, use:

    pip3 install -r requirements.txt

It's recommended to install numpy from distribution packages, not
pip because it's faster and more reliable.

You must also install `tensor2tensor` using our own fork, using the version
indicated in requirements.txt.
Do not install tensor2tensor from pypi, as that is not compatible.

Please see the Tensorflow documentation if you wish to use GPU
acceleration. You'll need to install nvidia+cuda or amdgpu-pro+rocm
drivers, and then install "tensorflow-gpu" from pip, or build
Tensorflow from source.

genie-parser has been tested successfully on Fedora 25 to 28
x86_64 with CPU and Nvidia GPU acceleration, as well as Ubuntu 16.04,
18.04. On RHEL/CentOS 7, you will need to use python 3.6 from the `rh-python36`
software collection. On Fedora 29 and later, the default is python 3.7, which is not
compatible with Tensorflow, so you will need to install python 3.6 separately.

It is also possible to use the [pipenv](https://pipenv.readthedocs.io/en/latest/)
tool to set up a virtualenv and install the dependencies inside it.

## Training

To train a new model, you should do the follow:

1. Unpack the dataset (*.tsv files) into a `dataset/` directory.

2. Download the word embeddings. We recommend using Glove 42B 300-dimensional,
   which can be downloaded from <https://nlp.stanford.edu/data/glove.42B.300d.zip>,
   or from our mirror at <https://oval.cs.stanford.edu/data/glove/glove.42B.300d.zip>.
   Set the `GLOVE` environment variable to the path of uncompressed text file.
   
   You can skip this step, in which case, the `luinet-datagen` script will download
   the recommended GloVe file automatically.
 
3. Prepare the working directory:
   ```
   luinet-datagen --src_data_dir ./dataset --data_dir ./workdir --thingpedia_snapshot [SNAPSHOT]
      --problem semparse_thingtalk_noquote
   ```
   This script computes the input dictionary, downloads a snapshot of Thingpedia,
   and computes a subset of the word embedding matrix corresponding to the dictionary.
   
   Use the optional SNAPSHOT argument to choose which Thingpedia snapshot to train
   against, or pass -1 for the latest content of Thingpedia. Be aware that
   using -1 might make the results impossible to reproduce.
   This script will verify that the dataset is compatible with the
   Thingpedia snapshot.
   
   The `--problem` parameter should be used to chosen to match the target language.
   The meaning is similar to that in the [tensor2tensor](https://github.com/Stanford-Mobisocial-IoT-Lab/tensor2tensor)
   library. See [genieparser/tasks/__init__.py](genieparser/tasks/__init__.py) for a
   list of available problems.
   
4. Train:
   ```
   genie-trainer --data_dir ./workdir --output_dir ./workdir/model
     --model genie_copy_seq2seq
     --hparams_set 'lstm_genie'
     --hparams_overrides ''
     --decode_hparams 'beam_size=20,return_beams=true'
     --problem 'semparse_thingtalk_noquote'
     --eval_early_stopping_metric 'metrics-semparse_thingtalk_noquote/accuracy'
     --noeval_early_stopping_metric_minimize
   ```
   See `luinet-trainer --help` for a description of each options. Available hparams sets are
   in [genieparser/layers/hparams.py](genieparser/layers/hparams.py), and available models
   are at [genieparser/models/__init__.py](genieparser/models/__init__.py).
    
   During training, you can use [tensorboard](https://github.com/tensorflow/tensorboard) to visualize
   progress:
   ```
   tensorboard --logdir ./workdir/model
   ```
    
   After training, you can extract the metrics of the best model on the validation set with:
   ```
   genie-print-metrics --output_dir ./workdir/model
     --eval_early_stopping_metric 'metrics-semparse_thingtalk_noquote/accuracy'
     --noeval_early_stopping_metric_minimize
   ```
    
5. Evaluate:
   ```
   genie-trainer --data_dir ./workdir --output_dir ./workdir/model
     --model genie_copy_seq2seq
     --hparams_set 'lstm_genie'
     --hparams_overrides ''
     --decode_hparams 'beam_size=20,return_beams=true'
     --problem 'semparse_thingtalk_noquote'
     --eval_early_stopping_metric 'metrics-semparse_thingtalk_noquote/accuracy'
     --noeval_early_stopping_metric_minimize
     --schedule evaluate
   ```
    
   Add `--eval_use_test_set` to use the test set instead of the validation set.
    
   You can also evaluate on a specific saved model (such as the best model according to the metric
   on the validation set) using the flag `--checkpoint_path ./workdir/model/export/best/.../variables/variables`
    
6. To deploy the model, point the server to a saved model directory from `workdir/model/export/best`,
   by writing a configuration file `server.conf` containing:
   ```
   [models]
   en=<path-to-saved-model>
   ```

   Run the server with:
   ```
   genie-server --config-file <path-to-server.conf>
   ```

   By default, the server runs at port 8400. You can change that by editing the server.conf file.
   An the example server.conf file is provided in the `data` folder, which describes all available
   options, including SSL and privilege separation.

   The server expects to connect to a TokenizerService (provided by [Almond Tokenizer](https://github.com/Stanford-Mobisocial-IoT-Lab/almond-tokenizer)) on
   localhost, port 8888.

