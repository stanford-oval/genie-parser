# Almond NN-Parser

A neural-network based semantic parser for the Almond
virtual assistant.

Almond is a research project of the Mobile and Social
Computing Lab at Stanford University.  You can find more
information at <http://almond.stanford.edu/about> and
<https://mobisocial.stanford.edu>.

## Installation

Almond NN-Parser depends on numpy, Tensorflow 1.2 for Python 3, and
the OrderedSet module from PIP:

    sudo dnf install python3-numpy
    pip3 install tensorflow orderedset

It's recommended to install numpy from distribution packages, not
pip because it's faster and more reliable. The rest is not available
at least in Fedora.
Optionally, you'll want to install python3-matplotlib-gtk3 for the
visualizations.

Please see the Tensorflow documentation if you wish to use GPU
acceleration (you'll need to install nvidia+cuda or amdgpu-pro+rocm
drivers).

Almond NN-Parser has been tested successfully on Fedora 25 and 26
x86_64 with CPU and Nvidia GPU acceleration.

## Training

Training assumes that you acquired the dataset in preprocessed form
from somewhere (e.g. as part of the supplementary material for a paper
that uses Almond NN-Parser).

1. Set the `DATASET` environment variable to the directory containing
   the dataset *.tsv files
2. Download the word embeddings. We recommend using Glove 42B 300-dimensional,
   which can be downloaded from <http://nlp.stanford.edu/data/glove.42B.300d.zip>.
   Set the `GLOVE` environment variable to the path of uncompressed text file.
3. Prepare the working directory:
```
    mkdir ~/workdir
    cd ~/workdir
    ~/almond-nnparser/scripts/prepare.sh . ${SNAPSHOT}
```
   This script computes the input dictionary, downloads a snapshot of Thingpedia,
   and computes a subset of the word embedding matrix to make it faster to
   load. Use the snapshot argument to choose which Thingpedia snapshot to train
   against, or pass -1 for the latest content of Thingpedia. Be aware that
   using -1 might make the results impossible to reproduce.
4. Check that the dataset is compatible with the Thingpedia snapshot:
```
    cut -f2 ${DATASET}/*.tsv > programs.txt
    cd ~/almond-nnparser
    python3 -m grammar.thingtalk ~/workdir/thingpedia.txt < ~/workdir/programs.txt
```
5. Prepare a model directory, eg. `model.1`, inside the working directory,
   and create a `model.conf` inside it. Edit any model parameters that you
   wish.
6. Train:
```
    ~/almond-nnparser/run_train.py ./model.1 ${DATASET}/train.tsv ${DATASET}/dev.tsv
```
7. Visualize the training:
```
    ~/almond-nnparser/scripts/plot_learning.py ./model.1/train-stats.json
```
8. Test:
```
    ~/almond-nnparser/run_test.py ./model.1 ${DATASET}/test.tsv
```

## Running the server

After training, a server which is compatible with Almond can be run from the
trained working directory:

    ~/almond-nnparser/run_server.py en:./model.1

By default, the server runs at port 8400. You can change that with a server.conf
file. You can change the paths in model.conf if you wish to run the server in a different
directory.

The server can also be run for multiple languages, as in:

    ~/almond-nnparser/run_server.py en:./model.1 es:./model.2 zh:./model.3

The server expects to connect to a TokenizerService (provided by SEMPRE) on
localhost, port 8888.
