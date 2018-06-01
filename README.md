# Almond NN-Parser

A neural-network based semantic parser for the Almond virtual assistant.

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
acceleration. You'll need to install nvidia+cuda or amdgpu-pro+rocm
drivers, and then install "tensorflow-gpu" from pip, or build
Tensorflow from source.

Almond NN-Parser has been tested successfully on Fedora 25 to 27
x86_64 with CPU and Nvidia GPU acceleration.

## Running a pre-trained model

Pretrained models are available for download from <https://parmesan.stanford.edu/dataset/en/current.tar.xz>.

You should unpack the tarball in a directory named `en`. After doing so, you can test the model interactively
using:

    ~/almond-nnparser/run_interactive.py ./en/model

Note that you must run this command from the directory that contains the `en` subdirectory.

The interactive shell accepts virtual assistant commands, such as "post on twitter" or "get a cat picture",
and outputs the corresponding ThingTalk program. The command must be preprocessed already, e.g.
"post QUOTED\_STRING\_0 on twitter" instead of "post 'hello' on twitter".

Basic model evaluation can also be perfomed as:

    ~/almond-nnparser/run_test.py ./en/model ./en/dataset/paraphrasing-dev.tsv

Finally, a server which is compatible with Almond can be run as:

    ~/almond-nnparser/run_server.py

By default, the server runs at port 8400. You can change that by creating a server.conf file
in the current working directory.
An example server.conf file is provided in the `data` folder.

The server expects to connect to a TokenizerService (provided by Almond Tokenizer) on
localhost, port 8888.

## Training

Training assumes that you acquired the dataset in preprocessed form
from somewhere (e.g. as part of the supplementary material for a paper
that uses Almond NN-Parser).

1. Set the `DATASET` environment variable to the directory containing
   the dataset *.tsv files. If you use the datasets linked above, this should
   be `./en/dataset`.
2. Download the word embeddings. We recommend using Glove 42B 300-dimensional,
   which can be downloaded from <http://nlp.stanford.edu/data/glove.42B.300d.zip>.
   Set the `GLOVE` environment variable to the path of uncompressed text file.
   
   You can skip this step, in which case, the prepare.py script will download
   the recommended GloVe file automatically.
3. Prepare the working directory:
   ```
    mkdir ./en
    ~/almond-nnparser/prepare.py ./en ${DATASET} [SNAPSHOT] [EMBED_SIZE] [EXTRA_WORD_FILE]
   ```
   This script computes the input dictionary, downloads a snapshot of Thingpedia,
   and computes a subset of the word embedding matrix corresponding to the dictionary.
   
   Use the optional SNAPSHOT argument to choose which Thingpedia snapshot to train
   against, or pass -1 for the latest content of Thingpedia. Be aware that
   using -1 might make the results impossible to reproduce.
   Use the EMBED_SIZE argument to choose the size (number of dimensions) of the embedding
   to download. Defaults to 300.
   Use EXTRA_WORD_FILE to specify additional words that should be included in the
   dictionary (one per line); words that are not in the dataset, in the canonical forms
   for Thingpedia functions or in this additional dictionary file will be mapped to the unknown
   token `<unk>`.
   
   This script will also verify that the dataset is compatible with the
   Thingpedia snapshot, and will create a default directory called `model` to
   host the model and its configuration file.
   
4. (Optional) Edit any model parameters that you wish in `./en/model/model.conf`.
6. Train:
    ```
    ~/almond-nnparser/run_train.py ./en/model train:${DATASET}/train.tsv dev:${DATASET}/dev.tsv
    ```
    
    If you wish to use curriculum learning (recommended), do:
    
    ```
    ~/almond-nnparser/run_train.py ./en/model train:${DATASET}/synthetic-train.tsv train:${DATASET}/train-nosynthetic.tsv dev:${DATASET}/dev.tsv
    ```
    
7. Visualize the training:
    ```
    ~/almond-nnparser/scripts/plot_learning.py ./en/model/train-stats.json
    ```
7. Test:
    ```
    ~/almond-nnparser/run_test.py ./en/model ${DATASET}/test.tsv
    ```
    
    This will produce a `stats_test.txt` file for error analysis. The file is tab-separated and has many columns;
    the first column is the input sentence, then the gold program, then the predicted program, then `True` or `False`
    whether it was predicted correctly or not, various `Correct` or `Incorrect` property for partial accuracy.
    This command will also produce `test-function-f1.tsv`, reporting precision, recall and F1 score for each function
    in the test set, as a binary prediction. It will produce `test-f1.tsv`, reporting precision, recall and F1 score,
    for each grammar reduction (parse action), based on the confusion matrix between the prediction and the gold sequence.
    NOTE: if the input file is called `foo.tsv`, the output file will be called `stats_foo.txt`.
