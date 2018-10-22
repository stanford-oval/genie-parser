#!/bin/bash

set -e
set -x
SRCDIR=`dirname $0`

# Functional tests
# for each combination of supported tasks and options
# create a workdir and train a model

GLOVE=${GLOVE:-glove.42B.300d.txt}
test -f $GLOVE || ( wget --no-verbose https://nlp.stanford.edu/data/glove.42B.300d.zip ; unzip glove.42B.300d.zip ; rm glove.42B.300d.zip )
export GLOVE

declare -A model_hparams
model_hparams[luinet_copy_transformer]=transformer_tiny_luinet
model_hparams[luinet_copy_seq2seq]=lstm_luinet



for problem in semparse_thingtalk_noquote semparse_thingtalk ; do
    workdir=`mktemp -d -p . luinet-tests-XXXXXX`
    pipenv run $SRCDIR/../luinet-datagen --problem $problem --src_data_dir $SRCDIR/dataset/$problem --data_dir $workdir --thingpedia_snapshot 6
    pipenv run python3 $SRCDIR/../luinet/scripts/retrieval.py --input_vocab $workdir/input_words.txt --thingpedia_snapshot $workdir/thingpedia.json --train_set $SRCDIR/dataset/$problem/train.tsv --test_set $SRCDIR/dataset/$problem/dev.tsv --cached_grammar $workdir/cached_grammar.pkl --cached_embeddings $workdir/input_embeddings.npy

    i=0
    for model in luinet_copy_seq2seq luinet_copy_transformer ; do
        for grammar in linear bottomup ; do
            for options in "" "pointer_layer=decaying_attentive" ; do
                pipenv run $SRCDIR/../luinet-trainer --problem $problem --data_dir $workdir --output_dir $workdir/model.$i --model $model --hparams_set ${model_hparams[$model]} --hparams "grammar_direction=$grammar,$options" --export_saved_model --schedule test

                # greedy decode
                pipenv run $SRCDIR/../luinet-decoder --problem $problem --data_dir $workdir --output_dir $workdir/model.$i --model $model --hparams_set ${model_hparams[$model]} --hparams "grammar_direction=$grammar,$options" --decode_hparams 'beam_size=1,alpha=0.6'
                    
                # beam search decode
                pipenv run $SRCDIR/../luinet-decoder --problem $problem --data_dir $workdir --output_dir $workdir/model.$i --model $model --hparams_set ${model_hparams[$model]} --hparams "grammar_direction=$grammar,$options" --decode_hparams 'beam_size=4,alpha=0.6'
                
                # we cannot test this until the t2t patch is merged
                #test -d $workdir/model.$i/export/best/*
                #test -f $workdir/model.$i/export/best/*/variables

                i=$((i+1))
            done
        done
    done
    rm -fr $workdir

done
