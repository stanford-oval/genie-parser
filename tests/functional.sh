#!/bin/bash

set -e
set -x
SRCDIR=`dirname $0`

# Functional tests
# for each combination of supported tasks and options
# create a workdir and train a model

GLOVE=${GLOVE:-glove.42B.300d.txt}
test -f $GLOVE || ( wget https://nlp.stanford.edu/data/glove.42B.300d.zip ; unzip glove.42B.300d.zip ; rm glove.42B.300d.zip )
export GLOVE


workdir=`mktemp -d -p . luinet-tests-XXXXXX`
for problem in semparse_thingtalk_noquote ; do

    pipenv run $SRCDIR/../luinet-datagen --problem $problem --src_data_dir $SRCDIR/dataset/$problem --data_dir $workdir --thingpedia_snapshot 6

    i=0
    for model in luinet_copy_transformer ; do
        for grammar in bottomup topdown linear ; do
            for hparams in transformer_tiny_luinet ; do
                for options in "pointer_layer=attentive" ; do
                    pipenv run $SRCDIR/../luinet-trainer --problem $problem --data_dir $workdir --output_dir $workdir/model.$i --model $model --hparams_set $hparams --hparams "grammar_direction=$grammar,$options" --export_saved_model --schedule test
                    pipenv run $SRCDIR/../luinet-decoder --problem $problem --data_dir $workdir --output_dir $workdir/model.$i --model $model --hparams_set $hparams --hparams "grammar_direction=$grammar,$options" --decode_hparams 'beam_size=1,alpha=0.6'
                    # we cannot test this until the t2t patch is merged
                    #test -d $workdir/model.$i/export/best/*
                    #test -f $workdir/model.$i/export/best/*/variables

                    i=$((i+1))
                done
            done
        done
    done

    rm -fr $workdir
done