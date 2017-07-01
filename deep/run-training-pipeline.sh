#!/bin/bash

die() {
    echo "$@"
    exit 1
}

set -e

LANGUAGE_TAG="$1"
shift
export LANGUAGE_TAG
test -n "${LANGUAGE_TAG}" || die "language must be specified as an argument to this script"

# default to everything, run-all-experiments.sh will override
TRAINING=${TRAINING:-thingpedia,turking-prim0,turking-prim1,turking-prim2,turking-prim3,turking-compound0,turking-compound1,turking-compound2,turking-compound3,turking-compound4,turking3-compound0,turking3-compound1,turking3-compound2,turking3-compound3,turking3-compound4,turking3-compound5,turking3-compound6,online,online-bookkeeping,ifttt-train}
export TRAINING
TESTING=${TESTING:-test-prim0,test-prim1,test-prim2,test-prim3,test-compound0,test-compound1,test-compound2,test-compound3,test-compound4,test3-compound0,test3-compound1,test3-compound2,test3-compound3,test3-compound4,test3-compound4,test3-compound5,test3-compound6,ifttt-test}
export TESTING

DATABASE_URL="mysql://sempre:${DATABASE_PW}@thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com:3306/thingengine?charset=utf8mb4_bin&ssl=Amazon%20RDS"
export DATABASE_URL

set -x

EMBEDDINGS=${EMBEDDINGS:-/srv/data/glove/glove.840B.300d.txt}

SRCDIR=`dirname $0`
SRCDIR=`realpath "${SRCDIR}"`
SEMPRE=`dirname ${SRCDIR}`

WORKDIR=`mktemp -t -d workdir-XXXXXX`
WORKDIR=`realpath "${WORKDIR}"`

on_error() {
    rm -fr ${WORKDIR}
}
#trap on_error ERR INT TERM

# extract the data
cd ${SEMPRE}
./run-prepare-seq2seq.sh \
    -ThingpediaDatabase.dbPw ${DATABASE_PW} \
    -Dataset.splitDevFromTrain true \
    -ThingpediaDataset.testTypes $(echo $TESTING | tr ',' ' ') \
	-ThingpediaDataset.trainTypes $(echo $TRAINING | tr ',' ' ') \
	-PrepareSeq2Seq.trainFile ${WORKDIR}/train.tsv \
	-PrepareSeq2Seq.devFile ${WORKDIR}/dev.tsv \
	-PrepareSeq2Seq.testFile ${WORKDIR}/test.tsv \
	> ${WORKDIR}/seq2seq.log 2>&1

cd ${SRCDIR}
# update the grammar, remove reddit and hackernews 
python ${SRCDIR}/scripts/gen_grammar.py ${DATABASE_PW} | grep -v -E ':reddit\.|:hackernews\.' > ${WORKDIR}/thingpedia.txt

# update the input tokens
cat ${WORKDIR}/train.tsv ${WORKDIR}/dev.tsv ${WORKDIR}/test.tsv | cut -f1 | tr ' ' '\n' | sort -u > ${WORKDIR}/input_words.txt
python ${SRCDIR}/scripts/trim_embeddings.py ${WORKDIR}/input_words.txt < ${EMBEDDINGS} > ${WORKDIR}/embeddings.txt

# actually run the model
cd ${WORKDIR}
python ${SRCDIR}/run_train.py tt ${WORKDIR}/input_words.txt ${WORKDIR}/embeddings.txt ${WORKDIR}/model ${WORKDIR}/train.tsv ${WORKDIR}/dev.tsv > ${WORKDIR}/train.log 2>&1
python ${SRCDIR}/run_test.py tt ${WORKDIR}/input_words.txt ${WORKDIR}/embeddings.txt ${WORKDIR}/model ${WORKDIR}/test.tsv > ${WORKDIR}/test.log 2>&1

# copy the workdir back to where we started
for i in `seq 1 10`; do
	if ! test -d ${SRCDIR}/workdir.$i ; then
		mv ${WORKDIR} ${SRCDIR}/workdir.$i
		break
	fi
done
