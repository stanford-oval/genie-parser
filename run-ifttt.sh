#!/bin/bash

die() {
    echo "$@"
    exit 1
}

set -e

originalpwd=`pwd`

LANGUAGE_TAG=en
export LANGUAGE_TAG
MODULE=ifttt
export MODULE
MODULES=core,corenlp,overnight,thingtalk,ifttt
export MODULES

SEMPRE=${SEMPRE:-`pwd`}

test -f ${SEMPRE}/ifttt/train.tsv || die "Must obtain train.tsv file by running ./ifttt/build_lexicon.js with the original data, then splitting 80%-20%"

set -x

WORKDIR=`mktemp -t -d workdir-XXXXXX`
WORKDIR=`realpath "${WORKDIR}"`
mkdir ${WORKDIR}/ifttt
cd ${WORKDIR}

on_error() {
    rm -fr ${WORKDIR}
}
#trap on_error ERR INT TERM

# run the berkeley aligner
${SEMPRE}/run-berkeley-aligner.sh ${SEMPRE}/ifttt/train.tsv ${SEMPRE}

# actually run sempre
${SEMPRE}/run-sempre-training.sh \
    -Builder.dataset ifttt.IftttDataset \
    -IftttDataset.trainFile ${SEMPRE}/ifttt/train.tsv \
    -IftttDataset.testFile ${SEMPRE}/ifttt/test.tsv \
    -ThingpediaDatabase.dbUrl mysql://localhost/ifttt \
    -ThingpediaDatabase.dbUser ifttt \
    -ThingpediaDatabase.dbPw ifttt \
    "$@"

# copy the workdir back to where we were launched from
for i in `seq 1 10`; do
	if ! test -d ${originalpwd}/workdir.$i ; then
		mv ${WORKDIR} ${originalpwd}/workdir.$i
		break
	fi
done
