#!/bin/bash

# Usage:
#
# ./scripts/prepare.sh ${WORKDIR} ${DATABASE_PW}

download_glove() {
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -O ${WORKDIR}/glove.42B.300d.zip
unzip ${WORKDIR}/glove.42B.300d.zip
rm ${WORKDIR}/glove.42B.300d.zip
mv ${WORKDIR}/glove.42B.300d.txt ${GLOVE}
}

set -x
set -e

WORKDIR=${1:-.}
DATABASE_PW=${DATABASE_PW:-$2}
srcdir=`dirname $0`/
srcdir=`realpath ${srcdir}`
GLOVE=${GLOVE:-$WORKDIR/glove.txt}

test -f ${GLOVE} || download_glove

python3 ${srcdir}/scripts/gen_grammar.py "${DATABASE_PW}" > ${WORKDIR}/thingpedia.txt
cat ${WORKDIR}/*.tsv | cut -f1 | tr " " "\n" | sort -u > ${WORKDIR}/input_words.txt
python3 ${srcdir}/scripts/trim_embeddings.py ${WORKDIR}/input_words.txt < ${GLOVE} > ${WORKDIR}/embeddings.txt
cp ${srcdir}/data/default.conf ${WORKDIR}/default.conf
