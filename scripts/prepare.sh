#!/bin/bash
#
# Copyright 2017 Giovanni Campagna <gcampagn@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 

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
EMBED_SIZE=${3:-300}
srcdir=`dirname $0`/..
srcdir=`realpath ${srcdir}`
DATASET=${DATASET:-$WORKDIR}
GLOVE=${GLOVE:-$WORKDIR/glove.txt}

test -f ${GLOVE} || download_glove

python3 ${srcdir}/scripts/gen_grammar.py "${DATABASE_PW}" > ${WORKDIR}/thingpedia.txt
cat ${DATASET}/*.tsv | cut -f2 | tr " " "\n" | sort -u > ${WORKDIR}/input_words.txt
python3 ${srcdir}/scripts/trim_embeddings.py ${WORKDIR}/input_words.txt ${EMBED_SIZE} < ${GLOVE} > ${WORKDIR}/embeddings-${EMBED_SIZE}.txt
cp ${srcdir}/data/default.conf ${WORKDIR}/default.conf
