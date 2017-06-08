#!/bin/sh

set -e
set -x

MODULE=${MODULE:-almond}
CANONICALS=${1:-./${MODULE}/${MODULE}.canonicals.tsv}
SEMPREDIR=`dirname $0`/..
SEMPREDIR=`realpath ${SEMPREDIR}`
BERKELEYALIGNER=${BERKELEYALIGNER:-${SEMPREDIR}/../berkeleyaligner}
WORKDIR=${WORKDIR:-.}

# prepare the berkeley aligner input
rm -fr ${WORKDIR}/berkeleyaligner.tmp
mkdir -p ${WORKDIR}/berkeleyaligner.tmp/${MODULE}/test
mkdir -p ${WORKDIR}/berkeleyaligner.tmp/${MODULE}/train
cut -f1 < ${CANONICALS} > ${WORKDIR}/berkeleyaligner.tmp/${MODULE}/train/train.f
cut -f2 < ${CANONICALS} > ${WORKDIR}/berkeleyaligner.tmp/${MODULE}/train/train.e

# run the berkeley aligner
( cd ${WORKDIR}/berkeleyaligner.tmp ; java -ea -jar ${BERKELEYALIGNER}/berkeleyaligner.jar ++${SEMPREDIR}/${MODULE}/${MODULE}.berkeleyaligner.conf )
paste ${WORKDIR}/berkeleyaligner.tmp/output/training.f \
      ${WORKDIR}/berkeleyaligner.tmp/output/training.e \
      ${WORKDIR}/berkeleyaligner.tmp/output/training.align > ${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.source.${LANGUAGE_TAG}

# convert the berkeley aligner format to something sempre likes
java -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
	-Dmodules=core,corenlp,overnight,thingtalk \
	edu.stanford.nlp.sempre.overnight.Aligner \
	${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.source.${LANGUAGE_TAG} \
	${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.${LANGUAGE_TAG} berkeley 2
rm -fr ${WORKDIR}/berkeleyaligner.tmp

# run the phrase aligner
java -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
	-Dmodules=core,corenlp,overnight,thingtalk \
	edu.stanford.nlp.sempre.overnight.PhraseAligner \
	${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.source.${LANGUAGE_TAG} \
	${WORKDIR}/${MODULE}/${MODULE}.phrase_alignments.${LANGUAGE_TAG} \
	${LANGUAGE_TAG} 2
