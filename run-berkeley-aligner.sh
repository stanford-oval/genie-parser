#!/bin/sh

set -e
set -x

MODULE=${MODULE:-sabrina}
CANONICALS=${1:-./${MODULE}/sabrina.canonicals.tsv}
SEMPREDIR=${2:-`pwd`}

# prepare the berkeley aligner input
rm -fr ./berkeleyaligner.tmp
mkdir -p ./berkeleyaligner.tmp/sabrina/test
mkdir -p ./berkeleyaligner.tmp/sabrina/train
java -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
	-Dmodules=core,corenlp,overnight,thingtalk \
	edu.stanford.nlp.sempre.overnight.Tokenizer \
	${CANONICALS} \
	./berkeleyaligner.tmp/sabrina/train/train.f \
	./berkeleyaligner.tmp/sabrina/train/train.e \
	${LANGUAGE_TAG}

# run the berkeley aligner
( cd ./berkeleyaligner.tmp ; java -ea -jar /home/gcampagn/mobisocial/berkeleyaligner/berkeleyaligner.jar ++${SEMPREDIR}/sabrina/sabrina.berkeleyaligner.conf )
paste ./berkeleyaligner.tmp/output/training.f \
      ./berkeleyaligner.tmp/output/training.e \
      ./berkeleyaligner.tmp/output/training.align > ./${MODULE}/${MODULE}.word_alignments.berkeley.source.${LANGUAGE_TAG}

# convert the berkeley aligner format to something sempre likes
java -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
	-Dmodules=core,corenlp,overnight,thingtalk \
	edu.stanford.nlp.sempre.overnight.Aligner \
	./${MODULE}/${MODULE}.word_alignments.berkeley.source.${LANGUAGE_TAG} \
	./${MODULE}/${MODULE}.word_alignments.berkeley.${LANGUAGE_TAG} berkeley 2
rm -fr ./berkeleyaligner.tmp

# run the phrase aligner
java -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
	-Dmodules=core,corenlp,overnight,thingtalk \
	edu.stanford.nlp.sempre.overnight.PhraseAligner \
	$CANONICALS \
	./${MODULE}/${MODULE}.phrase_alignments.${LANGUAGE_TAG} \
	./${MODULE}/${MODULE}.word_alignments.berkeley.${LANGUAGE_TAG} \
	${LANGUAGE_TAG} 2
