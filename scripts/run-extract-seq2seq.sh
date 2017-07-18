#!/bin/sh

set -x

SEMPREDIR=`dirname $0`/..
SEMPREDIR=`realpath ${SEMPREDIR}`
MODULE=${MODULE:-almond}

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
EXTRA_ARGS="$@"

JAVA=${JAVA:-java}
BASE_ARGS="-ea $JAVA_ARGS -Djava.library.path=${SEMPREDIR}/jni -Dmodules=core,corenlp,overnight,thingtalk,api -cp ${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*"

exec ${JAVA} ${BASE_ARGS} edu.stanford.nlp.sempre.thingtalk.seq2seq.ExtractSeq2Seq \
	++${SEMPREDIR}/${MODULE}/${MODULE}.seq2seq.conf \
	-ExtractSeq2Seq.languageTag ${LANGUAGE_TAG} \
	-CoreNLPAnalyzer.languageTag ${LANGUAGE_TAG} \
	-ThingpediaDataset.languageTag ${LANGUAGE_TAG} \
	${EXTRA_ARGS}
