#!/bin/sh

set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
SEMPREDIR=`dirname $0`/..
SEMPREDIR=`realpath ${SEMPREDIR}`
MODULE=${MODULE:-almond}
OUTPUT="$1"
shift

EXTRA_ARGS="$@"

JAVA=${JAVA:-java}
BASE_ARGS="-ea $JAVA_ARGS -Djava.library.path=${SEMPREDIR}/jni -Dmodules=core,corenlp,overnight,thingtalk,api -cp ${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*"

exec ${JAVA} ${BASE_ARGS} edu.stanford.nlp.sempre.thingtalk.DownloadDataset \
	++${SEMPREDIR}/${MODULE}/${MODULE}.training.conf \
	-Grammar.inPaths ${SEMPREDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.grammar \
	-DownloadDataset.outputFile ${OUTPUT} \
	-CoreNLPAnalyzer.languageTag ${LANGUAGE_TAG} \
	-ThingpediaDataset.languageTag ${LANGUAGE_TAG} \
	${EXTRA_ARGS}
