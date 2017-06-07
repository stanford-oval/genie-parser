#!/bin/sh

usage() {
	echo "run.sh: [interactive|training|server] [extra SEMPRE args...]"
}

SEMPREDIR=`dirname $0`
SEMPREDIR=`realpath $SEMPREDIR`
MODULE=${MODULE:-sabrina}
LANGUAGE_TAG=${LANGUAGE_TAG:-en}
WORKDIR=${WORKDIR:-.}

MODE=$1
shift

EXTRA_ARGS="$@"

JAVA=${JAVA:-java}
BASE_ARGS="-ea $JAVA_ARGS -Djava.library.path=${SEMPREDIR}/jni -Dmodules=core,corenlp,overnight,thingtalk,api -cp ${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*"

case $MODE in
	interactive)
		TARGET=edu.stanford.nlp.sempre.Main
		MODE_ARGS="++${MODULE}/${MODULE}.interactive.conf -Builder.inParamsPath ${MODULE}/${MODULE}.${LANGUAGE_TAG}.params"
		;;
    eval)
        TARGET=edu.stanford.nlp.sempre.Main
		MODE_ARGS="++${MODULE}/${MODULE}.training.conf -Builder.inParamsPath ${MODULE}/${MODULE}.${LANGUAGE_TAG}.params -Dataset.maxExamples train:0"
		;;
	training)
		TARGET=edu.stanford.nlp.sempre.Main
		MODE_ARGS="++${MODULE}/${MODULE}.training.conf"
		;;
	server)
		TARGET=edu.stanford.nlp.sempre.api.APIServer
		MODE_ARGS="++${MODULE}/${MODULE}.server.conf"
		;;
	interactive)
		usage
		exit 1
		;;
esac

if test $MODE != "server" ; then
MODE_ARGS="${MODE_ARGS} 
-Grammar.inPaths ${SEMPREDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.grammar 
-CoreNLPAnalyzer.languageTag ${LANGUAGE_TAG} 
-ThingpediaDataset.languageTag ${LANGUAGE_TAG} 
-FeatureExtractor.languageTag ${LANGUAGE_TAG} 
-OvernightFeatureComputer.wordAlignmentPath ${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley 
-OvernightFeatureComputer.phraseAlignmentPath ${WORKDIR}/${MODULE}/${MODULE}.phrase_alignments 
-PPDBModel.ppdbModelPath ${SEMPREDIR}/data/ppdb.txt"
fi

exec ${JAVA} ${BASE_ARGS} ${TARGET} ${MODE_ARGS} ${EXTRA_ARGS}
