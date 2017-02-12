#!/bin/sh

set -x

SEMPREDIR=`dirname $0`

EXTRA_ARGS="$@"

JAVA=${JAVA:-java}
BASE_ARGS="-ea $JAVA_ARGS -Dmodules=core,corenlp,overnight,thingtalk,api -cp ${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*"

exec ${JAVA} ${BASE_ARGS} edu.stanford.nlp.sempre.thingtalk.ThingpediaLexiconEvaluator ++sabrina/sabrina.training.conf ${EXTRA_ARGS}
