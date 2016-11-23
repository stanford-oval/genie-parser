#!/bin/sh

set -e
set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
WORKDIR=`pwd`
export WORKDIR
SEMPREDIR=`dirname $0`
export SEMPREDIR

rm -fr ${WORKDIR}/sempre.tmp
test ${SEMPREDIR} != "." && cp ${SEMPREDIR}/module-classes.txt .

cat > ${WORKDIR}/sabrina/sabrina.training.conf <<EOF
!include ${SEMPREDIR}/sabrina/sabrina.conf
execDir ${WORKDIR}/sempre.tmp

# Grammar
Grammar.inPaths ${SEMPREDIR}/sabrina/sabrina.${LANGUAGE_TAG}.grammar

# Dataset
Builder.dataset thingtalk.ThingpediaDataset
ThingpediaDataset.languageTag ${LANGUAGE_TAG}
Dataset.devFrac 0.1
Dataset.trainFrac 0.9
Dataset.splitDevFromTrain false
# note that we don't set splitDevFromTrain here
# run-manual-training-pipeline.sh would do that, if needed

# Features
FeatureExtractor.languageTag ${LANGUAGE_TAG}
OvernightFeatureComputer.wordAlignmentPath ${WORKDIR}/sabrina/sabrina.word_alignments.berkeley
OvernightFeatureComputer.phraseAlignmentPath ${WORKDIR}/sabrina/sabrina.phrase_alignments

# Training
Learner.maxTrainIters 2
Learner.numThreads 8
Learner.batchSize 75
Params.l1Reg nonlazy
Params.l1RegCoeff 0.0001
EOF

${SEMPREDIR}/run.sh training "$@"

# move the generated file where APIServer will know to look for
cp ${WORKDIR}/sempre.tmp/params.2 ${WORKDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.params
#rm -fr ${WORKDIR}/sempre.tmp
