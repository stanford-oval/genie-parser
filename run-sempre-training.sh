#!/bin/sh

set -e
set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
WORKDIR=`pwd`
SEMPREDIR=`dirname $0`

rm -fr ${WORKDIR}/sempre.tmp
cp ${SEMPREDIR}/module-classes.txt .
java -Xmx32G -ea -Dmodules=core,corenlp,overnight,thingtalk \
              -Djava.library.path=jni \
              -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
              edu.stanford.nlp.sempre.Main \
              -execDir ${WORKDIR}/sempre.tmp \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -CoreNLPAnalyzer.entityRecognizers corenlp.PhoneNumberEntityRecognizer corenlp.EmailEntityRecognizer \
              -CoreNLPAnalyzer.yearsAsNumbers -CoreNLPAnalyzer.splitHyphens false \
              -CoreNLPAnalyzer.languageTag ${LANGUAGE_TAG} \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths ${SEMPREDIR}/sabrina/sabrina.${LANGUAGE_TAG}.grammar \
              -Grammar.tags floatingargs floatingnames floatingstrings \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -FeatureExtractor.languageTag ${LANGUAGE_TAG} \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code strvalue \
              -FloatingParser.maxDepth 14 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 8 \
              -Learner.maxTrainIters 2 \
              -wordAlignmentPath ${WORKDIR}/sabrina/sabrina.word_alignments.berkeley.${LANGUAGE_TAG} \
              -phraseAlignmentPath ${WORKDIR}/sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath ${SEMPREDIR}/sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -ThingpediaDataset.onlineLearnFile ${WORKDIR}/sabrina/sabrina.${LANGUAGE_TAG}.online_learn \
              -ThingpediaDataset.languageTag ${LANGUAGE_TAG} \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              "$@"

# move the generated file where APIServer will know to look for
cp ${WORKDIR}/sempre.tmp/params.2 ${WORKDIR}/sabrina/sabrina.${LANGUAGE_TAG}.params
#rm -fr ${WORKDIR}/sempre.tmp
