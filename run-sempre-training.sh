#!/bin/sh

set -e
set -x

MODULE=${MODULE:-sabrina}
LANGUAGE_TAG=${LANGUAGE_TAG:-en}
WORKDIR=`pwd`
SEMPREDIR=`dirname $0`

MODULES=${MODULES:-core,corenlp,overnight,thingtalk}

rm -fr ${WORKDIR}/sempre.tmp
cp ${SEMPREDIR}/module-classes.txt .
java -Xmx32G -ea -Dmodules=${MODULES} \
              -Djava.library.path=jni \
              -cp "${SEMPREDIR}/libsempre/*:${SEMPREDIR}/lib/*" \
              edu.stanford.nlp.sempre.Main \
              -execDir ${WORKDIR}/sempre.tmp \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -CoreNLPAnalyzer.entityRecognizers corenlp.PhoneNumberEntityRecognizer corenlp.EmailEntityRecognizer \
               corenlp.QuotedStringEntityRecognizer corenlp.URLEntityRecognizer \
              -CoreNLPAnalyzer.regularExpressions 'USERNAME:[@](.+)' 'HASHTAG:[#](.+)' \
              -CoreNLPAnalyzer.yearsAsNumbers -CoreNLPAnalyzer.splitHyphens false \
              -CoreNLPAnalyzer.languageTag ${LANGUAGE_TAG} \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths ${SEMPREDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.grammar \
              -Grammar.tags includebookkeeping \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -FeatureExtractor.languageTag ${LANGUAGE_TAG} \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code strvalue \
              -FloatingParser.maxDepth 10 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 14 \
              -Learner.maxTrainIters 2 \
              -wordAlignmentPath ${WORKDIR}/${MODULE}/${MODULE}.word_alignments.berkeley.${LANGUAGE_TAG} \
              -phraseAlignmentPath ${WORKDIR}/${MODULE}/${MODULE}.phrase_alignments \
              -PPDBModel.ppdbModelPath ${SEMPREDIR}/sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -ThingpediaDataset.languageTag ${LANGUAGE_TAG} \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              "$@"

# move the generated file where APIServer will know to look for
cp ${WORKDIR}/sempre.tmp/params.2 ${WORKDIR}/${MODULE}/${MODULE}.${LANGUAGE_TAG}.params
#rm -fr ${WORKDIR}/sempre.tmp
