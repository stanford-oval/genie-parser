#!/bin/sh

set -e
set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}

rm -fr ./sempre.tmp
java -Xmx12G -ea -Dmodules=core,corenlp,overnight,freebase,thingtalk \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.Main \
              -execDir ./sempre.tmp \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -CoreNLPAnalyzer.entityRecognizers corenlp.PhoneNumberEntityRecognizer corenlp.EmailEntityRecognizer \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths sabrina/sabrina.${LANGUAGE_TAG}.grammar \
              -Grammar.tags floatingargs floatingnames floatingstrings \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code \
              -FloatingParser.maxDepth 16 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 20 \
              -Learner.maxTrainIters 3 \
              -Learner.reduceParserScoreNoise \
              -Parser.derivationScoreNoise 1 \
              -wordAlignmentPath sabrina/sabrina.word_alignments.berkeley \
              -phraseAlignmentPath sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -ThingpediaDataset.onlineLearnFile sabrina/sabrina.${LANGUAGE_TAG}.online_learn \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              "$@"

# move the generated file where APIServer will know to look for
cp sempre.tmp/params.3 ./sabrina/sabrina.${LANGUAGE_TAG}.params

rm -fr ./sempre.tmp
