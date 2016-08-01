#!/bin/sh

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -Xmx12G -ea -Dmodules=core,overnight,freebase,thingtalk \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.Main \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths sabrina/sabrina.en.grammar \
              -Grammar.tags floatingargs floatingnames \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code \
              -FloatingParser.maxDepth 16 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 20 \
              -Learner.maxTrainIters 1 \
              -Learner.reduceParserScoreNoise \
              -Parser.derivationScoreNoise 1 \
              -wordAlignmentPath sabrina/sabrina.word_alignments.berkeley \
              -phraseAlignmentPath sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              -Main.interactive true \
              "$@"
