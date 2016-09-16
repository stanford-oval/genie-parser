#!/bin/sh

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -Xmx12G -ea -Dmodules=core,corenlp,overnight,thingtalk \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.Main \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -CoreNLPAnalyzer.entityRecognizers corenlp.PhoneNumberEntityRecognizer corenlp.EmailEntityRecognizer \
              -CoreNLPAnalyzer.yearsAsNumbers -CoreNLPAnalyzer.splitHyphens false \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths sabrina/sabrina.en.grammar \
              -Grammar.tags floatingargs floatingnames floatingstrings \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code strvalue \
              -FloatingParser.maxDepth 8 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 8 \
              -Learner.maxTrainIters 1 \
              -wordAlignmentPath sabrina/sabrina.word_alignments.berkeley \
              -phraseAlignmentPath sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -ThingpediaDataset.onlineLearnFile sabrina/sabrina.en.online_learn \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              -Main.interactive true \
              "$@"
