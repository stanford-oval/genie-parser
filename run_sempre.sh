#!/bin/sh

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -ea -Dmodules=core,corenlp,overnight,freebase,thingtalk,api \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.api.APIServer \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -JavaExecutor.unpackValues false \
              -Grammar.languageInPaths en:sabrina/sabrina.en.grammar it:sabrina/sabrina.it.grammar es:sabrina/sabrina.es.grammar \
              -Grammar.tags floatingargs floatingnames \
              -Builder.languageInParamsPath en:sabrina/sabrina.en.params it:sabrina/sabrina.it.params es:sabrina/sabrina.es.params \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code \
              -FloatingParser.maxDepth 8 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 20 \
              -wordAlignmentPath sabrina/sabrina.word_alignments.berkeley \
              -phraseAlignmentPath sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              "$@"
