#!/bin/sh

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -ea -Dmodules=core,corenlp,overnight,thingtalk,api \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.api.APIServer \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -CoreNLPAnalyzer.entityRecognizers corenlp.PhoneNumberEntityRecognizer corenlp.EmailEntityRecognizer \
              -CoreNLPAnalyzer.yearsAsNumbers -CoreNLPAnalyzer.splitHyphens false \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -JavaExecutor.unpackValues false \
              -Grammar.languageInPaths en:sabrina/sabrina.en.grammar it:sabrina/sabrina.it.grammar es:sabrina/sabrina.es.grammar zh:sabrina/sabrina.zh.grammar \
              -Grammar.tags floatingargs floatingnames \
              -Builder.languageInParamsPath en:sabrina/sabrina.en.params it:sabrina/sabrina.it.params es:sabrina/sabrina.es.params zh:sabrina/sabrina.zh.params \
              -APIServer.onlineLearnFiles en:sabrina/sabrina.en.online_learn it:sabrina/sabrina.it.online_learn es:sabrina/sabrina.es.online_learn zh:sabrina/sabrina.zh.online_learn \
              -APIServer.testSetFiles en:sabrina/sabrina.en.test it:sabrina/sabrina.it.test es:sabrina/sabrina.es.test zh:sabrina/sabrina.zh.test \
              -APIServer.utteranceLogFile sabrina/sabrina.log \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code strvalue \
              -FloatingParser.maxDepth 16 \
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
