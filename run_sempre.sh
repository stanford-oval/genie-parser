#!/bin/sh

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -ea '-Dmodules=core,corenlp,overnight,freebase,thingtalk' \
              '-cp' 'libsempre/*:lib/*' \
              'edu.stanford.nlp.sempre.APIServer' \
              '-LanguageAnalyzer' 'corenlp.CoreNLPAnalyzer' \
              '-Builder.parser' 'BeamParser' \
              '-Builder.executor' 'JavaExecutor' \
              '-JavaExecutor.unpackValues' 'false' \
              '-Grammar.languageInPaths' 'en:sabrina/sabrina.en.grammar' 'it:sabrina/sabrina.it.grammar' \
              '-Grammar.tags' 'parse' '+Grammar.tags' 'general' \
              '-FeatureExtractor.featureDomains' 'denotation' 'rule' \
              '-FeatureExtractor.featureComputers' 'overnight.OvernightFeatureComputer' \
              '-OvernightFeatureComputer.featureDomains' \
              'match' 'ppdb' 'skip-bigram' 'root' 'alignment' 'lexical' \
              'root_lexical' \
              '-FloatingParser.maxDepth' '12' \
              '-Parser.beamSize' '9' \
              '-wordAlignmentPath' 'sabrina/sabrina.word_alignments.berkeley' \
              '-phraseAlignmentPath' 'sabrina/sabrina.phrase_alignments' \
              '-PPDBModel.ppdbModelPath' 'sabrina/sabrina-ppdb.txt' \
              '-Learner.maxTrainIters' '1' \
              '-SimpleLexicon.inPaths' 'sabrina/sabrina.lexicon' \
              '-DataSet.inPaths' 'train:sabrina/sabrina.examples' \
              '-ThingpediaLexicon.dbUrl' 'jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine' \
              '-ThingpediaLexicon.dbUser' 'sempre' \
              '-BeamParser.executeAllDerivations' 'true' \
              '-FloatingParser.executeAllDerivations' 'true' \
              "$@"
