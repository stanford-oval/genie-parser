#!/bin/sh

#              '-Grammar.tags' 'parse' '+Grammar.tags' 'general' \
#              '-DataSet.inPaths' 'train:thingtalk/thingtalk.examples' \

# The overnight paper does not include "rule" features
# I have them because they help the parser ignore too many
# derivation that use $StringValue (which is a catch all)
exec java -Xmx12G -ea '-Dmodules=core,overnight,freebase,thingtalk' \
              '-cp' 'libsempre/*:lib/*' \
              'edu.stanford.nlp.sempre.Main' \
              '-LanguageAnalyzer' 'corenlp.CoreNLPAnalyzer' \
              '-Builder.parser' 'FloatingParser' \
              '-Builder.executor' 'JavaExecutor' \
              '-JavaExecutor.unpackValues' 'false' \
              '-Builder.dataset' 'thingtalk.ThingpediaDataset' \
              '-Builder.valueEvaluator' 'thingtalk.JsonValueEvaluator' \
              '-Grammar.inPaths' 'sabrina/sabrina.grammar' \
              '-FeatureExtractor.featureComputers' 'overnight.OvernightFeatureComputer' \
              '-OvernightFeatureComputer.featureDomains' \
              'match' 'ppdb' 'skip-bigram' 'skip-ppdb' 'root' 'alignment' 'lexical' \
              'root_lexical' \
              '-FloatingParser.maxDepth' '12' \
              '-Parser.beamSize' '20' \
              '-Learner.maxTrainIters' '1' \
              '-SimpleLexicon.inPaths' 'sabrina/sabrina.lexicon' \
              '-wordAlignmentPath' 'sabrina/sabrina.word_alignments.berkeley' \
              '-phraseAlignmentPath' 'sabrina/sabrina.phrase_alignments' \
              '-PPDBModel.ppdbModelPath' 'sabrina/sabrina-ppdb.txt' \
              '-ThingpediaDatabase.dbUrl' 'jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine' \
              '-ThingpediaDatabase.dbUser' 'sempre' \
              '-Main.interactive' 'true' \
              '-BeamParser.executeAllDerivations' 'true' \
              '-FloatingParser.executeAllDerivations' 'true' \
              "$@"
