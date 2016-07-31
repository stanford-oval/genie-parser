#!/bin/sh

die() {
    echo "$@"
    exit 1
}

set -e

test -n ${DATABASE_URL} || die "DATABASE_URL must be set";
set -x

LANGUAGE_TAG=${LANGUAGE_TAG:-en}
THINGENGINE=${THINGENGINE:-`pwd`/../thingengine-platform-cloud}
BERKELEY_ALIGNER=${BERKELEY_ALIGNER:-`pwd`/../berkeleyaligner}

# extract the canonicals from the db
node ${THINGENGINE}/scripts/reconstruct_canonicals.js ./sabrina/sabrina.canonicals.tsv

# run the berkeley aligner
rm -fr ./berkeleyaligner.tmp
mkdir -p ./berkeleyaligner.tmp/sabrina/test
mkdir -p ./berkeleyaligner.tmp/sabrina/train
cut -f1 ./sabrina/sabrina.canonicals.tsv > ./berkeleyaligner.tmp/sabrina/train/train.f
cut -f2 ./sabrina/sabrina.canonicals.tsv > ./berkeleyaligner.tmp/sabrina/train/train.e
( cd ./berkeleyaligner.tmp ; java -ea -jar $BERKELEY_ALIGNER/berkeleyaligner.jar ++../sabrina/sabrina.berkeleyaligner.conf )
paste ./berkeleyaligner.tmp/output/training.f ./berkeleyaligner.tmp/output/training.e ./berkeleyaligner.tmp/output/training.align > ./sabrina/sabrina.word_alignments.berkeley.source

# convert the berkeley aligner format to something sempre likes
java -cp 'libsempre/*:lib/*' -Dmodules=core,corenlp,overnight,thingtalk edu.stanford.nlp.sempre.overnight.Aligner ./sabrina/sabrina.word_alignments.berkeley.source ./sabrina/sabrina.word_alignments.berkeley berkeley 2

# here would optionally clean up the ppdb, but we don't yet

# actually run sempre
rm -fr ./sempre.tmp
java -Xmx12G -ea -Dmodules=core,corenlp,overnight,freebase,thingtalk \
              -cp 'libsempre/*:lib/*' \
              edu.stanford.nlp.sempre.Main \
              -execDir ./sempre.tmp \
              -LanguageAnalyzer corenlp.CoreNLPAnalyzer \
              -Builder.parser FloatingParser \
              -Builder.executor JavaExecutor \
              -Builder.valueEvaluator thingtalk.JsonValueEvaluator \
              -JavaExecutor.unpackValues false \
              -Builder.dataset thingtalk.ThingpediaDataset \
              -Grammar.inPaths sabrina/sabrina.${LANGUAGE_TAG}.grammar \
              -Grammar.tags floatingargs floatingnames \
              -FeatureExtractor.featureDomains rule \
              -FeatureExtractor.featureComputers overnight.OvernightFeatureComputer thingtalk.ThingTalkFeatureComputer \
              -OvernightFeatureComputer.featureDomains \
              match ppdb skip-bigram skip-ppdb root alignment lexical \
              root_lexical \
              -ThingTalkFeatureComputer.featureDomains anchorBoundaries code paramVerbAlign \
              -FloatingParser.maxDepth 8 \
              -FloatingParser.useAnchorsOnce \
              -Parser.beamSize 40 \
              -Learner.maxTrainIters 3 \
              -Learner.reduceParserScoreNoise \
              -Parser.derivationScoreNoise 4 \
              -wordAlignmentPath sabrina/sabrina.word_alignments.berkeley \
              -phraseAlignmentPath sabrina/sabrina.phrase_alignments \
              -PPDBModel.ppdbModelPath sabrina/sabrina-ppdb.txt \
              -PPDBModel.ppdb false \
              -ThingpediaDatabase.dbUrl jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine \
              -ThingpediaDatabase.dbUser sempre \
              -BeamParser.executeAllDerivations true \
              -FloatingParser.executeAllDerivations true \
              "$@"

# move the generated file where APIServer will know to look for
cp sempre.tmp/params.3 ./sabrina/sabrina.${LANGUAGE_TAG}.params

rm -fr ./sempre.tmp
rm -fr ./berkeleyaligner.tmp
