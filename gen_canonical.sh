#!/bin/sh


exec java -Dmodules=core,overnight,freebase,thingtalk -Xmx8g \
	-cp libsempre/*:lib/* -ea edu.stanford.nlp.sempre.overnight.GenerationMain \
	-GenerationMain.varyMaxDepth true \
	-LanguageAnalyzer corenlp.CoreNLPAnalyzer \
	-execDir genthingtalk.out -overwriteExecDir -addToView 0 \
	-JoinFn.typeInference true -JoinFn.specializedTypeCheck false \
	-JavaExecutor.unpackValues false -JavaExecutor.printStackTrace false \
	-Grammar.inPaths sabrina/sabrina.grammar \
	-initialization "denotation :: error,-1000" "denotation :: empty,-100" "paraphrase :: size,+0.01" "denotation :: value_in_formula,-100" \
	-FeatureExtractor.featureComputers overnight.OvernightFeatureComputer -OvernightFeatureComputer.featureDomains  \
	-OvernightFeatureComputer.itemAnalysis false \
	-Builder.parser FloatingParser -maxDepth 15 -beamSize 15 -derivationScoreNoise 1 \
	-Builder.executor JavaExecutor \
	-SimpleLexicon.inPaths sabrina/sabrina.lexicon \
	-Builder.dataset thingtalk.ThingpediaDataset -Derivation.showUtterance \
	-FeatureExtractor.featureDomains denotation -printAllPredictions \
	-printPredictedUtterances -FloatingParser.executeAllDerivations \
	-BeamParser.executeAllDerivations -Parser.pruneErrorValues true \
	-Grammar.tags generate +Grammar.tags general \
  '-ThingpediaDatabase.dbUrl' 'jdbc:mysql://thingengine.crqccvnuyu19.us-west-2.rds.amazonaws.com/thingengine' \
  '-ThingpediaDatabase.dbUser' 'sempre' \
	"$@"
