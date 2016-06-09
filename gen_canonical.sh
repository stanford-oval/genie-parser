#!/bin/sh


exec java -Dmodules=core,overnight,freebase,thingtalk -Xmx2g \
	-cp libsempre/*:lib/* -ea edu.stanford.nlp.sempre.overnight.GenerationMain \
	-GenerationMain.varyMaxDepth true \
	-LanguageAnalyzer corenlp.CoreNLPAnalyzer \
	-execDir genthingtalk.out -overwriteExecDir -addToView 0 \
	-JoinFn.typeInference true -JoinFn.specializedTypeCheck false \
	-JavaExecutor.convertNumberValues false -JavaExecutor.printStackTrace false \
	-Grammar.inPaths thingtalk/thingtalk.grammar \
	-initialization "denotation :: error,-1000" "denotation :: empty,-100" "paraphrase :: size,+0.01" "denotation :: value_in_formula,-100" \
	-FeatureExtractor.featureComputers overnight.OvernightFeatureComputer -OvernightFeatureComputer.featureDomains  \
	-OvernightFeatureComputer.itemAnalysis false \
	-Builder.parser FloatingParser -maxDepth 15 -beamSize 15 -derivationScoreNoise 1 \
	-Builder.executor JavaExecutor \
	-SimpleLexicon.inPaths thingtalk/thingtalk.lexicon \
	-Dataset.inPaths train:thingtalk/thingtalk-canonical.examples -Derivation.showUtterance \
	-FeatureExtractor.featureDomains denotation -printAllPredictions \
	-printPredictedUtterances -executeAllDerivations -Parser.pruneErrorValues true \
	-Grammar.tags generate +Grammar.tags general 
