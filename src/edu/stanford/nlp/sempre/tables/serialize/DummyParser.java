package edu.stanford.nlp.sempre.tables.serialize;

import edu.stanford.nlp.sempre.*;
import gnu.trove.map.hash.TObjectDoubleHashMap;

public class DummyParser extends Parser {

  public DummyParser(Spec spec) {
    super(spec);
  }

  @Override
  public ParserState newParserState(Params params, Example ex, boolean computeExpectedCounts) {
    return new DummyParserState(this, params, ex, computeExpectedCounts);
  }

}

class DummyParserState extends ParserState {

  public DummyParserState(Parser parser, Params params, Example ex, boolean computeExpectedCounts) {
    super(parser, params, ex, computeExpectedCounts);
  }

  @Override
  public void infer() {
    // Assume that the example already has all derivations.
    for (Derivation deriv : ex.predDerivations) {
      featurizeAndScoreDerivation(deriv);
      deriv.compatibility = parser.valueEvaluator.getCompatibility(ex.targetValue, deriv.value);
      predDerivations.add(deriv);
    }
    if (computeExpectedCounts) {
      expectedCounts = new TObjectDoubleHashMap<>();
      ParserState.computeExpectedCounts(predDerivations, expectedCounts);
    }
  }

}
