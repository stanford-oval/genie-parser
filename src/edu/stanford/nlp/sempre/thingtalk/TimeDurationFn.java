package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

/**
 * Created by rakesh on 9/13/16.
 */
public class TimeDurationFn extends SemanticFn {
  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        String value = ex.languageInfo.getNormalizedNerSpan("SET", c.getStart(), c.getEnd());
        if (value == null)
          value = ex.languageInfo.getNormalizedNerSpan("DURATION", c.getStart(), c.getEnd());
          if(value == null)
            return null;
        NumberValue numValue = NumberValue.parseDurationValue(value);
        if (numValue == null)
          return null;
        return new Derivation.Builder()
                .withCallable(c)
                .formula(new ValueFormula<>(numValue))
                .type(SemType.dateType)
						.meetCache(
								value.equals("PRESENT_REF") ? Derivation.Cacheability.NON_DETERMINISTIC : Derivation.Cacheability.CACHEABLE)
                .createDerivation();
      }
    };
  }
}
