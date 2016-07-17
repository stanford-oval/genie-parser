package edu.stanford.nlp.sempre;

import edu.stanford.nlp.sempre.Derivation.Cacheability;

/**
 * Maps a string to a Date.
 *
 * @author Percy Liang
 */
public class DateFn extends SemanticFn {
  @Override
public DerivationStream call(final Example ex, final Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        String value = ex.languageInfo.getNormalizedNerSpan("DATE", c.getStart(), c.getEnd());
        if (value == null)
          return null;
        DateValue dateValue = DateValue.parseDateValue(value);
        if (dateValue == null)
          return null;
        return new Derivation.Builder()
                .withCallable(c)
                .formula(new ValueFormula<>(dateValue))
                .type(SemType.dateType)
						.meetCache(
								value.equals("PRESENT_REF") ? Cacheability.NON_DETERMINISTIC : Cacheability.CACHEABLE)
                .createDerivation();
      }
    };
  }
}
