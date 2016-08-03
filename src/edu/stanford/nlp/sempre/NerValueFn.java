package edu.stanford.nlp.sempre;

import fig.basic.LispTree;

/**
 * Similar to FilterNerSpanFn, but it will return the NER values
 * instead of the raw tokens
 * 
 * @author gcampagn
 */
public class NerValueFn extends SemanticFn {
  // Accepted NER tag (PERSON, LOCATION, ORGANIZATION, etc)
  private String tag;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    tag = tree.child(1).value;
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        String value = ex.languageInfo.getNormalizedNerSpan(tag, c.getStart(), c.getEnd());
        if (value == null)
          return null;
        return new Derivation.Builder()
            .withCallable(c)
            .formula(new ValueFormula<>(new StringValue(value)))
            .type(SemType.entityType)
            .createDerivation();
      }
    };
  }

}
