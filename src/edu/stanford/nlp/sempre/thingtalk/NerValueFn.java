package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

/**
 * Similar to FilterNerSpanFn, but it will return the NER values
 * instead of the raw tokens, and will return a TypedStringValue instead of a
 * StringValue
 * 
 * @author gcampagn
 */
public class NerValueFn extends SemanticFn {
  // Accepted NER tag (PERSON, LOCATION, ORGANIZATION, etc)
  private String tag;
  private String type;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    tag = tree.child(1).value;
    type = tree.child(2).value;
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
            .formula(new ValueFormula<>(new TypedStringValue(type, value)))
            .nerUtterance(tag)
            .type(SemType.entityType)
            .createDerivation();
      }
    };
  }

}
