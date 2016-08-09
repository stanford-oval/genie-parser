package edu.stanford.nlp.sempre;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import fig.basic.LispTree;

public class FilterRegexFn extends SemanticFn {
  public Pattern pattern;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    pattern = Pattern.compile(tree.child(1).value);
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        if (c.getStart() != c.getEnd() - 1)
          return null;

        Matcher matcher = pattern.matcher(ex.token(c.getStart()));
        if (!matcher.matches())
          return null;

        return new Derivation.Builder()
            .withCallable(c)
            .formula(new ValueFormula<>(new StringValue(matcher.group(1))))
            .type(SemType.stringType)
            .createDerivation();
      }
    };
  }

}
