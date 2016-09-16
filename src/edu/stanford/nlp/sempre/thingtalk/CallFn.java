package edu.stanford.nlp.sempre.thingtalk;

import java.util.Collections;

import com.beust.jcommander.internal.Lists;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

public class CallFn extends SemanticFn {
  private String function;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    function = tree.child(1).value;
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        Formula f;
        int n = c.getChildren().size(); 
        if (n == 1)
          f = new CallFormula(function, Collections.singletonList(c.child(0).formula));
        else if (n == 2)
          f = new CallFormula(function, Lists.newArrayList(c.child(0).formula, c.child(1).formula));
        else
          throw new RuntimeException();

        return new Derivation.Builder()
            .withCallable(c)
            .formula(f)
            .type(SemType.anyType)
            .createDerivation();
      }
    };
  }

}
