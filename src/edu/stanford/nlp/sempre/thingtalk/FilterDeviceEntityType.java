package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class FilterDeviceEntityType extends SemanticFn {

  @Override
  public DerivationStream call(Example ex, Callable c) {
    return new SingleDerivationStream() {

      @Override
      public Derivation createDerivation() {
        Derivation child = c.child(0);
        TypedStringValue value = (TypedStringValue) child.value;
        if (!value.type.equals("Entity(tt:device)"))
          return null;

        return new Derivation.Builder()
            .withCallable(c)
            .withFormulaFrom(child)
            .createDerivation();
      }
    };
  }

}
