package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class FilterInvalidInvocationFn extends SemanticFn {
  private static boolean operatorOk(String operator, boolean isAction) {
    if (isAction)
      return operator.equals("is");
    else
      return true;
  }

  static boolean paramTypeOk(ParamNameValue param, ChannelNameValue channel) {
    return param.type.equals(channel.getArgType(param.argname));
  }

  private static boolean valueOk(Value value) {
    if (!(value instanceof ParametricValue))
      return true;

    boolean isAction = value instanceof ActionValue;

    ParametricValue pv = (ParametricValue) value;
    for (ParamValue param : pv.params) {
      if (!ArgFilterHelpers.valueOk(param))
        return false;
      if (!paramTypeOk(param.name, pv.name) || !operatorOk(param.operator, isAction))
        return false;
    }

    return true;
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        Derivation child = c.child(0);
        if (child.getValue() != null && !valueOk(child.getValue()))
          return null;

        return new Derivation.Builder()
            .withCallable(c)
            .withFormulaFrom(child)
            .createDerivation();
      }
    };
  }

}
