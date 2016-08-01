package edu.stanford.nlp.sempre.thingtalk;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

import edu.stanford.nlp.sempre.*;

public class ApplyFn extends SemanticFn {

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    final Derivation left = c.child(0);
    if (left.value == null || !(left.value instanceof ParametricValue))
      throw new IllegalArgumentException("ApplyFn used incorrectly");

    final Iterator<Derivation> pseudoRight;
    if (c.getChildren().size() >= 2) {
      Derivation right = c.child(1);

      if (right.value == null || !(right.value instanceof ParamNameValue))
        throw new IllegalArgumentException("ApplyFn used incorrectly");

      if (!FilterInvalidInvocationFn.paramTypeOk((ParamNameValue) right.value, ((ParametricValue) left.value).name))
        return new EmptyDerivationStream();

      pseudoRight = Collections.singletonList(right).iterator();
    } else {
      // try all possible arguments to this channel
      ParametricValue pv = (ParametricValue) left.value;
      pseudoRight = pv.name.argtypes.entrySet().stream().map(e -> {
        String argname = e.getKey();
        String argtype = e.getValue();

        String argcanonical = argname.replace("_", " ").replaceAll("([^A-Z])([A-Z])", "$1 $2");

        // build a pseudo-derivation with the formula and the canonical
        Derivation.Builder bld = new Derivation.Builder().canonicalUtterance(argcanonical)
            .formula(new ValueFormula<>(new ParamNameValue(argname, argtype)));
        return bld.createDerivation();
      }).iterator();
    }

    return new MultipleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        if (!pseudoRight.hasNext())
          return null;

        Derivation right = pseudoRight.next();
        // build a call formula that calls ThingTalk.addParam with the
        // action name, the param name, the operator and the value
        Formula[] params = new Formula[] { left.formula, right.formula, new VariableFormula("op"),
            new VariableFormula("value") };
        CallFormula cf = new CallFormula(ThingTalk.class.getName() + ".addParam", Arrays.asList(params));

        // build a lambda with two arguments, first the value and then
        // the operator
        LambdaFormula l1 = new LambdaFormula("value", cf);
        LambdaFormula l2 = new LambdaFormula("op", l1);

        Derivation.Builder bld = new Derivation.Builder().withCallable(c).formula(l2).type(SemType.anyAnyAnyFunc)
            .canonicalUtterance(left.canonicalUtterance + " " + right.canonicalUtterance);
        return bld.createDerivation();
      }
    };
  }

}
