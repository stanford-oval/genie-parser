package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class AddValueFn extends SemanticFn {
  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        Derivation left = c.child(0);
        Derivation right = c.child(1);

        // FIXME check this is used correctly

        LambdaFormula lf = (LambdaFormula) left.formula;
        CallFormula cf = (CallFormula) lf.body;
        ValueFormula<?> vf = (ValueFormula<?>) cf.args.get(1);

        ParamNameValue param = (ParamNameValue) vf.value;

        String haveType = ThingTalk.typeFromValue(right.value);
        if (!ArgFilterHelpers.typeOk(haveType, param.type, right.value) &&
            !ArgFilterHelpers.typeOkArray(haveType, param.type, right.value))
          return null;

        return new Derivation.Builder().withCallable(c).formula(Formulas.lambdaApply(lf, right.formula))
            .type(SemType.anyType)
            .createDerivation();
      }
    };
  }
}
