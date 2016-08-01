package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class AddOperatorFn extends SemanticFn {
  private static boolean operatorOk(String paramType, String operator) {
    switch (operator) {
    case "is":
      return true;
    case "contains":
      return paramType.equals("String");
    case "has":
      return paramType.startsWith("Array(");
    case ">":
    case "<":
      return paramType.equals("Number") || paramType.startsWith("Measure(");
    default:
      return true;
    }
  }

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new SingleDerivationStream() {
      @Override
      public Derivation createDerivation() {
        Derivation left = c.child(0);
        Derivation right = c.child(1);
        
        // FIXME check this is used correctly
        
        LambdaFormula lf2 = (LambdaFormula) left.formula;
        LambdaFormula lf1 = (LambdaFormula) lf2.body;
        CallFormula cf = (CallFormula) lf1.body;
        ValueFormula<?> vf = (ValueFormula<?>) cf.args.get(1);
        
        ParamNameValue param = (ParamNameValue) vf.value;
        StringValue operator = (StringValue) right.value;
        
        if (!operatorOk(param.type, operator.value))
          return null;

        return new Derivation.Builder().withCallable(c).formula(Formulas.lambdaApply(lf2, right.formula))
            .type(SemType.anyAnyFunc)
            .createDerivation();
      }
    };
  }
}
