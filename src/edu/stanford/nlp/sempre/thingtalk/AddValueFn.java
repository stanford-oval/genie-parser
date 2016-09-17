package edu.stanford.nlp.sempre.thingtalk;

import java.util.Iterator;

import com.google.common.base.Joiner;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

public class AddValueFn extends SemanticFn {
  private boolean isAction;
  private String withToken;
  private String opToken;
  private String operator;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    isAction = tree.child(1).value.equals("action");
    withToken = tree.child(2).value;

    if (tree.child(3).isLeaf())
      opToken = tree.child(3).value;
    else
      opToken = Joiner.on(' ').join(tree.child(3).children);
    operator = tree.child(4).value;
  }

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new AddValueStream(ex, c);
  }

  private static boolean operatorOk(String paramType, String operator) {
    switch (operator) {
    case "is":
      return !paramType.startsWith("Array(");
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

  private class AddValueStream extends MultipleDerivationStream {
    private final Example ex;
    private final Callable callable;
    private final ParametricValue invocation;
    private final Iterator<String> argnameIter;
    private String currentArgname;

    public AddValueStream(Example ex, Callable callable) {
      this.ex = ex;
      this.callable = callable;

      Derivation left = callable.child(0);
      if (left.value == null || !(left.value instanceof ParametricValue))
        throw new IllegalArgumentException("AddValueFn used incorrectly");

      invocation = (ParametricValue) left.value;
      argnameIter = invocation.name.argtypes.keySet().iterator();
    }

    @Override
    public Derivation createDerivation() {
      while (true) {
        if (!argnameIter.hasNext())
          return null;

        currentArgname = argnameIter.next();
        if (isAction && invocation.hasParamName(currentArgname))
          continue;

        ParamNameValue param = new ParamNameValue(currentArgname, invocation.name.getArgType(currentArgname));

        if (!operatorOk(param.type, operator))
          continue;

        Derivation left = callable.child(0);
        Derivation right = callable.child(1);
        Value toAdd = right.value;
        String haveType = ThingTalk.typeFromValue(toAdd);

        if (!ArgFilterHelpers.typeOk(haveType, param.type, toAdd) &&
            !ArgFilterHelpers.typeOkArray(haveType, param.type, toAdd))
          continue;

        ParamValue pv = new ParamValue(param, haveType, operator, toAdd);

        ParametricValue newInvocation = invocation.clone();
        newInvocation.add(pv);

        String opPart = opToken.length() > 0 ? " " + opToken + " " : " ";

        String canonical = left.canonicalUtterance + " " + withToken + " " +
            invocation.name.getArgCanonical(currentArgname) + opPart
            + right.canonicalUtterance;

        Derivation.Builder bld = new Derivation.Builder().withCallable(callable)
            .formula(new ValueFormula<>(newInvocation)).type(SemType.entityType)
            .canonicalUtterance(canonical);
        return bld.createDerivation();
      }
    }
  }
}
