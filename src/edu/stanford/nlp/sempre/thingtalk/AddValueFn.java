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

  private static boolean operatorOk(Type paramType, String operator) {
    switch (operator) {
    case "is":
      return !(paramType instanceof Type.Array);
    case "contains":
      return paramType == Type.String;
    case "has":
      return paramType instanceof Type.Array;
    case ">":
    case "<":
      return paramType == Type.Number || paramType instanceof Type.Measure;
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

    private Derivation findLastAnchoredArg(Derivation deriv) {
      Derivation lastArg = null;
      if (deriv.children != null && deriv.children.size() == 2)
        lastArg = deriv.child(1);
      else
        return null;
      if (lastArg.getCat().equals("$PersonValue"))
        return null;
      if (lastArg.spanStart == -1 || lastArg.spanEnd == -1)
        return findLastAnchoredArg(deriv.child(0));
      else
        return lastArg;
    }

    @Override
    public Derivation createDerivation() {
      Derivation lastArg = findLastAnchoredArg(callable.child(0));
      if (lastArg != null && !lastArg.isLeftOf(callable.child(1)))
        return null;

      while (true) {
        if (!argnameIter.hasNext())
          return null;

        currentArgname = argnameIter.next();
        if (operator.equals("is") && invocation.hasParamName(currentArgname))
          continue;
        if (currentArgname.startsWith("__")) // compat argument
          continue;

        ParamNameValue param = new ParamNameValue(currentArgname, invocation.name.getArgType(currentArgname));

        if (!operatorOk(param.type, operator))
          continue;

        Derivation left = callable.child(0);
        Derivation right = callable.child(1);
        Value toAdd = right.value;
        String sempreType = ThingTalk.typeFromValue(toAdd);
        Type haveType = Type.fromString(sempreType);

        if (!ArgFilterHelpers.typeOk(haveType, param.type, toAdd) &&
            !ArgFilterHelpers.typeOkArray(haveType, param.type, toAdd))
          continue;

        ParamValue pv = new ParamValue(param, sempreType, operator, toAdd);

        ParametricValue newInvocation = invocation.clone();
        newInvocation.add(pv);

        String opPart = opToken.length() > 0 ? " " + opToken + " " : " ";

        String canonical = left.canonicalUtterance + " " + withToken + " " +
            invocation.name.getArgCanonical(currentArgname) + opPart
            + right.canonicalUtterance;
        String nerCanonical = left.nerUtterance + " " + withToken + " "
            + invocation.name.getArgCanonical(currentArgname) + opPart
            + right.nerUtterance;

        Derivation.Builder bld = new Derivation.Builder()
            .withCallable(callable)
            .formula(new ValueFormula<>(newInvocation))
            .type(SemType.entityType)
            .canonicalUtterance(canonical)
            .nerUtterance(nerCanonical);
        Derivation deriv = bld.createDerivation();

        int spanMin = callable.child(1).spanStart;
        if (spanMin > 0 && ThingTalkFeatureComputer.opts.featureDomains.contains("thingtalk_params_leftword"))
          deriv.addFeature("thingtalk_params_leftword",
              ex.token(spanMin - 1) + "---" + pv.name.argname + "," + pv.operator);

        return deriv;
      }
    }
  }
}
