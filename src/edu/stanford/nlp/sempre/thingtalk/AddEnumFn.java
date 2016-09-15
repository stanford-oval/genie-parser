package edu.stanford.nlp.sempre.thingtalk;

import java.util.Arrays;
import java.util.Iterator;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

public class AddEnumFn extends SemanticFn {
  private String opToken;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    opToken = tree.child(1).value;
  }

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new AddEnumStream(ex, c);
  }

  private class AddEnumStream extends MultipleDerivationStream {
    private final Example ex;
    private final Callable callable;
    private final ParametricValue invocation;
    private final Iterator<String> argnameIter;
    private String currentArgname;
    private Iterator<String> currentEnumIter;

    public AddEnumStream(Example ex, Callable callable) {
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
        if (currentEnumIter == null) {
          if (!argnameIter.hasNext())
            return null;

          currentArgname = argnameIter.next();
          if (invocation.hasParamName(currentArgname))
            continue;

          String enumType = invocation.name.getArgType(currentArgname);
          if (!enumType.startsWith("Enum("))
            continue;

          String[] enumList = enumType.substring("Enum(".length(), enumType.length()-1).split(",");
          currentEnumIter = Arrays.asList(enumList).iterator();
        }
        if (!currentEnumIter.hasNext()) {
          currentEnumIter = null;
          continue;
        }

        ParamNameValue param = new ParamNameValue(currentArgname, invocation.name.getArgType(currentArgname));
        String enumValue = currentEnumIter.next();

        Derivation left = callable.child(0);
        Value toAdd = new StringValue(enumValue);

        ParamValue pv = new ParamValue(param, "Enum", "is", toAdd);

        ParametricValue newInvocation = invocation.clone();
        newInvocation.add(pv);

        String canonical = left.canonicalUtterance + " arg " +
            invocation.name.getArgCanonical(currentArgname) + " " + opToken + " "
            + enumValue;

        Derivation.Builder bld = new Derivation.Builder().withCallable(callable)
            .formula(new ValueFormula<>(newInvocation)).type(SemType.entityType)
            .canonicalUtterance(canonical);
        return bld.createDerivation();
      }
    }
  }
}
