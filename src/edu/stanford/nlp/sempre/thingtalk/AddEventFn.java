package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import com.google.common.base.Joiner;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

public class AddEventFn extends SemanticFn {
  private boolean applyToAction;
  private String eventVar;
  private String eventToken;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    applyToAction = tree.child(1).value.equals("action");
    eventVar = tree.child(2).value;
    if (tree.child(3).isLeaf())
      eventToken = tree.child(3).value;
    else
      eventToken = Joiner.on(' ').join(tree.child(3).children);
  }

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new CompositionStream(ex, c);
  }

  private class CompositionStream extends MultipleDerivationStream {
    private final List<String> actionArgs;
    private final Iterator<String> actionArgsIter;
    private final RuleValue rv;
    private final Callable callable;

    public CompositionStream(Example ex, Callable c) {
      this.callable = c;

      Derivation left = c.child(0);
      if (left.value == null || !(left.value instanceof RuleValue))
        throw new IllegalArgumentException("AddCompositionFn used incorrectly");

      rv = (RuleValue) left.value;

      Set<String> usedArgs = new HashSet<>();
      if (applyToAction) {
        for (ParamValue pv : rv.action.params) {
          usedArgs.add(pv.name.argname);
        }
      } else {
        for (ParamValue pv : rv.query.params) {
          if (pv.operator.equals("is"))
            usedArgs.add(pv.name.argname);
        }
      }

      actionArgs = new ArrayList<>();
      if (applyToAction) {
        for (String name : rv.action.name.getArgNames()) {
          String type = rv.action.name.getArgType(name);
          if (!type.equals("String"))
            continue;
          if (!usedArgs.contains(name))
            actionArgs.add(name);
        }
      } else {
        for (String name : rv.query.name.getArgNames()) {
          String type = rv.query.name.getArgType(name);
          if (!type.equals("String"))
            continue;
          if (!usedArgs.contains(name))
            actionArgs.add(name);
        }
      }

      actionArgsIter = actionArgs.iterator();
    }

    @Override
    public Derivation createDerivation() {
      while (true) {
        if (!actionArgsIter.hasNext())
          return null;

        String currentActionArg = actionArgsIter.next();
        ParamNameValue actionParamName = new ParamNameValue(currentActionArg, "String");
        String actionArgCanonical;
        if (applyToAction)
          actionArgCanonical = rv.action.name.argcanonicals.get(currentActionArg);
        else
          actionArgCanonical = rv.query.name.argcanonicals.get(currentActionArg);

        RuleValue clone = rv.clone();
        ParamValue nextPv = new ParamValue(actionParamName, "VarRef", "is",
            new NameValue("tt:param." + eventVar));

        if (applyToAction)
          clone.action.add(nextPv);
        else
          clone.query.add(nextPv);

        Derivation deriv = new Derivation.Builder().withCallable(callable).formula(new ValueFormula<>(clone))
            .type(SemType.entityType)
            .canonicalUtterance(callable.child(0).canonicalUtterance + " " + callable.child(1).canonicalUtterance + " "
                    + actionArgCanonical + " " + eventToken)
            .nerUtterance(callable.child(0).nerUtterance + " " + callable.child(1).nerUtterance + " "
                + actionArgCanonical + " " + eventToken)
            .createDerivation();

        deriv.addFeature("thingtalk_composition", "names = " + currentActionArg + "---" + eventVar);

        // some triggers are more prone to formatting than others
        // add a prior to that
        if (applyToAction && rv.query != null)
          deriv.addFeature("thingtalk_composition",
              String.format("formatted_trigger=%s:%s", rv.query.name.kind, rv.query.name.channelName));
        else
          deriv.addFeature("thingtalk_composition",
              String.format("formatted_trigger=%s:%s", rv.trigger.name.kind, rv.trigger.name.channelName));

        return deriv;
      }
    };
  }
}
