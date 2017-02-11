package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;

public class AddCompositionFn extends SemanticFn {
  private boolean applyToAction;

  @Override
  public void init(LispTree tree) {
    super.init(tree);
    applyToAction = tree.child(1).value.equals("action");
  }

  @Override
  public DerivationStream call(final Example ex, final Callable c) {
    return new CompositionStream(ex, c);
  }

  private class CompositionStream extends MultipleDerivationStream {
    private final List<String> actionArgs;
    private final Iterator<String> actionArgsIter;
    private final Map<String, String> scope;
    private final Map<String, String> canonicals;
    private String currentActionArg;
    private Iterator<String> triggerArgs;
    private final RuleValue rv;
    private final Callable callable;

    public CompositionStream(Example ex, Callable c) {
      this.callable = c;

      Derivation left = c.child(0);
      if (left.value == null || !(left.value instanceof RuleValue))
        throw new IllegalArgumentException("AddCompositionFn used incorrectly");

      rv = (RuleValue) left.value;

      scope = new HashMap<>();
      canonicals = new HashMap<>();
      Set<String> boundArgs = new HashSet<>();
      if (rv.trigger != null) {
        for (ParamValue pv : rv.trigger.params) {
          if (pv.operator.equals("is"))
            boundArgs.add(pv.name.argname);
        }
        for (String name : rv.trigger.name.getArgNames()) {
          if (!boundArgs.contains(name)) {
            scope.put(name, rv.trigger.name.getArgType(name));
            canonicals.put(name, rv.trigger.name.getArgCanonical(name));
          }
        }
      }

      if (applyToAction && rv.query != null) {
        for (ParamValue pv : rv.query.params) {
          if (pv.operator.equals("is"))
            boundArgs.add(pv.name.argname);
        }
        for (String name : rv.query.name.getArgNames()) {
          if (!boundArgs.contains(name)) {
            scope.put(name, rv.query.name.getArgType(name));
            canonicals.put(name, rv.query.name.getArgCanonical(name));
          }
        }
      }

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
          if (!usedArgs.contains(name))
            actionArgs.add(name);
        }
      } else {
        for (String name : rv.query.name.getArgNames()) {
          if (!usedArgs.contains(name))
            actionArgs.add(name);
        }
      }

      actionArgsIter = actionArgs.iterator();
    }

    @Override
    public Derivation createDerivation() {
      while (true) {
        if (triggerArgs == null) {
          if (!actionArgsIter.hasNext())
            return null;

          currentActionArg = actionArgsIter.next();
          triggerArgs = scope.keySet().iterator();
        }
        if (!triggerArgs.hasNext()) {
          triggerArgs = null;
          continue;
        }

        String candidateTriggerArg = triggerArgs.next();
        String triggerType = scope.get(candidateTriggerArg);

        String actionType;
        if (applyToAction)
          actionType = rv.action.name.getArgType(currentActionArg);
        else
          actionType = rv.query.name.getArgType(currentActionArg);

        if (!actionType.equals(triggerType))
          continue;

        ParamNameValue actionParamName = new ParamNameValue(currentActionArg, actionType);
        String actionArgCanonical;
        if (applyToAction)
          actionArgCanonical = rv.action.name.argcanonicals.get(currentActionArg);
        else
          actionArgCanonical = rv.query.name.argcanonicals.get(currentActionArg);
        String triggerArgCanonical = canonicals.get(candidateTriggerArg);

        RuleValue clone = rv.clone();
        ParamValue nextPv = new ParamValue(actionParamName, "VarRef", "is",
            new NameValue("tt:param." + candidateTriggerArg));

        if (applyToAction)
          clone.action.add(nextPv);
        else
          clone.query.add(nextPv);

        Derivation deriv = new Derivation.Builder().withCallable(callable).formula(new ValueFormula<>(clone))
            .type(SemType.entityType)
            .canonicalUtterance(callable.child(0).canonicalUtterance + " " + callable.child(1).canonicalUtterance + " "
                + actionArgCanonical + " " + triggerArgCanonical)
            .nerUtterance(callable.child(0).nerUtterance + " " + callable.child(1).nerUtterance + " "
                + actionArgCanonical + " " + triggerArgCanonical)
            .createDerivation();

        if (ThingTalkFeatureComputer.opts.featureDomains.contains("thingtalk_composition") &&
            ThingTalkFeatureComputer.opts.logicalFormFeatures) {
          deriv.addFeature("thingtalk_composition", "names = " + currentActionArg + "---" + candidateTriggerArg);
          deriv.addFeature("thingtalk_composition", "type = " + actionType);

          // note that it's a different feature compared to param=%s.%s:%s
          // the latter is about values that are explicitly given by the user, this one is about values that
          // can be extracted from the context and it's a still a good idea to extract
          // param=%s.%s:%s in here would be a bad feature because it would overfit towards arguments that
          // are easy to specify explicitly (of which we see plenty cases in the thingpedia dataset)
          //
          // the interesting example is "get cat picture then send picture on gmail with url is image url"
          // because of the many examples of send picture on gmail that have message and subject prefilled,
          // if we use param=%s.%s:%s here we first try to match those to one of the several values from "get cat picture"
          // (eg image id, or link url), causing the good parse to fall off the beam
          if (applyToAction)
            deriv.addFeature("thingtalk_composition",
                String.format("composed_param=%s.%s:%s", rv.action.name.kind, rv.action.name.channelName,
                    currentActionArg));
          else
            deriv.addFeature("thingtalk_composition",
                String.format("composed_param=%s.%s:%s", rv.query.name.kind, rv.query.name.channelName,
                    currentActionArg));
        }

        return deriv;
      }
    };
  }
}
