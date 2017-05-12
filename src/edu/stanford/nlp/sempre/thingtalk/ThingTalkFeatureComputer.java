package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashSet;
import java.util.Set;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.util.ArraySet;
import fig.basic.Option;

public class ThingTalkFeatureComputer implements FeatureComputer {
  public static class Options {
    @Option(gloss = "Set of paraphrasing feature domains to include")
    public Set<String> featureDomains = new HashSet<>();

    @Option
    public boolean logicalFormFeatures = true;
  }

  public static Options opts = new Options();

  @Override
  public void extractLocal(Example ex, Derivation deriv) {
    if (opts.featureDomains.contains("anchorBoundaries"))
      extractAnchorBoundaries(ex, deriv);

    if (opts.featureDomains.contains("strvalue"))
      extractStrValue(ex, deriv);

    if (opts.featureDomains.contains("thingtalk_root"))
      extractRootCodeFeatures(ex, deriv);

    if (opts.featureDomains.contains("thingtalk_params") && opts.logicalFormFeatures)
      extractCodeFeatures(ex, deriv);

    extractMeasureFeatures(ex, deriv);
    extractPersonFeatures(ex, deriv);
  }

  private void extractPersonFeatures(Example ex, Derivation deriv) {
    if (!deriv.getCat().equals("$PersonValue"))
      return;

    Derivation usernameDeriv = deriv.child(0);
    int usernameTokenPos = usernameDeriv.getStart();
    assert usernameTokenPos >= 0;
    if (usernameTokenPos < ex.numTokens() - 1) {
      String nextToken = ex.token(usernameTokenPos + 1);
      deriv.addFeature("username_bigram", "USERNAME " + nextToken);
    }
    if (usernameTokenPos > 0) {
      String previousToken = ex.token(usernameTokenPos - 1);
      deriv.addFeature("username_bigram", previousToken + " USERNAME");
    }
  }

  private void extractMeasureFeatures(Example ex, Derivation deriv) {
    if (!deriv.getCat().equals("$MeasureValue"))
      return;

    Derivation numberDeriv = deriv.child(0);
    Derivation unitDeriv = deriv.child(1);

    NumberValue nv = (NumberValue)deriv.value;
    deriv.addFeature("units", nv.unit + "---" + tokenAfter(ex, numberDeriv));
  }

  private String tokenBefore(Example ex, Derivation deriv) {
    if (deriv.start == 0)
      return null;
    if (ex.token(deriv.start - 1).equals("''") || ex.token(deriv.start - 1).equals("``"))
      return deriv.start > 1 ? ex.token(deriv.start - 2) : null;
    else
      return ex.token(deriv.start - 1);
  }

  private String tokenAfter(Example ex, Derivation deriv) {
    int n = ex.numTokens();
    if (deriv.end == n)
      return null;
    if (ex.token(deriv.end).equals("''"))
      return deriv.end < n - 1 ? ex.token(deriv.end + 1) : null;
    else
      return ex.token(deriv.end);
  }

  private void extractAnchorBoundaries(Example ex, Derivation deriv) {
    if (!deriv.rule.isAnchored() || !"$StrValue".equals(deriv.rule.lhs))
      return;

    deriv.addFeature("anchorBoundaries", "left=" + tokenBefore(ex, deriv));
    deriv.addFeature("anchorBoundaries", "right=" + tokenAfter(ex, deriv));
  }

  private void extractStrValue(Example ex, Derivation deriv) {
    if (!deriv.rule.isAnchored() || !"$StrValue".equals(deriv.rule.lhs))
      return;

    deriv.addFeature("strvalue", "size", deriv.end - deriv.start);

    // often times string values are proper names, eg when saying
    // show soccer matches of Juventus
    // or
    // monitor tweets from POTUS
    for (int i = deriv.start; i < deriv.end; i++)
      deriv.addFeature("strvalue", "pos=" + ex.languageInfo.posTags.get(i));

    // often CoreNLP even comes in our help by recognizing organizations
    // (when properly spelled)
    // let's not waste that effort
    if (ex.languageInfo.isMaximalNerSpan("ORGANIZATION", deriv.start, deriv.end) ||
        ex.languageInfo.isMaximalNerSpan("PERSON", deriv.start, deriv.end))
      deriv.addFeature("strvalue", "isEntity");
  }

  private void extractRootCodeFeatures(Example ex, Derivation deriv) {
    if (deriv.value == null)
      return;
    if (!deriv.isRootCat())
      return;

    Value value = deriv.child(0).value;

    if (value instanceof TriggerValue)
      deriv.addGlobalFeature("thingtalk_root", ex.token(0) + "---trigger");
    else if (value instanceof QueryValue)
      deriv.addGlobalFeature("thingtalk_root", ex.token(0) + "---query");
    else if (value instanceof ActionValue)
      deriv.addGlobalFeature("thingtalk_root", ex.token(0) + "---action");
  }

  private void extractCodeFeatures(Example ex, Derivation deriv) {
    if (deriv.value == null)
      return;

    Value value = deriv.value;
    if (deriv.isRootCat()) {
      // HACK: the root is a StringValue in json form, so we descend one level
      // to find the value
      value = deriv.child(0).value;
    }

    if (value instanceof ParametricValue) {
      extractParametricValueFeatures(deriv, (ParametricValue) value);
    } else if (value instanceof RuleValue) {
      RuleValue rule = (RuleValue) value;
      extractParametricValueFeatures(deriv, rule.trigger);
      extractParametricValueFeatures(deriv, rule.query);
      extractParametricValueFeatures(deriv, rule.action);
    }
  }

  private void extractParametricValueFeatures(Derivation deriv, ParametricValue pv) {
    if (pv == null)
      return;
    
    ArraySet<String> params = new ArraySet<>();
    ArraySet<ParamValue> duplicates = new ArraySet<>();
    for (ParamValue p : pv.params) {
      if (params.contains(p.name.argname))
        duplicates.add(p);
      else
        params.add(p.name.argname);
    }

    for (ParamValue p : duplicates) {
      if (p.operator.equals("is"))
        deriv.addGlobalFeature("thingtalk_complexity", "dupIsParam=" + p.name.argname);
      else
        deriv.addGlobalFeature("thingtalk_complexity", "dupNonIsParam=" + p.name.argname);
    }
    deriv.addGlobalFeature("thingtalk_complexity", "nparams", params.size());
    
    // add a feature for each pair (channel name, parameter)
    // this is a bias towards choosing certain params for certain channels
    // for example, it's very likely that @phone.send_sms will be paired with
    // to, while it is less likely to be paired with message
    // because the sentence "send sms to foo" is more common that the sentence
    // "send sms saying foo"
    // this becomes even more important for trigger/queries, where parameters
    // are optional
    for (ParamValue p : pv.params) {
      if (p.tt_type.equals("VarRef"))
        continue;

      deriv.addGlobalFeature("thingtalk_params",
          String.format("param=%s.%s:%s", pv.name.kind, pv.name.channelName, p.name.argname));
    }

    // don't add operator features for actions (because their operators are
    // always "is" and don't really have any meaning)
    //if (pv instanceof ActionValue)
    //  return;

    for (ParamValue p : pv.params) {
      if (p.tt_type.equals("VarRef"))
        continue;

      // add a feature for the pair (argtype, operator)
      // this is to bias against certain operators that, while legal, don't make much
      // sense, for example @thermostat.temperature(value), value = 73 F, because it will never be exactly 73 F

      if (!p.tt_type.equals("String"))
          deriv.addGlobalFeature("thingtalk_params", "operatortype=" + p.tt_type + ":" + p.operator);

      // add a feature for the triple (argname, type, operator)
      // this is to bias towards certain operators for certain arguments
      // for example
      // this is a refinement of the previous, to catch cases like
      // @twitter.source(text, ...), text = "foo"
      // "=" is a fine operator for Strings in general, but for the specific case
      // of text it is wrong
      deriv.addGlobalFeature("thingtalk_params", "operator=" + p.name.argname + ":" + p.operator);
    }
  }
}
