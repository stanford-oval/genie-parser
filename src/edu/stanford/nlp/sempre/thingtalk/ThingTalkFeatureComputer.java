package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.FeatureComputer;
import edu.stanford.nlp.util.ArraySet;
import fig.basic.Option;

public class ThingTalkFeatureComputer implements FeatureComputer {
  public static class Options {
    @Option(gloss = "Set of paraphrasing feature domains to include")
    public Set<String> featureDomains = new HashSet<>();
  }

  public static Options opts = new Options();

  @Override
  public void extractLocal(Example ex, Derivation deriv) {
    if (opts.featureDomains.contains("anchorBoundaries"))
      extractAnchorBoundaries(ex, deriv);

    if (opts.featureDomains.contains("code"))
      extractCodeFeatures(ex, deriv);

    if (opts.featureDomains.contains("paramVerbAlign"))
      extractParamVerbAlign(ex, deriv);
  }

  private Collection<String> findVerbs(Example ex) {
    Collection<String> verbs = new ArrayList<>();

    for (int i = 0; i < ex.numTokens(); i++) {
      // from penn tree bank:
      //     VB  Verb, base form
      //     VBD     Verb, past tense
      //     VBG     Verb, gerund or present participle
      //     VBN     Verb, past participle
      //     VBP     Verb, non-3rd person singular present
      //     VBZ     Verb, 3rd person singular present 
      if (ex.posTag(i).equals("VB") || ex.posTag(i).equals("VBN") || ex.posTag(i).equals("VBP")
          || ex.posTag(i).equals("VBZ"))
        verbs.add(ex.token(i));
    }

    // GIANT HACK: for some reason corenlp recognizes post and tweet
    // as nouns (probably because it was trained before these two came
    // into mainstream use as verbs)
    // we add them as verbs if they appear in first position (which
    // would make them imperatives) - in other positions they are
    // either nouns or properly tagged by corenlp
    // we only do this if we didn't find a verb though
    if (verbs.size() == 0) {
      if (ex.numTokens() > 0 && ex.token(0).equals("post") || ex.token(0).equals("tweet"))
        verbs.add(ex.token(0));
    }

    return verbs;
  }

  private void extractParamVerbAlign(Example ex, Derivation deriv) {
    if (deriv.value == null)
      return;

    if (!(deriv.value instanceof ParamNameValue))
      return;

    ParamNameValue pnv = (ParamNameValue) deriv.value;

    // SEMI HACK: our goal is to make sure that we score high only
    // the parameters that are good with the current action/trigger/query
    // otherwise only the bad ones will be left in the beam and we'll
    // be left with no valid invocations after the correctness filter
    // unfortunately, we cannot look at "sibling" rules to see which
    // actions are more likely to be picked
    // instead, we look at the sentence globally and align ourselves
    // with the verbs, which we assume are good proxies for the action
    Collection<String> verbs = findVerbs(ex);

    for (String verb : verbs)
      deriv.addFeature("paramVerbAlign", verb + "---" + pnv.argname + "(" + pnv.type + ")");
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
    if (deriv.rule.isAnchored() && deriv.rule.lhs.equals("$StrValue")) {
      deriv.addFeature("anchorBoundaries", "left=" + tokenBefore(ex, deriv));
      deriv.addFeature("anchorBoundaries", "right=" + tokenAfter(ex, deriv));
    }
  }

  private void extractCodeFeatures(Example ex, Derivation deriv) {
    if (deriv.value == null)
      return;
    
    if (deriv.value instanceof ParametricValue)
      extractParametricValueFeatures(deriv, (ParametricValue) deriv.value);
  }

  private void extractParametricValueFeatures(Derivation deriv, ParametricValue pv) {
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
        deriv.addFeature("code", "dupIsParam=" + p.name.argname);
      else
        deriv.addFeature("code", "dupNonIsParam=" + p.name.argname);
    }
    deriv.addFeature("code", "nparams", params.size());
    
    // add a feature for each pair (channel name, parameter)
    // this is a bias towards choosing certain params for certain channels
    // for example, it's very likely that @phone.send_sms will be paired with
    // to, while it is less likely to be paired with message
    // because the sentence "send sms to foo" is more common that the sentence
    // "send sms saying foo"
    // this becomes even more important for trigger/queries, where parameters
    // are optional
    for (String p : params)
      deriv.addFeature("code", String.format("param=%s.%s:%s", pv.name.kind, pv.name.channelName, p));

    // don't add operator features for actions (because their operators are
    // always "is" and don't really have any meaning)
    if (pv instanceof ActionValue)
      return;

    for (ParamValue p : pv.params) {
      // add a feature for the pair (argtype, operator)
      // this is to bias against certain operators that, while legal, don't make much
      // sense, for example @thermostat.temperature(value), value = 73 F, because it will never be exactly 73 F
      // note that we use the actual ThingTalk type from param.name.type, not the looser
      // param.type
      deriv.addFeature("code", "operatortype=" + p.name.type + ":" + p.operator);

      // add a feature for the triple (argname, type, operator)
      // this is to bias towards certain operators for certain arguments
      // for example
      // this is a refinement of the previous, to catch cases like
      // @twitter.source(text, ...), text = "foo"
      // "=" is a fine operator for Strings in general, but for the specific case
      // of text it is wrong
      deriv.addFeature("code", "operator=" + p.name.argname + "," + p.name.type + ":" + p.operator);
    }
  }
}
