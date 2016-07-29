package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashSet;
import java.util.Set;

import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.FeatureComputer;
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
      extractAnchorBoundaries(ex, deriv, deriv);
  }

  private void extractAnchorBoundaries(Example ex, Derivation deriv, Derivation grandparent) {
    if (deriv.rule.isAnchored()) {
      if (deriv.start > 0)
        grandparent.addFeature("anchorBoundary", "left=" + ex.token(deriv.start - 1));
      else
        grandparent.addFeature("anchorBoundary", "left=null");
      if (deriv.end < ex.numTokens() - 1)
        grandparent.addFeature("anchorBoundary", "right=" + ex.token(deriv.start + 1));
      else
        grandparent.addFeature("anchorBoundary", "right=null");
    } else {
      for (Derivation child : deriv.children)
        extractAnchorBoundaries(ex, child, grandparent);
    }
  }

}
