package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Derivation;
import edu.stanford.nlp.sempre.DerivationPruner;
import edu.stanford.nlp.sempre.DerivationPruningComputer;

public class ThingTalkDerivationPruner extends DerivationPruningComputer {
  public ThingTalkDerivationPruner(DerivationPruner pruner) {
    super(pruner);
  }

  @Override
  public boolean isPruned(Derivation deriv) {
    if (!deriv.isRootCat())
      return false;

    String canonical = deriv.canonicalUtterance;
    String tokens[] = canonical.split("\\s+");

    for (String token : tokens) {
      if (token.startsWith("$"))
        return true;
    }

    return false;
  }
}
