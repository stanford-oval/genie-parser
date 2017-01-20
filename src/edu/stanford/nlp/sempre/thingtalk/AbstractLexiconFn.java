package edu.stanford.nlp.sempre.thingtalk;

import java.util.Iterator;

import edu.stanford.nlp.sempre.*;

public abstract class AbstractLexiconFn extends SemanticFn {
  private AbstractLexicon lexicon;

  protected void setLexicon(AbstractLexicon lexicon) {
    this.lexicon = lexicon;
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    String phrase = c.childStringValue(0);
    return new LexiconDerivationStream(ex, c, lexicon.lookup(phrase), phrase);
  }

  public class LexiconDerivationStream extends MultipleDerivationStream {
    private Example ex;
    private Callable callable;
    private Iterator<AbstractLexicon.Entry> entries;
    private String phrase;

    public LexiconDerivationStream(Example ex, Callable c, Iterator<AbstractLexicon.Entry> entries,
        String phrase) {
      this.ex = ex;
      this.callable = c;
      this.entries = entries;
      this.phrase = phrase;
    }

    @Override
    public Derivation createDerivation() {
      if (!entries.hasNext())
        return null;

      LocationLexicon.Entry entry = entries.next();
      Derivation deriv = new Derivation.Builder().withCallable(callable)
          .formula(entry.formula)
          .canonicalUtterance(entry.rawPhrase)
          .nerUtterance(entry.nerTag)
          .type(SemType.entityType)
          .createDerivation();

      // Doesn't generalize, but add it for now, otherwise not separable
      if (FeatureExtractor.containsDomain("lexAlign"))
        deriv.addFeature("lexAlign", phrase + " --- " + entry.formula);

      if (SemanticFn.opts.trackLocalChoices)
        deriv.addLocalChoice("SimpleLexiconFn " + deriv.startEndString(ex.getTokens()) + " " + entry);

      return deriv;
    }
  }
}
