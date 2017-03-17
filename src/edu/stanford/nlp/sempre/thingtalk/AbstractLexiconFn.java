package edu.stanford.nlp.sempre.thingtalk;

import java.util.Iterator;

import edu.stanford.nlp.sempre.*;

public abstract class AbstractLexiconFn<E extends Value> extends SemanticFn {
  private AbstractLexicon<E> lexicon;

  protected void setLexicon(AbstractLexicon<E> lexicon) {
    this.lexicon = lexicon;
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    String phrase = c.childStringValue(0);
    return new LexiconDerivationStream<>(ex, c, lexicon.lookup(phrase).iterator(), phrase);
  }

  public class LexiconDerivationStream<E extends Value> extends MultipleDerivationStream {
    private Example ex;
    private Callable callable;
    private Iterator<AbstractLexicon.Entry<E>> entries;
    private String phrase;

    public LexiconDerivationStream(Example ex, Callable c, Iterator<AbstractLexicon.Entry<E>> entries,
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

      LocationLexicon.Entry<E> entry = entries.next();
      Derivation deriv = new Derivation.Builder().withCallable(callable)
          .formula(entry.formula)
          .canonicalUtterance(entry.rawPhrase)
          .nerUtterance(entry.rawPhrase)
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
