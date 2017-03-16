package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.DerivationStream;
import edu.stanford.nlp.sempre.EmptyDerivationStream;
import edu.stanford.nlp.sempre.Example;
import fig.basic.LispTree;
import fig.basic.Option;

public class LocationLexiconFn extends AbstractLexiconFn<LocationValue> {
  public static class Options {
    @Option(gloss = "Filter by CoreNLP NER tag")
    public boolean filterNerTag = true;
  }

  public static Options opts = new Options();

  @Override
  public void init(LispTree tree) {
    super.init(tree);

    String languageTag = tree.child(1).value;
    setLexicon(LocationLexicon.getForLanguage(languageTag));
  }

  @Override
  public DerivationStream call(Example ex, Callable c) {
    if (opts.filterNerTag && !ex.languageInfo.isMaximalNerSpan("LOCATION", c.getStart(), c.getEnd()))
      return new EmptyDerivationStream();
    return super.call(ex, c);
  }
}
