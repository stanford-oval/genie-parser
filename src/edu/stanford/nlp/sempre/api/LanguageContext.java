package edu.stanford.nlp.sempre.api;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;

class LanguageContext {
  public final String tag;
  public final Parser parser;
  public final Params params;
  public final LanguageAnalyzer analyzer;
  public final QueryCache cache = new QueryCache(256);
  public final Learner learner;

  public LanguageContext(String tag) {
    this.tag = tag;

    Builder builder = new Builder();
    builder.buildForLanguage(tag);
    parser = builder.parser;
    params = builder.params;
    analyzer = new CoreNLPAnalyzer(tag);
    learner = new Learner(builder.parser, builder.params, null);
  }
}