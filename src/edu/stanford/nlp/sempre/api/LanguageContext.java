package edu.stanford.nlp.sempre.api;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;

class LanguageContext {
  public final Parser parser;
  public final Params params;
  public final LanguageAnalyzer analyzer;
  public final QueryCache cache = new QueryCache(256);

  public LanguageContext(String tag) {
    Builder builder = new Builder();
    builder.buildForLanguage(tag);
    parser = builder.parser;
    params = builder.params;
    analyzer = new CoreNLPAnalyzer(tag);
  }
}