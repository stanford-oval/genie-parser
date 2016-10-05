package edu.stanford.nlp.sempre.api;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;

class OnlineLearnEntry {
  private final String utterance;
  private final String targetJson;

  public OnlineLearnEntry(String utterance, String targetJson) {
    this.utterance = utterance;
    this.targetJson = targetJson;
  }

  @Override
  public String toString() {
    return utterance + "\t" + targetJson;
  }
}

class LanguageContext {
  public final String tag;
  public final Parser parser;
  public final Params params;
  public final LanguageAnalyzer analyzer;
  public final QueryCache cache = new QueryCache(256);
  public final Learner learner;
  public final ExactMatcherLayer exactMatch;

  public LanguageContext(String tag) {
    this(tag, new CoreNLPAnalyzer(tag), null);
  }

  public LanguageContext(String tag, LanguageAnalyzer analyzer, ExactMatcherLayer exactMatch) {
    this.tag = tag;
    this.analyzer = analyzer;
    if (exactMatch != null)
      this.exactMatch = exactMatch;
    else
      this.exactMatch = new ExactMatcherLayer(tag, analyzer);

    Builder builder = new Builder();
    builder.buildForLanguage(tag);
    parser = builder.parser;
    params = builder.params;
    learner = new Learner(builder.parser, builder.params, null);
  }
}