package edu.stanford.nlp.sempre.api;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.LanguageAnalyzer;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaDataset;

class ExactMatcherLayer {
  private static class ExactMatchKey {
    private List<String> tokens;

    public ExactMatchKey(Example ex) {
      tokens = ex.getTokens();
    }

    public ExactMatchKey(LanguageAnalyzer analyzer, String utterance) {
      Example ex = new Example.Builder().setUtterance(utterance).createExample();
      ex.preprocess(analyzer);
      tokens = ex.getTokens();
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + tokens.hashCode();
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      ExactMatchKey other = (ExactMatchKey) obj;
      if (!tokens.equals(other.tokens))
        return false;
      return true;
    }
  }

  private final Map<ExactMatchKey, String> mem = new ConcurrentHashMap<>();
  private final String languageTag;
  private final LanguageAnalyzer analyzer;

  public ExactMatcherLayer(String languageTag, LanguageAnalyzer analyzer) {
    this.languageTag = languageTag;
    this.analyzer = analyzer;
  }

  public void load() throws IOException {
    ThingpediaDataset.getRawExamples(languageTag, (utterance, json) -> {
      mem.put(new ExactMatchKey(analyzer, utterance), json);
    });
  }

  public void store(Example ex, String targetJson) {
    mem.put(new ExactMatchKey(ex), targetJson);
  }

  public void store(String utterance, String targetJson) {
    mem.put(new ExactMatchKey(analyzer, utterance), targetJson);
  }

  public String hit(Example query) {
    return mem.get(new ExactMatchKey(query));
  }
}
