package edu.stanford.nlp.sempre.api;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import edu.stanford.nlp.sempre.thingtalk.ThingpediaDataset;

class ExactMatcherLayer {
  private final Map<String, String> mem = new ConcurrentHashMap<>();
  private final String languageTag;

  public ExactMatcherLayer(String languageTag) {
    this.languageTag = languageTag;
  }

  public void load() throws IOException {
    ThingpediaDataset.getRawExamples(mem, languageTag);
  }

  public void store(String query, String targetJson) {
    mem.put(query, targetJson);
  }

  public void expire(String query) {
    mem.remove(query);
  }

  public String hit(String query) {
    return mem.get(query);
  }
}
