package edu.stanford.nlp.sempre.api;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;

class ExactMatcherLayer {
  private final Map<String, String> mem = new ConcurrentHashMap<>();

  public void load(String filename) throws IOException {
    try {
      for (String line : IOUtils.readLines(filename)) {
        String[] parts = line.split("\t");
        store(parts[0], parts[1]);
      }
    } catch (RuntimeIOException e) {
      if (e.getCause() instanceof IOException)
        throw ((IOException) e.getCause());
      else
        throw e;
    }
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
