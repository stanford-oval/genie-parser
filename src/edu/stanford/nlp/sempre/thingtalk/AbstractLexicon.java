package edu.stanford.nlp.sempre.thingtalk;

import java.util.Collection;
import java.util.Iterator;

import edu.stanford.nlp.sempre.Formula;
import edu.stanford.nlp.sempre.GenericObjectCache;
import fig.basic.LogInfo;
import fig.basic.Option;

public abstract class AbstractLexicon {
  public static class Options {
    @Option
    public int verbose = 0;
  }

  public static Options opts = new Options();

  public static class Entry {
    public final String nerTag;
    public final Formula formula;
    public final String rawPhrase;

    public Entry(String nerTag, Formula formula, String rawPhrase) {
      this.nerTag = nerTag;
      this.formula = formula;
      this.rawPhrase = rawPhrase;
    }
  }

  private final GenericObjectCache<String, Collection<Entry>> cache = new GenericObjectCache<>(256);

  protected abstract Collection<Entry> doLookup(String rawPhrase);

  public Iterator<Entry> lookup(String rawPhrase) {
    if (opts.verbose >= 2)
      LogInfo.logs("AbstractLexicon.lookup %s", rawPhrase);
    Collection<Entry> fromCache = cache.hit(rawPhrase);
    if (fromCache != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("AbstractLexicon.cacheHit");
      return fromCache.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("AbstractLexicon.cacheMiss");

    fromCache = doLookup(rawPhrase);
    // cache location lookups forever
    // if memory pressure occcurs, the cache will automatically
    // downsize itself
    cache.store(rawPhrase, fromCache, Long.MAX_VALUE);
    return fromCache.iterator();
  }
}

