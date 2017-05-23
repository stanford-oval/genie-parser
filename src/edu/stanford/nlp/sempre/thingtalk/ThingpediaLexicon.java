package edu.stanford.nlp.sempre.thingtalk;

import java.sql.*;
import java.util.*;

import javax.sql.DataSource;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import fig.basic.LogInfo;
import fig.basic.Option;

public class ThingpediaLexicon {
  public static final long CACHE_AGE = 1000 * 3600 * 1; // cache lexicon lookups for 1 hour

  public static class Options {
    @Option
    public int verbose = 0;
  }

  public static Options opts = new Options();

  public enum Mode {
    TRIGGER, ACTION, QUERY;
  };

  public static abstract class Entry {
    public abstract Formula toFormula();

    public abstract String getRawPhrase();

    public void addFeatures(FeatureVector vec) {
    }

    public boolean applyFilter(Object filter) {
      return true;
    }

    @Override
    public String toString() {
      return "[ " + getRawPhrase() + " => " + toFormula() + " ]";
    }
  }

  private static class ParamEntry extends Entry {
    private final String rawPhrase;
    private final String argname;
    private final String type;

    public ParamEntry(String rawPhrase, String argname, String type) {
      this.rawPhrase = rawPhrase;
      this.argname = argname;
      this.type = type;
    }

    @Override
    public String getRawPhrase() {
      return rawPhrase;
    }

    @Override
    public boolean applyFilter(Object filter) {
      return type.equals(filter);
    }

    @Override
    public Formula toFormula() {
      return new ValueFormula<>(
          new ParamNameValue(argname, type));
    }
  }

  private static class AppEntry extends Entry {
    private final String rawPhrase;
    private final long userId;
    private final String appId;

    public AppEntry(String rawPhrase, long userId, String appId) {
      this.rawPhrase = rawPhrase;
      this.userId = userId;
      this.appId = appId;
    }

    @Override
    public String getRawPhrase() {
      return rawPhrase;
    }

    @Override
    public Formula toFormula() {
      return new ValueFormula<>(new NameValue("tt:app." + userId + "." + appId));
    }
  }

  private static class ChannelEntry extends Entry {
    private final String rawPhrase;
    private final String kind;
    private final String name;
    private final List<String> argnames;
    private final List<String> argcanonicals;
    private final List<String> argtypes;
    private final String search;

    public ChannelEntry(String rawPhrase, String kind, String name, String argnames, String argcanonicals,
        String argtypes, String search)
        throws JsonProcessingException {
      this.rawPhrase = rawPhrase;
      this.kind = kind;
      this.name = name;
      this.search = search;

      TypeReference<List<String>> typeRef = new TypeReference<List<String>>() {
      };
      this.argnames = Json.readValueHard(argnames, typeRef);
      this.argcanonicals = Json.readValueHard(argcanonicals, typeRef);
      this.argtypes = Json.readValueHard(argtypes, typeRef);
    }

    @Override
    public String getRawPhrase() {
      return rawPhrase;
    }

    @Override
    public Formula toFormula() {
      return new ValueFormula<>(new ChannelNameValue(kind, name, argnames, argcanonicals, argtypes));
    }

    @Override
    public void addFeatures(FeatureVector vec) {
      if (ThingTalkFeatureComputer.opts.featureDomains.contains("thingtalk_lexicon"))
        vec.add("thingtalk_lexicon", search + "---" + kind);
    }
  }

  private static class KindEntry extends Entry {
    private final String kindCanonical;
    private final String kind;

    public KindEntry(String kindCanonical, String kind) {
      this.kindCanonical = kindCanonical;
      this.kind = kind;
    }

    @Override
    public String getRawPhrase() {
      return kindCanonical;
    }

    @Override
    public Formula toFormula() {
      return new ValueFormula<>(new NameValue("tt:device." + kind));
    }
  }

  private static class LexiconKey {
    private final Mode mode;
    private final String query;

    public LexiconKey(Mode mode, String query) {
      this.mode = mode;
      this.query = query;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((mode == null) ? 0 : mode.hashCode());
      result = prime * result + ((query == null) ? 0 : query.hashCode());
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
      LexiconKey other = (LexiconKey) obj;
      if (mode != other.mode)
        return false;
      if (query == null) {
        if (other.query != null)
          return false;
      } else if (!query.equals(other.query))
        return false;
      return true;
    }
  }

  private static final Map<String, ThingpediaLexicon> instances = new HashMap<>();
  private final GenericObjectCache<LexiconKey, List<Entry>> cache = new GenericObjectCache<>(1024);
  private final DataSource dataSource;
  private final String languageTag;

  private ThingpediaLexicon(String languageTag) {
    dataSource = ThingpediaDatabase.getSingleton();
    this.languageTag = languageTag;
  }

  public synchronized static ThingpediaLexicon getForLanguage(String languageTag) {
    ThingpediaLexicon instance = instances.get(languageTag);
    if (instance == null) {
      instance = new ThingpediaLexicon(languageTag);
      instances.put(languageTag, instance);
    }
    return instance;
  }

  public synchronized static void clearAllCaches() {
    for (ThingpediaLexicon lexicon : instances.values()) {
      lexicon.clear();
    }
  }

  public void clear() {
    cache.clear();
  }

  public Iterator<Entry> lookupChannel(String phrase, Mode channel_type) throws SQLException {
    if (opts.verbose >= 2)
      LogInfo.logs("ThingpediaLexicon.lookupChannel(%s) %s", channel_type, phrase);

    String token = LexiconUtils.preprocessRawPhrase(phrase);
    if (token == null)
      return Collections.emptyIterator();

    List<Entry> entries = cache.hit(new LexiconKey(channel_type, phrase));
    if (entries != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("ThingpediaLexicon cacheHit");
      return entries.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("ThingpediaLexicon cacheMiss");

    String query = "select dscc.canonical,ds.kind,dsc.name,dsc.argnames,dscc.argcanonicals,dsc.types from "
        + " device_schema_channels dsc, device_schema ds, device_schema_channel_canonicals dscc, lexicon2 lex "
        + " where dsc.schema_id = ds.id and dsc.version = ds.developer_version and dscc.schema_id = dsc.schema_id "
        + " and dscc.version = dsc.version and dscc.name = dsc.name and dscc.language = ? and channel_type = ? and "
        + " lex.schema_id = ds.id and ds.kind_type <> 'primary' and lex.channel_name = dsc.name and lex.language = ? "
        + " and lex.token = ? limit "
        + (3 * Parser.opts.beamSize);

    long now = System.currentTimeMillis();

    String search, key;
    try (Connection con = dataSource.getConnection(); PreparedStatement stmt = con.prepareStatement(query)) {
      stmt.setString(1, languageTag);
      String channelType = channel_type.toString().toLowerCase();
      stmt.setString(2, channelType);
      stmt.setString(3, languageTag);
      String stemmed = LanguageUtils.stem(token);
      stmt.setString(4, stemmed);
      key = stemmed;

      entries = new LinkedList<>();
      try (ResultSet rs = stmt.executeQuery()) {
        while (rs.next())
          entries.add(new ChannelEntry(rs.getString(1), rs.getString(2), rs.getString(3),
              rs.getString(4), rs.getString(5), rs.getString(6), key));
      } catch (SQLException | JsonProcessingException e) {
        if (opts.verbose > 0)
          LogInfo.logs("Exception during lexicon lookup: %s", e.getMessage());
      }
      cache.store(new LexiconKey(channel_type, phrase), Collections.unmodifiableList(entries),
          now + CACHE_AGE);
      return entries.iterator();
    }
  }
}
