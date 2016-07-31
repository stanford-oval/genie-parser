package edu.stanford.nlp.sempre.thingtalk;

import java.sql.*;
import java.util.*;

import javax.sql.DataSource;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;

import edu.stanford.nlp.sempre.*;
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
    APP, KIND, PARAM, TRIGGER, ACTION, QUERY;
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
    private final List<String> argtypes;

    public ChannelEntry(String rawPhrase, String kind, String name, String argnames, String argtypes)
        throws JsonProcessingException {
      this.rawPhrase = rawPhrase;
      this.kind = kind;
      this.name = name;

      TypeReference<List<String>> typeRef = new TypeReference<List<String>>() {
      };
      this.argnames = Json.readValueHard(argnames, typeRef);
      this.argtypes = Json.readValueHard(argtypes, typeRef);
    }

    @Override
    public String getRawPhrase() {
      return rawPhrase;
    }

    @Override
    public Formula toFormula() {
      return new ValueFormula<>(new ChannelNameValue(kind, name, argnames, argtypes));
    }

    @Override
    public void addFeatures(FeatureVector vec) {
      vec.add("kinds", kind);
    }
  }

  private static class KindEntry extends Entry {
    private final String kind;

    public KindEntry(String kind) {
      this.kind = kind;
    }

    @Override
    public String getRawPhrase() {
      return kind;
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

  private final DataSource dataSource;
  private final String languageTag;
  private final GenericObjectCache<LexiconKey, List<Entry>> cache = new GenericObjectCache<>(
      1024);

  public ThingpediaLexicon(String languageTag) {
    dataSource = ThingpediaDatabase.getSingleton();
    this.languageTag = languageTag;
  }

  public Iterator<Entry> lookupApp(String phrase) throws SQLException {
    if (opts.verbose >= 2)
      LogInfo.logs("ThingpediaLexicon.lookupApp %s", phrase);

    List<Entry> entries = cache.hit(new LexiconKey(Mode.APP, phrase));
    if (entries != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("ThingpediaLexicon cacheHit");
      return entries.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("ThingpediaLexicon cacheMiss");

    String query;
    if (Builder.opts.parser.equals("BeamParser")) {
      query = "select canonical,owner,appId from app where canonical = ? limit " + Parser.opts.beamSize;
    } else {
      query = "select canonical,owner,appId from app where match canonical against (? in natural language mode) limit "
          + Parser.opts.beamSize;
    }

    long now = System.currentTimeMillis();

    try (Connection con = dataSource.getConnection()) {
      try (PreparedStatement stmt = con.prepareStatement(query)) {
        stmt.setString(1, phrase);

        entries = new LinkedList<>();
        try (ResultSet rs = stmt.executeQuery()) {
          while (rs.next())
            entries.add(new AppEntry(rs.getString(1), rs.getLong(2), rs.getString(3)));
        } catch (SQLException e) {
          if (opts.verbose > 0)
            LogInfo.logs("SQL exception during lexicon lookup: %s", e.getMessage());
        }
        cache.store(new LexiconKey(Mode.APP, phrase), Collections.unmodifiableList(entries), now + CACHE_AGE);
        return entries.iterator();
      }
    }
  }

  public Iterator<Entry> lookupKind(String phrase) throws SQLException {
    if (opts.verbose >= 2)
      LogInfo.logs("ThingpediaLexicon.lookupKind %s", phrase);

    List<Entry> entries = cache.hit(new LexiconKey(Mode.KIND, phrase));
    if (entries != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("ThingpediaLexicon cacheHit");
      return entries.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("ThingpediaLexicon cacheMiss");

    String query = "select kind from device_schema where kind = ? limit " + Parser.opts.beamSize;

    long now = System.currentTimeMillis();

    try (Connection con = dataSource.getConnection()) {
      try (PreparedStatement stmt = con.prepareStatement(query)) {
        stmt.setString(1, phrase);

        entries = new LinkedList<>();
        try (ResultSet rs = stmt.executeQuery()) {
          while (rs.next())
            entries.add(new KindEntry(rs.getString(1)));
        } catch (SQLException e) {
          if (opts.verbose > 0)
            LogInfo.logs("SQL exception during lexicon lookup: %s", e.getMessage());
        }
        cache.store(new LexiconKey(Mode.KIND, phrase), Collections.unmodifiableList(entries), now + CACHE_AGE);
        return entries.iterator();
      }
    }
  }

  public Iterator<Entry> lookupParam(String phrase) throws SQLException {
    if (opts.verbose >= 2)
      LogInfo.logs("ThingpediaLexicon.lookupParam %s", phrase);

    String query;
    if (phrase == null) {
      query = "select distinct canonical, argname, argtype from device_schema_arguments join device_schema "
          + "on schema_id = id and version = approved_version and kind_type <> 'primary'";
    } else {
      String[] tokens = phrase.split(" ");
      if (Builder.opts.parser.equals("BeamParser")) {
        if (tokens.length > 4)
          return Collections.emptyIterator();

        query = "select distinct canonical, argname, argtype from device_schema_arguments join device_schema "
            + "on schema_id = id and version = approved_version and canonical = ? and kind_type <> 'primary' "
            + "limit " + Parser.opts.beamSize;
      } else {
        if (tokens.length > 1)
          return Collections.emptyIterator();

        query = "select distinct canonical, argname, argtype from device_schema_arguments join device_schema "
            + "on schema_id = id and version = approved_version and match canonical against (? in natural language "
            + "mode) and kind_type <> 'primary' limit " + Parser.opts.beamSize;
      }
    }

    List<Entry> entries = cache.hit(new LexiconKey(Mode.PARAM, phrase));
    if (entries != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("ThingpediaLexicon cacheHit");
      return entries.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("ThingpediaLexicon cacheMiss");

    long now = System.currentTimeMillis();

    try (Connection con = dataSource.getConnection()) {
      try (PreparedStatement stmt = con.prepareStatement(query)) {
        if (phrase != null)
          stmt.setString(1, phrase);

        entries = new LinkedList<>();
        try (ResultSet rs = stmt.executeQuery()) {
          while (rs.next())
            entries.add(new ParamEntry(rs.getString(1), rs.getString(2), rs.getString(3)));
        } catch (SQLException e) {
          if (opts.verbose > 0)
            LogInfo.logs("SQL exception during lexicon lookup: %s", e.getMessage());
        }
        cache.store(new LexiconKey(Mode.PARAM, phrase), Collections.unmodifiableList(entries), now + CACHE_AGE);
        return entries.iterator();
      }
    }
  }

  // a list of words that appear often in our examples (and thus are frequent queries to
  // the lexicon), but are not useful to lookup canonical forms
  // with FloatingParser, if the lookup word is in this array, we just return no
  // derivations
  private static final String[] IGNORED_WORDS = { "in", "on", "a", "to", "with", "and",
      "me", "if", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwz" };
  static {
    Arrays.sort(IGNORED_WORDS);
  }

  public Iterator<Entry> lookupChannel(String phrase, Mode channel_type) throws SQLException {
    if (opts.verbose >= 2)
      LogInfo.logs("ThingpediaLexicon.lookupChannel(%s) %s", channel_type, phrase);

    String[] tokens = phrase.split(" ");
    if (Builder.opts.parser.equals("BeamParser")) {
      if (tokens.length < 3 || tokens.length > 7)
        return Collections.emptyIterator();
      if (!"on".equals(tokens[tokens.length - 2]))
        return Collections.emptyIterator();
    } else {
      if (tokens.length > 1)
        return Collections.emptyIterator();
      if (Arrays.binarySearch(IGNORED_WORDS, tokens[0]) >= 0)
        return Collections.emptyIterator();
    }

    List<Entry> entries = cache.hit(new LexiconKey(channel_type, phrase));
    if (entries != null) {
      if (opts.verbose >= 3)
        LogInfo.logs("ThingpediaLexicon cacheHit");
      return entries.iterator();
    }
    if (opts.verbose >= 3)
      LogInfo.logs("ThingpediaLexicon cacheMiss");

    String query;
    if (Builder.opts.parser.equals("BeamParser")) {
      query = "select dsc.canonical,ds.kind,dsc.name,dsc.argnames,dsc.types from device_schema_channels dsc, device_schema ds "
          + " where dsc.schema_id = ds.id and dsc.version = ds.approved_version and channel_type = ? "
          + " and canonical = ? and ds.kind_type <> 'primary' limit " + Parser.opts.beamSize;
    } else {
      query = "select dsc.canonical,ds.kind,dsc.name,dsc.argnames,dsc.types from device_schema_channels dsc, device_schema ds "
          + " where dsc.schema_id = ds.id and dsc.version = ds.approved_version and channel_type = ? and "
          + "match canonical against (? in natural language mode) and ds.kind_type <> 'primary' limit "
          + Parser.opts.beamSize;
    }

    long now = System.currentTimeMillis();

    try (Connection con = dataSource.getConnection()) {
      try (PreparedStatement stmt = con.prepareStatement(query)) {
        stmt.setString(1, channel_type.toString().toLowerCase());
        stmt.setString(2, phrase);

        entries = new LinkedList<>();
        try (ResultSet rs = stmt.executeQuery()) {
          while (rs.next())
            entries.add(new ChannelEntry(rs.getString(1), rs.getString(2), rs.getString(3),
                rs.getString(4), rs.getString(5)));
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
}
