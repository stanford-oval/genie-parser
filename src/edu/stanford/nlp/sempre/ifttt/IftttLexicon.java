package edu.stanford.nlp.sempre.ifttt;

import java.sql.*;
import java.util.*;
import java.util.stream.Collectors;

import javax.sql.DataSource;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import edu.stanford.nlp.sempre.thingtalk.ChannelNameValue;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaDatabase;
import edu.stanford.nlp.sempre.thingtalk.Type;
import fig.basic.LogInfo;
import fig.basic.Option;

public class IftttLexicon {
  public static final long CACHE_AGE = 1000 * 3600 * 1; // cache lexicon lookups for 1 hour

  public static class Options {
    @Option
    public int verbose = 0;
  }

  public static Options opts = new Options();

  public enum Mode {
    TRIGGER, ACTION;
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

  private static class ChannelEntry extends Entry {
    private final String rawPhrase;
    private final String kind;
    private final String name;
    private final List<String> argnames;
    private final List<String> argcanonicals;
    private final List<Type> argtypes;
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
      this.argtypes = Json.readValueHard(argtypes, typeRef).stream().map((s) -> Type.fromString(s))
          .collect(Collectors.toList());
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
      vec.add("kinds", kind);

      // this overfits wildly, but makes sure that certain words like xkcd or twitter
      // (when they appear) are immediately recognized as the right kind so that we don't
      // go in the woods because of a "get" that's too generic
      vec.add("thingtalk_lexicon", search + "---" + this.kind);
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

  private static final Map<String, IftttLexicon> instances = new HashMap<>();
  private final GenericObjectCache<LexiconKey, List<Entry>> cache = new GenericObjectCache<>(1024);
  private final DataSource dataSource;
  private final String languageTag;

  private IftttLexicon(String languageTag) {
    dataSource = ThingpediaDatabase.getSingleton();
    this.languageTag = languageTag;
  }

  public synchronized static IftttLexicon getForLanguage(String languageTag) {
    IftttLexicon instance = instances.get(languageTag);
    if (instance == null) {
      instance = new IftttLexicon(languageTag);
      instances.put(languageTag, instance);
    }
    return instance;
  }

  // a list of words that appear often in our examples (and thus are frequent queries to
  // the lexicon), but are not useful to lookup canonical forms
  // with FloatingParser, if the lookup word is in this array, we just return no
  // derivations
  private static final String[] IGNORED_WORDS = { "in", "on", "a", "to", "with", "and",
      "me", "the", "if", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwz" };
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
    boolean isBeam;
    if (Builder.opts.parser.equals("BeamParser")) {
      isBeam = true;
      query = "select canonical,channel,name from interfaces where type = ? "
          + " and canonical = ?";
    } else {
      isBeam = false;
      query = "select canonical,channel,name from interfaces where type = ?  "
          + " and match (canonical) against (? in natural language mode)";
    }

    long now = System.currentTimeMillis();

    String search, key;
    try (Connection con = dataSource.getConnection()) {
      try (PreparedStatement stmt = con.prepareStatement(query)) {
        stmt.setString(1, channel_type.toString().toLowerCase());
        if (isBeam) {
          search = phrase;
          key = phrase;
          stmt.setString(2, phrase);
        } else {
          search = "";
          key = "";
          for (int i = 0; i < tokens.length; i++) {
            search += (i > 0 ? " " : "") + tokens[i];
            search += " " + LanguageUtils.stem(tokens[i]);
            key += (i > 0 ? " " : "") + tokens[i];
          }
          stmt.setString(2, search);
        }

        entries = new LinkedList<>();
        try (ResultSet rs = stmt.executeQuery()) {
          while (rs.next())
            entries.add(new ChannelEntry(rs.getString(1), rs.getString(2), rs.getString(3),
                "[]", "[]", "[]", key));
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
