package edu.stanford.nlp.sempre.thingtalk;

import java.sql.*;
import java.util.*;

import javax.sql.DataSource;

import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import edu.stanford.nlp.sempre.ValueFormula;
import fig.basic.LogInfo;

public class EntityLexicon extends AbstractLexicon {
  private static final Map<String, EntityLexicon> instances = new HashMap<>();

  private final String languageTag;
  private final DataSource dataSource;

  private EntityLexicon(String languageTag) {
    dataSource = ThingpediaDatabase.getSingleton();
    this.languageTag = languageTag;
  }

  public synchronized static EntityLexicon getForLanguage(String languageTag) {
    EntityLexicon instance = instances.get(languageTag);
    if (instance == null) {
      instance = new EntityLexicon(languageTag);
      instances.put(languageTag, instance);
    }
    return instance;
  }

  private static final String QUERY = "select entity_id,entity_value,entity_canonical from entity_lexicon where language = ? and token in (?, ?)";

  @Override
  protected Collection<Entry> doLookup(String rawPhrase) {
    String token = LexiconUtils.preprocessRawPhrase(rawPhrase);
    if (token == null)
      return Collections.emptySet();
  
    Collection<Entry> entries = new LinkedList<>();

    try (Connection con = dataSource.getConnection(); PreparedStatement stmt = con.prepareStatement(QUERY)) {
      stmt.setString(1, languageTag);
      stmt.setString(2, token);
      stmt.setString(3, LanguageUtils.stem(token));

      try (ResultSet rs = stmt.executeQuery()) {
        while (rs.next()) {
          String id = rs.getString(1);
          String entityValue = rs.getString(2);
          String entityCanonical = rs.getString(3);

          String type = "Entity(" + id + ")";
          entries.add(new Entry("GENERIC_ENTITY_" + id,
              new ValueFormula<>(new TypedStringValue(type, entityValue)),
              entityCanonical));
        }
      }
    } catch (SQLException e) {
      if (opts.verbose > 0)
        LogInfo.logs("Exception during lexicon lookup: %s", e.getMessage());
    }
    return entries;
  }
}
