package edu.stanford.nlp.sempre.thingtalk;

import java.sql.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.sql.DataSource;

import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.LanguageAnalyzer;
import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import fig.basic.Pair;

public class LexiconBuilder {
  public static class Function {
    final int schemaId;
    final String name;

    public Function(int schemaId, String name) {
      this.schemaId = schemaId;
      this.name = name;
    }

    @Override
    public String toString() {
      return schemaId + ":" + name;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((name == null) ? 0 : name.hashCode());
      result = prime * result + schemaId;
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
      Function other = (Function) obj;
      if (name == null) {
        if (other.name != null)
          return false;
      } else if (!name.equals(other.name))
        return false;
      if (schemaId != other.schemaId)
        return false;
      return true;
    }
  }

  private final String languageTag;
  private final boolean deletePrevious;
  private final String onlyKind;
  private final LanguageAnalyzer analyzer;

  private static final String ALL_QUERY = "select utterance,target_json from example_utterances where language = ? "
              + "and (type in ('thingpedia', 'online') or type like 'turking%') and not is_base lock in share mode";
  private static final String ONLY_KIND_QUERY = "select utterance,target_json from example_utterances where language = ? "
      + "and type = 'thingpedia' and schema_id = ? and not is_base lock in share mode";

  public LexiconBuilder(LanguageAnalyzer analyzer, String languageTag, String onlyKind) {
    this.analyzer = analyzer;
    this.languageTag = languageTag;
    this.onlyKind = onlyKind;
    this.deletePrevious = onlyKind == null;
  }

  public void build() throws SQLException {
    DataSource db = ThingpediaDatabase.getSingleton();

    try (Connection connection = db.getConnection()) {
      // start a transaction for the whole operation
      // this effectively locks the whole db mostly read only but that's
      // ok
      connection.setAutoCommit(false);

      // Cache all mappings from kinds to schema IDs
      Map<String, Integer> schemaIdMap = new HashMap<>();

      try (Statement s = connection.createStatement()) {
        try (ResultSet rs = s
            .executeQuery("select id,kind from device_schema where kind_type <> 'primary' lock in share mode")) {
          while (rs.next()) {
            schemaIdMap.put(rs.getString(2), rs.getInt(1));
          }
        }
      }

      // The lexicon itself
      Map<String, Set<Function>> newLexicon = new HashMap<>();
      Map<String, Map<Function, Double>> lexicon = new HashMap<>();
      // The priors on the functions
      Map<Function, Double> priors = new HashMap<>();

      Pattern namePattern = Pattern.compile("^tt:([^\\.]+)\\.(.+)$");

      int count = 0;
      try (PreparedStatement s = connection.prepareStatement(onlyKind == null ? ALL_QUERY : ONLY_KIND_QUERY)) {
        s.setString(1, languageTag);
        if (onlyKind != null)
          s.setInt(2, schemaIdMap.get(onlyKind));
        try (ResultSet rs = s.executeQuery()) {
          while (rs.next()) {
            ++count;
            if (count % 100 == 0)
              System.err.println("Example #" + count);
            String utterance = rs.getString(1);
            String targetJson = rs.getString(2);

            Map<String, Object> json = Json.readMapHard(targetJson);
            List<Map<String, Object>> invocations;

            if (json.containsKey("trigger"))
              invocations = Collections.singletonList((Map<String, Object>) json.get("trigger"));
            else if (json.containsKey("action"))
              invocations = Collections.singletonList((Map<String, Object>) json.get("action"));
            else if (json.containsKey("query"))
              invocations = Collections.singletonList((Map<String, Object>) json.get("query"));
            else {
              if (!json.containsKey("rule"))
                continue;
              invocations = new ArrayList<>();
              Map<String, Object> rule = (Map<String, Object>) json.get("rule");
              if (rule.containsKey("trigger"))
                invocations.add((Map<String, Object>) rule.get("trigger"));
              if (rule.containsKey("query"))
                invocations.add((Map<String, Object>) rule.get("query"));
              if (rule.containsKey("action"))
                invocations.add((Map<String, Object>) rule.get("action"));
            }

            double weight = 1. / invocations.size();

            LanguageInfo utteranceInfo = analyzer.analyze(utterance);

            for (Map<String, Object> inv : invocations) {
              Map<String, Object> name = (Map<String, Object>) inv.get("name");
              Matcher match = namePattern.matcher((String) name.get("id"));
              if (!match.matches())
                throw new RuntimeException("Channel name " + name.get("id") + " not in proper format");
              String kind = match.group(1);
              String channelName = match.group(2);

              if (!schemaIdMap.containsKey(kind)) {
                System.err.println("Invalid kind " + kind);
              } else {
                int schemaId = schemaIdMap.get(kind);
                Function fn = new Function(schemaId, channelName);

                priors.compute(fn, (existingFn, existingWeight) -> {
                  if (existingWeight == null)
                    return weight;
                  else
                    return existingWeight + weight;
                });

                for (int i = 0; i < utteranceInfo.numTokens(); i++) {
                  if (utteranceInfo.nerValues.get(i) != null)
                    continue;

                  if (!LanguageUtils.isContentWord(utteranceInfo.posTags.get(i)))
                    continue;
                  if (LexiconUtils.isIgnored(utteranceInfo.tokens.get(i)))
                    continue;
                  String token = LanguageUtils.stem(utteranceInfo.tokens.get(i));

                  lexicon.compute(token, (tkn, functions) -> {
                    if (functions == null)
                      functions = new HashMap<>();
                    functions.compute(fn, (existingFn, existingWeigth) -> {
                      if (existingWeigth == null)
                        return weight;
                      else
                        return existingWeigth + weight;
                    });
                    return functions;
                  });
                }
              }
            }
          }
        }

        for (Map.Entry<String, Map<Function, Double>> entry : lexicon.entrySet()) {
          String token = entry.getKey();
          Map<Function, Double> functions = entry.getValue();
          List<Pair<Function, Double>> functionList = new ArrayList<>();

          functions.forEach((fn, weight) -> {
            functionList.add(new Pair<>(fn, weight / priors.get(fn)));
          });
          functionList.sort((p1, p2) -> {
            return (int) Math.signum(p2.getSecond() - p1.getSecond());
          });

          for (int i = 0; i < Math.min(20, functionList.size()); i++) {
            Pair<Function, Double> p = functionList.get(i);
            newLexicon.computeIfAbsent(token, (key) -> new HashSet<>()).add(p.getFirst());
          }
        }

        if (deletePrevious) {
          try (Statement stmt = connection.createStatement()) {
            stmt.execute("delete from lexicon2");
          }
        }

        count = 0;
        try (PreparedStatement ps = connection.prepareStatement("insert ignore into lexicon2 values (?, ?, ?, ?)")) {
          count++;
          if (count % 10 == 0)
            System.err.println("Token #" + count + "/" + newLexicon.size());
          for (Map.Entry<String, Set<Function>> entry : newLexicon.entrySet()) {
            String token = entry.getKey();
            for (Function fn : entry.getValue()) {
              //System.out.println(token + "\t" + fn.schemaId + "\t" + fn.name);
              ps.setString(1, languageTag);
              ps.setString(2, token);
              ps.setInt(3, fn.schemaId);
              ps.setString(4, fn.name);
              ps.addBatch();
            }
          }
          ps.executeBatch();
        }
        connection.commit();
      }
    }
  }
}
