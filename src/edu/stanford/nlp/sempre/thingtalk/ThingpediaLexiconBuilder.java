package edu.stanford.nlp.sempre.thingtalk;

import java.sql.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.sql.DataSource;

import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import edu.stanford.nlp.sempre.Master;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import fig.basic.Option;
import fig.basic.Pair;
import fig.exec.Execution;

/**
 * Analyzes all the examples and builds a lexicon mapping
 * tokens to functions
 * 
 * @author gcampagn
 *
 */
public class ThingpediaLexiconBuilder implements Runnable {
  public static class Options {
    @Option
    public String languageTag = "en";
  }

  public static final Options opts = new Options();

  private static class Function {
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

  private ThingpediaLexiconBuilder() {
  }

  @Override
  public void run() {
    CoreNLPAnalyzer.opts.annotators = Lists.newArrayList("ssplit", "pos", "lemma", "ner");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer(opts.languageTag);
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

      // First import the old lexicon
      try (Statement s = connection.createStatement()) {
        try (ResultSet rs = s
            .executeQuery("select token,schema_id,channel_name from lexicon where language = 'en'")) {
          while (rs.next()) {
            String token = LanguageUtils.stem(rs.getString(1));
            int schemaId = rs.getInt(2);
            String channelName = rs.getString(3);
            newLexicon.computeIfAbsent(token, (key) -> new HashSet<>()).add(new Function(schemaId, channelName));
          }
        }
      }

      // The lexicon itself
      Map<String, Map<Function, Double>> lexicon = new HashMap<>();
      // The priors on the functions
      Map<Function, Double> priors = new HashMap<>();

      Pattern namePattern = Pattern.compile("^tt:([^\\.]+)\\.(.+)$");

      int count = 0;
      try (PreparedStatement s = connection.prepareStatement(
          "select utterance,target_json from example_utterances where language = ? "
              + "and (type = 'thingpedia') and not is_base "
              + "lock in share mode")) {
        s.setString(1, opts.languageTag);
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

        count = 0;
        try (PreparedStatement ps = connection.prepareStatement("insert into lexicon2 values (?, ?, ?, ?)")) {
          count++;
          if (count % 10 == 0)
            System.err.println("Token #" + count + "/" + newLexicon.size());
          for (Map.Entry<String, Set<Function>> entry : newLexicon.entrySet()) {
            String token = entry.getKey();
            for (Function fn : entry.getValue()) {
              //System.out.println(token + "\t" + fn.schemaId + "\t" + fn.name);
              ps.setString(1, opts.languageTag);
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
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new ThingpediaLexiconBuilder(), Master.getOptionsParser());
  }
}
