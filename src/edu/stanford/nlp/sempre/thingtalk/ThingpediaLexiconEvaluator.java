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
import fig.exec.Execution;

/**
 * Analyzes all the examples and computes the recall of the lexicon
 * 
 * @author gcampagn
 *
 */
public class ThingpediaLexiconEvaluator implements Runnable {
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

  private ThingpediaLexiconEvaluator() {
  }

  @Override
  public void run() {
    CoreNLPAnalyzer.opts.annotators = Lists.newArrayList("ssplit", "pos", "lemma", "ner");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer(opts.languageTag);
    DataSource db = ThingpediaDatabase.getSingleton();

    try (Connection connection = db.getConnection()) {
      // Cache all mappings from kinds to schema IDs
      Map<String, Integer> schemaIdMap = new HashMap<>();
      try (Statement s = connection.createStatement()) {
        try (ResultSet rs = s
            .executeQuery("select id,kind from device_schema where kind_type <> 'global'")) {
          while (rs.next()) {
            schemaIdMap.put(rs.getString(2), rs.getInt(1));
          }
        }
      }

      // Cache the whole lexicon
      Map<String, Set<Function>> lexicon = new HashMap<>();
      try (PreparedStatement s = connection.prepareStatement(
          "select token,schema_id,channel_name from lexicon2 where language = ?")) {
        s.setString(1, opts.languageTag);
        try (ResultSet rs = s.executeQuery()) {
          while (rs.next()) {
            String token = LanguageUtils.stem(rs.getString(1));
            int schemaId = rs.getInt(2);
            String channelName = rs.getString(3);
            lexicon.computeIfAbsent(token, (key) -> new HashSet<>()).add(new Function(schemaId, channelName));
          }
        }
      }

      Pattern namePattern = Pattern.compile("^tt:([^\\.]+)\\.(.+)$");

      int success = 0;
      int count = 0;
      try (PreparedStatement s = connection.prepareStatement(
          "select utterance,target_json from example_utterances where language = ? "
              + "and type like 'test%' and not is_base")) {
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

            boolean all = true;
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
                Function target = new Function(schemaId, channelName);

                boolean found = false;
                for (int i = 0; i < utteranceInfo.numTokens(); i++) {
                  if (utteranceInfo.nerValues.get(i) != null)
                    continue;

                  if (LexiconUtils.isIgnored(utteranceInfo.tokens.get(i)))
                    continue;
                  String token = LanguageUtils.stem(utteranceInfo.tokens.get(i));

                  for (Function fn : lexicon.getOrDefault(token, Collections.emptySet())) {
                    if (fn.equals(target)) {
                      found = true;
                      break;
                    }
                  }
                  if (found)
                    break;
                }
                if (!found) {
                  all = false;
                  break;
                }
              }
            }

            if (all)
              success++;
          }

          System.err.println("Recall: " + (double) success / count);
        }
      }
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new ThingpediaLexiconEvaluator(), Master.getOptionsParser());
  }
}
