package edu.stanford.nlp.sempre.thingtalk.seq2seq;

import java.io.*;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaDatabase;
import fig.basic.Option;
import fig.exec.Execution;

public class ExtractSeq2Seq implements Runnable {
  public static class Options {
    @Option
    public String languageTag = "en";

    @Option
    public List<String> types = new ArrayList<>();

    @Option
    public String output = null;
  }

  public static Options opts = new Options();

  private static final String QUERY_TMPL = "select id, type, utterance, target_json from example_utterances where language = ? and is_base = 0 and type in (%s)";

  private static String addTypesToQuery() {
    return String.format(QUERY_TMPL, opts.types.stream()
        .map((t) -> "'" + t + "'").collect(Collectors.joining(",")));
  }

  private ExtractSeq2Seq() {
  }

  @Override
  public void run() {
    Seq2SeqConverter converter = new Seq2SeqConverter(opts.languageTag);

    try (Writer writer = new BufferedWriter(new FileWriter(opts.output));
        Connection conn = ThingpediaDatabase.getSingleton().getConnection();
        PreparedStatement stmt = conn.prepareStatement(addTypesToQuery())) {
      stmt.setString(1, opts.languageTag);

      try (ResultSet set = stmt.executeQuery()) {
        while (set.next()) {
          int id = set.getInt(1);
          String type = set.getString(2);
          String utterance = set.getString(3);
          String targetJson = set.getString(4);
          Value targetValue = new StringValue(targetJson);

          Example ex = new Example.Builder()
              .setId(type + "_" + Integer.toString(id))
              .setUtterance(utterance)
              .setTargetValue(targetValue)
              .createExample();

          ex.preprocess();

          Seq2SeqConverter.writeSequences(converter.run(ex), writer);
        }
      }
    } catch (SQLException | IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new ExtractSeq2Seq(), Master.getOptionsParser());
  }
}
