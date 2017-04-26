package edu.stanford.nlp.sempre.thingtalk;

import java.sql.SQLException;

import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.Master;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import fig.basic.Option;
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

  private ThingpediaLexiconBuilder() {
  }

  @Override
  public void run() {
    CoreNLPAnalyzer.opts.annotators = Lists.newArrayList("ssplit", "pos", "lemma", "ner");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer(opts.languageTag);
    try {
      LexiconBuilder builder = new LexiconBuilder(analyzer, opts.languageTag, null);
      builder.build();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new ThingpediaLexiconBuilder(), Master.getOptionsParser());
  }
}
