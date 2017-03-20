package edu.stanford.nlp.sempre.thingtalk.seq2seq;

import java.io.*;
import java.util.List;

import edu.stanford.nlp.sempre.AbstractDataset;
import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.Master;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaDataset;
import fig.basic.Option;
import fig.exec.Execution;

public class PrepareSeq2Seq implements Runnable {
  public static class Options {
    @Option
    public String languageTag = "en";

    @Option
    public String trainFile = "deep/train.tsv";

    @Option
    public String testFile = "deep/test.tsv";

    @Option
    public String devFile = "deep/dev.tsv";
  }

  public static final Options opts = new Options();

  private PrepareSeq2Seq() {
  }

  private static void processGroup(AbstractDataset dataset, String groupName, String fileName) {
    List<Example> group = dataset.examples(groupName);
    if (group == null)
      return;

    Seq2SeqConverter converter = new Seq2SeqConverter(opts.languageTag);

    try (Writer writer = new BufferedWriter(new FileWriter(fileName))) {
      for (Example ex : group) {
        Seq2SeqConverter.writeSequences(converter.run(ex), writer);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void run() {
    AbstractDataset dataset = new ThingpediaDataset();
    try {
      dataset.read();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    processGroup(dataset, "train", opts.trainFile);
    processGroup(dataset, "test", opts.testFile);
    processGroup(dataset, "dev", opts.devFile);
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new PrepareSeq2Seq(), Master.getOptionsParser());
  }
}
