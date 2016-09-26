package edu.stanford.nlp.sempre.ifttt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import edu.stanford.nlp.sempre.*;
import fig.basic.LogInfo;
import fig.basic.Option;

public class IftttDataset extends AbstractDataset {
  public static class Options {
    @Option
    public String trainFile = null;
    @Option
    public String testFile = null;
  }

  public static Options opts = new Options();

  public IftttDataset() {
  }

  private void readFromFile(int maxExamples, List<Example> examples, String prefix, String filename) throws IOException {
    int count = 0;
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
      String line;
      while (examples.size() < maxExamples && (line = reader.readLine()) != null) {
        String[] parts = line.split("\t");

        Example ex = new Example.Builder()
            .setId(prefix + "_" + Integer.toString(count++))
            .setUtterance(parts[0])
            .setTargetValue(new StringValue(parts[2]))
            .createExample();

        addOneExample(ex, maxExamples, examples);
      }
    }
  }

  private void readTrain() throws IOException {
    int maxExamples = getMaxExamplesForGroup("train");
    List<Example> examples = getOrCreateGroup("train");

    if (opts.trainFile == null)
      return;

    readFromFile(maxExamples, examples, "train", opts.trainFile);
  }

  private void readTest() throws IOException {
    int maxExamples = getMaxExamplesForGroup("test");
    List<Example> examples = getOrCreateGroup("test");

    if (opts.testFile == null)
      return;

    readFromFile(maxExamples, examples, "test", opts.testFile);
  }

  @Override
  public void read() throws IOException {
    LogInfo.begin_track_printAll("IftttDataset.read");

    readTrain();

    if (Dataset.opts.splitDevFromTrain)
      splitDevFromTrain();

    readTest();
    collectStats();

    LogInfo.end_track();
  }
}
