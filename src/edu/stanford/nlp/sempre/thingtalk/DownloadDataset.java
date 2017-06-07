package edu.stanford.nlp.sempre.thingtalk;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import com.google.common.base.Joiner;

import edu.stanford.nlp.sempre.*;
import fig.basic.Option;
import fig.exec.Execution;

public class DownloadDataset implements Runnable {
  public static class Options {
    @Option
    public String outputFile;
  }

  public static final Options opts = new Options();

  private Writer writer;
  private CanonicalReconstructor reconstructor;

  private DownloadDataset() {
  }

  private void writeUtterance(Example ex) throws IOException {
    LanguageInfo utteranceInfo = ex.languageInfo;

    List<String> inputTokens = new ArrayList<>();
    for (int i = 0; i < utteranceInfo.tokens.size(); i++) {
      String token, tag, current;

      tag = utteranceInfo.nerTags.get(i);
      token = utteranceInfo.tokens.get(i);

      if (!"O".equals(tag)) {
        if (i < utteranceInfo.tokens.size() - 1 &&
            utteranceInfo.nerTags.get(i + 1).equals(tag) &&
            Objects.equals(utteranceInfo.nerValues.get(i), utteranceInfo.nerValues.get(i + 1)))
          continue;

        current = tag;
      } else {
        current = token;
      }
      inputTokens.add(current);
    }

    writer.write(Joiner.on(' ').join(inputTokens));
  }

  private void writeCanonical(Example ex) throws IOException {
    writer.write(reconstructor.reconstruct(ex));
  }

  @Override
  public void run() {
    Builder builder = new Builder();
    builder.build();
    reconstructor = new CanonicalReconstructor(builder.grammar);
    try (Writer writer = new FileWriter(opts.outputFile)) {
      this.writer = writer;

      for (Example ex : builder.dataset.examples("train")) {
        writeUtterance(ex);
        writer.write('\t');
        writeCanonical(ex);
        writer.write('\t');
        writer.write(((StringValue) ex.targetValue).value);
        writer.write('\n');
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new DownloadDataset(), Master.getOptionsParser());
  }

}
