package edu.stanford.nlp.sempre.overnight;

import java.io.FileWriter;
import java.io.IOException;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;

public class Tokenizer {
  private final String languageTag;

  public Tokenizer(String languageTag) {
    this.languageTag = languageTag;
  }

  public void tokenize(String exampleFile, String utteranceFile, String originalFile) {
    CoreNLPAnalyzer.opts.annotators = Lists.newArrayList("ssplit");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer(languageTag);

    try (FileWriter utteranceWriter = new FileWriter(utteranceFile);
        FileWriter originalWriter = new FileWriter(originalFile)) {
      for (String line : IOUtils.readLines(exampleFile)) {
        String[] phrases = line.split("\t");
        if (phrases.length < 2) {
          System.err.println("Unexpected badly formatted line: " + line);
          continue;
        }
        String utterance = phrases[0];
        String original = phrases[1];

        LanguageInfo utteranceInfo = analyzer.analyze(utterance);

        utteranceWriter.append(Joiner.on(' ').join(utteranceInfo.tokens) + "\n");
        originalWriter.append(original);
        originalWriter.append("\n");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  //args[0] - example file with utterance and original
  //args[1] - utterance output file
  //args[2] - canonical (original) output file
  //args[3] - language tag
  public static void main(String[] args) {
    Tokenizer tokenizer = new Tokenizer(args[3]);
    tokenizer.tokenize(args[0], args[1], args[2]);
  }
}
