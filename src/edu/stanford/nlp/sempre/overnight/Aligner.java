package edu.stanford.nlp.sempre.overnight;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.google.common.base.Joiner;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import fig.basic.LispTree;
import fig.basic.MapUtils;

/**
 * Word-aligns original utterances with their paraphrases
 * Created by joberant on 2/20/15.
 */
public class Aligner {

  private final Map<String, Counter<String>> model = new HashMap<>();

  public double getCondProb(String target, String source) {
    if (model.containsKey(source)) {
      if (model.get(source).containsKey(target))
        return model.get(source).getCount(target);
    }
    return 0d;
  }

  public int size() { return model.size(); }

  //takes an example file and creates a model
  public void heuristicAlign(String exampleFile, int threshold) {

    model.clear();
    Iterator<LispTree> iter = LispTree.proto.parseFromFile(exampleFile);

    while (iter.hasNext()) {
      LispTree tree = iter.next();
      LispTree utteranceTree = tree.child(1);
      LispTree originalTree = tree.child(2);
      String utterance = preprocessUtterance(utteranceTree.child(1).value);
      String original = preprocessUtterance(originalTree.child(1).value);
      String[] utteranceTokens = utterance.split("\\s+");
      String[] originalTokens = original.split("\\s+");

      align(utteranceTokens, originalTokens);
    }
    normalize(threshold);
  }

  public void heuristicTsvAlign(String exampleFile, int threshold) {
    model.clear();
    for (String line : IOUtils.readLines(exampleFile)) {
      String[] phrases = line.split("\t");
      String utterance = phrases[0];
      String original = phrases[1];
      String[] utteranceTokens = utterance.split("\\s+");
      String[] originalTokens = original.split("\\s+");

      align(utteranceTokens, originalTokens);
    }
    normalize(threshold);
  }

  public void saveModel(String out) throws IOException {
    PrintWriter writer = IOUtils.getPrintWriter(out);
    for (String source: model.keySet()) {
      Counter<String> counts = model.get(source);
      for (String target: counts.keySet()) {
        writer.println(Joiner.on('\t').join(source, target, counts.getCount(target)));
      }
    }
  }

  //normalize all the counts to get conditional probabilities
  private void normalize(int threshold) {
    for (String source: model.keySet()) {
      Counter<String> counts = model.get(source);
      Counters.removeKeys(counts, Counters.keysBelow(counts, threshold));
      Counters.normalize(counts);
    }
  }

  //count every co-occurrence
  private void align(String[] utteranceTokens, String[] originalTokens) {
    for (String utteranceToken: utteranceTokens) {
      for (String originalToken: originalTokens) {
        MapUtils.putIfAbsent(model, utteranceToken.toLowerCase(), new ClassicCounter<>());
        MapUtils.putIfAbsent(model, originalToken.toLowerCase(), new ClassicCounter<>());
        model.get(utteranceToken.toLowerCase()).incrementCount(originalToken.toLowerCase());
        model.get(originalToken.toLowerCase()).incrementCount(utteranceToken.toLowerCase());
      }
    }
  }

  //remove '?' and '.'
  public String preprocessUtterance(String utterance) {
    if (utterance.endsWith("?"))
      return utterance.substring(0, utterance.length() - 1);
    if (utterance.endsWith("."))
      return utterance.substring(0, utterance.length() - 1);
    return utterance;
  }

  //read from serialized file
  public static Aligner read(File from) {
    Aligner res = new Aligner();
    try {
      for (String line : edu.stanford.nlp.io.IOUtils.readLines(from)) {
        String[] tokens = line.split("\t");
        MapUtils.putIfAbsent(res.model, tokens[0], new ClassicCounter<>());
        res.model.get(tokens[0]).incrementCount(tokens[1], Double.parseDouble(tokens[2]));
      }
    } catch (RuntimeIOException e) {
      if (!(e.getCause() instanceof FileNotFoundException))
        throw e;
    }
    return res;
  }

  public static Aligner read(String path, String languageTag) {
    // try path.languageTag, if that fails, read just path
    File withLanguage = new File(path + "." + languageTag);
    if (withLanguage.exists())
      return read(withLanguage);
    else
      return read(new File(path));
  }

  private void berkeleyAlign(String file, int threshold) {
    for (String line: IOUtils.readLines(file)) {
      String[] tokens = line.split("\t");
      if (tokens.length < 3) {
        System.err.println("Skipped line " + line);
        continue;
      }
      String[] sourceTokens = tokens[0].split("\\s+");
      String[] targetTokens = tokens[1].split("\\s+");
      String[] alignmentTokens = tokens[2].split("\\s+");
      for (String alignmentToken: alignmentTokens) {
        String[] alignment = alignmentToken.split("-");
        Integer source = Integer.parseInt(alignment[0]);
        Integer target = Integer.parseInt(alignment[1]);
        MapUtils.putIfAbsent(model, sourceTokens[source], new ClassicCounter<>());
        MapUtils.putIfAbsent(model, targetTokens[target], new ClassicCounter<>());
        model.get(sourceTokens[source]).incrementCount(targetTokens[target]);
        model.get(targetTokens[target]).incrementCount(sourceTokens[source]);
      }
    }
    normalize(threshold);
  }

  //args[0] - example file with utterance and original
  //args[1] - output file
  //args[2] - heuristic or berkeley
  //args[3] - threshold
  public static void main(String[] args) {
    Aligner aligner = new Aligner();
    int threshold = Integer.parseInt(args[3]);
    switch (args[2]) {
    case "heuristic":
      aligner.heuristicAlign(args[0], threshold);
      break;
    case "berkeley":
      aligner.berkeleyAlign(args[0], threshold);
      break;
    case "heuristic-tsv":
      aligner.heuristicTsvAlign(args[0], threshold);
      break;
    default:
      throw new RuntimeException("bad alignment mode: " + args[2]);
    }
    try {
      aligner.saveModel(args[1]);
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    }
  }


}
