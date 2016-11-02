package edu.stanford.nlp.sempre.overnight;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import com.google.common.base.Joiner;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import fig.basic.MapUtils;

public class PhraseAligner {

  private final Map<String, Counter<String>> model = new HashMap<>();
  private final String languageTag;

  public PhraseAligner(String languageTag) {
    this.languageTag = languageTag;
  }

  // This function implements the phrase-align algorithm from
  // "The Alignment Template Approach to Statistical Machine Translation"
  // Franz Josef Och, Hermann Ney, ACL 2004
  //
  // as cited by the Overnight paper
  public void phraseAlign(String exampleFile) {
    model.clear();

    for (String line : IOUtils.readLines(exampleFile)) {
      String[] phrases = line.split("\t");
      if (phrases.length < 2) {
        System.err.println("Unexpected badly formatted line: " + line);
        continue;
      }
      String utterance = phrases[0];
      String original = phrases[1];

      List<String> utteranceTokens = Arrays.asList(utterance.split(" "));
      List<String> originalTokens = Arrays.asList(original.split(" "));

      int[] utteranceAlignment = new int[utteranceTokens.size()];

      // we use -1 to mean "not aligned"
      for (int i = 0; i < utteranceAlignment.length; i++)
          utteranceAlignment[i] = -1;
      int[] originalAlignment = new int[originalTokens.size()];
      for (int i = 0; i < originalAlignment.length; i++)
          originalAlignment[i] = -1;
      
      String[] alignInfo = phrases[2].split(" ");
      for (String alignToken : alignInfo) {
        String[] tokens = alignToken.split("-");
        int utteranceToken = Integer.valueOf(tokens[0]);
        int originalToken = Integer.valueOf(tokens[1]);

        utteranceAlignment[utteranceToken] = originalToken;
        originalAlignment[originalToken] = utteranceToken;
      }
      
      for (int i1 = 0; i1 < utteranceTokens.size(); i1++) {
        for (int i2 = i1 + 1; i2 <= utteranceTokens.size(); i2++) {
          // span utterance tokens [i1, i2)

          // TP = target-phrase
          // (terminology from the paper)
          int[] TP = new int[i2 - i1];
          for (int k = 0; k < i2 - i1; k++)
            TP[k] = utteranceAlignment[i1 + k];

          int jmin = -1, jmax = -1;
          boolean anyAligned = false;
          for (int k = 0; k < i2 - i1; k++) {
            if (TP[k] >= 0) {
              anyAligned = true;
              jmin = TP[k];
              jmax = TP[k];
              break;
            }
          }
          if (!anyAligned)
            continue;
          boolean quasiConsecutive = true;
          for (int k = 0; k < i2 - i1; k++) {
            if (TP[k] < 0)
              continue;
            if (TP[k] < jmin - 1 || TP[k] > jmax + 1) {
              quasiConsecutive = false;
              break;
            }
            if (TP[k] < jmin)
              jmin = TP[k];
            if (TP[k] > jmax)
              jmax = TP[k];
          }
          if (!quasiConsecutive)
            continue;

          int[] SP = new int[jmax - jmin + 1];
          for (int k = 0; k < jmax - jmin + 1; k++)
            SP[k] = originalAlignment[jmin + k];

          boolean inSourceSpan = true;
          for (int k = 0; k < jmax - jmin + 1; k++) {
            if (SP[k] < 0)
              continue;
            if (SP[k] < i1 || SP[k] >= i2) {
              inSourceSpan = false;
              break;
            }
          }
          if (!inSourceSpan)
            continue;

          // ignore 1 to 1 mappings (handled by the berkeley aligner files instead)
          if (i2 - i1 == 1 && jmax - jmin == 0)
            continue;
          // HEURISTIC: ignore many to 1 mappings
          if (jmax - jmin == 0)
            continue;

          hit(Joiner.on(' ').join(utteranceTokens.subList(i1, i2)),
              Joiner.on(' ').join(originalTokens.subList(jmin, jmax + 1)));
        }
      }
      
      //System.out.printf("processed %s to %s\n", utterance, original);
    }
  }

  //count every co-occurrence
  private void hit(String utterancePhrase, String originalPhrase) {
    MapUtils.putIfAbsent(model, originalPhrase, new ClassicCounter<>());
    model.get(originalPhrase).incrementCount(utterancePhrase);
  }

  private void applyThreshold(int threshold) {
    for (String source : model.keySet()) {
      Counter<String> counts = model.get(source);
      Counters.removeKeys(counts, Counters.keysBelow(counts, threshold));
    }
  }

  public void saveModel(String out) throws IOException {
    PrintWriter writer = IOUtils.getPrintWriter(out);
    for (String source : model.keySet()) {
      Counter<String> counts = model.get(source);
      for (String target : counts.keySet()) {
        writer.println(Joiner.on('\t').join(source, target, counts.getCount(target)));
      }
    }
  }

  //args[0] - example file with utterance, original and alignment
  //args[1] - output file
  //args[2] - language tag
  //args[3] - threshold
  public static void main(String[] args) {
    PhraseAligner aligner = new PhraseAligner(args[2]);
    int threshold = Integer.parseInt(args[3]);
    aligner.phraseAlign(args[0]);
    // do not normalize
    aligner.applyThreshold(threshold);
    try {
      aligner.saveModel(args[1]);
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    }
  }
}
