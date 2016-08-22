package edu.stanford.nlp.sempre.overnight;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import fig.basic.BipartiteMatcher;
import fig.basic.MapUtils;

public class PhraseAligner {

  private final Map<String, Counter<String>> model = new HashMap<>();
  private final Aligner berkeleyAligner;
  private final String languageTag;

  public PhraseAligner(String berkeleyFile, String languageTag) {
    this.languageTag = languageTag;
    berkeleyAligner = Aligner.read(berkeleyFile, languageTag);
  }

  private double[][] computeLexicalAlignmentMatrix(List<String> utteranceTokens, List<String> originalTokens) {
    double[][] res = new double[utteranceTokens.size() + originalTokens.size()][utteranceTokens.size()
        + originalTokens.size()];
    //init with -infnty and low score on the diagonal
    for (int i = 0; i < res.length - 1; i++) {
      for (int j = i; j < res.length; j++) {
        if (i == j) {
          res[i][j] = 0d;
          res[j][i] = 0d;
        } else {
          res[i][j] = -1000d;
          res[j][i] = -1000d;
        }
      }
    }

    for (int i = 0; i < utteranceTokens.size(); ++i) {
      for (int j = 0; j < originalTokens.size(); ++j) {
        String inputToken = utteranceTokens.get(i);
        String derivToken = originalTokens.get(j);

        double product = berkeleyAligner.getCondProb(inputToken, derivToken)
            * berkeleyAligner.getCondProb(derivToken, inputToken);
        res[i][utteranceTokens.size() + j] = product;
        res[utteranceTokens.size() + j][i] = product;
      }
    }
    return res;
  }

  // This function implements the phrase-align algorithm from
  // "The Alignment Template Approach to Statistical Machine Translation"
  // Franz Josef Och, Hermann Ney, ACL 2004
  //
  // as cited by the Overnight paper
  public void phraseAlign(String exampleFile, int threshold) {
    model.clear();

    CoreNLPAnalyzer.opts.annotators = Lists.newArrayList("ssplit");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer(languageTag);

    for (String line : IOUtils.readLines(exampleFile)) {
      String[] phrases = line.split("\t");
      if (phrases.length < 2) {
        System.err.println("Unexpected badly formatted line: " + line);
        continue;
      }
      String utterance = phrases[0];
      String original = phrases[1];

      LanguageInfo utteranceInfo = analyzer.analyze(utterance);
      LanguageInfo originalInfo = analyzer.analyze(original);

      double[][] alignmentMatrix = computeLexicalAlignmentMatrix(utteranceInfo.tokens, originalInfo.tokens);
      BipartiteMatcher bMatcher = new BipartiteMatcher();
      int[] assignment = bMatcher.findMaxWeightAssignment(alignmentMatrix);

      for (int i1 = 0; i1 < utteranceInfo.numTokens(); i1++) {
        for (int i2 = i1+1; i2 <= utteranceInfo.numTokens(); i2++) {
          // span utterance tokens [i1, i2)
          
          // TP = target-phrase
          // (terminology from the paper)
          int[] TP = new int[i2-i1];
          // we use -1 to mean "not aligned"
          for (int k = 0; k < i2-i1; k++) {
            TP[k] = assignment[i1+k] >= utteranceInfo.numTokens() ? assignment[i1+k] - utteranceInfo.numTokens() : -1;
          }
          
          int jmin = -1, jmax = -1;
          boolean anyAligned = false;
          for (int k = 0; k < i2-i1; k++) {
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
          for (int k = 0; k < jmax - jmin + 1; k++) {
            SP[k] = assignment[utteranceInfo.numTokens() + jmin + k] != utteranceInfo.numTokens() + jmin + k
                ? assignment[utteranceInfo.numTokens() + jmin + k] : -1;
          }

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

          hit(Joiner.on(' ').join(utteranceInfo.tokens.subList(i1, i2)),
              Joiner.on(' ').join(originalInfo.tokens.subList(jmin, jmax + 1)));
        }
      }
      
      System.out.printf("processed %s to %s\n", utterance, original);
    }

    // do not normalize
    applyThreshold(threshold);
  }

  //count every co-occurrence
  private void hit(String utterancePhrase, String originalPhrase) {
    MapUtils.putIfAbsent(model, utterancePhrase.toLowerCase(), new ClassicCounter<>());
    MapUtils.putIfAbsent(model, originalPhrase.toLowerCase(), new ClassicCounter<>());
    model.get(utterancePhrase.toLowerCase()).incrementCount(originalPhrase.toLowerCase());
    model.get(originalPhrase.toLowerCase()).incrementCount(utterancePhrase.toLowerCase());
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

  //args[0] - example file with utterance and original
  //args[1] - output file
  //args[2] - berkeley align file
  //args[3] - language tag
  //args[4] - threshold
  public static void main(String[] args) {
    PhraseAligner aligner = new PhraseAligner(args[2], args[3]);
    int threshold = Integer.parseInt(args[4]);
    aligner.phraseAlign(args[0], threshold);
    try {
      aligner.saveModel(args[1]);
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    }
  }
}
