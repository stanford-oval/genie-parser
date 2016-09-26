package edu.stanford.nlp.sempre.ifttt;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import fig.basic.Pair;

public class ExtractEntities {
  public static void main(String[] params) {
    CoreNLPAnalyzer.opts.entityRecognizers = Lists.newArrayList("corenlp.PhoneNumberEntityRecognizer",
        "corenlp.EmailEntityRecognizer");
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer("en");

    try (Scanner scanner = new Scanner(System.in)) {

      while (scanner.hasNext()) {
        String line = scanner.nextLine();

        // id, description, iftttcode
        String[] split = line.split("\t+");
        if (split.length < 3) {
          System.err.println("Skipped badly formatted line " + line);
          continue;
        }
        String id = split[0];
        String description = split[1];
        String ifttt = split[2];

        LanguageInfo info = analyzer.analyze(description);

        String previousTag = null, previousValue = null;

        List<Pair<String, String>> entities = new ArrayList<>();
        for (int i = 0; i < info.numTokens(); i++) {
          String token = info.tokens.get(i);
          String nerTag = info.nerTags.get(i);
          String nerValue = info.nerValues.get(i);
          if ("O".equals(nerTag))
            nerTag = null;

          if (nerTag == null && token.startsWith("#")) {
            nerTag = "HASHTAG";
            nerValue = token.substring(1);
          }
          if (nerTag == null && token.startsWith("@")) {
            nerTag = "USER";
            nerValue = token.substring(1);
          }

          if (nerTag == null || nerValue == null) {
            previousTag = nerTag;
            previousValue = nerValue;
            continue;
          }
          if (nerTag.equals(previousTag) || nerValue.equals(previousValue))
            continue;

          entities.add(new Pair<>(nerTag, nerValue));

          previousTag = nerTag;
          previousValue = nerValue;
        }

        System.out.print(id + "\t" + description + "\t" + ifttt + "\t");
        boolean first = true;
        for (Pair<String, String> pair : entities) {
          String tag = pair.getFirst();
          String value = pair.getSecond();
          if (!first)
            System.out.print(",");
          first = false;
          System.out.print(tag + "=\"" + value + "\"");
        }
        System.out.println();
      }
    }
  }
}
