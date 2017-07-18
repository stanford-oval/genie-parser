package edu.stanford.nlp.sempre.corenlp;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.sempre.LanguageInfo;

public class URLEntityRecognizer implements NamedEntityRecognizer {
  private static final Pattern REGEXP = Pattern
      .compile("(https?://(?:www\\.|(?!www))[^\\.]+\\..{2,}|www\\..+\\..{2,}|.{2,}\\.(?:com|net|org))");

  @Override
  public void recognize(LanguageInfo info) {
    for (int i = 0; i < info.numTokens(); i++) {
      Matcher matcher = REGEXP.matcher(info.tokens.get(i));
      if (matcher.matches()) {
        String url = matcher.group();
        if (!url.startsWith("http"))
          url = "http://" + url;

        info.nerTags.set(i, "URL");
        info.nerValues.set(i, url);
      }
    }
  }
}
