package edu.stanford.nlp.sempre.corenlp;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.sempre.LanguageInfo;

public class RegexpEntityRecognizer implements NamedEntityRecognizer {
  private final Pattern regexp;
  private final String tag;

  public RegexpEntityRecognizer(String tag, String regexp) {
    this.tag = tag;
    this.regexp = Pattern.compile(regexp);
  }

  @Override
  public void recognize(LanguageInfo info) {
    for (int i = 0; i < info.numTokens(); i++) {
      if ("QUOTED_STRING".equals(info.nerTags.get(i)))
        continue;
      Matcher matcher = regexp.matcher(info.tokens.get(i));
      if (matcher.matches()) {
        info.nerTags.set(i, tag);
        info.nerValues.set(i, matcher.group(1));
      }
    }
  }

}
