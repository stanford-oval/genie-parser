package edu.stanford.nlp.sempre.corenlp;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.sempre.LanguageInfo;

public class EmailEntityRecognizer implements NamedEntityRecognizer {
  private static final Pattern EMAIL_REGEX = Pattern.compile("^[^@]+@\\p{Alnum}+\\.[\\p{Alnum}\\.]+$",
      Pattern.UNICODE_CHARACTER_CLASS);

  @Override
  public void recognize(LanguageInfo info) {
    int n = info.numTokens();
    for (int i = 0; i < n; i++) {
      Matcher matcher = EMAIL_REGEX.matcher(info.tokens.get(i));
      if (matcher.matches()) {
        info.nerTags.set(i, "EMAIL_ADDRESS");
        info.nerValues.set(i, matcher.group());
      }
    }
  }

  public static void main(String[] args) {
    Scanner scanner = new Scanner(System.in);

    String line;
    while ((line = scanner.nextLine()) != null) {
      Matcher matcher = EMAIL_REGEX.matcher(line);
      System.out.println(matcher.matches() ? matcher.group() : "null");
    }
  }
}
