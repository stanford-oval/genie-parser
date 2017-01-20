package edu.stanford.nlp.sempre.thingtalk;

import java.util.Arrays;

import edu.stanford.nlp.sempre.Builder;

class LexiconUtils {
  private LexiconUtils() {
  }

  // a list of words that appear often in our examples (and thus are frequent queries to
  // the lexicon), but are not useful to lookup canonical forms
  // with FloatingParser, if the lookup word is in this array, we just return no
  // derivations
  private static final String[] IGNORED_WORDS = { "in", "is", "of", "or", "not", "my", "i",
      "at", "as", "by",
      "from", "for", "an", "on", "a", "to", "with", "and", "'s", "'", "s", "when",
      "notify", "monitor", "it", "?", "me", "the", "if", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwz" };
  static {
    Arrays.sort(IGNORED_WORDS);
  }

  public static String preprocessRawPhrase(String rawPhrase) {
    String[] tokens = rawPhrase.split(" ");
    if (Builder.opts.parser.equals("BeamParser")) {
      if (tokens.length < 3 || tokens.length > 7)
        return null;
      if (!"on".equals(tokens[tokens.length - 2]))
        return null;
      return rawPhrase;
    } else {
      if (tokens.length > 1)
        return null;
      if (Arrays.binarySearch(IGNORED_WORDS, tokens[0]) >= 0)
        return null;
      return tokens[0];
    }
  }
}
