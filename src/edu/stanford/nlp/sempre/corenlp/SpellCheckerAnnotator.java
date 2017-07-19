package edu.stanford.nlp.sempre.corenlp;

import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

import com.google.common.collect.Sets;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.sempre.corenlp.QuotedStringAnnotator.QuoteAnnotation;
import edu.stanford.nlp.util.ArraySet;
import fig.basic.LogInfo;
import fig.basic.Option;

public class SpellCheckerAnnotator implements Annotator {
  public static class Options {
    @Option
    public String dictionaryDirectory = "/usr/share/myspell";

    @Option
    public Set<String> extraDictionary = Sets.newHashSet("bodytrace", "bmi", "dm", "dms",
        "gifs", "giphy", "github", "gmail", "gmails", "hashtag", "hashtags", "heatpad", "hvac",
        "icalendar", "ig", "imgflip", "images", "inbox", "insta", "linkedin", "mlb", "n't", "omlet",
        "onedrive", "parklon", "phd", "phdcomics", "popup", "powerpost", "reblog", "reddit", "retweet",
        "retweeted", "rss", "'s", "sms", "sportradar", "tumblr", "uber", "unfollow", "unmute", "wapo",
        "weatherapi", "webos", "wsj", "xkcd", "facebook", "youtube", "'m", "'ll", "'ve", "'re", "lakers", "bing",
        "repo", "yandex", "aapl", "haz", "moar", "lol", "juventus", "fitbit", "kcal", "ncaafb", "unclosed",
        "twilio", "mms", "bing", "xkcds", "kbs", "btc", "momma", "gps", "wtf", "apod", "diy", "smss",
        "preselect", "vid", "vids", "ubering", "urls");
  }

  public static Options opts = new Options();

  private final HunspellDictionary dictionary;

  private static final Pattern NUMERIC_PATTERN = Pattern.compile("[-+0-9:/.]+.*");
  private static final Pattern BLANK_PATTERN = Pattern.compile("_+");
  private static final Set<String> PTB_PUNCTUATION = Sets.newHashSet("-lrb-", "-lsb-", "-rrb-", "-rsb-", "'", "`", "''",
      "``", ",", "-LRB-", "-RRB-", "%", ">", "<", "-LSB-", "-RSB-");

  private static final Map<String, String> HARDCODED_REPLACEMENTS = new HashMap<>();
  static {
    // missing brands or words that hunspell doesn't recognize
    // it's possible that hunspell would do these on its own
    HARDCODED_REPLACEMENTS.put("allmend", "almond");
    HARDCODED_REPLACEMENTS.put("bling", "bing");
    HARDCODED_REPLACEMENTS.put("bingg", "bing");
    HARDCODED_REPLACEMENTS.put("bodytrance", "bodytrace");
    HARDCODED_REPLACEMENTS.put("bodytracescale", "bodytrace scale");
    HARDCODED_REPLACEMENTS.put("hastag", "hashtag");
    HARDCODED_REPLACEMENTS.put("hastags", "hashtags");
    HARDCODED_REPLACEMENTS.put("headpad", "heatpad");
    HARDCODED_REPLACEMENTS.put("heapad", "heatpad");
    HARDCODED_REPLACEMENTS.put("ingmail", "in gmail");
    HARDCODED_REPLACEMENTS.put("imgfli", "imgflip");
    HARDCODED_REPLACEMENTS.put("instgram", "instagram");
    HARDCODED_REPLACEMENTS.put("linkediin", "linkedin");
    HARDCODED_REPLACEMENTS.put("linkenin", "linkedin");
    HARDCODED_REPLACEMENTS.put("LinkdIn", "linkedin");
    HARDCODED_REPLACEMENTS.put("mens", "men 's");
    HARDCODED_REPLACEMENTS.put("mygmail", "my gmail");
    HARDCODED_REPLACEMENTS.put("nasas", "nasa 's");
    HARDCODED_REPLACEMENTS.put("omle", "omlet");
    HARDCODED_REPLACEMENTS.put("parklod", "parklon");
    HARDCODED_REPLACEMENTS.put("parkon", "parklon");
    HARDCODED_REPLACEMENTS.put("peklon", "parklon");
    HARDCODED_REPLACEMENTS.put("redditt", "reddit");
    HARDCODED_REPLACEMENTS.put("sportrader", "sportradar");
    HARDCODED_REPLACEMENTS.put("tmblr", "tumblr");
    HARDCODED_REPLACEMENTS.put("tumbr", "tumblr");
    HARDCODED_REPLACEMENTS.put("ubert", "uber");
    HARDCODED_REPLACEMENTS.put("wenos", "webos");
    HARDCODED_REPLACEMENTS.put("weos", "webos");
    HARDCODED_REPLACEMENTS.put("xkdc", "xkcd");
    HARDCODED_REPLACEMENTS.put("xldc", "xkcd");
    HARDCODED_REPLACEMENTS.put("xjcd", "xkcd");

    // for some reason hunspell thinks these have to do with bullfights
    HARDCODED_REPLACEMENTS.put("evey", "every");
    HARDCODED_REPLACEMENTS.put("frontpage", "front page");
    HARDCODED_REPLACEMENTS.put("lighbulb", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbub", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbul", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbulbif", "light bulb if");
    HARDCODED_REPLACEMENTS.put("lightbulp", "light bulb");
    HARDCODED_REPLACEMENTS.put("mylightbulb", "my light bulb");
    HARDCODED_REPLACEMENTS.put("secion", "section");
    HARDCODED_REPLACEMENTS.put("timestap", "time stamp");

    // PTB tokenizer weirdness
    HARDCODED_REPLACEMENTS.put("earth.at", "earth at");
  }

  public SpellCheckerAnnotator(String name, Properties props) {
    this(props == null ? "en_US" : (String) props.getOrDefault("spellcheck.dictPath", "en_US"));
  }

  private SpellCheckerAnnotator(String languageTag) {
    switch (languageTag) {
    case "en":
      languageTag = "en_US";
      break;
    case "it":
      languageTag = "it_IT";
      break;
    default:
      // don't add a country, and hope for the best  
    }

    try {
      dictionary = new HunspellDictionary(opts.dictionaryDirectory + "/" + languageTag);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private boolean slashCompoundWord(CoreLabel token, String word, List<CoreLabel> newTokens) {
    String[] split = word.split("/");
    if (split.length <= 1)
      return false;
    
    for (String splitword : split) {
      if (!dictionary.spell(splitword))
        return false;
    }
    
    String[] lemmaSplit = token.get(CoreAnnotations.LemmaAnnotation.class).split("/");

    int begin = token.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
    for (int i = 0; i < split.length; i++) {
      if (i > 0) {
        CoreLabel slashToken = new CoreLabel(token);
        slashToken.set(CoreAnnotations.TextAnnotation.class, "/");
        //slashToken.set(CoreAnnotations.LemmaAnnotation.class, "/");
        slashToken.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, begin);
        slashToken.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, begin + 1);
        begin++;
        newTokens.add(slashToken);
      }
      CoreLabel newToken = new CoreLabel(token);
      newToken.set(CoreAnnotations.TextAnnotation.class, split[i]);
      //newToken.set(CoreAnnotations.LemmaAnnotation.class, lemmaSplit[i]);
      newToken.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, begin);
      newToken.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, begin + split[i].length());
      begin += split[i].length();
      newTokens.add(newToken);
    }

    return true;
  }

  @Override
  public void annotate(Annotation annotation) {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

    List<CoreLabel> newTokens = new ArrayList<>();
    for (CoreLabel token : tokens) {
      String word = token.get(CoreAnnotations.TextAnnotation.class);

      if (token.get(QuoteAnnotation.class) != null ||
          PTB_PUNCTUATION.contains(word) ||
          word.contains("@") || word.startsWith("#") || word.length() <= 2 ||
          NUMERIC_PATTERN.matcher(word).matches() ||
          BLANK_PATTERN.matcher(word).matches() ||
          word.startsWith("http") || word.startsWith("www") ||
          (token.ner() != null && !"O".equals(token.ner()))) {
        newTokens.add(token);
        continue;
      }

      if (opts.extraDictionary.contains(word.toLowerCase()) || dictionary.spell(word)) {
        newTokens.add(token);
        continue;
      }

      if (HARDCODED_REPLACEMENTS.containsKey(word.toLowerCase())) {
        doReplace(token, newTokens, word, HARDCODED_REPLACEMENTS.get(word.toLowerCase()));
        continue;
      }

      if (slashCompoundWord(token, word, newTokens)) {
        continue;
      }

      List<String> replacements = dictionary.suggest(word);
      if (replacements.isEmpty()) {
        LogInfo.logs("Found no replacement for mispelled word %s", word);
        newTokens.add(token);
        continue;
      } else {
        doReplace(token, newTokens, word, replacements.get(0));
      }
    }
    assert newTokens.size() >= tokens.size();

    for (int i = 0; i < newTokens.size(); i++) {
      newTokens.get(i).setIndex(i);
    }

    annotation.set(CoreAnnotations.TokensAnnotation.class, newTokens);
  }

  private void doReplace(CoreLabel token, List<CoreLabel> newTokens, String word, String replacement) {
    if (word.equalsIgnoreCase(replacement)) {
      newTokens.add(token);
      return;
    }

    String[] newWords = replacement.split("\\s+");

    for (String newWord : newWords) {
      CoreLabel newToken = new CoreLabel(token);
      newToken.set(CoreAnnotations.TextAnnotation.class, newWord);
      // we're not going to rerun the lemmatizer, so just go with it
      newToken.set(CoreAnnotations.LemmaAnnotation.class, newWord);
      newTokens.add(newToken);
    }
    LogInfo.logs("Replaced mispelled word %s as %s", word, replacement);
  }

  @Override
  public Set<Class<? extends CoreAnnotation>> requirementsSatisfied() {
    return Collections.emptySet();
  }

  @Override
  public Set<Class<? extends CoreAnnotation>> requires() {
    // TODO Auto-generated method stub
    return Collections.unmodifiableSet(new ArraySet<>(Arrays.asList(
        CoreAnnotations.TextAnnotation.class,
        CoreAnnotations.TokensAnnotation.class,
        CoreAnnotations.PositionAnnotation.class)));
  }

}
