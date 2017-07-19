package edu.stanford.nlp.sempre.corenlp;

import java.io.IOException;
import java.util.*;

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
    public Set<String> extraDictionary = Sets.newHashSet("bodytrace", "bodytracescale", "bmi", "dm", "dms",
        "file/folder", "gifs", "giphy", "github", "gmail", "gmails", "hashtag", "hashtags", "heatpad", "hvac",
        "icalendar", "ig", "imgflip", "images", "inbox", "insta", "linkedin", "mlb", "n't", "omlet",
        "onedrive", "parklon", "phd", "phdcomics", "popup", "powerpost", "reblog", "reddit", "retweet",
        "retweeted", "rss", "'s", "sms", "sportradar", "tumblr", "uber", "unfollow", "unmute", "wapo",
        "weatherapi", "webos", "wsj", "xkcd");
  }

  public static Options opts = new Options();

  private final HunspellDictionary dictionary;

  private static final Set<String> PTB_PUNCTUATION = Sets.newHashSet("-lrb-", "-lsb-", "-rrb-", "-rsb-", "'", "`", "''",
      "``");

  private static final Map<String, String> HARDCODED_REPLACEMENTS = new HashMap<>();
  static {
    // missing brands or words that hunspell doesn't recognize
    // it's possible that hunspell would do these on its own 
    HARDCODED_REPLACEMENTS.put("bling", "bing");
    HARDCODED_REPLACEMENTS.put("bodytrance", "bodytrace");
    HARDCODED_REPLACEMENTS.put("hastag", "hashtag");
    HARDCODED_REPLACEMENTS.put("headpad", "heatpad");
    HARDCODED_REPLACEMENTS.put("heapad", "heatpad");
    HARDCODED_REPLACEMENTS.put("ingmail", "in gmail");
    HARDCODED_REPLACEMENTS.put("linkediin", "linkedin");
    HARDCODED_REPLACEMENTS.put("linkenin", "linkedin");
    HARDCODED_REPLACEMENTS.put("mygmail", "my gmail");
    HARDCODED_REPLACEMENTS.put("nasas", "nasa 's");
    HARDCODED_REPLACEMENTS.put("omle", "omlet");
    HARDCODED_REPLACEMENTS.put("parklod", "parklon");
    HARDCODED_REPLACEMENTS.put("parkon", "parklon");
    HARDCODED_REPLACEMENTS.put("redditt", "reddit");
    HARDCODED_REPLACEMENTS.put("sportrader", "sportradar");
    HARDCODED_REPLACEMENTS.put("tmblr", "tumblr");
    HARDCODED_REPLACEMENTS.put("ubert", "uber");
    HARDCODED_REPLACEMENTS.put("wenos", "webos");
    HARDCODED_REPLACEMENTS.put("xkdc", "xkcd");

    // for some reason hunspell thinks these have to do with bullfights
    HARDCODED_REPLACEMENTS.put("mylightbulb", "my light bulb");
    HARDCODED_REPLACEMENTS.put("lighbulb", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbub", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbul", "light bulb");
    HARDCODED_REPLACEMENTS.put("lightbulbif", "light bulb if");
    HARDCODED_REPLACEMENTS.put("lightbulp", "light bulb");
    HARDCODED_REPLACEMENTS.put("secion", "section");
    HARDCODED_REPLACEMENTS.put("timestap", "timestamp");

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

  private boolean slashCompoundWord(String word) {
    String[] split = word.split("/");
    if (split.length <= 1)
      return false;
    
    for (String splitword : split) {
      if (!dictionary.spell(splitword))
        return false;
    }
    
    return true;
  }

  @Override
  public void annotate(Annotation annotation) {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);

    List<CoreLabel> newTokens = new ArrayList<>();
    for (CoreLabel token : tokens) {
      String word = token.get(CoreAnnotations.TextAnnotation.class);

      if (token.get(QuoteAnnotation.class) != null) {
        newTokens.add(token);
        continue;
      }

      if (PTB_PUNCTUATION.contains(word) || opts.extraDictionary.contains(word.toLowerCase())) {
        newTokens.add(token);
        continue;
      }

      if (dictionary.spell(word)) {
        newTokens.add(token);
        continue;
      }

      if (slashCompoundWord(word)) {
        newTokens.add(token);
        continue;
      }

      if (HARDCODED_REPLACEMENTS.containsKey(word.toLowerCase())) {
        doReplace(token, newTokens, word, HARDCODED_REPLACEMENTS.get(word.toLowerCase()));
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
