package edu.stanford.nlp.sempre.corenlp;

import static java.lang.System.err;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.pascal.ISODateInstance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.util.*;

/**
 * A copy of edu.stanford.nlp.ie.regexp.QuantifiableEntityNormalizer, with
 * some bugfixes and without sutime and a ton of other unneeded stuff.
 * 
 * Have been removed:
 * - SUTime
 * - ranges
 * - millions abbreviated as "m" or "m."
 * - less than / greater than
 * 
 * The rest of the CoreNLP doc-string follows:
 * 
 * Various methods for normalizing Money, Date, Percent, Time, and
 * Number, Ordinal amounts.
 * These matchers are generous in that they try to quantify something
 * that's already been labelled by an NER system; don't use them to make
 * classification decisions. This class has a twin in the pipeline world:
 * {@link edu.stanford.nlp.pipeline.QuantifiableEntityNormalizingAnnotator}.
 * Please keep the substantive content here, however, so as to lessen code
 * duplication.
 * <p>
 * <i>Implementation note:</i> The extensive test code for this class is
 * now in a separate JUnit Test class. This class depends on the background
 * symbol for NER being the default background symbol. This should be fixed
 * at some point.
 *
 * @author Chris Cox
 * @author Christopher Manning (extended for RTE)
 * @author Anna Rafferty
 */
public class QuantifiableEntityNormalizer {

  private static final boolean DEBUG = false;
  private static final boolean DEBUG2 = false;  // String normlz functions

  public static final String BACKGROUND_SYMBOL = "O";

  private static final Pattern timePattern = Pattern
      .compile("([0-2]?[0-9])((?::[0-5][0-9]){0,2})([PpAa]\\.?[Mm]\\.?)?");

  private static final Pattern moneyPattern = Pattern
      .compile("([$\u00A3\u00A5\u20AC#]?)(-?[0-9,]+(?:\\.[0-9]*)?|\\.[0-9]+)[-a-zA-Z]*");

  //Collections of entity types
  private static final Set<String> quantifiable;  //Entity types that are quantifiable
  private static final Set<String> collapseBeforeParsing;
  private static final Map<String, String> timeUnitWords;
  private static final Map<String, Double> moneyMultipliers;
  private static final Map<String, Character> currencyWords;
  public static final ClassicCounter<String> wordsToValues;
  public static final ClassicCounter<String> ordinalsToValues;

  static {

    quantifiable = Generics.newHashSet();
    quantifiable.add("MONEY");
    quantifiable.add("TIME");
    quantifiable.add("DATE");
    quantifiable.add("PERCENT");
    quantifiable.add("NUMBER");
    quantifiable.add("ORDINAL");
    quantifiable.add("DURATION");

    collapseBeforeParsing = Generics.newHashSet();
    collapseBeforeParsing.add("PERSON");
    collapseBeforeParsing.add("ORGANIZATION");
    collapseBeforeParsing.add("LOCATION");

    timeUnitWords = Generics.newHashMap();
    timeUnitWords.put("second", "S");
    timeUnitWords.put("seconds", "S");
    timeUnitWords.put("minute", "m");
    timeUnitWords.put("minutes", "m");
    timeUnitWords.put("hour", "H");
    timeUnitWords.put("hours", "H");
    timeUnitWords.put("day", "D");
    timeUnitWords.put("days", "D");
    timeUnitWords.put("week", "W");
    timeUnitWords.put("weeks", "W");
    timeUnitWords.put("month", "M");
    timeUnitWords.put("months", "M");
    timeUnitWords.put("year", "Y");
    timeUnitWords.put("years", "Y");

    currencyWords = Generics.newHashMap();
    currencyWords.put("dollars?", '$');
    currencyWords.put("cents?", '$');
    currencyWords.put("pounds?", '\u00A3');
    currencyWords.put("pence|penny", '\u00A3');
    currencyWords.put("yen", '\u00A5');
    currencyWords.put("euros?", '\u20AC');
    currencyWords.put("won", '\u20A9');
    currencyWords.put("\\$", '$');
    currencyWords.put("\u00A2", '$');  // cents
    currencyWords.put("\u00A3", '\u00A3');  // pounds
    currencyWords.put("#", '\u00A3');      // for Penn treebank
    currencyWords.put("\u00A5", '\u00A5');  // Yen
    currencyWords.put("\u20AC", '\u20AC');  // Euro
    currencyWords.put("\u20A9", '\u20A9');  // Won
    currencyWords.put("yuan", '\u5143');   // Yuan

    moneyMultipliers = Generics.newHashMap();
    moneyMultipliers.put("trillion", 1000000000000.0);  // can't be an integer
    moneyMultipliers.put("billion", 1000000000.0);
    moneyMultipliers.put("bn", 1000000000.0);
    moneyMultipliers.put("million", 1000000.0);
    moneyMultipliers.put("thousand", 1000.0);
    moneyMultipliers.put("hundred", 100.0);
    moneyMultipliers.put("b.", 1000000000.0);
    moneyMultipliers.put(" k ", 1000.0);
    moneyMultipliers.put("dozen", 12.0);

    wordsToValues = new ClassicCounter<>();
    wordsToValues.setCount("zero", 0.0);
    wordsToValues.setCount("one", 1.0);
    wordsToValues.setCount("two", 2.0);
    wordsToValues.setCount("three", 3.0);
    wordsToValues.setCount("four", 4.0);
    wordsToValues.setCount("five", 5.0);
    wordsToValues.setCount("six", 6.0);
    wordsToValues.setCount("seven", 7.0);
    wordsToValues.setCount("eight", 8.0);
    wordsToValues.setCount("nine", 9.0);
    wordsToValues.setCount("ten", 10.0);
    wordsToValues.setCount("eleven", 11.0);
    wordsToValues.setCount("twelve", 12.0);
    wordsToValues.setCount("thirteen", 13.0);
    wordsToValues.setCount("fourteen", 14.0);
    wordsToValues.setCount("fifteen", 15.0);
    wordsToValues.setCount("sixteen", 16.0);
    wordsToValues.setCount("seventeen", 17.0);
    wordsToValues.setCount("eighteen", 18.0);
    wordsToValues.setCount("nineteen", 19.0);
    wordsToValues.setCount("twenty", 20.0);
    wordsToValues.setCount("thirty", 30.0);
    wordsToValues.setCount("forty", 40.0);
    wordsToValues.setCount("fifty", 50.0);
    wordsToValues.setCount("sixty", 60.0);
    wordsToValues.setCount("seventy", 70.0);
    wordsToValues.setCount("eighty", 80.0);
    wordsToValues.setCount("ninety", 90.0);

    ordinalsToValues = new ClassicCounter<>();
    ordinalsToValues.setCount("zeroth", 0.0);
    ordinalsToValues.setCount("first", 1.0);
    ordinalsToValues.setCount("second", 2.0);
    ordinalsToValues.setCount("third", 3.0);
    ordinalsToValues.setCount("fourth", 4.0);
    ordinalsToValues.setCount("fifth", 5.0);
    ordinalsToValues.setCount("sixth", 6.0);
    ordinalsToValues.setCount("seventh", 7.0);
    ordinalsToValues.setCount("eighth", 8.0);
    ordinalsToValues.setCount("ninth", 9.0);
    ordinalsToValues.setCount("tenth", 10.0);
    ordinalsToValues.setCount("eleventh", 11.0);
    ordinalsToValues.setCount("twelfth", 12.0);
    ordinalsToValues.setCount("thirteenth", 13.0);
    ordinalsToValues.setCount("fourteenth", 14.0);
    ordinalsToValues.setCount("fifteenth", 15.0);
    ordinalsToValues.setCount("sixteenth", 16.0);
    ordinalsToValues.setCount("seventeenth", 17.0);
    ordinalsToValues.setCount("eighteenth", 18.0);
    ordinalsToValues.setCount("nineteenth", 19.0);
    ordinalsToValues.setCount("twentieth", 20.0);
    ordinalsToValues.setCount("twenty-first", 21.0);
    ordinalsToValues.setCount("twenty-second", 22.0);
    ordinalsToValues.setCount("twenty-third", 23.0);
    ordinalsToValues.setCount("twenty-fourth", 24.0);
    ordinalsToValues.setCount("twenty-fifth", 25.0);
    ordinalsToValues.setCount("twenty-sixth", 26.0);
    ordinalsToValues.setCount("twenty-seventh", 27.0);
    ordinalsToValues.setCount("twenty-eighth", 28.0);
    ordinalsToValues.setCount("twenty-ninth", 29.0);
    ordinalsToValues.setCount("thirtieth", 30.0);
    ordinalsToValues.setCount("thirty-first", 31.0);
    ordinalsToValues.setCount("fortieth", 40.0);
    ordinalsToValues.setCount("fiftieth", 50.0);
    ordinalsToValues.setCount("sixtieth", 60.0);
    ordinalsToValues.setCount("seventieth", 70.0);
    ordinalsToValues.setCount("eightieth", 80.0);
    ordinalsToValues.setCount("ninetieth", 90.0);
    ordinalsToValues.setCount("hundredth", 100.0);
    ordinalsToValues.setCount("thousandth", 1000.0);
    ordinalsToValues.setCount("millionth", 1000000.0);
    ordinalsToValues.setCount("billionth", 1000000000.0);
    ordinalsToValues.setCount("trillionth", 1000000000000.0);
  }

  private QuantifiableEntityNormalizer() {
  } // this is all static

  /**
   * This method returns the closest match in set such that the match
   * has more than three letters and differs from word only by one substitution,
   * deletion, or insertion. If not match exists, returns null.
   */
  private static String getOneSubstitutionMatch(String word, Set<String> set) {
    // TODO (?) pass the EditDistance around more places to make this
    // more efficient.  May not really matter.
    EditDistance ed = new EditDistance();
    for (String cur : set) {
      if (isOneSubstitutionMatch(word, cur, ed))
        return cur;
    }
    return null;
  }

  private static boolean isOneSubstitutionMatch(String word, String match,
      EditDistance ed) {
    if (word.equalsIgnoreCase(match))
      return true;
    if (match.length() > 3) {
      if (ed.score(word, match) <= 1)
        return true;
    }
    return false;
  }

  /**
   * Convert the content of a List of CoreMaps to a single
   * space-separated String. This grabs stuff based on the
   * get(CoreAnnotations.NamedEntityTagAnnotation.class) field.
   * [CDM: Changed to look at NamedEntityTagAnnotation not AnswerClass Jun 2010,
   * hoping that will fix a bug.]
   *
   * @param l
   *          The List
   * @return one string containing all words in the list, whitespace separated
   */
  public static <E extends CoreMap> String singleEntityToString(List<E> l) {
    String entityType = l.get(0).get(CoreAnnotations.NamedEntityTagAnnotation.class);
    StringBuilder sb = new StringBuilder();
    for (E w : l) {
      assert (w.get(CoreAnnotations.NamedEntityTagAnnotation.class).equals(entityType));
      sb.append(w.get(CoreAnnotations.TextAnnotation.class));
      sb.append(' ');
    }
    return sb.toString();
  }

  /**
   * Provided for backwards compatibility; see normalizedDateString(s,
   * openRangeMarker)
   */
  static String normalizedDateString(String s) {
    return normalizedDateString(s, ISODateInstance.NO_RANGE);
  }

  /**
   * Returns a string that represents either a single date or a range of
   * dates. Representation pattern is roughly ISO8601, with some extensions
   * for greater expressivity; see {@link ISODateInstance} for details.
   * 
   * @param s
   *          Date string to normalize
   * @param openRangeMarker
   *          a marker for whether this date is not involved in
   *          an open range, is involved in an open range that goes forever
   *          backward and
   *          stops at s, or is involved in an open range that goes forever
   *          forward and
   *          starts at s. See {@link ISODateInstance}.
   * @return A yyyymmdd format normalized date
   */
  static String normalizedDateString(String s, String openRangeMarker) {
    ISODateInstance d = new ISODateInstance(s, openRangeMarker);
    if (DEBUG2)
      err.println("normalizeDate: " + s + " to " + d.getDateString());
    return (d.getDateString());
  }

  static String normalizedDurationString(String s) {
    s = s.trim();
    int space = s.lastIndexOf(' ');
    if (space < 0)
      return null;
    String timeword = s.substring(space+1);
    String numword = s.substring(0, space);
    
    String multiplier = timeUnitWords.get(timeword);
    if (multiplier == null)
      return null;
    
    String number = normalizedNumberString(numword);
    return "P" + number + multiplier;
  }

  /**
   * Concatenates separate words of a date or other numeric quantity into one
   * node (e.g., 3 November -> 3_November)
   * Tag is CD or NNP, and other words are added to the remove list
   */
  static <E extends CoreMap> void concatenateNumericString(List<E> words, List<E> toRemove) {
    if (words.size() <= 1)
      return;
    boolean first = true;
    StringBuilder newText = new StringBuilder();
    E foundEntity = null;
    for (E word : words) {
      if (foundEntity == null && (word.get(CoreAnnotations.PartOfSpeechAnnotation.class).equals("CD")
          || word.get(CoreAnnotations.PartOfSpeechAnnotation.class).equals("NNP"))) {
        foundEntity = word;
      }
      if (first) {
        first = false;
      } else {
        newText.append('_');
      }
      newText.append(word.get(CoreAnnotations.TextAnnotation.class));
    }
    if (foundEntity == null) {
      foundEntity = words.get(0);//if we didn't find one with the appropriate tag, just take the first one
    }
    toRemove.addAll(words);
    toRemove.remove(foundEntity);
    foundEntity.set(CoreAnnotations.PartOfSpeechAnnotation.class, "CD");  // cdm 2008: is this actually good for dates??
    String collapsed = newText.toString();
    foundEntity.set(CoreAnnotations.TextAnnotation.class, collapsed);
    foundEntity.set(CoreAnnotations.OriginalTextAnnotation.class, collapsed);
  }

  public static String normalizedTimeString(String s) {
    return normalizedTimeString(s, null);
  }

  public static String normalizedTimeString(String s, String ampm) {
    if (DEBUG2)
      err.println("normalizingTime: " + s);
    s = s.replaceAll("[ \t\n\0\f\r]", "");
    Matcher m = timePattern.matcher(s);
    if (s.equalsIgnoreCase("noon")) {
      return "12:00pm";
    } else if (s.equalsIgnoreCase("midnight")) {
      return "00:00am";  // or "12:00am" ?
    } else if (s.equalsIgnoreCase("morning")) {
      return "M";
    } else if (s.equalsIgnoreCase("afternoon")) {
      return "A";
    } else if (s.equalsIgnoreCase("evening")) {
      return "EN";
    } else if (s.equalsIgnoreCase("night")) {
      return "N";
    } else if (s.equalsIgnoreCase("day")) {
      return "D";
    } else if (s.equalsIgnoreCase("suppertime")) {
      return "EN";
    } else if (s.equalsIgnoreCase("lunchtime")) {
      return "MD";
    } else if (s.equalsIgnoreCase("midday")) {
      return "MD";
    } else if (s.equalsIgnoreCase("teatime")) {
      return "A";
    } else if (s.equalsIgnoreCase("dinnertime")) {
      return "EN";
    } else if (s.equalsIgnoreCase("dawn")) {
      return "EM";
    } else if (s.equalsIgnoreCase("dusk")) {
      return "EN";
    } else if (s.equalsIgnoreCase("sundown")) {
      return "EN";
    } else if (s.equalsIgnoreCase("sunup")) {
      return "EM";
    } else if (s.equalsIgnoreCase("daybreak")) {
      return "EM";
    } else if (m.matches()) {
      if (DEBUG2) {
        err.printf("timePattern matched groups: |%s| |%s| |%s| |%s|\n", m.group(0), m.group(1), m.group(2), m.group(3));
      }
      // group 1 is hours, group 2 is minutes and maybe seconds; group 3 is am/pm
      StringBuilder sb = new StringBuilder();
      sb.append(m.group(1));
      if (m.group(2) == null || "".equals(m.group(2))) {
        sb.append(":00");
      } else {
        sb.append(m.group(2));
      }
      if (m.group(3) != null) {
        String suffix = m.group(3);
        suffix = suffix.replaceAll("\\.", "");
        suffix = suffix.toLowerCase();
        sb.append(suffix);
      } else if (ampm != null) {
        sb.append(ampm);
      } else {
        // Do nothing; leave ambiguous
        // sb.append("pm");
      }
      if (DEBUG2) {
        err.println("normalizedTimeString new str: " + sb.toString());
      }
      return sb.toString();
    } else if (DEBUG) {
      err.println("Quantifiable: couldn't normalize " + s);
    }
    return null;
  }

  /**
   * Heuristically decides if s is in American (42.33) or European (42,33)
   * number format
   * and tries to turn European version into American.
   *
   */
  private static String convertToAmerican(String s) {
    if (s.contains(",")) {
      //turn all but the last into blanks - this isn't really correct, but it's close enough for now
      while (s.indexOf(',') != s.lastIndexOf(','))
        s = s.replaceFirst(",", "");
      int place = s.lastIndexOf(',');
      //if it's american, should have at least three characters after it
      if (place >= s.length() - 3 && place != s.length() - 1) {
        s = s.substring(0, place) + '.' + s.substring(place + 1);
      } else {
        s = s.replace(",", "");
      }
    }
    return s;
  }

  static String normalizedMoneyString(String s) {
    //first, see if it looks like european style
    s = convertToAmerican(s);
    // clean up string
    s = s.replaceAll("[ \t\n\0\f\r,]", "");
    s = s.toLowerCase();
    if (DEBUG2) {
      err.println("normalizedMoneyString: Normalizing " + s);
    }

    double multiplier = 1.0;

    // do currency words
    char currencySign = '$';
    for (String currencyWord : currencyWords.keySet()) {
      if (StringUtils.find(s, currencyWord)) {
        if (DEBUG2) {
          err.println("Found units: " + currencyWord);
        }
        if (currencyWord.equals("pence|penny") || currencyWord.equals("cents?") || currencyWord.equals("\u00A2")) {
          multiplier *= 0.01;
        }
        // if(DEBUG){err.println("Quantifiable: Found "+ currencyWord);}
        s = s.replaceAll(currencyWord, "");
        currencySign = currencyWords.get(currencyWord);
      }
    }

    // process rest as number
    String value = normalizedNumberStringQuiet(s, multiplier);
    if (value == null) {
      return null;
    } else {
      return currencySign + value;
    }
  }

  public static String normalizedNumberString(String s) {
    if (DEBUG2) {
      err.println("normalizedNumberString: normalizing " + s);
    }
    return normalizedNumberStringQuiet(s, 1.0);
  }

  private static final Pattern allSpaces = Pattern.compile(" *");

  public static String normalizedNumberStringQuiet(String s,
      double multiplier) {

    // clean up string
    s = s.replaceAll("[ \t\n\0\f\r]+", " ");
    if (allSpaces.matcher(s).matches()) {
      return null;
    }
    //see if it looks like european style
    s = convertToAmerican(s);
    // remove parenthesis around numbers
    // if PTBTokenized, this next bit should be a no-op
    // in some contexts parentheses might indicate a negative number, but ignore that.
    if (s.startsWith("(") && s.endsWith(")")) {
      s = s.substring(1, s.length() - 1);
      if (DEBUG2)
        err.println("Deleted (): " + s);
    }
    s = s.toLowerCase();
    
    // handle numbers written in words
    String[] parts = s.split("[ -]");
    if (DEBUG2)
      err.println("Looking for number words in |" + s + "|; multiplier is " + multiplier);
    
    double value = Double.NaN;
    for (String part : parts) {
      // check for hyphenated word like 4-Ghz: delete final -
      if (part.endsWith("-")) {
        part = part.substring(0, part.length() - 1);
      }

      Matcher m = moneyPattern.matcher(part);
      if (m.matches()) {
        if (DEBUG2) {
          err.println("Number matched with |" + m.group(2) + "| |" +
              m.group(3) + '|');
        }
        String numStr = m.group(2).replace(",", "");
        double v = Double.parseDouble(numStr);
        if (Double.isNaN(value))
          value = v;
        else
          value += v;
        continue;
      }

      // get multipliers like "billion"
      boolean found = false;
      for (String moneyTag : moneyMultipliers.keySet()) {
        if (part.equals(moneyTag)) {
          // if (DEBUG) {err.println("Quantifiable: Found "+ moneyTag);}
          if (Double.isNaN(value))
            value = moneyMultipliers.get(moneyTag);
          else
            value *= moneyMultipliers.get(moneyTag);
          found = true;
          break;
        } else {
          EditDistance ed = new EditDistance();
          if (isOneSubstitutionMatch(part,
              moneyTag, ed)) {
            if (Double.isNaN(value))
              value = moneyMultipliers.get(moneyTag);
            else
              value *= moneyMultipliers.get(moneyTag);
            found = true;
            break;
          }
        }
      }
      if (found)
        continue;
      
      if (wordsToValues.containsKey(part)) {
        if (Double.isNaN(value))
          value = wordsToValues.getCount(part);
        else
          value += wordsToValues.getCount(part);
      } else {
        String partMatch = getOneSubstitutionMatch(part, wordsToValues.keySet());
        if (partMatch != null) {
          if (Double.isNaN(value))
            value = wordsToValues.getCount(part);
          else
            value += wordsToValues.getCount(part);
        }
      }
    }
    if (!Double.isNaN(value))
      return Double.toString(value * multiplier);
    else
      return null;
  }

  public static String normalizedOrdinalString(String s) {
    if (DEBUG2) {
      err.println("normalizedOrdinalString: normalizing " + s);
    }
    return normalizedOrdinalStringQuiet(s);
  }

  public static final Pattern numberPattern = Pattern.compile("([0-9.]+)");

  public static String normalizedOrdinalStringQuiet(String s) {
    // clean up string
    s = s.replaceAll("[ \t\n\0\f\r,]", "");
    // remove parenthesis around numbers
    // if PTBTokenized, this next bit should be a no-op
    // in some contexts parentheses might indicate a negative number, but ignore that.
    if (s.startsWith("(") && s.endsWith(")")) {
      s = s.substring(1, s.length() - 1);
      if (DEBUG2)
        err.println("Deleted (): " + s);
    }
    s = s.toLowerCase();

    if (DEBUG2)
      err.println("Looking for ordinal words in |" + s + '|');
    if (Character.isDigit(s.charAt(0))) {
      Matcher matcher = numberPattern.matcher(s);
      matcher.find();
      // just parse number part, assuming last two letters are st/nd/rd
      return normalizedNumberStringQuiet(matcher.group(), 1.0);
    } else if (ordinalsToValues.containsKey(s)) {
      return Double.toString(ordinalsToValues.getCount(s));
    } else {
      String val = getOneSubstitutionMatch(s, ordinalsToValues.keySet());
      if (val != null)
        return Double.toString(ordinalsToValues.getCount(val));
      else
        return null;
    }
  }

  public static String normalizedPercentString(String s) {
    if (DEBUG2) {
      err.println("normalizedPercentString: " + s);
    }
    s = s.replaceAll("\\s", "");
    s = s.toLowerCase();
    if (s.contains("%") || s.contains("percent")) {
      s = s.replaceAll("percent|%", "");
    }
    String norm = normalizedNumberStringQuiet(s, 1.0);
    if (norm == null) {
      return null;
    }
    return '%' + norm;
  }

  private static <E extends CoreMap> List<E> processEntity(List<E> l,
      String entityType, String compModifier) {
    assert (quantifiable.contains(entityType));
    if (DEBUG) {
      System.err.println("Quantifiable.processEntity: " + l);
    }
    String s;
    if (entityType.equals("TIME")) {
      s = timeEntityToString(l);
    } else {
      s = singleEntityToString(l);
    }

    if (DEBUG)
      System.err.println("Quantifiable: working on " + s);
    String p = null;
    switch (entityType) {
    case "NUMBER": {
      p = "";
      if (compModifier != null) {
        p = compModifier;
      }
      String q = normalizedNumberString(s);
      if (q != null) {
        p = p.concat(q);
      } else {
        p = null;
      }
      break;
    }
    case "ORDINAL":
      p = normalizedOrdinalString(s);
      break;
    case "DURATION":
      p = normalizedDurationString(s);
      break;
    case "MONEY": {
      p = "";
      if (compModifier != null) {
        p = compModifier;
      }
      String q = normalizedMoneyString(s);
      if (q != null) {
        p = p.concat(q);
      } else {
        p = null;
      }
      break;
    }
    case "DATE":
      p = normalizedDateString(s);
      break;
    case "TIME": {
      p = "";
      if (compModifier != null && !compModifier.matches("am|pm")) {
        p = compModifier;
      }
      String q = normalizedTimeString(s, compModifier != null ? compModifier : "");
      if (q != null && q.length() == 1 && !q.equals("D")) {
        p = p.concat(q);
      } else {
        p = q;
      }
      break;
    }
    case "PERCENT": {
      p = "";
      if (compModifier != null) {
        p = compModifier;
      }
      String q = normalizedPercentString(s);
      if (q != null) {
        p = p.concat(q);
      } else {
        p = null;
      }
      break;
    }
    }
    if (DEBUG) {
      err.println("Quantifiable: Processed '" + s + "' as '" + p + '\'');
    }

    int i = 0;
    for (E wi : l) {
      if (p != null) {
        if (DEBUG) {
          System.err.println("#4: Changing normalized NER from "
              + wi.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class) + " to " + p + " at index " + i);
        }
        wi.set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, p);
      }
      //currently we also write this into the answers;
      //wi.setAnswer(wi.get(CoreAnnotations.AnswerAnnotation.class)+"("+p+")");
      i++;
    }
    return l;
  }

  /**
   * @param l
   *          The list of tokens in a time entity
   * @return the word in the time word list that should be normalized
   */
  private static <E extends CoreMap> String timeEntityToString(List<E> l) {
    String entityType = l.get(0).get(CoreAnnotations.AnswerAnnotation.class);
    int size = l.size();
    for (E w : l) {
      assert (w.get(CoreAnnotations.AnswerAnnotation.class) == null ||
          w.get(CoreAnnotations.AnswerAnnotation.class).equals(entityType));
      Matcher m = timePattern.matcher(w.get(CoreAnnotations.TextAnnotation.class));
      if (m.matches())
        return w.get(CoreAnnotations.TextAnnotation.class);
    }
    if (DEBUG) {
      System.err.println("default: " + l.get(size - 1).get(CoreAnnotations.TextAnnotation.class));
    }
    return l.get(size - 1).get(CoreAnnotations.TextAnnotation.class);
  }

  /**
   * Takes the output of an {@link AbstractSequenceClassifier} and marks up
   * each document by normalizing quantities. Each {@link CoreLabel} in any
   * of the documents which is normalizable will receive a "normalizedQuantity"
   * attribute.
   *
   * @param l
   *          a {@link List} of {@link List}s of {@link CoreLabel}s
   * @return The list with normalized entity fields filled in
   */
  public static List<List<CoreLabel>> normalizeClassifierOutput(List<List<CoreLabel>> l) {
    for (List<CoreLabel> doc : l) {
      addNormalizedQuantitiesToEntities(doc);
    }
    return l;
  }

  private static String earlyOneWord = "early";
  private static String earlyTwoWords = "(?:dawn|eve|beginning) of";
  private static String earlyThreeWords = "early in the";
  private static String lateOneWord = "late";
  private static String lateTwoWords = "late at|end of";
  private static String lateThreeWords = "end of the";
  private static String middleTwoWords = "(?:middle|midst) of";
  private static String middleThreeWords = "(?:middle|midst) of the";

  private static String amOneWord = "[Aa]\\.?[Mm]\\.?";
  private static String pmOneWord = "[Pp]\\.?[Mm]\\.?";
  private static String amThreeWords = "in the morning";
  private static String pmTwoWords = "at night";
  private static String pmThreeWords = "in the (?:afternoon|evening)";

  /**
   * Takes the strings of the three previous words to a quantity and detects a
   * quantity modifier like "less than", "more than", etc.
   * Any of these words may be <code>null</code> or an empty String.
   */
  private static <E extends CoreMap> String detectTimeOfDayModifier(List<E> list, int beforeIndex, int afterIndex) {
    String prev = (beforeIndex >= 0) ? list.get(beforeIndex).get(CoreAnnotations.TextAnnotation.class).toLowerCase()
        : "";
    String prev2 = (beforeIndex - 1 >= 0)
        ? list.get(beforeIndex - 1).get(CoreAnnotations.TextAnnotation.class).toLowerCase() : "";
    String prev3 = (beforeIndex - 2 >= 0)
        ? list.get(beforeIndex - 2).get(CoreAnnotations.TextAnnotation.class).toLowerCase() : "";
    int sz = list.size();
    String next = (afterIndex < sz) ? list.get(afterIndex).get(CoreAnnotations.TextAnnotation.class).toLowerCase() : "";
    String next2 = (afterIndex + 1 < sz)
        ? list.get(afterIndex + 1).get(CoreAnnotations.TextAnnotation.class).toLowerCase() : "";
    String next3 = (afterIndex + 2 < sz)
        ? list.get(afterIndex + 2).get(CoreAnnotations.TextAnnotation.class).toLowerCase() : "";

    String longPrev = prev3 + ' ' + prev2 + ' ' + prev;
    if (longPrev.matches(earlyThreeWords)) {
      return "E";
    } else if (longPrev.matches(lateThreeWords)) {
      return "L";
    } else if (longPrev.matches(middleThreeWords)) {
      return "M";
    }

    longPrev = prev2 + ' ' + prev;
    if (longPrev.matches(earlyTwoWords)) {
      return "E";
    } else if (longPrev.matches(lateTwoWords)) {
      return "L";
    } else if (longPrev.matches(middleTwoWords)) {
      return "M";
    }

    if (prev.matches(earlyOneWord) || prev2.matches(earlyOneWord)) {
      return "E";
    } else if (prev.matches(lateOneWord) || prev2.matches(lateOneWord)) {
      return "L";
    }

    String longNext = next3 + ' ' + next2 + ' ' + next;
    if (longNext.matches(pmThreeWords)) {
      return "pm";
    }
    if (longNext.matches(amThreeWords)) {
      return "am";
    }

    longNext = next2 + ' ' + next;
    if (longNext.matches(pmTwoWords)) {
      return "pm";
    }

    if (next.matches(amOneWord) || next2.matches("morning") || next3.matches("morning")) {
      return "am";
    }
    if (next.matches(pmOneWord) || next2.matches("afternoon") || next3.matches("afternoon")
        || next2.matches("night") || next3.matches("night")
        || next2.matches("evening") || next3.matches("evening")) {
      return "pm";
    }

    return "";
  }

  /**
   * Identifies contiguous MONEY, TIME, DATE, or PERCENT entities
   * and tags each of their constituents with a "normalizedQuantity"
   * label which contains the appropriate normalized string corresponding to
   * the full quantity. Quantities are not concatenated
   *
   * @param l
   *          A list of {@link CoreMap}s representing a single
   *          document. Note: the Labels are updated in place.
   */
  public static <E extends CoreMap> void addNormalizedQuantitiesToEntities(List<E> l) {
    addNormalizedQuantitiesToEntities(l, false, false);
  }

  public static <E extends CoreMap> void addNormalizedQuantitiesToEntities(List<E> l, boolean concatenate) {
    addNormalizedQuantitiesToEntities(l, concatenate, false);
  }

  private static boolean checkStrings(String s1, String s2) {
    if (s1 == null || s2 == null) {
      return s1 == s2;
    } else {
      return s1.equals(s2);
    }
  }

  private static boolean checkNumbers(Number n1, Number n2) {
    if (n1 == null || n2 == null) {
      return n1 == n2;
    } else {
      return n1.equals(n2);
    }
  }

  public static <E extends CoreMap> boolean isCompatible(String tag, E prev, E cur) {
    if ("NUMBER".equals(tag) || "ORDINAL".equals(tag)) {
      // Get NumericCompositeValueAnnotation and say two entities are incompatible if they are different
      Number n1 = cur.get(CoreAnnotations.NumericCompositeValueAnnotation.class);
      Number n2 = prev.get(CoreAnnotations.NumericCompositeValueAnnotation.class);
      boolean compatible = checkNumbers(n1, n2);
      if (!compatible)
        return compatible;
    }

    if ("TIME".equals(tag) || "SET".equals(tag) || "DATE".equals(tag) || "DURATION".equals(tag)) {
      // Check timex...
      Timex timex1 = cur.get(TimeAnnotations.TimexAnnotation.class);
      Timex timex2 = prev.get(TimeAnnotations.TimexAnnotation.class);
      String tid1 = (timex1 != null) ? timex1.tid() : null;
      String tid2 = (timex2 != null) ? timex2.tid() : null;
      boolean compatible = checkStrings(tid1, tid2);
      if (!compatible)
        return compatible;
    }

    return true;
  }

  /**
   * Identifies contiguous MONEY, TIME, DATE, or PERCENT entities
   * and tags each of their constituents with a "normalizedQuantity"
   * label which contains the appropriate normalized string corresponding to
   * the full quantity.
   *
   * @param list
   *          A list of {@link CoreMap}s representing a single
   *          document. Note: the Labels are updated in place.
   * @param concatenate
   *          true if quantities should be concatenated into one label, false
   *          otherwise
   */
  public static <E extends CoreMap> void addNormalizedQuantitiesToEntities(List<E> list, boolean concatenate,
      boolean usesSUTime) {
    List<E> toRemove = new ArrayList<>(); // list for storing those objects we're going to remove at the end (e.g., if concatenate, we replace 3 November with 3_November, have to remove one of the originals)

    // Goes through tokens and tries to fix up NER annotations
    fixupNerBeforeNormalization(list);

    // Now that NER tags has been fixed up, we do another pass to add the normalization
    String prevNerTag = BACKGROUND_SYMBOL;
    String timeModifier = "";
    ArrayList<E> collector = new ArrayList<>();
    for (int i = 0, sz = list.size(); i <= sz; i++) {
      E wi = null;
      String currNerTag = null;
      if (i < list.size()) {
        wi = list.get(i);
        if (DEBUG) {
          System.err.println("addNormalizedQuantitiesToEntities: wi is " + wi + "; collector is " + collector);
        }

        currNerTag = wi.get(CoreAnnotations.NamedEntityTagAnnotation.class);
        if ("TIME".equals(currNerTag)) {
          if (timeModifier.equals("")) {
            timeModifier = detectTimeOfDayModifier(list, i - 1, i + 1);
          }
        }
      }

      E wprev = (i > 0) ? list.get(i - 1) : null;
      // if the current wi is a non-continuation and the last one was a
      // quantity, we close and process the last segment.
      if ((currNerTag == null || !currNerTag.equals(prevNerTag) || !isCompatible(prevNerTag, wprev, wi))
          && quantifiable.contains(prevNerTag)) {
        // special handling of TIME
        switch (prevNerTag) {
        case "TIME":
          processEntity(collector, prevNerTag, timeModifier);
          break;
        case "DATE":
          processEntity(collector, prevNerTag, null);
          //now repair this date if it's more than one word
          //doesn't really matter which one we keep ideally we should be doing lemma/etc matching anyway
          //but we vaguely try to deal with this by choosing the NNP or the CD
          if (concatenate)
            concatenateNumericString(collector, toRemove);
          break;
        default:
          processEntity(collector, prevNerTag, null);
          if (concatenate) {
            concatenateNumericString(collector, toRemove);
          }
          break;
        }

        collector = new ArrayList<>();
        timeModifier = "";
      }

      // if the current wi is a quantity, we add it to the collector.
      // if its the first word in a quantity, we record index before it
      if (quantifiable.contains(currNerTag)) {
        collector.add(wi);
      }
      prevNerTag = currNerTag;
    }
    if (concatenate) {
      list.removeAll(toRemove);
    }
  }

  public static <E extends CoreMap> void fixupNerBeforeNormalization(List<E> list) {
    // Goes through tokens and tries to fix up NER annotations
    String prevNerTag = BACKGROUND_SYMBOL;
    String prevNumericType = null;
    Timex prevTimex = null;
    for (int i = 0, sz = list.size(); i < sz; i++) {
      E wi = list.get(i);

      String curWord = (wi.get(CoreAnnotations.TextAnnotation.class) != null
          ? wi.get(CoreAnnotations.TextAnnotation.class) : "");
      String currNerTag = wi.get(CoreAnnotations.NamedEntityTagAnnotation.class);

      if (DEBUG) {
        System.err.println("fixupNerBeforeNormalization: wi is " + wi);
      }
      // repairs commas in between dates...  String constant first in equals() in case key has null value....
      if ((i + 1) < sz && ",".equals(wi.get(CoreAnnotations.TextAnnotation.class)) && "DATE".equals(prevNerTag)) {
        if (prevTimex == null && prevNumericType == null) {
          E nextToken = list.get(i + 1);
          String nextNER = nextToken.get(CoreAnnotations.NamedEntityTagAnnotation.class);
          if (nextNER != null && nextNER.equals("DATE")) {
            wi.set(CoreAnnotations.NamedEntityTagAnnotation.class, "DATE");
          }
        }
      }

      //repairs mistagged multipliers after a numeric quantity
      if (!curWord.equals("") && (moneyMultipliers.containsKey(curWord) ||
          (getOneSubstitutionMatch(curWord, moneyMultipliers.keySet()) != null)) &&
          prevNerTag != null && (prevNerTag.equals("MONEY") || prevNerTag.equals("NUMBER"))) {
        wi.set(CoreAnnotations.NamedEntityTagAnnotation.class, prevNerTag);
      }

      //repairs four digit ranges (2002-2004) that have not been tagged as years - maybe bad? (empirically useful)
      if (curWord.contains("-")) {
        String[] sides = curWord.split("-");
        if (sides.length == 2) {
          try {
            int first = Integer.parseInt(sides[0]);
            int second = Integer.parseInt(sides[1]);
            //they're both integers, see if they're both between 1000-3000 (likely years)
            if (1000 <= first && first <= 3000 && 1000 <= second && second <= 3000) {
              wi.set(CoreAnnotations.NamedEntityTagAnnotation.class, "DATE");
              String dateStr = new ISODateInstance(new ISODateInstance(sides[0]), new ISODateInstance(sides[1]))
                  .getDateString();
              if (DEBUG) {
                System.err.println("#5: Changing normalized NER from " +
                    wi.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class) + " to " + dateStr + " at index "
                    + i);
              }
              wi.set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, dateStr);
              continue;
            }
          } catch (Exception e) {
            // they weren't numbers.
          }
        }
      }

      // Marks time units as DURATION if they are preceded by a NUMBER tag.  e.g. "two years" or "5 minutes"
      if (timeUnitWords.containsKey(curWord) &&
          (currNerTag == null || !"DURATION".equals(currNerTag)) &&
          ("NUMBER".equals(prevNerTag))) {
        wi.set(CoreAnnotations.NamedEntityTagAnnotation.class, "DURATION");
        for (int j = i - 1; j >= 0; j--) {
          E prev = list.get(j);
          if ("NUMBER".equals(prev.get(CoreAnnotations.NamedEntityTagAnnotation.class))) {
            prev.set(CoreAnnotations.NamedEntityTagAnnotation.class, "DURATION");
          }
        }
      }

      prevNerTag = currNerTag;
    }
  }

  /**
   * Runs a deterministic named entity classifier which is good at recognizing
   * numbers and money and date expressions not recognized by our statistical
   * NER. It then changes any BACKGROUND_SYMBOL's from the list to
   * the value tagged by this deterministic NER.
   * It then adds normalized values for quantifiable entities.
   *
   * @param l
   *          A document to label
   * @return The list with results of 'specialized' (rule-governed) NER filled
   *         in
   */
  public static <E extends CoreLabel> List<E> applySpecializedNER(List<E> l) {
    int sz = l.size();
    // copy l
    List<CoreLabel> copyL = new ArrayList<>(sz);
    for (int i = 0; i < sz; i++) {
      if (DEBUG2) {
        if (i == 1) {
          String tag = l.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
          if (tag == null || tag.equals("")) {
            err.println("Quantifiable: error! tag is " + tag);
          }
        }
      }
      copyL.add(new CoreLabel(l.get(i)));
    }
    // run NumberSequenceClassifier
    AbstractSequenceClassifier<CoreLabel> nsc = new NumberSequenceClassifier();
    copyL = nsc.classify(copyL);
    // update entity only if it was not O
    for (int i = 0; i < sz; i++) {
      E before = l.get(i);
      CoreLabel nscAnswer = copyL.get(i);
      if ((before.get(CoreAnnotations.NamedEntityTagAnnotation.class) == null
          || before.get(CoreAnnotations.NamedEntityTagAnnotation.class).equals(BACKGROUND_SYMBOL)) &&
          (nscAnswer.get(CoreAnnotations.AnswerAnnotation.class) != null
              && !nscAnswer.get(CoreAnnotations.AnswerAnnotation.class).equals(BACKGROUND_SYMBOL))) {
        before.set(CoreAnnotations.NamedEntityTagAnnotation.class,
            nscAnswer.get(CoreAnnotations.AnswerAnnotation.class));
      }
    }

    addNormalizedQuantitiesToEntities(l);
    return l;
  } // end applySpecializedNER

}