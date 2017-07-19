package edu.stanford.nlp.sempre.thingtalk.seq2seq;

import java.util.*;

import com.google.common.collect.Sets;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.thingtalk.*;
import fig.basic.Pair;

public class Seq2SeqTokenizer {
  public static class Value {
    public final String type;
    public final Object value;

    Value(String type, Object value) {
      this.type = type;
      this.value = value;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((type == null) ? 0 : type.hashCode());
      result = prime * result + ((value == null) ? 0 : value.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      Value other = (Value) obj;
      if (type == null) {
        if (other.type != null)
          return false;
      } else if (!type.equals(other.type))
        return false;
      if (value == null) {
        if (other.value != null)
          return false;
      } else if (!value.equals(other.value))
        return false;
      return true;
    }

    @Override
    public String toString() {
      return "(" + type + ": " + value + ")";
    }
  }

  public static class Result {
    public final List<String> tokens = new ArrayList<>();
    public final Map<Value, List<Integer>> entities = new HashMap<>();
  }

  private final boolean applyHeuristics;
  private final LocationLexicon locationLexicon;
  private final EntityLexicon entityLexicon;

  public Seq2SeqTokenizer(String languageTag, boolean applyHeuristics) {
    this.applyHeuristics = applyHeuristics;

    locationLexicon = LocationLexicon.getForLanguage(languageTag);
    entityLexicon = EntityLexicon.getForLanguage(languageTag);
  }

  public Result process(Example ex) {
    Map<String, Integer> nextInt = new HashMap<>();

    StringBuilder fullEntity = new StringBuilder();
    LanguageInfo utteranceInfo = ex.languageInfo;
    Result result = new Result();

    if (applyHeuristics) {
      // adjust the NER tag where the model fails

      for (int i = 0; i < utteranceInfo.tokens.size(); i++) {
        String token, tag;

        tag = utteranceInfo.nerTags.get(i);
        if ("O".equals(tag))
          tag = null;
        token = utteranceInfo.tokens.get(i);

        if (token.equals("san") && i < utteranceInfo.tokens.size() - 2
            && utteranceInfo.tokens.get(i + 1).equals("jose")
            && utteranceInfo.tokens.get(i + 2).startsWith("earthquake")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
          utteranceInfo.nerTags.set(i + 2, tag);
        }
        if (token.equals("nevada") && i < utteranceInfo.tokens.size() - 2
            && utteranceInfo.tokens.get(i + 1).equals("wolf")
            && utteranceInfo.tokens.get(i + 2).startsWith("pack")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
          utteranceInfo.nerTags.set(i + 2, tag);
        }
        if (token.equals("miami") && i < utteranceInfo.tokens.size() - 1
            && utteranceInfo.tokens.get(i + 1).equals("heat")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
        }

        // unsurprisingly, "wolf pack", "red hat" and "california bears" are fairly generic
        // words on their own, and corenlp does not tag them
        if (token.equals("wolf") && i < utteranceInfo.tokens.size() - 1
            && utteranceInfo.tokens.get(i + 1).startsWith("pack")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
        }
        if (token.equals("red") && i < utteranceInfo.tokens.size() - 1
            && utteranceInfo.tokens.get(i + 1).equals("hat")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
        }
        if (token.equals("california") && i < utteranceInfo.tokens.size() - 1
            && utteranceInfo.tokens.get(i + 1).equals("bears")) {
          tag = "ORGANIZATION";
          utteranceInfo.nerTags.set(i + 1, tag);
        }

        switch (token) {
        //case "google":
        case "warriors":
        case "stanford":
        case "apple":
        case "giants":
        case "cavaliers":
        case "sta":
        case "stan":
        case "msft":
        case "goog":
        case "cubs":
        case "aapl":

          // in our dataset, Barcelona refers to the team
        case "barcellona":
        case "barcelona":
          tag = "ORGANIZATION";
          break;

        case "italian":
        case "french":
        case "spanish":
        case "chinese":
        case "english":
        case "german":
          tag = "LANGUAGE";
          break;
        }

        if (tag != null && !utteranceInfo.nerTags.get(i).equals("QUOTED_STRING"))
          utteranceInfo.nerTags.set(i, tag);
      }
    }

    for (int i = 0; i < utteranceInfo.tokens.size(); i++) {
      String token, tag, current;

      tag = utteranceInfo.nerTags.get(i);
      token = utteranceInfo.tokens.get(i);

      if (!"O".equals(tag)) {
        if (fullEntity.length() != 0)
          fullEntity.append(" ");
        fullEntity.append(token);
        if (i < utteranceInfo.tokens.size() - 1 &&
            utteranceInfo.nerTags.get(i + 1).equals(tag) &&
            Objects.equals(utteranceInfo.nerValues.get(i), utteranceInfo.nerValues.get(i + 1)))
          continue;

        Pair<String, Object> value = nerValueToThingTalkValue(ex, tag, utteranceInfo.nerValues.get(i),
            fullEntity.toString());
        fullEntity.setLength(0);
        // ignore tt:device entities
        if (value != null && "GENERIC_ENTITY_tt:device".equals(value.getFirst()))
          value = null;
        if (value != null) {
          tag = value.getFirst();
          int id = nextInt.compute(tag, (oldKey, oldValue) -> {
            if (oldValue == null)
              oldValue = -1;
            return oldValue + 1;
          });
          result.entities.computeIfAbsent(new Value(tag, value.getSecond()), (key) -> new LinkedList<>()).add(id);
          current = tag + "_" + id;
        } else {
          current = token;
        }
      } else {
        current = token;
      }
      result.tokens.add(current);
    }

    return result;
  }

  private LocationValue findLocation(String entity) {
    // earth is not a location, and neiter is europe
    switch (entity) {
    case "earth":
    case "europe":
    case "uk":
    case "us":
    case "u.s.":
    case "usa":
    case "united states":
    case "america":

      // how sabrina could be a location is beyond me
    case "sabrina":
      return null;
    }

    Collection<LocationLexicon.Entry<LocationValue>> entries = locationLexicon.lookup(entity);
    if (entries.isEmpty())
      return null;

    LocationLexicon.Entry<LocationValue> first = entries.iterator().next();
    return (LocationValue) ((ValueFormula<?>) first.formula).value;
  }

  // refuse to return anything for yahoo, because otherwise every yahoo finance sentence
  // would have a very confusing two entities
  private static final Set<String> NOT_ENTITIES = Sets.newHashSet("wsj world news", "yahoo", "capital weather gang",
      "ac state");

  private Pair<String, Object> findEntity(Example ex, String entity, String hint) {
    // override the lexicon on this one
    if (applyHeuristics) {
      if (entity.equals("warriors"))
        return new Pair<>("GENERIC_ENTITY_sportradar:nba_team",
            new TypedStringValue("Entity(sportradar:nba_team)", "gsw", "Golden State Warriors"));
      if (entity.equals("cavaliers"))
        return new Pair<>("GENERIC_ENTITY_sportradar:nba_team",
            new TypedStringValue("Entity(sportradar:nba_team)", "cle", "Cleveland Cavaliers"));
      if (entity.equals("giants"))
        return new Pair<>("GENERIC_ENTITY_sportradar:mlb_team",
            new TypedStringValue("Entity(sportradar:mlb_team)", "sf", "San Francisco Giants"));
      if (entity.equals("cubs"))
        return new Pair<>("GENERIC_ENTITY_sportradar:mlb_team",
            new TypedStringValue("Entity(sportradar:mlb_team)", "chc", "Chicago Cubs"));
      if (entity.equals("wolf pack") || entity.equals("nevada wolf pack"))
        return new Pair<>("GENERIC_ENTITY_sportradar:ncaafb_team",
            new TypedStringValue("Entity(sportradar:ncaafb_team)", "nev", "Nevada Wolf Pack"));
      // Barcellona Pozzo di Grotto, obviously
      if (entity.equals("barcelona"))
        return new Pair<>("GENERIC_ENTITY_sportradar:eu_soccer_team",
            new TypedStringValue("Entity(sportradar:eu_soccer_team)", "bar", "FC Barcelona"));

      if (NOT_ENTITIES.contains(entity))
        return null;
    }

    String tokens[] = entity.split("\\s+");

    Set<EntityLexicon.Entry<TypedStringValue>> entitySet = new HashSet<>();

    for (String token : tokens)
      entitySet.addAll(entityLexicon.lookup(token));

    if (entitySet.isEmpty())
      return null;

    // (scare quotes) MACHINE LEARNING!
    int nfootball = 0;
    int nbasketball = 0;
    int nbaseball = 0;
    int nstock = 0;
    for (String token : ex.getTokens()) {
      switch (token) {
      case "football":
      case "ncaafb":
      case "nfl":
        nfootball++;
        break;

      case "ncaambb":
      case "nba":
      case "basketball":
        nbasketball++;
        break;

      case "mlb":
      case "baseball":
        nbaseball++;
        break;

      case "stock":
      case "finance":
      case "quote":
      case "dividend":
        nstock++;
      }
    }
    if (entity.equals("california bears")) {
      if (nfootball > nbasketball)
        return new Pair<>("GENERIC_ENTITY_sportradar:ncaafb_team",
            new TypedStringValue("Entity(sportradar:ncaafb_team)", "cal", "California Bears"));
      else if (nfootball < nbasketball)
        return new Pair<>("GENERIC_ENTITY_sportradar:ncaambb_team",
            new TypedStringValue("Entity(sportradar:ncaambb_team)", "cal", "California Golden Bears"));
    }
    if (entity.equals("google")) {
      if (nstock > 0)
        return new Pair<>("GENERIC_ENTITY_tt:stock_id",
            new TypedStringValue("Entity(tt:stock_id)", "goog", "Alphabet Inc."));
    }

    List<Pair<Pair<String, Object>, Double>> weights = new ArrayList<>();
    for (EntityLexicon.Entry<TypedStringValue> entry : entitySet) {
      String nerTag = entry.nerTag;
      TypedStringValue value = entry.formula.value;
      String[] canonicalTokens = entry.rawPhrase.split("\\s+");

      if (hint != null && !nerTag.endsWith(hint))
        continue;

      double weight = 0;
      if (nerTag.endsWith("sportradar:mlb_team"))
        weight += 0.25 * nbaseball;
      else if (nerTag.endsWith("sportradar:nba_team") || nerTag.endsWith("sportradar:ncaambb_team"))
        weight += 0.25 * nbasketball;
      else if (nerTag.endsWith("sportradar:nfl_team") || nerTag.endsWith("sportradar:ncaafb_team"))
        weight += 0.25 * nfootball;

      for (String canonicalToken : canonicalTokens) {
        boolean found = false;
        for (String token : tokens) {
          if (token.equals(canonicalToken)) {
            weight += 1;
            found = true;
          } else if (token.equals("cardinals") && canonicalToken.equals("cardinal")) {
            weight += 1;
            found = true;
          } else if (token.equals("la") && (canonicalToken.equals("los") || canonicalToken.equals("angeles"))) {
            weight += 0.5;
            found = true;
          }
        }
        if (!found)
          weight -= 0.125;
      }

      weights.add(new Pair<>(new Pair<>(nerTag, value), weight));
    }

    weights.sort((one, two) -> {
      double w1 = one.getSecond();
      double w2 = two.getSecond();

      if (w1 == w2)
        return 0;
      // sort highest weight first
      if (w1 < w2)
        return +1;
      else
        return -1;
    });

    double maxWeight = weights.get(0).getSecond();
    if (weights.size() > 1 && weights.get(1).getSecond() == maxWeight) {
      //System.out.println("Ambiguous entity " + entity + ", could be any of " + weights);
      return null;
    }

    return weights.get(0).getFirst();
  }

  private static TimeValue parseTimeValue(String nerValue) {
    DateValue date = DateValue.parseDateValue(nerValue);
    if (date == null)
      return null;
    return new TimeValue(date.hour, date.minute);
  }

  private Pair<String, Object> nerValueToThingTalkValue(Example ex, String nerType, String nerValue,
      String entity) {
    switch (nerType) {
    case "MONEY":
    case "PERCENT":
      try {
        if (nerValue == null)
          return null;
        if (nerValue.startsWith(">=") || nerValue.startsWith("<="))
          nerValue = nerValue.substring(3);
        else if (nerValue.startsWith(">") || nerValue.startsWith("<") || nerValue.startsWith("~"))
          nerValue = nerValue.substring(2);
        else
          nerValue = nerValue.substring(1);
        return new Pair<>("NUMBER", Double.valueOf(nerValue));
      } catch (NumberFormatException e) {
        return null;
      }

    case "NUMBER":
      if (nerValue == null)
        return null;
      try {
        if (nerValue.startsWith(">=") || nerValue.startsWith("<="))
          nerValue = nerValue.substring(2);
        else if (nerValue.startsWith(">") || nerValue.startsWith("<") || nerValue.startsWith("~"))
          nerValue = nerValue.substring(1);
        double v = Double.valueOf(nerValue);
        if (v == 1 || v == 0)
          return null;
        return new Pair<>(nerType, v);
      } catch (NumberFormatException e) {
        return null;
      }

    case "DATE": {
      DateValue date = DateValue.parseDateValue(nerValue);
      if (date == null)
        return null;
      return new Pair<>(nerType, date);
    }
    case "TIME":
      if (nerValue == null)
        return null;
      if (!nerValue.startsWith("T")) {
        // actually this is a date, not a time
        DateValue date = DateValue.parseDateValue(nerValue);
        if (date == null)
          return null;
        return new Pair<>("DATE", date);
      } else {
        TimeValue time = parseTimeValue(nerValue);
        if (time == null)
          return null;
        return new Pair<>(nerType, time);
      }

    case "USERNAME":
    case "HASHTAG":
    case "PHONE_NUMBER":
    case "EMAIL_ADDRESS":
    case "URL":
    case "QUOTED_STRING":
      return new Pair<>(nerType, nerValue);

    case "LOCATION":
      LocationValue loc = findLocation(entity);
      if (loc == null)
        return null;
      return new Pair<>(nerType, loc);

    case "ORGANIZATION":
      return findEntity(ex, entity, null);

    case "LANGUAGE":
      return findEntity(ex, entity, "tt:iso_lang_code");

    case "DURATION":
      if (nerValue != null) {
        NumberValue v = NumberValue.parseDurationValue(nerValue);
        if (v.value == 1 && v.unit.equals("day"))
          return null;
        return new Pair<>(nerType, v);
      } else {
        return null;
      }
    }

    return null;
  }

}
