package edu.stanford.nlp.sempre.thingtalk.seq2seq;

import java.io.IOException;
import java.io.Writer;
import java.util.*;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.thingtalk.*;
import fig.basic.Pair;

class Seq2SeqConverter {
  private static class Value {
    final String type;
    final Object value;

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

  private final LocationLexicon locationLexicon;
  private final EntityLexicon entityLexicon;

  private Example ex;
  private final List<String> inputTokens = new ArrayList<>();
  private final List<String> outputTokens = new ArrayList<>();
  private final Map<Value, List<Integer>> entities = new HashMap<>();

  public Seq2SeqConverter(String languageTag) {
    locationLexicon = LocationLexicon.getForLanguage(languageTag);
    entityLexicon = EntityLexicon.getForLanguage(languageTag);
  }

  public Pair<List<String>, List<String>> run(Example ex) {
    this.ex = ex;
    inputTokens.clear();
    outputTokens.clear();
    entities.clear();

    writeUtterance();
    writeOutput();

    return new Pair<>(Collections.unmodifiableList(inputTokens), Collections.unmodifiableList(outputTokens));
  }

  public static void writeSequences(Pair<List<String>, List<String>> sequences, Writer writer) throws IOException {
    List<String> input = sequences.getFirst();
    List<String> output = sequences.getSecond();

    boolean first = true;
    for (String t : input) {
      if (!first)
        writer.append(' ');
      first = false;
      writer.append(t);
    }
    writer.append('\t');
    first = true;
    for (String t : output) {
      if (!first)
        writer.append(' ');
      first = false;
      writer.append(t);
    }
    writer.append('\n');
  }

  private void writeUtterance() {
    Map<String, Integer> nextInt = new HashMap<>();

    StringBuilder fullEntity = new StringBuilder();
    LanguageInfo utteranceInfo = ex.languageInfo;

    // HACK HACK HACK
    // adjust the NER tag where the model fails
    // (eg for companies founded after 1993...)

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
      if (token.equals("la") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("lakers")) {
        tag = "ORGANIZATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("palo") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("alto")) {
        tag = "LOCATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("los") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("angeles")) {
        tag = "LOCATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("san") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("francisco")) {
        tag = "LOCATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("stanford") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("cardinals")) {
        tag = "ORGANIZATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("bayern") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("munchen")) {
        tag = "ORGANIZATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("chicago") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("cubs")) {
        tag = "ORGANIZATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }
      if (token.equals("toronto") && i < utteranceInfo.tokens.size() - 1
          && utteranceInfo.tokens.get(i + 1).equals("fc")) {
        tag = "ORGANIZATION";
        utteranceInfo.nerTags.set(i + 1, tag);
      }

      switch (token) {
      case "google":
      case "warriors":
      case "stanford":
      case "apple":
      case "giants":
      case "cavaliers":
      case "sta":
      case "stan":
      case "microsoft":
      case "juventus":
      case "msft":
      case "goog":
      case "cubs":
      case "aapl":

        // in our dataset, Barcellona refers to the team
      case "barcellona":
        tag = "ORGANIZATION";
        break;

      case "sunnyvale":
      case "paris":
        tag = "LOCATION";
        break;
      }

      if (tag != null)
        utteranceInfo.nerTags.set(i, tag);
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
        if (value != null) {
          tag = value.getFirst();
          int id = nextInt.compute(tag, (oldKey, oldValue) -> {
            if (oldValue == null)
              oldValue = -1;
            return oldValue + 1;
          });
          entities.computeIfAbsent(new Value(tag, value.getSecond()), (key) -> new LinkedList<>()).add(id);
          current = tag + "_" + id;
        } else {
          current = token;
        }
      } else {
        current = token;
      }
      inputTokens.add(current);
    }
  }

  private void writeOutput() {
    Map<?, ?> json = Json.readMapHard(((StringValue) ex.targetValue).value);

    if (json.containsKey("special"))
      writeSpecial((Map<?, ?>) json.get("special"));
    else if (json.containsKey("answer"))
      writeAnswer((Map<?, ?>) json.get("answer"));
    else if (json.containsKey("command"))
      writeCommand((Map<?, ?>) json.get("command"));
    else if (json.containsKey("rule"))
      writeRule((Map<?, ?>) json.get("rule"));
    else if (json.containsKey("trigger"))
      writeTopInvocation("trigger", (Map<?, ?>) json.get("trigger"));
    else if (json.containsKey("query"))
      writeTopInvocation("query", (Map<?, ?>) json.get("query"));
    else if (json.containsKey("action"))
      writeTopInvocation("action", (Map<?, ?>) json.get("action"));
  }

  private void writeSpecial(Map<?, ?> special) {
    outputTokens.add("special");
    String id = (String) special.get("id");
    outputTokens.add(id);
  }

  private void writeTopInvocation(String invocationType, Map<?, ?> map) {
    outputTokens.add(invocationType);
    writeInvocation(invocationType, map);
  }

  private void writeInvocation(String invocationType, Map<?, ?> invocation) {
    Map<?, ?> name = (Map<?, ?>) invocation.get("name");
    outputTokens.add(name.get("id").toString());

    List<?> arguments = (List<?>) invocation.get("args");
    for (Object o : arguments) {
      Map<?, ?> arg = (Map<?, ?>) o;
      Map<?, ?> argName = (Map<?, ?>) arg.get("name");
      outputTokens.add(argName.get("id").toString());
      outputTokens.add(arg.get("operator").toString());
      writeArgument(arg);
    }
  }

  private void writeRule(Map<?, ?> rule) {
    outputTokens.add("rule");
    if (rule.containsKey("trigger")) {
      //outputTokens.add("if");
      writeInvocation("trigger", (Map<?, ?>) rule.get("trigger"));
    }
    if (rule.containsKey("query")) {
      writeInvocation("query", (Map<?, ?>) rule.get("query"));
    }
    if (rule.containsKey("action")) {
      writeInvocation("action", (Map<?, ?>) rule.get("action"));
    }
  }

  private void writeCommand(Map<?, ?> command) {
    outputTokens.add("command");
    outputTokens.add((String) command.get("type"));

    Map<?, ?> value = (Map<?, ?>) command.get("value");
    if (value.containsKey("value"))
      outputTokens.add(value.get("value").toString());
    else if (value.containsKey("name"))
      outputTokens.add(value.get("name").toString());
    else
      outputTokens.add(value.get("id").toString());
  }

  private void writeAnswer(Map<?, ?> answer) {
    outputTokens.add("answer");
    writeArgument(answer);
  }

  private boolean writeValue(String type, Object value, boolean warn) {
    Value searchKey = new Value(type, value);
    List<Integer> ids = entities.getOrDefault(searchKey, Collections.emptyList());

    if (ids.isEmpty()) {
      if (warn) {
        // try again with QUOTED_STRING

        switch (type) {
        case "QUOTED_STRING":
          if (writeValue("USERNAME", value, false))
            return true;
          if (writeValue("HASHTAG", value, false))
            return true;
          break;
        case "USERNAME":
          if (writeValue("QUOTED_STRING", "@" + value, false))
            return true;
          break;
        case "HASHTAG":
          if (writeValue("QUOTED_STRING", "#" + value, false))
            return true;
          break;
        case "PHONE_NUMBER":
        case "EMAIL_ADDRESS":
        case "URL":
          if (writeValue("QUOTED_STRING", value, false))
            return true;
          break;
        }

        System.out
            .println(ex.id + ": cannot find value " + type + " " + value + ", have "
                + entities);
        // write the type and hope for the best
        outputTokens.add(type + "_0");
      }
      return false;
    }

    int id = ids.remove(0);

    outputTokens.add(type + "_" + id);
    return true;
  }

  private void writeValue(String type, Object value) {
    writeValue(type, value, true);
  }

  private void writeArgument(Map<?, ?> argument) {
    String type = (String) argument.get("type");
    Map<?, ?> value = (Map<?, ?>) argument.get("value");
    if (type.startsWith("Entity(")) {
      writeValue("GENERIC_ENTITY_" + type.substring("Entity(".length(), type.length() - 1),
          new TypedStringValue(type, value.get("value").toString()));
      return;
    }
    switch (type) {
    case "Location":
      String relativeTag = (String) value.get("relativeTag");
      if (relativeTag.equals("absolute")) {
        double latitude = ((Number) value.get("latitude")).doubleValue();
        double longitude = ((Number) value.get("longitude")).doubleValue();
        LocationValue loc = new LocationValue(latitude, longitude);
        writeValue("LOCATION", loc);
      } else {
        outputTokens.add(relativeTag);
      }
      break;

    case "Boolean":
    case "Bool":
    case "Enum":
      outputTokens.add(value.get("value").toString());
      break;

    case "VarRef":
      outputTokens.add(value.get("id").toString());
      break;

    case "String":
      writeValue("QUOTED_STRING", value.get("value").toString());
      break;

    case "Date":
      writeValue("DATE",
          new DateValue((Integer) value.get("year"), (Integer) value.get("month"), (Integer) value.get("day"),
              value.containsKey("hour") ? (int) (Integer) value.get("hour") : 0,
              value.containsKey("minute") ? (int) (Integer) value.get("minute") : 0,
              value.containsKey("second") ? (int) (Integer) value.get("second") : 0));
      break;

    case "Time":
      writeValue("TIME",
          new TimeValue((Integer) value.get("hour"), (Integer) value.get("minute")));
      break;

    case "Username":
      writeValue("USERNAME", value.get("value").toString());
      break;

    case "Hashtag":
      writeValue("HASHTAG", value.get("value").toString());
      break;

    case "Number":
      writeValue("NUMBER", ((Number) value.get("value")).doubleValue());
      break;

    case "Measure":
      if (ArgFilterHelpers.isTimeUnit(value.get("unit").toString())) {
        NumberValue numValue = new NumberValue(((Number) value.get("value")).doubleValue(),
            value.get("unit").toString());
        if (writeValue("DURATION", numValue, false))
          break;
        if (writeValue("SET", numValue, false))
          break;
      }
      writeValue("NUMBER", ((Number) value.get("value")).doubleValue());
      outputTokens.add(value.get("unit").toString());
      break;

    case "PhoneNumber":
      writeValue("PHONE_NUMBER", value.get("value").toString());
      break;

    case "EmailAddress":
      writeValue("EMAIL_ADDRESS", value.get("value").toString());
      break;

    case "URL":
      writeValue("URL", value.get("value").toString());
      break;

    default:
      throw new IllegalArgumentException("Invalid value type " + type);
    }
  }

  private LocationValue findLocation(String entity) {
    Collection<LocationLexicon.Entry<LocationValue>> entries = locationLexicon.lookup(entity);
    if (entries.isEmpty())
      return null;

    LocationLexicon.Entry<LocationValue> first = entries.iterator().next();
    return (LocationValue) ((ValueFormula<?>) first.formula).value;
  }

  private Pair<String, Object> findEntity(Example ex, String entity) {
    // override the lexicon on this one
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

    List<Pair<Pair<String, Object>, Double>> weights = new ArrayList<>();
    for (EntityLexicon.Entry<TypedStringValue> entry : entitySet) {
      String nerTag = entry.nerTag;
      TypedStringValue value = entry.formula.value;
      String[] canonicalTokens = entry.rawPhrase.split("\\s+");

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
      System.out.println("Ambiguous entity " + entity + ", could be any of " + weights);
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
        return new Pair<>(nerType, Double.valueOf(nerValue));
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
      return new Pair<>(nerType, findLocation(entity));

    case "ORGANIZATION":
    case "PERSON":
      return findEntity(ex, entity);

    case "SET":
    case "DURATION":
      if (nerValue != null)
        return new Pair<>(nerType, NumberValue.parseDurationValue(nerValue));
      else
        return null;
    }

    return null;
  }
}
