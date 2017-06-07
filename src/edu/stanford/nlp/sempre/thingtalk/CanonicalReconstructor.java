package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import com.google.common.base.Joiner;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaLexicon.Mode;
import fig.basic.LispTree;

public class CanonicalReconstructor {

  private static final Map<String, String> SPECIAL_TO_GRAMMAR = new HashMap<>();
  static {
    SPECIAL_TO_GRAMMAR.put("yes", "yes");
    SPECIAL_TO_GRAMMAR.put("no", "no");
    SPECIAL_TO_GRAMMAR.put("nevermind", "never_mind");
    SPECIAL_TO_GRAMMAR.put("makerule", "make_rule");
    SPECIAL_TO_GRAMMAR.put("failed", "failuretoparse");
  }

  private final Grammar grammar;
  private final List<String> buffer = new ArrayList<>();

  public CanonicalReconstructor(Grammar grammar_) {
    grammar = grammar_;
  }

  private static String lispTreeToString(LispTree tree) {
    if (tree.isLeaf())
      return tree.value;
    return Joiner.on(' ').join(tree.children.stream().map(c -> lispTreeToString(c)).iterator());
  }
  private String getToken(String token) {
    return lispTreeToString(grammar.getMacro("@" + token));
  }

  private String getId(Object obj) {
    if (obj instanceof String)
      return (String) obj;
    Map<?, ?> map = (Map<?, ?>) obj;
    if (map.containsKey("value"))
      return (String) map.get("value");
    return (String) map.get("id");
  }

  private String clean(String token) {
    return token.replace('_', ' ').replaceAll("([^A-Z])([A-Z])", "$1 $2").toLowerCase();
  }

  private String tokenize(String string) {
    return Joiner.on(' ').join(string.split("(\\s+|[,\\.\"\\'!\\?])"));
  }

  private void argToCanonical(Map<?, ?> arg, Map<String, String> scope) {
    if (arg.get("type").equals("VarRef")) {
      String id = getId(arg.get("value"));
      if (id.startsWith("tt:param.$event")) {
        switch (id) {
        case "tt:param.$event":
          buffer.add(getToken("event"));
          break;
        case "tt:param.$event.title":
          buffer.add(getToken("event_title"));
          break;
        case "tt:param.$event.body":
          buffer.add(getToken("event_body"));
          break;
        }
      } else {
        if (id.startsWith("tt:param."))
          id = id.substring("tt:param.".length());
        assert scope.containsKey(id);
        buffer.add(scope.get(id));
      }
      return;
    }

    Map<?, ?> value = (Map<?, ?>) arg.get("value");
    if (arg.get("type").equals("Enum")) {
      buffer.add(value.get("value").toString());
      return;
    }
    
    Type type = Type.fromString((String) arg.get("type"));

    if (type instanceof Type.Entity) {
      String entityType = ((Type.Entity) type).getType();
      switch (entityType) {
      case "tt:username":
      case "tt:contact_name":
        buffer.add("USERNAME");
        return;
      case "tt:hashtag":
        buffer.add("HASHTAG");
        return;
      case "tt:phone_number":
        buffer.add("PHONE_NUMBER");
        return;
      case "tt:email_address":
        buffer.add("EMAIL_ADDRESS");
        return;
      case "tt:url":
        buffer.add("URL");
        return;
      }
      
      if (value.containsKey("display"))
        buffer.add(tokenize((String) value.get("display")));
      else
        buffer.add((String) value.get("value"));
    } else if (type == Type.Location) {
      String relativeTag = (String) value.get("relativeTag");
      switch (relativeTag) {
      case "rel_current_location":
        buffer.add(getToken("here"));
        break;
      case "rel_home":
        buffer.add(getToken("at_home"));
        break;
      case "rel_work":
        buffer.add(getToken("at_work"));
        break;
      default:
        if (value.containsKey("display"))
          buffer.add(tokenize((String) value.get("display")));
        else
          buffer.add("LOCATION");
      }
    } else if (type == Type.Boolean || type instanceof Type.Enum) {
      buffer.add(getToken(value.get("value").toString()));
    } else if (type == Type.String) {
      buffer.add("QUOTED_STRING");
    } else if (type == Type.Date) {
      buffer.add("DATE");
    } else if (type == Type.Time) {
      buffer.add("TIME");
    } else if (type == Type.Number) {
      buffer.add("NUMBER");
    } else if (type instanceof Type.Measure) {
      buffer.add("NUMBER");
      buffer.add(value.get("unit").toString());
    } else {
      throw new RuntimeException("Invalid argument type " + type);
    }
  }

  public String reconstruct(Example ex) {
    buffer.clear();

    String targetJson = ((StringValue) ex.targetValue).value;
    Map<String, Object> json = Json.readMapHard(targetJson);

    if (json.containsKey("special")) {
      String special = getId(json.get("special"));
      if (special.startsWith("tt:root.special."))
        special = special.substring("tt:root.special.".length());
      if (special.equals("failed")) {
        buffer.add("failuretoparse");
      } else {
        assert SPECIAL_TO_GRAMMAR.containsKey(special);
        buffer.add(getToken(SPECIAL_TO_GRAMMAR.get(special)));
      }
    } else if (json.containsKey("command")) {
      Map<?, ?> command = (Map<?, ?>) json.get("command");
      String type = (String) command.get("type");
      if (!type.equals("help"))
        throw new RuntimeException("Invalid {\"command\"} type " + type);
      buffer.add(getToken("help"));
      String value = getId(command.get("value"));
      if (value.startsWith("tt:device."))
        value = value.substring("tt:device.".length());
      buffer.add(clean(value));
    } else if (json.containsKey("answer")) {
      argToCanonical((Map<?, ?>) json.get("answer"), null);
    } else if (json.containsKey("setup")) {
      buffer.add("tell");
      buffer.add("USERNAME");
      reconstructPrimRule((Map<?, ?>) json.get("setup"));
    } else {
      reconstructPrimRule(json);
    }

    return Joiner.on(' ').join(buffer);
  }

  private String getChannel(Map<?, ?> invocation) {
    String id = getId(invocation.get("name"));
    if (id.startsWith("tt:"))
      id = id.substring("tt:".length());
    return id;
  }

  private void invocationToCanonical(Map<?, ?> invocation, ThingpediaLexicon.Entry invocationMeta,
      Map<String, String> scope) {
    List<?> args = (List<?>) invocation.get("args");
    
    ChannelNameValue channel = invocationMeta.toValue();
    buffer.add(invocationMeta.getRawPhrase());

    if (invocation.containsKey("person")) {
      buffer.add(getToken("of"));
      buffer.add("USERNAME");
    }

    for (Object o : args) {
      Map<?, ?> arg = (Map<?, ?>) o;
      buffer.add(getToken("with"));

      String argname = getId(arg.get("name"));
      if (argname.startsWith("tt:param."))
        argname = argname.substring("tt:param.".length());
      String argcanonical = channel.getArgCanonical(argname);
      assert argcanonical != null;
      buffer.add(argcanonical);

      switch ((String) arg.get("operator")) {
      case "is":
        break;
      case "<":
        buffer.add(getToken("less_than"));
        break;
      case ">":
        buffer.add(getToken("greater_than"));
        break;
      case "contains":
        buffer.add(getToken("containing"));
        break;
      case "has":
        buffer.add(getToken("having"));
        break;
      default:
        throw new RuntimeException("Invalid operator " + arg.get("operator"));
      }

      argToCanonical(arg, scope);
    }

    for (String argname : channel.getArgNames())
      scope.put(argname, channel.getArgCanonical(argname));
  }

  private void reconstructPrimRule(Map<?, ?> map) {
    Map<?, ?> trigger = null, query = null, action = null;
    ThingpediaLexicon.Entry triggerMeta = null, queryMeta = null, actionMeta = null;

    boolean isRule = false;
    if (map.containsKey("rule")) {
      isRule = true;
      Map<?, ?> rule = (Map<?, ?>) map.get("rule");
      trigger = (Map<?, ?>) rule.get("trigger");
      query = (Map<?, ?>) rule.get("query");
      action = (Map<?, ?>) rule.get("action");
    } else {
      trigger = (Map<?, ?>) map.get("trigger");
      query = (Map<?, ?>) map.get("query");
      action = (Map<?, ?>) map.get("action");
    }
    
    ThingpediaLexicon lexicon = ThingpediaLexicon.getForLanguage(getToken("language_tag"));

    if (trigger != null)
      triggerMeta = lexicon.lookupChannelByName(getChannel(trigger), Mode.TRIGGER);
    if (query != null)
      queryMeta = lexicon.lookupChannelByName(getChannel(query), Mode.QUERY);
    if (action != null)
      actionMeta = lexicon.lookupChannelByName(getChannel(action), Mode.ACTION);
    
    Map<String, String> scope = new HashMap<>();
    if (isRule) {
      if (trigger != null) {
        List<?> args = (List<?>) trigger.get("args");
        if (getChannel(trigger).equals("builtin:timer") && args.size() == 1) {
          buffer.add(getToken("every"));
          argToCanonical((Map<?,?>)args.get(0), scope);
        } else if (getChannel(trigger).equals("builtin.at") && args.size() == 1) {
          buffer.add(getToken("every_day_at"));
          argToCanonical((Map<?,?>)args.get(0), scope);
        } else {
          buffer.add(getToken("if"));
          invocationToCanonical(trigger, triggerMeta, scope);
          buffer.add(getToken("then"));
        }
      }
      if (query != null)
        invocationToCanonical(query, queryMeta, scope);
      if (query != null && action != null)
        buffer.add(getToken("then"));
      if (action != null)
        invocationToCanonical(action, actionMeta, scope);
    } else if (action != null) {
      invocationToCanonical(action, actionMeta, scope);
    } else if (query != null) {
      invocationToCanonical(query, queryMeta, scope);
    } else if (trigger != null) {
      buffer.add(getToken("monitor_if"));
      invocationToCanonical(trigger, triggerMeta, scope);
    }
  }

}
