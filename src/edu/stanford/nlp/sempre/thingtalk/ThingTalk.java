package edu.stanford.nlp.sempre.thingtalk;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.sempre.*;

/**
 * Functions for supporting thingtalk
 *
 * @author Rakesh Ramesh
 */
public final class ThingTalk {

  public static NumberValue measureValueCast(StringValue unit, NumberValue number) {
    NumberValue tempVal = new NumberValue(number.value, unit.value);
    return tempVal;
  }

  public static EmailAddressValue emailAddressCast(StringValue string) {
    return new EmailAddressValue(string.value);
  }

  public static PhoneNumberValue phoneNumberCast(StringValue string) {
    return new PhoneNumberValue(string.value);
  }

  //******************************************************************************************************************
  // Constructing the parameter value structure
  //******************************************************************************************************************
  public static ParamValue paramForm(StringValue tt_type, ParamNameValue tt_arg, StringValue operator, Value value) {
    ParamValue paramVal = new ParamValue(tt_arg, tt_type.value, operator.value, value);
    return paramVal;
  }

  private static String typeFromNumber(NumberValue value) {
    if (value.unit == null || value.unit.equals(NumberValue.unitless))
      return "Number";
    else
      return "Measure";
  }

  private static String typeFromDate(DateValue value) {
    if (value.day == -1 && value.month == -1 && value.year == -1)
      return "Time";
    else
      return "Date";
  }

  static String typeFromValue(Value value) {
    if (value instanceof NumberValue)
      return typeFromNumber((NumberValue) value);
    else if (value instanceof StringValue)
      return "String";
    else if (value instanceof TimeValue)
      return "Time";
    else if (value instanceof DateValue)
      return typeFromDate((DateValue) value);
    else if (value instanceof BooleanValue)
      return "Bool";
    else if (value instanceof LocationValue)
      return "Location";
    else if (value instanceof EmailAddressValue)
      return "EmailAddress";
    else if (value instanceof PhoneNumberValue)
      return "PhoneNumber";
    else
      throw new RuntimeException("Unexpected value " + value);
  }

  public static ParamValue paramForm(ParamNameValue tt_arg, StringValue operator, Value value) {
    return new ParamValue(tt_arg, typeFromValue(value), operator.value, value);
  }

  public static ParametricValue addParam(ParametricValue oldInvocation, ParamNameValue paramName, StringValue operator,
      Value value) {
    ParametricValue newInvocation = oldInvocation.clone();
    newInvocation.add(paramForm(paramName, operator, value));
    return newInvocation;
  }

  //******************************************************************************************************************
  // Constructing the trigger value structure
  //******************************************************************************************************************
  public static TriggerValue trigParam(ChannelNameValue triggerName) {
    TriggerValue triggerVal = new TriggerValue(triggerName);
    return triggerVal;
  }

  //******************************************************************************************************************
  // Constructing the query value structure
  //******************************************************************************************************************
  public static QueryValue queryParam(ChannelNameValue queryName) {
    QueryValue queryVal = new QueryValue(queryName);
    return queryVal;
  }

  //******************************************************************************************************************
  // Constructing the action value structure
  //******************************************************************************************************************
  public static ActionValue actParam(ChannelNameValue actionName) {
    ActionValue actionVal = new ActionValue(actionName);
    return actionVal;
  }

  //******************************************************************************************************************
  // Constructing the command value structure
  //******************************************************************************************************************
  public static CommandValue cmdForm(StringValue type, Value val) {
    CommandValue cmdVal = new CommandValue(type.value, val);
    return cmdVal;
  }

  //******************************************************************************************************************
  // Answers
  //******************************************************************************************************************
  public static ParamValue ansForm(StringValue type, Value val) {
    return new ParamValue(new ParamNameValue("answer", type.value), type.value, "is", val);
  }

  public static ParamValue ansForm(Value val) {
    // we don't need to give a type to ParamNameValue because we're not letting this
    // paramvalue through FilterInvalidArgFn
    return new ParamValue(new ParamNameValue("answer", null), typeFromValue(val), "is", val);
  }

  //******************************************************************************************************************
  // Specials handler -- Fragile!! Handle with care
  //******************************************************************************************************************
  public static StringValue special(NameValue spl) {
    Map<String, Object> json = new HashMap<>();
    json.put("special", spl.toJson());
    return (new StringValue(Json.writeValueAsStringHard(json)));
  }

  //******************************************************************************************************************
  // Constructing the rule value structure
  //******************************************************************************************************************
  public static RuleValue timeRule(DateValue time, Value action) {
    ParamNameValue timeName = new ParamNameValue("time", "String");
    ParamValue timeParam = new ParamValue(timeName, "Time", "is", time);
    TriggerValue timeTrigger = new TriggerValue(
        new ChannelNameValue("builtin", "at", Collections.singletonList("time"), Collections.singletonList("time"),
            Collections.singletonList("String")),
        Collections.singletonList(timeParam));

    if (action instanceof QueryValue)
      return new RuleValue(timeTrigger, (QueryValue) action, null);
    else if (action instanceof ActionValue)
      return new RuleValue(timeTrigger, null, (ActionValue) action);
    else
      throw new RuntimeException();
  }

  public static RuleValue timeSpanRule(NumberValue time, Value action) {
    ParamNameValue timeName = new ParamNameValue("interval", "Measure(ms)");
    ParamValue timeParam = new ParamValue(timeName, "Measure", "is", time);
    TriggerValue timeTrigger = new TriggerValue(new ChannelNameValue("builtin", "timer",
        Collections.singletonList("interval"), Collections.singletonList("interval"),
        Collections.singletonList("Measure(ms)")),
        Collections.singletonList(timeParam));

    if (action instanceof QueryValue)
      return new RuleValue(timeTrigger, (QueryValue) action, null);
    else if (action instanceof ActionValue)
      return new RuleValue(timeTrigger, null, (ActionValue) action);
    else
      throw new RuntimeException();
  }

  public static RuleValue ifttt(TriggerValue trigger, ActionValue action) {
    RuleValue ruleVal = new RuleValue(trigger, null, action);
    return ruleVal;
  }

  public static RuleValue ifttt(TriggerValue trigger, QueryValue action) {
    RuleValue ruleVal = new RuleValue(trigger, action, null);
    return ruleVal;
  }

  //******************************************************************************************************************
  // Constructing the rule value structure
  //******************************************************************************************************************
  public static Value jsonOut(Value val) {
    Map<String, Object> json = new HashMap<>();
    String label = "";
    if (val instanceof RuleValue)
      label = "rule";
    else if (val instanceof ActionValue)
      label = "action";
    else if (val instanceof CommandValue)
      label = "command";
    else if (val instanceof TriggerValue)
      label = "trigger";
    else if (val instanceof QueryValue)
      label = "query";
    else if (val instanceof ParamValue)
      label = "answer";
    else
      label = "error"; // FIXME: Error flow
    json.put(label, val.toJson());
    return (new StringValue(Json.writeValueAsStringHard(json)));
  }
}
