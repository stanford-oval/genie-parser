package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.sempre.*;

/**
 * Functions for supporting thingtalk
 *
 * @author Rakesh Ramesh
 */
public final class ThingTalk {

	public static NumberValue tempValueCast(StringValue unit, NumberValue number) {
		NumberValue tempVal = new NumberValue(number.value, unit.value);
		return tempVal;
	}

    //******************************************************************************************************************
    // Constructing the parameter value structure
    //******************************************************************************************************************
	public static ParamValue paramForm(StringValue tt_type, ParamNameValue tt_arg, StringValue operator, Value value) {
		ParamValue paramVal = new ParamValue(tt_arg, tt_type.value, operator.value, value);
        return paramVal;
    }

    //******************************************************************************************************************
    // Constructing the trigger value structure
    //******************************************************************************************************************
	public static TriggerValue trigParam(ChannelNameValue triggerName) {
        TriggerValue triggerVal = new TriggerValue(triggerName);
        return triggerVal;
    }
    public static TriggerValue trigParam(TriggerValue oldTrigger, ParamValue param) {
        // FIXME: Write a copy constructor
		TriggerValue newTrigger = (TriggerValue) oldTrigger.clone();
        newTrigger.add(param);
        return newTrigger;
    }

	//******************************************************************************************************************
	// Constructing the query value structure
	//******************************************************************************************************************
	public static QueryValue queryParam(ChannelNameValue queryName) {
		QueryValue queryVal = new QueryValue(queryName);
		return queryVal;
	}

	public static QueryValue queryParam(QueryValue oldQuery, ParamValue param) {
		// FIXME: Write a copy constructor
		QueryValue newQuery = (QueryValue) oldQuery.clone();
		newQuery.add(param);
		return newQuery;
	}

    //******************************************************************************************************************
    // Constructing the action value structure
    //******************************************************************************************************************
	public static ActionValue actParam(ChannelNameValue actionName) {
        ActionValue actionVal = new ActionValue(actionName);
        return actionVal;
    }
    public static ActionValue actParam(ActionValue oldAction, ParamValue param) {
		ActionValue newAction = (ActionValue) oldAction.clone();
        newAction.add(param);
        return newAction;
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

    //******************************************************************************************************************
    // Specials handler -- Fragile!! Handle with care
    //******************************************************************************************************************
    public static StringValue special(NameValue spl) {
        Map<String,Object> json = new HashMap<>();
        json.put("special",spl.toJson());
        return (new StringValue(Json.writeValueAsStringHard(json)));
    }

    //******************************************************************************************************************
    // Constructing the rule value structure
    //******************************************************************************************************************
    public static RuleValue ifttt(TriggerValue trigger, ActionValue action) {
        RuleValue ruleVal = new RuleValue(trigger, action);
        return ruleVal;
    }

    //******************************************************************************************************************
    // Constructing the rule value structure
    //******************************************************************************************************************
    public static Value jsonOut(Value val) {
        Map<String,Object> json = new HashMap<>();
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
