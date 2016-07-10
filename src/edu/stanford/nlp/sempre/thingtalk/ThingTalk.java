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

    //******************************************************************************************************************
    // Casting Java built-in types to Sempre Value structures
    //******************************************************************************************************************
    public static StringValue stringValueCast(String value) {
        StringValue stringVal = new StringValue(value);
        return stringVal;
    }

    public static NumberValue numValueCast(Integer value) {
        NumberValue numVal = new NumberValue(value);
        return numVal;
    }
    public static NumberValue numValueCast(Double value) {
        NumberValue numVal = new NumberValue(value);
        return numVal;
    }

    public static NumberValue tempValueCast(String unit, Integer value) {
        NumberValue tempVal = new NumberValue(value, unit);
        return tempVal;
    }
    public static NumberValue tempValueCast(String unit, Double value) {
        NumberValue tempVal = new NumberValue(value, unit);
        return tempVal;
    }

    public static BooleanValue boolValueCast(String value) {
        BooleanValue boolVal = new BooleanValue(value == "True");
        return boolVal;
    }

    //******************************************************************************************************************
    // Constructing the parameter value structure
    //******************************************************************************************************************
	public static ParamValue paramForm(String tt_type, ParamNameValue tt_arg, String operator, Value value) {
        ParamValue paramVal = new ParamValue(tt_arg, tt_type, operator, value);
        return paramVal;
    }

    //******************************************************************************************************************
    // Constructing the trigger value structure
    //******************************************************************************************************************
    public static TriggerValue trigParam(NameValue triggerName) {
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
	public static QueryValue queryParam(NameValue queryName) {
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
    public static ActionValue actParam(NameValue actionName) {
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
    public static CommandValue cmdForm(String type, Value val) {
        CommandValue cmdVal = new CommandValue(type, val);
        return cmdVal;
    }
    public static CommandValue cmdForm(String type, String val) {
        StringValue strVal = new StringValue(val);
        CommandValue cmdVal = new CommandValue(type, strVal);
        return cmdVal;
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
	public static RuleValue timeRule(DateValue time, Value action) {
		ParamNameValue timeName = new ParamNameValue("time", "String", new NameValue("tt:builtin.at"));
		ParamValue timeParam = new ParamValue(timeName, "Time", "is", time);
		TriggerValue timeTrigger = new TriggerValue(new NameValue("tt:builtin.at"),
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
        else
            label = "error"; // FIXME: Error flow
        json.put(label, val.toJson());
        return (new StringValue(Json.writeValueAsStringHard(json)));
    }
}
