package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

import java.util.HashMap;
import java.util.Map;

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
    public static ParamValue paramForm(String tt_type, NameValue tt_arg, String operator, Value value) {
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
        TriggerValue newTrigger = new TriggerValue(oldTrigger.name, oldTrigger.params);
        //for(Value param: params) {
        newTrigger.add(param);
        //}
        //return Collections.singletonList(newTrigger);
        return newTrigger;
    }

    //******************************************************************************************************************
    // Constructing the action value structure
    //******************************************************************************************************************
    public static ActionValue actParam(NameValue actionName) {
        ActionValue actionVal = new ActionValue(actionName);
        return actionVal;
    }
    public static ActionValue actParam(ActionValue oldAction, ParamValue param) {
        ActionValue newAction = new ActionValue(oldAction.name, oldAction.params);
        //for(Value param: params) {
        newAction.add(param);
        //}
        //return Collections.singletonList(newAction);
        return newAction;
    }

    //******************************************************************************************************************
    // Specials handler -- Fragile!! Handle with care
    //******************************************************************************************************************
    public static StringValue special(NameValue spl) {
        Map<String,Object> json = new HashMap<String,Object>();
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
    public static Value jsonOut(RuleValue rule) {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("rule",rule.toJson());
        return (new StringValue(Json.writeValueAsStringHard(json)));
    }
}
