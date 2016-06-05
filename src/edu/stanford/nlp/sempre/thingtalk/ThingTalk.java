package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.NameValue;
import edu.stanford.nlp.sempre.StringValue;
import edu.stanford.nlp.sempre.Value;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Functions for supporting thingtalk
 *
 * @author Rakesh Ramesh
 */
public final class ThingTalk {
    public static List<Value> ifttt(List<Value> trigger, List<Value> action) {
        // TODO: Create rule value
        // Following is hacky
        Map<String,Object> json = new HashMap<String,Object>();
        Map<String,Object> ruleJson = new HashMap<String,Object>();
        json.put("rule",ruleJson);
        ruleJson.put("trigger",((TriggerValue) trigger.get(0)).toJSON());
        ruleJson.put("action",((ActionValue) action.get(0)).toJSON());

        return Collections.singletonList(new StringValue(Json.writeValueAsStringHard(json)));
    }

    public static List<Value> trigParam(Value trigger) {
        return Collections.singletonList(new TriggerValue(((NameValue) trigger).id));
    }
    public static List<Value> trigParam(List<Value> trigger, String param) {
        return trigParam(trigger, Collections.singletonList(new StringValue(param)));
    }
    public static List<Value> trigParam(List<Value> trigger, Value param) {
        return trigParam(trigger, Collections.singletonList(param));
    }
    public static List<Value> trigParam(List<Value> trigger, List<Value> params) {
        TriggerValue oldTrigger = ((TriggerValue) (trigger.get(0)));
        TriggerValue newTrigger = new TriggerValue(oldTrigger.name, oldTrigger.params);
        for(Value param: params) {
            newTrigger.add((ParamValue) param);
        }
        return Collections.singletonList(newTrigger);
    }

    public static List<Value> actParam(Value action) {
        return Collections.singletonList(new ActionValue(((NameValue) action).id));
    }
    public static List<Value> actParam(List<Value> action, String param) {
        return actParam(action, Collections.singletonList(new StringValue(param)));
    }
    public static List<Value> actParam(List<Value> action, Value param) {
        return actParam(action, Collections.singletonList(param));
    }
    public static List<Value> actParam(List<Value> action, List<Value> params) {
        ActionValue oldAction = ((ActionValue) (action.get(0)));
        ActionValue newAction = new ActionValue(oldAction.name, oldAction.params);
        for(Value param: params) {
            newAction.add((ParamValue) param);
        }
        return Collections.singletonList(newAction);
    }

    public static List<Value> paramForm(String tt_type, String operator, Value tt_arg, String value) {
        ParamValue param = new ParamValue(((NameValue) tt_arg).id, tt_type, operator, value);
        return Collections.singletonList(param);
    }
    public static List<Value> paramForm(String tt_type, String operator, Value tt_arg, Double value) {
        ParamValue param = new ParamValue(((NameValue) tt_arg).id, tt_type, operator, String.valueOf(value));
        return Collections.singletonList(param);
    }
    public static List<Value> paramForm(String tt_type, String operator, Value tt_arg, Integer value) {
        ParamValue param = new ParamValue(((NameValue) tt_arg).id, tt_type, operator, String.valueOf(value));
        return Collections.singletonList(param);
    }
    public static List<Value> paramForm(String tt_type, String operator, Value tt_arg, Value value) {
        ParamValue param = new ParamValue(((NameValue) tt_arg).id, tt_type, operator, value.toString());
        return Collections.singletonList(param);
    }
    public static List<Value> paramForm(String temp_type, String tt_type, String operator, Value tt_arg, Integer value) {
        ParamValue param = new ParamValue(((NameValue) tt_arg).id, tt_type, operator, String.valueOf(value), temp_type);
        return Collections.singletonList(param);
    }

    public static List<Value> special(Value spl) {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("special",((NameValue)spl).id);
        return Collections.singletonList(new StringValue(Json.writeValueAsStringHard(json)));
    }

    public static List<Value> paramTop(List<Value> params) {
        Map<String,Object> json = new HashMap<String, Object>();
        json.put("answer",((ParamValue)(params.get(0))).toJSON());
        return Collections.singletonList(new StringValue(Json.writeValueAsStringHard(json)));
    }
}
