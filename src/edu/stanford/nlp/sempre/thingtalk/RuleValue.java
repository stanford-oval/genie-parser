package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents a thingtalk rule.
 * @author Rakesh Ramesh
 */
public class RuleValue extends Value {
    public final TriggerValue trigger;
    public final ActionValue action;

    public RuleValue(TriggerValue trigger, ActionValue action) {
        this.trigger = trigger;
        this.action = action;
    }
    public RuleValue(LispTree tree) {
        this.trigger = (TriggerValue) Values.fromLispTree(tree.child(1));
        this.action = (ActionValue) Values.fromLispTree(tree.child(2));
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("rule");
        tree.addChild(trigger.toLispTree());
        tree.addChild(action.toLispTree());
        return tree;
    }

    public Map<String,Object> toJson() {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("trigger", trigger.toJson());
        json.put("action", action.toJson());
        return json;
    }

    @Override public boolean equals(Object o) {
        if(this == o) return true;
        if(o == null || getClass() != o.getClass()) return false;
        RuleValue that = (RuleValue) o;
        if(!trigger.equals(that.trigger) || !action.equals(that.action)) return false;
        return true;
    }
    @Override public int hashCode() {
        return trigger.hashCode() ^ action.hashCode();
    }
}
