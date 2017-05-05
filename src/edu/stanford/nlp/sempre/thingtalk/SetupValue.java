package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents a rule to be setup remotely.
 * @author Rakesh Ramesh
 */
public final class SetupValue extends Value implements Cloneable{
    public final RuleValue rule;
    public final TriggerValue trigger;
    public final QueryValue query;
    public final ActionValue action;
    public final TypedStringValue person;

    public SetupValue(TypedStringValue person, RuleValue rule, TriggerValue trigger, QueryValue query, ActionValue action) {
        this.person = person;
        this.trigger = trigger;
        this.query = query;
        this.action = action;
        this.rule = rule;
    }

    public SetupValue(LispTree tree) {
        this.person = (TypedStringValue) Values.fromLispTree(tree.child(1));
        this.rule = (RuleValue) Values.fromLispTree(tree.child(2));
        this.trigger = (TriggerValue) Values.fromLispTree(tree.child(2));
        this.query = (QueryValue) Values.fromLispTree(tree.child(2));
        this.action = (ActionValue) Values.fromLispTree(tree.child(2));
    }

    private void addToLispTree(LispTree tree, Value val) {
        if(val == null)
            tree.addChild("null");
        else
            tree.addChild(val.toLispTree());
    }

    @Override
    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("setup");
        tree.addChild(this.person.toLispTree());
        addToLispTree(tree, this.rule);
        addToLispTree(tree, this.trigger);
        addToLispTree(tree, this.query);
        addToLispTree(tree, this.action);
        return tree;
    }

    @Override
    public Map<String, Object> toJson() {
        Map<String, Object> json = new HashMap<>();
        json.put("person", this.person.value);
        if(this.rule != null)
            json.put("rule", this.rule.toJson());
        if(this.trigger != null)
            json.put("trigger", this.trigger.toJson());
        if(this.query != null)
            json.put("query", this.query.toJson());
        if(this.action != null)
            json.put("action", this.action.toJson());
        return json;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        SetupValue other = (SetupValue) o;

        if(!person.equals(other.person))
            return false;

        if(rule == null) {
            if(other.rule != null)
                return false;
        } else if(!rule.equals(other.rule))
            return false;
        if (action == null) {
            if (other.action != null)
                return false;
        } else if (!action.equals(other.action))
            return false;
        if (query == null) {
            if (other.query != null)
                return false;
        } else if (!query.equals(other.query))
            return false;
        if (trigger == null) {
            if (other.trigger != null)
                return false;
        } else if (!trigger.equals(other.trigger))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + person.hashCode();
        result = prime * result + ((rule == null) ? 0 : rule.hashCode());
        result = prime * result + ((action == null) ? 0 : action.hashCode());
        result = prime * result + ((query == null) ? 0 : query.hashCode());
        result = prime * result + ((trigger == null) ? 0 : trigger.hashCode());
        return result;
    }

    @Override
    public SetupValue clone() {
        return new SetupValue(new TypedStringValue("Username", this.person.value),
                (this.rule == null) ? null : rule.clone(),
                (this.trigger == null) ? null : (TriggerValue) trigger.clone(),
                (this.query == null) ? null : (QueryValue) query.clone(),
                (this.action == null) ? null : (ActionValue) action.clone());
    }
}
