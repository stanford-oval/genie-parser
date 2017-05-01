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
    public final TypedStringValue person;

    public SetupValue(TypedStringValue person, RuleValue rule) {
        this.person = person;
        this.rule = rule;
    }

    public SetupValue(LispTree tree) {
        this.person = (TypedStringValue) Values.fromLispTree(tree.child(1));
        this.rule = (RuleValue) Values.fromLispTree(tree.child(2));
    }

    @Override
    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("setup");
        tree.addChild(this.person.toLispTree());
        tree.addChild(this.rule.toLispTree());
        return tree;
    }

    @Override
    public Map<String, Object> toJson() {
        Map<String, Object> json = new HashMap<>();
        json.put("person", this.person.value);
        json.put("rule", this.rule.toJson());
        return json;
    }

    @Override
    public boolean equals(Object o) {
        if(this == o)
            return true;
        if(o == null || getClass()  != o.getClass())
            return false;
        SetupValue that = (SetupValue) o;
        if (!person.equals(that.person) || !rule.equals(that.rule))
            return false;
        return true;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + person.hashCode();
        result = prime * result + ((rule == null) ? 0 : rule.hashCode());
        return result;
    }

    @Override
    public SetupValue clone() {
        return new SetupValue(new TypedStringValue("Username", this.person.value), rule.clone());
    }
}
