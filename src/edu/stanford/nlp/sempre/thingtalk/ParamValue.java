package edu.stanford.nlp.sempre.thingtalk;

import fig.basic.LispTree;
import fig.basic.LogInfo;
import edu.stanford.nlp.sempre.Value;

import java.util.*;


/**
 * Represents a parameter for thingtalk
 * @author Rakesh Ramesh
 */
public class ParamValue extends Value {
    public final String name;
    // type: "String", "Date", "List", "Number"
    public final String tt_type;
    public final String operator;
    public final String value;

    public ParamValue(LispTree tree) {
        this.name = tree.child(1).value;
        this.tt_type = tree.child(2).value;
        this.operator = tree.child(3).value;
        this.value = tree.child(4).value;
    }

    public ParamValue(String name, String tt_type, String operator, String value) {
        this.name = name;
        this.tt_type = tt_type;
        this.operator = operator;
        this.value = value;
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("param");
        tree.addChild(name);
        tree.addChild(tt_type);
        tree.addChild(operator);
        tree.addChild(value);
        return tree;
    }

    public Map<String, Object> toJSON() {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("name", name);
        json.put("type", tt_type);
        json.put("operator", operator);
        json.put("value", value);
        return json;
    }

    @Override public int hashCode() { return (name.hashCode() ^ operator.hashCode() ^ value.hashCode()); }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParamValue that = (ParamValue) o;
        // Note: only check name and the value
        return (this.name.equals(that.name) && this.operator.equals(that.operator) && this.value.equals(that.value));
    }
}