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
    // name: "from", "to", "at", "on", "text"
    public final String name;
    // type: "String", "Date", "List", "Number"
    public final String tt_type;
    public final String value;

    public ParamValue(LispTree tree) {
        this.name = tree.child(1).value;
        this.tt_type = tree.child(2).value;
        this.value = tree.child(3).value;
    }

    public ParamValue(String name, String tt_type, String value) {
        this.name = name;
        this.tt_type = tt_type;
        this.value = value;
    }

    public ParamValue(String name, String value) {
        this.name = name;
        switch(name) {
            case "at":
                this.tt_type = "Time"; break;
            case "on":
                this.tt_type = "Date"; break;
            case "from":
            case "to":
            default:
                this.tt_type = "String";
        }
        this.value = value;
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("param");
        tree.addChild(name);
        tree.addChild(tt_type);
        tree.addChild(value);
        return tree;
    }

    public Map<String, Object> toJSON() {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("name", name);
        json.put("type", tt_type);
        json.put("value", value);
        return json;
    }

    @Override public int hashCode() { return value.hashCode(); }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParamValue that = (ParamValue) o;
        // Note: only check name and the value
        return (this.name.equals(that.name) && this.value.equals(that.value));
    }
}