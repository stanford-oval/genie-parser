package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;


/**
 * Represents a parameter for thingtalk
 * @author Rakesh Ramesh
 */
public class ParamValue extends Value {
	public final ParamNameValue name;

    // type: "String", "Date", "List", "Number", "Measure"
    public final String tt_type;
    // operator: "is", "contains", "has"
    public final String operator;
    public final Value value;

    public ParamValue(LispTree tree) {
		this.name = (ParamNameValue) Values.fromLispTree(tree.child(1));
        this.tt_type = tree.child(2).value;
        this.operator = tree.child(3).value;
        this.value = Values.fromLispTree(tree.child(4));
    }

	public ParamValue(ParamNameValue name, String tt_type, String operator, Value value) {
        this.name = name;
        this.tt_type = tt_type;
        this.operator = operator;
        this.value = value;
    }

    @Override
	public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("param");
        tree.addChild(name.toLispTree());
        tree.addChild(tt_type);
        tree.addChild(operator);
        tree.addChild(value.toLispTree());
        return tree;
    }

    @Override
	public Map<String,Object> toJson() {
        Map<String,Object> json = new HashMap<>();
        json.put("name", name.toJson());
        json.put("type", tt_type);
        json.put("operator", operator);
        json.put("value", value.toJson());
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