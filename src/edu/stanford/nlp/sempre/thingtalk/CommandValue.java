package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents a thingtalk command
 * @author Rakesh Ramesh
 */
public class CommandValue extends Value {
    public final String type;
    public final Value value;

    public CommandValue(LispTree tree) {
        this.type = tree.child(1).value;
        this.value = Values.fromLispTree(tree.child(2));
    }

    public CommandValue(String type, Value value) {
        this.type = type;
        this.value = value;
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("command");
        tree.addChild(type);
        tree.addChild(value.toLispTree());
        return tree;
    }

    @Override public Map<String,Object> toJson() {
        Map<String,Object> json = new HashMap<String,Object>();
        json.put("type", type);
        json.put("value", value.toJson());
        return json;
    }

    @Override public boolean equals(Object o) {
        if(this == o) return true;
        if(o == null || getClass() != o.getClass()) return false;
        CommandValue that = (CommandValue) o;
        if(this.type != that.type || !value.equals(that.value)) return false;
        return true;
    }

    @Override public int hashCode() {
        return (type.hashCode() ^ value.hashCode());
    }
}
