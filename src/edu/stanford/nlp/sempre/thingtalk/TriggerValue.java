package edu.stanford.nlp.sempre.thingtalk;

import fig.basic.LispTree;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import com.google.common.collect.Lists;
import java.util.*;
/**
 * Represents a thingtalk trigger
 * @author Rakesh Ramesh
 */
public class TriggerValue extends Value {
    public final String name;
    public final List<ParamValue> params;

    public TriggerValue(LispTree tree) {
        this.name = tree.child(1).value;
        this.params = new ArrayList<ParamValue>();
        for(int i=2; i < tree.children.size(); i++) {
            this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
        }
    }
    public TriggerValue(String name, List<ParamValue> params) {
        this.name = name;
        this.params = new ArrayList<ParamValue>();
        this.params.addAll(params);
    }
    public TriggerValue(String name) {
        this.name = name;
        this.params = new ArrayList<ParamValue>();
    }

    public void add(ParamValue param) {
        assert (params != null) : param;
        params.add(param);
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("trigger");
        tree.addChild(this.name);
        for(ParamValue param : this.params)
            tree.addChild(param.toLispTree());
        return tree;
    }

    public Map<String, Object> toJSON() {
        Map<String,Object> json = new HashMap<String, Object>();
        json.put("name", name);
        List<Object> args = new ArrayList<Object>();
        json.put("args",args);
        for(ParamValue param: params) {
            args.add(param.toJSON());
        }
        return json;
    }

    @Override public boolean equals(Object o) {
        if(this == o) return true;
        if(o == null || getClass() != o.getClass()) return false;
        TriggerValue that = (TriggerValue) o;
        if(name != that.name || !params.equals(that.params)) return false;
        return true;
    }
    @Override public int hashCode() {
        return name.hashCode()^params.hashCode();
    }
}