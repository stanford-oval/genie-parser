package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.NameValue;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a thingtalk action
 * @author Rakesh Ramesh
 */
public class ActionValue extends Value {
    public final NameValue name;
    public final List<ParamValue> params;

    public ActionValue(LispTree tree) {
        this.name = (NameValue) Values.fromLispTree(tree.child(1));
        this.params = new ArrayList<ParamValue>();
        for(int i=2; i < tree.children.size(); i++) {
            this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
        }
    }
    public ActionValue(NameValue name, List<ParamValue> params) {
        this.name = name;
        this.params = new ArrayList<ParamValue>();
        this.params.addAll(params);
    }
    public ActionValue(NameValue name) {
        this.name = name;
        this.params = new ArrayList<ParamValue>();
    }

    public void add(ParamValue param) {
        assert (params != null) : param;
        params.add(param);
    }

    public LispTree toLispTree() {
        LispTree tree = LispTree.proto.newList();
        tree.addChild("action");
        tree.addChild(name.toLispTree());
        for(ParamValue param : this.params)
            tree.addChild(param.toLispTree());
        return tree;
    }

    public Map<String, Object> toJson() {
        Map<String,Object> json = new HashMap<String, Object>();
        json.put("name", name.toJson());
        List<Object> args = new ArrayList<Object>();
        json.put("args",args);
        for(ParamValue param: params) {
            args.add(param.toJson());
        }
        return json;
    }

    @Override public boolean equals(Object o) {
        if(this == o) return true;
        if(o == null || getClass() != o.getClass()) return false;
        ActionValue that = (ActionValue) o;
        if(!name.equals(that.name) || !params.equals(that.params)) return false;
        return true;
    }
    @Override public int hashCode() {
        return name.hashCode()^params.hashCode();
    }
}