package edu.stanford.nlp.sempre;

import fig.basic.LispTree;
import fig.basic.LogInfo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ListValue extends Value {
  public final List<Value> values;

  public ListValue(LispTree tree) {
    values = new ArrayList<Value>();
    for (int i = 1; i < tree.children.size(); i++)
      values.add(Values.fromLispTree(tree.child(i)));
  }

  public ListValue(List<Value> values) { this.values = values; }

  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild("list");
    for (Value value : values)
      tree.addChild(value == null ? LispTree.proto.newLeaf(null) : value.toLispTree());
    return tree;
  }

  public void log() {
    for (Value value : values)
      LogInfo.logs("%s", value);
  }

  public Map<String,Object> toJson() {
    Map<String,Object> json = new HashMap<String,Object>();
    List<Object> listJson = new ArrayList<Object>();
    json.put("list", listJson);
    for(Value value: values)
      listJson.add(value.toJson());
    return json;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ListValue that = (ListValue) o;
    if (!values.equals(that.values)) return false;
    return true;
  }

  @Override public int hashCode() { return values.hashCode(); }
}
