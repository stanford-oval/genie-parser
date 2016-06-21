package edu.stanford.nlp.sempre;

import fig.basic.LispTree;

import java.util.HashMap;
import java.util.Map;

public class UriValue extends Value {
  public final String value;

  public UriValue(LispTree tree) {
    this.value = tree.child(1).value;
  }

  public UriValue(String value) {
    this.value = value;
  }

  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild("url");
    tree.addChild(value != null ? value : "");
    return tree;
  }
  public Map<String,Object> toJson() {
    Map<String,Object> json = new HashMap<String,Object>();
    json.put("value", value);
    return json;
  }

  @Override public int hashCode() { return value.hashCode(); }
  @Override public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    UriValue that = (UriValue) o;
    return this.value.equals(that.value);
  }
}
