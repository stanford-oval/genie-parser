package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.sempre.StringValue;
import edu.stanford.nlp.sempre.Value;
import fig.basic.LispTree;

public class EmailAddressValue extends Value {
  public final String value;

  public EmailAddressValue(String value) {
    this.value = value;
  }

  public EmailAddressValue(LispTree tree) {
    this.value = tree.child(1).value;
  }

  @Override
  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild("emailaddress");
    tree.addChild(value);
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
    json.put("value", value);
    return json;
  }

  @Override
  public int hashCode() {
    return value.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    StringValue that = (StringValue) o;
    return this.value.equals(that.value);
  }
}