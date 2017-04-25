package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

/**
 * Base class for thingtalk entities that take parameters (actions, triggers,
 * queries)
 * 
 * @author Rakesh Ramesh & Giovanni Campagna
 */
public abstract class ParametricValue extends Value implements Cloneable {
  public final TypedStringValue person; // null if its me or else value supplied
  public final ChannelNameValue name;
  public ArrayList<ParamValue> params = new ArrayList<>();

  public ParametricValue(LispTree tree) {
    this.person = (TypedStringValue) Values.fromLispTree(tree.child(1));

    this.name = (ChannelNameValue) Values.fromLispTree(tree.child(2));

    for (int i = ((this.person == null) ? 2 : 3); i < tree.children.size(); i++) {
      this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
    }
  }

  public ParametricValue(ChannelNameValue name, List<ParamValue> params) {
    this.name = name;
    this.params.addAll(params);
    this.person = null;
  }

  public ParametricValue(TypedStringValue person, ChannelNameValue name) {
    this.name = name;
    this.person = person;
  }

  public ParametricValue(ChannelNameValue name) {
    this.name = name;
    this.person = null;
  }

  protected abstract String getLabel();

  public void add(ParamValue param) {
    assert (params != null) : param;
    params.add(param);
  }

  public boolean hasParamName(String name) {
    for (ParamValue p : params) {
      if (p.name.argname.equals(name))
        return true;
    }
    return false;
  }

  @Override
  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild(getLabel());
    if(this.person != null) tree.addChild(person.toLispTree());
    tree.addChild(name.toLispTree());
    for (ParamValue param : this.params)
      tree.addChild(param.toLispTree());
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
    if(this.person != null) json.put("person", person.toJson());
    json.put("name", name.toJson());
    List<Object> args = new ArrayList<>();
    json.put("args", args);
    for (ParamValue param : params) {
      args.add(param.toJson());
    }
    return json;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    ParametricValue that = (ParametricValue) o;
    if (!name.equals(that.name) || !params.equals(that.params) || (person != null && that.person != null && !person.equals(that.person)))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    int hashCode = name.hashCode() ^ params.hashCode();
    if(this.person != null) hashCode = hashCode ^ person.hashCode();
    return hashCode;
  }

  @Override
  public ParametricValue clone() {
    try {
      ParametricValue self = (ParametricValue) super.clone();
      self.params = new ArrayList<>(self.params);
      return self;
    } catch (CloneNotSupportedException e) {
      throw new RuntimeException(e);
    }
  }
}
