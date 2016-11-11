package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import com.google.common.base.Joiner;

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
  public final ChannelNameValue name;

  public ArrayList<ParamValue> params = new ArrayList<>();
  private List<String> tokens;
  private List<String> nerTokens;

  public ParametricValue(LispTree tree) {
    this((ChannelNameValue) Values.fromLispTree(tree.child(1)));

    for (int i = 2; i < tree.children.size(); i++) {
      this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
    }
  }

  public ParametricValue(ChannelNameValue name) {
    this.name = name;
    this.tokens = Arrays.asList(this.name.rule.split("\\s+"));
    this.nerTokens = new ArrayList<>(this.tokens);
  }

  protected abstract String getLabel();

  public String getCanonical() {
    return Joiner.on(' ').join(this.tokens);
  }

  public String getNerCanonical() {
    return Joiner.on(' ').join(this.nerTokens);
  }

  public boolean add(ParamValue param, String paramCanonical, String paramNerTag) {
    assert (params != null) : param;
    params.add(param);

    String pname = param.name.argname;
    boolean rval = false;
    for (int i = 0; i < tokens.size(); i++) {
      if (tokens.get(i).equals("$" + pname)) {
        tokens.set(i, paramCanonical);
        nerTokens.set(i, paramNerTag);
        rval = true;
      }
    }

    return rval;
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
    tree.addChild(name.toLispTree());
    for (ParamValue param : this.params)
      tree.addChild(param.toLispTree());
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
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
    if (!name.equals(that.name) || !params.equals(that.params))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    return name.hashCode() ^ params.hashCode();
  }

  @Override
  public ParametricValue clone() {
    try {
      ParametricValue self = (ParametricValue) super.clone();
      self.params = new ArrayList<>(self.params);
      self.tokens = new ArrayList<>(self.tokens);
      self.nerTokens = new ArrayList<>(self.nerTokens);
      return self;
    } catch (CloneNotSupportedException e) {
      throw new RuntimeException(e);
    }
  }
}
