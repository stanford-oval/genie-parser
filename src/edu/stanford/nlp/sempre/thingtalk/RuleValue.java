package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

/**
 * Represents a thingtalk rule.
 * @author Rakesh Ramesh
 */
public class RuleValue extends Value {
  public final TriggerValue trigger;
  public final QueryValue query;
  public final ActionValue action;

  public RuleValue(TriggerValue trigger, QueryValue query, ActionValue action) {
    this.trigger = trigger;
    this.query = query;
    this.action = action;
  }

  public RuleValue(LispTree tree) {
    this.trigger = (TriggerValue) Values.fromLispTree(tree.child(1));
    this.query = (QueryValue) Values.fromLispTree(tree.child(2));
    this.action = (ActionValue) Values.fromLispTree(tree.child(3));
  }

  @Override
  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild("rule");
    if (trigger != null)
      tree.addChild(trigger.toLispTree());
    else
      tree.addChild("null");
    if (query != null)
      tree.addChild(query.toLispTree());
    else
      tree.addChild("null");
    if (action != null)
      tree.addChild(action.toLispTree());
    else
      tree.addChild("null");
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
    if (trigger != null)
      json.put("trigger", trigger.toJson());
    if (query != null)
      json.put("query", query.toJson());
    if (action != null)
      json.put("action", action.toJson());
    return json;
  }

  // Generated with Eclipse 
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((action == null) ? 0 : action.hashCode());
    result = prime * result + ((query == null) ? 0 : query.hashCode());
    result = prime * result + ((trigger == null) ? 0 : trigger.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    RuleValue other = (RuleValue) obj;
    if (action == null) {
      if (other.action != null)
        return false;
    } else if (!action.equals(other.action))
      return false;
    if (query == null) {
      if (other.query != null)
        return false;
    } else if (!query.equals(other.query))
      return false;
    if (trigger == null) {
      if (other.trigger != null)
        return false;
    } else if (!trigger.equals(other.trigger))
      return false;
    return true;
  }
}
