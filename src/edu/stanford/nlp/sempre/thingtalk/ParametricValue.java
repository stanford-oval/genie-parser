package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.NameValue;
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
	public final NameValue name;
	public final List<ParamValue> params;

	public ParametricValue(LispTree tree) {
		this.name = (NameValue) Values.fromLispTree(tree.child(1));
		this.params = new ArrayList<>();
		for (int i = 2; i < tree.children.size(); i++) {
			this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
		}
	}

	public ParametricValue(NameValue name, List<ParamValue> params) {
		this.name = name;
		this.params = new ArrayList<>();
		this.params.addAll(params);
	}

	public ParametricValue(NameValue name) {
		this.name = name;
		this.params = new ArrayList<>();
	}

	protected abstract String getLabel();

	public void add(ParamValue param) {
		assert (params != null) : param;
		params.add(param);
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
			return (ParametricValue) super.clone();
		} catch (CloneNotSupportedException e) {
			throw new RuntimeException(e);
		}
	}
}
