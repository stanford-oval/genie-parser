package edu.stanford.nlp.sempre.thingtalk;

import java.util.List;

import fig.basic.LispTree;

/**
 * Represents a thingtalk action
 * @author Rakesh Ramesh
 */
public class ActionValue extends ParametricValue {
    public ActionValue(LispTree tree) {
		super(tree);
    }

	public ActionValue(ChannelNameValue name, List<ParamValue> params) {
		super(name, params);
    }

	public ActionValue(ChannelNameValue name) {
		super(name);
    }

	public ActionValue(TypedStringValue person, ChannelNameValue name) {
		super(person, name);
	}

	@Override
	public String getLabel() {
		return "action";
    }
}