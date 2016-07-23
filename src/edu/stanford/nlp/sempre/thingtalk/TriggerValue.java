package edu.stanford.nlp.sempre.thingtalk;

import java.util.List;

import fig.basic.LispTree;
/**
 * Represents a thingtalk trigger
 * @author Rakesh Ramesh
 */
public class TriggerValue extends ParametricValue {
	public TriggerValue(LispTree tree) {
		super(tree);
    }

	public TriggerValue(ChannelNameValue name, List<ParamValue> params) {
		super(name, params);
    }

	public TriggerValue(ChannelNameValue name) {
		super(name);
    }

	@Override
	public String getLabel() {
		return "trigger";
    }
}