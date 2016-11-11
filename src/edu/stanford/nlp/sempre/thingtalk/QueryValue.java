package edu.stanford.nlp.sempre.thingtalk;

import fig.basic.LispTree;

public class QueryValue extends ParametricValue {
	public QueryValue(LispTree tree) {
		super(tree);
    }

	public QueryValue(ChannelNameValue name) {
		super(name);
    }

	@Override
	public String getLabel() {
		return "query";
	}
}
