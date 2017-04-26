package edu.stanford.nlp.sempre.thingtalk;

import java.util.List;

import fig.basic.LispTree;

public class QueryValue extends ParametricValue {
	public QueryValue(LispTree tree) {
		super(tree);
    }

	public QueryValue(ChannelNameValue name, List<ParamValue> params) {
		super(name, params);
    }

	public QueryValue(ChannelNameValue name) {
		super(name);
    }

    public QueryValue(TypedStringValue person, ChannelNameValue name) {
		super(person, name);
	}

	@Override
	public String getLabel() {
		return "query";
	}
}
