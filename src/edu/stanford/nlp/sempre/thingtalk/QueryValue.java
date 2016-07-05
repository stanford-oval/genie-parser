package edu.stanford.nlp.sempre.thingtalk;

import java.util.List;

import edu.stanford.nlp.sempre.NameValue;
import fig.basic.LispTree;

public class QueryValue extends ParametricValue {
	public QueryValue(LispTree tree) {
		super(tree);
    }

	public QueryValue(NameValue name, List<ParamValue> params) {
		super(name, params);
    }

	public QueryValue(NameValue name) {
		super(name);
    }

	@Override
	public String getLabel() {
		return "query";
	}
}
