package edu.stanford.nlp.sempre.thingtalk;

import java.util.List;

import edu.stanford.nlp.sempre.NameValue;
import fig.basic.LispTree;
/**
 * Represents a thingtalk trigger
 * @author Rakesh Ramesh
 */
public class TriggerValue extends ParametricValue {
	public TriggerValue(LispTree tree) {
		super(tree);
    }
    public TriggerValue(NameValue name, List<ParamValue> params) {
		super(name, params);
    }
    public TriggerValue(NameValue name) {
		super(name);
    }

	@Override
	public String getLabel() {
		return "trigger";
    }
}