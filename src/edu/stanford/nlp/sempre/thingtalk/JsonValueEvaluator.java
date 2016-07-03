package edu.stanford.nlp.sempre.thingtalk;

import java.util.Map;

import edu.stanford.nlp.sempre.*;

public class JsonValueEvaluator implements ValueEvaluator {
	private final ExactValueEvaluator exact;

	public JsonValueEvaluator() {
		exact = new ExactValueEvaluator();
	}

	@Override
	public double getCompatibility(Value target, Value pred) {
		if (!(target instanceof StringValue) || !(pred instanceof StringValue))
			return exact.getCompatibility(target, pred);

		String targetString = ((StringValue) target).value;
		String predString = ((StringValue) pred).value;

		Map<String, Object> targetJson = Json.readMapHard(targetString);
		Map<String, Object> predJson = Json.readMapHard(predString);

		return targetJson.equals(predJson) ? 1 : 0;
	}
}
