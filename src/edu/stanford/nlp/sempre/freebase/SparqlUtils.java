package edu.stanford.nlp.sempre.freebase;

import edu.stanford.nlp.sempre.DateValue;
import edu.stanford.nlp.sempre.NameValue;
import edu.stanford.nlp.sempre.NumberValue;
import edu.stanford.nlp.sempre.PrimitiveFormula;
import edu.stanford.nlp.sempre.StringValue;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.ValueFormula;
import edu.stanford.nlp.sempre.VariableFormula;

public final class SparqlUtils {
	private SparqlUtils() {
	}

	public static String dateTimeStr(PrimitiveFormula formula) {
		return "xsd:datetime(" + plainStr(formula) + ")";
	}

	public static String plainStr(PrimitiveFormula formula) {
		if (formula instanceof VariableFormula)
			return ((VariableFormula) formula).name;

		Value value = ((ValueFormula) formula).value;

		if (value instanceof StringValue) {
			String s = ((StringValue) value).value;
			return "\"" + s.replaceAll("\"", "\\\\\"") + "\"" + (s.equals("en") ? "" : "@en");
		}
		if (value instanceof NameValue)
			return ((NameValue) value).id;
		if (value instanceof NumberValue)
			return ((NumberValue) value).value + "";
		if (value instanceof DateValue) {
			DateValue date = (DateValue) value;
			if (date.month == -1)
				return "\"" + date.year + "\"" + "^^xsd:datetime";
			if (date.day == -1)
				return "\"" + date.year + "-" + date.month + "\"" + "^^xsd:datetime";
			return "\"" + date.year + "-" + date.month + "-" + date.day + "\"" + "^^xsd:datetime";
		}
		throw new RuntimeException("Unhandled primitive: " + value);
	}
}