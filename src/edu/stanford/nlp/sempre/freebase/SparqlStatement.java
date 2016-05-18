package edu.stanford.nlp.sempre.freebase;

import edu.stanford.nlp.sempre.DateValue;
import edu.stanford.nlp.sempre.PrimitiveFormula;
import edu.stanford.nlp.sempre.ValueFormula;

public class SparqlStatement implements SparqlExpr {
	public final PrimitiveFormula arg1;
	public final String relation;
	public final PrimitiveFormula arg2;
	public boolean optional;
	public String options;

	public SparqlStatement(PrimitiveFormula arg1, String relation, PrimitiveFormula arg2, boolean optional) {
		this.arg1 = arg1;
		this.relation = relation;
		this.arg2 = arg2;
		this.optional = optional;
		this.options = null;
	}

	public SparqlStatement(PrimitiveFormula arg1, String relation, PrimitiveFormula arg2, boolean optional,
			String options) {
		this(arg1, relation, arg2, optional);
		this.options = options;
	}

	public static boolean isIndependent(String relation) {
		return relation.equals(":");
	}

	public static boolean isOperator(String relation) {
		return relation.equals("=") || relation.equals("!=") || relation.equals("<") || relation.equals(">")
				|| relation.equals("<=") || relation.equals(">=");
	}

	public static boolean isSpecialFunction(String relation) {
		return relation.equals("STRSTARTS") || relation.equals("STRENDS");
	}

	public String simpleString() {
		// Workaround for annoying dates:
		// http://answers.semanticweb.com/questions/947/dbpedia-sparql-endpoint-xsddate-comparison-weirdness
		if (arg2 instanceof ValueFormula && ((ValueFormula) arg2).value instanceof DateValue) {
			if (isOperator(relation)) {
				if (relation.equals("=")) {
					// (= (date 2000 -1 -1)) really means (>= (2000 -1 -1)) and
					// (< (2001 -1 -1))
					DateValue startDate = (DateValue) ((ValueFormula) arg2).value;
					DateValue endDate = advance(startDate);
					return SparqlUtils.dateTimeStr(arg1) + " >= "
							+ SparqlUtils.dateTimeStr(new ValueFormula<DateValue>(startDate)) + ") . FILTER ("
							+ SparqlUtils.dateTimeStr(arg1) + " < "
							+ SparqlUtils.dateTimeStr(new ValueFormula<DateValue>(endDate));
				}
				if (relation.equals("<=")) {
					// (<= (date 2000 -1 -1)) really means (< (date 2001 -1 -1))
					DateValue startDate = (DateValue) ((ValueFormula) arg2).value;
					DateValue endDate = advance(startDate);
					return SparqlUtils.dateTimeStr(arg1) + " < "
							+ SparqlUtils.dateTimeStr(new ValueFormula<DateValue>(endDate));
				}
				if (relation.equals(">")) {
					// (> (date 2000 -1 -1)) really means >= (date 2001 -1 -1)
					DateValue startDate = (DateValue) ((ValueFormula) arg2).value;
					DateValue endDate = advance(startDate);
					return SparqlUtils.dateTimeStr(arg1) + " >= "
							+ SparqlUtils.dateTimeStr(new ValueFormula<DateValue>(endDate));
				}
				if (relation.equals("<") || relation.equals(">="))
					return SparqlUtils.dateTimeStr(arg1) + " " + relation + " " + SparqlUtils.dateTimeStr(arg2);
				// Note: != is not treated specially
			}
		}

		return SparqlUtils.plainStr(arg1) + " " + relation + " " + SparqlUtils.plainStr(arg2);
	}

	private DateValue advance(DateValue date) {
		// TODO(pliang): deal with carrying over
		if (date.day != -1)
			return new DateValue(date.year, date.month, date.day + 1);
		if (date.month != -1)
			return new DateValue(date.year, date.month + 1, -1);
		return new DateValue(date.year + 1, -1, -1);
	}

	@Override
	public String toString() {
		String result;
		if (isSpecialFunction(relation)) { // Special functions
			result = "FILTER (" + relation + "(" + SparqlUtils.plainStr(arg1) + "," + SparqlUtils.plainStr(arg2) + "))";
		} else if (isOperator(relation)) {
			result = "FILTER (" + simpleString() + ")";
		} else if (optional) {
			result = "OPTIONAL { " + simpleString() + " }";
		} else {
			result = simpleString();
		}

		if (this.options != null)
			result += " " + this.options;

		return result;
	}
}