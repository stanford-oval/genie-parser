package edu.stanford.nlp.sempre.freebase;

import java.util.List;

import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.NameValue;
import edu.stanford.nlp.sempre.PrimitiveFormula;
import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.ValueFormula;
import fig.basic.StrUtils;

//Example: { <expr> ... <expr> }
public class SparqlBlock implements SparqlExpr {
	public final List<SparqlExpr> children = Lists.newArrayList();

	public SparqlBlock add(SparqlExpr expr) {
		if (expr instanceof SparqlBlock)
			this.children.addAll(((SparqlBlock) expr).children);
		else
			this.children.add(expr);
		return this;
	}

	private String getId(PrimitiveFormula formula) {
		if (!(formula instanceof ValueFormula))
			return null;
		Value value = ((ValueFormula) formula).value;
		if (!(value instanceof NameValue))
			return null;
		return ((NameValue) value).id;
	}

	private boolean isPrimitiveType(String id) {
		if (FreebaseInfo.BOOLEAN.equals(id))
			return true;
		if (FreebaseInfo.INT.equals(id))
			return true;
		if (FreebaseInfo.FLOAT.equals(id))
			return true;
		if (FreebaseInfo.DATE.equals(id))
			return true;
		if (FreebaseInfo.TEXT.equals(id))
			return true;
		return false;
	}

	public void addStatement(PrimitiveFormula arg1, String property, PrimitiveFormula arg2, boolean optional) {
		// if (!property.startsWith("fb:") &&
		// !SparqlStatement.isOperator(property)
		// && !SparqlStatement.isSpecialFunction(property) &&
		// !SparqlStatement.isIndependent(property))
		// throw new RuntimeException("Invalid SPARQL property: " + property);

		// Ignore statements like:
		// ?x fb:type.object.type fb:type.datetime
		// because we should have already captured the semantics using other
		// formulas that involve ?x.
		if (property.equals(FreebaseInfo.TYPE)) {
			String id = getId(arg2);
			if (isPrimitiveType(id) || FreebaseInfo.ANY.equals(id))
				return;
		}
		if (SparqlStatement.isIndependent(property))
			return; // Nothing connecting arg1 and arg2
		add(new SparqlStatement(arg1, property, arg2, optional));
	}

	@Override
	public String toString() {
		List<String> strings = Lists.newArrayList();
		for (SparqlExpr expr : children) {
			if (expr instanceof SparqlSelect)
				strings.add("{ " + expr + " }");
			else
				strings.add(expr.toString());
		}
		return "{ " + StrUtils.join(strings, " . ") + " }";
	}
}
