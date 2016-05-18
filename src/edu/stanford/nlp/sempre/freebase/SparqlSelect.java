package edu.stanford.nlp.sempre.freebase;

import java.util.List;

import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.PrimitiveFormula;
import edu.stanford.nlp.sempre.VariableFormula;

//Example: SELECT ?x ?y WHERE { ... } ORDER BY ?x ?y LIMIT 10 OFFSET 3
public class SparqlSelect implements SparqlExpr {
	public static class Var {
		public final VariableFormula var;
		public final String asValue; // for COUNT(?x3) as ?x2
		public final String unit; // Specifies the types of the variable (used
									// to parse back the results)
		public final boolean isAuxiliary; // Whether this is supporting
											// information (e.g., names)
		public final String description; // Human-friendly for display

		public Var(VariableFormula var, String asValue, String unit, boolean isAuxiliary, String description) {
			this.var = var;
			this.asValue = asValue;
			this.unit = unit;
			this.isAuxiliary = isAuxiliary;
			this.description = description;
		}

		@Override
		public String toString() {
			if (asValue == null)
				return var.name;
			return "(" + asValue + " AS " + var.name + ")";
		}
}

	public final List<Var> selectVars = Lists.newArrayList();

	public SparqlBlock where;
	public final List<VariableFormula> sortVars = Lists.newArrayList();
	public int offset = 0; // Start at this point when returning results
	public int limit = -1; // Number of results to return

	@Override
	public String toString() {
		StringBuilder out = new StringBuilder();
		out.append("SELECT DISTINCT");
		for (Var var : selectVars)
			out.append(" " + var.toString());
		out.append(" WHERE " + where);
		if (sortVars.size() > 0) {
			out.append(" ORDER BY");
			for (PrimitiveFormula sortVar : sortVars)
				out.append(" " + SparqlUtils.plainStr(sortVar));
		}
		if (limit != -1)
			out.append(" LIMIT " + limit);
		if (offset != 0)
			out.append(" OFFSET " + offset);
		return out.toString();
	}
}