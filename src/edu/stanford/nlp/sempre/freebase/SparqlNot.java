package edu.stanford.nlp.sempre.freebase;

//Example: FILTER NOT EXISTS { ... }
public class SparqlNot implements SparqlExpr {
	public final SparqlBlock block;

	public SparqlNot(SparqlBlock block) {
		this.block = block;
	}

	@Override
	public String toString() {
		return "FILTER NOT EXISTS " + block;
	}
}