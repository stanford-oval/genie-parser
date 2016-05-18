package edu.stanford.nlp.sempre.freebase;

import java.util.List;

import com.google.common.collect.Lists;

import fig.basic.StrUtils;

//Example: { ... } UNION { ... } UNION { ... }
public class SparqlUnion implements SparqlExpr {
	public final List<SparqlBlock> children = Lists.newArrayList();

	public SparqlUnion add(SparqlBlock block) {
		this.children.add(block);
		return this;
	}

	@Override
	public String toString() {
		return "{ " + StrUtils.join(children, " UNION ") + " }";
}
}