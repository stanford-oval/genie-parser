package edu.stanford.nlp.sempre;

import java.util.HashMap;
import java.util.Map;

import fig.basic.LispTree;

/**
 * Assign normal form semantics to each formula.
 *
 * ... except that we run some copy-pasted magic to convert questions into
 * sparql
 *
 * @author Giovanni Campagna
 */
public class NormalFormExecutor extends Executor {
	private static class NormalFormConverter {
		private Map<String, String> renames;
		private int index;

		private Formula reduce(Formula formula) {
			if (formula instanceof JoinFormula) {
				JoinFormula jf = (JoinFormula) formula;

				Formula call = reduce(jf.relation);

				// A Redex
				if (call instanceof LambdaFormula)
					return reduce(Formulas.lambdaApply((LambdaFormula) call, jf.child));

				// Not a Redex
				return new JoinFormula(jf.relation, reduce(jf.child));

			} else if (formula instanceof LambdaFormula) {
				LambdaFormula lf = (LambdaFormula) formula;
				return new LambdaFormula(lf.var, reduce(lf.body));
			} else {
				return formula;
			}
		}

		private Formula alphaNormalize(Formula formula) {
			index = 0;
			renames = new HashMap<String, String>();
			return this.doAlphaNormalize(formula);
		}

		private Formula doAlphaNormalize(Formula formula) {
			if (formula instanceof LambdaFormula) {
				int var = index++;
				LambdaFormula lf = (LambdaFormula) formula;
				String old = renames.get(lf.var);
				renames.put(lf.var, "x" + var);
				Formula body = doAlphaNormalize(lf.body);
				renames.put(lf.var, old);
				return new LambdaFormula("x" + var, body);
			} else if (formula instanceof JoinFormula) {
				JoinFormula jf = (JoinFormula) formula;
				Formula relation = doAlphaNormalize(jf.relation);
				Formula child = doAlphaNormalize(jf.child);
				return new JoinFormula(relation, child);
			} else if (formula instanceof VariableFormula) {
				VariableFormula vf = (VariableFormula) formula;
				if (renames.get(vf.name) != null)
					return new VariableFormula(renames.get(vf.name));
				else
					return vf;
			} else {
				return formula;
			}
		}

		private Formula etaReduce(Formula formula) {
			if (formula instanceof LambdaFormula) {
				LambdaFormula lf = (LambdaFormula) formula;

				Formula body = etaReduce(lf.body);
				if (body instanceof JoinFormula) {
					JoinFormula jf = (JoinFormula) body;
					if (jf.child instanceof VariableFormula) {
						VariableFormula vf = (VariableFormula) jf.child;
						if (vf.name.equals(lf.var))
							return etaReduce(jf.relation);
					}
				}
			}

			return formula;
		}
	}

	public static StringValue normalize(Formula formula) {
		NormalFormConverter cvt = new NormalFormConverter();
		Formula reduced = cvt.reduce(formula);
		Formula etaReduced = cvt.etaReduce(reduced);
		Formula alphaNormalized = cvt.alphaNormalize(etaReduced);
		LispTree tree = alphaNormalized.toLispTree();
		return new StringValue(tree.toString());
    }

	// work around JavaExecutor annoyances
	public static StringValue normalize(Value value) {
		return normalize(new ValueFormula<>(value));
	}

    @Override
	public Response execute(Formula formula, ContextValue context) {
		return new Response(normalize(formula));
	}
}
