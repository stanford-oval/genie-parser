package edu.stanford.nlp.sempre;

import fig.basic.*;

import java.util.Map;
import java.util.HashMap;

/**
 * Assign null semantics to each formula.
 *
 * @author Percy Liang
 */
public class NormalFormExecutor extends Executor {
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

    public Response execute(Formula formula, ContextValue context) {
        Formula reduced = this.reduce(formula);
        Formula etaReduced = this.etaReduce(reduced);
        Formula alphaNormalized = this.alphaNormalize(etaReduced);
        LispTree tree = alphaNormalized.toLispTree();
        StringValue treeString = new StringValue(tree.toString());
        return new Response(treeString);
  }
}
