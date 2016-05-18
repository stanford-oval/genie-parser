package edu.stanford.nlp.sempre;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.sempre.freebase.SparqlBlock;
import edu.stanford.nlp.sempre.freebase.SparqlNot;
import edu.stanford.nlp.sempre.freebase.SparqlSelect;
import edu.stanford.nlp.sempre.freebase.SparqlStatement;
import edu.stanford.nlp.sempre.freebase.SparqlUnion;
import fig.basic.LispTree;
import fig.basic.Ref;

/**
 * Assign normal form semantics to each formula.
 * 
 * ... except that we run some copy-pasted magic to convert questions into
 * sparql
 *
 * @author Giovanni Campagna
 */
public class NormalFormExecutor extends Executor {
	public static final String RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
	private Map<String, String> renames;
    private int index;

	/**
	 * Convert a lambda DCS form to Sparql
	 * 
	 * @author Percy Liang, Johnatan Berard (taken from freebase module and
	 *         stripped of freebase stuff)
	 *
	 */
	private static class SparqlConverter {
		private int numVars = 0; // Used to create new Sparql variables

		// For each variable, a description
		private Map<VariableFormula, String> descriptionsMap = new HashMap<VariableFormula, String>(); // ?y
		// =>
		// "Height
		// (meters)"

		private SparqlSelect query; // Resulting SPARQL expression

		// The state used to a SELECT statement (DRT box in which all variables
		// are existentially closed).
		private static class Box {
			// These are the variables that are first selected.
			List<VariableFormula> initialVars = new ArrayList<VariableFormula>();

			// Mapping from lambda DCS variables to SPARQL variables (which are
			// unique across the entire formula, not just this box).
			Map<VariableFormula, PrimitiveFormula> env = new LinkedHashMap<VariableFormula, PrimitiveFormula>();

			// Some SPARQL variables are bound to quantities based on the SELECT
			// statement (e.g., COUNT(?x)).
			Map<VariableFormula, String> asValuesMap = new HashMap<VariableFormula, String>();

			// e.g.,
			// ?y
			// =>
			// COUNT(?x)
			// or
			// ?y
			// =>
			// (?x1
			// +
			// ?x2)
		}

		public SparqlConverter(Formula rootFormula) throws BadFormulaException {
    		Ref<PrimitiveFormula> head = new Ref<PrimitiveFormula>();
    		Box box = new Box();

    		rootFormula = stripOuterLambdas(box, rootFormula);
    		SparqlBlock block = convert(rootFormula, head, null, box);
    		query = closeExistentialScope(block, head, box);
    	}

		public SparqlSelect getQuery() {
			return query;
		}

		// Strip off lambdas and add the variables to the environment
		// For example, in (lambda x (lambda y BODY)), we would create an
		// environment {x:NEW_VAR, y:NEW_VAR}, and interpret BODY as a unary.
		private Formula stripOuterLambdas(Box box, Formula formula) {
			while (formula instanceof LambdaFormula) {
				LambdaFormula lambda = ((LambdaFormula) formula);
				VariableFormula var = newVar();
				box.env.put(new VariableFormula(lambda.var), var);
				box.initialVars.add(var);
				formula = lambda.body;
			}
			return formula;
		}

		// Create a SELECT expression (a DRT box).
		private SparqlSelect closeExistentialScope(SparqlBlock block, Ref<PrimitiveFormula> head, Box box) {
			// Optimization: if block only contains one select statement, then
			// can optimize and just return that.
			if (block.children.size() == 1 && block.children.get(0) instanceof SparqlSelect
					&& box.initialVars.size() == 0)
				return (SparqlSelect) block.children.get(0);

			SparqlSelect select = new SparqlSelect();

			// Add initial variables
			for (VariableFormula var : box.initialVars)
				addSelectVar(box, select, block, var, false);

			// Add head variable (ensure that the head is a variable rather than
			// a primitive value)
			VariableFormula headVar = ensureIsVar(box, block, head);
			addSelectVar(box, select, block, headVar, false);

			// Add the other supporting variables in the environment (for
			// communicating with nested blocks, e.g. for superlatives).
			for (PrimitiveFormula formula : box.env.values()) {
				if (!(formula instanceof VariableFormula))
					continue;
				VariableFormula supportingVar = (VariableFormula) formula;
				addSelectVar(box, select, block, supportingVar, false);
			}

			select.where = block;
			return select;
		}

		private void addSelectVar(Box box, SparqlSelect select, SparqlBlock block, VariableFormula var,
				boolean isAuxiliary) {
			// Check if alrady exists; if so, don't add it again
			for (SparqlSelect.Var oldVar : select.selectVars)
				if (oldVar.var.equals(var))
					return;

			select.selectVars.add(
					new SparqlSelect.Var(var, box.asValuesMap.get(var), null, isAuxiliary,
					descriptionsMap.get(var)));
		}

		// Mutable |head| to make sure it contains a VariableFormula.
		private VariableFormula ensureIsVar(Box box, SparqlBlock block, Ref<PrimitiveFormula> head) {
			VariableFormula headVar;
			if (head.value instanceof VariableFormula) {
				headVar = (VariableFormula) head.value;
			} else {
				headVar = newVar();
				if (head.value != null) {
					// LogInfo.logs("ensureIsVar: %s : %s", headVar,
					// head.value);
					Value value = ((ValueFormula<?>) head.value).value;
					if (value instanceof NumberValue) { // encode as (3 as ?x1)
						// [FILTER doesn't work
						// for isolated numbers]
						box.asValuesMap.put(headVar, Formulas.getString(head.value));
					} else { // encode as (FILTER (?x1 = fb:en.barack_obama))
						addStatement(block, headVar, "=", head.value);
						addEntityStatement(block, headVar);
					}
				}
				head.value = headVar;
			}
			return headVar;
		}

		// Add statement as well as updating the units information.
		private void addOptionalStatement(SparqlBlock block, PrimitiveFormula arg1, String property,
				PrimitiveFormula arg2) {
			addStatement(block, arg1, property, arg2, true);
		}

		private void addStatement(SparqlBlock block, PrimitiveFormula arg1, String property, PrimitiveFormula arg2) {
			addStatement(block, arg1, property, arg2, false);
		}

		private void addStatement(SparqlBlock block, PrimitiveFormula arg1, String property, PrimitiveFormula arg2,
				boolean optional) {
			block.addStatement(arg1, property, arg2, optional);
		}

		private void addEntityStatement(SparqlBlock block, VariableFormula var) {
			// This is dangerous because in the DB, not all entities are
			// necessarily labeled with fb:common.topic
			// addStatement(block, var, FreebaseInfo.TYPE, new ValueFormula(new
			// NameValue(FreebaseInfo.ENTITY)));
			// Only needed when includeEntityNames = true.
			addStatement(block, var, RDF_TYPE, newVar());
		}

		private void updateAsValues(Box box, VariableFormula var, String asValue) {
			box.asValuesMap.put(var, asValue);
		}

		// Main conversion function.
		// head, modifier: SPARQL variables (e.g., ?x13)
		// box:
		// - env: mapping from lambda-DCS variables (e.g., ?city) to SPARQL
		// variables (?x13)
		// - asValuesMap: additional constraints
		private SparqlBlock convert(Formula rawFormula, Ref<PrimitiveFormula> head, Ref<PrimitiveFormula> modifier,
				Box box) {

			// Check binary/unary compatibility
			boolean isNameFormula = (rawFormula instanceof ValueFormula)
					&& (((ValueFormula<?>) rawFormula).value instanceof NameValue);
			// Either binary or unary

			boolean needsBinary = (modifier != null);
			boolean providesBinary = rawFormula instanceof LambdaFormula || rawFormula instanceof ReverseFormula;
			if (!isNameFormula && needsBinary != providesBinary) {
				throw new RuntimeException("Binary/unary mis-match: " + rawFormula + " is "
						+ (providesBinary ? "binary" : "unary") + ", but need " + (needsBinary ? "binary" : "unary"));
			}

			SparqlBlock block = new SparqlBlock();

			if (rawFormula instanceof ValueFormula) { // e.g.,
				// fb:en.barack_obama or
				// (number 3)
				@SuppressWarnings({ "unchecked" })
				ValueFormula<Value> formula = (ValueFormula<Value>) rawFormula;
				if (modifier != null) { // Binary predicate
					if (head.value == null)
						head.value = newVar();
					if (modifier.value == null)
						modifier.value = newVar();
					// Deal with primitive reverses
					// (!fb:people.person.date_of_birth)
					String property = ((NameValue) formula.value).id;
					PrimitiveFormula arg1, arg2;
					if (CanonicalNames.isReverseProperty(property)) {
						arg1 = modifier.value;
						property = property.substring(1);
						arg2 = head.value;
					} else {
						arg1 = head.value;
						arg2 = modifier.value;
					}

					// Annoying logic to deal with dates.
					// If we have
					// ?x fb:people.person.date_of_birth "2003"^xsd:datetime,
					// then create two statements:
					// ?x fb:people.person.date_of_birth ?v
					// ?v = "2003"^xsd:datetime [this needs to be transformed]
					if (!SparqlStatement.isOperator(property)) {
						if (arg2 instanceof ValueFormula) {
							Value value = ((ValueFormula<?>) arg2).value;
							if (value instanceof DateValue) {
								VariableFormula v = newVar();
								addStatement(block, v, "=", arg2);
								arg2 = v;
							}
						}
					}
					addStatement(block, arg1, property, arg2);
				} else { // Unary predicate
					unify(block, head, formula);
				}
			} else if (rawFormula instanceof VariableFormula) {
				VariableFormula var = (VariableFormula) rawFormula;
				PrimitiveFormula value = box.env.get(var);
				if (value == null)
					throw new RuntimeException("Unbound variable: " + var + ", env = " + box.env);
				unify(block, head, value);
			} else if (rawFormula instanceof NotFormula) {
				NotFormula formula = (NotFormula) rawFormula;
				block.add(new SparqlNot(convert(formula.child, head, null, box)));
			} else if (rawFormula instanceof MergeFormula) {
				MergeFormula formula = (MergeFormula) rawFormula;
				switch (formula.mode) {
				case and:
					block.add(convert(formula.child1, head, null, box));
					block.add(convert(formula.child2, head, null, box));
					break;
				case or:
					SparqlUnion union = new SparqlUnion();
					ensureIsVar(box, block, head);
					union.add(convert(formula.child1, head, null, box));
					union.add(convert(formula.child2, head, null, box));
					block.add(union);
					break;
				default:
					throw new RuntimeException("Unhandled mode: " + formula.mode);
				}
			} else if (rawFormula instanceof JoinFormula) {
				// Join
				JoinFormula formula = (JoinFormula) rawFormula;
				Ref<PrimitiveFormula> intermediate = new Ref<PrimitiveFormula>();
				block.add(convert(formula.child, intermediate, null, box));
				block.add(convert(formula.relation, head, intermediate, box));
			} else if (rawFormula instanceof ReverseFormula) {
				// Reverse
				ReverseFormula formula = (ReverseFormula) rawFormula;
				block.add(convert(formula.child, modifier, head, box)); // Switch
				// modifier
				// and
				// head
			} else if (rawFormula instanceof LambdaFormula) {
				// Lambda (new environment, same scope)
				LambdaFormula formula = (LambdaFormula) rawFormula;
				if (modifier.value == null)
					modifier.value = newVar();
				Box newBox = createNewBox(formula.body, box); // Create new
				// environment
				newBox.env.put(new VariableFormula(formula.var), modifier.value); // Map
				// variable
				// to
				// modifier
				block.add(convert(formula.body, head, null, newBox));
				// Place pragmatic constraint that head != modifier (for
				// symmetric relations like spouse)
				block.addStatement(head.value, "!=", modifier.value, false);
				returnAsValuesMap(box, newBox);
			} else if (rawFormula instanceof MarkFormula) {
				// Mark (new environment, same scope)
				MarkFormula formula = (MarkFormula) rawFormula;
				if (head.value == null)
					head.value = newVar();
				Box newBox = createNewBox(formula.body, box); // Create new
				// environment
				newBox.env.put(new VariableFormula(formula.var), head.value); // Map
				// variable
				// to
				// head
				// (ONLY
				// difference
				// with
				// lambda)
				block.add(convert(formula.body, head, null, newBox));
				returnAsValuesMap(box, newBox);
			} else if (rawFormula instanceof SuperlativeFormula) {
				// Superlative (new environment, close scope)
				SuperlativeFormula formula = (SuperlativeFormula) rawFormula;

				int rank = Formulas.getInt(formula.rank);
				int count = Formulas.getInt(formula.count);

				boolean useOrderBy = rank != 1 || count != 1;
				boolean isMax = formula.mode == SuperlativeFormula.Mode.argmax;
				if (useOrderBy) {
					// Method 1: use ORDER BY
					// + can deal with offset and limit
					// - but can't be nested
					// - doesn't handle ties at the top
					// Recurse
					Box newBox = createNewBox(formula.head, box); // Create new
					// environment
					SparqlBlock newBlock = convert(formula.head, head, null, newBox);
					Ref<PrimitiveFormula> degree = new Ref<PrimitiveFormula>();
					newBlock.add(convert(formula.relation, head, degree, newBox));

					// Apply the aggregation operation
					VariableFormula degreeVar = ensureIsVar(box, block, degree);

					// Force |degreeVar| to be selected as a variable.
					box.env.put(new VariableFormula("degree"), degreeVar);
					newBox.env.put(new VariableFormula("degree"), degreeVar);

					SparqlSelect select = closeExistentialScope(newBlock, head, newBox);
					select.sortVars.add(isMax ? new VariableFormula(applyVar("DESC", degreeVar)) : degreeVar);
					select.offset = rank - 1;
					select.limit = count;
					block.add(select);
				} else {
					// Method 2: use MAX
					// - can't deal with offset and limit
					// + can be nested
					// + handles ties at the top
					// (argmax 1 1 h r) ==> (h (r (mark degree (max ((reverse r)
					// e)))))
					AggregateFormula.Mode mode = isMax ? AggregateFormula.Mode.max : AggregateFormula.Mode.min;
					Formula best = new MarkFormula("degree", new AggregateFormula(mode,
							new JoinFormula(new ReverseFormula(formula.relation), formula.head)));
					Formula transformed = new MergeFormula(MergeFormula.Mode.and, formula.head,
							new JoinFormula(formula.relation, best));
					block.add(convert(transformed, head, null, box));
				}
			} else if (rawFormula instanceof AggregateFormula) {
				// Aggregate (new environment, close scope)
				AggregateFormula formula = (AggregateFormula) rawFormula;
				ensureIsVar(box, block, head);

				// Recurse
				Box newBox = createNewBox(formula.child, box);
				// Create new environment
				Ref<PrimitiveFormula> newHead = new Ref<PrimitiveFormula>(newVar());
				// Stores the aggregated value
				SparqlBlock newBlock = convert(formula.child, newHead, null, newBox);

				VariableFormula var = (VariableFormula) head.value; // e.g., ?x

				// Variable representing the aggregation
				VariableFormula newVar = (VariableFormula) newHead.value;
				// e.g., ?y = COUNT(?x)

				updateAsValues(newBox, var, applyVar(formula.mode.toString(), newVar));
				// ?var AS COUNT(?newVar)
				block.add(closeExistentialScope(newBlock, head, newBox));
			} else if (rawFormula instanceof ArithmeticFormula) {
				// (+ (number 3) (number 5))
				ArithmeticFormula formula = (ArithmeticFormula) rawFormula;
				Ref<PrimitiveFormula> newHead1 = new Ref<PrimitiveFormula>();
				Ref<PrimitiveFormula> newHead2 = new Ref<PrimitiveFormula>();
				block.add(convert(formula.child1, newHead1, null, box));
				block.add(convert(formula.child2, newHead2, null, box));
				if (head.value == null)
					head.value = newVar();
				VariableFormula var = (VariableFormula) head.value;
				PrimitiveFormula var1 = newHead1.value;
				PrimitiveFormula var2 = newHead2.value;
				updateAsValues(box, var, applyOpVar(ArithmeticFormula.modeToString(formula.mode), var1, var2));
			} else {
				throw new RuntimeException("Unhandled formula: " + rawFormula);
			}

			return block;
		}

		// Copy |box|'s |env|, but only keep the variables which are used in
		// |formula| (these are the free variables).
		// This is an important optimization for converting to SPARQL.
		private Box createNewBox(Formula formula, Box box) {
			Box newBox = new Box();
			for (VariableFormula key : box.env.keySet())
				if (Formulas.containsFreeVar(formula, key))
					newBox.env.put(key, box.env.get(key));
			return newBox;
		}

		// Copy asValuesMap constraints from newBox to box.
		// This is for when we create a new environment (newBox), but maintain
		// the same scope,
		// so we don't rely on closeExistentialScope to include the asValuesMap
		// constraints.
		private void returnAsValuesMap(Box box, Box newBox) {
			for (Map.Entry<VariableFormula, String> e : newBox.asValuesMap.entrySet()) {
				if (box.asValuesMap.containsKey(e.getKey()))
					throw new RuntimeException("Copying asValuesMap involves overwriting: " + box + " <- " + newBox);
				box.asValuesMap.put(e.getKey(), e.getValue());
			}
		}

		private void unify(SparqlBlock block, Ref<PrimitiveFormula> head, PrimitiveFormula value) {
			if (head.value == null) {
				// |head| is not set, just use |value|.
				head.value = value;
			} else {
				// |head| is already set, so add a constraint that it equals
				// |value|.
				// This happens when the logical form is just a single entity
				// (e.g., fb:en.barack_obama).
				addStatement(block, head.value, "=", value);
				if (head.value instanceof VariableFormula && value instanceof ValueFormula
						&& ((ValueFormula<?>) value).value instanceof NameValue)
					addEntityStatement(block, (VariableFormula) head.value);
			}
		}

		// Helper functions
		private String applyVar(String func, VariableFormula var) {
			return applyVar(func, var.name);
		}

		private String applyVar(String func, String var) {
			if (func.equals("count"))
				var = "DISTINCT " + var;
			return func + "(" + var + ")";
		}

		private String applyOpVar(String func, PrimitiveFormula var1, PrimitiveFormula var2) {
			return '(' + Formulas.getString(var1) + ' ' + func + ' ' + Formulas.getString(var2) + ')';
		}

		private VariableFormula newVar() {
			numVars++;
			return new VariableFormula("?x" + numVars);
		}
	}

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

    @Override
	public Response execute(Formula formula, ContextValue context) {
		if (formula instanceof JoinFormula) {
			JoinFormula jf = (JoinFormula) formula;
			if (jf.relation instanceof ValueFormula) {
				ValueFormula<?> rootLeft = (ValueFormula<?>) jf.relation;
				if (rootLeft.value instanceof NameValue) {
					String rootName = ((NameValue) (rootLeft.value)).id;

					if (rootName.equals("tt:root.question.value")) {
						SparqlConverter converter = new SparqlConverter(jf.child);
						SparqlSelect converted = converter.getQuery();
						StringValue sparqlString = new StringValue(converted.toString());

						Formula reduced = new JoinFormula(rootLeft, new ValueFormula<>(sparqlString));
						StringValue treeString = new StringValue(reduced.toLispTree().toString());
						return new Response(treeString);
					}
				}
			}
		}

        Formula reduced = this.reduce(formula);
        Formula etaReduced = this.etaReduce(reduced);
        Formula alphaNormalized = this.alphaNormalize(etaReduced);
        LispTree tree = alphaNormalized.toLispTree();
        StringValue treeString = new StringValue(tree.toString());
        return new Response(treeString);
  }
}
