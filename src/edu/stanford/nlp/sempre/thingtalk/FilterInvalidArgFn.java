package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class FilterInvalidArgFn extends SemanticFn {

	private static boolean typeOk(String have, String want) {
		if (have.equals(want))
			return true;

		// a mistake in the naming that is in too many places
		// to fix now
		if (have.equals("Bool") && want.equals("Boolean"))
			return true;

		// FIXME be stricter in handling measures
		// (not a problem for now because we only parse temperatures)
		if (have.equals("Measure") && want.startsWith("Measure("))
			return true;

		// time gets converted to String by Sabrina
		if (have.equals("Time") && want.equals("String"))
			return true;

		// String and Picture are the same type for compat with
		// type annotations that were written before Picture existed
		if ((have.equals("String") && want.equals("Picture")) ||
				(have.equals("Picture") && want.equals("String")))
			return true;

		return false;
	}

	private static boolean valueOk(Value value) {
		if (!(value instanceof ParamValue))
			return true;

		ParamValue pv = (ParamValue) value;
		return typeOk(pv.tt_type, pv.name.type);
	}

	@Override
	public DerivationStream call(Example ex, Callable c) {
		return new SingleDerivationStream() {
			@Override
			public Derivation createDerivation() {
				Derivation child = c.child(0);
				if (child.getValue() != null && !valueOk(child.getValue()))
					return null;

				return new Derivation.Builder()
						.withCallable(c)
						.withFormulaFrom(child)
						.createDerivation();
			}
		};
	}

}
