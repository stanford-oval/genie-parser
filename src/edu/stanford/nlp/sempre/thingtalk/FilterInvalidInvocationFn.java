package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;

public class FilterInvalidInvocationFn extends SemanticFn {
	private static boolean valueOk(Value value) {
		if (!(value instanceof ParametricValue))
			return true;

		ParametricValue pv = (ParametricValue) value;
		for (ParamValue param : pv.params) {
			if (!param.name.type.equals(pv.name.getArgType(param.name.argname)))
				return false;
		}

		return true;
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
