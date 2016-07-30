package edu.stanford.nlp.sempre;

public final class EmptyDerivationStream implements DerivationStream {
	@Override
	public boolean hasNext() {
		return false;
	}

	@Override
	public Derivation next() {
		return null;
	}

	@Override
	public Derivation peek() {
		return null;
	}

	@Override
	public int estimatedSize() {
		return 0;
	}

}
