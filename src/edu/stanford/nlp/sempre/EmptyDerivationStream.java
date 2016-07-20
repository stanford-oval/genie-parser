package edu.stanford.nlp.sempre;

import java.io.IOException;

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
	public void close() throws IOException {
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
