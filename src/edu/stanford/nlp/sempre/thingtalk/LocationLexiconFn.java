package edu.stanford.nlp.sempre.thingtalk;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import edu.stanford.nlp.sempre.*;

public class LocationLexiconFn extends SemanticFn {

	private final LocationLexicon lexicon;

	public LocationLexiconFn() {
		lexicon = LocationLexicon.get();
	}

	private static <E> boolean all(Collection<E> items, Predicate<E> p) {
		for (E i : items) {
			if (!p.test(i))
				return false;
		}
		return true;
	}

	@Override
	public DerivationStream call(Example ex, Callable c) {
		List<String> nerTags = ex.languageInfo.nerTags.subList(c.getStart(), c.getEnd());
		if (!all(nerTags, t -> t.equals("LOCATION")))
			return new EmptyDerivationStream();

		String phrase = c.childStringValue(0);
		return new LocationDerivationStream(ex, c, lexicon.lookup(phrase), phrase);
	}

	public class LocationDerivationStream extends MultipleDerivationStream {
		private Example ex;
		private Callable callable;
		private Iterator<LocationLexicon.Entry> entries;
		private String phrase;

		public LocationDerivationStream(Example ex, Callable c, Iterator<LocationLexicon.Entry> entries,
				String phrase) {
			this.ex = ex;
			this.callable = c;
			this.entries = entries;
			this.phrase = phrase;
		}

		@Override
		public Derivation createDerivation() {
			if (!entries.hasNext())
				return null;

			LocationLexicon.Entry entry = entries.next();
			Derivation deriv = new Derivation.Builder().withCallable(callable)
					.formula(entry.formula)
					.canonicalUtterance(entry.rawPhrase).type(SemType.entityType)
					.createDerivation();

			// Doesn't generalize, but add it for now, otherwise not separable
			if (FeatureExtractor.containsDomain("lexAlign"))
				deriv.addFeature("lexAlign", phrase + " --- " + entry.formula);

			if (SemanticFn.opts.trackLocalChoices)
				deriv.addLocalChoice("SimpleLexiconFn " + deriv.startEndString(ex.getTokens()) + " " + entry);

			return deriv;
		}
	}
}
