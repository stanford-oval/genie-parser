package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.*;
import fig.basic.LispTree;
import fig.basic.Option;

import java.sql.SQLException;

/**
 * Uses the ThingpediaLexicon.
 *
 * Example: (rule $ROOT ($PHRASE) (ThingpediaLexiconFn "channel"))
 * 			for triggers, actions or queries
 *
 * 			(rule $ROOT ($PHRASE) (ThingpediaLexiconFn "app"))
 * 			for an application
 *
 * @author Giovanni Campagna
 */
public class ThingpediaLexiconFn extends SemanticFn {
	public static class Options {
		@Option(gloss = "Verbosity level")
		public int verbose = 0;
	}

	public static Options opts = new Options();

	private final ThingpediaLexicon lexicon;

	private enum Mode {
		APP, CHANNEL;
	}

	private Mode mode;

	// Only return entries whose type matches this
	private SemType restrictType = SemType.topType;

	public ThingpediaLexiconFn() {
		lexicon = ThingpediaLexicon.getSingleton();
	}

	@Override
	public void init(LispTree tree) {
		super.init(tree);
		String value = tree.child(1).value;

		// mode
		if (value.equals("app"))
			this.mode = Mode.APP;
		else if (value.equals("channel"))
			this.mode = Mode.CHANNEL;
		else
			throw new RuntimeException("Invalid mode for ThingpediaLexiconFn");
	}

	@Override
	public DerivationStream call(Example ex, Callable c) {
		String phrase = c.childStringValue(0);
		ThingpediaLexicon.EntryStream entries;
		try {
			if (mode == Mode.APP)
				entries = lexicon.lookupApp(phrase);
			else if (mode == Mode.CHANNEL)
				entries = lexicon.lookupChannel(phrase);
			else
				throw new RuntimeException();
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}

		return new ThingpediaDerivationStream(ex, c, entries, phrase);
	}

	public class ThingpediaDerivationStream extends MultipleDerivationStream {
		private Example ex;
		private Callable callable;
		private ThingpediaLexicon.EntryStream entries;
		private String phrase;

		public ThingpediaDerivationStream(Example ex, Callable c, ThingpediaLexicon.EntryStream entries,
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

			ThingpediaLexicon.Entry entry = entries.next();
			FeatureVector features = new FeatureVector();
			Derivation deriv = new Derivation.Builder().withCallable(callable).formula(entry.toFormula())
					.localFeatureVector(features).createDerivation();

			// Doesn't generalize, but add it for now, otherwise not separable
			if (FeatureExtractor.containsDomain("lexAlign"))
				deriv.addFeature("lexAlign", phrase + " --- " + entry.toFormula());

			if (SemanticFn.opts.trackLocalChoices)
				deriv.addLocalChoice("SimpleLexiconFn " + deriv.startEndString(ex.getTokens()) + " " + entry);

			return deriv;
		}
	}

}
