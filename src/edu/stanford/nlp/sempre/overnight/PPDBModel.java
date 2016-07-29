package edu.stanford.nlp.sempre.overnight;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.sempre.LanguageInfo;
import fig.basic.LogInfo;
import fig.basic.Option;

/**
 * PPDBModel extracts and scores paraphrasing featues from derivations.
 * This model is intended to be used with FloatingParser
 *
 * @author Yushi Wang
 */

public final class PPDBModel {
  public static class Options {
    @Option(gloss = "Path to file with alignment table")
    public String ppdbModelPath = "regex/regex-ppdb.txt";

    @Option(gloss = "Using ppdb format")
    public boolean ppdb = true;
  }

  public static Options opts = new Options();

  public static PPDBModel model;

	// we use TreeSet instead of HashSet because this will grow very big
	// (gbs of data) and HashSet has poor memory consumption at large sizes
	private final Set<StringPair> table = new HashSet<>();

  // We should only have one paraphrase model
  public static PPDBModel getSingleton() {
    if (model == null) {
      model = new PPDBModel();
    }
    return model;
  }

  private PPDBModel() {
		loadPPDBModel(opts.ppdbModelPath);
  }

	private static class StringPair {
		private final String one;
		private final String two;

		public StringPair(String one, String two) {
			this.one = one;
			this.two = two;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + one.hashCode();
			result = prime * result + two.hashCode();
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			StringPair other = (StringPair) obj;
			if (!one.equals(other.one))
				return false;
			if (!two.equals(other.two))
				return false;
			return true;
		}
	}

  /**
   * Loading ppdb model from file
   */
	private void loadPPDBModel(String path) {
    LogInfo.begin_track("Loading ppdb model");

		try (BufferedReader reader = IOUtils.getBufferedFileReader(path)) {
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.length() == 0)
					continue;
				if (opts.ppdb) {
					String[] tokens = line.split("\\|\\|\\|");
					String first = tokens[1].trim();
					String second = tokens[2].trim();
					String stemmedFirst = LanguageInfo.LanguageUtils.stem(first);
					String stemmedSecond = LanguageInfo.LanguageUtils.stem(second);

					table.add(new StringPair(first, second));
					if ((!stemmedFirst.equals(first) || !stemmedSecond.equals(second)) &&
							!stemmedFirst.equals(stemmedSecond))
						table.add(new StringPair(stemmedFirst, stemmedSecond));
				} else {
					String[] tokens = line.split("\t");
					table.add(new StringPair(tokens[0], tokens[1]));
				}
			}
		} catch (IOException e) {
			LogInfo.logs("IOException loading ppdb model: %s", e.getMessage());
		}
		LogInfo.logs("ParaphraseUtils.loadPhraseTable: number of entries=%s", table.size());
    LogInfo.end_track();
  }

	public double get(String key, String token) {
		return table.contains(new StringPair(key, token)) ? 1.0 : 0.0;
  }
}
