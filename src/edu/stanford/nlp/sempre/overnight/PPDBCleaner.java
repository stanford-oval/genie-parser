package edu.stanford.nlp.sempre.overnight;

import java.io.*;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.sempre.*;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Utils;
import fig.exec.Execution;

public class PPDBCleaner implements Runnable {
	@Option
	public String inputFile;

	@Option
	public String outputFile;

	@Option
	public boolean useDataset;

	@Option
	public String dataset;

	private static final Pattern NUMBER = Pattern.compile("^[0-9.,/\\-]+$");

	private final Set<String> words = new HashSet<>();

	private void buildWordsSet() {
		AbstractDataset dataset = (AbstractDataset) Utils.newInstanceHard(this.dataset);

		for (Example ex : dataset.examples("train")) {
			words.addAll(ex.getTokens());
			words.addAll(ex.getLemmaTokens());
		}
	}

	@Override
	public void run() {
		if (useDataset)
			buildWordsSet();

    try (BufferedReader reader = IOUtils.readerFromString(inputFile)) {
			try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)))) {
			String line;
			while ((line = reader.readLine()) != null) {
					String[] tokens = line.split("\\|\\|\\|");
					String first = tokens[1].trim();
					String second = tokens[2].trim();
					// ppdb includes a large amount of entries that are just numbers with different
					// formatting
					// while they are obviously paraphrases, we definitely don't want them,
					// because CoreNLP parses numbers better than ppdb can ever hope to
					if (NUMBER.matcher(first).matches() && NUMBER.matcher(second).matches())
						continue;
					String stemmedFirst = LanguageInfo.LanguageUtils.stem(first);
					String stemmedSecond = LanguageInfo.LanguageUtils.stem(second);
					if (useDataset && !words.contains(first) && !words.contains(second) && !words.contains(stemmedFirst)
							&& !words.contains(stemmedSecond))
						continue;
					if (first.length() > 0 && second.length() > 0) {
						writer.write(first);
						writer.write("\t");
						writer.write(second);
						writer.write("\n");
					}
					if (stemmedFirst.length() > 0 && stemmedSecond.length() > 0) {
						writer.write(stemmedFirst);
						writer.write("\t");
						writer.write(stemmedSecond);
						writer.write("\n");
					}
				}
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) {
		LogInfo.writeToStdout = true;
		Execution.run(args, "Main", new PPDBCleaner(), Master.getOptionsParser());
	}
}
