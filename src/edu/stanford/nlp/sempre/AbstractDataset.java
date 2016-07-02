package edu.stanford.nlp.sempre;

import java.io.File;
import java.io.IOException;
import java.util.*;

import com.fasterxml.jackson.core.type.TypeReference;

import fig.basic.*;
import fig.exec.Execution;
import fig.prob.SampleUtils;

public abstract class AbstractDataset {

	// Group id -> examples in that group
	private LinkedHashMap<String, List<Example>> allExamples = new LinkedHashMap<>();

	// General statistics about the examples.
	private final HashSet<String> tokenTypes = new HashSet<>();
	private final StatFig numTokensFig = new StatFig(); // For each example,
														// number of tokens

	public Set<String> groups() {
		return allExamples.keySet();
	}

	protected Collection<Map.Entry<String, List<Example>>> groupEntries() {
		return allExamples.entrySet();
	}

	public List<Example> examples(String group) {
		return allExamples.get(group);
	}

	public abstract void read() throws IOException;

	protected List<Example> getOrCreateGroup(String group) {
		List<Example> examples = allExamples.get(group);
		if (examples == null)
			allExamples.put(group, examples = new ArrayList<>());
		return examples;
	}

	protected void splitDevFromTrain() {
		// Split original training examples randomly into train and dev.
		List<Example> origTrainExamples = allExamples.get("train");
		if (origTrainExamples != null) {
			int split1 = (int) (Dataset.opts.trainFrac * origTrainExamples.size());
			int split2 = (int) ((1 - Dataset.opts.devFrac) * origTrainExamples.size());
			int[] perm = SampleUtils.samplePermutation(Dataset.opts.splitRandom, origTrainExamples.size());

			List<Example> trainExamples = new ArrayList<>();
			allExamples.put("train", trainExamples);
			List<Example> devExamples = allExamples.get("dev");
			if (devExamples == null) {
				// Preserve order
				LinkedHashMap<String, List<Example>> newAllExamples = new LinkedHashMap<>();
				for (Map.Entry<String, List<Example>> entry : allExamples.entrySet()) {
					newAllExamples.put(entry.getKey(), entry.getValue());
					if (entry.getKey().equals("train"))
						newAllExamples.put("dev", devExamples = new ArrayList<>());
				}
				allExamples = newAllExamples;
			}
			for (int i = 0; i < split1; i++)
				trainExamples.add(origTrainExamples.get(perm[i]));
			for (int i = split2; i < origTrainExamples.size(); i++)
				devExamples.add(origTrainExamples.get(perm[i]));
		}
	}

	protected void addOneExample(Example ex, int maxExamples, List<Example> examples) {
		if (examples.size() >= maxExamples)
			return;

		ex.preprocess();

		// Skip example if too long
		if (ex.numTokens() > Dataset.opts.maxTokens)
			return;

		LogInfo.logs("Example %s (%d): %s => %s", ex.id, examples.size(), ex.getTokens(), ex.targetValue);

		examples.add(ex);
		numTokensFig.add(ex.numTokens());
		for (String token : ex.getTokens())
			tokenTypes.add(token);
	}

	protected void collectStats() {
		LogInfo.begin_track_printAll("Dataset stats");
		Execution.putLogRec("numTokenTypes", tokenTypes.size());
		Execution.putLogRec("numTokensPerExample", numTokensFig);
		for (Map.Entry<String, List<Example>> e : allExamples.entrySet())
			Execution.putLogRec("numExamples." + e.getKey(), e.getValue().size());
		LogInfo.end_track();
	}

	protected static int getMaxExamplesForGroup(String group) {
		int maxExamples = Integer.MAX_VALUE;
		for (Pair<String, Integer> maxPair : Dataset.opts.maxExamples)
			if (maxPair.getFirst().equals(group))
				maxExamples = maxPair.getSecond();
		return maxExamples;
	}

	public static void appendExampleToFile(String path, Example ex) {
		// JSON is an annoying format because we can't just append.
		// So currently we have to read the entire file in and write it out.
		List<Example> examples;
		if (new File(path).exists()) {
			examples = Json.readValueHard(IOUtils.openInHard(path), new TypeReference<List<Example>>() {
			});
		} else {
			examples = new ArrayList<>();
		}
		examples.add(ex);
		Json.prettyWriteValueHard(new File(path), examples);
	}
}
