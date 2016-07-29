package edu.stanford.nlp.sempre;

import java.util.*;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.type.TypeReference;
import com.google.common.collect.Lists;

import fig.basic.*;

/**
 * A dataset contains a set of examples, which are keyed by group (e.g., train,
 * dev, test).
 *
 * @author Percy Liang
 */
public class Dataset extends AbstractDataset {
  public static class Options {
    @Option(gloss = "Paths to read input files (format: <group>:<file>)")
    public ArrayList<Pair<String, String>> inPaths = new ArrayList<>();
    @Option(gloss = "Maximum number of examples to read")
    public ArrayList<Pair<String, Integer>> maxExamples = new ArrayList<>();

    // Training file gets split into:
    // |  trainFrac  -->  |           | <-- devFrac |
    @Option(gloss = "Fraction of trainExamples (from the beginning) to keep for training")
    public double trainFrac = 1;
    @Option(gloss = "Fraction of trainExamples (from the end) to keep for development")
    public double devFrac = 0;
    @Option(gloss = "Used to randomly divide training examples")
    public Random splitRandom = new Random(1);
    @Option(gloss = "whether to split dev from train")
    public boolean splitDevFromTrain = true;

    @Option(gloss = "Only keep examples which have at most this number of tokens")
    public int maxTokens = Integer.MAX_VALUE;
  }

  public static Options opts = new Options();

	/** For JSON. */
	static class GroupInfo {
		@JsonProperty
		final String group;
		@JsonProperty
		final List<Example> examples;
		String path; // Optional, used if this was read from a path.

		@JsonCreator
		public GroupInfo(@JsonProperty("group") String group, @JsonProperty("examples") List<Example> examples) {
			this.group = group;
			this.examples = examples;
		}
	}

	/** For JSON. */
	@JsonProperty("groups")
	public List<GroupInfo> getAllGroupInfos() {
		List<GroupInfo> all = Lists.newArrayList();
		for (Map.Entry<String, List<Example>> entry : groupEntries())
			all.add(new GroupInfo(entry.getKey(), entry.getValue()));
		return all;
	}

	/** For JSON. */
	// Allows us to creates dataset from arbitrary JSON, not requiring a
	// path from which to read.
	@JsonCreator
	public static Dataset fromGroupInfos(@JsonProperty("groups") List<GroupInfo> groups) {
		Dataset d = new Dataset();
		d.readFromGroupInfos(groups);
		return d;
	}

	@Override
	public void read() {
		readFromPathPairs(opts.inPaths);
	}

	public void readFromPathPairs(List<Pair<String, String>> pathPairs) {
		// Try to detect whether we need JSON.
		for (Pair<String, String> pathPair : pathPairs) {
			if (pathPair.getSecond().endsWith(".json")) {
				readJsonFromPathPairs(pathPairs);
				return;
			}
		}

		readLispTreeFromPathPairs(pathPairs);
	}

	private void readJsonFromPathPairs(List<Pair<String, String>> pathPairs) {
		List<GroupInfo> groups = Lists.newArrayListWithCapacity(pathPairs.size());
		for (Pair<String, String> pathPair : pathPairs) {
			String group = pathPair.getFirst();
			String path = pathPair.getSecond();
			List<Example> examples = Json.readValueHard(IOUtils.openInHard(path), new TypeReference<List<Example>>() {
			});
			GroupInfo gi = new GroupInfo(group, examples);
			gi.path = path;
			groups.add(gi);
		}
		readFromGroupInfos(groups);
	}

	private void readFromGroupInfos(List<GroupInfo> groupInfos) {
		LogInfo.begin_track_printAll("Dataset.read");

		for (GroupInfo groupInfo : groupInfos) {
			int maxExamples = getMaxExamplesForGroup(groupInfo.group);
			List<Example> examples = getOrCreateGroup(groupInfo.group);
			readHelper(groupInfo.examples, maxExamples, examples, groupInfo.path);
		}
		if (opts.splitDevFromTrain)
			splitDevFromTrain();
		collectStats();

		LogInfo.end_track();
	}

	private void readHelper(List<Example> incoming, int maxExamples, List<Example> examples, String path) {
		int i = 0;
		for (Example ex : incoming) {
			if (examples.size() >= maxExamples)
				return;

			// Specify a default id if it doesn't exist
			if (ex.id == null) {
				String id = (path != null ? path : "<nopath>") + ":" + i;
				ex = new Example.Builder().withExample(ex).setId(id).createExample();
			}

			addOneExample(ex, maxExamples, examples);
		}
	}

	private void readLispTreeFromPathPairs(List<Pair<String, String>> pathPairs) {
		LogInfo.begin_track_printAll("Dataset.read");
		for (Pair<String, String> pathPair : pathPairs) {
			String group = pathPair.getFirst();
			String path = pathPair.getSecond();
			int maxExamples = getMaxExamplesForGroup(group);
			List<Example> examples = getOrCreateGroup(group);
			readLispTreeHelper(path, maxExamples, examples);
		}
		if (opts.splitDevFromTrain)
			splitDevFromTrain();
		LogInfo.end_track();
	}

	private void readLispTreeHelper(String path, int maxExamples, List<Example> examples) {
		if (examples.size() >= maxExamples)
			return;
		LogInfo.begin_track("Reading %s", path);

		Iterator<LispTree> trees = LispTree.proto.parseFromFile(path);
		int n = 0;
		while (examples.size() < maxExamples && trees.hasNext()) {
			// Format: (example (id ...) (utterance ...) (targetFormula ...)
			// (targetValue ...))
			LispTree tree = trees.next();
			if (tree.children.size() < 2 && !"example".equals(tree.child(0).value))
				throw new RuntimeException("Invalid example: " + tree);

			// Specify a default id if it doesn't exist
			Example ex = Example.fromLispTree(tree, path + ":" + n);
			n++;

			addOneExample(ex, maxExamples, examples);
		}
		LogInfo.end_track();
	}
}
