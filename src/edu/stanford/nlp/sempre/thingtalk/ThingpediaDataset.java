package edu.stanford.nlp.sempre.thingtalk;

import java.io.IOException;
import java.sql.*;
import java.util.List;

import edu.stanford.nlp.sempre.*;
import fig.basic.LogInfo;

public class ThingpediaDataset extends AbstractDataset {
	private final ThingpediaDatabase dataSource;

	private static final String QUERY = "select id, utterance, target_json from example_utterances where not is_base";

	public ThingpediaDataset() {
		dataSource = ThingpediaDatabase.getSingleton();
	}

	@Override
	public void read() throws IOException {
		LogInfo.begin_track_printAll("ThingpediaDataset.read");

		// assume all examples are train for now
		int maxExamples = getMaxExamplesForGroup("train");
		List<Example> examples = getOrCreateGroup("train");
		
		try (Connection con = dataSource.getConnection()) {
			Statement stmt = con.createStatement();
			ResultSet set = stmt.executeQuery(QUERY);
			
			while (set.next()) {
				int id = set.getInt(1);
				String utterance = set.getString(2);
				String targetJson = set.getString(3);
				Value targetValue = new StringValue(targetJson);
				
				Example ex = new Example.Builder()
						.setId(Integer.toString(id))
						.setUtterance(utterance)
						.setTargetValue(targetValue)
						.createExample();
				
				addOneExample(ex, maxExamples, examples);
			}
		} catch (SQLException e) {
			throw new IOException(e);
		}

		if (Dataset.opts.splitDevFromTrain)
			splitDevFromTrain();
		collectStats();

		LogInfo.end_track();
	}
}
