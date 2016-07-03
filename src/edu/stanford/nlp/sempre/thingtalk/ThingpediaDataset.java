package edu.stanford.nlp.sempre.thingtalk;

import java.io.IOException;
import java.sql.*;
import java.util.List;

import com.fasterxml.jackson.core.type.TypeReference;

import edu.stanford.nlp.sempre.*;
import fig.basic.LogInfo;

public class ThingpediaDataset extends AbstractDataset {
	private final ThingpediaDatabase dataSource;

	private static final String CANONICAL_QUERY = "select dsc.canonical, ds.kind, dsc.name, dsc.channel_type, dsc.argnames,dsc.types "
			+ "from device_schema_channels dsc join device_schema ds on ds.id = dsc.schema_id and dsc.version = ds.developer_version "
			+ "where canonical is not null and channel_type = 'action' and ds.kind_type <> 'primary'";
	private static final String FULL_EXAMPLE_QUERY = "select id, utterance, target_json from example_utterances where not is_base";

	public ThingpediaDataset() {
		dataSource = ThingpediaDatabase.getSingleton();
	}

	private void readCanonicals(Connection con, int maxExamples, List<Example> examples) throws SQLException {
		try (Statement stmt = con.createStatement()) {
			try (ResultSet set = stmt.executeQuery(CANONICAL_QUERY)) {
				TypeReference<List<String>> typeRef = new TypeReference<List<String>>() {
				};

				while (set.next() && examples.size() < maxExamples) {
					String canonical = set.getString(1);
					String kind = set.getString(2);
					String name = set.getString(3);
					String channelType = set.getString(4);
					List<String> argnames = Json.readValueHard(set.getString(5), typeRef);
					List<String> argtypes = Json.readValueHard(set.getString(6), typeRef);
					ActionValue actionValue = ThingTalk.actParam(new ChannelNameValue(kind, name, argnames, argtypes));
					Value targetValue = ThingTalk.jsonOut(actionValue);

					Example ex = new Example.Builder()
							.setId("canonical_" + kind + "_" + name)
							.setUtterance(canonical)
							.setTargetValue(targetValue)
							.createExample();

					addOneExample(ex, maxExamples, examples);
				}
			}
		}
	}

	private void readFullExamples(Connection con, int maxExamples, List<Example> examples) throws SQLException {
		try (Statement stmt = con.createStatement()) {
			try (ResultSet set = stmt.executeQuery(FULL_EXAMPLE_QUERY)) {

				while (set.next() && examples.size() < maxExamples) {
					int id = set.getInt(1);
					String utterance = set.getString(2);
					String targetJson = set.getString(3);
					Value targetValue = new StringValue(targetJson);

					Example ex = new Example.Builder()
							.setId("full_" + Integer.toString(id))
							.setUtterance(utterance)
							.setTargetValue(targetValue)
							.createExample();

					addOneExample(ex, maxExamples, examples);
				}
			}
		}
	}

	@Override
	public void read() throws IOException {
		LogInfo.begin_track_printAll("ThingpediaDataset.read");

		// assume all examples are train for now
		int maxExamples = getMaxExamplesForGroup("train");
		List<Example> examples = getOrCreateGroup("train");
		
		try (Connection con = dataSource.getConnection()) {
			// we initially train with just the canonical forms
			// this is to "bias" the learner towards learning actions with
			// parameters
			// if we don't do that, with true examples the correct parse
			// always falls off the beam and we don't learn at all
			readCanonicals(con, maxExamples, examples);
			readFullExamples(con, maxExamples, examples);
		} catch (SQLException e) {
			throw new IOException(e);
		}

		if (Dataset.opts.splitDevFromTrain)
			splitDevFromTrain();
		collectStats();

		LogInfo.end_track();
	}
}
