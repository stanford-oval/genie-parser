package edu.stanford.nlp.sempre;

import java.io.IOException;
import java.util.List;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import fig.basic.LogInfo;

// A wrapper for Master that uses stdin/stdout for structured communication
public class StreamAPI {
	private final Master master;
	private final Builder builder;

	public StreamAPI(Master master, Builder builder) {
		this.master = master;
		this.builder = builder;
	}

	private void writeError(JsonGenerator writer, String error) throws IOException {
		writer.writeStartObject();
		writer.writeObjectField("error", error);
		writer.writeEndObject();
		writer.writeRaw("\n");
		writer.flush();
	}

	private List<Derivation> handleUtterance(Session session, String query) {
		session.updateContext();

		// Create example
		Example.Builder b = new Example.Builder();
		b.setId("session:" + session.id);
		b.setUtterance(query);
		b.setContext(session.context);
		Example ex = b.createExample();

		ex.preprocess();

		// Parse!
		builder.parser.parse(builder.params, ex, false);

		return ex.getPredDerivations();
	}

	public void run() {
		JsonFactory factory = new JsonFactory();
		ObjectMapper om = new ObjectMapper(factory);
		try {
			JsonGenerator writer = factory.createGenerator(LogInfo.stdout);

			writer.writeStartObject();
			writer.writeObjectField("status", "Ready");
			writer.writeEndObject();
			writer.writeRaw("\n");
			writer.flush();

			try {
				while (true) {
					String line = LogInfo.stdin.readLine();
					JsonNode node = om.readTree(line);

					if (node.has("command")) {
						String command = node.get("command").asText();

						if ("exit".equals(command))
							break;

						writeError(writer, "Invalid command");
						continue;
					}

					String sessionId = node.get("session").asText();
					String utterance = node.get("utterance").asText();

					Value value;
					try {
						Session session = master.getSession(sessionId);
						List<Derivation> derivations = handleUtterance(session, utterance);

						if (derivations.size() == 0)
							value = null;
						else
							value = derivations.get(0).value;
					} catch (Exception e) {
						e.printStackTrace();
						writeError(writer, e.getMessage());
						continue;
					}

					writer.writeStartObject();
					if (value == null)
						writer.writeObjectField("answer", null);
					else
						writer.writeObjectField("answer", ((StringValue) value).value);
					writer.writeEndObject();
					writer.writeRaw("\n");
					writer.flush();
				}
			} catch (IOException e) {
				writeError(writer, e.getMessage());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}