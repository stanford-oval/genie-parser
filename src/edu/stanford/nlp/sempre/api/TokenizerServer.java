package edu.stanford.nlp.sempre.api;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.channels.ServerSocketChannel;
import java.util.*;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.google.common.collect.Lists;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import edu.stanford.nlp.sempre.thingtalk.seq2seq.Seq2SeqTokenizer;
import fig.basic.Option;
import fig.exec.Execution;

public class TokenizerServer implements Runnable {
  public static class Options {
    @Option
    public List<String> languages = Lists.newArrayList("en", "it", "zh");
  };

  public static Options opts = new Options();

  private final ObjectMapper object = new ObjectMapper();
  private ServerSocket server;
  private final Map<String, LanguageAnalyzer> analyzers = new HashMap<>();
  private final Map<String, Seq2SeqTokenizer> tokenizers = new HashMap<>();
  private final Executor threadPool = Executors.newFixedThreadPool(2 * Runtime.getRuntime().availableProcessors());

  public static class Input {
    @JsonProperty
    int req;

    @JsonProperty
    String languageTag;

    @JsonProperty
    String utterance;
  }

  public static class Output {
    @JsonProperty
    int req;

    @JsonProperty
    final List<String> tokens = new ArrayList<>();

    @JsonProperty
    final Map<String, Map<String, Object>> values = new HashMap<>();
  }

  private synchronized void writeOutput(Writer outputStream, Output output) {
    ObjectWriter writer = object.writer().withType(Output.class);
    try {
      writer.writeValue(outputStream, output);
      outputStream.append('\n');
      outputStream.flush();
    } catch (IOException e) {
      System.err.println("Failed to write tokenizer output out: " + e.getMessage());
      e.printStackTrace(System.err);
    }
  }

  private void processInput(Writer outputStream, Input input) {
    LanguageAnalyzer analyzer = analyzers.get(input.languageTag);
    Seq2SeqTokenizer tokenizer = tokenizers.get(input.languageTag);

    Output output = new Output();
    output.req = input.req;

    Example ex = new Example.Builder().setUtterance(input.utterance).createExample();
    ex.preprocess(analyzer);

    Seq2SeqTokenizer.Result result = tokenizer.process(ex);

    output.tokens.addAll(result.tokens);

    for (Map.Entry<Seq2SeqTokenizer.Value, List<Integer>> entry : result.entities.entrySet()) {
      Seq2SeqTokenizer.Value entity = entry.getKey();
      String entityType = entity.type;
      for (int entityNum : entry.getValue()) {
        String entityToken = entityType + "_" + entityNum;

        Value entityValue;
        if (entity.value instanceof Double)
          entityValue = new NumberValue((double) entity.value);
        else if (entity.value instanceof String)
          entityValue = new StringValue((String) entity.value);
        else
          entityValue = (Value) entity.value;
        output.values.put(entityToken, entityValue.toJson());
      }
    }

    writeOutput(outputStream, output);
  }

  private void handleConnection(Socket s) {
    try (Socket socket = s) {
      Reader inputStream = new InputStreamReader(socket.getInputStream());
      Writer outputStream = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));

      ObjectReader objReader = object.reader().withType(Input.class);
      JsonParser parser = object.getFactory().createParser(inputStream);

      while (!socket.isClosed()) {
        JsonToken nextToken = parser.nextToken();
        if (nextToken == null) {
          // eof
          break;
        }
        Input next;
        try {
          next = objReader.readValue(parser);
        } catch (JsonProcessingException e) {
          System.err.println("Invalid JSON input: " + e.getMessage());
          e.printStackTrace();
          continue;
        }

        threadPool.execute(() -> processInput(outputStream, next));
      }
    } catch (EOFException e) {
      return;
    } catch (IOException e) {
      System.err.println("IO error on connection: " + e.getMessage());
      e.printStackTrace(System.err);
    }
  }

  @Override
  public void run() {
    for (String lang : opts.languages) {
      analyzers.put(lang, new CoreNLPAnalyzer(lang));
      tokenizers.put(lang, new Seq2SeqTokenizer(lang, false));
    }

    object.getFactory()
        .disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET)
        .disable(JsonParser.Feature.AUTO_CLOSE_SOURCE);

    try {
      ServerSocketChannel socketChannel = (ServerSocketChannel) System.inheritedChannel();
      if (socketChannel != null)
        server = socketChannel.socket();
      else
        server = new ServerSocket(8888);

      while (true) {
        Socket socket = server.accept();
        new Thread(() -> handleConnection(socket)).start();
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new TokenizerServer(), Master.getOptionsParser());
  }
}
