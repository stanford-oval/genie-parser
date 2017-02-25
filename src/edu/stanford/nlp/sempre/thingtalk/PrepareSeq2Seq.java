package edu.stanford.nlp.sempre.thingtalk;

import java.io.*;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.sempre.*;
import fig.basic.Option;
import fig.exec.Execution;

public class PrepareSeq2Seq implements Runnable {
  public static class Options {
    @Option
    public String languageTag = "en";

    @Option
    public String trainFile = "train.tsv";

    @Option
    public String testFile = "test.tsv";

    @Option
    public String devFile = "dev.tsv";
  }

  public static final Options opts = new Options();

  private PrepareSeq2Seq() {
  }

  private static void writeOutput(Writer writer, Example ex) throws IOException {
    Map<?, ?> json = Json.readMapHard(((StringValue) ex.targetValue).value);

    if (json.containsKey("special"))
      writeSpecial(writer, (Map<?, ?>) json.get("special"));
    else if (json.containsKey("answer"))
      writeAnswer(writer, (Map<?, ?>) json.get("answer"));
    else if (json.containsKey("command"))
      writeCommand(writer, (Map<?, ?>) json.get("command"));
    else if (json.containsKey("rule"))
      writeRule(writer, (Map<?, ?>) json.get("rule"));
    else if (json.containsKey("trigger"))
      writeTopInvocation(writer, "trigger", (Map<?, ?>) json.get("trigger"));
    else if (json.containsKey("query"))
      writeTopInvocation(writer, "query", (Map<?, ?>) json.get("query"));
    else if (json.containsKey("action"))
      writeTopInvocation(writer, "action", (Map<?, ?>) json.get("action"));
  }

  private static void writeSpecial(Writer writer, Map<?, ?> special) throws IOException {
    writer.write("special ");
    String id = (String) special.get("id");
    writer.write(id);
  }

  private static void writeTopInvocation(Writer writer, String invocationType, Map<?, ?> map) throws IOException {
    writer.write(invocationType);
    writer.write(' ');
    writeInvocation(writer, invocationType, map);
  }

  private static void writeInvocation(Writer writer, String invocationType, Map<?, ?> invocation) throws IOException {
    Map<?, ?> name = (Map<?, ?>) invocation.get("name");
    writer.write(name.get("id").toString());
    writer.write(" ");

    List<?> arguments = (List<?>) invocation.get("args");
    boolean first = true;
    for (Object o : arguments) {
      Map<?, ?> arg = (Map<?, ?>) o;
      Map<?, ?> argName = (Map<?, ?>) arg.get("name");
      if (!first)
        writer.write(" ");
      first = false;
      writer.write(argName.get("id").toString());
      writer.write(" ");
      writer.write(arg.get("operator").toString());
      writer.write(" ");
      writeArgument(writer, arg);
    }
  }

  private static void writeRule(Writer writer, Map<?, ?> rule) throws IOException {
    writer.write("rule ");
    if (rule.containsKey("trigger")) {
      writer.write("if ");
      writeInvocation(writer, "trigger", (Map<?, ?>) rule.get("trigger"));
      writer.write(" then ");
    }
    if (rule.containsKey("query")) {
      writeInvocation(writer, "query", (Map<?, ?>) rule.get("query"));
      if (rule.containsKey("action"))
        writer.write(" then ");
    }
    if (rule.containsKey("action")) {
      writeInvocation(writer, "action", (Map<?, ?>) rule.get("action"));
    }
  }

  private static void writeCommand(Writer writer, Map<?, ?> command) throws IOException {
    writer.write("command ");
    writer.write((String) command.get("type"));
    writer.write(" ");

    Map<?, ?> value = (Map<?, ?>) command.get("value");
    if (value.containsKey("value"))
      writer.write(value.get("value").toString());
    else if (value.containsKey("name"))
      writer.write(value.get("name").toString());
    else
      writer.write(value.get("id").toString());
  }

  private static void writeAnswer(Writer writer, Map<?, ?> answer) throws IOException {
    writer.write("answer ");
    writeArgument(writer, answer);
  }

  private static void writeArgument(Writer writer, Map<?, ?> argument) throws IOException {
    String type = (String) argument.get("type");
    Map<?, ?> value = (Map<?, ?>) argument.get("value");
    if (type.startsWith("Entity(")) {
      // FIXME entities
      //writer.write(value.get("value").toString());
      writer.write("ENTITY");
      return;
    }
    switch (type) {
    case "Location":
      String relativeTag = (String) value.get("relativeTag");
      writer.write(relativeTag);
      if (relativeTag.equals("absolute")) {
        // FIXME location
        //writer.write(" ");
        //writer.write(value.get("latitude").toString());
        //writer.write(" ");
        //writer.write(value.get("longitude").toString());
      }
      break;

    case "Boolean":
    case "Bool":
    case "Enum":
      writer.write(value.get("value").toString());
      break;

    case "VarRef":
      writer.write(value.get("id").toString());
      break;

    case "String":
      writer.write("QUOTED_STRING");
      break;

    case "Date":
      writer.write("DATE");
      break;

    case "Time":
      writer.write("TIME");
      break;

    case "Username":
      writer.write("USERNAME");
      break;

    case "Hashtag":
      writer.write("HASHTAG");
      break;

    case "Number":
      writer.write("NUMBER");
      break;

    case "Measure":
      writer.write("NUMBER");
      writer.write(" ");
      writer.write(value.get("unit").toString());
      break;

    case "PhoneNumber":
      writer.write("PHONE_NUMBER");
      break;

    case "EmailAddress":
      writer.write("EMAIL_ADDRESS");
      break;

    case "URL":
      writer.write("URL");
      break;

    default:
      throw new IllegalArgumentException("Invalid value type " + type);
    }
  }

  private static void writeUtterance(Writer writer, Example ex) throws IOException {
    String previousTag = null;
    LanguageInfo utteranceInfo = ex.languageInfo;
    for (int i = 0; i < utteranceInfo.tokens.size(); i++) {
      String current;

      if (utteranceInfo.nerValues.get(i) != null) {
        current = utteranceInfo.nerTags.get(i);
        if (current.equals(previousTag))
          continue;
        previousTag = utteranceInfo.nerTags.get(i);
      } else {
        current = utteranceInfo.tokens.get(i);
        previousTag = null;
      }
      if (i > 0)
        writer.append(' ');
      writer.append(current);
    }
  }

  private static void processGroup(AbstractDataset dataset, String groupName, String fileName) {
    List<Example> group = dataset.examples(groupName);
    if (group == null)
      return;
    try (Writer writer = new BufferedWriter(new FileWriter(fileName))) {
      for (Example ex : group) {
        writeUtterance(writer, ex);
        writer.append('\t');
        writeOutput(writer, ex);
        writer.append('\n');
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void run() {
    AbstractDataset dataset = new ThingpediaDataset();
    try {
      dataset.read();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    processGroup(dataset, "train", opts.trainFile);
    processGroup(dataset, "test", opts.testFile);
    processGroup(dataset, "dev", opts.devFile);
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new PrepareSeq2Seq(), Master.getOptionsParser());
  }
}
