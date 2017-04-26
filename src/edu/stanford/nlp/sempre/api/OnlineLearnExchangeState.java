package edu.stanford.nlp.sempre.api;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaDataset;
import fig.basic.LogInfo;
import fig.basic.Option;

public class OnlineLearnExchangeState extends AbstractHttpExchangeState {
  public static class Options {
    @Option(gloss = "Probability of storing the example in the test set instead of the train set.")
    public double testProbability = 0.1;
  }

  public static final Options opts = new Options();

  private final APIServer server;
  private final String sessionId;

  public OnlineLearnExchangeState(APIServer server, HttpExchange exchange) {
    super(exchange);
    this.server = server;

    if (reqParams.containsKey("sessionId"))
      sessionId = reqParams.get("sessionId");
    else
      sessionId = null;
  }

  private static final Pattern NAME_REGEX = Pattern.compile("^tt:([^\\.]+)\\.(.+)$");

  private static void extractSchema(List<String> into, Map<?, ?> invocation) {
    if (invocation == null)
      return;

    Object name = invocation.get("name");

    CharSequence fullName;
    if (name instanceof CharSequence)
      fullName = (CharSequence) name;
    else
      fullName = (CharSequence) ((Map<?, ?>) name).get("id");

    Matcher matcher = NAME_REGEX.matcher(fullName);
    if (!matcher.matches())
      return;

    into.add(matcher.group(1));
  }

  @Override
  protected void doHandle() throws IOException {
    try {
      if (sessionId == null)
        throw new IllegalStateException("Missing session ID");

      String localeTag = reqParams.get("locale");
      LanguageContext language = localeToLanguage(server.langs, localeTag);

      String query = reqParams.get("q");
      if (query == null)
        throw new IllegalArgumentException("Missing query");

      String targetJson = reqParams.get("target");
      if (targetJson == null)
        throw new IllegalArgumentException("Missing target");

      // check if the target parses as JSON
      Map<String, Object> parsed;
      try {
        parsed = Json.readMapHard(targetJson);
      } catch (Exception e) {
        throw new IllegalArgumentException("Target is not valid JSON");
      }

      // figure out what schemas are involved in this example
      List<String> schemas = new ArrayList<>();
      try {
        if (parsed.containsKey("rule")) {
          Map<?, ?> rule = (Map<?, ?>) parsed.get("rule");
          extractSchema(schemas, (Map<?, ?>) rule.get("trigger"));
          extractSchema(schemas, (Map<?, ?>) rule.get("query"));
          extractSchema(schemas, (Map<?, ?>) rule.get("action"));
        } else {
          extractSchema(schemas, (Map<?, ?>) parsed.get("trigger"));
          extractSchema(schemas, (Map<?, ?>) parsed.get("query"));
          extractSchema(schemas, (Map<?, ?>) parsed.get("action"));
        }
      } catch (ClassCastException e) {
        throw new IllegalArgumentException("Target is not valid SEMPRE JSON");
      }

      double diceRoll = ThreadLocalRandom.current().nextDouble();
      boolean storeAsTest = diceRoll > (1 - opts.testProbability);

      if (storeAsTest)
        LogInfo.logs("Storing %s as %s in the test set", query, targetJson);
      else
        LogInfo.logs("Learning %s as %s", query, targetJson);

      Session session = server.getSession(sessionId);
      Example ex = null;
      synchronized (session) {
        LogInfo.logs("session.lang %s, language.tag %s", session.lang, language.tag);
        if (session.lang != null && !session.lang.equals(language.tag))
          throw new IllegalArgumentException("Cannot change the language of an existing session");
        session.lang = language.tag;
        session.remoteHost = remoteHost;

        // we only learn in the ML sense if we still have the parsed example...
        if (!storeAsTest && session.lastEx != null && session.lastEx.utterance != null
            && session.lastEx.utterance.equals(query)) {
          ex = session.lastEx;

          ex.targetValue = new StringValue(targetJson);
          language.learner.onlineLearnExample(ex);
        }
      }

      String type;
      if (storeAsTest) {
        type = "test";
      } else {
        // ... but we always save the example in the database, just in case
        // potentially this allows someone to DDOS our server with bad data
        // we just hope the ML model is resilient to that (and it should be)
        type = "online";
      }

      // reuse the CoreNLP analysis if possible
      if (ex != null)
        language.exactMatch.store(ex, targetJson);
      else
        language.exactMatch.store(query, targetJson);

      ThingpediaDataset.storeExample(query, targetJson, language.tag, type, schemas);

      // we would need to remove all entries from the cache that are affected by this learning step
      // (which potentially is all of them)
      // that would mean too many evictions
      // instead, we let the normal cache aging pick it up, and only remove the current utterance,
      // which we know for sure has changed
      language.cache.clear(query);
    } catch(IllegalStateException|IllegalArgumentException e) {
      returnError(400, e, sessionId);
      return;
    } catch(Exception e) {
      returnError(500, e, sessionId);
      e.printStackTrace();
      return;
    }

    returnOk("Learnt successfully", sessionId);
  }

}
