package edu.stanford.nlp.sempre.api;

import static fig.basic.LogInfo.logs;

import java.io.IOException;
import java.util.*;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.api.SecureIdentifiers;

class ExchangeState extends AbstractHttpExchangeState {
  /**
   * 
   */
  private final APIServer server;

  static private final int MAX_ITEMS = 5;

  private final String sessionId;

  public ExchangeState(APIServer server, HttpExchange exchange) {
    super(exchange);
    this.server = server;

    if (reqParams.containsKey("sessionId"))
      sessionId = reqParams.get("sessionId");
    else
      sessionId = SecureIdentifiers.getId();
  }

  private Map<String, Object> makeJson(List<Derivation> response) {
    Map<String, Object> json = new HashMap<>();
    List<Object> items = new ArrayList<>();
    json.put("sessionId", sessionId);
    json.put("candidates", items);

    int nItems = MAX_ITEMS;
    for (Derivation deriv : response) {
      if (nItems == 0)
        break;
      nItems--;

      Map<String, Object> item = new HashMap<>();
      Value value = deriv.getValue();
      if (value == null)
        item.put("answer", null);
      else if (value instanceof StringValue)
        item.put("answer", ((StringValue) value).value);
      else if (value instanceof ListValue)
        item.put("answer", ((StringValue) ((ListValue) value).values.get(0)).value);
      item.put("score", deriv.score);
      item.put("prob", deriv.prob);
      items.add(item);
    }

    return json;
  }

  private Map<String, Object> makeJson(Exception e) {
    Map<String, Object> json = new HashMap<>();
    json.put("sessionId", sessionId);
    json.put("error", e.getMessage());
    return json;
  }

  private List<Derivation> handleUtterance(Session session, LanguageContext language, String query) {
    session.updateContext();

    // Create example
    Example.Builder b = new Example.Builder();
    b.setId("session:" + session.id);
    b.setUtterance(query);
    b.setContext(session.context);
    Example ex = b.createExample();

    ex.preprocess(language.analyzer);

    // Parse!
    language.parser.parse(language.params, ex, false);

    session.lastEx = ex;
    session.updateContext(ex, 1);

    return ex.getPredDerivations();
  }

  @Override
  protected void doHandle() throws IOException {
    String localeTag = reqParams.get("locale");
    LanguageContext language = null;

    if (localeTag != null) {
      String[] splitTag = localeTag.split("[_\\.\\-]");
      // try with language and country
      if (splitTag.length >= 2)
        language = this.server.langs.get(splitTag[0] + "_" + splitTag[1]);
      if (language == null && splitTag.length >= 1)
        language = this.server.langs.get(splitTag[0]);
    }
    // fallback to english if the language is not recognized or
    // locale was not specified
    if (language == null)
      language = this.server.langs.get("en");

    String query = reqParams.get("q");

    int exitStatus;
    List<Derivation> derivations = null;
    Exception error = null;

    try {
      if (query != null) {
        // try from cache
        derivations = language.cache.hit(query);
        if (APIServer.opts.verbose >= 3) {
          if (derivations != null)
            logs("cache hit");
          else
            logs("cache miss");
        }

        if (derivations == null) {
          Session session = this.server.getSession(sessionId);
          synchronized (session) {
            if (session.lang != null && !session.lang.equals(localeTag))
              throw new IllegalArgumentException("Cannot change the language of an existing session");
            session.lang = localeTag;
            session.remoteHost = remoteHost;
            derivations = handleUtterance(session, language, query);
          }
          language.cache.store(query, derivations);
        }
        exitStatus = 200;
      } else {
        exitStatus = 400;
        error = new RuntimeException("Bad Request");
      }
    } catch (Exception e) {
      exitStatus = 500;
      error = e;
      e.printStackTrace();
    }

    if (error != null)
      returnJson(exitStatus, makeJson(error));
    else
      returnJson(exitStatus, makeJson(derivations));
  }
}