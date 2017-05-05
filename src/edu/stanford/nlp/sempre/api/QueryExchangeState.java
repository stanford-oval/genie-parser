package edu.stanford.nlp.sempre.api;

import static fig.basic.LogInfo.logs;

import java.io.IOException;
import java.util.*;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.Derivation.Cacheability;
import edu.stanford.nlp.sempre.thingtalk.ArgFilterHelpers;
import edu.stanford.nlp.sempre.thingtalk.ThingTalk;
import fig.basic.LogInfo;

class QueryExchangeState extends AbstractHttpExchangeState {
  private final APIServer server;

  static private final int MAX_ITEMS = 5;

  private final String sessionId;

  public QueryExchangeState(APIServer server, HttpExchange exchange) {
    super(exchange);
    this.server = server;

    if (reqParams.containsKey("sessionId"))
      sessionId = reqParams.get("sessionId");
    else
      sessionId = SecureIdentifiers.getId();
  }

  private Map<String, Object> makeJson(List<Derivation> response, int nItems, boolean longResponse) {
    Map<String, Object> json = new HashMap<>();
    List<Object> items = new ArrayList<>();
    json.put("sessionId", sessionId);
    json.put("candidates", items);

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
      if (longResponse)
        item.put("canonical", deriv.canonicalUtterance);
      items.add(item);
    }

    return json;
  }

  private NumberValue findNumber(Example ex, boolean withUnit) {
    LanguageInfo info = ex.languageInfo;
    try {
      for (int i = 0; i < ex.numTokens(); i++) {
        if (info.nerTags.get(i).equals("NUMBER")) {
          NumberValue value = NumberValue.parseNumber(info.nerValues.get(i));
          if (!withUnit)
            return value;
          if (i < ex.numTokens() - 1) {
            String unit = ex.token(i + 1);
            unit = ArgFilterHelpers.getUnitCaseless(unit);
            if (unit != null) {
              value = new NumberValue(value.value, unit);
              return value;
            }
          }
          return value;
        }
      }
    } catch (NumberFormatException e) {
      LogInfo.warnings("NumberFn: Cannot convert NerSpan to a number");
    }

    return null;
  }

  private List<Derivation> handleUtterance(Session session, LanguageContext language, String query, String expect) {
    session.updateContext();

    // Create example
    Example.Builder b = new Example.Builder();
    b.setId("session:" + session.id);
    b.setUtterance(query);
    b.setContext(session.context);
    Example ex = b.createExample();
    ex.preprocess(language.analyzer);

    Value hackAnswer = null;
    if (expect != null) {
      if (expect.equals("Number")) {
        hackAnswer = findNumber(ex, false);
      } else if (expect.startsWith("Measure(")) {
        hackAnswer = findNumber(ex, true);
      }
    }

    // try from cache
    List<Derivation> derivations = language.cache.hit(query);
    if (APIServer.opts.verbose >= 3) {
      if (derivations != null)
        logs("cache hit");
      else
        logs("cache miss");
    }

    // Parse!
    if (derivations == null) {
      language.parser.parse(language.params, ex, false);
      derivations = ex.getPredDerivations();
      language.cache.store(query, derivations);
    } else {
      ex.predDerivations = derivations;
    }

    session.lastEx = ex;
    session.updateContext(ex, 1);

    // now put the hack answer if we have
    if (hackAnswer != null) {
      StringValue hackAnswerJson = (StringValue)ThingTalk.jsonOut(ThingTalk.ansForm(hackAnswer));

      Derivation deriv = new Derivation.Builder().canonicalUtterance(query).score(Double.POSITIVE_INFINITY).prob(1.0)
              .value(hackAnswerJson).meetCache(Cacheability.NON_DETERMINISTIC).createDerivation();
      derivations = new ArrayList<>(derivations);
      derivations.add(0, deriv);
    } else {
      // now try the exact match, and if it succeeds replace the choice in front
      String exactMatch = language.exactMatch.hit(ex);
      if (exactMatch != null) {
        Derivation deriv = new Derivation.Builder().canonicalUtterance(query).score(Double.POSITIVE_INFINITY).prob(1.0)
                .value(new StringValue(exactMatch)).meetCache(Cacheability.NON_DETERMINISTIC).createDerivation();

        // make a copy of the derivation list, so that we don't put
        // semi-garbage in the query cache or later when trying to online
        // learn
        derivations = new ArrayList<>(derivations);
        derivations.add(0, deriv);
      }
    }

    return derivations;
  }

  @Override
  protected void doHandle() throws IOException {
    String localeTag = reqParams.get("locale");
    LanguageContext language = localeToLanguage(server.langs, localeTag);

    String query = reqParams.get("q");

    int exitStatus;
    List<Derivation> derivations = null;
    Exception error = null;
    int limit = 0;
    String strLongResponse = reqParams.get("long");
    boolean longResponse = "1".equals(strLongResponse);

    // From ValueCategory on the client
    String expect = reqParams.get("expect");

    if(expect != null)
      logs("expect : " + expect);

    try {
      if (query == null)
        throw new IllegalArgumentException("Missing query");

      if (longResponse) {
        limit = Integer.MAX_VALUE;
      } else {
        String limitStr = reqParams.get("limit");
        if (limitStr != null)
          limit = Integer.valueOf(limitStr);
        else
          limit = MAX_ITEMS;
      }

      Session session = this.server.getSession(sessionId);
      synchronized (session) {
        if (session.lang != null && !session.lang.equals(language.tag))
          throw new IllegalArgumentException("Cannot change the language of an existing session");
        session.lang = language.tag;
        session.remoteHost = remoteHost;

        derivations = handleUtterance(session, language, query, reqParams.get("expect"));
        server.logUtterance(language.tag, query);
      }
      exitStatus = 200;
    } catch (IllegalArgumentException | IllegalStateException e) {
      exitStatus = 400;
      error = e;
    } catch (Exception e) {
      exitStatus = 500;
      error = e;
      e.printStackTrace();
    }

    if (error != null)
      returnError(exitStatus, error, sessionId);
    else
      returnJson(exitStatus, makeJson(derivations, limit, longResponse));
  }
}
