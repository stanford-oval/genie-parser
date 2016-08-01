package edu.stanford.nlp.sempre.api;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.Example;
import edu.stanford.nlp.sempre.Session;
import edu.stanford.nlp.sempre.StringValue;
import fig.basic.LogInfo;

class OnlineLearnExchangeState extends AbstractHttpExchangeState {
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

      Session session = server.getSession(sessionId);
      Example ex;
      synchronized (session) {
        if (session.lang == null || !session.lang.equals(language.tag))
          throw new IllegalArgumentException("Cannot change the language of an existing session");
        session.lang = language.tag;
        session.remoteHost = remoteHost;
        if (session.lastEx == null || session.lastEx.utterance == null || !session.lastEx.utterance.equals(query))
          throw new IllegalStateException("Query is not the last for the session (or the session expired)");
        ex = session.lastEx;

        LogInfo.logs("Learning %s as %s", ex.utterance, targetJson);

        ex.targetValue = new StringValue(targetJson);
        language.learner.onlineLearnExample(ex);
      }

      server.recordOnlineLearnExample(language.tag, query, targetJson);

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
