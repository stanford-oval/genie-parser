package edu.stanford.nlp.sempre.api;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.thingtalk.ThingpediaLexicon;
import edu.stanford.nlp.sempre.thingtalk.EntityLexicon;
import fig.basic.LogInfo;

class ClearCacheExchangeState extends AdminHttpExchangeState {
  private final APIServer server;

  public ClearCacheExchangeState(APIServer server, HttpExchange exchange) {
    super(exchange);
    this.server = server;
  }

  @Override
  protected void doHandle() throws IOException {
    if (!checkCredentials())
      return;

    try {
      String lang = reqParams.get("locale");
      if (lang == null) {
        returnError(400, new IllegalArgumentException("locale argument missing"));
        return;
      }

      String utterance = reqParams.get("q");

      if (utterance != null) {
        if (APIServer.opts.verbose >= 3)
          LogInfo.logs("Removing %s (locale = %s) from query cache", utterance, lang);
        this.server.langs.get(lang).cache.clear(utterance);
      } else {
        if (APIServer.opts.verbose >= 3)
          LogInfo.logs("Clearing query cache for locale = %s", lang);
        this.server.langs.get(lang).cache.clear();

        ThingpediaLexicon.clearAllCaches();
        EntityLexicon.clearAllCaches();
      }

      returnOk("Cache cleared");
    } catch (Exception e) {
      returnError(400, e);
    }
  }
}
