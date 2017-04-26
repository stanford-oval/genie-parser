package edu.stanford.nlp.sempre.api;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.thingtalk.EntityLexicon;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaLexicon;
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
      String localeTag = reqParams.get("locale");
      if (localeTag == null) {
        returnError(400, new IllegalArgumentException("locale argument missing"));
        return;
      }

      String utterance = reqParams.get("q");
      LanguageContext ctx = localeToLanguage(server.langs, localeTag);

      if (utterance != null) {
        if (APIServer.opts.verbose >= 3)
          LogInfo.logs("Removing %s (locale = %s) from query cache", utterance, ctx.tag);
        ctx.cache.clear(utterance);
      } else {
        if (APIServer.opts.verbose >= 3)
          LogInfo.logs("Clearing query cache for locale = %s", ctx.tag);
        ctx.cache.clear();

        ThingpediaLexicon.getForLanguage(ctx.tag).clear();
        EntityLexicon.getForLanguage(ctx.tag).clear();
      }

      returnOk("Cache cleared");
    } catch (Exception e) {
      returnError(400, e);
    }
  }
}