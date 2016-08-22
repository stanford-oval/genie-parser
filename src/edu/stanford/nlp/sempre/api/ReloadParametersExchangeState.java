package edu.stanford.nlp.sempre.api;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;

class ReloadParametersExchangeState extends AdminHttpExchangeState {
  private final APIServer server;

  public ReloadParametersExchangeState(APIServer server, HttpExchange exchange) {
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

      if (!this.server.langs.containsKey(lang)) {
        returnError(400, new IllegalArgumentException("invalid language tag"));
        return;
      }
      LanguageContext previous = this.server.langs.get(lang);
      this.server.langs.put(lang, new LanguageContext(lang, previous.onlineLearnSaveQueue));

      returnOk("Reloaded");
    } catch (Exception e) {
      returnError(400, e);
    }
  }
}
