package edu.stanford.nlp.sempre.api;

import java.io.IOException;

import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.thingtalk.LexiconBuilder;
import edu.stanford.nlp.sempre.thingtalk.ThingpediaLexicon;

public class IncrementalUpdateLexiconExchangeState extends AdminHttpExchangeState {
  private final APIServer server;

  public IncrementalUpdateLexiconExchangeState(APIServer server, HttpExchange exchange) {
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

      String kind = reqParams.get("kind");
      if (kind == null) {
        returnError(400, new IllegalArgumentException("kind argument missing"));
        return;
      }
      LanguageContext ctx = localeToLanguage(server.langs, localeTag);

      LexiconBuilder builder = new LexiconBuilder(ctx.analyzer, ctx.tag, kind);
      builder.build();

      ThingpediaLexicon.getForLanguage(ctx.tag).clear();

      returnOk("Lexicon updated");
    } catch (Exception e) {
      returnError(400, e);
    }
  }
}