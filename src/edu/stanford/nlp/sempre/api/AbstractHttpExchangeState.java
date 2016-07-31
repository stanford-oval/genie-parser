package edu.stanford.nlp.sempre.api;

import static fig.basic.LogInfo.logs;

import java.io.*;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

import edu.stanford.nlp.sempre.Json;
import fig.basic.LogInfo;

abstract class AbstractHttpExchangeState {
  protected final HttpExchange exchange;
  protected final Map<String, String> reqParams;
  protected final String remoteHost;

  public AbstractHttpExchangeState(HttpExchange exchange) {
    this.exchange = exchange;
    URI uri = exchange.getRequestURI();
    this.remoteHost = exchange.getRemoteAddress().toString();

    reqParams = parseQuery(uri.getRawQuery());

    if (APIServer.opts.verbose >= 2)
      LogInfo.logs("GET %s from %s", uri, remoteHost);
  }

  private static Map<String, String> parseQuery(String rawQuery) {
    if (rawQuery == null || rawQuery.length() == 0)
      return Collections.emptyMap();

    Map<String, String> ret = new HashMap<>();
    for (String s : rawQuery.split("[&;]")) {
      String[] kv = s.split("=", 2);
      try {
        String key = URLDecoder.decode(kv[0], "UTF-8");
        String value = URLDecoder.decode(kv[1], "UTF-8");
        if (APIServer.opts.verbose >= 5)
          logs("%s => %s", key, value);
        ret.put(key, value);
      } catch (UnsupportedEncodingException e) {
        throw new RuntimeException(e);
      }
    }

    return ret;
  }

  protected void returnError(int status, Exception e) throws IOException {
    Map<String, Object> json = new HashMap<>();
    json.put("error", e.getMessage());
    returnJson(status, json);
  }

  protected void returnJson(int status, Map<String, Object> json) throws IOException {
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", "application/json;charset=utf8");
    headers.set("Access-Control-Allow-Origin", "*");
    exchange.sendResponseHeaders(status, 0);

    PrintWriter out = new PrintWriter(new OutputStreamWriter(exchange.getResponseBody(), Charset.forName("UTF-8")));
    out.println(Json.writeValueAsStringHard(json));
    out.close();
  }

  protected void returnOk(String ok) throws IOException {
    Map<String, Object> json = new HashMap<>();
    json.put("result", ok);
    returnJson(200, json);
  }

  protected abstract void doHandle() throws IOException;

  public void run() {
    try {
      doHandle();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      exchange.close();
    }
  }
}

abstract class AdminHttpExchangeState extends AbstractHttpExchangeState {
  public AdminHttpExchangeState(HttpExchange exchange) {
    super(exchange);
  }

  protected boolean checkCredentials() throws IOException {
    if (APIServer.opts.accessToken == null || !APIServer.opts.accessToken.equals(reqParams.get("accessToken"))) {
      if (APIServer.opts.verbose >= 2)
        LogInfo.logs("Invalid access token to admin endpoint");
      returnError(403, new SecurityException("Invalid access token"));
      return false;
    }
    return true;
  }
}