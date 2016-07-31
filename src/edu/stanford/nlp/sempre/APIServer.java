package edu.stanford.nlp.sempre;

import static fig.basic.LogInfo.logs;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.Charset;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.sun.net.httpserver.*;

import edu.stanford.nlp.sempre.corenlp.CoreNLPAnalyzer;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class APIServer implements Runnable {
  public static class Options {
    @Option
    public int port = 8400;
    @Option
    public int numThreads = 4;
    @Option
    public int verbose = 1;
    @Option
    public List<String> languages = Arrays.asList(new String[] { "en", "it", "es" });
    @Option
    public String accessToken = null;
  }

  public static Options opts = new Options();

  private static class LanguageContext {
    public final Parser parser;
    public final Params params;
    public final LanguageAnalyzer analyzer;
    public final QueryCache cache = new QueryCache(256);

    public LanguageContext(String tag) {
      Builder builder = new Builder();
      builder.buildForLanguage(tag);
      parser = builder.parser;
      params = builder.params;
      analyzer = new CoreNLPAnalyzer(tag);
    }
  }

  private final Map<String, Session> sessionMap = new HashMap<>();
  private final Map<String, LanguageContext> langs = new ConcurrentHashMap<>();

  private static abstract class AbstractHttpExchangeState {
    protected final HttpExchange exchange;
    protected final Map<String, String> reqParams;
    protected final String remoteHost;

    public AbstractHttpExchangeState(HttpExchange exchange) {
      this.exchange = exchange;
      URI uri = exchange.getRequestURI();
      this.remoteHost = exchange.getRemoteAddress().toString();

      reqParams = parseQuery(uri.getRawQuery());

      if (opts.verbose >= 2)
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
          if (opts.verbose >= 5)
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

  private abstract class AdminHttpExchangeState extends AbstractHttpExchangeState {
    public AdminHttpExchangeState(HttpExchange exchange) {
      super(exchange);
    }

    protected boolean checkCredentials() throws IOException {
      if (opts.accessToken == null || !opts.accessToken.equals(reqParams.get("accessToken"))) {
        if (opts.verbose >= 2)
          LogInfo.logs("Invalid access token to admin endpoint");
        returnError(403, new SecurityException("Invalid access token"));
        return false;
      }
      return true;
    }
  }

  private class ClearCacheExchangeState extends AdminHttpExchangeState {
    public ClearCacheExchangeState(HttpExchange exchange) {
      super(exchange);
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
          if (opts.verbose >= 3)
            LogInfo.logs("Removing %s (locale = %s) from query cache", utterance, lang);
          langs.get(lang).cache.clear(utterance);
        } else {
          if (opts.verbose >= 3)
            LogInfo.logs("Clearing query cache for locale = %s", lang);
          langs.get(lang).cache.clear();
        }

        returnOk("Cache cleared");
      } catch (Exception e) {
        returnError(400, e);
      }
    }
  }

  private class ReloadParametersExchangeState extends AdminHttpExchangeState {
    public ReloadParametersExchangeState(HttpExchange exchange) {
      super(exchange);
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

        if (!langs.containsKey(lang)) {
          returnError(400, new IllegalArgumentException("invalid language tag"));
          return;
        }
        langs.put(lang, new LanguageContext(lang));
      } catch (Exception e) {
        returnError(400, e);
      }
    }
  }

  private class ExchangeState extends AbstractHttpExchangeState {
    static private final int MAX_ITEMS = 5;

    private final String sessionId;

    public ExchangeState(HttpExchange exchange) {
      super(exchange);

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
          language = langs.get(splitTag[0] + "_" + splitTag[1]);
        if (language == null && splitTag.length >= 1)
          language = langs.get(splitTag[0]);
      }
      // fallback to english if the language is not recognized or
      // locale was not specified
      if (language == null)
        language = langs.get("en");

      String query = reqParams.get("q");

      int exitStatus;
      List<Derivation> derivations = null;
      Exception error = null;

      try {
        if (query != null) {
          // try from cache
          derivations = language.cache.hit(query);
          if (opts.verbose >= 3) {
            if (derivations != null)
              logs("cache hit");
            else
              logs("cache miss");
          }

          if (derivations == null) {
            Session session = getSession(sessionId);
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

  private class SessionGCTask extends TimerTask {
    @Override
    public void run() {
      gcSessions();
    }
  }

  private synchronized Session getSession(String sessionId) {
    if (sessionMap.containsKey(sessionId)) {
      return sessionMap.get(sessionId);
    } else {
      Session newSession = new Session(sessionId);
      sessionMap.put(sessionId, newSession);
      return newSession;
    }
  }

  private synchronized void gcSessions() {
    Iterator<Session> iter = sessionMap.values().iterator();
    long now = System.currentTimeMillis();

    while (iter.hasNext()) {
      Session session = iter.next();
      synchronized (session) {
        long lastTime = session.getLastAccessTime();
        if (lastTime - now > 300 * 1000) // 5 min
          iter.remove();
      }
    }
  }

  private void addLanguage(String tag) {
    LanguageContext language = new LanguageContext(tag);
    langs.put(tag, language);
  }

  @Override
  public void run() {
    // Add supported languages
    for (String tag : opts.languages)
      addLanguage(tag);

    try {
      String hostname = fig.basic.SysInfoUtils.getHostName();
      HttpServer server = HttpServer.create(new InetSocketAddress(opts.port), 10);
      ExecutorService pool = Executors.newFixedThreadPool(opts.numThreads);
      server.createContext("/query", new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new ExchangeState(exchange).run();
        }
      });
      server.createContext("/admin/clear-cache", new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new ClearCacheExchangeState(exchange).run();
        }
      });
      server.createContext("/admin/reload", new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new ReloadParametersExchangeState(exchange).run();
        }
      });
      server.setExecutor(pool);
      server.start();
      LogInfo.logs("Server started at http://%s:%s/sempre", hostname, opts.port);

      Timer gcTimer = new Timer(true);
      gcTimer.schedule(new SessionGCTask(), 600000, 600000);

      try {
        while (!Thread.currentThread().isInterrupted())
          Thread.sleep(60000);
      } catch (InterruptedException e) {
      }

      server.stop(0);
      pool.shutdown();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new APIServer(), Master.getOptionsParser());
  }
}
