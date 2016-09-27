package edu.stanford.nlp.sempre.api;

import java.io.*;
import java.math.BigInteger;
import java.net.InetSocketAddress;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.Security;
import java.util.*;
import java.util.concurrent.*;

import javax.net.ssl.SSLContext;

import com.sun.net.httpserver.*;

import edu.stanford.nlp.sempre.Master;
import edu.stanford.nlp.sempre.PosixHelper;
import edu.stanford.nlp.sempre.Session;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

final class SecureIdentifiers {
  private SecureIdentifiers() {
  }

  private static SecureRandom random = new SecureRandom();

  public static String getId() {
    return new BigInteger(130, random).toString(32);
  }
}

class LogFlusherThread<E> extends Thread {
  private final BlockingQueue<E> queue;
  private final String logFile;

  public LogFlusherThread(BlockingQueue<E> queue, String logFile) {
    this.queue = queue;
    this.logFile = logFile;

    setDaemon(true);
    setName("log flusher " + (new File(logFile).getName()));
  }

  @Override
  public void run() {
    try (BufferedWriter writer = new BufferedWriter(new FileWriter(logFile, true))) {
      E next;
      while ((next = queue.take()) != null) {
        writer.append(next.toString());
        writer.append("\n");
        writer.flush();
      }
    } catch (IOException e) {
      LogInfo.logs("IOException writing to the log file! %s", e.getMessage());
    } catch (InterruptedException e) {

    }
  }
}

public class APIServer implements Runnable {
  public static class Options {
    @Option
    public int port = 8400;
    @Option
    public int ssl_port = -1;
    @Option
    public int numThreads = 4;
    @Option
    public int verbose = 1;
    @Option
    public String chuid = null;
    @Option
    public List<String> languages = Arrays.asList(new String[] { "en", "it", "es", "zh" });
    @Option
    public String accessToken = null;
    @Option
    public String utteranceLogFile = null;
  }

  public static Options opts = new Options();

  private class LogEntry {
    private final String languageTag;
    private final String utterance;

    public LogEntry(String languageTag, String utterance) {
      this.languageTag = languageTag;
      this.utterance = utterance;
    }

    @Override
    public String toString() {
      return languageTag + "\t" + utterance;
    }
  }

  private final BlockingQueue<LogEntry> logQueue;

  private final Map<String, Session> sessionMap = new HashMap<>();
  final Map<String, LanguageContext> langs = new ConcurrentHashMap<>();

  public APIServer() {
    logQueue = new LinkedBlockingQueue<>();
  }

  synchronized Session getSession(String sessionId) {
    if (sessionMap.containsKey(sessionId)) {
      return sessionMap.get(sessionId);
    } else {
      Session newSession = new Session(sessionId);
      sessionMap.put(sessionId, newSession);
      return newSession;
    }
  }

  void logUtterance(String languageTag, String utterance) {
    if (opts.utteranceLogFile == null)
      return;

    logQueue.offer(new LogEntry(languageTag, utterance));
  }

  private class SessionGCTask extends TimerTask {
    @Override
    public void run() {
      gcSessions();
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
    // use a sane x509 key manager that does server-side SNI properly
    Security.setProperty("ssl.KeyManagerFactory.algorithm", "NewSunX509");

    try {
      // create the server early so we can bind and drop privileges

      HttpServer server = null;
      HttpsServer ssl_server = null;
      if (opts.port >= 0)
          server = HttpServer.create(new InetSocketAddress(opts.port), 10);
      if (opts.ssl_port >= 0) {
          try {
          SSLContext context = SSLContext.getDefault();
          ssl_server = HttpsServer.create(new InetSocketAddress(opts.ssl_port), 10);
          // use default configuration
          ssl_server.setHttpsConfigurator(new HttpsConfigurator(context));
        } catch (NoSuchAlgorithmException e) {
          throw new RuntimeException(e);
        }
      }
      if (server == null && ssl_server == null)
        throw new RuntimeException("Must specify one of port or ssl_port");

      if (opts.chuid != null)
        PosixHelper.setuid(opts.chuid);

      // Add supported languages
      for (String tag : opts.languages)
        addLanguage(tag);

      // open log files (after we dropped privileges, so the log files are not owned by root)
      if (opts.utteranceLogFile != null)
        new LogFlusherThread<>(logQueue, opts.utteranceLogFile).start();

      for (LanguageContext lang : langs.values())
        lang.exactMatch.load();

      String hostname = fig.basic.SysInfoUtils.getHostName();
      ExecutorService pool = Executors.newFixedThreadPool(opts.numThreads);

      HttpHandler query = new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new QueryExchangeState(APIServer.this, exchange).run();
        }
      };
      HttpHandler learn = new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new OnlineLearnExchangeState(APIServer.this, exchange).run();
        }
      };
      HttpHandler clearCache = new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new ClearCacheExchangeState(APIServer.this, exchange).run();
        }
      };
      HttpHandler reload = new HttpHandler() {
        @Override
        public void handle(HttpExchange exchange) {
          new ReloadParametersExchangeState(APIServer.this, exchange).run();
        }
      };

      if (server != null) {
        server.createContext("/query", query);
        server.createContext("/learn", learn);
        server.createContext("/admin/clear-cache", clearCache);
        server.createContext("/admin/reload", reload);
        server.setExecutor(pool);
        server.start();
        LogInfo.logs("Server started at http://%s:%s/", hostname, opts.port);
      }
      if (ssl_server != null) {
        ssl_server.createContext("/query", query);
        ssl_server.createContext("/learn", learn);
        ssl_server.createContext("/admin/clear-cache", clearCache);
        ssl_server.createContext("/admin/reload", reload);

        ssl_server.setExecutor(pool);
        ssl_server.start();
        LogInfo.logs("Server started at https://%s:%s/", hostname, opts.ssl_port);
      }

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
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) {
    Execution.run(args, "Main", new APIServer(), Master.getOptionsParser());
  }
}
