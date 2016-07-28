package edu.stanford.nlp.sempre;

import static fig.basic.LogInfo.logs;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.sun.net.httpserver.*;

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
	}

	public static Options opts = new Options();

	private static class LanguageContext {
		public final Parser parser;
		public final Params params;
		public final QueryCache cache = new QueryCache(256);

		public LanguageContext(String tag) {
			Builder builder = new Builder();
			builder.buildForLanguage(tag);
			parser = builder.parser;
			params = builder.params;
		}

		public void learn() {
			Dataset dataset = new Dataset();
			dataset.read();

			Learner learner = new Learner(parser, params, dataset);
			learner.learn();
		}
	}

	private class Handler implements HttpHandler {
		@Override
		public void handle(HttpExchange exchange) {
			try {
				new ExchangeState(exchange);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	private final Map<String, Session> sessionMap = new HashMap<>();
	private final Map<String, LanguageContext> langs = new HashMap<>();

	private class ExchangeState {
		static private final int MAX_ITEMS = 5;

		// Input
		private HttpExchange exchange;
		private Map<String, String> reqParams = new HashMap<>();
		private String remoteHost;

		// For header
		private HttpCookie cookie;
		private boolean isNewSession;

		public ExchangeState(HttpExchange exchange) throws IOException {
			this.exchange = exchange;

			URI uri = exchange.getRequestURI();
			this.remoteHost = exchange.getRemoteAddress().toString();

			// Don't use uri.getQuery: it can't distinguish between '+' and '-'
			String[] tokens = uri.toString().split("\\?");
			if (tokens.length == 2) {
				for (String s : tokens[1].split("&")) {
					String[] kv = s.split("=", 2);
					try {
						String key = URLDecoder.decode(kv[0], "UTF-8");
						String value = URLDecoder.decode(kv[1], "UTF-8");
						if (opts.verbose >= 5)
							logs("%s => %s", key, value);
						reqParams.put(key, value);
					} catch (UnsupportedEncodingException e) {
						throw new RuntimeException(e);
					}
				}
			}

			String cookieStr = exchange.getRequestHeaders().getFirst("Cookie");
			if (cookieStr != null) {  // Cookie already exists
				cookie = HttpCookie.parse(cookieStr).get(0);
				isNewSession = false;
			} else {
				cookie = new HttpCookie("sessionId", SecureIdentifiers.getId());
				isNewSession = true;  // Create a new cookie
			}

			String sessionId = null;
			if (cookie != null) sessionId = cookie.getValue();
			if (opts.verbose >= 2)
				LogInfo.logs("GET %s from %s (%ssessionId=%s)", uri, remoteHost, isNewSession ? "new " : "", sessionId);

			String uriPath = uri.getPath();
			if (uriPath.equals("/query")) {
				handleQuery(sessionId);
			} else {
				exchange.sendResponseHeaders(404, 0);
			}

			exchange.close();
		}

		private void setHeaders(int status) throws IOException {
			Headers headers = exchange.getResponseHeaders();
			headers.set("Content-Type", "application/json;charset=utf8");
			if (isNewSession && cookie != null)
				headers.set("Set-Cookie", cookie.toString());
			exchange.sendResponseHeaders(status, 0);
		}

		private String makeJson(List<Derivation> response) {
			Map<String, Object> json = new HashMap<>();
			List<Object> items = new ArrayList<>();
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

			return Json.writeValueAsStringHard(json);
		}

		private String makeJson(Exception e) {
			Map<String, Object> json = new HashMap<>();
			json.put("error", e.getMessage());
			return Json.writeValueAsStringHard(json);
		}

		private List<Derivation> handleUtterance(Session session, LanguageContext language, String query) {
			session.updateContext();

			// Create example
			Example.Builder b = new Example.Builder();
			b.setId("session:" + session.id);
			b.setUtterance(query);
			b.setContext(session.context);
			Example ex = b.createExample();

			ex.preprocess();

			// Parse!
			language.parser.parse(language.params, ex, false);

			return ex.getPredDerivations();
		}

		private void handleQuery(String sessionId) throws IOException {
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

			// Print header
			setHeaders(exitStatus);
			// Print body
			PrintWriter out = new PrintWriter(new OutputStreamWriter(exchange.getResponseBody()));
			if (error != null)
				out.println(makeJson(error));
			else
				out.println(makeJson(derivations));

			out.close();
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
		language.learn();
		langs.put(tag, language);
	}

	@Override
	public void run() {
		// Add supported languages
		addLanguage("en");
		addLanguage("it");

		try {
			String hostname = fig.basic.SysInfoUtils.getHostName();
			HttpServer server = HttpServer.create(new InetSocketAddress(opts.port), 10);
			ExecutorService pool = Executors.newFixedThreadPool(opts.numThreads);
			server.createContext("/", new Handler());
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
