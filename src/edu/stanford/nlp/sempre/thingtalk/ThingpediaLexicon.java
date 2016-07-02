package edu.stanford.nlp.sempre.thingtalk;

import java.io.Closeable;
import java.io.IOException;
import java.sql.*;
import java.util.Iterator;

import org.apache.commons.dbcp2.BasicDataSource;

import edu.stanford.nlp.sempre.*;
import fig.basic.LogInfo;
import fig.basic.Option;

public class ThingpediaLexicon {
	public static class Options {
		@Option
		public String dbUrl = "jdbc:mysql://localhost/thingengine";
		@Option
		public String dbUser = "thingengine";
		@Option
		public String dbPw = "thingengine";
		@Option
		public int verbose = 0;
	}

	public static Options opts = new Options();

	public enum Mode {
		APP, KIND, TRIGGER, ACTION, QUERY;
	};

	public static abstract class Entry {
		public abstract Formula toFormula();

		public abstract String getRawPhrase();

		public void addFeatures(FeatureVector vec) {
		}

		@Override
		public String toString() {
			return "[ " + getRawPhrase() + " => " + toFormula() + " ]";
		}
	}

	private static class AppEntry extends Entry {
		private final String rawPhrase;
		private final long userId;
		private final String appId;

		public AppEntry(String rawPhrase, long userId, String appId) {
			this.rawPhrase = rawPhrase;
			this.userId = userId;
			this.appId = appId;
		}

		@Override
		public String getRawPhrase() {
			return rawPhrase;
		}

		@Override
		public Formula toFormula() {
			return new ValueFormula<>(new NameValue("tt:app." + userId + "." + appId));
		}
	}

	private static class ChannelEntry extends Entry {
		private final String rawPhrase;
		private final String kind;
		private final String name;

		public ChannelEntry(String rawPhrase, String kind, String name) {
			this.rawPhrase = rawPhrase;
			this.kind = kind;
			this.name = name;
		}

		@Override
		public String getRawPhrase() {
			return rawPhrase + " on " + kind;
		}

		@Override
		public Formula toFormula() {
			return new ValueFormula<>(new NameValue("tt:" + kind + "." + name));
		}

		@Override
		public void addFeatures(FeatureVector vec) {
			vec.add("kinds", kind);
		}
	}

	private static class KindEntry extends Entry {
		private final String kind;

		public KindEntry(String kind) {
			this.kind = kind;
		}

		@Override
		public String getRawPhrase() {
			return kind;
		}

		@Override
		public Formula toFormula() {
			return new ValueFormula<>(new NameValue("tt:device." + kind));
		}
	}

	private static ThingpediaLexicon instance;

	private final BasicDataSource dataSource;

	private ThingpediaLexicon() {
		dataSource = new BasicDataSource();
		dataSource.setDriverClassName("com.mysql.jdbc.Driver");
		dataSource.setUrl(opts.dbUrl);
		dataSource.setUsername(opts.dbUser);
		dataSource.setPassword(opts.dbPw);
	}

	public static ThingpediaLexicon getSingleton() {
		if (instance == null)
			instance = new ThingpediaLexicon();

		return instance;
	}

	static {
		try {
			Class.forName("com.mysql.jdbc.Driver");
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	public static abstract class EntryStream implements Iterator<Entry>, Closeable, AutoCloseable
	{
		protected final ResultSet rs;
		private final Statement stmt;
		private final Connection con;
		private Entry nextEntry;

		public EntryStream(Connection con, Statement stmt, ResultSet rs) {
			this.con = con;
			this.stmt = stmt;
			this.rs = rs;
			nextEntry = null;
		}

		@Override
		public void close() throws IOException {
			try {
				con.close();
				stmt.close();
				rs.close();
			} catch (SQLException e) {
				throw new IOException(e);
			}
		}

		protected abstract Entry createEntry() throws SQLException;

		private void checkNext() {
			try {
			if (nextEntry != null)
				return;
			if (!rs.next())
				return;
			nextEntry = createEntry();
			} catch(SQLException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public boolean hasNext() {
			checkNext();
			return nextEntry != null;
		}

		@Override
		public Entry next() {
			checkNext();
			Entry next = nextEntry;
			nextEntry = null;
			return next;
		}
	}

	private static class AppEntryStream extends EntryStream {
		public AppEntryStream(Connection con, Statement stmt, ResultSet rs) {
			super(con, stmt, rs);
		}

		@Override
		protected AppEntry createEntry() throws SQLException {
			return new AppEntry(rs.getString(1), rs.getLong(2), rs.getString(3));
		}
	}

	private static class ChannelEntryStream extends EntryStream {
		public ChannelEntryStream(Connection con, Statement stmt, ResultSet rs) {
			super(con, stmt, rs);
		}

		@Override
		protected ChannelEntry createEntry() throws SQLException {
			return new ChannelEntry(rs.getString(1), rs.getString(2), rs.getString(3));
		}
	}

	private static class KindEntryStream extends EntryStream {
		public KindEntryStream(Connection con, Statement stmt, ResultSet rs) {
			super(con, stmt, rs);
		}

		@Override
		protected KindEntry createEntry() throws SQLException {
			return new KindEntry(rs.getString(1));
		}
	}

	public EntryStream lookupApp(String phrase) throws SQLException {
		if (opts.verbose >= 2)
			LogInfo.logs("ThingpediaLexicon.lookupApp %s", phrase);

		String query;
		if (Builder.opts.parser.equals("BeamParser")) {
			query = "select canonical,owner,appId from app where canonical = ?";
		} else {
			query = "select canonical,owner,appId from app where match canonical against (? in natural language mode)";
		}

		Connection con = dataSource.getConnection();
		PreparedStatement stmt = con.prepareStatement(query);
		stmt.setString(1, phrase);
		
		return new AppEntryStream(con, stmt, stmt.executeQuery());
	}

	public EntryStream lookupKind(String phrase) throws SQLException {
		if (opts.verbose >= 2)
			LogInfo.logs("ThingpediaLexicon.lookupKind %s", phrase);

		String query = "select kind from device_schema where kind = ?";

		Connection con = dataSource.getConnection();
		PreparedStatement stmt = con.prepareStatement(query);
		stmt.setString(1, phrase);

		return new KindEntryStream(con, stmt, stmt.executeQuery());
	}

	public EntryStream lookupChannel(String phrase, Mode channel_type) throws SQLException {
		if (opts.verbose >= 2)
			LogInfo.logs("ThingpediaLexicon.lookupChannel(%s) %s", channel_type, phrase);

		String query;
		if (Builder.opts.parser.equals("BeamParser")) {
			query = "select dsc.canonical,ds.kind,dsc.name from device_schema_channels dsc, device_schema ds "
					+ " where dsc.schema_id = ds.id and dsc.version = ds.approved_version and channel_type = ? and canonical = ?";
		} else {
			query = "select dsc.canonical,ds.kind,dsc.name from device_schema_channels dsc, device_schema ds "
					+ " where dsc.schema_id = ds.id and dsc.version = ds.approved_version and channel_type = ? and "
					+ "match canonical against (? in natural language mode)";
		}

		Connection con = dataSource.getConnection();
		PreparedStatement stmt = con.prepareStatement(query);
		stmt.setString(1, channel_type.name().toLowerCase());
		stmt.setString(2, phrase);

		return new ChannelEntryStream(con, stmt, stmt.executeQuery());
	}
}
