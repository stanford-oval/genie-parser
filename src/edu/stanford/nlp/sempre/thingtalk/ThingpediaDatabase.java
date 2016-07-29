package edu.stanford.nlp.sempre.thingtalk;

import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.logging.Logger;

import javax.sql.DataSource;

import org.apache.commons.dbcp2.BasicDataSource;

import fig.basic.Option;

public class ThingpediaDatabase implements DataSource {
	public static class Options {
		@Option
		public String dbUrl = "jdbc:mysql://localhost/thingengine";
		@Option
		public String dbUser = "thingengine";
		@Option
		public String dbPw = "thingengine";
	}

	public static Options opts = new Options();

	private static ThingpediaDatabase instance;

	private final BasicDataSource dataSource;

	private ThingpediaDatabase() {
		dataSource = new BasicDataSource();
		dataSource.setDriverClassName("com.mysql.jdbc.Driver");
		dataSource.setUrl(opts.dbUrl);
		dataSource.setUsername(opts.dbUser);
		dataSource.setPassword(opts.dbPw);
	}

	public static ThingpediaDatabase getSingleton() {
		if (instance == null)
			instance = new ThingpediaDatabase();

		return instance;
	}

	static {
		try {
			Class.forName("com.mysql.jdbc.Driver");
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public PrintWriter getLogWriter() throws SQLException {
		return dataSource.getLogWriter();
	}

	@Override
	public int getLoginTimeout() throws SQLException {
		return dataSource.getLoginTimeout();
	}

	@Override
	public Logger getParentLogger() throws SQLFeatureNotSupportedException {
		return dataSource.getParentLogger();
	}

	@Override
	public void setLogWriter(PrintWriter logWriter) throws SQLException {
		dataSource.setLogWriter(logWriter);
	}

	@Override
	public void setLoginTimeout(int loginTimeout) throws SQLException {
		dataSource.setLoginTimeout(loginTimeout);
	}

	@Override
	public boolean isWrapperFor(Class<?> iface) throws SQLException {
		return iface.isInstance(dataSource);
	}

	@Override
	public <T> T unwrap(Class<T> iface) throws SQLException {
		return iface.cast(dataSource);
	}

	@Override
	public Connection getConnection() throws SQLException {
		return dataSource.getConnection();
	}

	@Override
	public Connection getConnection(String user, String pass) throws SQLException {
		return dataSource.getConnection(user, pass);
	}
}
