package edu.stanford.nlp.sempre;

import java.io.IOException;

public class PosixHelper {
	static {
		System.loadLibrary("posix");
	}

	public static native void setuid(String chuid) throws IOException, SecurityException;
}
