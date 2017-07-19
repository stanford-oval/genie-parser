package edu.stanford.nlp.sempre.corenlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class HunspellDictionary {

  static {
    System.loadLibrary("hunspell_jni");
  }

  private long nativeLib;

  private static native long nativeLoadLib(String affPath, String dicPath);

  private static native boolean nativeSpell(long lib, String word);

  private static native List<String> nativeSuggest(long lib, String word);

  private static native void nativeFreeLib(long lib);

  public HunspellDictionary(String dictionaryPath) throws IOException {
    File aff = new File(dictionaryPath + ".aff");
    File dic = new File(dictionaryPath + ".dic");
    if (!aff.exists() || !dic.exists())
      throw new FileNotFoundException("Dictionary files not found");
    nativeLib = nativeLoadLib(aff.getAbsolutePath(), dic.getAbsolutePath());
    if (nativeLib == 0)
      throw new IOException("Failed to load Hunspell dictionary");
  }

  @Override
  protected void finalize() {
    if (nativeLib != 0)
      nativeFreeLib(nativeLib);
    nativeLib = 0;
  }

  public boolean spell(String word) {
    return nativeSpell(nativeLib, word);
  }

  public List<String> suggest(String word) {
    return nativeSuggest(nativeLib, word);
  }
}
