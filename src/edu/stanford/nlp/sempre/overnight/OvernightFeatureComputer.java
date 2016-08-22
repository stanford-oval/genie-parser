package edu.stanford.nlp.sempre.overnight;

import java.io.File;
import java.util.*;

import com.google.common.base.Joiner;
import com.google.common.collect.Sets;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.sempre.*;
import edu.stanford.nlp.sempre.LanguageInfo.LanguageUtils;
import fig.basic.BipartiteMatcher;
import fig.basic.LogInfo;
import fig.basic.Option;

/**
 * Define features on the input utterance and a partial canonical utterance.
 *
 * Feature computation recipe:
 * - For both the input and (partial) canonical utterance, extract a list of tokens
 *   (perhaps with POS tags).
 * - Given a list of tokens, extract a set of items, where an item is a (tag,
 *   data) pair, where the tag specifies the "type" of the data, and is used
 *   to determine features.  Example: ("bigram", "not contains"), ("unigram",
 *   "not"), ("unigram-RB", "not")
 * - Given the input and canonical items, define recall features (how much of
 *   the input items is the canononical covering).
 * This recipe allows us to decouple the extraction of items on one utterance
 * from the computation of actual precision/recall features.
 *
 * @author Percy Liang
 * @author Yushi Wang
 */
public final class OvernightFeatureComputer implements FeatureComputer {
  public static class Options {
    @Option(gloss = "Set of paraphrasing feature domains to include")
    public Set<String> featureDomains = new HashSet<>();

    @Option(gloss = "Whether or not to count intermediate categories for size feature")
    public boolean countIntermediate = true;

    @Option(gloss = "Whether or not to do match/ppdb analysis")
    public boolean itemAnalysis = true;

    @Option(gloss = "Whether or not to learn paraphrases")
    public boolean learnParaphrase = true;

    @Option(gloss = "Verbose flag")
    public int verbose = 0;

    @Option(gloss = "Path to alignment file")
    public String wordAlignmentPath;
    @Option(gloss = "Path to phrase alignment file")
    public String phraseAlignmentPath;
    @Option(gloss = "Threshold for phrase table co-occurrence")
    public int phraseTableThreshold = 3;
  }

  public static Options opts = new Options();

  // note: throughout this file we use ArrayList<> instead of the more
  // idiomatic List<> because we get
  // invokevirtual bytecodes instead of invokedynamic, and this is
  // very hot code

  // Represents a local pattern on an utterance.
  private static final class Item {
    public enum Tag {
      UNIGRAM, BIGRAM, SKIP_BIGRAM
    };

    public final Tag tag;
    public final String data1;
    public final String stem1;
    public final String ner1;
    public final String data2;
    public final String stem2;
    public final String ner2;

    public Item(Tag tag, String data1, String stem1, String ner1) {
      this.tag = tag;
      this.data1 = data1;
      this.stem1 = stem1;
      this.ner1 = ner1;
      this.data2 = null;
      this.stem2 = null;
      this.ner2 = null;
    }

    public Item(Tag tag, String data1, String stem1, String ner1, String data2, String stem2, String ner2) {
      this.tag = tag;
      this.data1 = data1;
      this.stem1 = stem1;
      this.ner1 = null;
      this.data2 = data2;
      this.stem2 = stem2;
      this.ner2 = null;
    }

    public Item(Tag tag, Item i1, Item i2) {
      assert tag != Tag.UNIGRAM;
      assert i1.tag == Tag.UNIGRAM;
      assert i2.tag == Tag.UNIGRAM;
      this.tag = tag;
      this.data1 = i1.data1;
      this.stem1 = i1.stem1;
      this.ner1 = i1.ner1;
      this.data2 = i2.data1;
      this.stem2 = i2.stem1;
      this.ner2 = i2.ner1;
    }

    @Override
    public String toString() {
      if (tag == Tag.UNIGRAM)
        return tag + ":" + data1;
      else
        return tag + ":" + data1 + " " + data2;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((data1 == null) ? 0 : data1.hashCode());
      result = prime * result + ((data2 == null) ? 0 : data2.hashCode());
      result = prime * result + ((tag == null) ? 0 : tag.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      return equals((Item) obj);
    }

    public boolean equals(Item other) {
      if (this == other)
        return true;
      if (tag != other.tag)
        return false;
      if (!data1.equals(other.data1))
        return false;
      if (data2 == null) {
        if (other.data2 != null)
          return false;
      } else if (!data2.equals(other.data2))
        return false;
      return true;
    }

    public boolean stemEquals(Item other) {
      if (this == other)
        return true;
      if (tag != other.tag)
        return false;
      if (!stem1.equals(other.stem1))
        return false;
      if (stem2 == null) {
        if (other.stem2 != null)
          return false;
      } else if (!stem2.equals(other.stem2))
        return false;
      return true;
    }
  }

  private static final class ItemList {
    public final ArrayList<Item> unigrams = new ArrayList<>();
    public final ArrayList<Item> bigrams = new ArrayList<>();
    public final ArrayList<Item> skipBigrams = new ArrayList<>();
    public boolean valid = false;

    @Override
    public String toString() {
      return "[ " + unigrams + ", " + bigrams + ", " + skipBigrams + " ]";
    }
  }

  // TODO(yushi): make this less hacky
  private static final Set<String> ENGLISH_STOP_WORDS = Sets
      .newHashSet("\' \" `` ` \'\' a an the that which . what ? is are am be of on".split(" "));
  // FIXME: I don't particularly agree that all these words are stop words...
  private static final Set<String> ITALIAN_STOP_WORDS = Sets
      .newHashSet("\' \" `` ` \'\' un uno il lo la i gli le che quale cosa . ? è sono sei di su".split(" "));
  private static final Map<String, Set<String>> STOP_WORDS = new HashMap<>();
  static {
    STOP_WORDS.put("en", ENGLISH_STOP_WORDS);
    STOP_WORDS.put("it", ITALIAN_STOP_WORDS);
  }
  /*private static final Set<String> entities = new HashSet<>(
      Arrays.asList("alice", "bob", "greenberg", "greenberg cafe", "central office",
          "sacramento", "austin", "california", "texas", "colorado", "colorado river", "red river", "lake tahoe",
          "tahoe", "lake huron", "huron", "mount whitney", "whitney", "mount rainier", "rainier", "death valley",
          "pacific ocean", "pacific",
          "sesame", "mission ave", "mission", "chelsea",
          "multivariate data analysis", "multivariate data", "data analysis", "multivariate", "data", "efron", "lakoff",
          "annals of statistics", "annals", "annals of", "of statistics", "statistics", "computational linguistics",
          "computational", "linguistics",
          "thai cafe", "pizzeria juno",
          "new york", "york", "beijing", "brown university", "ucla", "mckinsey", "google"));*/
  private static final Set<String> entities = Collections.emptySet();

  private final Map<String, Set<String>> phraseTable;
  private final Aligner aligner;
  private final PPDBModel ppdbModel;
  private final String languageTag;
  private final Set<String> stopWords;

  public OvernightFeatureComputer(String languageTag) {
    this.languageTag = languageTag;
    aligner = Aligner.read(opts.wordAlignmentPath, languageTag);
    stopWords = STOP_WORDS.getOrDefault(languageTag, Collections.emptySet());
    ppdbModel = PPDBModel.getSingleton();
    phraseTable = loadPhraseTable(languageTag);
  }

  private boolean isStopWord(String token) {
    return stopWords.contains(token);
  }

  @Override public void extractLocal(Example ex, Derivation deriv) {
    if (deriv.rule.rhs == null) return;

    // do not featurize $StrValue - the paraphrase model is not appropriate for it
    if (deriv.rule.lhs != null && deriv.rule.lhs.equals("$StrValue"))
      return;

    // Optimization: feature vector same as child, so don't do anything.
    if (deriv.rule.isCatUnary()) {
      if (deriv.isRootCat()) {
        extractValueInFormulaFeature(deriv);

        ItemList inputItems = computeInputItems(ex);
        ItemList candidateItems = computeCandidateItems(ex, deriv);
        extractRootFeatures(ex, deriv, inputItems.unigrams, candidateItems.unigrams);
      }
      return;
    }

    // Important!  We want to define the global feature vector for this
    // derivation, but we can only specify the local feature vector.  So to
    // make things cancel out, we subtract out the unwanted feature vectors of
    // descendents.
    subtractDescendentsFeatures(deriv, deriv);

    deriv.addFeature("paraphrase", "size", derivationSize(deriv));

    ItemList inputItems = computeInputItems(ex);
    ItemList candidateItems = computeCandidateItems(ex, deriv);

    extractRootFeatures(ex, deriv, inputItems.unigrams, candidateItems.unigrams);
    extractLexicalFeatures(ex, deriv, inputItems.unigrams, candidateItems.unigrams);
    extractPhraseAlignmentFeatures(ex, deriv, candidateItems.unigrams);
    extractLogicalFormFeatures(ex, deriv);

    if (!opts.itemAnalysis) return;
    
    boolean hasMatch = opts.featureDomains.contains("match");
    boolean hasPpdb = opts.featureDomains.contains("ppdb");
    boolean hasSkipBigram = opts.featureDomains.contains("skip-bigram");
    boolean hasSkipPpdb = opts.featureDomains.contains("skip-ppdb");

    for (Item input : inputItems.unigrams) {
      double match = 0;
      double ppdb = 0;
      for (Item candidate : candidateItems.unigrams) {
        match = Math.max(match, computeMatch(input, candidate));
        ppdb = Math.max(ppdb, computeParaphrase(input, candidate));
      }
      if (match > 0 && hasMatch)
        deriv.addFeature("paraphrase", "match");
      if (ppdb > 0 && hasPpdb)
        deriv.addFeature("paraphrase", "ppdb");
    }
    for (Item input : inputItems.bigrams) {
      double match = 0;
      double ppdb = 0;
      for (Item candidate : candidateItems.bigrams) {
        match = Math.max(match, computeMatch(input, candidate));
        ppdb = Math.max(ppdb, computeParaphrase(input, candidate));
      }
      if (match > 0 && hasMatch)
        deriv.addFeature("paraphrase", "match");
      if (ppdb > 0 && hasPpdb)
        deriv.addFeature("paraphrase", "ppdb");
    }
    for (Item input : inputItems.skipBigrams) {
      double skipBigram = 0;
      double skipPpdb = 0;
      for (Item candidate : candidateItems.skipBigrams) {
          skipBigram = Math.max(skipBigram, computeMatch(input, candidate));
          skipPpdb = Math.max(skipPpdb, computeParaphrase(input, candidate));
      }
      if (skipBigram > 0 && hasSkipBigram)
        deriv.addFeature("paraphrase", "skip-bigram");
      if (skipPpdb > 0 && hasSkipPpdb)
        deriv.addFeature("paraphrase", "skip-ppdb");
    }

    if (opts.verbose >= 1) {
      HashMap<String, Double> features = new LinkedHashMap<>();
      deriv.incrementAllFeatureVector(+1, features);
      LogInfo.logs("category %s, %s %s", deriv.cat, inputItems, candidateItems);
      FeatureVector.logFeatures(features);
    }
  }

  private void extractValueInFormulaFeature(Derivation deriv) {
    if (!opts.featureDomains.contains("denotation")) return;

    if (deriv.value instanceof StringValue) {

      //get strings from value
      List<String> valueList = new ArrayList<>();

      String value = ((StringValue) deriv.value).value;

      if (value.charAt(0) == '[')
        value = value.substring(1, value.length() - 1); //strip "[]"
      String[] tokens = value.split(",");
      for (String token : tokens) {
        token = token.trim(); //strip spaces
        if (token.length() > 0)
          valueList.add(token);
      }

      //get strings from formula
      List<Formula> formulaList = deriv.formula.mapToList(formula -> {
        List<Formula> res = new ArrayList<>();
        if (formula instanceof ValueFormula) {
          res.add(formula);
        }
        return res;
      }, true);

      for (Formula f : formulaList) {
        Value formulaValue = ((ValueFormula<?>) f).value;
        String valueStr = (formulaValue instanceof StringValue) ? ((StringValue) formulaValue).value : formulaValue.toString();
        if (valueList.contains(valueStr))
          deriv.addFeature("denotation", "value_in_formula");
      }
    }
  }

  private void extractRootFeatures(Example ex, Derivation deriv, ArrayList<Item> inputItems,
      ArrayList<Item> derivItems) {
    if (!deriv.isRootCat()) return;
    if (!opts.featureDomains.contains("root") && !opts.featureDomains.contains("root_lexical")) return;

    //alignment features
    BipartiteMatcher bMatcher = new BipartiteMatcher();
    ArrayList<Item> filteredInputTokens = filterStopWords(inputItems);
    ArrayList<Item> filteredDerivTokens = filterStopWords(derivItems);

    // the bipartite matcher explodes if filtered input tokens is zero length
    int[] assignment;
    if (!filteredInputTokens.isEmpty())
      assignment = bMatcher.findMaxWeightAssignment(buildAlignmentMatrix(filteredInputTokens, filteredDerivTokens));
    else
      assignment = new int[0];

    if (opts.featureDomains.contains("root")) {
      //number of unmathced words based on exact match and ppdb
      int matches = 0;
      for (int i = 0; i < filteredInputTokens.size(); ++i) {
        if (assignment[i] != i) {
          matches++;
        }
      }
      deriv.addFeature("root", "unmatched_input", filteredInputTokens.size() - matches);
      deriv.addFeature("root", "unmatched_deriv", filteredDerivTokens.size() - matches);
      if (deriv.value != null) {
        if (deriv.value instanceof ListValue) {
          ListValue list = (ListValue) deriv.value;
          deriv.addFeature("root", String.format("pos0=%s&returnType=%s", ex.posTag(0), list.values.get(0).getClass()));
        }
      }
    }

    if (opts.featureDomains.contains("root_lexical")) {
      for (int i = 0; i < assignment.length; ++i) {
        if (assignment[i] == i) {
          if (i < filteredInputTokens.size()) {
            Item inputToken = filteredInputTokens.get(i);
            if (inputToken.ner1 != null)
              deriv.addFeature("root_lexical", "deleted_token=" + inputToken.ner1);
            else
              deriv.addFeature("root_lexical", "deleted_token=" + inputToken.data1);
          }
          else {
            Item derivToken = filteredDerivTokens.get(i - filteredInputTokens.size());
            // deriv tokens never get ner tags (because it would be too expensive to run ner on the
            // canonical utterance)
            deriv.addFeature("root_lexical", "deleted_token=" + derivToken.data1);
          }
        }
      }
    }
  }

  private List<Formula> getCallFormulas(Derivation deriv) {
    return deriv.formula.mapToList(formula -> {
      List<Formula> res = new ArrayList<>();
      if (formula instanceof CallFormula) {
        res.add(((CallFormula) formula).func);
      }
      return res;
    }, true);
  }
  private void extractLogicalFormFeatures(Example ex, Derivation deriv) {
    if (!opts.featureDomains.contains("lf")) return;
    for (int i = 0; i < ex.numTokens(); ++i) {
      List<Formula> callFormulas = getCallFormulas(deriv);
      if (ex.posTag(i).equals("JJS")) {
        if (ex.token(i).equals("least") || ex.token(i).equals("most")) //at least and at most are not what we want
          continue;
        for (Formula callFormula: callFormulas) {
          String callFormulaDesc = callFormula.toString();
          //LogInfo.logs("SUPER: utterance=%s, formula=%s", ex.utterance, deriv.formula);
          deriv.addFeature("lf", callFormulaDesc + "& superlative");
        }
      }
    }
    if (!opts.featureDomains.contains("simpleworld")) return;
    //specific handling of simple world methods
    if (deriv.formula instanceof CallFormula) {
      CallFormula callFormula = (CallFormula) deriv.formula;
      String desc = callFormula.func.toString();
      switch (desc) {
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.filter":
          deriv.addFeature("simpleworld", "filter&" + callFormula.args.get(1));
          break;
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.getProperty":
          deriv.addFeature("simpleworld", "getProperty&" + callFormula.args.get(1));
          break;
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.superlative":
          deriv.addFeature("simpleworld", "superlative&" + callFormula.args.get(1) + "&" + callFormula.args.get(2));
          break;
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.countSuperlative":
          deriv.addFeature("simpleworld", "countSuperlative&" + callFormula.args.get(1) + "&" + callFormula.args.get(2));
          break;
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.countComparative":
          deriv.addFeature("simpleworld", "countComparative&" + callFormula.args.get(2) + "&" + callFormula.args.get(1));
          break;
        case "edu.stanford.nlp.sempre.overnight.SimpleWorld.aggregate":
          deriv.addFeature("simpleworld", "countComparative&" + callFormula.args.get(0));
          break;
        default: break;
      }
    }
  }

  private void extractPhraseAlignmentFeatures(Example ex, Derivation deriv, List<Item> derivTokens) {

    if (!opts.featureDomains.contains("alignment")) return;

    //get the tokens
    Set<String> inputSubspans = ex.languageInfo.getLowerCasedSpans();

    for (int i = 0; i < derivTokens.size(); ++i) {
      for (int j = i + 1; j <= derivTokens.size() && j <= i + 4; ++j) {

        String lhs = Joiner.on(' ').join(derivTokens.subList(i, j).stream().map(item -> item.data1).iterator());
        if (entities.contains(lhs)) continue; //optimization

        if (phraseTable.containsKey(lhs)) {
          Set<String> rhsCandidates = phraseTable.get(lhs);
          Set<String> intersection = Sets.intersection(rhsCandidates, inputSubspans);
          for (String rhs: intersection) {
            addAndFilterLexicalFeature(deriv, "alignment", rhs, lhs);
          }
        }
      }
    }
  }

  private static Map<String, Set<String>> loadPhraseTable(File fromFile) {
    Map<String, Set<String>> res = new HashMap<>();
    int num = 0;
    for (String line : IOUtils.readLines(fromFile)) {
      String[] tokens = line.split("\t");
      if (tokens.length != 3) throw new RuntimeException("Bad alignment line: " + line);
      if (!res.containsKey(tokens[0]))
        res.put(tokens[0], new HashSet<>());

      double value = Double.parseDouble(tokens[2]);
      if (value >= opts.phraseTableThreshold) {
        res.get(tokens[0]).add(tokens[1]);
        num++;
      }
    }
    LogInfo.logs("Number of entries=%s", num);
    return res;
  }

  private static Map<String, Set<String>> loadPhraseTable(String languageTag) {
    // try path.languageTag, if that fails, read just path
    File withLanguage = new File(opts.phraseAlignmentPath + "." + languageTag);
    if (withLanguage.exists())
      return loadPhraseTable(withLanguage);
    else
      return loadPhraseTable(new File(opts.phraseAlignmentPath));
  }

  private void addAndFilterLexicalFeature(Derivation deriv, String domain, String str1, String str2) {

    String[] str1Tokens = str1.split("\\s+");
    String[] str2Tokens = str2.split("\\s+");
    for (String str1Token: str1Tokens)
      if (entities.contains(str1Token)) return;
    for (String str2Token: str2Tokens)
      if (entities.contains(str2Token)) return;

    if (stopWords.contains(str1) || stopWords.contains(str2)) return;
    deriv.addFeature(domain, str1 + "--" + str2);
  }

  private void addAndFilterLexicalFeature(Derivation deriv, String domain, Item str1, Item str2) {
    if ((str1.tag == Item.Tag.UNIGRAM && stopWords.contains(str1.data1)) ||
        (str2.tag == Item.Tag.UNIGRAM && stopWords.contains(str2.data1)))
      return;

    String f1 = str1.ner1 != null ? str1.ner1 : str1.data1;
    if (str1.tag != Item.Tag.UNIGRAM)
      f1 += " " + (str1.ner2 != null ? str1.ner2 : str1.data2);
    String f2 = str2.ner1 != null ? str2.ner1 : str2.data1;
    if (str2.tag != Item.Tag.UNIGRAM)
      f2 += " " + (str2.ner2 != null ? str2.ner2 : str2.data2);
    deriv.addFeature(domain, f1 + "--" + f2);
  }

  private void extractLexicalFeatures(Example ex, Derivation deriv, ArrayList<Item> inputItems,
      ArrayList<Item> derivItems) {

    if (!opts.featureDomains.contains("lexical")) return;

    //alignment features
    BipartiteMatcher bMatcher = new BipartiteMatcher();
    ArrayList<Item> filteredInputTokens = filterStopWords(inputItems);
    ArrayList<Item> filteredDerivTokens = filterStopWords(derivItems);

    // the bipartite matcher explodes if filtered input tokens is zero length
    if (filteredInputTokens.isEmpty())
      return;

    double[][] alignmentMatrix = buildLexicalAlignmentMatrix(filteredInputTokens, filteredDerivTokens);
    int[] assignment = bMatcher.findMaxWeightAssignment(alignmentMatrix);
    for (int i = 0; i < filteredInputTokens.size(); ++i) {
      if (assignment[i] != i) {
        int derivIndex = assignment[i] - filteredInputTokens.size();
        Item inputToken = filteredInputTokens.get(i);

        assert inputToken.tag == Item.Tag.UNIGRAM;
        if (entities.contains(inputToken.data1))
          continue; //optimization - stop here

        Item derivToken = filteredDerivTokens.get(derivIndex);
        if (!inputToken.equals(derivToken)) {
          addAndFilterLexicalFeature(deriv, "lexical", inputToken, derivToken);
          extractStringSimilarityFeatures(deriv, inputToken.data1, derivToken.data1);

          //2:2 features
          if (i < filteredInputTokens.size() - 1) {
            if (assignment[i + 1] == assignment[i] + 1) {
              Item inputBigram = new Item(Item.Tag.BIGRAM, inputToken, filteredInputTokens.get(i + 1));
              Item derivBigram = new Item(Item.Tag.BIGRAM, derivToken, filteredDerivTokens.get(derivIndex + 1));
              if (!inputBigram.equals(derivBigram)) {
                addAndFilterLexicalFeature(deriv, "lexical", inputBigram, derivBigram);
              }
            }
          }
          //1:2 features
          if (derivIndex > 0) {
            addAndFilterLexicalFeature(deriv, "lexical", inputToken,
                new Item(Item.Tag.BIGRAM, filteredDerivTokens.get(derivIndex - 1),
                    filteredDerivTokens.get(derivIndex)));
          }
          if (derivIndex < filteredDerivTokens.size() - 1) {
            addAndFilterLexicalFeature(deriv, "lexical", inputToken,
                new Item(Item.Tag.BIGRAM, filteredDerivTokens.get(derivIndex),
                    filteredDerivTokens.get(derivIndex + 1)));
          }
        }
      }
    }
  }

  private void extractStringSimilarityFeatures(Derivation deriv, String inputToken, String derivToken) {
    if (inputToken.startsWith(derivToken) || derivToken.startsWith(inputToken))
      deriv.addFeature("lexical", "starts_with");
    else if (inputToken.length() > 4 && derivToken.length() > 4) {
      if (inputToken.substring(0, 4).equals(derivToken.substring(0, 4)))
        deriv.addFeature("lexical", "common_prefix");
    }
  }

  //return a list without wtop words
  private ArrayList<Item> filterStopWords(ArrayList<Item> items) {
    ArrayList<Item> res = new ArrayList<>();
    for (Item token : items) {
      assert token.tag == Item.Tag.UNIGRAM;
      if (!stopWords.contains(token.data1))
        res.add(token);
    }
    return res;
  }

  private double[][] buildAlignmentMatrix(ArrayList<Item> inputTokens, ArrayList<Item> derivTokens) {

    double[][] res = new double[inputTokens.size() + derivTokens.size()][inputTokens.size() + derivTokens.size()];
    for (int i = 0; i < inputTokens.size(); ++i) {
      for (int j = 0; j < derivTokens.size(); ++j) {
        Item inputToken = inputTokens.get(i);
        Item derivToken = derivTokens.get(j);

        if (computeMatch(inputToken, derivToken) > 0d) {
          res[i][inputTokens.size() + j] = 1d;
          res[inputTokens.size() + j][i] = 1d;
        }
        else if (computeParaphrase(inputToken, derivToken) > 0d) {
          res[i][inputTokens.size() + j] = 0.5d;
          res[inputTokens.size() + j][i] = 0.5d;
        }
      }
    }
    for (int i = 0; i < res.length - 1; i++) {
      for (int j = i + 1; j < res.length; j++) {
        if (i != j && res[i][j] < 1) {
          res[i][j] = Double.NEGATIVE_INFINITY;
          res[j][i] = Double.NEGATIVE_INFINITY;
        }
      }
    }
    return res;
  }

  private double[][] buildLexicalAlignmentMatrix(ArrayList<Item> filteredInputTokens,
      ArrayList<Item> filteredDerivTokens) {
    double[][] res = new double[filteredInputTokens.size() + filteredDerivTokens.size()][filteredInputTokens.size()
        + filteredDerivTokens.size()];
    //init with -infnty and low score on the diagonal
    for (int i = 0; i < res.length - 1; i++) {
      for (int j = i; j < res.length; j++) {
        if (i == j) {
          res[i][j] = 0d;
          res[j][i] = 0d;
        } else {
          res[i][j] = -1000d;
          res[j][i] = -1000d;
        }
      }
    }

    for (int i = 0; i < filteredInputTokens.size(); ++i) {
      for (int j = 0; j < filteredDerivTokens.size(); ++j) {
        Item inputToken = filteredInputTokens.get(i);
        Item derivToken = filteredDerivTokens.get(j);

        if (computeMatch(inputToken, derivToken) > 0) {
          res[i][filteredInputTokens.size() + j] = 1d;
          res[filteredInputTokens.size() + j][i] = 1d;
        } else if (computeParaphrase(inputToken, derivToken) > 0) {
          res[i][filteredInputTokens.size() + j] = 0.5d;
          res[filteredInputTokens.size() + j][i] = 0.5d;
        } else {
          double product = aligner.getCondProb(inputToken.data1, derivToken.data1)
              * aligner.getCondProb(derivToken.data1, inputToken.data1);
          res[i][filteredInputTokens.size() + j] = product;
          res[filteredInputTokens.size() + j][i] = product;
        }
      }
    }
    return res;
  }

  // Fetch items from the temporary state.
  // If it doesn't exist, create one.
  private static ItemList getItems(Map<String, Object> tempState) {
    ItemList items = (ItemList) tempState.get("items");
    if (items == null)
      tempState.put("items", items = new ItemList());
    return items;
  }

  private void populateItems(List<String> tokens, List<String> nerTags, List<String> nerValues, ItemList items) {
    ArrayList<String> prunedTokens = new ArrayList<>();
    ArrayList<String> prunedStems = new ArrayList<>();
    ArrayList<String> prunedNer = new ArrayList<>();

    // Populate items with unpruned tokens
    String previousUnigram = null;
    String previousStem = null;
    String previousNer = null;

    for (int i = 0; i < tokens.size(); i++) {
      String unigram = tokens.get(i).toLowerCase();
      String stem = LanguageUtils.stem(unigram);

      String ner = null;
      if (nerTags != null && nerValues != null) {
        if (nerValues.get(i) != null)
          ner = nerTags.get(i);
      }
      items.unigrams.add(new Item(Item.Tag.UNIGRAM, unigram, stem, ner));
      if (i > 0)
        items.bigrams.add(new Item(Item.Tag.BIGRAM, previousUnigram, previousStem, previousNer, unigram, stem, ner));

      if (!isStopWord(unigram) || (i > 0 && (previousUnigram.equals('`') || previousUnigram.equals("``")))) {
        prunedTokens.add(unigram);
        prunedStems.add(stem);
        prunedNer.add(ner);
      }

      previousUnigram = unigram;
      previousStem = stem;
      previousNer = ner;
    }

    // Populate items with skip words removed
    previousUnigram = null;
    previousStem = null;
    previousNer = null;
    if (prunedTokens.size() > 0) {
      previousUnigram = prunedTokens.get(0);
      previousStem = prunedStems.get(0);
      previousNer = prunedNer.get(0);
    }
    for (int i = 1; i < prunedTokens.size(); i++) {
      String unigram = prunedTokens.get(i);
      String stem = prunedStems.get(i);
      String ner = prunedNer.get(i);
      items.skipBigrams.add(new Item(Item.Tag.SKIP_BIGRAM, previousUnigram, previousStem, previousNer,
          unigram, stem, ner));
      previousUnigram = unigram;
      previousStem = stem;
    }

    items.valid = true;
  }

  // Compute the items for the input utterance.
  private ItemList computeInputItems(Example ex) {
    ItemList items = getItems(ex.getTempState());
    if (items.valid)
      return items;
    populateItems(ex.getTokens(), ex.languageInfo.nerTags, ex.languageInfo.nerValues, items);
    LogInfo.logs("input %s, items %s", ex.utterance, items);
    return items;
  }

  // Return the set of tokens (partial canonical utterance) produced by the
  // derivation.
  public List<String> extractTokens(Example ex, Derivation deriv, List<String> tokens) {
    tokens.addAll(Arrays.asList(deriv.canonicalUtterance.split("\\s+")));
    return tokens;
  }

  // Compute the items for a partial canonical utterance.
  private ItemList computeCandidateItems(Example ex, Derivation deriv) {
    // Get tokens
    ArrayList<String> tokens = new ArrayList<>();
    extractTokens(ex, deriv, tokens);
    // Compute items
    ItemList items = new ItemList();
    populateItems(tokens, null, null, items);
    return items;
  }

  private static void subtractDescendentsFeatures(Derivation deriv, Derivation subderiv) {
    if (subderiv.children != null) {
      for (Derivation child : subderiv.children) {
        deriv.getLocalFeatureVector().add(-1, child.getLocalFeatureVector(), new FeatureMatcher() {
          @Override
          public boolean matches(String feature) {
            return feature.startsWith("paraphrase :: ") || feature.startsWith("lexical :: ")
                || feature.startsWith("alignment :: ");
          }

        });
        subtractDescendentsFeatures(deriv, child);
      }
    }
  }

  // Return the "complexity" of the given derivation.
  private static int derivationSize(Derivation deriv) {
    if (deriv.rule.isAnchored())
      return 1;
    int sum = 0;
    if (opts.countIntermediate || !(deriv.rule.lhs.contains("Intermediate"))) sum++;
    if (deriv.children != null) {
      for (Derivation child : deriv.children)
        sum += derivationSize(child);
    }
    return sum;
  }

  private static double computeMatch(Item a, Item b) {
    if (a.equals(b))
      return 1;
    if (a.stemEquals(b))
      return 1;
    return 0;
  }

  private double computeParaphrase(Item a, Item b) {
    if (computeMatch(a, b) >  0) return 0;

    // we know we never compare items of different types
    assert a.tag == b.tag;

    int numPpdb = 0;
    // if we're comparing unigrams, we don't need to check a.data1 == b.data1
    // because we just did it earlier when we called computeMatch
    if (a.tag == Item.Tag.UNIGRAM || (!a.data1.equals(b.data1) && !a.stem1.equals(b.stem1))) {
      if (ppdbModel.get(a.data1, b.data1) > 0 || ppdbModel.get(a.stem1, b.stem1) > 0) {
        numPpdb++;
      } else {
        return 0;
      }
    }
    if (a.tag != Item.Tag.UNIGRAM) {
      if (!a.data2.equals(b.data2) && !a.stem2.equals(b.stem2)) {
        if (ppdbModel.get(a.data2, b.data2) > 0 || ppdbModel.get(a.stem2, b.stem2) > 0) {
          numPpdb++;
        } else {
          return 0;
        }
      }
    }
    return numPpdb <= 1 ? 1d : 0d;
  }
}
