package edu.stanford.nlp.sempre.corenlp;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.sempre.LanguageAnalyzer;
import edu.stanford.nlp.sempre.LanguageInfo;
import edu.stanford.nlp.sempre.LanguageInfo.DependencyEdge;
import edu.stanford.nlp.sempre.SempreUtils;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.util.CoreMap;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Utils;

/**
 * CoreNLPAnalyzer uses Stanford CoreNLP pipeline to analyze an input string utterance
 * and return a LanguageInfo object
 *
 * @author akchou
 */
public class CoreNLPAnalyzer extends LanguageAnalyzer {
  public static class Options {
    @Option(gloss = "What CoreNLP annotators to run")
    public List<String> annotators = Lists.newArrayList("ssplit", "pos", "lemma", "ner", "parse");

    @Option(gloss = "Whether to use case-sensitive models")
    public boolean caseSensitive = false;

    @Option(gloss = "What language to use (as a two letter tag)")
    public String languageTag = "en";

    @Option(gloss = "Additional named entity recognizers to run")
    public List<String> entityRecognizers = new ArrayList<>();

    @Option(gloss = "Additional regular expressions to apply to tokens")
    public List<String> regularExpressions = new ArrayList<>();

    @Option(gloss = "Ignore DATE tags on years (numbers between 1000 and 3000) and parse them as numbers")
    public boolean yearsAsNumbers = false;

    @Option(gloss = "Whether to split hyphens or not")
    public boolean splitHyphens = true;
  }

  public static Options opts = new Options();

  // TODO(pliang): don't muck with the POS tag; instead have a separate flag
  // for isContent which looks at posTag != "MD" && lemma != "be" && lemma !=
  // "have"
  // Need to update TextToTextMatcher
  private static final String[] AUX_VERB_ARR = new String[] {"is", "are", "was",
      "were", "am", "be", "been", "will", "shall", "have", "has", "had",
      "would", "could", "should", "do", "does", "did", "can", "may", "might",
      "must", "seem" };
  private static final Set<String> AUX_VERBS = new HashSet<>(Arrays.asList(AUX_VERB_ARR));
  private static final String AUX_VERB_TAG = "VBD-AUX";

  private static final Set<String> NOT_A_NUMBER = Sets.newHashSet("9gag");

  private static final Pattern INTEGER_PATTERN = Pattern.compile("[0-9]+");

  private final String languageTag;
  private final StanfordCoreNLP pipeline;
  private final NamedEntityRecognizer[] extraRecognizers;

  public CoreNLPAnalyzer() {
    this(opts.languageTag);
  }

  public CoreNLPAnalyzer(String languageTag) {
    this.languageTag = languageTag;

    Properties props = new Properties();
    
    switch (languageTag) {
    case "en":
    case "en_US":
      if (opts.caseSensitive) {
        props.put("pos.model",
            "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
        props.put("ner.model",
            "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.conll.4class.distsim.crf.ser.gz");
      } else {
        props.put("pos.model", "edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger");
        props.put("ner.model",
            "edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz,edu/stanford/nlp/models/ner/english.conll.4class.caseless.distsim.crf.ser.gz");
      }
      break;

    case "de":
      loadResource("StanfordCoreNLP-german.properties", props);
      if (!opts.caseSensitive) {
        props.put("pos.model", "edu/stanford/nlp/models/pos-tagger/german/german-fast-caseless.tagger");
      }
      break;

    case "fr":
      loadResource("StanfordCoreNLP-french.properties", props);
      break;

    case "zh":
      loadResource("StanfordCoreNLP-chinese.properties", props);
      break;

    case "es":
      loadResource("StanfordCoreNLP-spanish.properties", props);
      break;

    default:
      LogInfo.logs("Unrecognized language %s, analysis will not work!", languageTag);
    }

    String annotators = Joiner.on(',').join(opts.annotators);
    if (languageTag.equals("zh"))
      props.put("annotators", "segment," + annotators);
    else
      props.put("annotators", "tokenize," + annotators);

    // force the numeric classifiers on, even if the props file would say otherwise
    // this is to make sure we can understands at least numbers in number form
    props.put("ner.applyNumericClassifiers", "true");
    props.put("ner.useSUTime", "true");

    pipeline = new StanfordCoreNLP(props);

    extraRecognizers = new NamedEntityRecognizer[opts.entityRecognizers.size() + opts.regularExpressions.size()];
    for (int i = 0; i < opts.entityRecognizers.size(); i++)
      extraRecognizers[i] = (NamedEntityRecognizer) Utils
          .newInstanceHard(SempreUtils.resolveClassName(opts.entityRecognizers.get(i)));
    for (int i = 0; i < opts.regularExpressions.size(); i++) {
      String spec = opts.regularExpressions.get(i);
      int split = spec.indexOf(':');
      extraRecognizers[opts.entityRecognizers.size() + i] = new RegexpEntityRecognizer(spec.substring(0, split),
          spec.substring(split + 1));
    }
  }

  private static void loadResource(String name, Properties into) {
    try {
      InputStream stream = Thread.currentThread().getContextClassLoader().getResourceAsStream(name);
      into.load(stream);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  // Stanford tokenizer doesn't break hyphens.
  // Replace hypens with spaces for utterances like
  // "Spanish-speaking countries" but not for "2012-03-28".
  // Also not break hyphens with spaces for things like 1-800-GOT-MILK
  public static String breakHyphens(String utterance) {
    StringBuilder buf = new StringBuilder(utterance);

    boolean seenHyphen = false;
    for (int i = 1; i < buf.length() - 1; i++) {
      char c = buf.charAt(i);
      if (c == '-') {
        if (!seenHyphen && Character.isLetter(buf.charAt(i - 1)) && Character.isLetter(buf.charAt(i + 1)))
          buf.setCharAt(i, ' ');
        else
          seenHyphen = true;
      } else if (Character.isWhitespace(c))
        seenHyphen = false;
    }
    return buf.toString();
  }

  @Override
  public LanguageInfo analyze(String utterance) {
    LanguageInfo languageInfo = new LanguageInfo();

    // Clear these so that analyze can hypothetically be called
    // multiple times.
    languageInfo.tokens.clear();
    languageInfo.posTags.clear();
    languageInfo.nerTags.clear();
    languageInfo.nerValues.clear();
    languageInfo.lemmaTokens.clear();
    languageInfo.dependencyChildren.clear();

    // Break hyphens
    if (opts.splitHyphens)
      utterance = breakHyphens(utterance);

    // Run Stanford CoreNLP
    Annotation annotation = pipeline.process(utterance);

    for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
      String word = token.get(TextAnnotation.class);
      String wordLower = word.toLowerCase();
      String nerTag = token.get(NamedEntityTagAnnotation.class);
      if (nerTag == null)
        nerTag = "O";
      String nerValue = token.get(NormalizedNamedEntityTagAnnotation.class);
      if(nerValue == null) {
        Timex nerValue_ = token.get(TimeAnnotations.TimexAnnotation.class);
        if(nerValue_ != null) {
          nerValue = nerValue_.value();
        }
      }
      String posTag = token.get(PartOfSpeechAnnotation.class);

      if (opts.yearsAsNumbers && nerTag.equals("DATE") && INTEGER_PATTERN.matcher(nerValue).matches()) {
        nerTag = "NUMBER";
      } else if (nerTag.equals("NUMBER") && NOT_A_NUMBER.contains(wordLower)) {
        nerTag = "O";
        posTag = "NNP";
        nerValue = null;
      }

      if (LanguageAnalyzer.opts.lowerCaseTokens) {
        languageInfo.tokens.add(wordLower);
      } else {
        languageInfo.tokens.add(word);
      }
      if (languageTag.equals("en")) {
        languageInfo.posTags.add(AUX_VERBS.contains(wordLower) ? AUX_VERB_TAG : posTag);
      } else {
        languageInfo.posTags.add(token.get(PartOfSpeechAnnotation.class));
      }
      languageInfo.lemmaTokens.add(token.get(LemmaAnnotation.class));
      languageInfo.nerTags.add(nerTag);
      languageInfo.nerValues.add(nerValue);
    }

    // Run additional entity recognizers
    for (NamedEntityRecognizer r : extraRecognizers)
      r.recognize(languageInfo);

    // Fills in a stanford dependency graph for constructing a feature
    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      SemanticGraph ccDeps = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
      if (ccDeps == null) continue;
      int sentenceBegin = sentence.get(CoreAnnotations.TokenBeginAnnotation.class);

      // Iterate over all tokens and their dependencies
      for (int sourceTokenIndex = sentenceBegin;
          sourceTokenIndex < sentence.get(CoreAnnotations.TokenEndAnnotation.class); sourceTokenIndex++) {
        final ArrayList<DependencyEdge> outgoing = new ArrayList<>();
        languageInfo.dependencyChildren.add(outgoing);
        IndexedWord node = ccDeps.getNodeByIndexSafe(sourceTokenIndex - sentenceBegin + 1);  // + 1 for ROOT
        if (node != null) {
          for (SemanticGraphEdge edge : ccDeps.outgoingEdgeList(node)) {
            final String relation = edge.getRelation().toString();
            final int targetTokenIndex = sentenceBegin + edge.getTarget().index() - 1;
            outgoing.add(new DependencyEdge(relation, targetTokenIndex));
          }
        }
      }
    }
    return languageInfo;
  }

  // Test on example sentence.
  public static void main(String[] args) {
    CoreNLPAnalyzer analyzer = new CoreNLPAnalyzer();
    while (true) {
      try {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("Enter some text:");
        String text = reader.readLine();
        LanguageInfo langInfo = analyzer.analyze(text);
        LogInfo.begin_track("Analyzing \"%s\"", text);
        LogInfo.logs("tokens: %s", langInfo.tokens);
        LogInfo.logs("lemmaTokens: %s", langInfo.lemmaTokens);
        LogInfo.logs("posTags: %s", langInfo.posTags);
        LogInfo.logs("nerTags: %s", langInfo.nerTags);
        LogInfo.logs("nerValues: %s", langInfo.nerValues);
        LogInfo.logs("dependencyChildren: %s", langInfo.dependencyChildren);
        LogInfo.end_track();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }
}
