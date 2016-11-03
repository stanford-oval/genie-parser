package edu.stanford.nlp.sempre;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;

import fig.basic.*;
import gnu.trove.impl.Constants;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

/**
 * Params contains the parameters of the model. Currently consists of a map from
 * features to weights.
 *
 * @author Percy Liang
 */
public class Params {
  public static class Options {
    @Option(gloss = "By default, all features have this weight")
    public double defaultWeight = 0;
    @Option(gloss = "Randomly initialize the weights")
    public boolean initWeightsRandomly = false;
    @Option(gloss = "Randomly initialize the weights")
    public Random initRandom = new Random(1);

    @Option(gloss = "Initial step size") public double initStepSize = 1;
    @Option(gloss = "How fast to reduce the step size")
    public double stepSizeReduction = 0;
    @Option(gloss = "Use the AdaGrad algorithm (different step size for each coordinate)")
    public boolean adaptiveStepSize = true;
    @Option(gloss = "Use dual averaging") public boolean dualAveraging = false;
    @Option(gloss = "Whether to do lazy l1 reg updates") public String l1Reg = "none";
    @Option(gloss = "L1 reg coefficient") public double l1RegCoeff = 0d;
    @Option(gloss = "Lazy L1 full update frequency") public int lazyL1FullUpdateFreq = 5000;
  }
  public static Options opts = new Options();
  public enum L1Reg {
    LAZY,
    NONLAZY,
    NONE;
  }
  private L1Reg parseReg(String l1Reg) {
    if ("lazy".equals(l1Reg)) return L1Reg.LAZY;
    if ("nonlazy".equals(l1Reg)) return L1Reg.NONLAZY;
    if ("none".equals(l1Reg)) return L1Reg.NONE;
    throw new RuntimeException("not legal l1reg");
  }

  private final L1Reg l1Reg = parseReg(opts.l1Reg);

  // Discriminative weights
  private final TObjectDoubleMap<String> weights = new TObjectDoubleHashMap<>(Constants.DEFAULT_CAPACITY,
      Constants.DEFAULT_LOAD_FACTOR, opts.defaultWeight);

  // For AdaGrad
  private final TObjectDoubleMap<String> sumSquaredGradients = new TObjectDoubleHashMap<>();

  // For dual averaging
  private final TObjectDoubleMap<String> sumGradients = new TObjectDoubleHashMap<>();

  // Number of stochastic updates we've made so far (for determining step size).
  private int numUpdates;

  // for lazy l1-reg update
  private final TObjectIntMap<String> l1UpdateTimeMap = new TObjectIntHashMap<>();

  // multi-thread synchronization
  private final ReadWriteLock lock = new ReentrantReadWriteLock(true);

  public void readLock() {
    lock.readLock().lock();
  }

  public void readUnlock() {
    lock.readLock().unlock();
  }

  private void writeLock() {
    lock.writeLock().lock();
  }

  private void writeUnlock() {
    lock.writeLock().unlock();
  }

  // Initialize the weights
  public void init(List<Pair<String, Double>> initialization) {
    writeLock();
    try {
      if (!weights.isEmpty())
        throw new RuntimeException("Initialization is not legal when there are non-zero weights");
      for (Pair<String, Double> pair : initialization)
        weights.put(pair.getFirst(), pair.getSecond());
    } finally {
      writeUnlock();
    }
  }

  // Read parameters from |path|.
  public void read(String path) {
    LogInfo.begin_track("Reading parameters from %s", path);
    try (BufferedReader in = IOUtils.openIn(path)) {
      String line;
      writeLock();
      try {
        while ((line = in.readLine()) != null) {
          String[] pair = Lists.newArrayList(Splitter.on('\t').split(line)).toArray(new String[2]);
          weights.put(pair[0], Double.parseDouble(pair[1]));
        }
      } finally {
        writeUnlock();
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    LogInfo.logs("Read %s weights", weights.size());
    LogInfo.end_track();
  }

  // Read parameters from |path|.
  public void read(String path, String prefix) {
    LogInfo.begin_track("Reading parameters from %s", path);
    try (BufferedReader in = IOUtils.openIn(path)) {
      String line;
      writeLock();
      try {
        while ((line = in.readLine()) != null) {
          String[] pair = Lists.newArrayList(Splitter.on('\t').split(line)).toArray(new String[2]);
          weights.put(pair[0], Double.parseDouble(pair[1]));
          weights.put(prefix + pair[0], Double.parseDouble(pair[1]));
        }
      } finally {
        writeUnlock();
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    LogInfo.logs("Read %s weights", weights.size());
    LogInfo.end_track();
  }

  // Update weights by adding |gradient| (modified appropriately with step size).
  public void update(TObjectDoubleMap<String> gradient) {
    writeLock();
    try {
      gradient.forEachEntry((f, g) -> {
        if (g * g == 0)
          return true;  // In order to not divide by zero

        if (l1Reg == L1Reg.LAZY)
          lazyL1Update(f);
        double stepSize = computeStepSize(f, g);

        if (opts.dualAveraging) {
          if (!opts.adaptiveStepSize && opts.stepSizeReduction != 0)
            throw new RuntimeException("Dual averaging not supported when " +
                "step-size changes across iterations for " +
                "features for which the gradient is zero");
          sumGradients.adjustOrPutValue(f, g, g);
          weights.put(f, stepSize * sumGradients.get(f));
        } else {
          if (stepSize * g == Double.POSITIVE_INFINITY || stepSize * g == Double.NEGATIVE_INFINITY) {
            LogInfo.logs("WEIRD FEATURE UPDATE: feature=%s, currentWeight=%s, stepSize=%s, gradient=%s", f,
                getWeight(f), stepSize, g);
            throw new RuntimeException("Gradient absolute value is too large or too small");
          }
          weights.adjustOrPutValue(f, stepSize * g, stepSize * g);
          if (l1Reg == L1Reg.LAZY)
            l1UpdateTimeMap.put(f, numUpdates);
        }
        return true;
      });
      // non lazy implementation goes over all weights
      if (l1Reg == L1Reg.NONLAZY) {
        Set<String> features = new HashSet<>(weights.keySet());
        for (String f : features) {
          double stepSize = computeStepSize(f, 0d); // no update for gradient here
          double update = opts.l1RegCoeff * -Math.signum(weights.get(f));
          clipUpdate(f, stepSize * update);
        }
      }
      numUpdates++;
      if (l1Reg == L1Reg.LAZY && opts.lazyL1FullUpdateFreq > 0 && numUpdates % opts.lazyL1FullUpdateFreq == 0) {
        LogInfo.begin_track("Fully apply L1 regularization.");
        finalizeWeights();
        System.gc();
        LogInfo.end_track();
      }
    } finally {
      writeUnlock();
    }
  }

  private double computeStepSize(String feature, double gradient) {
    if (opts.adaptiveStepSize) {
      sumSquaredGradients.adjustOrPutValue(feature, gradient * gradient, gradient * gradient);
      // ugly - adding one to the denominator when using l1 reg.
      if (l1Reg != L1Reg.NONE)
        return opts.initStepSize / (Math.sqrt(sumSquaredGradients.get(feature) + 1));
      else
        return opts.initStepSize / Math.sqrt(sumSquaredGradients.get(feature));
    } else {
      return opts.initStepSize / Math.pow(numUpdates, opts.stepSizeReduction);
    }
  }

  private static <K> double getDouble(TObjectDoubleMap<K> map, K key, double defaultValue) {
    if (map.containsKey(key))
      return map.get(key);
    else
      return defaultValue;
  }

  /*
   * If the update changes the sign, remove the feature
   */
  private void clipUpdate(String f, double update) {
    double currWeight = getDouble(weights, f, 0);
    if (currWeight == 0)
      return;

    if (currWeight * (currWeight + update) < 0.0)  {
      weights.remove(f);
    } else {
      weights.adjustOrPutValue(f, update, update);
    }
  }

  private void lazyL1Update(String f) {
    if (weights.get(f) == 0)
      return;
    // For pre-initialized weights, which have no updates yet
    if (!sumSquaredGradients.containsKey(f) || !l1UpdateTimeMap.containsKey(f)) {
      l1UpdateTimeMap.put(f, numUpdates);
      sumSquaredGradients.put(f, 0.0);
      return;
    }
    int numOfIter = numUpdates - l1UpdateTimeMap.get(f);
    if (numOfIter == 0) return;
    if (numOfIter < 0) throw new RuntimeException("l1UpdateTimeMap is out of sync.");

    double stepSize = (numOfIter * opts.initStepSize) / (Math.sqrt(sumSquaredGradients.get(f) + 1));
    double update = -opts.l1RegCoeff * Math.signum(getDouble(weights, f, 0.0));
    clipUpdate(f, stepSize * update);
    if (weights.containsKey(f))
      l1UpdateTimeMap.put(f, numUpdates);
    else
      l1UpdateTimeMap.remove(f);
  }

  // must be called with read lock held
  public double getWeight(String f) {
    if (l1Reg == L1Reg.LAZY)
      lazyL1Update(f);
    if (opts.initWeightsRandomly) {
      if (!weights.containsKey(f))
        weights.put(f, 2 * opts.initRandom.nextDouble() - 1);
      return weights.get(f);
    } else {
      return weights.get(f);
    }
  }

  // must be called with read lock held
  public Map<String, Double> getWeights() {
    final Map<String, Double> hashMap = new HashMap<>();
    weights.forEachEntry((feature, value) -> {
      hashMap.put(feature, value);
      return true;
    });
    return hashMap;
  }

  public void write(PrintWriter out) { write(null, out); }

  public void write(String prefix, PrintWriter out) {
    readLock();
    try {
      List<Map.Entry<String, Double>> entries = new ArrayList<>(getWeights().entrySet());
      Collections.sort(entries, new ValueComparator<String, Double>(true));
      for (Map.Entry<String, Double> entry : entries) {
        double value = entry.getValue();
        out.println((prefix == null ? "" : prefix + "\t") + entry.getKey() + "\t" + value);
      }
    } finally {
      readUnlock();
    }
  }

  public void write(String path) {
    LogInfo.begin_track("Params.write(%s)", path);
    try (PrintWriter out = IOUtils.openOutHard(path)) {
      write(out);
    }
    LogInfo.end_track();
  }

  public void log() {
    LogInfo.begin_track("Params");
    List<Map.Entry<String, Double>> entries = new ArrayList<>(getWeights().entrySet());
    Collections.sort(entries, new ValueComparator<String, Double>(true));
    for (Map.Entry<String, Double> entry : entries) {
      double value = entry.getValue();
      LogInfo.logs("%s\t%s", entry.getKey(), value);
    }
    LogInfo.end_track();
  }

  public void finalizeWeights() {
    writeLock();
    try {
      if (l1Reg == L1Reg.LAZY) {
        Set<String> features = new HashSet<>(weights.keySet());
        for (String f : features)
          lazyL1Update(f);
      }
    } finally {
      writeUnlock();
    }
  }

  public Params copyParams()  {
    readLock();
    try {
      Params result = new Params();
      for (String feature : this.weights.keySet()) {
        result.weights.put(feature, this.getWeight(feature));
      }
      return result;
    } finally {
      readUnlock();
    }
  }

  // copy params starting with prefix and drop the prefix
  public Params copyParamsByPrefix(String prefix)  {
    Params result = new Params();
    readLock();
    try {
      for (String feature : this.getWeights().keySet()) {
        if (feature.startsWith(prefix)) {
          String newFeature = feature.substring(prefix.length());
          result.weights.put(newFeature, this.getWeight(feature));
        }
      }
      return result;
    } finally {
      readUnlock();
    }
  }

  public boolean isEmpty() {
    readLock();
    try {
      return weights.size() == 0;
    } finally {
      readUnlock();
    }
  }

  public Params getRandomWeightParams()  {
    Random rand = new Random();
    Params result = new Params();
    for (String feature : this.getWeights().keySet()) {
      result.weights.put(feature, 2 * rand.nextDouble() - 1); // between -1 and 1
    }
    return result;
  }
}
