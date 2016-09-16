package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.*;

public class JsonValueEvaluator implements ValueEvaluator {
  private final ExactValueEvaluator exact;

  public JsonValueEvaluator() {
    exact = new ExactValueEvaluator();
  }

  @SuppressWarnings("unchecked")
  private Set<Object> normalize(List<?> list) {
    ListIterator<?> li = list.listIterator();
    Set<Object> res = new HashSet<>();

    while (li.hasNext()) {
      Object o = li.next();
      if (o instanceof Number && !(o instanceof Double)) {
        res.add(((Number) o).doubleValue());
      } else if (o instanceof Map<?, ?>) {
        normalize((Map<?, Object>) o);
        res.add(o);
      } else if (o instanceof List<?>) {
        res.add(normalize((List<Object>) o));
      } else {
        res.add(o);
      }
    }

    return res;
  }

  @SuppressWarnings("unchecked")
  private void normalize(Map<?, Object> map) {
    for (Map.Entry<?, Object> e : map.entrySet()) {
      Object v = e.getValue();
      if (v instanceof Number && !(v instanceof Double))
        e.setValue(((Number) v).doubleValue());
      else if (v instanceof List<?>)
        e.setValue(normalize((List<?>) v));
      else if (v instanceof Map<?, ?>)
        normalize((Map<?, Object>) v);
    }
  }

  @Override
  public double getCompatibility(Value target, Value pred) {
    if (!(target instanceof StringValue) || !(pred instanceof StringValue))
      return exact.getCompatibility(target, pred);

    String targetString = ((StringValue) target).value;
    String predString = ((StringValue) pred).value;

    Map<String, Object> targetJson = Json.readMapHard(targetString);
    normalize(targetJson);
    Map<String, Object> predJson = Json.readMapHard(predString);
    normalize(predJson);

    return targetJson.equals(predJson) ? 1 : 0;
  }

  public static void main(String[] args) {
    try (Scanner scanner = new Scanner(System.in)) {
      JsonValueEvaluator eval = new JsonValueEvaluator();

      while (scanner.hasNext()) {
        String one = scanner.nextLine();
        String two = scanner.nextLine();

        System.out.println(eval.getCompatibility(new StringValue(one), new StringValue(two)));
      }
    }
  }
}
