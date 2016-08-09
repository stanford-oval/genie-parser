package edu.stanford.nlp.sempre.thingtalk;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;

import edu.stanford.nlp.sempre.NumberValue;
import edu.stanford.nlp.sempre.Value;

class ArgFilterHelpers {
  private ArgFilterHelpers() {
  }

  private static final Set<String> TIME_UNITS = Sets.newHashSet("ms", "s", "min", "h", "day", "week", "month", "year");
  private static final Set<String> TEMP_UNITS = Sets.newHashSet("C", "F");
  private static final Map<String, Set<String>> ALLOWED_UNITS = new HashMap<>();
  static {
    ALLOWED_UNITS.put("ms", TIME_UNITS);
    ALLOWED_UNITS.put("C", TEMP_UNITS);
  }

  private static boolean unitOk(String have, String want) {
    return ALLOWED_UNITS.get(want).contains(have);
  }

  static boolean typeOk(String have, String want, Value value) {
    if (have.equals(want))
      return true;

    // a mistake in the naming that is in too many places
    // to fix now
    if (have.equals("Bool") && want.equals("Boolean"))
      return true;

    // FIXME be stricter in handling measures
    // (not a problem for now because we only parse temperatures)
    if (have.equals("Measure") && want.startsWith("Measure(") && value instanceof NumberValue)
      return unitOk(((NumberValue) value).unit, want.substring("Measure(".length(), want.length() - 1));

    // time gets converted to String by Sabrina
    if (have.equals("Time") && want.equals("String"))
      return true;

    // String and Picture are the same type for compat with
    // type annotations that were written before Picture existed
    if ((have.equals("String") && want.equals("Picture")) ||
        (have.equals("Picture") && want.equals("String")))
      return true;

    return false;
  }

  static boolean typeOkArray(String have, String argtype, Value value) {
    if (!argtype.startsWith("Array("))
      return false;

    // remove initial Array( and final )
    String eltype = argtype.substring("Array(".length(), argtype.length() - 1);
    return typeOk(have, eltype, value);
  }

  private static boolean operatorOk(String type, String operator) {
    switch (operator) {
    case "is":
      return true;
    case "contains":
      return type.equals("String");
    case ">":
    case "<":
      return type.equals("Number") || type.equals("Measure");
    default:
      throw new RuntimeException("Unexpected operator " + operator);
    }
  }

  static boolean valueOk(Value value) {
    if (!(value instanceof ParamValue))
      return true;

    ParamValue pv = (ParamValue) value;

    if (pv.operator.equals("has"))
      return typeOkArray(pv.tt_type, pv.name.type, pv.value);
    return typeOk(pv.tt_type, pv.name.type, pv.value) &&
        operatorOk(pv.tt_type, pv.operator);
  }

}
