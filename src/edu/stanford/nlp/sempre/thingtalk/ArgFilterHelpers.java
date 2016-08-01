package edu.stanford.nlp.sempre.thingtalk;

import edu.stanford.nlp.sempre.Value;

class ArgFilterHelpers {
  private ArgFilterHelpers() {
  }

  static boolean typeOk(String have, String want) {
    if (have.equals(want))
      return true;

    // a mistake in the naming that is in too many places
    // to fix now
    if (have.equals("Bool") && want.equals("Boolean"))
      return true;

    // FIXME be stricter in handling measures
    // (not a problem for now because we only parse temperatures)
    if (have.equals("Measure") && want.startsWith("Measure("))
      return true;

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

  static boolean typeOkArray(String have, String argtype) {
    if (!argtype.startsWith("Array("))
      return false;

    // remove initial Array( and final )
    String eltype = argtype.substring("Array(".length(), argtype.length() - 1);
    return typeOk(have, eltype);
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
      return typeOkArray(pv.tt_type, pv.name.type);
    return typeOk(pv.tt_type, pv.name.type) &&
        operatorOk(pv.tt_type, pv.operator);
  }

}
