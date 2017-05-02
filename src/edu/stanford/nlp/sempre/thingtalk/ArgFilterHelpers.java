package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import com.google.common.collect.Sets;

import edu.stanford.nlp.sempre.NumberValue;
import edu.stanford.nlp.sempre.Value;

public class ArgFilterHelpers {
  private ArgFilterHelpers() {
  }

  private static final Set<String> TIME_UNITS = Sets.newHashSet("ms", "s", "min", "h", "day", "week", "month", "year");
  private static final Set<String> TEMP_UNITS = Sets.newHashSet("C", "F");
  private static final Set<String> LENGTH_UNITS = Sets.newHashSet("m", "km", "mm", "cm", "mi", "in", "ft");
  private static final Set<String> SPEED_UNITS = Sets.newHashSet("mps", "kmph", "mph");
  private static final Set<String> WEIGHT_UNITS = Sets.newHashSet("kg", "g", "lb", "oz");
  private static final Set<String> PRESSURE_UNITS = Sets.newHashSet("Pa", "bar", "psi", "mmHg", "inHg", "atm");
  private static final Set<String> ENERGY_UNITS = Sets.newHashSet("kcal", "kJ");
  private static final Set<String> HEARTRATE_UNITS = Collections.singleton("bpm");
  private static final Set<String> FILESIZE_UNITS = Sets.newHashSet("byte", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB",
      "TiB");
  private static final Map<String, Set<String>> ALLOWED_UNITS = new HashMap<>();
  private static final Set<String> ALL_UNITS = new HashSet<>();
  private static final Map<String, String> ALL_LOWERCASE_UNITS = new HashMap<>();
  static {
    ALLOWED_UNITS.put("ms", TIME_UNITS);
    ALLOWED_UNITS.put("C", TEMP_UNITS);
    ALLOWED_UNITS.put("m", LENGTH_UNITS);
    ALLOWED_UNITS.put("mps", SPEED_UNITS);
    ALLOWED_UNITS.put("kg", WEIGHT_UNITS);
    ALLOWED_UNITS.put("mmHg", PRESSURE_UNITS);
    ALLOWED_UNITS.put("kcal", ENERGY_UNITS);
    ALLOWED_UNITS.put("bpm", HEARTRATE_UNITS);
    ALLOWED_UNITS.put("byte", FILESIZE_UNITS);

    ALLOWED_UNITS.forEach((type, units) -> {
      for (String unit : units) {
        ALL_UNITS.add(unit);
        ALL_LOWERCASE_UNITS.put(unit.toLowerCase(), unit);
      }
    });
  }

  public static boolean isUnit(String unit) {
    return ALL_UNITS.contains(unit);
  }

  public static String getUnitCaseless(String unit) {
    return ALL_LOWERCASE_UNITS.get(unit);
  }

  public static boolean isTimeUnit(String unit) {
    return TIME_UNITS.contains(unit);
  }

  private static boolean unitOk(String have, String want) {
    if (!ALLOWED_UNITS.containsKey(want))
      throw new RuntimeException("Invalid required unit " + want);
    return ALLOWED_UNITS.get(want).contains(have);
  }

  private static boolean isEntity(String type) {
    if (type.startsWith("Entity("))
      return true;

    return type.equals("Username") || type.equals("Hashtag");
  }

  private static final Map<String, String> RENAMED_ENTITY_TYPES = new HashMap<>();

  static {
    RENAMED_ENTITY_TYPES.put("Username", "tt:username");
    RENAMED_ENTITY_TYPES.put("Hashtag", "tt:hashtag");
    RENAMED_ENTITY_TYPES.put("EmailAddress", "tt:email_address");
    RENAMED_ENTITY_TYPES.put("Picture", "tt:picture");
    RENAMED_ENTITY_TYPES.put("URL", "tt:url");
    RENAMED_ENTITY_TYPES.put("PhoneNumber", "tt:phone_number");
  }

  static boolean typeOk(String have, String want, Value value) {
    if (have.equals(want))
      return true;

    if (have.equals("Enum") && want.startsWith("Enum("))
      return true;

    // a mistake in the naming that is in too many places
    // to fix now
    if (have.equals("Bool") && want.equals("Boolean"))
      return true;

    // String is acceptable for entity types
    if (have.equals("String") && isEntity(want))
      return true;
    
    // Renamed entity types
    if (RENAMED_ENTITY_TYPES.containsKey(have) && want.equals("Entity(" + RENAMED_ENTITY_TYPES.get(have) + ")"))
      return true;

    if (have.equals("Measure") && want.startsWith("Measure(") && value instanceof NumberValue)
      return unitOk(((NumberValue) value).unit, want.substring("Measure(".length(), want.length() - 1));

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
