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

  static boolean typeOk(Type have, Type want, Value value) {
    if (have instanceof Type.Measure && want instanceof Type.Measure && value instanceof NumberValue)
      return unitOk(((NumberValue) value).unit, ((Type.Measure) want).getUnit());
    return want.isAssignable(have);
  }

  static boolean typeOkArray(Type have, Type argtype, Value value) {
    return argtype instanceof Type.Array && typeOk(have, ((Type.Array) argtype).getElementType(), value);
  }
}
