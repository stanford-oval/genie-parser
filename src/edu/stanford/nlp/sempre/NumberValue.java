package edu.stanford.nlp.sempre;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import fig.basic.Fmt;
import fig.basic.LispTree;
import fig.basic.LogInfo;

/**
 * Represents a numerical value (optionally comes with a unit).
 * In the future, might want to split this into an Integer version?
 *
 * @author Percy Liang
 */
public class NumberValue extends Value {
  public static final String unitless = "fb:en.unitless";
  public static final String yearUnit = "fb:en.year";

  public final double value;
  public final String unit;  // What measurement (e.g., "fb:en.meter" or unitless)

  public static final Pattern PATTERN = Pattern.compile("(P|PT)([0-9\\.]+)([mMSDHYW])");

  public static NumberValue parseNumber(String value) {
    if (value.startsWith(">=") || value.startsWith("<="))
      value = value.substring(2);
    else if (value.startsWith(">") || value.startsWith("<") || value.startsWith("~"))
      value = value.substring(1);
    return new NumberValue(Double.parseDouble(value));
  }

  public static NumberValue parseDurationValue(String durationStr) {
    if(!PATTERN.matcher(durationStr).matches())
      return null;

    Matcher m = PATTERN.matcher(durationStr);
    if(m.find()) {
      boolean dailyValue = false;
      if(m.group(1).equals("PT"))
        dailyValue = true;

      String unitStr = m.group(3);
      String unit;
      if(unitStr.equals("S"))
        unit = "s";
      else if (unitStr.equals("m"))
        unit = "min";
      else if(unitStr.equals("M"))
        unit = dailyValue ? "min" : "month";
      else if(unitStr.equals("H"))
        unit = "h";
      else if(unitStr.equals("D"))
        unit = "day";
      else if(unitStr.equals("W"))
        unit = "week";
      else if(unitStr.equals("Y"))
        unit = "year";
      else {
        LogInfo.warnings("Got unknown unit %s", unitStr);
        return null;
      }

      try {
        return new NumberValue(Double.parseDouble(m.group(2)), unit);
      } catch(NumberFormatException e) {
        LogInfo.warnings("Cannot parse %s as a number", m.group(1));
        return null;
      }
    } else {
      LogInfo.warning("Cannot parse duration string");
      return null;
    }
  }

  public NumberValue(double value) {
    this(value, unitless);
  }

  public NumberValue(double value, String unit) {
    this.value = value;
    this.unit = unit;
  }

  public NumberValue(LispTree tree) {
    this.value = Double.parseDouble(tree.child(1).value);
    this.unit = 2 < tree.children.size() ? tree.child(2).value : unitless;
  }

  @Override
  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild("number");
    tree.addChild(Fmt.D(value));
    if (!unit.equals(unitless))
      tree.addChild(unit);
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
    json.put("value", value);
    if(!unit.equals(unitless))
      json.put("unit", unit);
    return json;
  }

  @Override public int hashCode() { return Double.valueOf(value).hashCode(); }
  @Override public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    NumberValue that = (NumberValue) o;
    if (this.value != that.value) return false;  // Warning: doing exact equality checking
    if (!this.unit.equals(that.unit)) return false;
    return true;
  }
}
