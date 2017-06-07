package edu.stanford.nlp.sempre.thingtalk;

import fig.basic.LogInfo;
import java.util.*;

import com.google.common.base.Joiner;

public class Type {
  public static final Type Any = new Type() {
    @Override
    public String toString() {
      return "Any";
    }
  };
  public static final Type Boolean = new Type() {
    @Override
    public String toString() {
      return "Boolean";
    }
  };
  public static final Type String = new Type() {
    @Override
    public String toString() {
      return "String";
    }
  };
  public static final Type Number = new Type() {
    @Override
    public String toString() {
      return "Number";
    }
  };
  public static final Type Time = new Type() {
    @Override
    public String toString() {
      return "Time";
    }
  };
  public static final Type Date = new Type() {
    @Override
    public String toString() {
      return "Date";
    }
  };
  public static final Type Location = new Type() {
    @Override
    public String toString() {
      return "Location";
    }
  };

  private Type() {
  }

  public static class Entity extends Type {
    private final String entityType;

    public Entity(String entityType) {
      this.entityType = entityType;
    }

    public String getType() {
      return entityType;
    }

    @Override
    public String toString() {
      return "Entity(" + entityType + ")";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((entityType == null) ? 0 : entityType.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      Entity other = (Entity) obj;
      if (entityType == null) {
        if (other.entityType != null)
          return false;
      } else if (!entityType.equals(other.entityType))
        return false;
      return true;
    }
  }

  public static class Measure extends Type {
    private final String unit;

    public Measure(String unit) {
      this.unit = unit;
    }

    public String getUnit() {
      return unit;
    }

    @Override
    public String toString() {
      return "Measure(" + unit + ")";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((unit == null) ? 0 : unit.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      Measure other = (Measure) obj;
      if (unit == null) {
        if (other.unit != null)
          return false;
      } else if (!unit.equals(other.unit))
        return false;
      return true;
    }
  }

  public static class Enum extends Type {
    private final List<String> entries = new ArrayList<>();

    public Enum(String[] entries) {
      this.entries.addAll(Arrays.asList(entries));
    }

    public List<String> getEntries() {
      return Collections.unmodifiableList(entries);
    }

    @Override
    public String toString() {
      return "Enum(" + Joiner.on(',').join(entries) + ")";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((entries == null) ? 0 : entries.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      Enum other = (Enum) obj;
      if (entries == null) {
        if (other.entries != null)
          return false;
      } else if (!entries.equals(other.entries))
        return false;
      return true;
    }
  }

  public static class Array extends Type {
    private final Type elemType;

    public Array(Type elemType) {
      this.elemType = elemType;
    }

    public Type getElementType() {
      return elemType;
    }

    @Override
    public String toString() {
      return "Array(" + elemType + ")";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((elemType == null) ? 0 : elemType.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      Array other = (Array) obj;
      if (elemType == null) {
        if (other.elemType != null)
          return false;
      } else if (!elemType.equals(other.elemType))
        return false;
      return true;
    }
  }

  public static Type fromString(String typeStr) {
    if (typeStr.startsWith("Entity("))
      return new Entity(typeStr.substring("Entity(".length(), typeStr.length() - 1));
    if (typeStr.startsWith("Measure("))
      return new Measure(typeStr.substring("Measure(".length(), typeStr.length() - 1));
    if (typeStr.startsWith("Enum("))
      return new Enum(typeStr.substring("Enum(".length(), typeStr.length() - 1).split(","));
    if (typeStr.startsWith("Array("))
      return new Array(fromString(typeStr.substring("Array(".length(), typeStr.length() - 1)));
    switch (typeStr) {
    case "Any":
      return Any;
    case "Boolean":
    case "Bool":
      return Boolean;
    case "String":
      return String;
    case "Number":
      return Number;
    case "Time":
      return Time;
    case "Date":
      return Date;
    case "Location":
      return Location;

    // compat types
    case "Username":
      return new Entity("tt:username");
    case "Hashtag":
      return new Entity("tt:hashtag");
    case "PhoneNumber":
      return new Entity("tt:phone_number");
    case "EmailAddress":
      return new Entity("tt:email_address");
    case "Picture":
      return new Entity("tt:picture");
    case "Resource":
      return new Entity("tt:rdf_resource");
    case "URL":
      return new Entity("tt:url");
    case "Measure":
      return new Measure(null);
    default:
      throw new IllegalArgumentException("Invalid type " + typeStr);
    }
  }

  private static boolean entitySubType(String assignableTo, String assign) {
    if (assign.equals("tt:contact_name"))
      return assignableTo.equals("tt:phone_number") || assignableTo.equals("tt:email_address")
          || assignableTo.equals("tt:contact");
    if (assign.equals("tt:picture"))
      return assignableTo.equals("tt:url");
    return false;
  }

  /**
   * Check if {@link other} is assignable to this
   * 
   * @param other
   *          the type to assign
   * @return true if other can be assigned to this, false otherwise
   */
  public boolean isAssignable(Type other) {
    if (this.equals(other))
      return true;

    if (this == Any || other == Any)
      return true;
    if (this instanceof Entity && other == String)
      return true;
    //if (this == String && other instanceof Entity)
    //  return true;
    if (this instanceof Entity && other instanceof Entity)
      return entitySubType(((Entity) this).entityType, (((Entity) other).entityType));
    return false;
  }
}
