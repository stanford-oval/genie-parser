package edu.stanford.nlp.sempre.thingtalk;

import java.util.*;

import edu.stanford.nlp.sempre.Value;
import edu.stanford.nlp.sempre.Values;
import fig.basic.LispTree;

class ParamValueComparator implements Comparator<ParamValue>, Cloneable {
  @Override
  public int compare(ParamValue o1, ParamValue o2) {
    return o1.name.argname.compareTo(o2.name.argname);
  }
}

class SortedList<E> extends AbstractCollection<E> {
  private final Comparator<E> comp;
  private final List<E> backingStore = new ArrayList<>();

  public SortedList(Comparator<E> comp) {
    this.comp = comp;
  }

  public SortedList(Comparator<E> comp, Collection<? extends E> from) {
    this(comp);
    this.addAll(from);
  }

  @Override
  public Iterator<E> iterator() {
    return backingStore.iterator();
  }

  @Override
  public int size() {
    return backingStore.size();
  }

  @Override
  public boolean addAll(Collection<? extends E> from) {
    boolean changed = backingStore.addAll(from);
    if (changed)
      backingStore.sort(comp);
    return changed;
  }

  @Override
  public boolean add(E el) {
    int i;
    for (i = 0; i < backingStore.size(); i++) {
      if (comp.compare(el, backingStore.get(i)) < 0)
        break;
    }
    backingStore.add(i, el);
    return true;
  }
}

/**
 * Base class for thingtalk entities that take parameters (actions, triggers,
 * queries)
 * 
 * @author Rakesh Ramesh & Giovanni Campagna
 */
public abstract class ParametricValue extends Value implements Cloneable {
  public final ChannelNameValue name;

  public final Collection<ParamValue> params = new SortedList<>(new ParamValueComparator());

  public ParametricValue(LispTree tree) {
    this.name = (ChannelNameValue) Values.fromLispTree(tree.child(1));

    for (int i = 2; i < tree.children.size(); i++) {
      this.params.add(((ParamValue) Values.fromLispTree(tree.child(i))));
    }
  }

  public ParametricValue(ChannelNameValue name, List<ParamValue> params) {
    this.name = name;
    this.params.addAll(params);
  }

  public ParametricValue(ChannelNameValue name) {
    this.name = name;
  }

  protected abstract String getLabel();

  public void add(ParamValue param) {
    assert (params != null) : param;
    params.add(param);
  }

  @Override
  public LispTree toLispTree() {
    LispTree tree = LispTree.proto.newList();
    tree.addChild(getLabel());
    tree.addChild(name.toLispTree());
    for (ParamValue param : this.params)
      tree.addChild(param.toLispTree());
    return tree;
  }

  @Override
  public Map<String, Object> toJson() {
    Map<String, Object> json = new HashMap<>();
    json.put("name", name.toJson());
    List<Object> args = new ArrayList<>();
    json.put("args", args);
    for (ParamValue param : params) {
      args.add(param.toJson());
    }
    return json;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    ParametricValue that = (ParametricValue) o;
    if (!name.equals(that.name) || !params.equals(that.params))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    return name.hashCode() ^ params.hashCode();
  }

  @Override
  public ParametricValue clone() {
    try {
      return (ParametricValue) super.clone();
    } catch (CloneNotSupportedException e) {
      throw new RuntimeException(e);
    }
  }
}
