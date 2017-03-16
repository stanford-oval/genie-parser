package edu.stanford.nlp.sempre.thingtalk;

import fig.basic.LispTree;

public class EntityLexiconFn extends AbstractLexiconFn<TypedStringValue> {
  @Override
  public void init(LispTree tree) {
    super.init(tree);

    String languageTag = tree.child(1).value;
    setLexicon(EntityLexicon.getForLanguage(languageTag));
  }
}