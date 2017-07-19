package edu.stanford.nlp.sempre.corenlp;

import java.util.Arrays;
import java.util.Collections;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.sempre.corenlp.QuotedStringAnnotator.QuoteAnnotation;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.CoreMap;

/**
 * Transfer quote annotation to NamedEntityTag and NormalizedNamedEntityTag
 * 
 * This runs after the ner annotator, which in turns runs after spellcheck,
 * which in turn run after quote2
 * 
 * @author gcampagn
 *
 */
public class QuotedStringEntityAnnotator implements Annotator {

  @Override
  public void annotate(Annotation annotation) {
    for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
      CoreMap quote = token.get(QuoteAnnotation.class);
      if (quote == null)
        continue;

      token.set(CoreAnnotations.NamedEntityTagAnnotation.class, "QUOTED_STRING");
      token.set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class,
          quote.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class));
    }
  }

  @Override
  public Set<Class<? extends CoreAnnotation>> requirementsSatisfied() {
    return Collections.emptySet();
  }

  @Override
  public Set<Class<? extends CoreAnnotation>> requires() {
    // TODO Auto-generated method stub
    return Collections.unmodifiableSet(new ArraySet<>(Arrays.asList(
        CoreAnnotations.TextAnnotation.class,
        CoreAnnotations.TokensAnnotation.class,
        CoreAnnotations.PositionAnnotation.class,
        CoreAnnotations.QuotationsAnnotation.class)));
  }

}
