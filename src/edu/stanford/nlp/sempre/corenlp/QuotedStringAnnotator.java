package edu.stanford.nlp.sempre.corenlp;

import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.QuoteAnnotator;
import edu.stanford.nlp.util.CoreMap;

public class QuotedStringAnnotator extends QuoteAnnotator {
  public static class QuoteAnnotation implements CoreAnnotation<CoreMap> {
    @Override
    public Class<CoreMap> getType() {
      return CoreMap.class;
    }
  }

  public QuotedStringAnnotator(Properties props) {
    super(props, false);
  }

  @Override
  public void annotate(Annotation annotation) {
    super.annotate(annotation);

    String text = annotation.get(CoreAnnotations.TextAnnotation.class);
    List<CoreMap> quotations = annotation.get(CoreAnnotations.QuotationsAnnotation.class);

    for (CoreMap quote : quotations) {
      for (CoreLabel quoteToken : quote.get(CoreAnnotations.TokensAnnotation.class)) {
        quoteToken.set(QuoteAnnotation.class, quote);
      }
      
      int begin = quote.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
      int end = quote.get(CoreAnnotations.CharacterOffsetEndAnnotation.class);

      String quoteString = text.substring(begin, end);
      if (quoteString.startsWith("``") || quoteString.startsWith("''"))
        quoteString = text.substring(begin + 2, end - 2);
      else
        quoteString = text.substring(begin + 1, end - 1);
      quote.set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, quoteString);
    }
  }

}
