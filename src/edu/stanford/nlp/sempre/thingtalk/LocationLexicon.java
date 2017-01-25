package edu.stanford.nlp.sempre.thingtalk;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLEncoder;
import java.util.*;
import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.type.TypeReference;

import edu.stanford.nlp.sempre.Json;
import edu.stanford.nlp.sempre.ValueFormula;
import fig.basic.LogInfo;
import fig.basic.Option;

public class LocationLexicon extends AbstractLexicon {
  public static class Options {
    @Option
    public String email = null;
    @Option
    public int verbose = 0;
  }

  public static Options opts = new Options();

  @JsonIgnoreProperties({ "boundingbox", "licence" })
  private static class NominatimEntry {
    @JsonProperty
    public String category;
    @JsonProperty
    public String display_name;
    @JsonProperty
    public String icon;
    @JsonProperty
    public double importance;
    @JsonProperty
    public double lat;
    @JsonProperty
    public double lon;
    @JsonProperty
    public String osm_id;
    @JsonProperty
    public String osm_type;
    @JsonProperty
    public String place_id;
    @JsonProperty
    public String place_rank;
    @JsonProperty
    public String type;
  }

  private static final String URL_TEMPLATE = "http://nominatim.openstreetmap.org/search/?format=jsonv2&accept-language=%s&limit=5&q=%s";
  private static final Map<String, LocationLexicon> instances = new HashMap<>();

  private final String languageTag;

  private LocationLexicon(String languageTag) {
    this.languageTag = languageTag;
  }

  public synchronized static LocationLexicon getForLanguage(String languageTag) {
    LocationLexicon instance = instances.get(languageTag);
    if (instance == null) {
      instance = new LocationLexicon(languageTag);
      instances.put(languageTag, instance);
    }
    return instance;
  }

  private static <E1, E2> Collection<E2> map(Collection<E1> from, Function<E1, E2> f) {
    Collection<E2> to = new ArrayList<>();
    for (E1 e : from)
      to.add(f.apply(e));
    return to;
  }

  @Override
  protected Collection<Entry> doLookup(String rawPhrase) {
    try {
      URL url = new URL(String.format(URL_TEMPLATE, languageTag, URLEncoder.encode(rawPhrase, "utf-8")));
      if (opts.verbose >= 3)
        LogInfo.logs("LocationLexicon HTTP call to %s", url);

      URLConnection connection = url.openConnection();
      connection.setRequestProperty("User-Agent", String.format("SEMPRE/2.1 JavaSE/1.8 (operated by %s)", opts.email));
      connection.setUseCaches(true);

      try (Reader reader = new InputStreamReader(connection.getInputStream())) {
        return map(Json.readValueHard(reader, new TypeReference<List<NominatimEntry>>() {
        }), (NominatimEntry entry) -> {
          String canonical = entry.display_name.toLowerCase().replaceAll("[,\\s+]", " ");
          return new Entry("LOCATION", new ValueFormula<>(
              new LocationValue(entry.lat, entry.lon, entry.display_name)),
              canonical);
        });
      }
    } catch (IOException e) {
      LogInfo.logs("Failed to contact location server: %s", e.getMessage());
      return Collections.emptyList();
    }
  }
}
