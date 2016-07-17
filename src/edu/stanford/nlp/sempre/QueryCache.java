package edu.stanford.nlp.sempre;

import java.lang.ref.SoftReference;
import java.util.*;

import edu.stanford.nlp.sempre.Derivation.Cacheability;

public class QueryCache {
	public static final long LEXICON_CACHE_AGE = 1000 * 3600 * 1; // cache lexicon lookups for 1 hour

	private static class QueryCacheItem {
		private final String query;
		private List<Derivation> derivations;
		private long expires;

		public QueryCacheItem(String query, List<Derivation> derivations) {
			this.query = query;
			this.derivations = Collections.unmodifiableList(derivations);

			// be conservative in cacheability
			Cacheability cache = Cacheability.CACHEABLE;
			for (Derivation d : derivations)
				cache = cache.meet(d.cache);
			switch (cache) {
			case CACHEABLE:
				this.expires = Long.MAX_VALUE;
				break;
			case LEXICON_DEPENDENT:
				this.expires = System.currentTimeMillis() + LEXICON_CACHE_AGE;
				break;
			default:
				this.expires = 0;
			}
		}
	}

	private static class ListBucket extends LinkedList<SoftReference<QueryCacheItem>> {
		private static final long serialVersionUID = 1L;
	};

	private final int maxBucketSize;
	private final ListBucket[] buckets;

	public QueryCache(int nBuckets) {
		maxBucketSize = 3;
		buckets = new ListBucket[nBuckets];
		for (int i = 0; i < nBuckets; i++)
			buckets[i] = new ListBucket();
	}

	private ListBucket getBucket(String query) {
		int bucket = query.hashCode() % buckets.length;
		if (bucket < 0)
			bucket += buckets.length;
		return buckets[bucket];
	}

	public List<Derivation> hit(String query) {
		ListBucket bucket = getBucket(query);
		long now = System.currentTimeMillis();

		synchronized (bucket) {
			Iterator<SoftReference<QueryCacheItem>> it = bucket.iterator();
			while (it.hasNext()) {
				QueryCacheItem item = it.next().get();
				if (item == null || item.expires <= now) {
					it.remove();
					continue;
				}
				if (query.equals(item.query))
					return item.derivations;
			}
		}

		return null;
	}

	public void store(String query, List<Derivation> derivations) {
		QueryCacheItem item = new QueryCacheItem(query, derivations);
		if (item.expires == 0) // ignore if not cacheable
			return;

		ListBucket bucket = getBucket(query);
		long now = System.currentTimeMillis();
		synchronized (bucket) {
			Iterator<SoftReference<QueryCacheItem>> it = bucket.iterator();
			while (it.hasNext()) {
				QueryCacheItem other = it.next().get();
				if (other == null || other.expires <= now) {
					it.remove();
					continue;
				}
				if (query.equals(other.query)) {
					// some other thread added it
					if (item.expires > other.expires) {
						// possibly freshen it
						other.expires = item.expires;
						other.derivations = item.derivations;
					}
					return;
				}
			}

			if (bucket.size() > maxBucketSize) {
				// too many items in this bucket, policy: purge the one expiring first
				long minExpire = Long.MAX_VALUE;
				SoftReference<QueryCacheItem> toRemove = null;
				for (SoftReference<QueryCacheItem> maybeother : bucket) {
					QueryCacheItem other = maybeother.get();
					if (other == null || other.expires < minExpire) {
						toRemove = maybeother;
						minExpire = other == null ? 0 : minExpire;
					}
				}
				bucket.remove(toRemove);
			}
			bucket.add(new SoftReference(item));
		}
	}
}
