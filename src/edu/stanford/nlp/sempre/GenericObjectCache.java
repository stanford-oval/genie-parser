package edu.stanford.nlp.sempre;

import java.lang.ref.SoftReference;
import java.util.*;

public class GenericObjectCache<K, V> {
	private static class CacheItem<K, V> {
		private final K key;
		private V value;
		private long expires;

		public CacheItem(K key, V value, long expires) {
			this.key = key;
			this.value = value;
			this.expires = expires;
		}
	}

	private static class ListBucket<K, V> extends LinkedList<SoftReference<CacheItem<K, V>>> {
		private static final long serialVersionUID = 1L;
	};

	private final int maxBucketSize;
	private final List<ListBucket<K, V>> buckets;

	public GenericObjectCache(int nBuckets) {
		maxBucketSize = 3;
		buckets = new ArrayList<>();
		for (int i = 0; i < nBuckets; i++)
			buckets.add(new ListBucket<K, V>());
	}

	private ListBucket<K, V> getBucket(K key) {
		int bucket = key.hashCode() % buckets.size();
		if (bucket < 0)
			bucket += buckets.size();
		return buckets.get(bucket);
	}

	public V hit(K key) {
		ListBucket<K, V> bucket = getBucket(key);
		long now = System.currentTimeMillis();

		synchronized (bucket) {
			Iterator<SoftReference<CacheItem<K, V>>> it = bucket.iterator();
			while (it.hasNext()) {
				CacheItem<K, V> item = it.next().get();
				if (item == null || item.expires <= now) {
					it.remove();
					continue;
				}
				if (key.equals(item.key))
					return item.value;
			}
		}

		return null;
	}

  public void clear() {
    for (ListBucket<K, V> bucket : buckets) {
      synchronized (bucket) {
        bucket.clear();
      }
    }
  }

  public void clear(K key) {
    ListBucket<K, V> bucket = getBucket(key);

    synchronized (bucket) {
      Iterator<SoftReference<CacheItem<K, V>>> it = bucket.iterator();
      while (it.hasNext()) {
        CacheItem<K, V> other = it.next().get();
        if (other == null || other.key.equals(key))
          it.remove();
      }
    }
  }

	public void store(K key, V value, long expires) {
		if (expires == 0) // ignore if not cacheable
			return;

		CacheItem<K, V> item = new CacheItem<>(key, value, expires);
		ListBucket<K, V> bucket = getBucket(key);

		long now = System.currentTimeMillis();
		synchronized (bucket) {
			Iterator<SoftReference<CacheItem<K, V>>> it = bucket.iterator();
			while (it.hasNext()) {
				CacheItem<K, V> other = it.next().get();
				if (other == null || other.expires <= now) {
					it.remove();
					continue;
				}
				if (key.equals(other.key)) {
					// some other thread added it
					if (item.expires > other.expires) {
						// possibly freshen it
						other.expires = item.expires;
						other.value = item.value;
					}
					return;
				}
			}

			if (bucket.size() > maxBucketSize) {
				// too many items in this bucket, policy: purge the one expiring first
				long minExpire = Long.MAX_VALUE;
				SoftReference<CacheItem<K, V>> toRemove = null;
				for (SoftReference<CacheItem<K, V>> maybeother : bucket) {
					CacheItem<K, V> other = maybeother.get();
					if (other == null || other.expires < minExpire) {
						toRemove = maybeother;
						minExpire = other == null ? 0 : minExpire;
					}
				}
				bucket.remove(toRemove);
			}
			bucket.add(new SoftReference<>(item));
		}
	}
}
