# Caching with Hash Tables

## Introduction to Caching

**Caching** is a technique that stores copies of frequently accessed data in a faster storage system to improve access times in subsequent requests. Hash tables are an ideal data structure for implementing caches due to their O(1) average time complexity for lookups and insertions.

## Why Use Hash Tables for Caching?

Hash tables provide several advantages for caching implementations:

1. **Fast Retrieval**: O(1) average time complexity for lookups
2. **Efficient Storage**: Direct mapping of keys to values
3. **Flexible Key Types**: Support for various types of cache keys
4. **Dynamic Resizing**: Ability to grow or shrink based on cache demand

## Common Caching Architectures

### Local Memory Cache

A simple in-memory cache using hash tables:

```java
public class MemoryCache<K, V> {
    private final HashMap<K, V> cache;
    private final int capacity;
    
    public MemoryCache(int capacity) {
        this.cache = new HashMap<>(capacity);
        this.capacity = capacity;
    }
    
    public V get(K key) {
        return cache.get(key);
    }
    
    public void put(K key, V value) {
        if (cache.size() >= capacity && !cache.containsKey(key)) {
            evictRandom();
        }
        cache.put(key, value);
    }
    
    private void evictRandom() {
        K keyToRemove = cache.keySet().iterator().next();
        cache.remove(keyToRemove);
    }
    
    public boolean contains(K key) {
        return cache.containsKey(key);
    }
    
    public int size() {
        return cache.size();
    }
    
    public void clear() {
        cache.clear();
    }
}
```

### Distributed Cache

Distributed systems often implement caching across multiple nodes:

```python
class DistributedCache:
    def __init__(self, nodes):
        self.nodes = nodes  # List of cache server nodes
        
    def get_node_for_key(self, key):
        # Consistent hashing to determine which node holds the key
        hash_val = hash(key)
        node_index = hash_val % len(self.nodes)
        return self.nodes[node_index]
    
    def get(self, key):
        node = self.get_node_for_key(key)
        return node.get_from_local_cache(key)
    
    def put(self, key, value):
        node = self.get_node_for_key(key)
        node.store_in_local_cache(key, value)
```

## Cache Eviction Policies

When a cache reaches its capacity limit, it must decide which items to evict. Common policies include:

### Least Recently Used (LRU)

Removes the least recently accessed items first. Implemented using a hash table and a doubly linked list:

```java
public class LRUCache<K, V> {
    private class Node {
        K key;
        V value;
        Node prev;
        Node next;
        
        Node(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }
    
    private final int capacity;
    private final HashMap<K, Node> cache;
    private final Node head;  // Most recently used
    private final Node tail;  // Least recently used
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new HashMap<>();
        this.head = new Node(null, null);
        this.tail = new Node(null, null);
        head.next = tail;
        tail.prev = head;
    }
    
    public V get(K key) {
        Node node = cache.get(key);
        if (node == null) {
            return null;
        }
        // Move to front (most recently used)
        moveToHead(node);
        return node.value;
    }
    
    public void put(K key, V value) {
        Node node = cache.get(key);
        
        if (node == null) {
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            addNode(newNode);
            
            if (cache.size() > capacity) {
                // Remove the least recently used item
                Node tail = removeTail();
                cache.remove(tail.key);
            }
        } else {
            // Update value and move to front
            node.value = value;
            moveToHead(node);
        }
    }
    
    private void addNode(Node node) {
        // Add right after head
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
    
    private void removeNode(Node node) {
        // Remove from doubly linked list
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void moveToHead(Node node) {
        removeNode(node);
        addNode(node);
    }
    
    private Node removeTail() {
        Node res = tail.prev;
        removeNode(res);
        return res;
    }
}
```

### Least Frequently Used (LFU)

Removes items that are used least frequently:

```java
public class LFUCache<K, V> {
    private final int capacity;
    private final HashMap<K, V> valueMap;
    private final HashMap<K, Integer> frequencyMap;
    private final HashMap<Integer, LinkedHashSet<K>> frequencyListMap;
    private int minFrequency;
    
    public LFUCache(int capacity) {
        this.capacity = capacity;
        this.valueMap = new HashMap<>();
        this.frequencyMap = new HashMap<>();
        this.frequencyListMap = new HashMap<>();
        this.minFrequency = 0;
    }
    
    public V get(K key) {
        if (!valueMap.containsKey(key)) {
            return null;
        }
        
        // Update frequency
        int frequency = frequencyMap.get(key);
        frequencyMap.put(key, frequency + 1);
        frequencyListMap.get(frequency).remove(key);
        
        // If no keys with the minimum frequency exist, increase min
        if (frequency == minFrequency && frequencyListMap.get(frequency).size() == 0) {
            minFrequency++;
        }
        
        // Add to the higher frequency list
        frequencyListMap.computeIfAbsent(frequency + 1, k -> new LinkedHashSet<>())
                         .add(key);
        
        return valueMap.get(key);
    }
    
    public void put(K key, V value) {
        if (capacity <= 0) {
            return;
        }
        
        // If key exists, update its value and frequency
        if (valueMap.containsKey(key)) {
            valueMap.put(key, value);
            get(key);  // This will update frequency
            return;
        }
        
        // If capacity is reached, remove the least frequent item
        if (valueMap.size() >= capacity) {
            K evict = frequencyListMap.get(minFrequency).iterator().next();
            frequencyListMap.get(minFrequency).remove(evict);
            valueMap.remove(evict);
            frequencyMap.remove(evict);
        }
        
        // Add new item with frequency 1
        valueMap.put(key, value);
        frequencyMap.put(key, 1);
        minFrequency = 1;
        frequencyListMap.computeIfAbsent(1, k -> new LinkedHashSet<>())
                         .add(key);
    }
}
```

### First-In-First-Out (FIFO)

Removes the oldest items first:

```java
public class FIFOCache<K, V> {
    private final LinkedHashMap<K, V> cache;
    private final int capacity;
    
    public FIFOCache(int capacity) {
        // Last parameter false means access order is not considered
        this.cache = new LinkedHashMap<>(capacity, 0.75f, false) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > capacity;
            }
        };
        this.capacity = capacity;
    }
    
    public V get(K key) {
        return cache.get(key);
    }
    
    public void put(K key, V value) {
        cache.put(key, value);
    }
}
```

### Random Replacement (RR)

Removes a random item when the cache is full:

```java
public class RandomCache<K, V> {
    private final HashMap<K, V> cache;
    private final int capacity;
    private final Random random;
    
    public RandomCache(int capacity) {
        this.cache = new HashMap<>(capacity);
        this.capacity = capacity;
        this.random = new Random();
    }
    
    public V get(K key) {
        return cache.get(key);
    }
    
    public void put(K key, V value) {
        if (cache.size() >= capacity && !cache.containsKey(key)) {
            // Get a random key to remove
            K[] keys = (K[]) cache.keySet().toArray();
            K randomKey = keys[random.nextInt(keys.length)];
            cache.remove(randomKey);
        }
        cache.put(key, value);
    }
}
```

## Time-Based Caching

Many caching solutions incorporate time-based expiration:

```java
public class TimedCache<K, V> {
    private class TimedValue {
        V value;
        long expiryTime;
        
        TimedValue(V value, long ttlMillis) {
            this.value = value;
            this.expiryTime = System.currentTimeMillis() + ttlMillis;
        }
        
        boolean isExpired() {
            return System.currentTimeMillis() > expiryTime;
        }
    }
    
    private final ConcurrentHashMap<K, TimedValue> cache = new ConcurrentHashMap<>();
    
    public V get(K key) {
        TimedValue timedValue = cache.get(key);
        if (timedValue == null) {
            return null;
        }
        
        if (timedValue.isExpired()) {
            cache.remove(key);
            return null;
        }
        
        return timedValue.value;
    }
    
    public void put(K key, V value, long ttlMillis) {
        cache.put(key, new TimedValue(value, ttlMillis));
    }
    
    // Periodic cleanup to remove expired entries
    public void cleanupExpiredEntries() {
        for (Map.Entry<K, TimedValue> entry : cache.entrySet()) {
            if (entry.getValue().isExpired()) {
                cache.remove(entry.getKey());
            }
        }
    }
}
```

## Real-World Caching Systems

### Web Browser Cache

Browsers cache web resources using hash tables for fast lookups:

```javascript
class BrowserCache {
    constructor(maxSize) {
        this.cache = new Map();
        this.maxSize = maxSize;
    }
    
    fetchResource(url) {
        if (this.cache.has(url)) {
            const resource = this.cache.get(url);
            if (!this.isExpired(resource)) {
                console.log("Cache hit for " + url);
                return resource.data;
            }
            // Resource expired, remove it
            this.cache.delete(url);
        }
        
        console.log("Cache miss for " + url);
        // Fetch from network and store in cache
        const data = fetchFromNetwork(url);
        this.cache.set(url, {
            data: data,
            timestamp: Date.now(),
            expiresAt: Date.now() + 3600000 // 1 hour
        });
        
        if (this.cache.size > this.maxSize) {
            // Remove oldest entry
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        
        return data;
    }
    
    isExpired(resource) {
        return Date.now() > resource.expiresAt;
    }
    
    clearCache() {
        this.cache.clear();
    }
}
```

### Database Query Cache

Database systems often implement query result caching:

```python
class QueryCache:
    def __init__(self, capacity=100):
        self.cache = {}
        self.capacity = capacity
        self.lru = []  # List to track query access order
    
    def get_result(self, query_hash):
        if query_hash in self.cache:
            # Update LRU order
            self.lru.remove(query_hash)
            self.lru.append(query_hash)
            return self.cache[query_hash]
        return None
    
    def put_result(self, query_hash, result):
        if len(self.cache) >= self.capacity and query_hash not in self.cache:
            # Remove least recently used query
            lru_key = self.lru.pop(0)
            del self.cache[lru_key]
        
        # Add or update query result
        self.cache[query_hash] = result
        if query_hash in self.lru:
            self.lru.remove(query_hash)
        self.lru.append(query_hash)
    
    def invalidate(self, table_name):
        # Invalidate all queries related to the given table
        keys_to_remove = []
        for query_hash in self.cache:
            if self._query_uses_table(query_hash, table_name):
                keys_to_remove.append(query_hash)
        
        for key in keys_to_remove:
            del self.cache[key]
            self.lru.remove(key)
    
    def _query_uses_table(self, query_hash, table_name):
        # Implementation depends on how query information is stored
        # This is a simplification
        return table_name in self.cache[query_hash]['tables_used']
```

## Cache Performance Metrics

When implementing caches with hash tables, several metrics are important:

1. **Hit Rate**: Percentage of requests fulfilled from cache
   ```
   Hit Rate = Cache Hits / Total Requests
   ```

2. **Miss Rate**: Percentage of requests not found in cache
   ```
   Miss Rate = Cache Misses / Total Requests
   ```

3. **Latency**: Time taken to retrieve an item from cache

4. **Throughput**: Number of cache requests handled per unit of time

5. **Eviction Rate**: Frequency at which items are removed from cache

## Cache-Aside vs. Write-Through Patterns

### Cache-Aside Pattern

The application is responsible for reading/writing from both the cache and the primary data store:

```java
public class CacheAsideService<K, V> {
    private final DataStore<K, V> dataStore;
    private final Cache<K, V> cache;
    
    public CacheAsideService(DataStore<K, V> dataStore, Cache<K, V> cache) {
        this.dataStore = dataStore;
        this.cache = cache;
    }
    
    public V get(K key) {
        // Try to get from cache first
        V value = cache.get(key);
        if (value == null) {
            // Cache miss - read from data store
            value = dataStore.get(key);
            if (value != null) {
                // Update cache
                cache.put(key, value);
            }
        }
        return value;
    }
    
    public void put(K key, V value) {
        // Write to data store
        dataStore.put(key, value);
        // Invalidate cache
        cache.remove(key);
    }
}
```

### Write-Through Pattern

Every write goes through the cache before updating the main data store:

```java
public class WriteThroughService<K, V> {
    private final DataStore<K, V> dataStore;
    private final Cache<K, V> cache;
    
    public WriteThroughService(DataStore<K, V> dataStore, Cache<K, V> cache) {
        this.dataStore = dataStore;
        this.cache = cache;
    }
    
    public V get(K key) {
        // Try to get from cache first
        V value = cache.get(key);
        if (value == null) {
            // Cache miss - read from data store
            value = dataStore.get(key);
            if (value != null) {
                // Update cache
                cache.put(key, value);
            }
        }
        return value;
    }
    
    public void put(K key, V value) {
        // Update cache first
        cache.put(key, value);
        // Then update data store
        dataStore.put(key, value);
    }
}
```

## Cache Consistency Challenges

### The Cache Invalidation Problem

One of the hardest problems in computer science:

1. **Stale Data**: Cache contains outdated information
2. **Inconsistent Views**: Different components see different values
3. **Write Conflicts**: Concurrent updates cause data corruption

### Solutions

1. **Time-To-Live (TTL)**: Items automatically expire after a set time
2. **Write-Through**: Update cache and backend simultaneously
3. **Cache Invalidation**: Actively remove or update entries when data changes
4. **Event-Based Systems**: Use message queues to notify cache of changes

## Memory Considerations and Optimizations

### Memory Management

1. **Item Size Limits**: Restrict the maximum size of cached objects
2. **Compression**: Compress large values to reduce memory footprint
3. **Serialization**: Convert objects to byte arrays for storage efficiency

### Performance Optimizations

1. **Prefetching**: Load related items into cache in anticipation of future requests
2. **Warm-up**: Pre-populate cache with common items before peak load
3. **Read-Through**: Automatically load missing items from the backend

## Conclusion

Hash tables provide the foundation for efficient caching systems due to their O(1) average time complexity for key operations. By combining hash tables with appropriate eviction policies and consistency mechanisms, developers can build high-performance caching solutions that significantly improve application responsiveness and reduce load on backend systems.

Whether implementing a simple in-memory cache or a distributed caching system, understanding the principles of hash table-based caching is essential for optimal system design and performance tuning.
