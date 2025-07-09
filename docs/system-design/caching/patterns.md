# Cache Patterns & Strategies 🚀

Advanced caching patterns for building high-performance distributed systems. This guide covers various caching strategies, implementation patterns, and optimization techniques.

## 🎯 Core Caching Patterns

### 1. **Cache-Aside (Lazy Loading)**

**Pattern**: Application manages cache directly

**Flow**:
```text
Read Flow:
1. Check cache for data
2. If cache hit: return data
3. If cache miss: fetch from database
4. Store in cache
5. Return data

Write Flow:
1. Write to database
2. Invalidate cache entry
3. Next read will populate cache
```

**Implementation**:
```python
def get_user(user_id):
    # Check cache first
    user = cache.get(f"user:{user_id}")
    if user:
        return user
    
    # Cache miss - fetch from database
    user = database.get_user(user_id)
    if user:
        cache.set(f"user:{user_id}", user, ttl=3600)
    
    return user

def update_user(user_id, user_data):
    # Update database
    database.update_user(user_id, user_data)
    
    # Invalidate cache
    cache.delete(f"user:{user_id}")
```

**Benefits**:
- **Simple to implement**
- **Fault tolerant**: Cache failures don't affect reads
- **Selective caching**: Only cache what's requested
- **Consistent**: Cache misses always get fresh data

**Drawbacks**:
- **Cache miss penalty**: First request always slow
- **Stale data risk**: Between invalidation and repopulation
- **Cache stampede**: Multiple requests for same expired data

**Use Cases**:
- **User profiles**: Frequently accessed but slowly changing
- **Product catalogs**: Read-heavy with occasional updates
- **Configuration data**: Rarely changing system settings

### 2. **Write-Through Cache**

**Pattern**: Write to cache and database simultaneously

**Flow**:
```text
Write Flow:
1. Write to cache
2. Write to database
3. Return success only if both succeed

Read Flow:
1. Read from cache
2. Cache always has data (if it exists)
```

**Implementation**:
```python
def save_user(user_id, user_data):
    # Write to cache first
    cache.set(f"user:{user_id}", user_data, ttl=3600)
    
    # Then write to database
    database.save_user(user_id, user_data)
    
    return user_data

def get_user(user_id):
    # Cache should always have data
    return cache.get(f"user:{user_id}")
```

**Benefits**:
- **Data consistency**: Cache and database always in sync
- **Fast reads**: Cache always populated
- **Reliability**: Database has all data even if cache fails

**Drawbacks**:
- **Write latency**: Every write hits both cache and database
- **Unused data**: Might cache data that's never read
- **Complexity**: Need to handle partial failures

**Use Cases**:
- **Critical data**: Financial transactions, user authentication
- **Frequently accessed**: Data that's written and read often
- **Consistent reads**: When stale data is unacceptable

### 3. **Write-Behind (Write-Back) Cache**

**Pattern**: Write to cache immediately, database asynchronously

**Flow**:
```text
Write Flow:
1. Write to cache
2. Mark as dirty
3. Return success immediately
4. Asynchronously write to database

Read Flow:
1. Read from cache
2. Cache has latest data
```

**Implementation**:
```python
class WriteBehindCache:
    def __init__(self):
        self.cache = {}
        self.dirty_keys = set()
        self.flush_queue = Queue()
        self.start_background_writer()
    
    def save_user(self, user_id, user_data):
        key = f"user:{user_id}"
        
        # Write to cache immediately
        self.cache[key] = user_data
        self.dirty_keys.add(key)
        
        # Queue for background write
        self.flush_queue.put((key, user_data))
        
        return user_data
    
    def background_writer(self):
        while True:
            key, data = self.flush_queue.get()
            try:
                database.save_user(key, data)
                self.dirty_keys.remove(key)
            except Exception as e:
                # Handle write failure
                self.handle_write_error(key, data, e)
```

**Benefits**:
- **Fast writes**: No database latency for writes
- **High throughput**: Can handle burst writes
- **Batch optimization**: Can batch database writes

**Drawbacks**:
- **Data loss risk**: Cache failure loses unwritten data
- **Complexity**: Need background processes
- **Inconsistency**: Database temporarily behind cache

**Use Cases**:
- **High write volume**: Logging, metrics, analytics
- **Acceptable data loss**: Non-critical data
- **Batch processing**: Data that can be written in batches

### 4. **Refresh-Ahead Cache**

**Pattern**: Proactively refresh cache before expiration

**Flow**:
```text
Read Flow:
1. Check cache for data
2. If data exists and TTL > threshold: return data
3. If data exists but TTL < threshold: 
   - Return current data
   - Trigger background refresh
4. If cache miss: fetch and cache data

Background Refresh:
1. Fetch fresh data from database
2. Update cache with new data
3. Reset TTL
```

**Implementation**:
```python
import time
from threading import Thread

class RefreshAheadCache:
    def __init__(self, refresh_threshold=0.8):
        self.cache = {}
        self.refresh_threshold = refresh_threshold
    
    def get_user(self, user_id):
        key = f"user:{user_id}"
        cached_item = self.cache.get(key)
        
        if cached_item:
            data, timestamp, ttl = cached_item
            age = time.time() - timestamp
            
            # Check if refresh needed
            if age > (ttl * self.refresh_threshold):
                # Trigger background refresh
                Thread(target=self.refresh_data, args=(key,)).start()
            
            return data
        
        # Cache miss - fetch immediately
        return self.fetch_and_cache(key)
    
    def refresh_data(self, key):
        try:
            fresh_data = database.get_user(key)
            self.cache[key] = (fresh_data, time.time(), 3600)
        except Exception as e:
            # Handle refresh failure
            pass
```

**Benefits**:
- **Consistent performance**: No cache miss penalty
- **Fresh data**: Data stays relatively fresh
- **High availability**: Cache always populated

**Drawbacks**:
- **Resource usage**: Background refresh consumes resources
- **Complexity**: Need background job management
- **Stale data**: Still possible during refresh

**Use Cases**:
- **Performance critical**: Applications needing consistent low latency
- **Predictable access**: Data with known access patterns
- **Expensive operations**: Costly database queries or API calls

## 🏗️ Advanced Caching Strategies

### 1. **Cache Hierarchies**

**Multi-Level Caching**:
```text
L1 Cache: Application Memory (fastest, smallest)
L2 Cache: Shared Memory (Redis/Memcached)
L3 Cache: Distributed Cache (Hazelcast)
L4 Cache: Database Buffer Pool
```

**Implementation**:
```python
class HierarchicalCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory
        self.l2_cache = redis.Redis()  # Redis
        self.l3_cache = database  # Database
    
    def get(self, key):
        # Check L1 cache
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2 cache
        data = self.l2_cache.get(key)
        if data:
            self.l1_cache[key] = data  # Promote to L1
            return data
        
        # Check L3 (database)
        data = self.l3_cache.get(key)
        if data:
            self.l2_cache.set(key, data)  # Store in L2
            self.l1_cache[key] = data     # Store in L1
            return data
        
        return None
```

### 2. **Cache Partitioning**

**Horizontal Partitioning**:
```text
Partition Strategies:
- Hash-based: hash(key) % num_partitions
- Range-based: Key ranges to different partitions
- Geographic: Partition by user location
- Functional: Partition by data type
```

**Implementation**:
```python
class PartitionedCache:
    def __init__(self, num_partitions=4):
        self.partitions = [redis.Redis(host=f"cache-{i}") 
                          for i in range(num_partitions)]
        self.num_partitions = num_partitions
    
    def get_partition(self, key):
        return hash(key) % self.num_partitions
    
    def get(self, key):
        partition = self.get_partition(key)
        return self.partitions[partition].get(key)
    
    def set(self, key, value, ttl=3600):
        partition = self.get_partition(key)
        return self.partitions[partition].set(key, value, ex=ttl)
```

### 3. **Cache Warming**

**Pre-populate cache with likely-to-be-requested data**:

**Strategies**:
- **Batch warming**: Load popular data on startup
- **Predictive warming**: Use analytics to predict needed data
- **User-based warming**: Warm cache when user logs in
- **Time-based warming**: Warm cache before peak hours

**Implementation**:
```python
class CacheWarmer:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def warm_popular_users(self):
        # Get most accessed users from analytics
        popular_users = self.database.get_popular_users(limit=1000)
        
        for user_id in popular_users:
            user_data = self.database.get_user(user_id)
            self.cache.set(f"user:{user_id}", user_data, ttl=3600)
    
    def warm_user_data(self, user_id):
        # Warm cache for specific user
        user_data = self.database.get_user(user_id)
        user_posts = self.database.get_user_posts(user_id)
        user_friends = self.database.get_user_friends(user_id)
        
        self.cache.set(f"user:{user_id}", user_data, ttl=3600)
        self.cache.set(f"posts:{user_id}", user_posts, ttl=1800)
        self.cache.set(f"friends:{user_id}", user_friends, ttl=7200)
```

## 🔄 Cache Invalidation Strategies

### 1. **Time-Based Invalidation (TTL)**

**Simple expiration after fixed time**:

```python
# Simple TTL
cache.set("user:123", user_data, ttl=3600)  # Expire in 1 hour

# Sliding window TTL
def get_with_sliding_ttl(key, ttl=3600):
    data = cache.get(key)
    if data:
        # Reset TTL on access
        cache.expire(key, ttl)
    return data
```

**TTL Selection Guidelines**:
- **Frequently changing data**: Short TTL (5-15 minutes)
- **Moderately changing data**: Medium TTL (1-6 hours)
- **Rarely changing data**: Long TTL (24+ hours)
- **Static data**: Very long TTL (weeks/months)

### 2. **Event-Based Invalidation**

**Invalidate cache based on data changes**:

```python
class EventBasedCache:
    def __init__(self):
        self.cache = redis.Redis()
        self.event_bus = EventBus()
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        self.event_bus.subscribe("user_updated", self.invalidate_user_cache)
        self.event_bus.subscribe("post_created", self.invalidate_user_posts)
    
    def invalidate_user_cache(self, event):
        user_id = event.data['user_id']
        self.cache.delete(f"user:{user_id}")
        self.cache.delete(f"posts:{user_id}")
        self.cache.delete(f"friends:{user_id}")
    
    def update_user(self, user_id, user_data):
        # Update database
        database.update_user(user_id, user_data)
        
        # Publish event
        self.event_bus.publish("user_updated", {"user_id": user_id})
```

### 3. **Tag-Based Invalidation**

**Group related cache entries for bulk invalidation**:

```python
class TaggedCache:
    def __init__(self):
        self.cache = redis.Redis()
        self.tags = {}  # tag -> set of keys
    
    def set_with_tags(self, key, value, tags, ttl=3600):
        # Store data
        self.cache.set(key, value, ex=ttl)
        
        # Associate with tags
        for tag in tags:
            if tag not in self.tags:
                self.tags[tag] = set()
            self.tags[tag].add(key)
    
    def invalidate_by_tag(self, tag):
        if tag in self.tags:
            keys_to_delete = self.tags[tag]
            
            # Delete cache entries
            if keys_to_delete:
                self.cache.delete(*keys_to_delete)
            
            # Clear tag association
            del self.tags[tag]

# Usage
cache = TaggedCache()
cache.set_with_tags("user:123", user_data, ["user", "user:123", "department:eng"])
cache.set_with_tags("user:456", user_data, ["user", "user:456", "department:eng"])

# Invalidate all engineering department users
cache.invalidate_by_tag("department:eng")
```

## 📊 Cache Performance Optimization

### 1. **Cache Sizing**

**Memory Management**:
```python
# LRU Cache with size limit
from functools import lru_cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
```

### 2. **Cache Compression**

**Reduce memory usage with compression**:

```python
import gzip
import pickle

class CompressedCache:
    def __init__(self, cache_backend):
        self.cache = cache_backend
    
    def set(self, key, value, ttl=3600):
        # Serialize and compress
        serialized = pickle.dumps(value)
        compressed = gzip.compress(serialized)
        
        return self.cache.set(key, compressed, ex=ttl)
    
    def get(self, key):
        compressed = self.cache.get(key)
        if compressed:
            # Decompress and deserialize
            serialized = gzip.decompress(compressed)
            return pickle.loads(serialized)
        return None
```

### 3. **Connection Pooling**

**Optimize connections to cache servers**:

```python
from redis.connection import ConnectionPool

class OptimizedCacheClient:
    def __init__(self, host='localhost', port=6379, max_connections=50):
        self.pool = ConnectionPool(
            host=host,
            port=port,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        self.redis = redis.Redis(connection_pool=self.pool)
    
    def get_pool_stats(self):
        return {
            'created_connections': self.pool.created_connections,
            'available_connections': len(self.pool._available_connections),
            'in_use_connections': len(self.pool._in_use_connections)
        }
```

## 🔧 Cache Monitoring and Debugging

### 1. **Cache Metrics**

**Essential metrics to monitor**:
- **Hit ratio**: cache hits / total requests
- **Miss ratio**: cache misses / total requests
- **Eviction rate**: items evicted per second
- **Memory usage**: cache memory consumption
- **Response time**: cache operation latency

```python
class MonitoredCache:
    def __init__(self):
        self.cache = redis.Redis()
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def get(self, key):
        data = self.cache.get(key)
        if data:
            self.metrics['hits'] += 1
        else:
            self.metrics['misses'] += 1
        return data
    
    def get_hit_ratio(self):
        total = self.metrics['hits'] + self.metrics['misses']
        return self.metrics['hits'] / total if total > 0 else 0
```

### 2. **Cache Debugging**

**Tools for troubleshooting cache issues**:

```python
class CacheDebugger:
    def __init__(self, cache):
        self.cache = cache
    
    def analyze_key_distribution(self):
        keys = self.cache.keys()
        patterns = {}
        
        for key in keys:
            pattern = self.extract_pattern(key)
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return patterns
    
    def find_large_keys(self, threshold=1024):
        large_keys = []
        for key in self.cache.keys():
            size = len(self.cache.get(key) or b'')
            if size > threshold:
                large_keys.append((key, size))
        
        return sorted(large_keys, key=lambda x: x[1], reverse=True)
    
    def get_ttl_distribution(self):
        ttls = {}
        for key in self.cache.keys():
            ttl = self.cache.ttl(key)
            ttl_bucket = self.get_ttl_bucket(ttl)
            ttls[ttl_bucket] = ttls.get(ttl_bucket, 0) + 1
        
        return ttls
```

## 🧮 Cache Eviction Algorithms

When caches reach capacity, eviction algorithms determine which data to remove. The choice of algorithm significantly impacts cache performance and hit rates.

### 1. **LRU (Least Recently Used)**

**Strategy**: Evict the least recently accessed item

**Use Cases**:
- **General-purpose caching**: Good default choice
- **Sequential access patterns**: Web page caching
- **Temporal locality**: Recently accessed data likely to be accessed again

**Implementation**:
```python
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
        
        self.access_times[key] = time.time()
    
    def evict(self):
        """Manually evict LRU item"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            return oldest_key
        return None

# Usage example
lru_cache = LRUCache(capacity=3)
lru_cache.put("user:1", {"name": "Alice"})
lru_cache.put("user:2", {"name": "Bob"})
lru_cache.put("user:3", {"name": "Charlie"})

# Access user:1 (makes it most recently used)
user1 = lru_cache.get("user:1")

# Adding new item evicts user:2 (least recently used)
lru_cache.put("user:4", {"name": "David"})
```

**Time Complexity**: O(1) for all operations
**Space Complexity**: O(capacity)

### 2. **LFU (Least Frequently Used)**

**Strategy**: Evict the least frequently accessed item

**Use Cases**:
- **Popularity-based caching**: Video streaming, content delivery
- **Long-running systems**: Where frequency patterns emerge
- **Hot data identification**: Keeping most popular items

**Implementation**:
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.frequencies = defaultdict(int)
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        # Update frequency
        self._update_frequency(key)
        return self.cache[key]
    
    def put(self, key, value):
        if self.capacity <= 0:
            return
        
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self._update_frequency(key)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                self._evict_lfu()
            
            self.cache[key] = value
            self.frequencies[key] = 1
            self.freq_to_keys[1][key] = True
            self.min_freq = 1
    
    def _update_frequency(self, key):
        freq = self.frequencies[key]
        
        # Remove from current frequency group
        del self.freq_to_keys[freq][key]
        
        # Update frequency
        self.frequencies[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = True
        
        # Update min_freq if necessary
        if freq == self.min_freq and not self.freq_to_keys[freq]:
            self.min_freq += 1
    
    def _evict_lfu(self):
        # Get least frequently used key
        lfu_key = next(iter(self.freq_to_keys[self.min_freq]))
        
        # Remove from all data structures
        del self.freq_to_keys[self.min_freq][lfu_key]
        del self.frequencies[lfu_key]
        del self.cache[lfu_key]
    
    def get_frequency_stats(self):
        """Get frequency distribution for monitoring"""
        stats = {}
        for freq, keys in self.freq_to_keys.items():
            stats[freq] = len(keys)
        return stats

# Usage example
lfu_cache = LFUCache(capacity=3)
lfu_cache.put("video:1", {"title": "Popular Video"})
lfu_cache.put("video:2", {"title": "Trending Video"})
lfu_cache.put("video:3", {"title": "New Video"})

# Access video:1 multiple times (increases frequency)
for _ in range(5):
    lfu_cache.get("video:1")

# Access video:2 fewer times
for _ in range(2):
    lfu_cache.get("video:2")

# Adding new item evicts video:3 (least frequently used)
lfu_cache.put("video:4", {"title": "Another Video"})
```

**Time Complexity**: O(1) for all operations
**Space Complexity**: O(capacity)

### 3. **FIFO (First In, First Out)**

**Strategy**: Evict the oldest item regardless of access pattern

**Use Cases**:
- **Simple caching**: When implementation simplicity matters
- **Uniform access patterns**: All items equally likely to be accessed
- **Memory-constrained environments**: Minimal overhead

**Implementation**:
```python
from collections import deque

class FIFOCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.insertion_order = deque()
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing key (don't change order)
            self.cache[key] = value
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Remove oldest item
                oldest_key = self.insertion_order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.insertion_order.append(key)
    
    def evict(self):
        """Manually evict FIFO item"""
        if self.insertion_order:
            oldest_key = self.insertion_order.popleft()
            del self.cache[oldest_key]
            return oldest_key
        return None

# Usage example
fifo_cache = FIFOCache(capacity=3)
fifo_cache.put("session:1", {"user_id": 1})
fifo_cache.put("session:2", {"user_id": 2})
fifo_cache.put("session:3", {"user_id": 3})

# Adding new item evicts session:1 (oldest)
fifo_cache.put("session:4", {"user_id": 4})
```

**Time Complexity**: O(1) for all operations
**Space Complexity**: O(capacity)

### 4. **Random Replacement**

**Strategy**: Evict a randomly selected item

**Use Cases**:
- **Uniform access patterns**: When no clear access pattern exists
- **Low-overhead systems**: Minimal computational cost
- **Testing baselines**: Comparison with other algorithms

**Implementation**:
```python
import random

class RandomCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                # Remove random item
                random_key = random.choice(list(self.cache.keys()))
                del self.cache[random_key]
            
            self.cache[key] = value
    
    def evict(self):
        """Manually evict random item"""
        if self.cache:
            random_key = random.choice(list(self.cache.keys()))
            del self.cache[random_key]
            return random_key
        return None
```

### 5. **2Q (Two Queue)**

**Strategy**: Maintains two queues to handle different access patterns

**Use Cases**:
- **Mixed access patterns**: Both sequential and random access
- **Database buffer pools**: Balancing scan resistance with locality
- **File system caches**: Handling different I/O patterns

**Implementation**:
```python
from collections import deque

class TwoQCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.main_cache = {}
        self.fifo_queue = deque()  # First access queue
        self.lru_queue = deque()   # Frequently accessed queue
        
        # Split capacity between queues
        self.fifo_size = capacity // 4
        self.lru_size = capacity - self.fifo_size
    
    def get(self, key):
        if key in self.main_cache:
            if key in self.lru_queue:
                # Move to end of LRU queue
                self.lru_queue.remove(key)
                self.lru_queue.append(key)
            elif key in self.fifo_queue:
                # Promote to LRU queue
                self.fifo_queue.remove(key)
                self._add_to_lru(key)
            
            return self.main_cache[key]
        return None
    
    def put(self, key, value):
        if key in self.main_cache:
            self.main_cache[key] = value
            # Update position in appropriate queue
            if key in self.lru_queue:
                self.lru_queue.remove(key)
                self.lru_queue.append(key)
        else:
            self.main_cache[key] = value
            self._add_to_fifo(key)
    
    def _add_to_fifo(self, key):
        if len(self.fifo_queue) >= self.fifo_size:
            # Remove from FIFO
            old_key = self.fifo_queue.popleft()
            del self.main_cache[old_key]
        
        self.fifo_queue.append(key)
    
    def _add_to_lru(self, key):
        if len(self.lru_queue) >= self.lru_size:
            # Remove from LRU
            old_key = self.lru_queue.popleft()
            del self.main_cache[old_key]
        
        self.lru_queue.append(key)
```

### 6. **Adaptive Replacement Cache (ARC)**

**Strategy**: Dynamically balances between LRU and LFU based on workload

**Use Cases**:
- **Unknown access patterns**: Adapts to changing workloads
- **Database systems**: PostgreSQL, ZFS use variants
- **High-performance systems**: Optimal for mixed workloads

**Implementation**:
```python
class ARCCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.p = 0  # Adaptation parameter
        
        # Four lists as per ARC algorithm
        self.t1 = {}  # Recent cache entries
        self.t2 = {}  # Frequent cache entries
        self.b1 = {}  # Ghost entries for T1
        self.b2 = {}  # Ghost entries for T2
        
        # Maintain order
        self.t1_order = deque()
        self.t2_order = deque()
        self.b1_order = deque()
        self.b2_order = deque()
    
    def get(self, key):
        if key in self.t1:
            # Move to T2 (frequent)
            value = self.t1[key]
            del self.t1[key]
            self.t1_order.remove(key)
            self.t2[key] = value
            self.t2_order.append(key)
            return value
        
        elif key in self.t2:
            # Move to end of T2
            self.t2_order.remove(key)
            self.t2_order.append(key)
            return self.t2[key]
        
        return None
    
    def put(self, key, value):
        if key in self.t1 or key in self.t2:
            # Update existing
            if key in self.t1:
                self.t1[key] = value
            else:
                self.t2[key] = value
        else:
            # New entry
            if key in self.b1:
                # Increase preference for T1
                self.p = min(self.p + max(len(self.b2) // len(self.b1), 1), self.capacity)
                self._replace(key)
                
                # Remove from B1 and add to T2
                del self.b1[key]
                self.b1_order.remove(key)
                self.t2[key] = value
                self.t2_order.append(key)
            
            elif key in self.b2:
                # Increase preference for T2
                self.p = max(self.p - max(len(self.b1) // len(self.b2), 1), 0)
                self._replace(key)
                
                # Remove from B2 and add to T2
                del self.b2[key]
                self.b2_order.remove(key)
                self.t2[key] = value
                self.t2_order.append(key)
            
            else:
                # Completely new
                if len(self.t1) + len(self.t2) >= self.capacity:
                    self._replace(key)
                
                # Add to T1
                self.t1[key] = value
                self.t1_order.append(key)
    
    def _replace(self, key):
        # ARC replacement logic
        if len(self.t1) >= max(1, self.p):
            # Remove from T1
            old_key = self.t1_order.popleft()
            del self.t1[old_key]
            
            # Add to B1
            self.b1[old_key] = True
            self.b1_order.append(old_key)
        else:
            # Remove from T2
            old_key = self.t2_order.popleft()
            del self.t2[old_key]
            
            # Add to B2
            self.b2[old_key] = True
            self.b2_order.append(old_key)
```

### 🎯 Algorithm Selection Guide

| Algorithm | Best For | Strengths | Weaknesses |
|-----------|----------|-----------|------------|
| **LRU** | General purpose, temporal locality | Simple, good hit rates | Poor for scan patterns |
| **LFU** | Popularity-based, long-running systems | Excellent for hot data | Slow adaptation to changes |
| **FIFO** | Simple systems, uniform access | Minimal overhead | No access pattern awareness |
| **Random** | Unknown patterns, baseline | Ultra-low overhead | Unpredictable performance |
| **2Q** | Mixed access patterns | Scan resistant | More complex implementation |
| **ARC** | Unknown/changing patterns | Adaptive, optimal for mixed workloads | Complex, patent concerns |

### 🔧 Implementation Best Practices

1. **Choose Based on Access Pattern**:
   - **Temporal locality**: Use LRU
   - **Popularity-based**: Use LFU
   - **Sequential scans**: Use 2Q or ARC
   - **Unknown patterns**: Use ARC

2. **Monitor and Adapt**:
   - Track hit rates for different algorithms
   - Use A/B testing to validate choices
   - Consider hybrid approaches

3. **Performance Considerations**:
   - All operations should be O(1) or O(log n)
   - Memory overhead should be minimal
   - Thread safety for concurrent access

---

## 💡 Best Practices

### 1. **Cache Design Principles**

**Design Guidelines**:
- **Fail gracefully**: Cache failures shouldn't break the application
- **Consistent naming**: Use clear, consistent key naming conventions
- **Appropriate TTLs**: Set TTLs based on data volatility
- **Monitor performance**: Track hit ratios and response times
- **Plan for scale**: Design for horizontal scaling

### 2. **Common Anti-Patterns to Avoid**

**What NOT to do**:
- **Caching everything**: Only cache frequently accessed data
- **Ignoring cache warming**: Cold caches hurt performance
- **No invalidation strategy**: Stale data causes issues
- **Oversized cache entries**: Large entries waste memory
- **No monitoring**: You can't optimize what you don't measure

### 3. **Security Considerations**

**Cache Security**:
- **Sensitive data**: Don't cache passwords, tokens, or PII
- **Access control**: Secure cache infrastructure
- **Data encryption**: Encrypt sensitive cached data
- **Network security**: Use secure connections to cache servers

## 🎓 Summary

Effective caching requires:

- **Right pattern**: Choose appropriate caching strategy
- **Proper invalidation**: Keep data fresh and consistent
- **Performance monitoring**: Track and optimize cache performance
- **Scalability planning**: Design for growth
- **Security awareness**: Protect sensitive data

Remember: **Caching is not just about speed—it's about building resilient, scalable systems that provide consistent user experiences.**

---

*"The fastest code is code that doesn't run. The second fastest is cached code."*
