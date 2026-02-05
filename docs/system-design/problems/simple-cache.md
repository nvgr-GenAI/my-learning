# Design a Simple Cache

Design a simple in-memory cache system that sits between application servers and a database to reduce database load and improve response times.

**Difficulty:** 🟢 Easy | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 25-35 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100K QPS, 1GB cache size, 80% hit rate target |
| **Key Challenges** | Cache invalidation, eviction policies, write strategies, staleness |
| **Core Concepts** | Cache-aside, write-through, write-back, TTL, cache warming |
| **Companies** | All companies (fundamental concept) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Read** | Fetch data from cache or database | P0 (Must have) |
    | **Write** | Update cache and database | P0 (Must have) |
    | **Eviction** | Remove items when cache is full | P0 (Must have) |
    | **Invalidation** | Remove stale data | P1 (Should have) |
    | **Cache Warming** | Pre-populate frequently accessed data | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Distributed caching (Redis, Memcached)
    - Cache replication
    - Advanced data structures (sorted sets, lists)
    - Pub/Sub messaging

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Hit Rate** | > 80% | Cache effectiveness |
    | **Latency (Cache Hit)** | < 1ms | Fast in-memory access |
    | **Latency (Cache Miss)** | < 50ms | Database query time |
    | **Availability** | 99.9% | High reliability |
    | **Consistency** | Eventual consistency | Balance between performance and freshness |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total QPS: 100,000 requests/second
    - Read: 90,000 QPS (90%)
    - Write: 10,000 QPS (10%)

    With 80% hit rate:
    - Cache hits: 72,000 QPS
    - Cache misses: 18,000 QPS (to database)
    - Database writes: 10,000 QPS

    Peak QPS: 2x average = 200,000 QPS
    ```

    ### Storage Estimates

    ```
    Cache size: 1 GB
    Average object size: 1 KB
    Total cached objects: 1M objects

    Database size: 100 GB (100x larger)
    Total objects in DB: 100M objects
    Working set (hot data): 1% = 1M objects
    ```

    ### Performance Impact

    ```
    Without cache:
    - All reads go to DB: 90,000 QPS
    - DB latency: 50ms average
    - User experience: Slow

    With cache (80% hit rate):
    - Cache hits: 72,000 QPS at 1ms = great UX
    - DB load: 18,000 QPS (80% reduction!)
    - Average read latency: 0.8 × 1ms + 0.2 × 50ms = 10.8ms
    ```

    ---

    ## Key Assumptions

    1. Single application server (not distributed)
    2. In-memory cache storage
    3. PostgreSQL as backing database
    4. Read-heavy workload (90% reads)
    5. LRU eviction policy

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Cache-aside pattern** - Application manages cache explicitly
    2. **Write-through strategy** - Write to cache and DB synchronously
    3. **TTL-based expiration** - Auto-invalidate after time period
    4. **LRU eviction** - Remove least recently used items when full
    5. **Simple and fast** - In-memory hash map for storage

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Application Layer"
            App[Application Server]
        end

        subgraph "Cache Layer"
            Cache[In-Memory Cache<br/>HashMap + LRU]
            Stats[Cache Statistics<br/>Hit Rate, Miss Rate]
        end

        subgraph "Database Layer"
            DB[(PostgreSQL<br/>Persistent Storage)]
        end

        App -->|1. Check cache| Cache
        Cache -->|2. Cache miss| App
        App -->|3. Query| DB
        DB -->|4. Return data| App
        App -->|5. Store in cache| Cache

        Cache -.->|Metrics| Stats

        style Cache fill:#e1f5ff
        style DB fill:#fff4e1
    ```

    ---

    ## Cache Strategies

    ### 1. Cache-Aside (Lazy Loading)

    **How it works:**
    - Application checks cache first
    - On cache miss, query database
    - Store result in cache for future requests

    ```python
    def get_user(user_id: int) -> User:
        # 1. Check cache
        cached_user = cache.get(f"user:{user_id}")
        if cached_user:
            return cached_user

        # 2. Cache miss - query database
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)

        # 3. Store in cache
        cache.set(f"user:{user_id}", user, ttl=3600)

        return user
    ```

    **Pros:**
    - Only cache what's actually requested (efficient)
    - Resilient to cache failures (app still works)

    **Cons:**
    - Cache miss penalty (extra latency)
    - Potential cache stampede

    ---

    ### 2. Write-Through

    **How it works:**
    - Write to cache and database synchronously
    - Cache always consistent with database

    ```python
    def update_user(user_id: int, data: dict):
        # 1. Write to database
        db.execute("UPDATE users SET ... WHERE id = ?", user_id)

        # 2. Update cache
        cache.set(f"user:{user_id}", data, ttl=3600)
    ```

    **Pros:**
    - Cache always fresh
    - No stale data issues

    **Cons:**
    - Write latency (wait for both cache and DB)
    - Writes to cache even if data rarely read

    ---

    ### 3. Write-Back (Write-Behind)

    **How it works:**
    - Write to cache immediately
    - Asynchronously write to database later

    ```python
    def update_user_async(user_id: int, data: dict):
        # 1. Write to cache immediately
        cache.set(f"user:{user_id}", data, ttl=3600)

        # 2. Queue database write
        write_queue.enqueue({
            "action": "update",
            "table": "users",
            "id": user_id,
            "data": data
        })
    ```

    **Pros:**
    - Fast writes
    - Can batch database writes

    **Cons:**
    - Risk of data loss if cache fails
    - More complex implementation

    ---

    ## API Design

    ### Simple Cache Interface

    ```python
    class SimpleCache:
        """Simple in-memory cache with TTL and LRU eviction"""

        def get(self, key: str) -> Any:
            """Get value from cache"""
            pass

        def set(self, key: str, value: Any, ttl: int = None):
            """Set value in cache with optional TTL"""
            pass

        def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass

        def clear(self):
            """Clear all cache entries"""
            pass

        def stats(self) -> dict:
            """Get cache statistics"""
            pass
    ```

    ---

    ## Data Flow

    ### Read Flow (Cache-Aside)

    ```mermaid
    sequenceDiagram
        participant App as Application
        participant Cache as Cache
        participant DB as Database

        App->>Cache: GET user:123

        alt Cache Hit
            Cache-->>App: Return user data
            Note over App: Latency: ~1ms
        else Cache Miss
            Cache-->>App: Key not found
            App->>DB: SELECT * FROM users WHERE id = 123
            DB-->>App: Return user data
            Note over App,DB: Latency: ~50ms
            App->>Cache: SET user:123 (ttl=3600)
            Cache-->>App: OK
        end

        App-->>App: Return user to client
    ```

    ---

    ### Write Flow (Write-Through)

    ```mermaid
    sequenceDiagram
        participant App as Application
        participant Cache as Cache
        participant DB as Database

        App->>DB: UPDATE users SET name = 'John' WHERE id = 123
        DB-->>App: Success

        App->>Cache: SET user:123 (updated data)
        Cache-->>App: OK

        App-->>App: Return success to client
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. Cache Invalidation

    **"There are only two hard things in Computer Science: cache invalidation and naming things."** - Phil Karlton

    **Strategies:**

    **Time-based (TTL):**
    ```python
    # Set with expiration
    cache.set("user:123", user_data, ttl=3600)  # 1 hour
    ```

    **Event-based:**
    ```python
    # On user update
    def update_user(user_id: int, data: dict):
        db.update(user_id, data)
        cache.delete(f"user:{user_id}")  # Invalidate
    ```

    **Version-based:**
    ```python
    # Include version in key
    cache_key = f"user:{user_id}:v{version}"
    ```

    ---

    ### 2. Cache Stampede Problem

    **Problem:**
    - Popular item expires
    - Many requests simultaneously query database
    - Database gets overwhelmed

    **Solution 1: Lock/Mutex**
    ```python
    def get_with_lock(key: str):
        value = cache.get(key)
        if value:
            return value

        # Acquire lock to prevent stampede
        with cache.lock(f"lock:{key}", timeout=10):
            # Double-check after acquiring lock
            value = cache.get(key)
            if value:
                return value

            # Only one request queries database
            value = db.query(key)
            cache.set(key, value, ttl=3600)
            return value
    ```

    **Solution 2: Probabilistic Early Expiration**
    ```python
    import random

    def get_with_early_refresh(key: str, ttl: int):
        value, expiry_time = cache.get_with_expiry(key)
        if not value:
            value = db.query(key)
            cache.set(key, value, ttl=ttl)
            return value

        # Probabilistically refresh before expiration
        time_to_expiry = expiry_time - time.time()
        if time_to_expiry < ttl * 0.1:  # Last 10% of TTL
            if random.random() < 0.1:  # 10% chance to refresh
                value = db.query(key)
                cache.set(key, value, ttl=ttl)

        return value
    ```

    ---

    ### 3. Cache Warming

    **Problem:** Cold cache has 0% hit rate initially

    **Solution: Pre-populate hot data**
    ```python
    def warm_cache():
        """Pre-populate cache with frequently accessed data"""
        # Get most popular users
        popular_users = db.query("""
            SELECT id, data FROM users
            ORDER BY access_count DESC
            LIMIT 10000
        """)

        for user in popular_users:
            cache.set(f"user:{user.id}", user.data, ttl=3600)
    ```

    ---

    ### 4. Monitoring and Metrics

    **Key metrics to track:**

    ```python
    class CacheStats:
        def __init__(self):
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.errors = 0

        def hit_rate(self) -> float:
            total = self.hits + self.misses
            if total == 0:
                return 0.0
            return self.hits / total

        def miss_rate(self) -> float:
            return 1 - self.hit_rate()
    ```

    **Target metrics:**
    - Hit rate: > 80%
    - Average latency: < 10ms
    - Eviction rate: < 5% of requests
    - Error rate: < 0.1%

=== "⚡ Step 4: Scale & Optimize"

    ## Optimization Techniques

    ### 1. Multi-Level Caching

    ```python
    class MultiLevelCache:
        """L1 (in-process) + L2 (Redis) cache"""

        def __init__(self):
            self.l1_cache = LRUCache(capacity=10000)  # Small, fast
            self.l2_cache = RedisCache()  # Large, shared

        def get(self, key: str):
            # Check L1 first (fastest)
            value = self.l1_cache.get(key)
            if value:
                return value

            # Check L2 (slower but shared)
            value = self.l2_cache.get(key)
            if value:
                # Promote to L1
                self.l1_cache.set(key, value)
                return value

            # Cache miss - query database
            value = db.query(key)
            self.l1_cache.set(key, value)
            self.l2_cache.set(key, value)
            return value
    ```

    ---

    ### 2. Cache Partitioning

    **Partition by data type:**
    ```python
    user_cache = SimpleCache(capacity=100000)
    product_cache = SimpleCache(capacity=50000)
    order_cache = SimpleCache(capacity=20000)
    ```

    **Benefits:**
    - Different eviction policies per partition
    - Isolate hot data from cold data
    - Better cache utilization

    ---

    ### 3. Compression

    ```python
    import zlib
    import json

    def set_compressed(key: str, value: dict):
        # Serialize and compress
        json_data = json.dumps(value)
        compressed = zlib.compress(json_data.encode())
        cache.set(key, compressed)

    def get_compressed(key: str) -> dict:
        compressed = cache.get(key)
        if not compressed:
            return None
        # Decompress and deserialize
        json_data = zlib.decompress(compressed).decode()
        return json.loads(json_data)
    ```

    **Benefits:**
    - 5-10x size reduction for text data
    - Cache more items in same memory
    - Trade CPU for memory

    ---

    ### 4. Cache Monitoring Dashboard

    | Metric | Formula | Alert Threshold |
    |--------|---------|-----------------|
    | **Hit Rate** | hits / (hits + misses) | < 70% |
    | **Miss Rate** | misses / (hits + misses) | > 30% |
    | **Eviction Rate** | evictions / total_requests | > 10% |
    | **Memory Usage** | used_memory / max_memory | > 90% |
    | **Average Latency** | total_time / total_requests | > 5ms |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Cache-aside pattern** - Application controls cache explicitly
    2. **Write-through for consistency** - Keep cache and DB in sync
    3. **TTL-based expiration** - Automatic invalidation
    4. **LRU eviction** - Remove least recently used items
    5. **Statistics tracking** - Monitor hit rate and performance

    ## Interview Tips

    ✅ **Start with requirements** - Clarify read/write ratio, consistency needs
    ✅ **Choose cache strategy** - Cache-aside for read-heavy workloads
    ✅ **Discuss eviction** - LRU is simple and effective
    ✅ **Address invalidation** - TTL + event-based invalidation
    ✅ **Consider stampede** - Mention locking or probabilistic refresh

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"What if cache goes down?"** | App should gracefully degrade to database-only mode |
    | **"How to invalidate related data?"** | Tag-based invalidation or namespace patterns |
    | **"Cache-aside vs write-through?"** | Cache-aside for read-heavy, write-through for consistency |
    | **"How to prevent cache stampede?"** | Locking, probabilistic early expiration, or cache warming |
    | **"How to scale beyond single node?"** | Distributed cache (Redis, Memcached) with consistent hashing |

    ## Real-World Examples

    - **Facebook**: Multi-level caching (TAO, Memcached, CDN)
    - **Twitter**: Timeline caching with Redis
    - **Netflix**: Multi-region cache with EVCache
    - **Amazon**: Product catalog caching

---

**Difficulty:** 🟢 Easy | **Interview Time:** 25-35 minutes | **Companies:** All companies

---

*This is a foundational problem that demonstrates caching patterns, performance optimization, and trade-offs between consistency and availability.*
