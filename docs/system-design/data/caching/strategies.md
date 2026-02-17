# Caching Strategies

Every distributed system faces a fundamental trade-off between speed and freshness. A cache sits between your application and its data source, storing recently or frequently accessed data in fast memory so that repeated requests avoid the slower round-trip to a database or remote service. The art of caching lies in choosing the right strategy for populating, updating, and evicting cached data so that the system remains both fast and correct. Get the strategy wrong and you face stale reads, inconsistent state, or catastrophic cache failures that bring down the very database the cache was supposed to protect.

This page covers the four pillars of caching strategy: how data enters the cache (read strategies), how writes propagate (write strategies), how the cache makes room when full (eviction policies), and what goes wrong when caching fails (cache problems).

## Strategy Decision Tree

Before diving into details, use this decision tree to identify which strategy fits your workload.

```text
What does your workload look like?
│
├── Mostly reads, occasional writes
│   └── Cache-Aside (lazy loading)
│       Most common pattern. App checks cache, falls back to DB.
│
├── Reads that must always hit cache (never DB directly)
│   └── Read-Through
│       Cache itself fetches from DB on miss. Simpler app code.
│
├── Writes need strong consistency with cache
│   └── Write-Through
│       Every write goes to cache AND DB synchronously.
│
├── Writes are bursty / high-throughput
│   └── Write-Behind (Write-Back)
│       Write to cache immediately, flush to DB asynchronously.
│
└── Writes are infrequent, reads dominate
    └── Write-Around
        Write to DB only. Cache fills on next read.
```

---

=== "Read Strategies"

    ## Read Strategies

    Read strategies determine how your application populates the cache when a user requests data. The two primary patterns are cache-aside, where the application manages the cache explicitly, and read-through, where the cache itself is responsible for fetching missing data.

    ### Cache-Aside (Lazy Loading)

    Cache-aside is the most widely deployed caching pattern on the internet. The application treats the cache as a simple key-value store and takes full responsibility for the three-step dance: check cache, query database on miss, then populate the cache for next time.

    ```
    ┌─────────┐     1. GET user:123     ┌───────────┐
    │  Client  │ ──────────────────────> │    App     │
    └─────────┘                          └─────┬─────┘
                                               │
                                     2. GET user:123
                                               │
                                               v
                                        ┌─────────────┐
                                        │    Cache     │
                                        └──────┬──────┘
                                               │
                                    3. MISS (not found)
                                               │
                                               v
                                        ┌─────────────┐
                                        │  Database    │
                                        └──────┬──────┘
                                               │
                                    4. Return user data
                                               │
                                               v
                                        ┌─────────────┐
                              5. SET    │    Cache     │
                             user:123   └──────┬──────┘
                                               │
                                    6. Return to client
    ```

    The pattern in pseudo-code is straightforward:

    ```python
    def get_user(user_id):
        user = cache.get(f"user:{user_id}")
        if user:
            return user                        # cache hit
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)
        cache.set(f"user:{user_id}", user, ttl=3600)
        return user
    ```

    Cache-aside is resilient because a cache failure simply means every request becomes a database query -- the system degrades rather than breaks. The downside is the first request for any key is always slow (a cold miss), and there is a brief window after a write where the cache holds stale data until the entry is invalidated or expires.

    ### Read-Through

    In a read-through configuration, the cache itself knows how to fetch data from the backing store. The application only ever talks to the cache, never directly to the database. When a miss occurs, the cache fetches the value, stores it, and returns it transparently.

    Read-through simplifies application code because the caching logic lives in the cache layer rather than scattered across services. The trade-off is that the cache must be configured with a loader function for each data source, making the cache layer more complex to operate.

    ### Comparison

    | Aspect              | Cache-Aside                  | Read-Through                 |
    |---------------------|------------------------------|------------------------------|
    | Cache miss handling | Application fetches from DB  | Cache fetches from DB        |
    | App complexity      | Higher (manages cache logic) | Lower (cache handles misses) |
    | Cache coupling      | Loose (app controls flow)    | Tight (cache knows DB)       |
    | Fault tolerance     | High (graceful degradation)  | Medium (depends on cache)    |
    | Best fit            | General-purpose workloads    | Uniform data access patterns |

    ### Real-World: Facebook TAO

    Facebook's TAO (The Associations and Objects) system is the canonical example of cache-aside at massive scale. TAO sits between Facebook's application tier and MySQL, caching the social graph -- every friend connection, like, comment, and post relationship.

    TAO serves over 10 billion read requests per day with a 99.8% cache hit rate. Each cache miss costs approximately 10ms (a MySQL query), while a cache hit returns in under 1ms. At Facebook's scale, even a 0.2% miss rate generates millions of database queries per day, which is why TAO uses a leader-follower cache topology: a single leader cache per region handles writes and invalidations, while multiple follower caches serve reads. This architecture keeps the cache consistent without requiring distributed locking.

    The lesson from TAO is that cache-aside works extraordinarily well for read-heavy social workloads, but at extreme scale you need careful attention to invalidation propagation and cache topology.

=== "Write Strategies"

    ## Write Strategies

    Write strategies determine what happens to the cache when data changes. The choice directly affects consistency guarantees, write latency, and risk of data loss.

    ### Write-Through

    In write-through caching, every write operation updates both the cache and the database synchronously. The write is only acknowledged as successful after both stores confirm the update.

    ```
    ┌─────────┐   1. Write    ┌───────┐   2. Write    ┌──────────┐
    │  Client  │ ───────────> │  App  │ ───────────> │  Cache    │
    └─────────┘               └───┬───┘               └────┬─────┘
                                  │                        │
                                  │   3. Write             │
                                  └───────────────> ┌──────┴─────┐
                                                    │  Database   │
                                                    └──────┬─────┘
                                                           │
                                        4. Both confirm    │
                                  <────────────────────────┘
                                  5. ACK to client
    ```

    Write-through guarantees that the cache never contains data the database does not also have. This makes it ideal for workloads where consistency is paramount, such as financial ledgers or authentication tokens. The cost is write latency: every write pays the penalty of two synchronous operations.

    ### Write-Behind (Write-Back)

    Write-behind decouples the client-facing write from the database write. The application writes to the cache and immediately acknowledges success. A background process asynchronously flushes dirty entries to the database, often batching multiple writes together for efficiency.

    ```
    ┌─────────┐  1. Write   ┌───────┐  2. Write   ┌──────────┐
    │  Client  │ ──────────> │  App  │ ──────────> │  Cache    │
    └─────────┘              └───────┘              └────┬─────┘
                                                        │
                              3. ACK (immediate)        │
                              <─────────────────────────┘
                                                        │
                                         4. Async flush │
                                         (batched)      │
                                                        v
                                                 ┌──────────┐
                                                 │ Database  │
                                                 └──────────┘
    ```

    Write-behind offers the lowest write latency and highest write throughput because the client never waits for the database. It also enables write coalescing: if the same key is updated ten times in one second, only the final value needs to be flushed. The critical risk is data loss -- if the cache node crashes before flushing, uncommitted writes are gone.

    ### Write-Around

    Write-around skips the cache entirely on writes. Data goes straight to the database, and the cache only gets populated on the next read (via cache-aside or read-through). This avoids polluting the cache with data that may never be read, but it means the first read after a write always incurs a cache miss.

    ### Consistency Implications

    | Strategy      | Write Latency | Data Loss Risk | Consistency    | Best For                    |
    |---------------|---------------|----------------|----------------|-----------------------------|
    | Write-Through | High (2x)     | None           | Strong         | Financial data, auth tokens |
    | Write-Behind  | Low           | Yes (on crash) | Eventual       | Analytics, logging, metrics |
    | Write-Around  | Medium        | None           | Stale on read  | Write-once, read-many data  |

    ### Real-World: DynamoDB DAX

    Amazon DynamoDB Accelerator (DAX) is a write-through cache for DynamoDB. When an application writes through DAX, the item is written to both the DAX cache and the DynamoDB table before the operation returns. This means subsequent reads from DAX always reflect the latest write, providing read-after-write consistency without any application-level cache invalidation logic.

    DAX delivers single-digit millisecond reads for DynamoDB tables that normally have single-digit millisecond writes. For read-heavy DynamoDB workloads (common in gaming leaderboards, ad-tech bidding, and session stores), DAX reduces read costs by up to 10x because cached reads are cheaper than table reads.

    The DAX example illustrates when write-through makes sense: when the same data is read far more often than it is written, and when consistency between cache and store is non-negotiable.

=== "Eviction Policies"

    ## Eviction Policies

    Every cache has finite memory. When the cache is full and a new entry needs space, the eviction policy decides which existing entry to remove. The choice of eviction policy can mean the difference between a 95% hit rate and a 60% hit rate on the same workload.

    ### LRU (Least Recently Used)

    LRU evicts the entry that has not been accessed for the longest time. It operates on the principle of temporal locality: data accessed recently is likely to be accessed again soon. LRU is the default eviction policy in Redis, Memcached, and most application-level caches.

    Conceptually, LRU maintains entries in access-time order. Every read or write moves the entry to the front. When eviction is needed, the entry at the back (least recently touched) is removed.

    ```python
    # Conceptual LRU: OrderedDict gives O(1) move-to-end and pop-first
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = OrderedDict()

        def get(self, key):
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

        def put(self, key, value):
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # evict oldest
    ```

    LRU's weakness is scan pollution. If a batch job iterates through a large dataset once, it pushes out frequently accessed hot data even though the scanned data will never be requested again. This is why databases like PostgreSQL use clock-sweep (an approximation of LRU that resists single-pass scans) rather than pure LRU.

    ### LFU (Least Frequently Used)

    LFU tracks how many times each entry has been accessed and evicts the entry with the lowest access count. This makes LFU excellent for workloads with stable popularity distributions, such as content delivery networks where a small set of videos or images accounts for most traffic.

    Redis added LFU support in version 4.0 (the `allkeys-lfu` and `volatile-lfu` policies). Redis implements an approximation called LFU with logarithmic frequency counting and a decay mechanism so that entries which were popular in the past but are no longer accessed eventually get evicted.

    The downside of LFU is slow adaptation. A newly added entry starts with a low frequency count and is vulnerable to immediate eviction, even if it is about to become very popular. The decay mechanism in Redis partially addresses this, but pure LFU can still struggle with rapidly shifting access patterns.

    ### TTL-Based Expiry

    TTL (Time To Live) is the simplest eviction mechanism: each entry has a timestamp, and the cache removes it after a fixed duration regardless of access patterns. TTL does not require tracking access order or frequency, making it the lowest-overhead policy.

    TTL is best used in combination with another policy. For example, Redis can run LRU eviction for memory pressure while also honoring TTL expirations. This layered approach means stale data expires automatically (via TTL) and memory stays bounded (via LRU).

    ### Comparison

    | Policy | Best For              | Weakness              | Used By            |
    |--------|-----------------------|-----------------------|--------------------|
    | LRU    | General purpose       | Scan pollution        | Redis, Memcached   |
    | LFU    | Popular content       | Slow to adapt         | Redis (since 4.0)  |
    | TTL    | Time-sensitive data   | No frequency awareness| All cache systems   |
    | Random | Simple systems        | Unpredictable         | Some CDNs           |

    ### Redis Eviction Policies

    Redis provides eight eviction policies that combine the concepts above. Understanding when to use each is essential for production Redis deployments.

    **volatile-lru** and **volatile-lfu** only evict keys that have a TTL set, using LRU or LFU ordering respectively. Use these when you mix persistent keys (no TTL) with ephemeral cached keys in the same Redis instance and want to protect the persistent keys from eviction.

    **allkeys-lru** and **allkeys-lfu** consider all keys for eviction regardless of TTL. Use `allkeys-lru` as the default for general-purpose caching. Use `allkeys-lfu` when your workload has clear popularity skew (a few keys get most of the traffic).

    **volatile-ttl** evicts keys with the shortest remaining TTL first. This is useful when you intentionally set shorter TTLs on less important data.

    **volatile-random** and **allkeys-random** evict a random key from the eligible set. Random eviction has surprisingly decent average-case performance and is sometimes used in CDN edge caches where the overhead of maintaining access-order metadata is unacceptable.

    **noeviction** refuses to evict anything and returns errors when memory is full. Use this when data loss is unacceptable and you would rather fail loudly than silently drop entries.

=== "Cache Problems"

    ## Cache Problems

    Caching introduces failure modes that do not exist in a simple application-to-database architecture. Understanding these failure modes is as important as understanding the strategies themselves, because a cache failure at scale can be worse than having no cache at all.

    ### Cache Stampede (Thundering Herd)

    A cache stampede occurs when a popular cache entry expires and many concurrent requests simultaneously discover the miss, all rushing to the database to regenerate the value.

    ```
    Cache key "trending_posts" expires at T=0

    T=0.001  Request A ──> Cache MISS ──> Query DB ─┐
    T=0.002  Request B ──> Cache MISS ──> Query DB ─┤
    T=0.003  Request C ──> Cache MISS ──> Query DB ─┤  DB overwhelmed
    T=0.004  Request D ──> Cache MISS ──> Query DB ─┤  by identical
      ...        ...           ...           ...    │  queries
    T=0.050  Request N ──> Cache MISS ──> Query DB ─┘
    ```

    If the cache entry serves 10,000 requests per second and the database query takes 200ms, a single expiration event can generate 2,000 simultaneous database queries for the same data. The database may slow down or crash, causing a cascading failure.

    **Solutions:**

    Mutex locking ensures that only one request regenerates the cache entry. All other requests wait for the lock holder to finish and then read the freshly cached value. This eliminates duplicate work but introduces lock contention.

    Probabilistic early expiry (also called stale-while-revalidate) has each request check whether the entry is "close to expiring" and probabilistically trigger a background refresh before the actual expiration. This spreads regeneration over time rather than concentrating it at the expiration moment.

    Request coalescing groups duplicate in-flight requests so that only one actually executes the database query, and all others receive the same result. Nginx and many CDNs implement this at the proxy level.

    ### Cache Penetration

    Cache penetration happens when requests repeatedly query for keys that do not exist in either the cache or the database. Since the data does not exist, there is nothing to cache, and every request falls through to the database.

    ```
    Attacker sends: GET /user/99999999 (non-existent ID)

    Request ──> Cache MISS ──> DB query ──> NULL ──> No cache entry
    Request ──> Cache MISS ──> DB query ──> NULL ──> No cache entry
    Request ──> Cache MISS ──> DB query ──> NULL ──> No cache entry
    (every request hits DB, cache provides zero protection)
    ```

    This can be an attack vector: an adversary sends millions of requests for random non-existent keys, bypassing the cache entirely and hammering the database.

    **Solutions:**

    Cache null results by storing a sentinel value (with a short TTL) for keys known to have no data. This converts future requests for the same non-existent key into cache hits.

    Bloom filters placed in front of the cache can quickly determine whether a key definitely does not exist, rejecting the request before it ever reaches the database. A bloom filter uses roughly 10 bits per entry, so even a billion-key dataset requires only about 1.2 GB of memory.

    ### Cache Avalanche

    Cache avalanche occurs when a large number of cache entries expire at the same time, causing a sudden flood of database queries. This differs from a stampede (which involves a single key) because it involves many keys simultaneously.

    ```
    All keys set with TTL=3600 at server start (T=0)

    T=3600: Thousands of keys expire simultaneously
            ┌──────────┐
            │  Cache   │  ──> Mass expiration
            └────┬─────┘
                 │
                 v  Thousands of DB queries
            ┌──────────┐
            │ Database │  ──> Overwhelmed
            └──────────┘
    ```

    This commonly happens when a cache is warmed at startup with uniform TTLs, or when a cache node restarts and all entries are cold.

    **Solutions:**

    TTL jitter adds a random offset to each key's TTL so that expirations are spread over time rather than synchronized. For example, instead of `TTL=3600`, use `TTL=3600 + random(0, 600)`, spreading expirations over a 10-minute window.

    Staggered warming populates the cache gradually rather than all at once, ensuring that entries created at different times will also expire at different times.

    Never-expire with background refresh keeps entries in cache indefinitely and uses a background process to update values. This eliminates expiration-triggered misses entirely, at the cost of slightly stale data.

    ### Real-World: Twitter During the World Cup

    During the 2014 FIFA World Cup, Twitter faced cache stampedes when trending tweets expired. A single popular tweet could generate hundreds of thousands of reads per second. When its cache entry expired, the resulting database stampede threatened to cascade into a site-wide outage.

    Twitter's engineering team solved this with a combination of early expiration (refreshing cache entries at 80% of their TTL) and request coalescing at the caching layer. They also introduced a "lease" mechanism similar to memcached's lease-get: the first request to encounter a miss receives a lease token granting it exclusive rights to regenerate the value, while all subsequent requests either wait or receive a slightly stale value. This approach reduced peak database load during major sporting events by over 90%.

---

## Eviction Policy Comparison

This table summarizes all eviction policies discussed above for quick reference when choosing a policy for your system.

| Policy | Mechanism                    | Hit Rate    | Overhead | Scan Resistant | Adaptive |
|--------|------------------------------|-------------|----------|----------------|----------|
| LRU    | Evict least recently used    | Good        | Low      | No             | No       |
| LFU    | Evict least frequently used  | Excellent*  | Medium   | Yes            | Slow     |
| FIFO   | Evict oldest entry           | Fair        | Very low | No             | No       |
| TTL    | Evict after time limit       | N/A (timed) | Very low | N/A            | N/A      |
| Random | Evict random entry           | Fair        | Very low | Yes            | No       |
| 2Q     | Two-queue promotion          | Very good   | Medium   | Yes            | Partial  |
| ARC    | Adaptive recency/frequency   | Excellent   | High     | Yes            | Yes      |

*LFU hit rate is excellent for stable popularity distributions but degrades when access patterns shift rapidly.

## Key Takeaways

1. Cache-aside is the right default for most read-heavy web applications. It is simple, resilient to cache failures, and pairs well with any eviction policy.

2. Write strategy choice is primarily a trade-off between consistency and latency. Write-through guarantees consistency at the cost of write speed. Write-behind maximizes throughput but risks data loss on failure.

3. LRU is the safest default eviction policy, but workloads with strong popularity skew benefit from LFU. Use Redis's `allkeys-lru` unless you have a specific reason not to.

4. Cache problems (stampede, penetration, avalanche) are not edge cases -- they are inevitable at scale. Design for them from the start with TTL jitter, null caching, and request coalescing.

5. Monitor hit rate religiously. A hit rate below 90% in a read-heavy workload usually indicates a problem with TTL tuning, key design, or cache sizing.

6. The best caching strategy is the one that matches your workload. There is no universal answer -- a system that needs strong consistency (banking) will use write-through, while a system that needs low-latency writes (analytics) will use write-behind.

## Related Topics

- [Databases Overview](../databases/index.md) -- understanding the data stores that caches sit in front of
- [Consistent Hashing](../../distributed-systems/consistent-hashing.md) -- how cache keys are distributed across multiple cache nodes
- [Performance Fundamentals](../../performance/fundamentals.md) -- broader performance optimization strategies beyond caching
