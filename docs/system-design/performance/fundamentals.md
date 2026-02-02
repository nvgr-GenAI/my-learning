# Performance Fundamentals

**Master system performance from response time to throughput** | ⚡ Metrics | 🎯 Optimization | 💼 Interview Ready

## Quick Reference

**Performance** - How well a system responds to workload demands:

| Metric | Target | Critical | What It Measures | Example |
|--------|--------|----------|-----------------|---------|
| **Latency (P95)** | <200ms | >1000ms | Response time for 95% of requests | API call: 150ms |
| **Throughput** | Per requirements | Declining | Requests processed per second | 10,000 RPS |
| **CPU Usage** | 70-80% | >90% | Processor utilization | 75% average |
| **Memory Usage** | 80-85% | >95% | RAM consumption | 82% with GC headroom |
| **Error Rate** | <0.1% | >1% | Failed requests percentage | 0.05% errors |
| **Cache Hit Rate** | >90% | <70% | Requests served from cache | 94% cache hits |

**Key Insight:** **Measure first, optimize second.** Premature optimization wastes time. Profile to find bottlenecks, then fix the biggest impact items.

---

=== "🎯 Understanding Performance"

    ## What is Performance?

    **Performance** is how efficiently your system uses resources to deliver results to users.

    ### The Restaurant Analogy

    ```
    🍽️ Restaurant Performance Metrics:
    ─────────────────────────────────────────────

    Latency (Response Time):
    - How long customers wait for food
    - Average: 15 minutes (misleading!)
    - P95: 20 minutes (95% served within this)
    - P99: 30 minutes (slow 1% = angry customers)

    Throughput (Capacity):
    - How many customers served per hour
    - Lunch rush: 50 customers/hour
    - Dinner rush: 80 customers/hour
    - Max capacity: 100 customers/hour

    Resource Utilization:
    - Kitchen: 70% busy (healthy)
    - Staff: 80% busy (healthy)
    - Oven: 90% busy (bottleneck!)

    Key Insight: Optimize the oven (bottleneck), not the staff!
    ```

    ---

    ## Performance vs Scalability

    | Aspect | Performance | Scalability |
    |--------|------------|-------------|
    | **Definition** | Speed of operations | Growth capacity |
    | **Focus** | Optimize existing system | Add resources |
    | **Metric** | Latency (response time) | Throughput (requests/sec) |
    | **Solution** | Algorithm optimization, caching | Add servers, sharding |
    | **Example** | Query: 500ms → 50ms (10x faster) | 1 server → 10 servers (10x capacity) |
    | **When to Use** | System is slow | System can't handle load |

    ---

    ## The Performance Mindset

    ```mermaid
    graph TB
        A[Measure<br/>Profile & monitor] --> B[Identify<br/>Find bottlenecks]
        B --> C[Analyze<br/>Root cause]
        C --> D[Optimize<br/>Fix bottleneck]
        D --> E[Validate<br/>Measure again]
        E --> A

        style A fill:#51cf66
        style D fill:#ff6b6b
    ```

    **Golden Rule:** Don't guess, measure!

=== "📊 Core Metrics"

    ## The 6 Essential Metrics

    === "1. Latency"

        ### Response Time (User-Facing)

        **What It Measures:**
        ```
        Latency = Time from request → response
        ─────────────────────────────────────────────

        User clicks button → [100ms] → Data appears

        Components:
        - Network time: 20ms
        - Server processing: 50ms
        - Database query: 25ms
        - Rendering: 5ms
        Total: 100ms
        ```

        ---

        ### Why Percentiles Matter

        ```
        Example API with 1000 requests:
        ─────────────────────────────────────────────
        Average latency: 100ms ← Misleading!

        Reality:
        - 900 requests: 50ms (fast)
        - 90 requests: 200ms (slow)
        - 9 requests: 1000ms (very slow)
        - 1 request: 10000ms (timeout!)

        Average = 100ms (looks good!)
        But 10% of users have bad experience!

        Better Metrics:
        - P50 (median): 50ms ✓
        - P95: 200ms (95% under this)
        - P99: 1000ms (99% under this)
        - P99.9: 10000ms (worst case)
        ```

        ---

        ### Industry Benchmarks

        | Operation | Target | Good | Bad |
        |-----------|--------|------|-----|
        | **Web Page Load** | <1s | <2s | >3s |
        | **API Call** | <50ms | <100ms | >500ms |
        | **Database Query** | <10ms | <50ms | >100ms |
        | **Cache Hit** | <1ms | <5ms | >10ms |
        | **Search Result** | <100ms | <500ms | >1s |

        ---

        ### Real-World Impact

        ```
        Google Study: Latency Impact
        ─────────────────────────────────────────────
        +100ms delay → 1% traffic loss
        +500ms delay → 20% traffic loss
        +1s delay → 50% traffic loss

        Amazon Study:
        +100ms delay → $1.6B annual revenue loss
        ```

    === "2. Throughput"

        ### Requests Per Second (System Capacity)

        **What It Measures:**
        ```
        Throughput = Requests handled per time unit
        ─────────────────────────────────────────────

        Example:
        - Peak traffic: 10,000 requests/second
        - Current capacity: 5,000 requests/second
        - Problem: Need 2x capacity!
        ```

        ---

        ### Throughput vs Latency

        ```
        The Inverse Relationship:
        ─────────────────────────────────────────────

        Low Load:
        - Latency: 50ms (fast!)
        - Throughput: 100 RPS

        Medium Load:
        - Latency: 100ms (slower)
        - Throughput: 500 RPS

        High Load:
        - Latency: 500ms (slow)
        - Throughput: 1,000 RPS (max capacity)

        Overload:
        - Latency: 5,000ms (timeout!)
        - Throughput: 800 RPS (declining!)
        - System: Degrading ❌

        Key: There's a sweet spot (70-80% utilization)
        ```

        ---

        ### Measuring Throughput

        | Metric | Description | Use Case |
        |--------|-------------|----------|
        | **RPS** | Requests Per Second | Web servers, APIs |
        | **QPS** | Queries Per Second | Databases |
        | **TPS** | Transactions Per Second | Payment systems |
        | **MPS** | Messages Per Second | Message queues |

    === "3. Resource Utilization"

        ### CPU, Memory, Disk, Network

        **Healthy Ranges:**

        | Resource | Healthy | Warning | Critical | Impact |
        |----------|---------|---------|----------|--------|
        | **CPU** | 60-75% | 75-85% | >90% | Slow processing |
        | **Memory** | 70-80% | 80-90% | >95% | Out of memory kills |
        | **Disk I/O** | <70% | 70-85% | >90% | Database bottleneck |
        | **Network** | <60% | 60-80% | >90% | Packet loss |

        ---

        ### Why Leave Headroom?

        ```
        Target: 70-80% Utilization
        ─────────────────────────────────────────────

        Reasons:
        1. Traffic Spikes
           - Normal: 1,000 RPS
           - Spike: 2,000 RPS (2x)
           - Headroom absorbs spike without overload

        2. Garbage Collection
           - Memory at 95% → Constant GC
           - GC pauses = Latency spikes

        3. Recovery Time
           - Server crash? Other servers absorb load
           - 80% utilization → Can handle 25% more

        4. Monitoring/Debugging
           - 100% CPU → Can't run profilers
           - Need space for troubleshooting
        ```

        ---

        ### The 90% Rule

        ```
        Why >90% is Dangerous:
        ─────────────────────────────────────────────

        Queueing Theory:
        - At 90% utilization, queue time doubles
        - At 95% utilization, queue time quadruples
        - At 99% utilization, queues explode

        Example:
        Normal (70% CPU): 100ms latency
        High (90% CPU): 300ms latency
        Critical (95% CPU): 800ms latency
        Danger (99% CPU): 5,000ms+ latency
        ```

    === "4. Error Rate"

        ### Failed Requests Percentage

        **What It Measures:**
        ```
        Error Rate = Failed Requests / Total Requests
        ─────────────────────────────────────────────

        Example:
        - Total requests: 10,000
        - Failed requests: 50
        - Error rate: 0.5%

        Types of Errors:
        - 4xx: Client errors (bad requests)
        - 5xx: Server errors (system failures)
        - Timeouts: Request too slow
        - Connection errors: Network issues
        ```

        ---

        ### Error Rate Targets

        | Service Level | Error Rate | 9's | Downtime/Year |
        |--------------|-----------|-----|---------------|
        | **Acceptable** | <0.1% | 99.9% | 8.7 hours |
        | **Good** | <0.01% | 99.99% | 52 minutes |
        | **Excellent** | <0.001% | 99.999% | 5.2 minutes |

    === "5. Cache Hit Rate"

        ### Cache Effectiveness

        **What It Measures:**
        ```
        Hit Rate = Cache Hits / Total Requests
        ─────────────────────────────────────────────

        Example:
        - 1,000 requests
        - 950 served from cache
        - 50 hit database
        - Cache hit rate: 95% ✓

        Performance Impact:
        - Cache hit: 5ms
        - Database query: 100ms
        - 20x faster with cache!
        ```

        ---

        ### The 80/20 Rule

        ```
        Pareto Principle:
        ─────────────────────────────────────────────
        20% of data = 80% of requests

        Strategy:
        1. Identify hot data (top 20%)
        2. Cache only hot data
        3. Achieve 80%+ hit rate

        Example: E-commerce
        - Total products: 100,000
        - Popular products: 20,000 (20%)
        - Cache: 20,000 products
        - Hit rate: 85% ✓
        - Memory saved: 80%
        ```

    === "6. Apdex Score"

        ### Application Performance Index

        **What It Measures:**
        ```
        User Satisfaction Metric
        ─────────────────────────────────────────────

        Apdex = (Satisfied + Tolerating/2) / Total

        Classification:
        - Satisfied: Response time ≤ T (e.g., ≤ 500ms)
        - Tolerating: T < Response ≤ 4T (500ms - 2s)
        - Frustrated: Response > 4T (> 2s)

        Example (T = 500ms):
        - 700 requests: ≤ 500ms (satisfied)
        - 200 requests: 500-2000ms (tolerating)
        - 100 requests: > 2000ms (frustrated)

        Apdex = (700 + 200/2) / 1000 = 0.8

        Score Interpretation:
        - 1.0: Perfect
        - 0.94-1.0: Excellent
        - 0.85-0.93: Good
        - 0.70-0.84: Fair
        - 0.50-0.69: Poor
        - <0.50: Unacceptable
        ```

=== "🏗️ Optimization Techniques"

    ## The 5-Layer Optimization Strategy

    === "1. Algorithm Optimization"

        ### Choose the Right Algorithm

        **Time Complexity Impact:**

        | Complexity | 100 items | 10,000 items | 1,000,000 items |
        |-----------|-----------|--------------|-----------------|
        | **O(1)** | 1 operation | 1 operation | 1 operation |
        | **O(log n)** | 7 operations | 13 operations | 20 operations |
        | **O(n)** | 100 operations | 10,000 operations | 1,000,000 operations |
        | **O(n log n)** | 700 operations | 130,000 operations | 20,000,000 operations |
        | **O(n²)** | 10,000 operations | 100,000,000 operations | 1,000,000,000,000 operations |

        ---

        ### Real-World Example

        ```
        Problem: Find user by ID in list
        ─────────────────────────────────────────────

        Bad (Linear Search - O(n)):
        for user in users:
            if user.id == target_id:
                return user
        Time: 500ms for 1M users

        Good (Hash Map - O(1)):
        users_map = {user.id: user for user in users}
        return users_map[target_id]
        Time: 0.001ms for 1M users

        Result: 500,000x faster!
        ```

    === "2. Database Optimization"

        ### The 3 Database Killers

        **1. N+1 Query Problem:**
        ```
        Bad (N+1 queries):
        ─────────────────────────────────────────────
        users = db.query("SELECT * FROM users")      # 1 query
        for user in users:
            orders = db.query(f"SELECT * FROM orders
                              WHERE user_id = {user.id}")  # N queries

        Total: 1 + N queries (N = 1000 → 1001 queries!)
        Time: 10 seconds

        Good (JOIN):
        ─────────────────────────────────────────────
        results = db.query("""
            SELECT users.*, orders.*
            FROM users
            LEFT JOIN orders ON users.id = orders.user_id
        """)

        Total: 1 query
        Time: 100ms

        Result: 100x faster!
        ```

        ---

        **2. Missing Indexes:**
        ```
        Bad (Table Scan):
        ─────────────────────────────────────────────
        SELECT * FROM users WHERE email = 'john@example.com';

        Without index:
        - Scans all 10M rows
        - Time: 5 seconds

        Good (Index Lookup):
        ─────────────────────────────────────────────
        CREATE INDEX idx_email ON users(email);
        SELECT * FROM users WHERE email = 'john@example.com';

        With index:
        - Direct lookup
        - Time: 5ms

        Result: 1,000x faster!
        ```

        ---

        **3. SELECT * (Fetching Too Much):**
        ```
        Bad:
        ─────────────────────────────────────────────
        SELECT * FROM users;  # 50 columns, 10MB data

        Good:
        ─────────────────────────────────────────────
        SELECT id, name, email FROM users;  # 3 columns, 1MB data

        Result: 10x less data transfer
        ```

        ---

        ### Connection Pooling

        ```
        Without Pooling:
        ─────────────────────────────────────────────
        For each request:
        1. Open connection (50ms)
        2. Execute query (10ms)
        3. Close connection (20ms)
        Total: 80ms per request

        With Pooling:
        ─────────────────────────────────────────────
        Pre-create 10-20 connections
        For each request:
        1. Get connection from pool (0.1ms)
        2. Execute query (10ms)
        3. Return to pool (0.1ms)
        Total: 10ms per request

        Result: 8x faster!
        ```

    === "3. Caching"

        ### Multi-Layer Caching Strategy

        **The Caching Hierarchy:**
        ```
        Request Flow:
        ─────────────────────────────────────────────

        1. Browser Cache (0ms)
           ↓ Miss
        2. CDN Edge (50ms) ← 95% of requests stop here
           ↓ Miss
        3. Redis Cache (5ms)
           ↓ Miss
        4. Database (100ms)

        Without cache: 100ms per request
        With 95% hit rate: 0.95 × 5ms + 0.05 × 100ms = 9.75ms
        Result: 10x faster!
        ```

        ---

        ### Cache-Aside Pattern

        ```
        Read:
        ─────────────────────────────────────────────
        1. Check cache
           └─ Hit? Return data (fast!)
           └─ Miss? Go to step 2

        2. Query database
        3. Store in cache (TTL = 5 minutes)
        4. Return data

        Write:
        ─────────────────────────────────────────────
        1. Update database
        2. Invalidate cache
        3. Next read will fetch fresh data
        ```

        ---

        ### Cache Eviction Policies

        | Policy | When to Remove | Use Case |
        |--------|---------------|----------|
        | **LRU** | Least recently accessed | General purpose, temporal locality |
        | **LFU** | Least frequently accessed | Popular content (videos, articles) |
        | **TTL** | After expiration time | Time-sensitive data (prices, stocks) |
        | **FIFO** | Oldest added first | Simple, predictable eviction |

        ---

        ### Cache Sizing

        ```
        80/20 Rule for Cache Sizing:
        ─────────────────────────────────────────────

        Total data: 100GB
        Hot data (20%): 20GB
        Cache size: 30GB (20GB × 1.5 buffer)

        Expected results:
        - Hit rate: 80-85%
        - Memory: 30GB
        - Cost-effective!
        ```

    === "4. Async Processing"

        ### Offload Heavy Operations

        **Synchronous (Blocking):**
        ```
        User uploads image → [5 seconds] → Response

        Steps:
        1. Upload image (1s)
        2. Generate thumbnails (2s)
        3. Apply watermark (1s)
        4. Update database (1s)
        Total: 5s (user waits!)
        ```

        **Asynchronous (Non-Blocking):**
        ```
        User uploads image → [100ms] → "Processing..." Response

        Immediate:
        1. Upload image (100ms)
        2. Queue job
        3. Return immediately

        Background:
        4. Worker generates thumbnails
        5. Worker applies watermark
        6. Worker updates database
        7. Webhook notifies user

        Result: 50x faster response!
        ```

        ---

        ### When to Use Async

        | Operation | Should Be Async? | Why |
        |-----------|-----------------|-----|
        | **Email sending** | ✓ Yes | User doesn't need to wait |
        | **Report generation** | ✓ Yes | Takes minutes, can be queued |
        | **Image processing** | ✓ Yes | CPU-intensive, slow |
        | **Payment processing** | ✗ No | User needs immediate confirmation |
        | **Login authentication** | ✗ No | Must be synchronous |

    === "5. Network Optimization"

        ### Reduce Data Transfer

        **Compression:**
        ```
        Uncompressed Response:
        ─────────────────────────────────────────────
        JSON payload: 1MB
        Transfer time: 1,000ms (1Mbps connection)

        Gzip Compressed:
        ─────────────────────────────────────────────
        Compressed: 200KB (80% reduction)
        Transfer time: 200ms

        Result: 5x faster!

        Compression Ratios:
        - HTML/JSON: 70-90%
        - Images (PNG): 10-30% (already compressed)
        - Videos: 0-5% (don't compress)
        ```

        ---

        ### CDN Benefits

        ```
        Without CDN:
        ─────────────────────────────────────────────
        Tokyo user → US server
        - Distance: 10,000 km
        - Latency: 200ms
        - Bandwidth cost: High

        With CDN:
        ─────────────────────────────────────────────
        Tokyo user → Tokyo edge server
        - Distance: 50 km
        - Latency: 10ms (20x faster!)
        - Bandwidth: 95% served from edge (cheap)
        - Origin server: Only 5% of traffic

        Cost savings: ~$800/month for 1TB
        ```

        ---

        ### HTTP/2 Multiplexing

        ```
        HTTP/1.1:
        ─────────────────────────────────────────────
        6 parallel connections
        - Connection 1: HTML
        - Connection 2: CSS
        - Connection 3: JS
        - Connection 4: Image 1
        - Connection 5: Image 2
        - Connection 6: Image 3

        HTTP/2:
        ─────────────────────────────────────────────
        1 connection, multiplexed
        - All resources over single connection
        - No head-of-line blocking
        - Header compression (HPACK)

        Result: 30-50% faster page load
        ```

=== "💡 Interview Tips"

    ## Common Interview Questions

    **Q1: "Explain the difference between latency and throughput"**

    **Good Answer:**
    ```
    Latency (Response Time):
    ─────────────────────────────
    - How long ONE request takes
    - Measured in milliseconds
    - User-facing metric (feels slow/fast)
    - Example: API call takes 100ms

    Throughput (Capacity):
    ─────────────────────────────
    - How MANY requests system handles per second
    - Measured in requests/second
    - System capacity metric
    - Example: System handles 10,000 requests/second

    Key Relationship:
    ─────────────────────────────
    - Low load: Low latency, variable throughput
    - High load: High latency, max throughput
    - Overload: Very high latency, declining throughput

    Real-World Analogy:
    ─────────────────────────────
    Highway:
    - Latency: Time to drive from A to B
    - Throughput: Cars passing per hour

    At rush hour:
    - Latency increases (traffic jams)
    - Throughput reaches maximum (all lanes full)
    ```

    ---

    **Q2: "Why use P95/P99 instead of average latency?"**

    **Good Answer:**
    ```
    Average Hides Problems:
    ─────────────────────────────────────────────

    Example with 100 requests:
    - 90 requests: 50ms (fast)
    - 9 requests: 500ms (slow)
    - 1 request: 5000ms (very slow)

    Average: 95ms (looks good!)
    Reality: 10% of users have bad experience

    Percentiles Show Truth:
    ─────────────────────────────────────────────
    - P50 (median): 50ms (typical user)
    - P95: 500ms (95% of users under this)
    - P99: 5000ms (worst 1% - still important!)

    Why P95/P99 Matter:
    ─────────────────────────────────────────────
    1. Heavy users hit slowness more
    2. They're your most valuable customers
    3. They might have more data (edge cases)
    4. They influence churn decisions

    Industry Standard: Optimize for P95
    Google/Amazon: Optimize for P99
    ```

    ---

    **Q3: "How would you optimize a slow database query?"**

    **Good Answer:**
    ```
    Step-by-Step Approach:

    1. Measure First:
    ─────────────────────────────
    - Run EXPLAIN on query
    - Identify table scans
    - Check execution time

    2. Add Indexes:
    ─────────────────────────────
    Bad: SELECT * FROM users WHERE email = 'john@example.com'
         (Table scan: 5 seconds)

    Good: CREATE INDEX idx_email ON users(email);
          (Index lookup: 5ms → 1000x faster!)

    3. Fix N+1 Queries:
    ─────────────────────────────
    Bad: 1 query + 1000 queries in loop = 1001 queries
    Good: Single JOIN query = 1 query

    4. Select Only Needed Columns:
    ─────────────────────────────
    Bad: SELECT * FROM users (50 columns, 10MB)
    Good: SELECT id, name, email (3 columns, 1MB)

    5. Add Caching:
    ─────────────────────────────
    - Cache frequent queries (Redis)
    - TTL: 5 minutes
    - Hit rate: 90%+
    - Reduces DB load by 90%

    6. Read Replicas (if still slow):
    ─────────────────────────────
    - Separate read/write databases
    - Distribute reads across 3 replicas
    - 3x read capacity

    Priority: Fix algorithm > Add index > Cache > Scale
    ```

    ---

    **Q4: "What's the N+1 query problem?"**

    **Good Answer:**
    ```
    The Problem:
    ─────────────────────────────────────────────

    Fetch users:
    SELECT * FROM users;  # 1 query, returns 1000 users

    For each user, fetch their orders:
    for user in users:
        SELECT * FROM orders WHERE user_id = user.id;  # 1000 queries!

    Total: 1 + 1000 = 1001 queries
    Time: 10 seconds (10ms per query)

    The Solution:
    ─────────────────────────────────────────────

    Single JOIN query:
    SELECT users.*, orders.*
    FROM users
    LEFT JOIN orders ON users.id = orders.user_id;

    Total: 1 query
    Time: 100ms

    Result: 100x faster!

    Real-World Impact:
    ─────────────────────────────────────────────
    - Common in ORMs (Lazy loading)
    - Easy to miss in development (small datasets)
    - Catastrophic in production (large datasets)
    - Solution: Eager loading, batch queries
    ```

    ---

    **Q5: "How do you handle cache invalidation?"**

    **Good Answer:**
    ```
    The Three Strategies:

    1. TTL (Time To Live):
    ─────────────────────────────────────────────
    - Set expiration time (e.g., 5 minutes)
    - Pros: Simple, automatic
    - Cons: May serve stale data
    - Use: Frequently changing data (prices, stocks)

    2. Invalidate on Write:
    ─────────────────────────────────────────────
    - Delete cache entry when data updated
    - Pros: Always fresh
    - Cons: Cache miss on next read
    - Use: User profiles, settings

    3. Write-Through:
    ─────────────────────────────────────────────
    - Update both cache and database
    - Pros: Cache always valid
    - Cons: Write latency, sync complexity
    - Use: Critical data (inventory, balances)

    Real-World Pattern:
    ─────────────────────────────────────────────
    Combine strategies:
    - TTL: 5 minutes (safety net)
    - Invalidate on write (immediate freshness)
    - Best of both worlds

    The Hard Part:
    ─────────────────────────────────────────────
    "There are only two hard things in Computer Science:
     cache invalidation and naming things."
    - Phil Karlton
    ```

    ---

    ## Interview Cheat Sheet

    **Performance Targets:**

    | System | Latency | Throughput | Notes |
    |--------|---------|-----------|-------|
    | **Web Page** | <1s (P95) | 1,000+ RPS | First contentful paint |
    | **API** | <100ms (P95) | 10,000+ RPS | RESTful endpoints |
    | **Database** | <50ms (P95) | 10,000+ QPS | Simple queries |
    | **Cache** | <1ms (P95) | 100,000+ RPS | Redis/Memcached |
    | **CDN** | <50ms (P95) | 1M+ RPS | Edge locations |

    **Quick Wins (Biggest Impact):**
    1. **Add caching** (10x improvement for read-heavy)
    2. **Fix N+1 queries** (100x for ORM queries)
    3. **Add database indexes** (1000x for searches)
    4. **Use CDN** (20x for global users)
    5. **Enable compression** (5x for text content)

    **Resource Utilization Targets:**
    - CPU: 70-80% (leave headroom for spikes)
    - Memory: 80-85% (headroom for GC)
    - Disk: <80% (performance degrades above)
    - Network: <70% (packet loss above)

=== "⚠️ Common Mistakes"

    ## Performance Pitfalls

    | Mistake | Problem | Solution |
    |---------|---------|----------|
    | **Premature optimization** | Optimize before measuring | Profile first, optimize bottlenecks |
    | **Ignoring P95/P99** | Focus only on average | Optimize tail latency (worst 5%) |
    | **No caching** | Every request hits DB | Cache hot data (80/20 rule) |
    | **SELECT *** | Fetch all columns | Select only needed columns |
    | **N+1 queries** | Queries in loops | Use JOINs or batch queries |
    | **Missing indexes** | Table scans | Index WHERE/ORDER BY columns |
    | **Synchronous processing** | Block on slow operations | Queue heavy jobs (async) |
    | **No monitoring** | Don't know what's slow | APM, metrics, distributed tracing |

    ---

    ## Design Pitfalls

    | Pitfall | Impact | Prevention |
    |---------|--------|-----------|
    | **Single database** | Bottleneck at 10K QPS | Read replicas + caching |
    | **No connection pooling** | 50ms overhead per request | Pool with 10-20 connections |
    | **Large payloads** | Slow transfer, high bandwidth | Paginate, compress, CDN |
    | **Chatty APIs** | Multiple round trips | Batch requests, GraphQL |
    | **Hot partitions** | Uneven load distribution | Better shard key selection |
    | **Blocking I/O** | Thread starvation | Async I/O, event-driven |

    ---

    ## Interview Red Flags

    **Avoid Saying:**
    - ❌ "Just add more servers" (ignores algorithm issues)
    - ❌ "Caching solves everything" (cache invalidation is hard)
    - ❌ "Average latency is 100ms so we're good" (ignores tail latency)
    - ❌ "We'll optimize later" (performance is foundational)
    - ❌ "NoSQL is always faster" (depends on use case)

    **Say Instead:**
    - ✅ "Profile first to find bottlenecks, then optimize"
    - ✅ "Monitor P95/P99 latency, not just average"
    - ✅ "Cache read-heavy data with 80/20 rule"
    - ✅ "Fix algorithm issues before scaling infrastructure"
    - ✅ "Choose database based on access patterns"

---

## 🎯 Key Takeaways

**The 10 Rules of Performance:**

1. **Measure first** - Can't optimize what you don't measure. Profile before optimizing.

2. **Optimize for P95/P99** - Average hides problems. 5% of users matter too.

3. **Fix algorithms first** - O(n²) → O(n log n) beats any hardware upgrade.

4. **Cache aggressively** - 80/20 rule: Cache 20% of data, serve 80% of requests.

5. **Indexes are critical** - Missing index = 1000x slower queries.

6. **Avoid N+1 queries** - Use JOINs, not queries in loops.

7. **Async for heavy work** - Don't make users wait for emails/reports/processing.

8. **Leave headroom** - Target 70-80% utilization, not 100%.

9. **CDN for global users** - 10ms edge latency vs 200ms cross-continent.

10. **Monitor continuously** - Performance degrades over time. Watch trends.

---

## 📚 Further Reading

**Master these related concepts:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **Caching Strategies** | 10x performance boost | [Caching Patterns →](../performance/caching.md) |
| **Database Optimization** | Fix biggest bottleneck | [Database Performance →](../databases/performance.md) |
| **Load Testing** | Validate performance | [Testing Guide →](../testing/performance-testing.md) |
| **Monitoring** | Detect issues early | [Observability →](../monitoring/apm.md) |

**Practice with real systems:**
- [Design Twitter](../problems/twitter.md) - Cache optimization, read-heavy
- [Design Netflix](../problems/netflix.md) - CDN, video streaming
- [Design Amazon](../problems/amazon.md) - Database optimization, caching

---

**Master performance fundamentals and build blazingly fast systems! ⚡**
