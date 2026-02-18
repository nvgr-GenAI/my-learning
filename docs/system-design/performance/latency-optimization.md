# Latency Optimization Techniques

Latency is the time a user waits between making a request and seeing a result. While performance fundamentals cover *what* to measure, this page covers *where* to optimize. Every optimization technique here targets a specific segment of the request lifecycle, and the key insight is that not all segments contribute equally to total latency. The difference between a mediocre and excellent system is usually not a single breakthrough -- it is dozens of small improvements applied to the right places.

!!! tip "Core Principle"
    Optimization without measurement is guessing. Always profile first, then apply Amdahl's Law: improving a component that accounts for 5% of total latency can never yield more than a 5% improvement, no matter how much you optimize it.

---

## The Latency Budget

Every request passes through multiple stages. A **latency budget** decomposes end-to-end latency into per-stage allocations, making it visible where time is actually spent. This is the single most useful exercise before optimizing anything.

```
Request Lifecycle with Typical Latency Breakdown
================================================================

User clicks       DNS       TCP        TLS       Server      DB        Render
  button        Lookup    Connect    Handshake   Process    Query     Response
    |             |          |          |          |          |          |
    v             v          v          v          v          v          v
    +-----+   +-----+   +-----+   +------+   +------+   +------+   +-----+
    | 0ms |-->| 20ms|-->| 30ms|-->| 40ms |-->| 80ms |-->|150ms |-->| 50ms|
    +-----+   +-----+   +-----+   +------+   +------+   +------+   +-----+
    |                                                                      |
    |<=================== Total: ~370ms =================================>|

Where does time actually go? (typical web application)
================================================================

    Network transit      ████████████░░░░░░░░  25%   (DNS, TCP, TLS, transfer)
    Database queries     ████████████████░░░░  40%   (queries, connection wait)
    Application logic    ██████░░░░░░░░░░░░░░  15%   (compute, serialization)
    Client rendering     ████░░░░░░░░░░░░░░░░  10%   (DOM, paint, JS execution)
    Queueing/overhead    ██░░░░░░░░░░░░░░░░░░  10%   (GC, scheduling, retries)

    --> Database + Network = 65% of total latency in most applications
    --> This is where optimization effort should concentrate first
```

The latency budget makes a counterintuitive point clear: most teams spend their optimization effort on application logic (15% of the budget) and ignore database queries (40%). The budget forces you to optimize where the time actually is.

---

## Network Optimizations

Network latency is governed by physics (speed of light), protocol overhead (handshakes), and data volume (bytes on the wire). You cannot change physics, but you can reduce round trips, compress data, and move computation closer to the user.

### Connection Pooling and Keep-Alive

Every new TCP connection requires a three-way handshake (1 RTT), and every new TLS connection adds another 1-2 RTTs on top. For a user 100ms away from your server, that is 200-300ms before a single byte of application data moves.

```
Without Keep-Alive (HTTP/1.0 default)
========================================
Client         Server
  |--- SYN ------->|       }
  |<-- SYN-ACK ----|       } TCP handshake: 1 RTT
  |--- ACK ------->|       }
  |--- Request --->|
  |<-- Response ---|
  |--- FIN ------->|  Connection closed
  |                |
  |--- SYN ------->|  Repeat for EVERY request
  |<-- SYN-ACK ----|
  ...

With Keep-Alive (HTTP/1.1 default)
========================================
Client         Server
  |--- SYN ------->|       }
  |<-- SYN-ACK ----|       } TCP handshake: 1 RTT (once)
  |--- ACK ------->|       }
  |--- Request 1 ->|
  |<-- Response 1 -|
  |--- Request 2 ->|       Reuse same connection
  |<-- Response 2 -|
  |--- Request 3 ->|
  |<-- Response 3 -|
  ...
```

**Impact:** Eliminates 100-300ms per subsequent request. HTTP/1.1 enabled keep-alive by default, which is why most modern systems benefit automatically. The optimization now is ensuring your load balancers and reverse proxies do not close connections prematurely.

### Response Compression

Compressing response bodies reduces bytes on the wire, which directly reduces transfer time. The trade-off is CPU time on both sides: the server must compress, and the client must decompress.

| Algorithm | Compression Ratio | CPU Cost  | Decompression Speed | Best For |
|-----------|:-----------------:|:---------:|:-------------------:|----------|
| gzip      | Good (60-70%)     | Moderate  | Fast                | Universal compatibility |
| Brotli    | Better (70-80%)   | Higher    | Fast                | Static assets (pre-compressed) |
| zstd      | Better (70-80%)   | Lower     | Very fast           | API responses, real-time data |

!!! info "When Compression Hurts"
    Compression adds latency when the payload is small (under 1KB, overhead exceeds savings), already compressed (images, video), or when CPU is the bottleneck rather than bandwidth. Most CDNs and reverse proxies skip compression for small responses automatically.

**Real-world numbers:** Cloudflare reports that Brotli reduces HTML transfer sizes by 17-25% compared to gzip, which translates to 50-100ms faster page loads on 3G connections. For API responses returning JSON, gzip alone typically reduces payload size by 85-90%.

### HTTP/2 Multiplexing and HTTP/3 with QUIC

HTTP/1.1 suffers from **head-of-line blocking**: the browser can open 6 parallel TCP connections to a server, but each connection processes requests sequentially. If one request is slow, it blocks everything behind it on that connection.

```
HTTP/1.1: Head-of-Line Blocking
========================================
Connection 1:  [====Req A====]...[==Req D==]
Connection 2:  [=Req B=].........[====Req E (slow)====]...[Req G]
Connection 3:  [===Req C===].....[==Req F==]
               ^                  ^                        ^
               |                  |                        |
            All 3 start        B finishes early         G waits for
            together           but conn 2 is blocked    slow E

HTTP/2: Multiplexed Streams
========================================
Single Connection:
  Stream 1: [====Req A====]
  Stream 2: [=Req B=]
  Stream 3: [===Req C===]
  Stream 4:          [==Req D==]
  Stream 5: [====Req E (slow)====]
  Stream 6:          [==Req F==]
  Stream 7:    [Req G]  <-- Not blocked by E!

  All streams interleave on ONE connection.
  No head-of-line blocking at HTTP level.
```

**HTTP/3 with QUIC** goes further. HTTP/2 solved head-of-line blocking at the HTTP layer, but TCP still has its own: if a single TCP packet is lost, all streams on that connection stall until the packet is retransmitted. QUIC replaces TCP with UDP-based transport where each stream is independent -- a lost packet on stream 5 does not block stream 7.

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 (QUIC) |
|---------|:--------:|:------:|:--------------:|
| Connections per domain | 6 parallel | 1 multiplexed | 1 multiplexed |
| Head-of-line blocking | HTTP + TCP | TCP only | None |
| Connection setup | TCP + TLS (2-3 RTT) | TCP + TLS (2-3 RTT) | 0-1 RTT |
| Connection migration | No | No | Yes (WiFi to cell) |

**Impact:** Google measured a 3% reduction in search latency when deploying QUIC, with larger improvements on lossy mobile networks (8% on poor connections). YouTube saw a 30% reduction in rebuffering when switching to QUIC.

### Edge Computing and CDN

The speed of light imposes a hard floor: a round trip from New York to Singapore takes ~250ms. No amount of server optimization can fix that. The solution is to move computation and data closer to the user.

```
Without CDN: All requests go to origin
========================================

  User (Tokyo)  ------- 180ms -------> Origin (Virginia)
  User (London) ------- 80ms --------> Origin (Virginia)
  User (Sydney) ------- 200ms -------> Origin (Virginia)

With CDN: Edge nodes serve cached content
========================================

  User (Tokyo)  ---- 5ms ----> Edge (Tokyo) --cache hit--> Response
  User (London) ---- 5ms ----> Edge (London) --cache miss--> Origin
  User (Sydney) ---- 5ms ----> Edge (Sydney) --cache hit--> Response

Latency reduction: 95-97% for cached content
```

Modern edge platforms (Cloudflare Workers, AWS Lambda@Edge, Vercel Edge Functions) go beyond static caching. They run application logic at the edge -- authentication checks, A/B test routing, personalization, and API response assembly. This moves the "application logic" portion of the latency budget from origin to edge, saving 50-200ms per request for geographically distant users.

---

## Database Optimizations

Databases are the largest single contributor to backend latency in most applications. The optimizations here target connection overhead, query execution time, and read distribution.

### Connection Pooling

Establishing a database connection involves TCP handshake, authentication, SSL negotiation, and session setup. A single PostgreSQL connection takes 20-50ms to establish. Without pooling, an application opening a connection per request at 1,000 RPS would spend 20-50 seconds of cumulative connection time every second.

```
Without Pool: New connection per query
========================================
App --> [connect 25ms] --> [query 5ms] --> [disconnect]
App --> [connect 25ms] --> [query 5ms] --> [disconnect]
App --> [connect 25ms] --> [query 5ms] --> [disconnect]
Total: 90ms for 3 queries  (83% connection overhead!)

With Pool: Reuse pre-established connections
========================================
                    +-- conn 1 (idle) --+
App --> Pool -------+-- conn 2 (idle) --+----> Database
                    +-- conn 3 (idle) --+

App --> [acquire 0.1ms] --> [query 5ms] --> [release to pool]
App --> [acquire 0.1ms] --> [query 5ms] --> [release to pool]
App --> [acquire 0.1ms] --> [query 5ms] --> [release to pool]
Total: 15.3ms for 3 queries  (6x faster)
```

| Tool | Database | Mode | Best For |
|------|----------|------|----------|
| PgBouncer | PostgreSQL | Transaction pooling | High-concurrency apps, thousands of clients |
| ProxySQL | MySQL | Connection multiplexing | MySQL clusters, read/write splitting |
| pgcat | PostgreSQL | Session/transaction | Sharded PostgreSQL, load balancing |

!!! warning "Pool Sizing Trap"
    More connections is not better. PostgreSQL performance degrades sharply above ~100-200 active connections due to lock contention and context switching. The formula from the PostgreSQL wiki: `connections = (2 * CPU cores) + disk spindles`. For a 16-core server with SSD, that is roughly 33 connections -- far fewer than most developers expect.

### Query Optimization and Covering Indexes

The difference between a table scan and an index lookup is the difference between reading every row in a million-row table versus reading exactly the rows you need. At scale, this is the difference between a 500ms query and a 2ms query.

**Covering indexes** are the most powerful single optimization for read-heavy queries. A covering index contains all columns referenced in a query, so the database never needs to visit the actual table rows at all.

```
Query: SELECT name, email FROM users WHERE status = 'active' AND country = 'US'

Without covering index:
  1. Scan index on (status, country) --> get row IDs       ~2ms
  2. For each row ID, fetch full row from table (random I/O) ~15ms
  Total: ~17ms

With covering index on (status, country, name, email):
  1. Scan index --> all needed data is IN the index         ~2ms
  2. No table access needed (index-only scan)               ~0ms
  Total: ~2ms  (8x faster)
```

!!! tip "The EXPLAIN Habit"
    Run `EXPLAIN ANALYZE` on every query that appears in your slow query log. Look for sequential scans on large tables, nested loop joins without indexes, and sort operations that spill to disk. These three patterns account for the majority of slow queries.

### Materialized Views

Some queries are inherently expensive -- aggregations over millions of rows, multi-table joins with complex filters, or analytics across time ranges. Even with perfect indexes, these queries take hundreds of milliseconds or seconds.

A **materialized view** precomputes the result and stores it as a table. Reads become simple table lookups (microseconds), at the cost of freshness -- the view must be refreshed periodically.

```
Expensive Query (runs every time a dashboard loads)
========================================
SELECT department, COUNT(*), AVG(salary), MAX(hire_date)
FROM employees JOIN departments ON ...
WHERE hire_date > '2024-01-01'
GROUP BY department
--> Scans 2M rows, joins 2 tables: ~800ms

Materialized View (refreshed every 15 minutes)
========================================
SELECT * FROM department_summary
--> Reads precomputed result: ~2ms

Trade-off:
  Speed:     800ms --> 2ms (400x faster)
  Freshness: Up to 15 minutes stale
  Storage:   Extra table to maintain
```

**When to use:** Dashboard queries, reporting endpoints, leaderboards, analytics aggregations -- anywhere where data can be slightly stale and the query is expensive. Netflix uses materialized views extensively for their personalization pipeline, precomputing recommendation scores rather than calculating them on each page load.

### Read Replicas

In most applications, reads outnumber writes by 10:1 to 100:1. Directing all reads to the primary database wastes capacity and creates unnecessary contention with writes. Read replicas distribute read load across multiple copies of the data.

```
Before: Single database handles everything
========================================
        +-- Read (40%) ---+
App --> +-- Read (40%) ---+--> Primary DB (overloaded)
        +-- Write (20%) --+

After: Reads distributed to replicas
========================================
        +-- Read -------> Replica 1
App --> +-- Read -------> Replica 2    (each handles 20% of total)
        +-- Read -------> Replica 3
        +-- Write ------> Primary --replicates--> Replicas

Result: Primary handles only writes (20% of traffic)
        Each replica handles 1/3 of reads
        Total DB capacity: ~3x higher
```

!!! warning "Replication Lag"
    Replicas are eventually consistent. After a write to the primary, there is a window (typically 10-100ms, but up to seconds under load) where replicas return stale data. The "read-your-own-writes" pattern handles this: after a user writes data, route their subsequent reads to the primary for a brief window, then fall back to replicas.

---

## Application Optimizations

Application-level optimizations reduce the work your server does per request. These often require code changes but yield significant improvements because they compound across every request.

### Pagination Strategies

Any endpoint returning a list of items needs pagination. The two primary approaches have fundamentally different performance characteristics.

```
Offset-Based:  GET /messages?offset=10000&limit=20
Cursor-Based:  GET /messages?after=msg_abc123&limit=20
```

**How offset pagination degrades:**

```
Offset = 0:      Scan 20 rows, return 20          ~2ms
Offset = 1,000:  Scan 1,020 rows, discard 1,000   ~15ms
Offset = 10,000: Scan 10,020 rows, discard 10,000 ~150ms
Offset = 100,000: Scan 100,020 rows, discard all   ~1,200ms

The database must count through ALL skipped rows every time.
There is no shortcut -- the DB cannot "jump" to row 10,000.
```

| Aspect | Offset-Based | Cursor-Based |
|--------|:------------:|:------------:|
| Performance at depth | Degrades linearly | Constant O(1) |
| Random page access | Yes (page 47 directly) | No (must traverse sequentially) |
| Consistency with inserts | Items shift (duplicates/gaps) | Stable (cursor is immutable anchor) |
| Implementation complexity | Simple | Moderate |
| Best for | Admin panels, small datasets | Feeds, timelines, mobile apps |

**Cursor-based pagination** uses an opaque token (usually an encoded primary key or timestamp) as the starting point. The database seeks directly to that position in the index, avoiding the scan-and-discard problem entirely.

!!! example "LinkedIn at Scale"
    LinkedIn moved their feed API from offset to cursor-based pagination. At the depth their power users reached (thousands of items deep), offset pagination queries were taking 2-3 seconds. After switching to cursor-based pagination, the same depth responded in under 10ms -- a 200x improvement that held constant regardless of depth.

### Batch Processing and Request Coalescing

Making individual requests for each item creates N round trips. Batching groups multiple operations into a single round trip.

```
N+1 Query Problem (common ORM trap)
========================================
SELECT * FROM orders WHERE user_id = 42;        -- 1 query
SELECT * FROM products WHERE id = 101;           -- N queries
SELECT * FROM products WHERE id = 102;           --   (one per
SELECT * FROM products WHERE id = 103;           --    order item)
...
--> 20 orders = 21 queries = 21 round trips

Batched:
SELECT * FROM orders WHERE user_id = 42;
SELECT * FROM products WHERE id IN (101, 102, 103, ...);
--> Always 2 queries, regardless of order count
```

**Request coalescing** goes further: when multiple clients request the same resource within a short window, only one actual fetch occurs, and all clients receive the result. This is how CDNs handle cache stampedes -- 1,000 simultaneous requests for the same expired cache entry trigger one origin fetch, not 1,000.

### Prefetching and Preloading

Instead of waiting for the user to request data, predict what they will need and fetch it before they ask.

```
Without Prefetching:
========================================
User opens inbox --> Load message list (200ms)
User clicks message --> Load message body (150ms)
                        Load attachments list (100ms)
Total perceived latency: 200ms + 150ms + 100ms = 450ms

With Prefetching:
========================================
User opens inbox --> Load message list (200ms)
                     Start prefetching top 5 message bodies (background)
User clicks message --> Already in memory (0ms)
                        Load attachments list (100ms)
Total perceived latency: 200ms + 0ms + 100ms = 300ms  (33% faster)
```

**Where prefetching works well:** Navigation that is predictable (next page of a paginated list), related data that users almost always need (user profile when loading their posts), and data that is expensive to compute but cheap to store temporarily.

**Where prefetching wastes resources:** When user behavior is unpredictable, when prefetched data is large, or when the data changes frequently enough that prefetched copies become stale.

### Async Processing (Moving Work Off the Hot Path)

The fastest work is work you do not do during the request. Any operation that the user does not need to see the result of immediately can be moved to a background process.

```
Synchronous (everything on the hot path)
========================================
POST /signup
  |-- Validate input              5ms
  |-- Insert user record          20ms
  |-- Send welcome email          800ms  <-- Why is this here?
  |-- Generate recommendations    500ms  <-- User hasn't even logged in yet
  |-- Log analytics event         50ms
  |-- Return response
Total: 1,375ms

Async (only essential work on hot path)
========================================
POST /signup
  |-- Validate input              5ms
  |-- Insert user record          20ms
  |-- Enqueue: send welcome email  2ms  --> Background worker handles it
  |-- Enqueue: generate recs       2ms  --> Background worker handles it
  |-- Enqueue: analytics           2ms  --> Background worker handles it
  |-- Return response
Total: 31ms  (44x faster)
```

!!! tip "The Hot Path Rule"
    Ask for every operation: "Does the user need this result to see a meaningful response?" If no, move it off the hot path. Email sending, analytics, notification fanout, image processing, search index updates -- none of these need to happen before returning a response.

Common async patterns:

- **Message queues** (Kafka, SQS, RabbitMQ) for durable task processing
- **Event-driven processing** for fan-out (one event triggers multiple downstream actions)
- **Eventual consistency** for non-critical data synchronization

---

## Caching Layers

Caching is covered in depth in the [Caching Strategies](../data/caching/strategies.md) page. Here, the focus is on how multiple cache layers compose into a latency-reduction system and what each layer contributes.

```
Multi-Tier Cache Architecture
================================================================

Request flow (left to right, stops at first hit):

  Client    Browser     CDN Edge    App Cache    DB Cache     Database
  Request   Cache       Cache       (Redis)      (Buffer)     (Disk)
    |         |           |           |            |             |
    v         v           v           v            v             v
    +------+------+-------+------+--------+------+--------+--------+
    |      | 0ms  | 1-5ms | 5ms  | 1-2ms  | 10ms | 1-5ms  | 50ms+  |
    | User | hit? | hit?  |      | hit?   |      | hit?   |        |
    +------+------+-------+------+--------+------+--------+--------+
              |       |              |               |          |
           95% of   80% of        90% of          80% of    Only
           static   dynamic       app data        working   ~1-2% of
           assets   content       requests        set       reads
           never    never         never hit                 reach
           leave    reach         the DB                    disk
           browser  origin

Effective hit rates compound:
  CDN hit rate:     80%  --> 80% of requests stop here
  App cache rate:   90%  --> 90% of remaining 20% stop here
  DB cache rate:    80%  --> 80% of remaining 2% stop here
  Combined:         99.6% of reads never touch disk
```

| Cache Layer | Latency | What It Caches | Typical TTL |
|-------------|:-------:|----------------|:-----------:|
| Browser/client | 0ms | Static assets, API responses | Minutes to days |
| CDN edge | 1-5ms | Static + dynamic content | Seconds to hours |
| Application (Redis/Memcached) | 1-2ms | Session data, computed results, hot queries | Seconds to minutes |
| Database buffer pool | Sub-ms | Recently accessed pages, index blocks | Automatic (LRU) |

!!! info "The Cold Start Problem"
    A freshly deployed cache has zero hit rate. All requests fall through to the database, potentially overwhelming it. Mitigation strategies: cache warming (preload popular keys on deploy), gradual traffic shifting (route 10% of traffic to new instances, then ramp), and stale-while-revalidate (serve slightly stale data while refreshing in background).

---

## Real-World Case Studies

### Amazon: 100ms = 1% Revenue

Amazon's research found that every **100ms of added latency costs 1% of sales**. For a company generating $500B+ annually, that is $5B per 100ms. This drove Amazon to build one of the world's most sophisticated latency optimization stacks:

- **DynamoDB** was built specifically because existing databases could not meet single-digit-millisecond latency SLOs at Amazon's scale
- **CloudFront** edge locations in 400+ cities reduce last-mile latency
- **Predictive prefetching** begins loading product pages before the user clicks, based on hover behavior

### Google: 500ms Slower = 20% Fewer Searches

Google experimented by artificially adding 500ms of latency to search results. The result: **20% fewer searches** and users began their next search session later. Even after removing the delay, affected users continued searching less for weeks -- latency had permanently changed their behavior.

This led to Google's obsession with sub-200ms search results:

- **Bigtable and Spanner** for distributed low-latency storage
- **QUIC protocol** (now HTTP/3) developed specifically to reduce connection latency
- **Precomputed search indexes** served from memory at the edge

### Discord: MongoDB to Cassandra for Messages

Discord initially stored messages in MongoDB. As they grew past 100M messages, read latency became unacceptable: queries for channel history took **hundreds of milliseconds** and degraded unpredictably during peak hours.

They migrated to Cassandra, choosing it specifically for:

- **Predictable latency** -- Cassandra's LSM-tree storage provides consistent read/write times regardless of data size
- **Time-series optimization** -- Messages are naturally time-ordered; Cassandra's partition key design maps perfectly to channel_id + time bucket
- **Linear scalability** -- Adding nodes reduces per-node load proportionally

Post-migration, P99 read latency dropped from **hundreds of milliseconds to under 10ms**. Discord later moved to ScyllaDB (a C++ Cassandra rewrite) for further latency improvements.

### LinkedIn: Cursor-Based Pagination at Scale

LinkedIn's feed serves 900M+ members. Their original offset-based pagination degraded as users scrolled deeper. The database had to skip millions of rows for power users, causing:

- P99 latency exceeding 3 seconds at depth > 1,000
- Database CPU spikes during peak hours
- Inconsistent results (posts appearing twice or being skipped as new content was inserted)

After migrating to cursor-based pagination:

- Latency became constant regardless of depth (under 10ms)
- Database load dropped 40% (no more scan-and-discard)
- Consistency improved (cursors are immutable position markers)

---

## Measuring and Monitoring Latency

Optimization is meaningless without measurement. The three pillars of latency measurement are percentiles, distributed tracing, and latency budgets.

### Percentile Metrics

!!! note "See Also"
    Percentile fundamentals are covered in [Performance Fundamentals](fundamentals.md). This section focuses on how to use percentiles for optimization decisions.

```
Which percentile to optimize for?
========================================

P50  (median):  What most users experience
                Optimize this for general user satisfaction
                Target: < 100ms for API, < 1s for page load

P95:            What your heaviest/most valuable users experience
                Optimize this to retain power users
                Target: < 500ms for API, < 3s for page load

P99:            Reveals systemic issues (GC pauses, lock contention,
                connection pool exhaustion, cache stampedes)
                Optimize this to prevent cascading failures
                Target: < 1s for API, < 5s for page load

P99.9:          Extreme tail -- usually infrastructure problems
                (disk I/O stalls, network partitions, DNS timeouts)
                Monitor but do not over-optimize
```

**The gap between P50 and P99 tells you about your system's consistency.** A small gap (P50=40ms, P99=80ms) indicates a predictable system. A large gap (P50=40ms, P99=2000ms) indicates intermittent problems -- usually garbage collection, lock contention, or resource exhaustion that strikes a small percentage of requests.

### Distributed Tracing

For a request that traverses multiple services, you need to know which service contributed how much latency. Distributed tracing (Jaeger, Zipkin, Datadog APT) propagates a trace ID through every service call, creating a flame graph of the entire request.

```
Trace: GET /api/feed (total: 340ms)
========================================
|-- API Gateway                          [  15ms  ]
|   |-- Auth Service                     [   8ms  ]
|   |-- Feed Service                     [ 290ms  ]
|       |-- User Cache Lookup (Redis)    [  2ms   ]
|       |-- Post Query (PostgreSQL)      [ 180ms  ] <-- BOTTLENECK
|       |-- Enrichment Service           [  95ms  ]
|           |-- Media Service            [  40ms  ]
|           |-- Like Count (Redis)       [   3ms  ]
|           |-- Author Profile (cache)   [   1ms  ]
|-- Response serialization               [   5ms  ]

The trace immediately reveals:
  1. Post Query is 53% of total latency --> optimize this first
  2. Enrichment calls are sequential --> parallelize them
  3. Auth and gateway add minimal overhead --> ignore for now
```

### Latency Budget Tracking

Set an explicit latency budget for each API endpoint and track actual spend against budget in dashboards.

```
Endpoint: GET /api/product/:id
Budget: 200ms (P99)

Current spend:
  Component          Budget    Actual    Status
  ─────────────────────────────────────────────
  Network/TLS          30ms     25ms     OK
  Auth middleware       10ms      8ms     OK
  Product query         50ms    120ms     OVER BUDGET
  Inventory check       30ms     15ms     OK
  Price calculation     20ms     18ms     OK
  Recommendation        40ms     35ms     OK
  Serialization         20ms     12ms     OK
  ─────────────────────────────────────────────
  Total               200ms    233ms     OVER by 33ms

Action: Product query is 70ms over budget.
        EXPLAIN ANALYZE reveals missing index on product_category.
```

---

## Decision Framework: Where to Optimize First

Not all optimizations yield equal returns. Amdahl's Law states that the maximum speedup from improving a component is limited by the fraction of time that component consumes. This creates a clear optimization priority order.

```
Amdahl's Law Applied to Latency
========================================

If database queries are 40% of total latency:
  Optimizing DB by 50% --> total improvement = 20%
  Optimizing DB by 90% --> total improvement = 36%

If app logic is 15% of total latency:
  Optimizing app by 50% --> total improvement = 7.5%
  Optimizing app by 90% --> total improvement = 13.5%

--> A 50% database improvement beats a 90% application improvement
```

### Optimization Priority Matrix

```
                        HIGH impact
                            |
    ┌───────────────────────┼───────────────────────┐
    |                       |                       |
    |  DB Connection Pool   |  Query Optimization   |
    |  Add Caching Layer    |  Read Replicas        |
    |  CDN for Static       |  Async Processing     |
    |                       |                       |
    |   QUICK WINS          |   STRATEGIC           |
    |   (do these first)    |   (plan and schedule)  |
LOW ────────────────────────┼──────────────────────── HIGH
effort                      |                       effort
    |   FILL-INS            |   LAST RESORT         |
    |   (when convenient)   |   (only if needed)    |
    |                       |                       |
    |  Compression tuning   |  HTTP/3 migration     |
    |  Keep-alive config    |  Database migration   |
    |  JSON serializer swap |  Service rewrite      |
    |                       |                       |
    └───────────────────────┼───────────────────────┘
                            |
                        LOW impact
```

### The Optimization Checklist

Work through this list in order. Stop when you meet your latency SLO.

| Step | Action | Expected Impact | Effort |
|:----:|--------|:---------------:|:------:|
| 1 | Profile with distributed tracing | Identifies bottleneck | 1 day |
| 2 | Add missing database indexes | 10-100x for affected queries | Hours |
| 3 | Enable connection pooling (DB + HTTP) | 2-8x connection overhead reduction | Hours |
| 4 | Add application-level caching (Redis) | 10-100x for cached reads | 1-2 days |
| 5 | Move async work off the hot path | 2-50x for affected endpoints | 1-3 days |
| 6 | Set up CDN for static + cacheable dynamic | 10-50x for edge-served content | 1 day |
| 7 | Switch to cursor-based pagination | Constant latency at any depth | 2-3 days |
| 8 | Add read replicas | 2-5x read throughput | 1-2 days |
| 9 | Materialize expensive queries | 100-1000x for dashboard queries | 1-2 days |
| 10 | Evaluate HTTP/2 or HTTP/3 | 5-30% for high-concurrency | 1-2 weeks |

!!! tip "The 80/20 Rule of Latency"
    Steps 1 through 5 typically resolve 80% of latency problems. Steps 6-10 are for teams that have already captured the easy wins and need to push further. If you find yourself reaching for database migrations or service rewrites, make absolutely sure you have exhausted the simpler options first.

---

## Key Takeaways

1. **Measure first** -- Distributed tracing and percentile metrics reveal where time actually goes, which is almost never where you think it goes.

2. **Database is usually the bottleneck** -- In most applications, 40%+ of latency comes from database queries. Connection pooling, indexes, and caching yield the highest returns.

3. **Network latency has a physics floor** -- You cannot make light travel faster, but you can reduce round trips (multiplexing, keep-alive), compress data, and move computation to the edge.

4. **Move work off the hot path** -- The fastest operation is one you do not perform during the request. Async processing is one of the highest-leverage optimizations available.

5. **Caching layers compound** -- A 80% CDN hit rate combined with a 90% application cache hit rate means 98% of requests never touch the database.

6. **Optimize in order** -- Follow Amdahl's Law. A modest improvement to the largest bottleneck beats a dramatic improvement to a small component.
