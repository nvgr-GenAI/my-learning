# Database Scaling Patterns

Every successful application eventually outgrows its database. What starts as a single PostgreSQL instance serving a few hundred users becomes a bottleneck when you reach millions of requests per day. The good news: you don't need to redesign everything at once. Database scaling follows a predictable progression — each phase builds on the last, and you should only move to the next when you've exhausted the current one.

## The Scaling Progression

```
Phase 1: Optimize          Phase 2: Scale Reads       Phase 3: Scale Writes      Phase 4: Distribute
─────────────────          ──────────────────         ───────────────────         ──────────────────
Single DB, tuned           Read replicas + cache      Partition + separate        Shard + federate

┌──────────┐               ┌──────────┐               ┌────────┐ ┌────────┐      ┌────┐┌────┐┌────┐
│    DB    │               │ Primary  │               │ Writes │ │ Reads  │      │ S1 ││ S2 ││ S3 │
│ (tuned)  │               │    │     │               │   DB   │ │   DB   │      └────┘└────┘└────┘
└──────────┘               │ ┌──┴──┐  │               └────────┘ └────────┘        Sharded cluster
                           │ R1  R2│  │
                           └───────┘
Handles:                   Handles:                   Handles:                   Handles:
~1K-10K QPS               ~50K-100K QPS              ~100K-500K QPS             ~1M+ QPS
```

| Phase | When to Use | Techniques | Complexity |
|-------|-------------|------------|------------|
| **Phase 1** | Slow queries, high CPU on single DB | Indexing, query tuning, connection pooling, vertical scaling | Low |
| **Phase 2** | Read latency growing, reads > 80% of traffic | Read replicas, application-level caching | Medium |
| **Phase 3** | Write throughput hitting single-node limits | Vertical partitioning, CQRS | Medium-High |
| **Phase 4** | Single domain too large for one node | Sharding, event sourcing, federation | High |

Most applications live comfortably in Phase 1 or 2. Stack Overflow runs on a handful of SQL Server instances. Many successful SaaS products never need sharding.

---

=== "Phase 1: Optimize"

    ## Optimize What You Have

    Before adding infrastructure, make your existing database work harder. Most databases operate well below their theoretical limits because of unoptimized queries, missing indexes, or misconfigured settings.

    ### Connection Pooling

    Every database connection consumes memory (typically 5-10MB in PostgreSQL). If your application opens a new connection per request, you'll hit the connection limit long before you hit CPU or I/O limits.

    A connection pool maintains a set of open connections and reuses them across requests. Instead of 500 application threads each holding a connection, a pool of 20-50 connections serves them all — because most requests hold a connection for only a few milliseconds.

    **PgBouncer** is the standard solution for PostgreSQL. It sits between your application and database, managing connections transparently. **Instagram** runs their entire PostgreSQL fleet behind PgBouncer, handling thousands of transactions per second with modest connection pools.

    ### Query Optimization

    The single highest-impact optimization in most applications:

    | Optimization | Impact | Example |
    |-------------|--------|---------|
    | **Add indexes** on WHERE/JOIN columns | Queries go from seconds → milliseconds | B-tree index on `user_id` |
    | **Eliminate N+1 queries** | Reduces query count by 10-100x | Use JOIN or batch query instead of per-row lookups |
    | **Use EXPLAIN ANALYZE** | Identifies exact bottlenecks | Find sequential scans, bad joins, unindexed sorts |

    See [Indexing](indexing.md) for a deep dive on index types and strategies.

    ### Vertical Scaling (Scale Up)

    The simplest scaling strategy: buy a bigger machine. More CPU cores, more RAM, faster SSDs.

    This is not a joke strategy — it's often the most cost-effective one. A single machine with 64 cores, 512GB RAM, and NVMe storage can handle remarkable workloads. **GitHub** ran on a single MySQL primary for years. **Stack Overflow** serves millions of users from a small number of SQL Server instances.

    The ceiling is real, though. The largest cloud instances top out around 128 cores and 4TB RAM. And vertical scaling provides zero redundancy — if that machine fails, everything stops.

=== "Phase 2: Scale Reads"

    ## Scale Reads

    Most applications are read-heavy. Social media feeds, product catalogs, user profiles — reads outnumber writes 10:1 or more. Phase 2 targets this imbalance.

    ### Read Replicas

    Create copies of your primary database that handle read queries. Writes still go to the primary, which replicates changes to the replicas asynchronously.

    ```
    Write path:                          Read path:

    App ──→ Primary DB                   App ──→ Load Balancer
               │                                   │
               │ async replication          ┌──────┼──────┐
               ├────────────────→ Replica 1 │      │      │
               ├────────────────→ Replica 2 R1     R2     R3
               └────────────────→ Replica 3
    ```

    **The trade-off is consistency.** Asynchronous replication means replicas are slightly behind the primary (typically milliseconds, but can be seconds under load). A user who just updated their profile might see stale data if the next read hits a replica that hasn't caught up.

    | Consistency Strategy | How It Works | Trade-off |
    |---------------------|-------------|-----------|
    | **Read-your-own-writes** | Route reads to primary for a few seconds after a write | Adds load to primary |
    | **Synchronous replication** | Primary waits for replica acknowledgment | Reduces write throughput |
    | **Causal consistency** | Track write timestamps, route to up-to-date replica | Application complexity |

    **Instagram** uses PostgreSQL with read replicas across multiple data centers. Read-heavy endpoints (feed, explore, search) go to replicas, while write-heavy endpoints (post creation, likes) go to the primary.

    ### Application-Level Caching

    Place a cache (typically Redis or Memcached) between your application and database. Check the cache first; only query the database on a cache miss.

    ```
    App ──→ Cache (Redis)
             │
             ├─ HIT → return cached data (sub-ms)
             │
             └─ MISS → query DB → store in cache → return
    ```

    A well-tuned cache with a 95% hit rate means your database handles only 5% of read traffic. **Facebook's** TAO cache serves billions of reads per day, with the underlying MySQL databases seeing a fraction of the total load.

    | Cache Pattern | Behavior | Best For |
    |--------------|----------|----------|
    | **Cache-aside** | App checks cache, misses go to DB, results stored | General purpose, simple |
    | **Write-through** | Every write updates both DB and cache | Strong consistency needed |
    | **TTL-based expiry** | Cache entries expire after a time window | Tolerance for brief staleness |

=== "Phase 3: Scale Writes"

    ## Scale Writes

    When your write volume exceeds what a single primary can handle, you need to split the write workload across multiple databases.

    ### Vertical Partitioning (Functional Splits)

    Split your database by domain — different services get different databases. This is often the first step toward microservices.

    ```
    Before:                              After:
    ┌─────────────────────┐              ┌──────────┐  ┌──────────┐  ┌──────────┐
    │     Monolith DB     │              │ User DB  │  │ Order DB │  │ Product  │
    │                     │              │          │  │          │  │    DB    │
    │ users, orders,      │     →        │ users    │  │ orders   │  │ products │
    │ products, analytics │              │ profiles │  │ payments │  │ catalog  │
    │                     │              │ sessions │  │ shipping │  │ reviews  │
    └─────────────────────┘              └──────────┘  └──────────┘  └──────────┘
    ```

    Each database handles only its domain's write load. They scale independently.

    **The trade-off:** Cross-domain queries become expensive — "show all orders with customer names" now requires querying two databases and joining in application code. Transactions spanning multiple databases require distributed coordination (two-phase commit or saga patterns).

    **Shopify** uses functional partitioning extensively — separate databases for shops, orders, products, and inventory.

    ### CQRS (Command Query Responsibility Segregation)

    Use different data models for reads and writes:

    ```
                        ┌─────────────────┐
    Write request ────→ │  Write Model    │ ──→ Event published
                        │ (normalized DB) │
                        └─────────────────┘
                                  │
                             event stream
                                  │
                                  ▼
                        ┌─────────────────┐
    Read request ─────→ │   Read Model    │
                        │ (denormalized)  │
                        └─────────────────┘
    ```

    | Aspect | Write Model | Read Model |
    |--------|------------|------------|
    | **Schema** | Normalized (3NF) | Denormalized (pre-joined) |
    | **Optimized for** | Consistency, validation | Query speed, read patterns |
    | **Example** | Separate orders, items, addresses tables | Single document with all order details |

    **When CQRS makes sense:** Systems where read and write patterns are fundamentally different — e-commerce writes orders as complex transactions but reads them as simple documents.

    **When it doesn't:** Simple CRUD applications where reads and writes use the same data shape. The complexity of maintaining two synchronized models isn't justified.

=== "Phase 4: Distribute"

    ## Distribute

    When functional splits aren't enough — when a single domain (like users or messages) generates more writes than one database can handle — you split data within a domain across multiple databases.

    ### Sharding (Horizontal Partitioning)

    Split rows across multiple database instances based on a shard key. See [Sharding](sharding.md) for a deep dive.

    | Strategy | How It Works | Good For | Watch Out For |
    |----------|-------------|----------|---------------|
    | **Hash-based** | `shard = hash(user_id) % N` | Even distribution | Adding shards requires rehashing |
    | **Range-based** | Users 1-1M → shard 1, 1M-2M → shard 2 | Range queries on shard key | Hot spots if ranges have uneven load |
    | **Directory-based** | Lookup table maps keys to shards | Flexible placement | Lookup service becomes a bottleneck |

    **Discord** shards by guild (server) ID — all messages for a guild live on the same shard, keeping the most common query local. **Slack** shards by workspace for the same reason.

    ### Event Sourcing

    Instead of storing current state, store the sequence of events that produced it:

    ```
    Traditional:                    Event Sourced:
    ┌──────────────────┐            ┌──────────────────────────────────┐
    │ account: 12345   │            │ 1. AccountOpened(12345, $0)      │
    │ balance: $750    │            │ 2. Deposited(12345, $1000)       │
    │ status: active   │            │ 3. Withdrawn(12345, $200)        │
    └──────────────────┘            │ 4. Deposited(12345, $50)         │
                                    │ 5. Withdrawn(12345, $100)        │
      Current state only            │    → Current balance: $750       │
                                    └──────────────────────────────────┘
                                      Complete history, replayable
    ```

    Events are immutable and append-only — perfect for distributed systems. Current state is derived by replaying events or maintained as a materialized view.

    **When it shines:** Financial systems (audit trail), collaborative editing (replay and merge), systems where "how we got here" matters as much as "where we are."

    ### Database Federation

    Combine multiple specialized databases behind an application layer:

    ```
    Application Layer
           │
      ┌────┼────────────────┐
      │    │                 │
      ▼    ▼                 ▼
    PostgreSQL    Redis     Elasticsearch
    (orders)     (sessions)   (search)
    ```

    Different data has different access patterns and should live in the database best suited for it. The application layer handles cross-database coordination.

---

## Key Takeaways

1. **Optimize first.** Indexing, query optimization, and connection pooling solve most performance problems without adding infrastructure.

2. **Read replicas + caching handle the majority of scaling needs.** If your workload is 90% reads (most are), Phase 2 gives you 10x headroom.

3. **Vertical partitioning is simpler than sharding.** Split by domain before you split within a domain.

4. **Each phase has a cost.** Read replicas introduce eventual consistency. Sharding complicates queries. CQRS doubles your data models. Only pay the cost when you need the benefit.

5. **Real companies delay complexity.** Instagram scaled to millions of users on PostgreSQL with replicas and caching before adding more complex patterns. Start simple.

6. **Monitor before you scale.** Without metrics (query latency, QPS, replication lag, cache hit rate), you're guessing where the bottleneck is.

---

## Related Topics

- **[Indexing](indexing.md)** — speed up queries before adding infrastructure
- **[Replication](replication.md)** — how read replicas and data copying work
- **[Sharding](sharding.md)** — deep dive into horizontal partitioning
- **[Database Types](database-types.md)** — choosing the right database for your workload
