# Database Sharding

You've added read replicas. You've added caching. You've optimized every query, added every useful index. And still, your single database is running at 90% capacity, backups take 6 hours, and your users are growing 30% per quarter. You've hit the wall — a single machine simply cannot hold or process all your data anymore.

This is where sharding comes in: **split your data across multiple database servers**, so each one handles only a fraction of the total.

---

## Before You Shard

Sharding is powerful but it's a one-way door — once you shard, your application becomes permanently more complex. Before committing, exhaust these simpler options:

**Vertical scaling** — get a bigger server. If you're at 70% capacity and growing linearly, a 2x hardware upgrade buys you significant runway. It's boring but effective.

**Read replicas** — if your workload is 80%+ reads (most web apps), replicas can absorb the read traffic while the primary handles writes. This alone can 5-10x your throughput.

**Caching** — a Redis layer in front of your database can serve hot data at 100,000+ QPS with sub-millisecond latency. If your read patterns are repetitive, caching may eliminate the need to shard entirely.

**Query optimization** — missing indexes, unoptimized queries, or full table scans can make a database appear overwhelmed when the real problem is inefficiency.

**Rule of thumb:** Only shard when your database is truly beyond what a single machine can handle — typically past 10,000+ write QPS, over 1TB of data, or when backups take so long they're no longer practical.

---

## What is Sharding?

Sharding is splitting a single large database into multiple smaller databases, each called a **shard**. Each shard holds a subset of the data and runs on its own server. Together, the shards form the complete dataset.

Think of a library that's outgrown its building. Instead of finding one enormous building, you open four branch locations — each branch holds books by different authors. A-F goes to Branch 1, G-L to Branch 2, and so on. The entire collection is available, just distributed across locations.

```
Before Sharding (single database):
┌─────────────────────────────────────┐
│         Single Database             │
│  1 Million Users                    │  ← Overwhelmed
│  10 Million Orders                  │  ← Slow queries
│  All on one server                  │  ← Single point of failure
└─────────────────────────────────────┘

After Sharding (distributed):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Shard 1  │  │ Shard 2  │  │ Shard 3  │  │ Shard 4  │
│ 250K usr │  │ 250K usr │  │ 250K usr │  │ 250K usr │
│ 2.5M ord │  │ 2.5M ord │  │ 2.5M ord │  │ 2.5M ord │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
  Each shard handles 25% of the load
```

The benefits are substantial: each shard is smaller and faster, queries touch less data, you can add shards as you grow, and if one shard goes down, 75% of your users are unaffected. But the complexity is real — your application must know which shard to talk to, cross-shard queries become difficult, and operations like backups and schema changes now happen across multiple databases.

---

## The Shard Key: Your Most Important Decision

Before you can split data, you need to decide **how to split it**. This decision — choosing the **shard key** — is the single most consequential choice in your sharding design. It's extremely difficult to change later because it determines how data is physically distributed.

The shard key is the column (or columns) used to determine which shard a row belongs to. When a query comes in, the routing layer looks at the shard key value to know which shard to contact.

### What Makes a Good Shard Key?

**High cardinality.** The key must have many distinct values. `user_id` with millions of unique values distributes well. `status` with only "active" and "inactive" puts 99% of data on one shard.

**Even distribution.** The values should be spread evenly across the keyspace. `user_id` (random UUIDs) distributes evenly. `country` does not — the US shard would be 100x larger than Iceland's.

**Present in most queries.** If 90% of your queries filter by `user_id`, that's your shard key. If queries rarely include it, every query becomes a scatter-gather across all shards.

**Immutable.** Moving a row between shards is expensive. A shard key that changes (like `location`) means constant reshuffling. `user_id` never changes — ideal.

### Shard Keys by Application

The right key depends on your domain:

- **Social media** (Instagram, Twitter): `user_id` — all of a user's posts, likes, and followers live on one shard
- **E-commerce** (Amazon, Shopify): `customer_id` — orders, cart, and preferences stay together
- **Multi-tenant SaaS** (Salesforce, Slack): `tenant_id` — natural isolation between customers
- **Messaging** (Discord): `channel_id` or `guild_id` — messages within a channel always on the same shard
- **Time-series / IoT**: `device_id` or `(device_id, time_bucket)` — device data stays together
- **Analytics / Logs**: `timestamp_bucket` (by day or week) — efficient time-range queries

### Bad Shard Keys

`status` (active/inactive) — only 2 values, massive imbalance. `created_date` — all new writes go to the newest shard, creating a permanent hotspot. `country` — wildly uneven population distribution. `is_premium` (boolean) — same problem as status. Any key with low cardinality is a recipe for hotspots.

---

=== "Sharding Strategies"

    Now that you understand shard keys, let's look at the four strategies for mapping a key to a shard. Each one has different strengths, and the right choice depends on your query patterns.

    | Strategy | Routing | Range Queries | Distribution | Best For |
    |---|---|---|---|---|
    | Hash-Based | `hash(key) % N` | Not supported | Excellent (even) | Key-value lookups, social media |
    | Range-Based | Key range boundaries | Efficient | Risk of hotspots | Time-series, logs, analytics |
    | Directory-Based | Lookup service | Depends on mapping | Flexible (arbitrary) | Multi-tenant SaaS |
    | Geographic | User region | Within-region only | Uneven (population) | Global apps, compliance (GDPR) |

    ### Hash-Based Sharding

    The most common strategy. Run the shard key through a hash function, then modulo by the number of shards: `shard = hash(key) % N`.

    ```
    user_12345  → hash() % 4 = 1  → Shard 1
    user_67890  → hash() % 4 = 2  → Shard 2
    user_11111  → hash() % 4 = 3  → Shard 3
    user_99999  → hash() % 4 = 0  → Shard 0
    ```

    The hash function produces an essentially random distribution, so data spreads evenly across shards regardless of the input pattern. Instagram uses hash-based sharding with `user_id` across 100+ shards to serve 1 billion+ users.

    **Strength:** Excellent even distribution. Simple O(1) routing. Scales linearly — double the shards, halve the load per shard.

    **Weakness:** Range queries are impossible. If you ask "find all users created between January and March," the hash has scattered those users randomly across all shards. You must query every shard (scatter-gather), which is slow. Also, adding or removing shards with naive modulo (`% N`) redistributes nearly all data — this is where consistent hashing helps (more on that later).

    **Best for:** Key-value lookups, social media (user profiles), sessions, gaming — any workload dominated by "get me everything for this specific entity."

    ### Range-Based Sharding

    Divide the key space into contiguous ranges and assign each range to a shard.

    ```
    Shard 1: user_id 1 – 1,000,000
    Shard 2: user_id 1,000,001 – 2,000,000
    Shard 3: user_id 2,000,001 – 3,000,000
    ```

    The key advantage over hash-based: **range queries are efficient**. "Get all orders from the last month" only needs to hit the shard(s) covering that time range, not all shards. This makes range-based sharding the natural choice for time-series data.

    **The hotspot problem:** If the shard key is sequential (like an auto-incrementing ID or a timestamp), all new writes go to the newest shard. The current-month shard gets hammered while older shards sit idle.

    ```
    Shard 1 (Jan-Mar): 10% load
    Shard 2 (Apr-Jun): 15% load
    Shard 3 (Jul-Sep): 25% load
    Shard 4 (Oct-Dec): 50% load  ← HOTSPOT!
    ```

    Mitigations include composite keys (e.g., `(region, timestamp)` to spread writes across regions), time-bucket sharding (create a new shard per month, archive old ones), and pre-splitting future ranges.

    **Best for:** Time-series data, logs, metrics, analytics — workloads where you frequently query date/time ranges. Cassandra, InfluxDB, and AWS CloudWatch all use range-based approaches.

    ### Directory-Based Sharding

    A separate lookup service maintains an explicit mapping of `key → shard`. Instead of computing the shard from the key, you ask the directory.

    ```
    Application Request: "Where is tenant_acme?"
        ↓
    ┌──────────────────────────────────────┐
    │  Directory Service (Redis/etcd)      │
    │  tenant_acme  → shard_dedicated_1    │
    │  tenant_xyz   → shard_shared_3       │
    │  tenant_big   → shard_premium_1      │
    └──────────────────────────────────────┘
        ↓
    Route to shard_dedicated_1
    ```

    The power of directory-based sharding is **arbitrary routing logic**. You can put enterprise customers on dedicated shards, route EU customers to GDPR-compliant infrastructure, or move a tenant between shards without resharding — just update the directory entry.

    **Weakness:** The directory service is additional infrastructure (and a potential single point of failure). Every request has an extra lookup hop. The directory itself must be highly available (typically replicated Redis or etcd/Consul).

    **Best for:** Multi-tenant SaaS where different tenants have different requirements. Salesforce, Slack, Shopify, and Atlassian all use directory-based routing to isolate large or enterprise customers on dedicated infrastructure.

    ### Geographic Sharding

    Route data to shards based on the user's geographic region — US data to US shards, EU data to EU shards, APAC data to APAC shards.

    ```
    US (Virginia)              EU (Frankfurt)           APAC (Tokyo)
    ┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
    │ US users & data │       │ EU users & data │      │ APAC users/data │
    │ Latency: ~20ms  │       │ Latency: ~20ms  │      │ Latency: ~20ms  │
    │ CCPA compliant  │       │ GDPR compliant  │      │ PDPA compliant  │
    └─────────────────┘       └─────────────────┘      └─────────────────┘
             Cross-region latency: 200-300ms (avoided!)
    ```

    The primary motivations are **latency** (users read/write to nearby shards) and **compliance** (GDPR requires EU citizen data to stay in the EU, and similar regulations exist worldwide).

    **Weakness:** Uneven distribution — if 60% of your users are in the US, the US shard is 3x larger. Cross-region queries are slow (200-300ms round trip). And "traveling users" create a dilemma: does a US user visiting Tokyo write to their home shard (slow) or the local shard (complex sync)?

    **Best for:** Global applications with regulatory requirements. Netflix (230M subscribers in 190+ countries), Stripe (PCI-DSS and GDPR across 40+ countries), and Spotify all use geographic sharding.

=== "Hard Parts"

    Sharding creates three problems that don't exist with a single database: cross-shard queries, cross-shard transactions, and resharding.

    | Problem | Impact | Primary Solutions | When to Worry |
    |---|---|---|---|
    | Cross-Shard Queries | Slow scatter-gather across all shards | Denormalization, app-level joins | When >10% of queries span shards |
    | Cross-Shard Transactions | No native ACID across shards | 2PC, Saga pattern, TCC | Financial systems, inventory |
    | Resharding | Massive data migration | Consistent hashing, split-based, live migration | When shards become unbalanced |

    ### Cross-Shard Queries

    Once data is spread across shards, queries that need data from multiple shards become difficult. If users are sharded by `user_id`, a query like "find all users named John" has no idea which shard the Johns are on — it must ask every shard.

    There are three strategies, in order of preference:

    **Denormalization** — duplicate data so common queries hit a single shard. Instead of joining users and orders across shards, store an `order_summary` directly on the user's shard. Twitter does this with `follower_count` — it's denormalized and updated asynchronously rather than computed with a cross-shard join. Fast reads, eventual consistency on writes.

    **Application-level joins** — query multiple shards independently and combine results in your application code. This works for occasional cross-shard needs (e.g., loading a user from Shard 1 and their orders from Shard 3, then combining them in the app layer).

    **Scatter-gather** — send the query to ALL shards in parallel, gather results, merge. This is a last resort — it's slow, doesn't scale (adding shards makes it slower), and should represent less than 1% of your queries. Use it for admin dashboards and reporting, not user-facing features.

    **Design principle:** 90%+ of your queries should hit a single shard. If they don't, your shard key is wrong.

    ### Cross-Shard Transactions

    A single database gives you ACID transactions for free. Once data spans multiple shards, atomic operations across them require coordination protocols.

    **Two-Phase Commit (2PC)** — a coordinator asks all involved shards "can you commit?" (prepare phase). If all say yes, the coordinator tells them to commit. If any says no, all abort. This guarantees atomicity but is slow (two round trips + lock holding time) and fragile (if the coordinator crashes while shards are prepared, they're stuck holding locks until it recovers).

    ```
    Transfer $100: Account A (Shard 1) → Account B (Shard 3)

    PREPARE:  Coordinator → Shard 1: "Can you debit $100?"  → "YES"
              Coordinator → Shard 3: "Can you credit $100?" → "YES"
    COMMIT:   Coordinator → Shard 1: "COMMIT" → Done
              Coordinator → Shard 3: "COMMIT" → Done

    If either says NO → ABORT all
    ```

    **Saga pattern** — break the distributed transaction into a sequence of local transactions, each with a compensating action if something fails later. Instead of one atomic operation, you have: debit Account A → credit Account B → record transfer. If crediting B fails, you run compensation: refund Account A.

    Sagas provide eventual consistency (not atomicity) and work well for multi-step business processes. Uber uses sagas for trip lifecycle, Amazon for order processing, Airbnb for booking flows.

    **TCC (Try-Confirm-Cancel)** — reserve resources first ("try"), then confirm or cancel. Useful for inventory and booking: reserve 2 concert tickets, then either confirm the purchase or release them. Booking.com and airline reservation systems use this pattern.

    **When to use what:** 2PC for financial systems that demand atomicity. Sagas for e-commerce and multi-service workflows that can tolerate eventual consistency. TCC for reservation/booking scenarios. And for analytics aggregation across shards — just accept eventual consistency.

    ### Resharding: When You Outgrow Your Topology

    Eventually, you'll need to change your shard layout — splitting overloaded shards, adding capacity, or restructuring. This is resharding, and with naive hash-based sharding (`hash % N`), changing N redistributes nearly all data.

    **Consistent hashing** solves this elegantly. Instead of `hash % N`, map both keys and shards onto a circular "ring." Each key goes to the next shard clockwise on the ring.

    ```
    Naive hashing: Add 1 shard (4→5) → ~80% of keys must move
    Consistent hashing: Add 1 shard (4→5) → only ~20% of keys move (1/N)
    ```

    To ensure even distribution, each physical shard gets 100-200 **virtual nodes** spread around the ring. DynamoDB, Cassandra, Riak, and Discord all use consistent hashing.

    **Split-based resharding** takes a different approach: split an overloaded shard into two by choosing a midpoint and migrating half the data. MongoDB does this automatically when chunks exceed 64MB. CockroachDB and Vitess also use split-based approaches.

    **Live migration** (dual-write approach) is how large companies do major resharding:

    1. Start writing to both old and new shard topology
    2. Backfill the new topology with historical data
    3. Verify consistency (checksums)
    4. Switch reads to the new topology
    5. Stop writing to the old topology

    Stripe and Shopify have used this approach for large-scale resharding with zero downtime. The process typically takes days to weeks.

=== "Operations"

    ### What Breaks

    Sharding introduces failure modes that don't exist with a single database.

    | Failure Mode | Impact | Mitigation |
    |---|---|---|
    | Single shard failure | ~25% of users affected (with 4 shards) | Per-shard replication + automatic failover |
    | Hotspot emergence | One shard overwhelmed, latency spikes | Monitor per-shard QPS; split or move hot entities |
    | Metadata service failure | No routing decisions possible | Cached routes + HA directory (replicated Redis/etcd) |
    | Cascading failure | Timeouts spread across shards via retries | Circuit breakers to stop traffic to failing shards |
    | Schema drift | Queries fail on shards with outdated schema | Automated DDL orchestration + central version tracking |

    **Single shard failure** — if one of four shards goes down, ~25% of users are affected. Each shard should have replicas (replication within the shard) with automatic failover. Tools like Orchestrator (MySQL), Patroni (PostgreSQL), and Redis Sentinel automate this.

    **Hotspot emergence** — one shard receives disproportionate traffic. Could be a celebrity user on that shard, a viral piece of content, or a time-based skew. Monitor per-shard QPS and latency; consider splitting the hot shard or moving the hot entity.

    **Metadata service failure** — if you use directory-based sharding, the directory going down means no routing decisions. Mitigate with cached routes (stale but functional) and a highly available directory (replicated Redis cluster, etcd, or Consul).

    **Cascading failure** — one overwhelmed shard starts timing out, causing the application to retry, which increases load on other shards, which start timing out too. **Circuit breakers** protect against this: if a shard's error rate exceeds a threshold, stop sending it traffic and return cached data or errors instead.

    **Schema drift** — with N shards, you have N independent databases. A schema migration applied to some shards but not others leads to queries failing on lagging shards. Use automated DDL orchestration (apply migrations shard-by-shard with verification) and track schema versions in a central metadata catalog.

    ### How Real Systems Shard

    #### Managed Services

    If you'd rather not build sharding infrastructure yourself, managed databases handle it transparently:

    | Service | Sharding Method | Key Feature | Caveat |
    |---|---|---|---|
    | DynamoDB (AWS) | Hash on partition key | Auto-splits and rebalances | Partition key immutable; hot partitions throttled |
    | Cloud Spanner (Google) | Auto range-based | Globally-consistent distributed SQL | 3-10x cost for multi-region |
    | Cosmos DB (Azure) | Hash on partition key | 5 tunable consistency levels | Cross-partition queries slower and costlier |
    | Vitess (open-source) | VSchema-defined | Adds sharding to MySQL | Used by YouTube, Slack, GitHub, Square |
    | CockroachDB | Auto range-based | Serializable distributed SQL | Cloud-agnostic (runs anywhere) |

    **DynamoDB** (AWS) — hash-based sharding on the partition key. Automatically splits partitions and rebalances. You choose the partition key; the rest is managed. Caveat: partition key can't be changed after table creation, and hot partitions can still be throttled.

    **Cloud Spanner** (Google) — automatically shards data by range and rebalances. Provides globally-consistent distributed SQL, which is remarkable — but comes at 3-10x the cost of single-region for multi-region deployments.

    **Cosmos DB** (Azure) — hash-based on the partition key with automatic scaling. Five tunable consistency levels from strong to eventual. Cross-partition queries are significantly slower and more expensive.

    **Vitess** (open-source, used by YouTube) — adds sharding to MySQL. You define VSchemas that specify how tables are sharded. Used by YouTube, Slack, GitHub, and Square.

    **CockroachDB** — automatically shards by range with automatic rebalancing. Provides serializable distributed SQL. Cloud-agnostic (runs anywhere).

    #### When to Self-Manage vs Use Managed

    For startups and small teams, managed services let you focus on your product instead of distributed database operations. For organizations already running MySQL or PostgreSQL, Vitess or Citus can add sharding to your existing database. Self-managed sharding only makes sense when you need extreme customization or are operating at a scale where the cost savings justify the engineering investment.

    ### Common Mistakes

    **Sharding too early.** Sharding is a one-way door that adds permanent complexity. Many teams shard when they should have just added read replicas, a caching layer, or optimized their queries. Make sure you've exhausted simpler options first.

    **Choosing the wrong shard key.** This is the most expensive mistake because it's the hardest to fix. A key with low cardinality creates hotspots. A key not present in most queries forces scatter-gather. Analyze your actual query patterns before choosing.

    **No plan for cross-shard operations.** If you shard by `user_id` but half your queries need data across users (leaderboards, search, recommendations), you'll be doing scatter-gather constantly. Design your data model for single-shard access from day one — denormalize aggressively.

    **No plan for resharding.** Starting with `hash % N` seems simple until you need to change N and discover that 80% of your data must move. Use consistent hashing from the beginning.

    **Ignoring per-shard monitoring.** Aggregate metrics hide shard-level problems. One shard could be at 95% capacity while the average shows 60%. Monitor QPS, latency, storage, and replication lag per shard.

---

## Key Takeaways

1. **Shard only when simpler solutions fail.** Read replicas, caching, query optimization, and vertical scaling should all be exhausted first.

2. **The shard key is everything.** It must have high cardinality, even distribution, and be present in most queries. It's extremely hard to change later — choose carefully.

3. **Design for single-shard queries.** 90%+ of your queries should hit exactly one shard. If they don't, rethink your shard key or denormalize your data model.

4. **Use consistent hashing from the start.** It makes adding and removing shards move only 1/N of the data instead of nearly all of it.

5. **Cross-shard operations are the tax you pay.** Denormalize to avoid cross-shard joins. Use sagas for distributed transactions. Accept that scatter-gather is slow and minimize it.

6. **Each shard is a database — treat it like one.** It needs replication, failover, backups, monitoring, and schema management. Multiply your operational burden by the number of shards.

7. **Consider managed services seriously.** DynamoDB, Spanner, and Cosmos DB handle resharding, replication, and failover automatically. The cost premium is often less than the engineering cost of doing it yourself.

---

## Related Topics

- **[Replication](replication.md)** — each shard needs its own replicas for high availability
- **[Indexing](indexing.md)** — indexes exist within each shard independently
- **[Database Types](database-types.md)** — NoSQL databases like Cassandra and DynamoDB have sharding built in
