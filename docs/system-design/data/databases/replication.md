# Database Replication

Your entire application — millions of users, years of data — runs on a single database server. One night, the hard drive fails. Or a data center loses power. Or traffic spikes 10x and that one server can't keep up. In any of these scenarios, your application is down, and there's nothing you can do about it.

This is the problem that replication solves: **keep copies of your data on multiple machines**, so that no single failure can take you down.

---

## Why Replicate?

A single database has three fundamental limitations:

**Availability.** If the server crashes, your application stops. There's no backup ready to take over. Every minute of downtime costs money and trust.

**Read throughput.** A single server can only handle so many queries per second. When your social media app has 100,000 users reading their feeds simultaneously, one database can't serve them all.

**Geographic latency.** If your database is in Virginia but your users are in Tokyo, every query crosses the Pacific Ocean and back — adding 150-200ms of latency that no amount of optimization can remove.

Replication addresses all three by maintaining copies (called **replicas**) of your database on separate machines, potentially in different data centers or even different continents.

---

## How Replication Works

At the most basic level, replication works through a **log**. Every change made to the primary database — every INSERT, UPDATE, DELETE — is recorded in a sequential log. Replicas read this log and apply the same changes to their own copy of the data.

```
Write Path:
  Client → Primary Database → Write data + append to replication log
                                                    │
                    ┌───────────────────────────────┘
                    ↓               ↓               ↓
               Replica 1       Replica 2       Replica 3
               (applies log)   (applies log)   (applies log)

Read Path:
  Client → Load Balancer → any replica (or primary)
```

This log-based approach means replicas don't need to understand SQL or business logic — they just replay the exact same changes in the exact same order. The log is the source of truth for "what changed and when."

### What Gets Replicated?

There are three approaches to what the log actually contains:

**Statement-based replication** sends the original SQL statements. Simple and compact, but breaks with non-deterministic functions — `NOW()` returns a different time on each replica, `RAND()` gives different random numbers.

**Row-based replication** sends the actual data changes: "row 42 changed column `last_login` from X to Y." More data to transfer, but perfectly deterministic. This is what most modern databases default to.

**Mixed replication** uses statement-based for simple queries and switches to row-based for non-deterministic ones. MySQL supports this as a middle ground.

---

## The Fundamental Trade-off: Consistency vs Speed

Here's the central tension in all of replication: **when a write happens on the primary, how long before replicas have the same data?**

This question leads to two fundamentally different approaches.

### Synchronous Replication

The primary waits for replicas to confirm they've received and applied the change before telling the client "write successful."

```
Client                Primary             Replica
  │                     │                    │
  │── INSERT ──────────→│                    │
  │                     │── replicate ──────→│
  │                     │                    │── apply changes
  │                     │←── ACK ───────────│
  │←── SUCCESS ─────────│                    │
  │                     │                    │
  Timeline: ════════════════════════════════════
  Client waits for replica to confirm (~5-50ms extra)
```

**Guarantee:** If the primary crashes right after acknowledging the write, the data is safe on at least one replica. Zero data loss.

**Cost:** Every write is slower by the round-trip time to the replica. If a replica is in another region, that's 50-200ms added to every write. If a replica goes down, writes are blocked entirely until it recovers (or is removed from the synchronous set).

### Asynchronous Replication

The primary acknowledges the write immediately after committing locally. Replication happens in the background.

```
Client                Primary             Replica
  │                     │                    │
  │── INSERT ──────────→│                    │
  │                     │── commit locally   │
  │←── SUCCESS ─────────│                    │
  │                     │── replicate ──────→│  (background)
  │                     │                    │── apply changes
  │                     │                    │
  Timeline: ════════════════════════════════════
  Client gets response immediately. Replica catches up later.
```

**Guarantee:** Fast writes. Replica failures don't affect the primary's ability to accept writes.

**Cost:** If the primary crashes before replication completes, recent writes are lost — they existed only on the primary. Replicas may serve **stale data** because they haven't caught up yet.

### Semi-Synchronous: The Practical Middle Ground

Most production systems use **semi-synchronous replication**: wait for at least one replica to acknowledge, then replicate to the rest asynchronously. This guarantees that data survives on at least two machines, while keeping latency manageable.

MySQL's semi-sync replication, PostgreSQL's `synchronous_commit`, and MongoDB's write concern `w:majority` all implement variants of this approach.

---

## Replication Lag: The Inevitable Consequence

With asynchronous replication, there's always a delay between when data is written to the primary and when it appears on replicas. This delay is called **replication lag**, and it's the source of most replication headaches.

Lag is typically sub-second under normal conditions but can spike during:

- Heavy write traffic (replicas fall behind)
- Network congestion between primary and replicas
- Long-running queries on replicas (blocks replay)
- Replica hardware being slower than the primary

### The Read-After-Write Problem

The most common user-visible symptom of replication lag:

```
1. User updates their profile name to "Alice"     → goes to Primary
2. User refreshes the page                         → reads from Replica
3. Replica hasn't caught up yet                    → shows old name "Bob"
4. User thinks the update failed!
```

This is frustrating and confusing. There are several strategies to handle it:

**Read-your-own-writes consistency.** After a user performs a write, route that user's subsequent reads to the primary (or to a replica known to be caught up) for a short window. Everyone else can still read from replicas normally.

**Monotonic reads.** Ensure each user always reads from the same replica within a session, so they never see data go "backwards" (seeing a newer state, then an older one on the next request).

**Causal consistency.** If operation B depends on operation A, ensure any replica that has seen B has also seen A. More complex to implement but prevents logical paradoxes.

---

## Primary-Replica (Master-Slave) Architecture

This is the most common replication architecture, used by the vast majority of production databases. One node is the **primary** (accepts all writes), and one or more **replicas** handle read traffic.

```
┌──────────────────────────────────────────────────┐
│              Application Layer                   │
│                                                  │
│  Writes ──→ Primary           Reads ──→ Replicas │
└──────────────────────────────────────────────────┘
              │                         │
              ▼                    ┌────┼────┐
       ┌──────────┐               ▼    ▼    ▼
       │ Primary  │         ┌────┐ ┌────┐ ┌────┐
       │ (R/W)    │────────→│ R1 │ │ R2 │ │ R3 │
       └──────────┘  repl.  │(RO)│ │(RO)│ │(RO)│
                     log    └────┘ └────┘ └────┘
```

### Why This Works Well

**Simplicity.** There's one source of truth for writes. No conflicts, no need to merge divergent data. The replication log flows in one direction.

**Read scaling.** Need to handle more read traffic? Add another replica. Each one can serve reads independently. Instagram uses this pattern to serve billions of reads per day — a handful of primaries with dozens of replicas.

**Dedicated workloads.** You can point your analytics queries at a dedicated replica, keeping heavy reporting jobs from slowing down the primary that serves your live application.

### The Write Bottleneck

The obvious limitation: all writes must go through a single primary. If your application is write-heavy (think IoT sensors sending millions of data points per second), one primary may not be enough. This is where you'd look at sharding (splitting data across multiple primaries) or multi-master replication.

### When to Use Primary-Replica

This pattern fits when your workload is **read-heavy** (the typical ratio is 80-95% reads), which covers most web applications:

- Social media feeds (reads vastly outnumber posts)
- E-commerce product catalogs (browsing vs purchasing)
- Content management systems (reading articles vs publishing)
- Analytics dashboards (queries vs data ingestion)

---

## Multi-Master Replication

In multi-master (or master-master) replication, **multiple nodes accept writes**. Each node replicates its changes to the others bidirectionally.

```
┌──────────────────────┐          ┌──────────────────────┐
│ Master A (US-East)   │◄────────►│ Master B (EU-West)   │
│ Accepts reads+writes │  bidir.  │ Accepts reads+writes │
│ from US users        │  repl.   │ from EU users        │
└──────────────────────┘          └──────────────────────┘
         │                                   │
    ┌────┼────┐                         ┌────┼────┐
    ▼    ▼    ▼                         ▼    ▼    ▼
   R1   R2   R3                        R4   R5   R6
  (read replicas)                     (read replicas)
```

The appeal is obvious: users in the US write to a local master, users in Europe write to their local master, and both sides stay in sync. No cross-ocean latency for writes. No single point of failure for writes.

### The Price: Conflicts

When two masters can accept writes independently, they can receive **conflicting writes** at the same time. User A updates their name to "Alice" on Master 1, while User B updates the same record to "Bob" on Master 2, before either change has replicated. Now what?

**Last-Write-Wins (LWW)** is the simplest strategy: attach a timestamp to every write, and when conflicts are detected, the write with the later timestamp wins. It's easy to implement but can silently lose data — the "losing" write simply vanishes.

**Vector clocks** track causality between events. Each node maintains a logical clock, and by comparing clocks you can determine whether two writes are causally related (one happened after the other) or truly concurrent (a genuine conflict that needs resolution).

**Application-level resolution** pushes the conflict to your code. CouchDB stores both conflicting versions and lets the application decide. Amazon's Dynamo (the internal system behind DynamoDB) famously uses this for shopping cart merges — if two conflicting cart versions exist, it takes the union (better to have a duplicate item than to lose one).

### The Split-Brain Problem

If the network between two masters goes down, each master continues accepting writes independently. When the network recovers, the masters have diverged — this is called **split-brain**.

```
Normal:
  Master A ◄──────────► Master B
             connected

Network partition:
  Master A    ✗    ✗    Master B
  (still accepting     (still accepting
   writes!)             writes!)

Recovery:
  Master A ◄──────────► Master B
  "I have 500 writes"  "I have 300 writes"
  "you haven't seen"   "you haven't seen"
  → Must reconcile all 800 writes
```

The standard defense is **quorum**: a node only accepts writes if it can communicate with a majority of other nodes. If you have 3 masters and one gets isolated, it goes read-only because it can't reach a majority. The other two continue operating normally.

### When to Use Multi-Master

Multi-master is significantly more complex to operate than primary-replica. Use it when you genuinely need:

- **Multi-region writes** — users on different continents need low-latency writes
- **No single point of failure for writes** — even primary-replica failover has a brief window of unavailability

Real-world examples: CouchDB, Cassandra, and DynamoDB are designed for multi-master. MySQL and PostgreSQL support it but it's operationally complex and rarely recommended as the default.

---

## Failover: When the Primary Dies

In a primary-replica setup, the primary failing is the most critical event. The system needs to **promote a replica to become the new primary** — quickly and with minimal data loss.

### Automatic Failover

```
Normal operation:
  Primary ──→ Replica A, Replica B, Replica C

Primary crashes:
  Primary ✗   Replica A, Replica B, Replica C
                  │
                  ↓ (health check detects failure)

Failover process:
  1. Detect primary is unreachable (missed heartbeats)
  2. Select best replica (most up-to-date replication position)
  3. Promote selected replica to primary
  4. Redirect all writes to new primary
  5. Point remaining replicas to the new primary

After failover:
  [old Primary offline]
  Replica A (now Primary) ──→ Replica B, Replica C
```

The critical decision is **which replica to promote**. You want the one with the most recent replication position — the one that had applied the most changes from the old primary before it died. Any changes that existed only on the old primary (committed but not yet replicated) are lost.

### The Danger of Automatic Failover

Automatic failover sounds great but has real risks:

**False positives.** The primary might be temporarily slow (heavy query, garbage collection pause) rather than actually dead. If you promote a replica prematurely, you end up with two nodes thinking they're the primary — split-brain.

**Data loss.** With async replication, the promoted replica may be missing the most recent writes. When the old primary comes back, it has data the new primary doesn't, leading to conflicts.

**Cascading failures.** If the failover itself triggers heavy load (all connections reconnecting, replicas reconfiguring), it can destabilize the remaining nodes.

This is why many teams keep failover **semi-automatic**: the system detects the failure and prepares the promotion, but a human confirms before executing it. The extra 30 seconds of downtime is often worth the safety.

### Planned Failover (Maintenance)

When you need to take the primary offline for maintenance, the process is cleaner:

1. Set primary to read-only (stop accepting new writes)
2. Wait for all replicas to fully catch up
3. Promote the target replica
4. Redirect traffic
5. Demote the old primary to a replica (or take it offline)

Since replicas are fully caught up before promotion, there's zero data loss.

---

## How Real Databases Implement Replication

### MySQL

MySQL uses **binary log (binlog)** replication. The primary writes changes to its binlog; replicas connect and read the binlog stream. MySQL supports statement-based, row-based, and mixed formats, with row-based being the recommended default.

**GTID (Global Transaction Identifiers)** — introduced in MySQL 5.6 — assigns a unique ID to every transaction, making it easy for replicas to track exactly which transactions they've applied. This simplifies failover because a replica can tell the new primary "I've applied transactions up to GTID X, send me everything after that."

MySQL also offers **Group Replication** for multi-master setups with built-in conflict detection, and **InnoDB Cluster** as a high-availability solution combining Group Replication with automatic failover.

### PostgreSQL

PostgreSQL uses **WAL (Write-Ahead Log) streaming**. The primary streams its WAL to replicas, which replay it. This is physical replication — replicas apply the exact same byte-level changes.

PostgreSQL also supports **logical replication** (since v10), which replicates at the row level and allows replicas to have different indexes, different schemas, or even be different PostgreSQL versions. This is useful for zero-downtime upgrades.

Key PostgreSQL features:

- **Synchronous replication** with configurable `synchronous_standby_names`
- **Streaming replication** for near-zero lag
- **Hot standby** — replicas can serve read queries while replaying WAL
- `pg_promote()` for replica promotion during failover

### MongoDB

MongoDB's replication unit is the **replica set** — a group of `mongod` instances where one is primary and the rest are secondaries. MongoDB uses an operation log (**oplog**) that secondaries continuously tail.

Elections are automatic: if the primary goes down, the remaining secondaries hold an election (using Raft-like consensus) and promote one of themselves. This is built into MongoDB's core — no external tooling needed.

**Write concern** controls durability: `w:1` acknowledges after the primary commits, `w:majority` waits for a majority of replica set members, `w:all` waits for every member. **Read preference** controls where reads go: `primary`, `secondary`, `primaryPreferred`, `secondaryPreferred`, or `nearest`.

### Cassandra

Cassandra takes a fundamentally different approach — it's a **masterless** (peer-to-peer) system. Every node can accept both reads and writes. Data is replicated to N nodes determined by the replication factor (typically 3).

Consistency is tunable per query: `ONE` (fastest, least consistent), `QUORUM` (majority must respond), `ALL` (slowest, strongest). The combination of write consistency and read consistency determines your actual guarantee — if `W + R > N` (where N is the replication factor), you get strong consistency.

---

## Choosing the Right Strategy

```
What's your primary concern?
    │
    ├─ High availability for a read-heavy app (80%+ reads)
    │   └─ Primary-Replica with async replication
    │      (MySQL, PostgreSQL, MongoDB replica sets)
    │
    ├─ Zero data loss is non-negotiable (financial, healthcare)
    │   └─ Primary-Replica with synchronous replication
    │      (PostgreSQL synchronous standby, MySQL semi-sync)
    │
    ├─ Multi-region low-latency writes
    │   └─ Multi-Master or masterless
    │      (Cassandra, CockroachDB, DynamoDB)
    │
    ├─ Write-heavy with tunable consistency
    │   └─ Masterless with quorum
    │      (Cassandra, DynamoDB)
    │
    └─ Simple setup, acceptable brief downtime on failure
        └─ Primary-Replica with manual failover
           (PostgreSQL streaming replication, MySQL binlog)
```

### The Spectrum of Consistency vs Performance

```
Strong ◄────────────────────────────────────────► Eventually
Consistent                                       Consistent

Sync replication    Semi-sync     Async          Masterless
(all replicas       (1+ replica   (primary       (any node
 confirm before     confirms,     confirms,      accepts
 acknowledging)     rest async)   rest catch up)  writes)

Slowest writes                                   Fastest writes
Zero data loss                                   Possible data loss
Lowest availability                              Highest availability
```

Most production systems land in the semi-sync zone — fast enough for users, safe enough for the business.

---

## Common Pitfalls

**Ignoring replication lag in application code.** If your app writes to the primary then immediately reads from a replica, it may not see its own write. Route reads-after-writes to the primary, or use session-based consistency.

**Not monitoring replication lag.** Lag can spike silently. Monitor it continuously and alert when it exceeds your acceptable threshold (typically 1-5 seconds for most applications). In PostgreSQL, check `pg_stat_replication`; in MySQL, check `Seconds_Behind_Master`; in MongoDB, check `rs.printReplicationInfo()`.

**Promoting an under-replicated replica.** During failover, always promote the replica with the most recent replication position. Promoting a lagging replica means losing all writes between its position and the primary's last position.

**Over-relying on automatic failover.** Automated systems can trigger false failovers during transient network blips, causing unnecessary disruption. Many production systems use a combination: automated detection with human-confirmed promotion, or automated failover with strict health-check thresholds (e.g., 3 consecutive failures before triggering).

**Ignoring the "old primary comes back" scenario.** After failover, the old primary will try to rejoin with data the new primary doesn't have. If you don't handle this (by wiping and re-syncing the old primary), you risk data corruption. PostgreSQL's `pg_rewind` and MySQL's GTID-based replication help automate this.

---

## Key Takeaways

1. **Replication keeps copies of your data on multiple machines** — for availability, read scaling, and geographic distribution.

2. **Synchronous replication guarantees zero data loss but slows writes.** Asynchronous is faster but risks losing recent writes on primary failure. Semi-synchronous is the practical middle ground.

3. **Replication lag is inevitable with async replication.** Design your application to handle it — route critical reads to the primary, use session consistency, monitor lag continuously.

4. **Primary-replica is the right default** for most applications. It's simple, well-understood, and handles read-heavy workloads (which is most workloads).

5. **Multi-master adds write scalability but introduces conflict resolution** — a significantly harder problem. Only use it when you genuinely need multi-region writes.

6. **Failover is the most critical operation.** Test it regularly, monitor replica lag, and decide in advance whether it's automatic or human-confirmed.

7. **Replication is not a backup.** If someone runs `DELETE FROM users` on the primary, that DELETE replicates to every replica instantly. You still need point-in-time backups.

---

## Related Topics

- **[Sharding](sharding.md)** — when one primary can't handle your write volume, split data across multiple primaries
- **[Database Types](database-types.md)** — replication models differ significantly between relational and distributed databases
- **[Indexing](indexing.md)** — replicas maintain their own indexes; each copy has the same index overhead
