# Data Consistency

In distributed systems, data is replicated across multiple nodes to improve
availability and reduce latency. Consistency determines what guarantees the
system provides about when and how replicas reflect the latest writes. Stronger
consistency is more intuitive for developers but costs more in latency and
availability. Weaker consistency is faster and more resilient but requires
applications to tolerate stale or conflicting data.

The consistency spectrum ranges from linearizable (where the system behaves as
if there is a single copy of the data) to weak (where reads may return
arbitrarily stale values). Most production systems choose a point on this
spectrum based on the business cost of a stale read.

```
Strong <----------------------------------------> Weak
(slowest, safest)                        (fastest, riskiest)

Linearizable   Sequential   Causal   Eventual   Weak
  Spanner       ZooKeeper   MongoDB  DynamoDB   Memcached
  ~200ms         ~50ms       ~30ms     ~5ms      ~1ms
```

---

=== "Consistency Models"

    Each consistency model offers a different contract between the system and the
    application. The models below are ordered from strongest to weakest.

    **Linearizable (Strong) Consistency** guarantees that every read returns the
    most recent completed write. The system behaves as though there is a single
    copy of the data, even when it is replicated across dozens of nodes. Once a
    write completes, every subsequent read from any client on any node reflects
    that write.

    ```
    Timeline: Linearizable Consistency

    Client A:  ---[ Write X=5 ]----->|
                                     | (write acknowledged)
    Client B:         ---[ Read X ]--|------> returns 5
    Client C:              ---[ Read X ]----> returns 5

    After the write completes, ALL reads see the new value.
    No client ever sees X with the old value.
    ```

    This is the model used by Google Spanner and etcd. It is the easiest to
    reason about but the most expensive to provide.

    **Sequential Consistency** guarantees that all nodes see operations in the
    same order, but that order does not have to match real-time wall-clock order.
    Two clients may disagree about which write happened "first" in absolute time,
    but they will agree on the sequence of events they observe.

    ```
    Timeline: Sequential Consistency

    Real time:   Client A writes X=1, then Client B writes X=2

    Node 1 sees: X=1, then X=2  (matches real time)
    Node 2 sees: X=2, then X=1  (different from real time, but consistent
                                  with what Node 2 always shows)

    All nodes see the SAME sequence. That sequence may not match
    the wall-clock order of writes.
    ```

    ZooKeeper provides sequential consistency. It is useful for coordination
    tasks such as leader election and configuration management, where all
    participants must agree on ordering but real-time guarantees are unnecessary.

    **Causal Consistency** preserves cause-and-effect relationships. If operation
    A causally precedes operation B (for example, B is a reply to A), then every
    node sees A before B. However, operations that are concurrent and unrelated
    may appear in different orders on different nodes.

    ```
    Timeline: Causal Consistency (Chat Example)

    Alice:  "What time is the meeting?"        (Message 1)
    Bob:    "3pm" (reply to Message 1)          (Message 2)
    Carol:  "Anyone want coffee?" (unrelated)   (Message 3)

    Valid orderings on any node:
      [1, 2, 3]  or  [1, 3, 2]  or  [3, 1, 2]

    Invalid ordering:
      [2, 1, 3]  -- reply before the question it answers
    ```

    MongoDB supports causal consistency within client sessions. This model is
    well suited for chat applications, comment threads, and collaborative editing
    where preserving the reply-to relationship matters more than global ordering.

    **Eventual Consistency** guarantees that if no new writes occur, all replicas
    will converge to the same value. The convergence window is typically
    milliseconds to low seconds, not minutes or hours. During that window,
    different clients may read different values from different replicas.

    ```
    Timeline: Eventual Consistency

    Client A:  ---[ Write X=5 to Replica 1 ]---->  (5ms, done)
    Replica 1: X=5
    Replica 2: X=old  ---(async replication)---> X=5  (after ~100ms)
    Replica 3: X=old  ---(async replication)---> X=5  (after ~200ms)

    During the convergence window, reads from Replica 2 or 3
    return the old value. After convergence, all replicas agree.
    ```

    DNS is an everyday example of eventual consistency. When you update a DNS
    record, the change propagates through caches based on TTL values. During
    propagation, some users resolve the old IP address while others get the new
    one. Amazon DynamoDB, Cassandra, and S3 all default to eventual consistency.

    **Weak Consistency** provides no ordering or convergence guarantees at all.
    After a write, there is no promise that any subsequent read will see it. This
    model is appropriate only when approximate or lossy data is acceptable, such
    as live video streaming (where dropping frames is preferable to buffering) or
    real-time analytics dashboards showing approximate visitor counts.

=== "Strong Consistency"

    Strong (linearizable) consistency makes a distributed system behave as if
    there is only one copy of the data. Every read returns the most recent write,
    regardless of which replica serves the request.

    **How it works.** The write path requires synchronous replication: the system
    does not acknowledge a write until all replicas (or a strict quorum) have
    confirmed it. This coordination ensures that no replica can serve a stale
    value. The cost is latency, because the write must wait for the slowest
    replica, including cross-datacenter round trips that can take 50 to 200
    milliseconds. Availability also suffers, because the system cannot accept
    writes if a required replica is unreachable.

    ```
    Strong Consistency Write Path

    Client --> Leader
                 |----> Replica A  (ack)
                 |----> Replica B  (ack)
                 |----> Replica C  (ack)
                 |
                 | All replicas acknowledged
                 |
    Client <-- Leader (write confirmed)

    If Replica C is down, the write BLOCKS until C recovers
    or the system times out and rejects the write.
    ```

    **Google Spanner** achieves linearizable consistency at global scale using
    TrueTime, a clock system based on GPS receivers and atomic clocks in every
    datacenter. TrueTime provides bounded clock uncertainty (typically under 7
    milliseconds), which Spanner uses to order transactions globally without
    requiring all replicas to communicate on every read. Spanner serves Google
    AdWords and Google Play, handling millions of transactions per second with
    commit latencies around 5 to 10 milliseconds within a region and 50 to 200
    milliseconds across continents.

    **PostgreSQL synchronous replication** provides strong consistency within a
    cluster. Setting `synchronous_commit = on` ensures that a transaction is not
    acknowledged until at least one standby has written the WAL record to disk.
    This adds 1 to 5 milliseconds of latency within a datacenter but guarantees
    no data loss on primary failure.

    **When you need strong consistency.** The decision comes down to the business
    cost of a stale read. Financial transactions require it because a stale
    balance can lead to double-spending. Inventory systems require it because a
    stale stock count leads to overselling. Distributed locks require it because
    two holders of the same lock corrupt shared state. In each case, the cost of
    inconsistency (lost money, angry customers, data corruption) far exceeds the
    cost of higher latency.

    A useful heuristic: if a stale read can cause a financial loss or a safety
    violation, use strong consistency. If a stale read merely causes a minor user
    experience issue, weaker models are appropriate.

=== "Eventual Consistency"

    Eventual consistency allows the system to acknowledge a write as soon as one
    replica records it, then replicates asynchronously to the others. This makes
    writes fast and the system highly available, but clients may read stale data
    during the convergence window.

    **How it works.** A write goes to the nearest replica (or any available
    replica), which acknowledges it immediately. Background replication processes
    propagate the write to other replicas over the network. The convergence
    window, the time until all replicas agree, is typically 50 to 500
    milliseconds within a region and 1 to 5 seconds across regions.

    ```
    Eventual Consistency Write Path

    Client --> Replica 1 (local)
                 |  Write acknowledged immediately (5ms)
                 |
                 |~~~> Replica 2  (async, ~100ms later)
                 |~~~> Replica 3  (async, ~200ms later)

    During the convergence window, reads from Replica 2 or 3
    may return stale data. After convergence, all agree.
    ```

    The central challenge of eventual consistency is conflict resolution. When
    two clients write different values to the same key on different replicas
    before replication occurs, the system must decide which value wins.

    **Last-write-wins (LWW)** picks the write with the highest timestamp. It is
    simple and automatic but discards the losing write entirely. DynamoDB and
    Cassandra use LWW by default. This works well for data where only the latest
    value matters (user settings, session tokens) but is dangerous for data that
    accumulates (shopping carts, counters).

    **Vector clocks** track causal relationships between writes. When two writes
    are concurrent (neither caused the other), the system detects the conflict
    and presents both versions to the application for resolution. Riak used this
    approach, returning "sibling" values that the application merged on read.

    **CRDTs (Conflict-free Replicated Data Types)** are data structures
    mathematically guaranteed to converge without conflicts. A G-Counter, for
    example, tracks per-node increments and sums them on read, so concurrent
    increments never conflict. Redis and Riak support CRDTs for counters, sets,
    and maps.

    **Amazon DynamoDB and the shopping cart.** Amazon's foundational Dynamo paper
    described the shopping cart as a system that must always accept writes (high
    availability) even at the cost of temporary inconsistency. If a customer adds
    items from two devices simultaneously, Dynamo stores both versions and merges
    them on the next read, using application-level logic to union the item lists.
    DynamoDB serves Amazon retail at over 10 million requests per second with
    single-digit millisecond latency.

    **DNS as everyday eventual consistency.** When you update a DNS A record, the
    authoritative server propagates the change, but resolvers worldwide cache the
    old record until its TTL expires. With a 300-second TTL, some users see the
    old IP for up to 5 minutes while others see the new one. This is eventual
    consistency that billions of users interact with daily without noticing.

    **When eventual consistency is the right choice.** Social media likes and
    view counts are classic examples. Instagram serves over 2 billion monthly
    active users; displaying a like count that is a few seconds stale has no
    business impact, but requiring strong consistency for every like would
    multiply read latency and infrastructure cost. User activity feeds, product
    catalog browsing, and CDN content delivery all fit this model.

=== "Choosing a Model"

    The right consistency model depends on the business impact of a stale or
    incorrect read, not on technical preference. The question to ask is: what
    happens if a user sees data that is one second old? If the answer is
    "nothing noticeable," eventual consistency is appropriate. If the answer is
    "we lose money or violate a contract," strong consistency is required.

    **Per-operation consistency.** Most production systems do not use a single
    consistency model for all data. Netflix uses strong consistency for billing
    (a customer must not be double-charged) but eventual consistency for viewing
    history (showing a recently watched title a few seconds late is harmless).
    Uber uses strong consistency for ride assignment (two drivers must not be
    assigned the same rider) but eventual consistency for the map display
    (driver positions can lag by a second).

    | Model | Latency | Availability | Data Safety | Best For |
    |-------|---------|--------------|-------------|----------|
    | Linearizable | 50-200ms | Lower (blocks on failure) | Highest | Finance, inventory, locks |
    | Sequential | 20-100ms | Medium | High | Coordination, config |
    | Causal | 10-50ms | High | Medium | Chat, comments, collab editing |
    | Eventual | 1-10ms | Highest | Lower | Feeds, likes, catalogs |
    | Weak | <5ms | Highest | Lowest | Streaming, analytics |

    **Tunable consistency.** Some databases let you choose consistency on a
    per-query basis rather than system-wide. Cassandra uses quorum reads and
    writes controlled by three parameters: N (total replicas), W (write
    acknowledgments required), and R (read acknowledgments required). When
    W + R > N, at least one replica that acknowledged the write will be included
    in the read quorum, guaranteeing strong consistency for that query. When
    W + R <= N, the system operates with eventual consistency but lower latency.

    ```
    Cassandra Tunable Consistency

    N = 3 replicas

    Strong:   W=2, R=2  -->  2+2=4 > 3  (overlap guaranteed)
    Eventual: W=1, R=1  -->  1+1=2 < 3  (no overlap, faster)

    Per-query examples:
      INSERT ... USING CONSISTENCY QUORUM;   -- strong
      SELECT ... USING CONSISTENCY ONE;      -- eventual, fast
    ```

    This lets a single Cassandra cluster serve both billing queries (QUORUM) and
    analytics queries (ONE) without separate infrastructure. DynamoDB offers a
    similar choice with its `ConsistentRead` parameter: set it to true for
    strong reads on critical paths, leave it false for eventually consistent
    reads everywhere else.

    **Decision framework.** For each piece of data in your system, answer these
    questions: (1) What is the cost of a stale read? (2) What is the acceptable
    staleness window? (3) Can conflicts be resolved automatically? If the cost
    is high, use strong consistency. If the cost is low and the staleness window
    is seconds, use eventual consistency. If causal relationships matter (replies
    must follow messages), use causal consistency. Document these decisions
    explicitly so the team understands why each choice was made.

---

## Key Takeaways

1. Consistency is a spectrum, not a binary choice. There are at least five
   distinct models between linearizable and weak, each with different latency,
   availability, and safety trade-offs.

2. Choose the weakest model that satisfies business requirements. Stronger
   consistency costs more in latency and availability. Using strong consistency
   where eventual would suffice wastes resources and limits scalability.

3. Most production systems use different consistency levels for different data.
   Netflix, Uber, and Amazon all mix strong and eventual consistency within the
   same application based on the business impact of stale reads.

4. Eventual does not mean "eventually, sometime." Convergence windows are
   typically milliseconds to low seconds. Define an SLA (for example, "99% of
   reads consistent within 200ms") and monitor it.

5. Conflict resolution must be planned upfront. Last-write-wins is simple but
   loses data. Vector clocks detect conflicts but require application logic.
   CRDTs guarantee convergence but support only specific data structures.

6. Tunable consistency, available in Cassandra and DynamoDB, lets you choose
   per-query rather than per-cluster, giving you strong consistency where it
   matters and eventual consistency where speed matters.

---

## Related Topics

- [CAP Theorem](cap-theorem.md) -- the fundamental trade-off between consistency and availability during network partitions
- [Distributed Systems Principles](principles.md) -- broader principles that inform consistency choices
- [Database Replication](../data/databases/replication.md) -- the mechanisms (synchronous, asynchronous, quorum) that underpin consistency guarantees
- [Consensus Algorithms](../distributed-systems/consensus.md) -- Paxos, Raft, and ZAB, the protocols that enable strong consistency
