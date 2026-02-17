# CAP Theorem

Brewer's theorem, proven by Seth Gilbert and Nancy Lynch in 2002, states that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance. Since network partitions are inevitable in any distributed system -- switches fail, cables get cut, packets get dropped -- the real choice during a partition is between consistency and availability. This realization transforms CAP from a theoretical "pick two of three" puzzle into a practical engineering decision: when a partition occurs, does your system refuse requests to stay correct, or does it keep serving requests at the risk of returning stale data?

---

=== "The Theorem"

    ## The Three Properties

    Each of CAP's three properties has a precise technical meaning that is worth understanding exactly, because vague definitions lead to incorrect architectural decisions.

    **Consistency** means linearizability: every read receives the most recent write or an error. If a client writes value v2 to the system and that write completes successfully, then every subsequent read from any node must return v2 (or a later value). This is the strongest form of consistency and is much stricter than "eventual consistency" or "read-your-writes." A system where two clients can read different values for the same key at the same moment -- even briefly -- does not satisfy CAP consistency.

    **Availability** means that every request to a non-failing node receives a non-error response, with no timeouts. The system does not simply "try its best" -- it guarantees a response for every request. A system that returns a 503 error because it cannot reach a quorum is, by CAP's definition, unavailable. Note that CAP availability does not specify how fast the response must arrive, only that one must eventually come. It also does not require the response to contain the most recent data -- just that it is a valid, non-error response.

    **Partition tolerance** means the system continues to operate despite arbitrary message loss or delay between nodes. A partition is not a theoretical edge case. It is a regular occurrence in any system that spans more than one machine. The system must continue to function (satisfying either C or A) even when some nodes cannot communicate with others.

    ### Why Partition Tolerance Is Non-Negotiable

    Network partitions happen in every datacenter, at every scale. Google reported in a 2011 study that network partitions occur roughly once every few weeks across their infrastructure. A study of Microsoft Azure found that network redundancy reduces but does not eliminate partitions. Aphyr's Jepsen testing project has documented partition-related consistency violations in nearly every distributed database tested, including MongoDB, Cassandra, CockroachDB, and Elasticsearch.

    Even within a single rack, a faulty top-of-rack switch can isolate a group of machines from the rest of the cluster. Across datacenters, the causes multiply: undersea cable damage (the 2008 Mediterranean cable cuts took down internet connectivity for millions), BGP misconfigurations (Cloudflare's 2020 backbone outage), or simple congestion causing packet loss that makes nodes appear unreachable.

    Because you cannot prevent partitions, every distributed system must tolerate them. The "CA" combination -- consistency plus availability without partition tolerance -- is only possible on a single machine. The moment you add a second node, you must plan for the possibility that the two nodes cannot communicate. The CAP choice therefore reduces to: during a partition, do you sacrifice consistency or availability?

    ### The Partition Scenario

    The following diagram illustrates the fundamental decision:

    ```
        Client                    Client
          |                         |
       Node A ----X---- Node B
       (has v2)         (has v1)

       Partition! Node A and Node B cannot communicate.
       CP choice: Node B rejects reads (unavailable) to avoid returning stale v1
       AP choice: Node B returns v1 (stale but available)
    ```

    When the network link between Node A and Node B breaks, Node A has received a write updating the value to v2, but Node B still holds the old value v1. A CP system forces Node B to reject requests until it can confirm the latest state, ensuring no client ever sees stale data. An AP system lets Node B keep serving requests with its local (possibly stale) copy, keeping the system responsive at the cost of correctness.

    Consider a more concrete scenario with a 5-node cluster:

    ```
    Normal operation (all nodes connected):

    Node A ------- Node B ------- Node C
      |               |               |
      +------ Node D ------+--- Node E

    Partition splits the cluster:

    +------------------+     +------------------+
    | Partition 1      |     | Partition 2      |
    | Node A, B, D     |     | Node C, E        |
    | (majority: 3/5)  |     | (minority: 2/5)  |
    +------------------+     +------------------+

    CP behavior:
      Partition 1: continues operating (has quorum of 3)
      Partition 2: stops accepting writes (no quorum)

    AP behavior:
      Partition 1: continues operating
      Partition 2: also continues operating (may diverge)
    ```

    This is not an abstract concern. It is the decision that determines how your banking system, shopping cart, or social media feed behaves when infrastructure fails.

    ### The CAP Decision Summary

    The following table summarizes what each combination means in practice:

    | Choice | During Partition           | Trade-off                          | Example Systems            |
    |--------|---------------------------|------------------------------------|----------------------------|
    | CP     | Reject requests if stale  | Some users see errors              | MongoDB, HBase, ZooKeeper  |
    | AP     | Serve possibly stale data | Some users see outdated values     | Cassandra, DynamoDB, Riak  |
    | CA     | Not possible (distributed)| Only works on a single machine     | Single-node PostgreSQL     |

    ### A Common Misconception

    Many descriptions of CAP say "pick any two of three," implying three equal combinations: CA, CP, and AP. This framing is misleading. In a distributed system, you cannot choose to ignore partitions -- they will happen whether you plan for them or not. A system that does not handle partitions is not a distributed system; it is a single point of failure. The only valid combinations for distributed systems are CP and AP, and the choice between them only matters during the (hopefully brief) periods when a partition is active. During normal operation, a well-designed system can provide both consistency and availability.

    Another common misunderstanding is confusing CAP consistency with ACID consistency. ACID consistency means a transaction moves the database from one valid state to another (referential integrity, constraints). CAP consistency means linearizability -- all nodes agree on the same value at the same time. A system can satisfy ACID consistency while violating CAP consistency, and vice versa. These are fundamentally different properties despite sharing a name.

=== "CP Systems"

    ## Consistency and Partition Tolerance

    A CP system prioritizes correctness: during a partition, nodes that cannot confirm they have the latest data will refuse to serve requests. The system becomes partially or fully unavailable, but every response it does return is guaranteed to be correct. The philosophy is straightforward: it is better to say "I don't know" than to give a wrong answer.

    ### How a CP System Handles a Partition

    ```
    Before partition:
      Client --> Node A (leader) --> replicates to Node B, Node C
      All nodes agree on current value v5

    Partition occurs (Node C isolated):
      Client --> Node A --> replicates to Node B (success)
      Quorum: 2 of 3 nodes agree = majority met
      Node A accepts the write, updates to v6

      Node C: cannot reach leader
              cannot confirm whether new writes have occurred
              rejects all read and write requests
              returns "503 Service Unavailable"

    After partition heals:
      Node C contacts leader, discovers it missed v6
      Node C syncs with leader, catches up to current state
      All three nodes serving consistent data again
    ```

    The minority side of the partition (Node C in this example) becomes unavailable. This is the core trade-off: you lose access to some fraction of your capacity, but you never serve stale data. Clients routed to Node C will see errors, but clients reaching Node A or Node B will see correct, up-to-date values.

    The quorum mechanism is central to how CP systems work. For a cluster of N nodes, a quorum typically requires a majority: (N/2) + 1 nodes. With 3 nodes, you need 2. With 5 nodes, you need 3. The key insight is that any two majorities must overlap by at least one node, so a read quorum and a write quorum are guaranteed to share at least one node that has the latest value:

    ```
    W + R > N  guarantees consistency
    where W = write quorum, R = read quorum, N = total replicas

    Example: N=5, W=3, R=3
    Write quorum:  [A, B, C]        (3 nodes)
    Read quorum:   [B, D, E]        (3 nodes)
    Overlap:       [B]              (at least 1 node has latest write)
    ```

    ### CP Systems in Practice

    **MongoDB** with write concern `w:majority` and journal enabled (`j:true`) operates as a CP system. Writes must be acknowledged by a majority of replica set members before the driver reports success. If a primary cannot reach a majority, it steps down and the affected shard becomes read-only (or unavailable) until a new primary is elected. MongoDB's replica sets typically contain 3 or 5 members, and primary election takes 10 to 12 seconds during which the affected shard cannot accept writes.

    **HBase** uses a single RegionServer per region, coordinated through ZooKeeper. If the RegionServer fails, the region becomes unavailable until ZooKeeper detects the failure and assigns a new RegionServer -- typically 30 to 60 seconds of downtime for that region. Facebook uses HBase to store messages for over 1.5 billion users, accepting brief unavailability windows in exchange for strong consistency of message ordering.

    **ZooKeeper** and **etcd** are coordination services built on consensus protocols (ZAB and Raft respectively). They require a quorum to process any request. In a 5-node ZooKeeper ensemble, if 3 nodes become isolated, the remaining 2 cannot form a quorum and stop accepting writes entirely. Kubernetes depends on etcd for cluster state, which is why etcd clusters are typically deployed across 3 or 5 availability zones -- losing one zone still leaves a quorum.

    **Redis Cluster** uses a primary-replica model where writes go to the primary. If the primary is partitioned away from the majority of nodes, the cluster will elect a new primary after a configurable timeout (default 15 seconds), and the old primary stops accepting writes. During this failover window, writes to that shard are unavailable.

    ### Use Cases for CP

    Banking and financial systems are the canonical CP use case. If a customer has $1,000 in their account and initiates two $800 transfers simultaneously from different ATMs, the system must reject one. Showing a stale balance and allowing both transfers would create $600 out of thin air. JPMorgan Chase processes roughly 10 billion transactions per year, and Visa handles over 65,000 transactions per second at peak -- every one must be consistent because financial regulators require accurate accounting at all times.

    Inventory management at scale requires CP behavior. Amazon sells approximately 4,000 items per minute during peak periods. During Prime Day 2023, customers purchased over 375 million items in 48 hours. If the inventory system showed stale counts, the same last unit could be sold to multiple customers, causing overselling, cancellations, and damaged trust. Airlines face the same challenge: a Boeing 737 has exactly 189 seats, and double-booking even one creates a cascading customer service problem.

    Distributed locks and leader election must be consistent. If two nodes both believe they are the leader (a split-brain scenario), they may issue conflicting commands that corrupt data or cause duplicate processing. Systems like etcd and ZooKeeper exist precisely to prevent this. When a distributed job scheduler elects a leader, it must be certain that exactly one node holds the lock -- an AP system that allows two leaders during a partition could launch the same batch job twice.

    ### The Cost of CP: Quantifying Unavailability

    CP systems accept unavailability as a trade-off, but how much unavailability? The answer depends on how often partitions occur and how quickly the system recovers.

    ```
    Unavailability during partition:

    Event                     Duration    Impact
    -----------------------------------------------
    Leader election            10-30 sec   Shard cannot accept writes
    Region server failover     30-60 sec   Region offline
    Cross-datacenter partition 1-30 min    Minority DC rejects requests
    Full datacenter outage     1-4 hours   All traffic rerouted

    For a 3-node cluster with 99.9% network uptime:
      ~8.7 hours of partition time per year
      During those hours, minority partition (1 node) is unavailable
      Effective availability: ~99.9% (with proper quorum sizing)
    ```

    The key insight is that CP does not mean "always unavailable." It means "unavailable only during partitions, and only on the minority side." A well-designed CP system with proper quorum sizes and fast leader election can still achieve very high availability numbers -- just not the 100% that AP systems target.

    ### Google Spanner: Pushing the Boundaries

    Google Spanner is technically a CP system, but it achieves near-CA behavior through an extraordinary engineering effort. Spanner uses TrueTime, a globally distributed clock system built on GPS receivers and atomic clocks in every datacenter. Each Google datacenter has multiple GPS receivers and atomic clocks, and the TrueTime API reports the current time along with an uncertainty bound (typically under 7 milliseconds).

    This bounded uncertainty allows Spanner to order transactions globally without the round-trip latency penalty of traditional consensus protocols. When committing a transaction, Spanner waits out the uncertainty window (a few milliseconds) to guarantee that no other transaction could have a conflicting timestamp. The result is a system that is externally consistent (stronger than linearizability) with 99.999% availability across globally distributed datacenters -- the closest any production system has come to "beating" CAP.

    ```
    How Spanner's TrueTime works:

    Traditional consensus:
      Node A --> "What time?" --> GPS/Atomic Clock --> "12:00:00.000 +/- 7ms"
      Node B --> "What time?" --> GPS/Atomic Clock --> "12:00:00.003 +/- 7ms"

      Commit at Node A:
        timestamp = TrueTime.now()     --> returns [earliest, latest]
        wait until TrueTime.after(timestamp)  --> ~7ms wait
        commit with timestamp

      Result: globally ordered transactions without cross-node communication
    ```

    Google achieves this by investing heavily in hardware (GPS receivers and atomic clocks in every datacenter are not cheap) and accepting a small latency floor (the TrueTime wait) for every write. CockroachDB and YugabyteDB take inspiration from Spanner but use hybrid logical clocks instead of TrueTime, achieving similar semantics without dedicated clock hardware at the cost of slightly higher commit latencies.

=== "AP Systems"

    ## Availability and Partition Tolerance

    An AP system prioritizes responsiveness: during a partition, every node continues to accept reads and writes. The system stays up, but different nodes may temporarily hold different versions of the same data. Once the partition heals, the system reconciles the divergent state through a conflict resolution mechanism. The philosophy is the mirror of CP: it is better to give a potentially stale answer than to give no answer at all.

    ### How an AP System Handles a Partition

    ```
    Before partition:
      Client A --> Node A    Client B --> Node B
      Both nodes hold value v5, replicating asynchronously

    Partition occurs (Node A and Node B cannot communicate):
      Client A writes v6 to Node A (succeeds immediately, local write)
      Client B writes v7 to Node B (succeeds immediately, local write)
      Both clients get fast responses, neither sees an error
      Node A thinks current value is v6
      Node B thinks current value is v7

    After partition heals:
      Node A has v6, Node B has v7 -- conflict detected!
      System applies conflict resolution strategy:
        Last-write-wins:  compare timestamps, keep v7 (later timestamp)
        Vector clocks:    detect concurrent writes, flag for resolution
        CRDT:             merge both values automatically (if data type supports it)

    Post-resolution:
      Both nodes agree on resolved value
      System is fully consistent again
    ```

    The key insight is that AP systems accept temporary inconsistency as the price of continuous operation. The inconsistency window -- the time between a partition occurring and the system resolving conflicts after it heals -- is typically seconds to minutes, not hours or days.

    ### AP Systems in Practice

    **Cassandra** is the most widely deployed AP database. It writes to the local node immediately and replicates asynchronously to other nodes in the cluster. Netflix runs Cassandra at a scale of over 10,000 nodes across multiple AWS regions, serving tens of millions of requests per second for its streaming catalog, user profiles, and viewing history. A brief partition between regions means users in one region might not see a title added in another for a few seconds -- completely acceptable for a streaming catalog. Apple reportedly runs one of the largest Cassandra deployments in the world, with over 150,000 nodes supporting iCloud and iTunes services.

    **DynamoDB** was designed from the ground up for availability. Amazon's original Dynamo paper (2007) described how the shopping cart service must always accept writes -- it is better to have a slightly confused cart than to show a customer an error page during checkout. At Amazon's scale (over 300 million active customer accounts, processing millions of orders per day), even a few seconds of cart unavailability during peak traffic translates to significant revenue loss. DynamoDB uses vector clocks to detect conflicting writes and resolves them on read, either automatically or by presenting all conflicting versions to the application.

    **CouchDB** takes a multi-version approach, storing all conflicting revisions and letting the application decide which to keep. This is particularly useful for offline-first applications where devices may be partitioned from the server for hours or days. A mobile app that stores data in a local CouchDB instance can continue working without network access, then sync and resolve conflicts when connectivity returns.

    **Riak** supports configurable conflict resolution including CRDTs (Conflict-free Replicated Data Types), which are data structures mathematically guaranteed to converge without coordination. Counters, sets, maps, and registers can be updated independently on different nodes and merged automatically when those nodes reconnect.

    ### Conflict Resolution Strategies

    When an AP system discovers that two nodes have divergent values for the same key, it must resolve the conflict. There are three main strategies, each with distinct trade-offs.

    **Last-write-wins** (LWW) is the simplest strategy: compare timestamps and keep the value with the latest one. Cassandra uses this by default. It works well when writes are independent (like updating a user's last-login timestamp) but can silently lose data when concurrent writes are both meaningful. If two users simultaneously add different items to a shared shopping list and the system uses LWW, one addition will be discarded. The simplicity of LWW makes it attractive, but teams must understand where it can lose data.

    **Vector clocks** track the causal history of each write as a vector of per-node counters. If write A happened before write B (A's vector is component-wise less than or equal to B's), the system can automatically discard A. If A and B are concurrent (neither vector dominates the other), the system detects a true conflict and escalates it -- either to the application for manual resolution or to a deterministic merge function. Amazon's Dynamo used vector clocks, though DynamoDB has since moved toward simpler last-writer-wins semantics for most use cases.

    **CRDTs** (Conflict-free Replicated Data Types) are data structures designed so that concurrent updates always converge to the same result regardless of the order in which updates are applied. A G-Counter (grow-only counter), for example, maintains a separate count per node. Each node increments only its own counter, and the merged value is the sum across all nodes. No matter the order of merging, the result is the same. CRDTs eliminate the need for conflict resolution at the cost of restricting what operations are supported -- you cannot, for example, build a CRDT that supports arbitrary key-value overwrites without some form of conflict policy.

    ### Use Cases for AP

    Social media feeds tolerate staleness well. Facebook serves over 2 billion daily active users across dozens of datacenters on four continents. If a user in Tokyo posts a photo and a friend in London does not see it for 3 to 5 seconds, that delay is imperceptible. But if the London user got an error page instead of their feed, they might close the app entirely. Instagram processes over 95 million photos per day -- each one replicated asynchronously to datacenters worldwide.

    Shopping carts benefit from AP design. Amazon's Dynamo paper established the principle that the cart should always accept additions. If a network partition causes two versions of a cart to diverge (perhaps the customer added an item on their phone while their laptop session also modified the cart), the system merges them on the next read. The customer might see a previously removed item reappear, but that is far better than losing the entire cart or blocking the purchase. During peak events like Prime Day, cart availability directly correlates with revenue.

    DNS is inherently an AP system. When you update a DNS record, the change propagates across the internet over minutes to hours depending on TTL settings. Every DNS resolver continues serving its cached (potentially stale) records during this propagation window. The global internet depends on this AP behavior -- if DNS required strong consistency, a single partition could make domains unreachable worldwide.

    ### The Cost of AP: Understanding the Inconsistency Window

    AP systems accept temporary inconsistency, but how inconsistent, and for how long? The answer varies dramatically by system and configuration.

    ```
    Typical inconsistency windows:

    System          Normal operation     During partition     After partition heals
    ---------------------------------------------------------------------------
    Cassandra       50-200 ms            Seconds to minutes   Seconds (anti-entropy)
    DynamoDB        < 1 second           Seconds              Milliseconds (read repair)
    CouchDB         Seconds              Minutes to hours     Minutes (bulk sync)
    DNS             Minutes to hours     Hours (TTL-based)    Hours (cache expiry)

    Key metric: Replication lag
      Cassandra cross-DC:    50-500 ms (normal), unbounded (partition)
      DynamoDB global table: < 1 second (normal), seconds (partition)
    ```

    The inconsistency window during normal operation is typically very short -- milliseconds for well-configured systems. It is only during actual partitions that the window grows. And even then, the system continues serving requests. For many applications, the probability of a user noticing stale data during the brief inconsistency window is extremely low compared to the certainty of them noticing an error page.

=== "Beyond CAP"

    ## The PACELC Theorem

    CAP only describes system behavior during a partition, which is actually the exceptional case. Most of the time, your system is running without any partition at all. Daniel Abadi proposed PACELC in 2010 to capture the full picture: even when the system is running normally (no partition), there is a fundamental trade-off between latency and consistency.

    PACELC stands for: if there is a **P**artition, choose between **A**vailability and **C**onsistency; **E**lse (normal operation), choose between **L**atency and **C**onsistency.

    The PACELC decision tree:

    ```
    Is there a network partition?
    |
    +-- YES (PAC): Choose between Availability and Consistency
    |   |
    |   +-- PA: Stay available, accept stale reads
    |   +-- PC: Stay consistent, reject requests from minority partition
    |
    +-- NO (ELC): Choose between Latency and Consistency
        |
        +-- EL: Favor low latency, replicate asynchronously
        +-- EC: Favor consistency, replicate synchronously (higher latency)
    ```

    This model explains why systems with the same CAP classification can feel very different in practice. Two CP systems might behave identically during a partition, but one might be fast during normal operation (asynchronous replication, reading from replicas) while the other is slow (synchronous replication, reading from the leader).

    | System     | During partition | During normal operation | PACELC     |
    |------------|-----------------|------------------------|------------|
    | DynamoDB   | Available (PA)  | Low latency (EL)       | PA/EL      |
    | Cassandra  | Available (PA)  | Tunable (EL or EC)     | PA/EL      |
    | MongoDB    | Consistent (PC) | Low latency (EL)       | PC/EL      |
    | Spanner    | Consistent (PC) | Consistent (EC)        | PC/EC      |
    | PNUTS      | Consistent (PC) | Low latency (EL)       | PC/EL      |

    MongoDB is PC/EL: it sacrifices availability during partitions but favors latency during normal operation (reads from secondaries are fast but may be slightly stale unless you explicitly request a primary read). Spanner is PC/EC: it is consistent in all conditions, paying a latency cost even during normal operation because every write must wait out the TrueTime uncertainty window.

    PACELC is more useful than CAP for day-to-day engineering because partitions are rare events (minutes per year in well-run infrastructure), while the latency-consistency trade-off affects every single request. When an architect says "we chose Cassandra for low latency," they are making a PACELC decision (EL), not a CAP decision. When they say "we need Spanner for global consistency," they are accepting the EC trade-off for correctness even during normal operation.

    ## Real Systems Are Tunable

    Most modern distributed databases do not fit neatly into "CP" or "AP." They offer tunable consistency that lets operators choose the trade-off per query or per table, making the binary CAP classification an oversimplification.

    Cassandra's consistency levels illustrate this spectrum clearly. With consistency level ONE, a write succeeds as soon as a single replica acknowledges it -- fast but weakly consistent. With QUORUM, a majority of replicas must acknowledge -- slower but strongly consistent if both reads and writes use QUORUM. With ALL, every replica must acknowledge -- slowest but guarantees no stale reads. A single Cassandra cluster can serve some queries at ONE (user activity tracking, where staleness is fine) and others at QUORUM (account balance lookups, where correctness matters).

    ```
    Cassandra consistency spectrum:

    ONE ----------- QUORUM ----------- ALL
    |               |                   |
    Fast            Balanced            Slow
    Weakly          Strongly            Fully
    consistent      consistent          consistent
    AP-like         Tunable             CP-like

    Read at ONE:    returns from first responding replica (may be stale)
    Read at QUORUM: waits for majority, returns most recent among them
    Read at ALL:    waits for every replica (guaranteed latest)
    ```

    DynamoDB offers similar tunability through its strongly consistent read option. By default, reads are eventually consistent (fast, served from any replica). But for critical operations, you can request a strongly consistent read that goes to the leader node -- trading latency for correctness on a per-request basis. This means a single DynamoDB table can serve both AP-style reads (for the product catalog browsing page) and CP-style reads (for the checkout inventory check).

    MongoDB's read concern levels provide another example. Read concern "local" returns data from the node that received the request (fast, possibly stale). Read concern "majority" returns only data that has been acknowledged by a majority of nodes (slower, consistent). Read concern "linearizable" provides the strongest guarantee but with the highest latency.

    ## Harvest and Yield

    In 1999, Eric Brewer (before formally stating the CAP conjecture) introduced the concepts of harvest and yield as a more nuanced way to think about degraded operation.

    **Yield** is the probability of completing a request -- analogous to availability but expressed as a continuous value rather than a binary property. A system with 99.99% yield fails to respond to 1 in 10,000 requests.

    **Harvest** is the fraction of the complete data reflected in the response -- analogous to consistency but again continuous rather than binary. A search engine that returns results from 99 of 100 index shards has high yield (it responded) and 99% harvest (it returned results from almost all of its data, but missed one shard).

    ```
    Traditional CAP view:        Harvest/Yield view:

    Consistent OR Available      Harvest: 0% -------- 100%
    (binary choice)              Yield:   0% -------- 100%
                                 (continuous trade-off)

    Example: Search with 100 shards, 1 shard down
      CAP says:     system is "inconsistent" (missing data)
      Harvest says: 99% harvest (99/100 shards responded)
                    100% yield  (request completed)
    ```

    This framing is often more practical than a binary CP/AP classification because real systems degrade gracefully rather than failing completely. An e-commerce search that drops results from one out of fifty index partitions is far more useful than one that returns an error page. The harvest-and-yield model lets engineers reason about partial degradation, which is how most outages actually manifest.

    ### Putting It All Together: Choosing for a Real System

    Consider designing an e-commerce platform. Different components have different requirements, and the CAP/PACELC framework helps make these decisions explicit:

    ```
    E-commerce platform CAP decisions:

    Component           CAP choice   PACELC    Reasoning
    -------------------------------------------------------------------
    Product catalog     AP           PA/EL     Stale price for 5 sec is OK
    Search index        AP           PA/EL     Missing 1 product briefly is OK
    Shopping cart       AP           PA/EL     Always accept adds (merge later)
    Inventory count     CP           PC/EL     Cannot oversell last unit
    Payment processing  CP           PC/EC     Money must be exactly right
    Order history       AP           PA/EL     Slight delay showing new order is OK
    User sessions       AP           PA/EL     Logging user out briefly is worse
    Distributed locks   CP           PC/EC     Must prevent double-processing
    ```

    No single CAP choice is correct for the entire system. The art of distributed systems design is identifying which components need which guarantees, and selecting (or configuring) the appropriate database or service for each.

---

## Key Takeaways

1. CAP is not "pick any two." Network partitions are unavoidable in distributed systems, so partition tolerance is mandatory. The real choice during a partition is between consistency and availability.

2. Consistency in CAP means linearizability -- the strongest form. Every read sees the most recent write. This is stricter than "eventual consistency" or "read-your-writes" guarantees.

3. CP systems (MongoDB, HBase, ZooKeeper, etcd) sacrifice availability during partitions to guarantee correctness. Choose CP when inconsistency has severe consequences: financial transactions, inventory counts, distributed locks.

4. AP systems (Cassandra, DynamoDB, CouchDB, Riak) sacrifice consistency during partitions to stay responsive. Choose AP when downtime is more costly than brief staleness: social feeds, shopping carts, DNS.

5. Most production systems are not purely CP or AP. They are tunable, and different parts of the same application often make different trade-offs. An e-commerce site might use AP for its product catalog and CP for its payment system.

6. PACELC extends CAP to capture the latency-consistency trade-off that exists even when no partition is occurring, which is actually the more common operating condition and the one engineers deal with on every request.

---

## Related Topics

- [Data Consistency](data-consistency.md) -- consistency models, eventual consistency, and PACELC in depth
- [Distributed System Principles](principles.md) -- foundational concepts that underpin CAP
- [Consensus Algorithms](../distributed-systems/consensus.md) -- Raft, Paxos, and how CP systems achieve agreement
- [Database Replication](../data/databases/replication.md) -- the mechanics of synchronous and asynchronous replication
