# System Design Trade-offs

Every design decision in system design is a trade-off. There is no perfect
architecture, no universally correct database, no one-size-fits-all pattern.
When you optimize for one property, you sacrifice another. The goal is not to
eliminate trade-offs but to make them deliberately, guided by requirements.

This page catalogs the ten most common trade-offs you will encounter in system
design discussions and interviews. Each includes the core tension, a scenario,
and real-world examples.

```
            The Trade-off Mindset

     "We could do X, which gives us A but costs B,
      or we could do Y, which gives us C but costs D.
      Given our requirements, we should choose..."

     Requirements ──> Identify Constraints ──> Evaluate Options
                                                     │
                                          ┌──────────┴──────────┐
                                          │                     │
                                      Option A              Option B
                                    (optimizes X,         (optimizes Y,
                                     sacrifices Y)         sacrifices X)
                                          │                     │
                                          └──────────┬──────────┘
                                                     │
                                              Pick based on
                                            business priorities
```

!!! tip "The Golden Rule"
    There is no "best" choice in system design. There is only the best choice
    **for your specific requirements, constraints, and scale**.

---

## Quick Reference

| Trade-off | When to favor Side A | When to favor Side B |
|---|---|---|
| **Consistency** vs **Availability** | Financial transactions, inventory | Social feeds, analytics |
| **Latency** vs **Throughput** | User-facing APIs, gaming | Batch ETL, log processing |
| **Read** vs **Write Optimization** | Read-heavy dashboards, catalogs | Write-heavy event logging |
| **Strong** vs **Eventual Consistency** | Banking, booking systems | Shopping carts, likes counters |
| **Simplicity** vs **Flexibility** | Early-stage startups, small teams | Large orgs, independent scaling |
| **Cost** vs **Performance** | Predictable traffic, budget constraints | Revenue-critical, latency-sensitive |
| **Accuracy** vs **Speed** | Billing, compliance reporting | Trending topics, view counters |
| **Synchronous** vs **Asynchronous** | Simple flows, immediate feedback | Resilience, decoupled services |
| **Centralized** vs **Distributed** | Small scale, strong consistency needs | High scale, fault tolerance |
| **Push** vs **Pull** | Small follower counts, real-time needs | Celebrity accounts, cost control |

---

=== "Consistency vs Availability"

    ## Consistency vs Availability

    This is the fundamental distributed systems trade-off, formalized by the
    CAP theorem. When a network partition occurs, the system must choose: reject
    requests to preserve correctness (CP), or keep serving potentially stale data
    to stay responsive (AP). For a deep dive, see [CAP Theorem](cap-theorem.md).

    ```
    Network Partition Occurs
    ┌─────────────────┐         ┌─────────────────┐
    │    Partition A   │  X──X   │    Partition B   │
    │                  │         │                  │
    │  Node 1 (v=42)  │         │  Node 2 (v=41)  │
    │  Node 3 (v=42)  │         │  Node 4 (v=41)  │
    └─────────────────┘         └─────────────────┘

    CP Choice: Node 2 & 4 refuse reads → users see errors
    AP Choice: Node 2 & 4 return v=41  → users see stale data
    ```

    | Factor | Favor Consistency (CP) | Favor Availability (AP) |
    |--------|----------------------|------------------------|
    | **Business cost of wrong data** | High (double-charging, overbooking) | Low (stale like count) |
    | **User expectation** | "Show me the truth or nothing" | "Show me something quickly" |
    | **Recovery complexity** | Simple (one source of truth) | Complex (conflict resolution) |
    | **Latency during normal ops** | Higher (quorum reads/writes) | Lower (local reads) |
    | **Example systems** | ZooKeeper, etcd, Spanner | Cassandra, DynamoDB, Riak |

    ### Real-World Scenario: Airline Seat Booking

    An airline cannot sell the same seat twice. During a network partition between
    datacenters, the booking system must reject requests rather than risk a
    double-booking. This is a textbook CP choice -- the business cost of
    inconsistency (two passengers with the same seat) far exceeds the cost of
    temporary unavailability (a user retries in 30 seconds).

    Contrast this with a social media "like" counter. If two partitions
    independently count likes and the total is briefly off by a few, no one
    notices or cares. Availability wins.

    !!! info "Beyond Binary"
        Most production systems are not purely CP or AP. They tune consistency
        per operation. DynamoDB lets you choose strong consistency for critical
        reads and eventual consistency for everything else -- on the same table.

---

=== "Latency vs Throughput"

    ## Latency vs Throughput

    Latency is the time to complete a single operation. Throughput is the number
    of operations completed per unit of time. These often conflict because
    optimizing for one can hurt the other.

    ```
    Low Latency (process immediately)
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Request │────>│ Process │────>│ Response│    1 item at a time
    └─────────┘     └─────────┘     └─────────┘    Latency: 5ms
                                                    Throughput: 200/sec

    High Throughput (batch and process)
    ┌─────────┐
    │ Req 1   │──┐
    ├─────────┤  │  ┌──────────────┐  ┌──────────┐
    │ Req 2   │──┼─>│ Batch Process│─>│ Response │  100 items at once
    ├─────────┤  │  └──────────────┘  └──────────┘  Latency: 50ms
    │ Req 3   │──┘                                   Throughput: 2000/sec
    └─────────┘
         Wait for batch to fill...
    ```

    | Scenario | Optimize For | Why |
    |----------|-------------|-----|
    | User clicks "Buy Now" | Latency | User is waiting; every ms matters |
    | Processing overnight log files | Throughput | No user waiting; cost per record matters |
    | Real-time game server | Latency | 100ms delay breaks the experience |
    | Sending bulk email notifications | Throughput | 10M emails; speed per email is irrelevant |
    | Autocomplete suggestions | Latency | Must respond within 100ms to feel instant |
    | Data warehouse ETL pipeline | Throughput | Processing TB of data; batch efficiency wins |

    ### Real-World Example: Kafka

    Apache Kafka achieves high throughput (millions of messages/sec) by batching
    writes to disk sequentially. A single message might wait a few milliseconds
    to be included in a batch, which adds latency. Kafka's `linger.ms` setting
    controls this trade-off directly: set it to 0 for lowest latency (send
    immediately), or to 50ms to accumulate larger batches for higher throughput.

    | Kafka Setting | Latency | Throughput | Use Case |
    |---|---|---|---|
    | `linger.ms=0` | ~2ms | ~100K msg/sec | Real-time alerting |
    | `linger.ms=50` | ~55ms | ~1M msg/sec | Log aggregation |
    | `linger.ms=200` | ~205ms | ~2M msg/sec | Batch analytics pipeline |

    !!! warning "The Coordination Tax"
        Distributed systems pay a coordination cost. Every cross-node
        synchronization step adds latency. The more nodes involved in a single
        request, the higher the latency -- but the system's aggregate throughput
        may increase because more nodes handle more total work.

---

=== "Read vs Write Optimization"

    ## Read vs Write Optimization

    Most systems are either read-heavy or write-heavy. Optimizing for reads
    (fast lookups, denormalized data, caches) makes writes slower and more
    complex. Optimizing for writes (append-only logs, normalized data) makes
    reads require more work.

    ```
    Read-Optimized (Denormalized)
    ┌────────────────────────────────────┐
    │ user_timeline table                │
    │ user_id │ tweet_text │ author_name │   Write: update author_name
    │         │            │ author_pic  │   in EVERY row (expensive)
    │         │            │ timestamp   │
    └────────────────────────────────────┘   Read: single query (fast)

    Write-Optimized (Normalized)
    ┌──────────────┐    ┌──────────────┐
    │ tweets       │    │ users        │
    │ tweet_id     │    │ user_id      │   Write: update author_name
    │ author_id ───┼───>│ name         │   in ONE place (fast)
    │ text         │    │ profile_pic  │
    └──────────────┘    └──────────────┘   Read: JOIN required (slower)
    ```

    | Approach | Read Cost | Write Cost | Best For |
    |----------|-----------|------------|----------|
    | **Denormalization** | Fast (single lookup) | Slow (update many copies) | Product catalogs, dashboards |
    | **Normalization** | Slow (joins needed) | Fast (update one place) | Transactional systems |
    | **Materialized Views** | Fast (precomputed) | Medium (async rebuild) | Analytics, reporting |
    | **CQRS** | Fast (read model) | Fast (write model) | High-scale systems with both |

    ### Pattern: CQRS (Command Query Responsibility Segregation)

    CQRS separates the read and write paths entirely, allowing each to be
    optimized independently.

    ```
    ┌─────────┐  write  ┌──────────────┐  events  ┌──────────────┐
    │ Client  │────────>│ Write Model  │─────────>│ Event Store  │
    └─────────┘         │ (normalized) │          └──────┬───────┘
                        └──────────────┘                 │
                                                    projection
        ┌─────────┐  read   ┌──────────────┐            │
        │ Client  │<────────│ Read Model   │<───────────┘
        └─────────┘         │(denormalized)│
                            └──────────────┘

    Write path: optimized for correctness and consistency
    Read path: optimized for query speed and flexibility
    ```

    ### Real-World Example: Amazon Product Pages

    Amazon's product catalog is extremely read-heavy (~500M page views/day vs
    ~10M updates/day, a 50:1 ratio). Amazon denormalizes product data into
    read-optimized stores and caches, accepting that a price update might take
    seconds to propagate. The write path uses a normalized transactional system
    for correctness; the read path serves from denormalized, cached views.

---

=== "Strong vs Eventual Consistency"

    ## Strong vs Eventual Consistency

    Strong consistency guarantees that after a write completes, every subsequent
    read returns that write's value. Eventual consistency guarantees only that
    if no new writes occur, all replicas will eventually converge to the same
    value. The gap between "eventually" and "immediately" is where the trade-off
    lives. See [Data Consistency](data-consistency.md) for the full spectrum.

    ```
    Strong Consistency (Synchronous Replication)
    ┌────────┐ write  ┌────────┐ replicate ┌────────┐ replicate ┌────────┐
    │ Client │───────>│ Node 1 │──────────>│ Node 2 │──────────>│ Node 3 │
    └────────┘        └────────┘           └────────┘           └────────┘
                           │  wait for all acks...  │                │
                           │<───────── ack ─────────│                │
                           │<────────────────────── ack ────────────│
                           │
                      ack to client (all nodes confirmed)
                      Latency: ~50-200ms

    Eventual Consistency (Asynchronous Replication)
    ┌────────┐ write  ┌────────┐
    │ Client │───────>│ Node 1 │──── ack to client (immediately)
    └────────┘        └────────┘     Latency: ~5ms
                           │
                      background replication (seconds later)
                           │
                      ┌────────┐           ┌────────┐
                      │ Node 2 │           │ Node 3 │
                      └────────┘           └────────┘
    ```

    | Factor | Strong Consistency | Eventual Consistency |
    |--------|-------------------|---------------------|
    | **Read correctness** | Always current | May be stale |
    | **Write latency** | Higher (wait for replicas) | Lower (ack immediately) |
    | **Availability during partitions** | Reduced (needs quorum) | Full (any node serves) |
    | **Conflict resolution** | Not needed | Required (last-write-wins, CRDTs) |
    | **Developer complexity** | Low (intuitive behavior) | Higher (handle stale reads) |

    ### When Each Makes Sense

    | Domain | Consistency Choice | Reasoning |
    |--------|-------------------|-----------|
    | Bank account balance | Strong | Overdraft costs real money |
    | Hotel room booking | Strong | Double-booking loses customers |
    | Social media likes | Eventual | Off by 3 likes is invisible |
    | Shopping cart contents | Eventual | Merge on checkout; no real harm |
    | Inventory count (last few items) | Strong | Overselling is costly |
    | Inventory count (1000s in stock) | Eventual | Small inaccuracy is harmless |

    !!! example "Real-World: DynamoDB"
        DynamoDB defaults to eventually consistent reads (~5ms) but offers
        strongly consistent reads (~15ms) at double the read capacity cost.
        Most applications use eventual consistency for 90%+ of reads and
        reserve strong consistency for operations where correctness is
        critical, like checking a unique constraint before inserting.

---

=== "Simplicity vs Flexibility"

    ## Simplicity vs Flexibility

    A monolithic architecture is simpler to build, deploy, debug, and reason
    about. A microservices architecture gives each team independence, allows
    services to scale independently, and lets teams choose different tech stacks.
    The flexibility comes at the cost of distributed systems complexity: network
    failures, distributed tracing, service discovery, data consistency across
    service boundaries.

    ```
    Monolith                              Microservices
    ┌──────────────────────┐         ┌────────┐ ┌────────┐ ┌────────┐
    │   Single Deployable  │         │ Users  │ │ Orders │ │ Search │
    │                      │         │Service │ │Service │ │Service │
    │  Users + Orders +    │         └───┬────┘ └───┬────┘ └───┬────┘
    │  Search + Payments   │             │          │          │
    │                      │         ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
    │  One DB, one deploy  │         │ DB     │ │ DB     │ │ DB     │
    └──────────────────────┘         └────────┘ └────────┘ └────────┘

    Deploy: 1 artifact                Deploy: N artifacts
    Debug: grep one log               Debug: distributed tracing
    Scale: scale everything           Scale: scale what's hot
    Team: everyone in one codebase    Team: independent ownership
    ```

    | Factor | Monolith (Simplicity) | Microservices (Flexibility) |
    |--------|----------------------|---------------------------|
    | **Team size** | <10 engineers | 50+ engineers, multiple teams |
    | **Deployment speed** | Minutes (one artifact) | Minutes per service, but coordination overhead |
    | **Debugging** | Stack traces, local logs | Distributed tracing (Jaeger, Zipkin) |
    | **Data consistency** | ACID transactions | Saga pattern, eventual consistency |
    | **Tech diversity** | One stack | Team chooses best tool per service |
    | **Scaling** | Scale entire app | Scale bottleneck service only |
    | **Initial velocity** | Fast (no infra overhead) | Slower (service mesh, CI/CD per service) |

    ### Real-World Example: Shopify

    Shopify runs one of the largest Ruby on Rails monoliths in the world. Rather
    than splitting into microservices, they invested in modular monolith
    architecture -- organizing code into well-defined components with enforced
    boundaries, while keeping a single deployable. This gives them team
    independence without distributed systems complexity. They handle Black Friday
    traffic (80K+ requests/sec at peak) with this approach.

    !!! tip "When to Split"
        Start with a monolith. Split into services only when you have a
        concrete problem that a monolith cannot solve: independent scaling
        needs, team ownership boundaries, or deployment coupling that slows
        teams down. "We might need it someday" is not a reason.

---

=== "Cost vs Performance"

    ## Cost vs Performance

    Higher performance costs more: faster machines, more replicas, bigger caches,
    premium cloud tiers. The trade-off is between spending money on
    infrastructure versus accepting lower performance. The goal is finding the
    point where additional spending yields diminishing returns.

    ```
    Performance vs Cost Curve

    Perf │
    100% │                          ·····················
         │                    ·····
     80% │               ····
         │           ···
     60% │        ··
         │      ··
     40% │    ··     <-- "sweet spot": 80% of performance
         │   ·           for 30% of max cost
     20% │  ·
         │ ·
      0% │·
         └──────────────────────────────────────────────
          $0     $100    $500   $1000   $2000   $5000
                           Monthly Cost
    ```

    | Strategy | Cost | Performance | Risk |
    |----------|------|-------------|------|
    | **Over-provision** | High (always paying for peak) | Consistently high | Wasted spend during off-peak |
    | **Auto-scale** | Medium (pay for what you use) | High during scale-up, lag during spike | Cold start latency |
    | **Under-provision** | Low | Degrades under load | User-facing failures at peak |
    | **Reserved instances** | Low (committed spend) | Consistent | Locked in if needs change |
    | **Spot/preemptible** | Very low (60-90% discount) | High when available | Interruptions at any time |

    ### Real-World Example: Netflix

    Netflix uses reserved instances for baseline load, auto-scaling for traffic
    spikes, and spot instances (up to 90% cheaper) for video transcoding. If a
    spot instance is reclaimed, they restart that encoding job on another --
    possible because transcoding is idempotent and stateless.

    | Netflix Workload | Strategy | Why |
    |---|---|---|
    | Streaming API servers | Reserved + auto-scale | Predictable base, spiky peaks |
    | Video encoding | Spot instances | Stateless, retryable, cost-sensitive |
    | Control plane (Zuul, Eureka) | Reserved | Must be always available |

    !!! warning "The Premature Optimization Tax"
        Do not over-optimize cost for a system that does not exist yet.
        Engineering time spent on auto-scaling before product-market fit
        has a higher opportunity cost than the cloud bill.

---

=== "Accuracy vs Speed"

    ## Accuracy vs Speed

    Exact answers require reading and processing all data. Approximate answers
    use probabilistic data structures and sampling to give "close enough" results
    in a fraction of the time and memory. When you have billions of events,
    exact counting may be impossible in real time.

    ```
    Exact Count                         Approximate Count
    ┌─────────────────────┐             ┌─────────────────────┐
    │ Count distinct users│             │ HyperLogLog         │
    │                     │             │                     │
    │ Store every user ID │             │ Store 12 KB of      │
    │ in a Set            │             │ register buckets     │
    │                     │             │                     │
    │ Memory: O(n)        │             │ Memory: O(1)        │
    │ 1B users = ~8 GB    │             │ 1B users = 12 KB    │
    │ Time: scan all data │             │ Time: constant      │
    │ Error: 0%           │             │ Error: ~0.81%       │
    └─────────────────────┘             └─────────────────────┘
    ```

    | Data Structure | What It Does | Accuracy | Memory | Use Case |
    |---|---|---|---|---|
    | **HyperLogLog** | Count distinct elements | ~99.2% (0.81% error) | 12 KB fixed | Unique visitors, distinct IPs |
    | **Bloom Filter** | "Is X in the set?" | No false negatives, ~1% false positives | Bits per element | Cache lookups, spam filtering |
    | **Count-Min Sketch** | Frequency of element X | Over-counts by small margin | Configurable | Top-K queries, rate limiting |
    | **t-digest** | Percentile estimates | ~0.1% at extremes | ~10 KB | P99 latency monitoring |

    ### Real-World Example: Redis at Scale

    Redis provides built-in HyperLogLog support. A single HyperLogLog key uses
    only 12 KB of memory regardless of how many elements are added. Companies
    use this to count unique visitors across billions of page views.

    | Approach | Memory for 1B Unique Users | Query Time | Error |
    |----------|---------------------------|------------|-------|
    | Exact (SET) | ~8 GB | O(1) lookup, O(n) count | 0% |
    | HyperLogLog | 12 KB | O(1) | 0.81% |
    | Bloom Filter (membership) | ~1.2 GB at 1% FP | O(k) | 1% false positive |

    !!! info "When Approximate Is Wrong"
        Never use approximate counting for billing, financial reporting,
        compliance audits, or any domain where "close enough" has legal or
        financial consequences. A 0.81% error on 1 billion transactions is
        8.1 million miscounted transactions.

---

=== "Sync vs Async"

    ## Synchronous vs Asynchronous

    In synchronous communication, the caller waits for the response before
    proceeding. In asynchronous communication, the caller sends the request and
    continues without waiting. Sync is simpler but creates tight coupling and
    cascading failures. Async is more resilient but harder to reason about.

    ```
    Synchronous                         Asynchronous
    ┌────────┐     ┌────────┐          ┌────────┐     ┌────────┐
    │Service │────>│Service │          │Service │────>│ Queue  │
    │   A    │     │   B    │          │   A    │     │        │
    │        │     │        │          │        │     └───┬────┘
    │ waiting│<────│response│          │ done!  │         │
    │  ...   │     │        │          └────────┘    ┌────┴───┐
    └────────┘     └────────┘                        │Service │
                                                     │   B    │
    If B is slow, A is slow.                         │(later) │
    If B is down, A fails.                           └────────┘
                                        If B is slow, A doesn't care.
                                        If B is down, message waits.
    ```

    | Factor | Synchronous | Asynchronous |
    |--------|------------|--------------|
    | **Complexity** | Low (request-response) | Higher (queues, retries, idempotency) |
    | **Coupling** | Tight (A depends on B) | Loose (A depends on queue) |
    | **Failure propagation** | Cascading (B failure breaks A) | Isolated (B failure is invisible to A) |
    | **Latency** | Sum of all service latencies | Only the write-to-queue latency |
    | **Debugging** | Easy (synchronous stack trace) | Hard (trace across queue boundaries) |
    | **Ordering** | Natural (sequential calls) | Must be engineered (partitioned queues) |

    ### When Each Makes Sense

    | Operation | Pattern | Why |
    |-----------|---------|-----|
    | User login | Sync | User must wait for auth result |
    | Send welcome email | Async | User doesn't need to wait for email delivery |
    | Payment processing | Sync (then async) | Authorize sync, settle async |
    | Image thumbnail generation | Async | Can take seconds; show placeholder |
    | Real-time search | Sync | User expects immediate results |
    | Order fulfillment notification | Async | Fire and forget to notification service |

    !!! example "Real-World: Slack"
        Slack uses asynchronous message delivery with eventual consistency.
        When you send a message, Slack acknowledges the send immediately
        (your message appears in your client) and asynchronously propagates
        it to all other channel members. In rare cases, two users may see
        messages in slightly different order for a brief moment before
        the ordering converges.

---

=== "Centralized vs Distributed"

    ## Centralized vs Distributed

    A centralized system has a single coordinating node that makes decisions.
    A distributed system spreads decision-making across multiple nodes. Centralized
    is simpler and provides strong consistency naturally, but creates a single
    point of failure and a scaling bottleneck. Distributed removes the bottleneck
    but introduces coordination complexity.

    ```
    Centralized                          Distributed
    ┌──────────────┐                    ┌────────┐
    │   Leader     │                    │ Node A │<──────>┌────────┐
    │  (single     │                    └────────┘        │ Node B │
    │   point of   │                         ^            └────────┘
    │   failure)   │                         |                 ^
    └──────┬───────┘                         v                 |
           │                            ┌────────┐             v
     ┌─────┼─────┐                      │ Node C │<──────>┌────────┐
     v     v     v                      └────────┘        │ Node D │
    ┌──┐  ┌──┐  ┌──┐                                      └────────┘
    │W1│  │W2│  │W3│                   No single leader
    └──┘  └──┘  └──┘                   All nodes coordinate
    Simple, but leader is SPOF         Complex, but no SPOF
    ```

    | Factor | Centralized | Distributed |
    |--------|------------|-------------|
    | **Consistency** | Easy (single source of truth) | Hard (consensus protocols needed) |
    | **Fault tolerance** | Low (leader failure = outage) | High (survive node failures) |
    | **Latency** | Low for co-located clients | Varies (depends on node proximity) |
    | **Scaling ceiling** | Limited by leader capacity | Near-unlimited horizontal scale |
    | **Operational complexity** | Low | High (monitoring, debugging, deploys) |
    | **Examples** | Single-leader DB, DNS root | Cassandra, CockroachDB, DynamoDB |

    ### Real-World Example: DNS

    DNS is a hybrid. The root zone is centralized (13 root server clusters),
    but resolution is massively distributed through caching resolvers worldwide.
    A root server failure does not take down the internet because resolvers
    cache responses for hours or days.

    !!! note "The Coordination Cost"
        Distributed systems must solve problems centralized systems avoid:
        leader election, consensus (Paxos, Raft), conflict resolution, and
        clock synchronization. Do not distribute a system that fits on one
        machine.

---

=== "Push vs Pull"

    ## Push vs Pull

    In a push model (fan-out on write), the system precomputes and delivers
    results to recipients when data changes. In a pull model (fan-out on read),
    the system computes results on demand when a recipient requests them. The
    choice depends on the ratio of writes to reads and the distribution of
    follower counts.

    ```
    Push (Fan-out on Write)
    ┌─────────┐  tweet   ┌──────────────┐  write to each
    │  User   │─────────>│ Fan-out      │  follower's timeline
    │  posts  │          │ Service      │──────┐──────┐──────┐
    └─────────┘          └──────────────┘      v      v      v
                                            ┌─────┐┌─────┐┌─────┐
                                            │TL-A ││TL-B ││TL-C │
                                            └─────┘└─────┘└─────┘
    Write cost: O(followers)               1000 followers = 1000 writes
    Read cost: O(1)                        Reading timeline = 1 read

    Pull (Fan-out on Read)
    ┌─────────┐  tweet   ┌──────────────┐
    │  User   │─────────>│ Tweet Store  │  store once
    │  posts  │          └──────────────┘
    └─────────┘

    ┌─────────┐  read    ┌──────────────┐  fetch from each
    │ Reader  │─────────>│ Timeline     │  followed user
    │ opens   │          │ Service      │──────┐──────┐──────┐
    │ feed    │          └──────────────┘      v      v      v
    └─────────┘                             ┌─────┐┌─────┐┌─────┐
                                            │User1││User2││User3│
                                            └─────┘└─────┘└─────┘
    Write cost: O(1)                       Store tweet once
    Read cost: O(following)                Following 500 users = 500 reads
    ```

    | Factor | Push (Fan-out on Write) | Pull (Fan-out on Read) |
    |--------|------------------------|----------------------|
    | **Write cost** | High (copy to all followers) | Low (store once) |
    | **Read cost** | Low (precomputed timeline) | High (aggregate on demand) |
    | **Staleness** | None (always current) | Depends on cache TTL |
    | **Celebrity problem** | Severe (1 tweet = 100M writes) | None |
    | **Storage** | High (duplicated data) | Low (single copy) |
    | **Latency to read** | Very low | Higher (compute on demand) |

    ### Real-World Example: Twitter's Hybrid Approach

    Twitter famously uses a hybrid push-pull model. Most users' tweets are
    pushed to their followers' timelines at write time (fan-out on write). But
    for users with millions of followers (celebrities, politicians), the tweet
    is stored once and merged into followers' timelines at read time (fan-out
    on read). This avoids the "celebrity problem" where one tweet would trigger
    hundreds of millions of writes.

    ```
    Twitter's Hybrid Model

    Regular user (5K followers) ──> Push to 5K timelines
                                    (fast, manageable fan-out)

    Celebrity (50M followers)   ──> Store tweet once
                                    │
    Reader opens timeline ──> Merge: precomputed timeline
                                    + latest tweets from
                                      followed celebrities
                                    (computed at read time)
    ```

    | User Type | Follower Count | Strategy | Why |
    |-----------|---------------|----------|-----|
    | Regular user | <50K | Push | Fan-out cost is small |
    | Verified/celebrity | >50K | Pull | 50K+ writes per tweet is too expensive |
    | Timeline read | -- | Merge | Combine precomputed + on-demand |

    !!! tip "The Threshold"
        The exact follower threshold for switching from push to pull depends
        on your infrastructure. Twitter's reported threshold was around 50K
        followers, though the exact number has changed over time as their
        infrastructure evolved.

---

## How to Discuss Trade-offs in Interviews

The ability to reason about trade-offs is the single most important skill in
a system design interview. Interviewers are not looking for the "right" answer
-- they are looking for structured reasoning about **why** you would choose one
approach over another.

### The Trade-off Framework

Use this structure every time you make a design decision:

```
"We could do [Option A], which gives us [Benefit A] but costs us [Drawback A].
 Alternatively, we could do [Option B], which gives us [Benefit B] but costs
 us [Drawback B]. Given that our requirements prioritize [Requirement], I'd
 recommend [Choice] because [Reasoning]."
```

!!! example "Example: Design a URL Shortener"
    "We could use PostgreSQL, which gives us ACID transactions, but read
    latency at scale would be 5-10ms. Alternatively, Redis gives us
    sub-millisecond reads but we'd handle persistence ourselves. Given
    that URL redirection is extremely read-heavy and latency-sensitive,
    I'd recommend Redis with a PostgreSQL backing store for durability."

### Interview Anti-Patterns

| Anti-Pattern | Why It's Bad | Better Approach |
|---|---|---|
| "We should use microservices" | No justification | "Given our team size of 5, a monolith is simpler..." |
| "NoSQL is better than SQL" | No context | "Our data is hierarchical with no joins, so a document store..." |
| "We need Kafka" | Jumping to implementation | "We need async processing because..." |
| "Let's add a cache" | Without sizing | "Reads are 100:1 vs writes, so a cache would absorb 99%..." |

### Requirements-First Approach

Always anchor trade-off decisions to requirements:

```
Requirements for a Banking System:
  ├── Consistency: CRITICAL (cannot show wrong balance)
  ├── Latency: moderate (users tolerate 1-2 sec for transfers)
  ├── Availability: high (but can sacrifice for correctness)
  └── Scale: millions of accounts, thousands of TPS

These requirements tell us:
  ├── Choose strong consistency over eventual
  ├── Choose CP over AP during partitions
  ├── Synchronous replication is acceptable (latency budget allows it)
  └── Relational DB with ACID (not DynamoDB with eventual consistency)
```

---

## Trade-off Decision Matrix

Use this template when evaluating design options. Score each factor 1-5 based
on how well the option satisfies it, weighted by how important that factor is
to your specific requirements.

```
┌──────────────────┬────────┬──────────┬──────────┬──────────┐
│ Factor           │ Weight │ Option A │ Option B │ Option C │
├──────────────────┼────────┼──────────┼──────────┼──────────┤
│ Consistency      │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Availability     │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Latency          │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Throughput       │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Cost             │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Simplicity       │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
│ Team familiarity │  [1-5] │   [1-5]  │   [1-5]  │   [1-5]  │
├──────────────────┼────────┼──────────┼──────────┼──────────┤
│ Weighted Total   │        │   [sum]  │   [sum]  │   [sum]  │
└──────────────────┴────────┴──────────┴──────────┴──────────┘

Weighted Score = SUM(Weight_i * Score_i) for each factor
```

!!! note "The Matrix Is a Starting Point"
    Do not treat the matrix as a formula that produces the "correct" answer.
    It is a tool for structuring your thinking and making implicit priorities
    explicit. The conversation about weights is often more valuable than the
    final scores.

---

## Real-World Case Studies

### Twitter: Push vs Pull Timeline

Twitter's timeline evolution is the canonical example of trade-off thinking
at scale.

| Phase | Approach | Problem |
|-------|----------|---------|
| **v1 (2006)** | Pull: query all followed users on each load | Slow at scale; timeline load = hundreds of DB queries |
| **v2 (2012)** | Push: fan-out on write to Redis timelines | Celebrity tweets caused write storms (Lady Gaga = 30M writes) |
| **v3 (current)** | Hybrid: push for regular users, pull for celebrities | Balances write cost with read latency |

The key insight: neither pure push nor pure pull works at Twitter's scale.
The distribution of follower counts (power law) means a one-size-fits-all
approach will always break for some segment of users.

### DynamoDB: Tunable Consistency for Lower Latency

DynamoDB lets developers choose consistency per read operation, not per table.
This enables a pattern where the same application uses both:

| Read Type | Consistency | Latency | Cost (RCU) | Use Case |
|-----------|------------|---------|------------|----------|
| Default | Eventual | ~5ms | 1 RCU per 8KB | Product browsing, feed loading |
| Opt-in | Strong | ~15ms | 2 RCU per 4KB | Checking inventory before purchase |

This per-operation choice lets applications be precise about where they need
correctness versus speed, rather than paying the consistency tax everywhere.

### Slack: Eventual Consistency for Messages

Slack chose eventual consistency for message delivery. When you send a
message, it is acknowledged immediately by the nearest server (5ms) and
asynchronously replicated to other members' connections (50-200ms). In rare
cases, two messages may appear briefly out of order before converging.

Slack's reasoning: perceived responsiveness matters more than strict ordering
in chat. A 5ms ack feels instant. A 200ms synchronous replication wait would
feel sluggish on every single message. The rare ordering glitch is a trivial
cost compared to always-fast sending.

```
Slack's Consistency Choice

User sends message
    │
    v
┌───────────────────┐
│ Nearest Server    │──> Ack to sender (5ms)
│ (writes locally)  │
└───────┬───────────┘
        │
   async replication (50-200ms)
        │
   ┌────┴─────┐──────┐
   v          v      v
┌──────┐  ┌──────┐  ┌──────┐
│ US-E │  │ EU   │  │ APAC │
│Server│  │Server│  │Server│
└──────┘  └──────┘  └──────┘
```

---

## Summary

Every trade-off in this page follows the same pattern: you gain something
valuable by giving up something else valuable. The skill is not memorizing
which side is "better" but understanding the conditions under which each side
wins.

| When You Hear... | Think About... |
|---|---|
| "We need strong consistency" | What is the cost of a stale read in dollars? |
| "Let's use microservices" | Do we have the team size and operational maturity? |
| "Add a cache" | What is the read:write ratio? What about invalidation? |
| "Use async messaging" | Can we handle idempotency and out-of-order delivery? |
| "Scale horizontally" | What state needs to be shared or partitioned? |
| "Use a distributed database" | Does our data fit on one machine? |

!!! tip "The Ultimate Interview Signal"
    The strongest signal in a system design interview is not knowing the
    "right" answer. It is saying: "It depends on our requirements. If we
    need X, I'd choose A because... but if we need Y, I'd choose B
    because..." That demonstrates engineering judgment, which is what the
    interview is actually testing.
