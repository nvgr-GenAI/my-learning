# Consensus Algorithms

Getting multiple machines to agree on something sounds simple until you realize that
networks drop messages, clocks drift, and servers crash at the worst possible moment.
Consensus is the foundational problem of distributed systems: how do N nodes agree on
a single value when any of them might fail, and the network between them is unreliable?

Every time Kubernetes schedules a pod, every time a distributed database commits a
transaction, and every time a service registry updates its membership, consensus is
happening behind the scenes. The algorithms discussed here -- Raft, Paxos, ZAB, and
others -- are the machinery that makes modern cloud infrastructure possible.

---

=== "The Problem"

    ## Why Consensus Is Hard

    Imagine a database cluster of five nodes. The leader accepts a write, but before it
    can replicate to all followers, the network partitions into two groups: {A, B} and
    {C, D, E}. Both groups believe they might be authoritative. If both sides start
    accepting writes independently, you get **split brain** -- conflicting state that may
    be impossible to reconcile after the partition heals.

    ```
    SPLIT-BRAIN SCENARIO

    Before partition:
    ┌─────────────────────────────────────────────┐
    │  A(leader)  B  C  D  E                      │
    │  All nodes agree: x = 42                    │
    └─────────────────────────────────────────────┘

    Network partition occurs:
    ┌──────────────┐     ┌────────────────────────┐
    │  A(leader) B │     │  C   D   E             │
    │  x = 42      │ ╳╳╳ │  x = 42               │
    └──────────────┘     └────────────────────────┘

    Without consensus, both sides accept writes:
    ┌──────────────┐     ┌────────────────────────┐
    │  A(leader) B │     │  C(new leader?) D  E   │
    │  x = 100     │ ╳╳╳ │  x = 200              │
    └──────────────┘     └────────────────────────┘

    Partition heals -- which value of x is correct?
    Neither side knows. Data is corrupted.
    ```

    Consensus algorithms solve this by requiring a **majority quorum** (more than half
    the nodes) to agree before any decision is final. In a five-node cluster, that means
    three nodes. Since any two majorities must overlap by at least one node, it is
    impossible for two conflicting decisions to both achieve quorum. The side with only
    two nodes ({A, B}) cannot form a majority and must stop accepting writes, while the
    side with three nodes ({C, D, E}) can safely elect a new leader and continue.

    ### The FLP Impossibility Result

    In 1985, Fischer, Lynch, and Paterson proved that no deterministic consensus
    algorithm can guarantee termination in an asynchronous system where even a single
    node may crash. This sounds devastating, but practical systems work around it using
    randomized timeouts (Raft uses election timeouts between 150-300ms) and partial
    synchrony assumptions. The result does not say consensus is impossible -- it says you
    cannot guarantee it completes in bounded time in the worst case.

    ### Byzantine vs Crash Failures

    Most consensus algorithms in mainstream infrastructure (Raft, Paxos, ZAB) handle
    **crash failures** -- a node either works correctly or stops entirely. **Byzantine
    failures**, where a node actively sends wrong or malicious data, require far more
    complex protocols like PBFT (Practical Byzantine Fault Tolerance), which needs 3f+1
    nodes to tolerate f failures compared to 2f+1 for crash failures. Byzantine
    consensus is mainly used in blockchain systems and high-security environments. For
    typical data center deployments, crash-fault-tolerant algorithms are sufficient.

    | Failure Type | Tolerance Formula | Nodes for f=1 | Primary Use |
    |---|---|---|---|
    | Crash failure | 2f + 1 | 3 | Databases, service discovery |
    | Byzantine failure | 3f + 1 | 4 | Blockchains, adversarial environments |

=== "Raft"

    ## Raft: Consensus Made Understandable

    Raft was designed in 2014 by Diego Ongaro and John Ousterhout at Stanford with a
    single explicit goal: be as easy to understand as possible while providing the same
    guarantees as Paxos. Their user study showed that students learning Raft scored
    significantly higher on comprehension tests than those learning Paxos. This
    understandability is not just an academic nicety -- it directly translates to fewer
    bugs in production implementations.

    Raft decomposes consensus into three relatively independent subproblems: leader
    election, log replication, and safety.

    ### Leader Election

    Every Raft node starts as a **follower**. Followers listen for heartbeats from a
    leader. If no heartbeat arrives within a randomized timeout (typically 150-300ms),
    the follower becomes a **candidate**, increments its term number, votes for itself,
    and asks other nodes for votes. A candidate that receives votes from a majority
    becomes the **leader**. The randomized timeout is the key to avoiding split votes --
    nodes will not all time out simultaneously.

    ```
    RAFT NODE STATE TRANSITIONS

    ┌──────────┐  election timeout  ┌──────────┐  wins majority  ┌──────────┐
    │ Follower ├───────────────────>│Candidate ├────────────────>│  Leader  │
    └────┬─────┘                    └────┬─────┘                 └────┬─────┘
         ^                               │                            │
         │      discovers higher term    │     discovers higher term  │
         └───────────────────────────────┴────────────────────────────┘
    ```

    A critical rule: each node votes for at most one candidate per term, and it only
    votes for a candidate whose log is at least as up-to-date as its own. This prevents
    a node with a stale log from becoming leader and overwriting committed entries.

    ```
    LEADER ELECTION EXAMPLE (5-node cluster, term 4)

    Time 0: Leader (Node 1) crashes
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │  XX  │ │  F   │ │  F   │ │  F   │ │  F   │
    │crash │ │ t=3  │ │ t=3  │ │ t=3  │ │ t=3  │
    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

    Time 1: Node 3 times out first (randomized), becomes candidate
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │  XX  │ │  F   │ │  C   │ │  F   │ │  F   │
    │      │ │ t=3  │ │ t=4  │ │ t=3  │ │ t=3  │
    └──────┘ └──────┘ └──┬───┘ └──────┘ └──────┘
                         │ RequestVote(term=4)
                    ┌────┼─────┬─────────┐
                    v    v     v         v
                   Yes  self  Yes       Yes
                        vote
    Result: Node 3 gets 4 votes (majority=3), becomes leader for term 4
    ```

    ### Log Replication

    Once elected, the leader handles all client requests. For each request, it appends
    an entry to its local log, then sends AppendEntries RPCs to every follower. When a
    majority of nodes have written the entry to their logs, the leader considers it
    **committed** and applies it to the state machine. The leader then notifies followers
    that the entry is committed on the next heartbeat.

    ```
    LOG REPLICATION FLOW

    Client: "SET x = 7"
         │
         v
    ┌─────────┐  AppendEntries   ┌──────────┐
    │ Leader  ├─────────────────>│Follower 1│ ACK
    │ log:[7] ├─────────────────>│Follower 2│ ACK
    │         ├──────── X ──────>│Follower 3│ (network drop)
    │         ├─────────────────>│Follower 4│ ACK
    └─────────┘                  └──────────┘

    3 ACKs + leader = 4 out of 5 --> majority reached
    Leader commits entry, applies SET x = 7
    Responds success to client
    Follower 3 catches up on next heartbeat
    ```

    This mechanism means committed entries survive leader failures. Any node elected as
    the new leader is guaranteed to have all committed entries in its log, because it
    needed a majority vote, and a majority of nodes have the committed entry.

    ### Who Uses Raft

    Raft powers some of the most critical infrastructure in modern systems. **etcd**, the
    configuration store behind every Kubernetes cluster, uses Raft to replicate cluster
    state across typically 3 or 5 nodes. At scale, a Kubernetes deployment at companies
    like Shopify (6,000+ nodes) depends on etcd's Raft consensus for every pod
    scheduling decision and config change.

    **HashiCorp Consul** uses Raft for service discovery and health checking across data
    centers. Consul deployments at companies like Stripe manage thousands of services
    across multiple availability zones. **CockroachDB**, used by companies like Netflix
    and Bose, employs Raft at the range level -- each range (typically 512 MB of data)
    has its own Raft group, meaning a large cluster runs thousands of concurrent Raft
    instances. **TiKV**, the storage layer behind TiDB, uses the same approach to support
    multi-terabyte transactional workloads at companies like PingCAP and Zhihu.

=== "Paxos and Others"

    ## Paxos: The Original Consensus Algorithm

    Leslie Lamport published Paxos in 1989 (though it was not widely understood until
    its re-publication in 1998 and a follow-up "Paxos Made Simple" paper in 2001). The
    core idea is elegant: a proposer sends a numbered proposal to acceptors, who promise
    not to accept lower-numbered proposals. If a majority of acceptors promise, the
    proposer can ask them to accept the value. Once a majority accepts, consensus is
    reached.

    ```
    BASIC PAXOS (single-decree)

    Proposer               Acceptors (5 nodes)
       │                   │  │  │  │  │
       │──Prepare(n=1)────>│  │  │  │  │
       │<──Promise(n=1)────│  │  │  │  │
       │──Prepare(n=1)──────->│  │  │  │
       │<──Promise(n=1)───────│  │  │  │
       │──Prepare(n=1)────────-->│  │  │
       │<──Promise(n=1)─────────│  │  │
       │                   3 promises = majority
       │──Accept(n=1,v=X)─>│  │  │  │  │
       │<──Accepted────────│  │  │  │  │
       │──Accept(n=1,v=X)────>│  │  │  │
       │<──Accepted───────────│  │  │  │
       │──Accept(n=1,v=X)────────>│  │  │
       │<──Accepted───────────────│  │  │
       │                   3 accepts = consensus on value X
    ```

    The fundamental challenge with basic Paxos is that it decides a single value. Real
    systems need to decide a sequence of values (a replicated log), which requires
    **Multi-Paxos**. Multi-Paxos elects a distinguished proposer (effectively a leader)
    who can skip the Prepare phase for subsequent proposals, reducing each decision to a
    single round trip. However, Lamport's paper left many practical details unspecified,
    leading to widely varying implementations that are difficult to compare or verify.

    Lamport famously claimed Paxos is simple. The distributed systems community
    overwhelmingly disagrees -- Google's Chubby team reported that "there are significant
    gaps between the description of the Paxos algorithm and the needs of a real-world
    system" and that their implementation required substantial engineering beyond the
    paper.

    ### ZAB: ZooKeeper Atomic Broadcast

    Apache ZooKeeper uses ZAB (ZooKeeper Atomic Broadcast) rather than Paxos or Raft.
    ZAB is optimized for the primary-backup pattern: all writes go through a single
    leader, and state changes are broadcast to followers in strict order. Unlike Paxos,
    ZAB guarantees that if a leader proposes changes A then B, every node processes them
    in that order.

    ZooKeeper is used at enormous scale. LinkedIn runs ZooKeeper ensembles managing
    configuration for over 1,000 microservices. Yahoo (where ZooKeeper originated)
    deployed it across thousands of machines. Kafka relied on ZooKeeper for broker
    coordination and partition leader election through version 3.x (KRaft, based on Raft,
    replaces it in Kafka 4.0+).

    ### Viewstamped Replication

    Viewstamped Replication (VR), designed by Brian Oki and Barbara Liskov in 1988,
    predates both Paxos's publication and Raft. It uses a leader-based approach with
    "view changes" when the leader fails -- conceptually similar to Raft's term changes.
    VR is not widely deployed in modern systems but is historically significant as one of
    the first practical consensus protocols.

    ### When to Use Which

    | Algorithm | Best For | Real-World Systems | Key Trade-Off |
    |---|---|---|---|
    | Raft | New projects needing consensus | etcd, Consul, CockroachDB, TiKV | Understandable but leader bottleneck |
    | Multi-Paxos | Custom high-performance systems | Google Chubby, Spanner | Flexible but hard to implement |
    | ZAB | Ordered broadcast, config mgmt | Apache ZooKeeper, Kafka (pre-4.0) | Strong ordering but ZK-specific |
    | PBFT | Byzantine fault tolerance | Hyperledger Fabric | Secure but O(n^2) message cost |

=== "Practical Usage"

    ## How Consensus Shows Up in Production

    Most engineers never implement a consensus algorithm directly. Instead, they use
    systems built on top of one. Understanding what these systems provide -- and their
    limits -- is more practically useful than knowing the algorithm internals.

    ### Leader Election

    When you have a distributed service where only one instance should perform a task
    (e.g., cron-job scheduling, queue processing), you need leader election. The typical
    pattern is to use a consensus-backed store to create an ephemeral, time-limited key.
    The node that successfully creates it is the leader. If the leader crashes, its
    session expires, and another node claims leadership.

    ```
    LEADER ELECTION VIA etcd (conceptual)

    Node A: PUT /leader = "node-a" (with lease TTL=10s)   --> SUCCESS (leader)
    Node B: PUT /leader = "node-b" (with lease TTL=10s)   --> FAIL (key exists)
    Node C: PUT /leader = "node-c" (with lease TTL=10s)   --> FAIL (key exists)

    Node A must renew lease every ~5s to stay leader.
    If Node A crashes, lease expires after 10s.
    Node B or C retry and one becomes the new leader.
    ```

    **Google Chubby** is the canonical example. Chubby is a distributed lock service
    built on Multi-Paxos, used internally at Google for leader election in GFS (choosing
    a master), BigTable (tablet server coordination), and MapReduce (master election).
    Chubby typically runs as a five-node cell and handles thousands of clients per cell.
    It was so critical that a Chubby outage at Google could cascade into failures across
    dozens of dependent services.

    ### Distributed Locks

    Distributed locks extend the leader election pattern. A service acquires a lock by
    writing a key with a fence token (monotonically increasing number). Any downstream
    service checks that the fence token is higher than the last one it saw, preventing
    stale lock holders from making changes after their lock has expired and been
    re-acquired. ZooKeeper's sequential ephemeral znodes provide this natively.

    ### Configuration Management and Service Discovery

    Consul, backed by Raft, stores service definitions and health check results. When a
    service instance registers or deregisters, the change is replicated through Raft
    consensus, ensuring all Consul agents have a consistent view. At HashiCorp's reported
    scale, Consul clusters manage tens of thousands of service instances with sub-second
    convergence on membership changes.

    etcd serves a similar role for Kubernetes. Every API object (pods, services,
    deployments) is stored in etcd. The Kubernetes API server is essentially a frontend
    for etcd's consensus-replicated key-value store. Production clusters typically run
    etcd with 3 or 5 nodes, and etcd recommends keeping the data size under 8 GB for
    optimal performance.

    ### Comparison of Consensus-Backed Systems

    | System | Algorithm | Typical Cluster Size | Max Recommended Data | Notable Users |
    |---|---|---|---|---|
    | etcd | Raft | 3-5 nodes | 8 GB | Kubernetes, CoreDNS |
    | ZooKeeper | ZAB | 3-5 nodes | Hundreds of MB | Kafka, Hadoop, HBase |
    | Consul | Raft | 3-5 nodes | Varies | Stripe, Twitch |
    | Google Chubby | Multi-Paxos | 5 nodes per cell | Small (lock service) | GFS, BigTable, Spanner |

    A common pattern across all these systems: consensus clusters are kept small (3-5
    nodes) because every write must reach a majority. Adding more nodes increases
    durability but also increases write latency. For read-heavy workloads, systems like
    etcd support linearizable reads from the leader or serializable (stale) reads from
    any node.

---

## Key Takeaways

Consensus solves the fundamental problem of getting distributed nodes to agree on
state despite crashes and network partitions. The majority quorum requirement (needing
more than half the nodes) is the core mechanism that prevents split brain.

Raft is the default choice for new systems because it was explicitly designed for
understandability and has battle-tested implementations in etcd, Consul, and
CockroachDB. Paxos is theoretically important and powers some of the largest systems at
Google, but its implementation complexity makes it impractical for most teams. ZAB
serves ZooKeeper well for ordered broadcast but is not a general-purpose consensus
library.

In practice, most engineers interact with consensus indirectly through systems like
etcd, ZooKeeper, or Consul. The practical skill is knowing when you need strong
consensus (leader election, distributed locks, configuration that must be consistent)
versus when eventual consistency is acceptable (analytics, caching, social media
counters).

---

## Related Topics

- [Distributed Systems Overview](index.md) - Broader distributed systems challenges
- [Consistent Hashing](consistent-hashing.md) - Data partitioning across nodes
- [Database Replication](../data/databases/replication.md) - Replication strategies and consistency
- [CAP Theorem](../fundamentals/cap-theorem.md) - Consistency and availability trade-offs
