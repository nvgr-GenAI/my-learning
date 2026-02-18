# Time, Clocks, and Ordering

In a single-process program, events have a natural total order -- one thing happens, then
the next. In a distributed system, there is no shared global clock. Each node has its own
clock, and those clocks inevitably drift apart. Messages between nodes take variable time
to arrive. The result is a fundamental question that underpins nearly every distributed
system design: **if two events happen on different machines, which one happened first?**

This question is not academic. Getting the answer wrong causes lost writes in databases,
violated invariants in replicated state machines, and phantom reads in distributed
transactions. Every system that claims consistency -- from Google Spanner to Amazon
DynamoDB -- must have a principled answer to how it orders events across machines.

```
THE CORE PROBLEM: NO SHARED CLOCK

Node A              Node B              Node C
┌─────────┐        ┌─────────┐        ┌─────────┐
│ Clock:   │        │ Clock:   │        │ Clock:   │
│ 10:00:00 │        │ 10:00:03 │        │ 09:59:58 │
│ .000     │        │ .217     │        │ .891     │
└─────────┘        └─────────┘        └─────────┘
     │                   │                   │
     │  "x = 5" at       │  "x = 7" at      │
     │  10:00:00.500     │  10:00:00.500     │
     │                   │                   │
     └───────── Which write wins? ───────────┘

Node A says it wrote at 10:00:00.500 (its time)
Node B says it wrote at 10:00:00.500 (its time)
But Node B's clock is 3.2 seconds ahead...
so Node B actually wrote ~2.7 seconds EARLIER in real time.
```

---

=== "Physical Clocks"

    ## Physical Clocks: Wall-Clock Time

    Physical clocks attempt to track real-world (wall-clock) time. Every computer has one,
    typically a quartz crystal oscillator that vibrates at a known frequency. The problem is
    that "known frequency" is only approximate -- crystals vary with temperature, age, and
    manufacturing tolerances.

    ### Clock Drift and Skew

    **Clock drift** is the rate at which a clock diverges from real time. A typical server
    crystal drifts at 10-100 parts per million (ppm), meaning:

    | Drift Rate | Error After 1 Hour | Error After 1 Day | Error After 1 Month |
    |---|---|---|---|
    | 10 ppm | 36 ms | 0.86 sec | 26 sec |
    | 50 ppm | 180 ms | 4.3 sec | 130 sec |
    | 100 ppm | 360 ms | 8.6 sec | 260 sec |

    **Clock skew** is the difference between two clocks at any given instant. Even if you
    synchronize two servers perfectly at noon, after one hour their clocks could differ by
    hundreds of milliseconds due to different drift rates.

    ```
    CLOCK DRIFT OVER TIME

    Real time:  ──────────────────────────────────────>

    Server A:   ──────────────────────────────────>     (fast crystal, +50 ppm)
                                               ↑ 180ms ahead after 1 hour

    Server B:   ─────────────────────────────────────>  (accurate)

    Server C:   ────────────────────────────────────>   (slow crystal, -30 ppm)
                                               ↑ 108ms behind after 1 hour

    Clock skew between A and C after 1 hour: ~288ms
    ```

    ### NTP: Network Time Protocol

    NTP synchronizes clocks by querying time servers over the network. A client sends a
    request, records when it sent it, records when the reply arrives, and estimates the
    server's time by accounting for network round-trip time.

    ```
    NTP SYNCHRONIZATION

    Client                        Time Server
      │                                │
      │──── Request (t1) ────────────>│
      │                                │  Server receives at t2
      │                                │  Server sends at t3
      │<──── Response (t4) ───────────│
      │                                │

    Round-trip delay = (t4 - t1) - (t3 - t2)
    Clock offset     = ((t2 - t1) + (t3 - t4)) / 2
    ```

    NTP achieves accuracy of **1-10 ms** within a data center and **10-100 ms** across the
    internet. This sounds good, but for a database processing thousands of transactions per
    second (each taking microseconds), a 10 ms uncertainty window means thousands of
    transactions cannot be reliably ordered by wall-clock time alone.

    !!! warning "NTP Can Move Time Backward"
        When NTP discovers a clock is ahead, it may **step** the clock backward (for large
        corrections) or **slew** it (gradually slow down for small corrections). A clock
        going backward can break assumptions in software that expects monotonically
        increasing timestamps. This is why many systems use `CLOCK_MONOTONIC` instead of
        `CLOCK_REALTIME` for measuring elapsed time.

    ### Why Physical Clocks Are Not Enough

    Physical clocks have three fundamental problems for distributed ordering:

    1. **Skew**: Two servers never agree on the exact time
    2. **Drift**: The disagreement grows over time between synchronizations
    3. **Non-monotonicity**: NTP corrections can make time appear to go backward

    These limitations motivated the invention of logical clocks -- mechanisms that do not
    try to track real time at all, but instead capture the **causal order** of events.

=== "Logical Clocks"

    ## Lamport Timestamps: Capturing Causality

    In 1978, Leslie Lamport published "Time, Clocks, and the Ordering of Events in a
    Distributed System," one of the most cited papers in computer science. His key insight:
    **you don't need to know what time it is -- you just need to know what happened before
    what.**

    A Lamport timestamp is a single integer counter. The rules are remarkably simple:

    1. Each node maintains a counter, starting at 0
    2. Before each local event, increment the counter
    3. When sending a message, attach the current counter value
    4. When receiving a message, set counter = max(local, received) + 1

    ```
    STEP-BY-STEP LAMPORT TIMESTAMP TRACE

    Node A              Node B              Node C
    L=0                 L=0                 L=0
     │                   │                   │
     ● L=1 (event a1)   │                   │
     │                   │                   │
     │── send(L=1) ────>│                   │
     │                   ● L=max(0,1)+1=2    │
     │                   │  (event b1)       │
     │                   │                   │
     │                   │── send(L=2) ────>│
     │                   │                   ● L=max(0,2)+1=3
     │                   │                   │  (event c1)
     │                   │                   │
     ● L=2 (event a2)   │                   │
     │                   │                   │
     │                   │                   ● L=4 (event c2)
     │                   │                   │
     │<──────────────── send(L=4) ──────────│
     ● L=max(2,4)+1=5   │                   │
     │  (event a3)       │                   │

    Causal chain: a1 -> b1 -> c1 -> c2 -> a3
    Lamport:       1     2     3     4     5   (correctly ordered!)
    ```

    ### What Lamport Timestamps Guarantee (and Don't)

    **Guarantee**: If event A causally precedes event B (written A -> B), then
    L(A) < L(B). Causality is always preserved in the timestamp ordering.

    **Limitation**: The converse is NOT true. If L(A) < L(B), you cannot conclude that A
    happened before B. They might be **concurrent** -- causally unrelated events that
    happened to get different counter values.

    !!! info "Lamport Clock Pseudo-Code"
        ```
        on local event:
            counter = counter + 1

        on send(message):
            counter = counter + 1
            attach counter to message

        on receive(message, received_counter):
            counter = max(counter, received_counter) + 1
        ```

    This limitation is fundamental: a single integer cannot distinguish "definitely happened
    before" from "happened concurrently but got a smaller number." To detect concurrency,
    you need vector clocks.

=== "Vector Clocks"

    ## Vector Clocks: Detecting Concurrency

    A vector clock is an array of counters, one per node in the system. Instead of a single
    number, each event is stamped with a vector that records how many events each node has
    performed. This extra information lets you determine not just ordering but also
    **concurrency** -- whether two events are causally related or independent.

    ### Rules

    For a system with N nodes, each node maintains a vector of N counters:

    1. Initialize all counters to 0
    2. Before each local event, increment your own position in the vector
    3. When sending a message, attach your entire vector
    4. When receiving, take the element-wise max of both vectors, then increment your own position

    ### Comparing Vector Clocks

    Given two vector timestamps V1 and V2:

    - **V1 < V2** (V1 happened before V2): Every element of V1 is less than or equal to the
      corresponding element of V2, and at least one is strictly less
    - **V1 = V2**: All elements are equal
    - **V1 || V2** (concurrent): Neither V1 <= V2 nor V2 <= V1 -- each vector has at least
      one element greater than the other

    ```
    VECTOR CLOCKS WITH 3 NODES

    Node A              Node B              Node C
    [0,0,0]             [0,0,0]             [0,0,0]
      │                   │                   │
      ● [1,0,0]          │                   │
      │  write(x=5)      │                   │
      │                   │                   │
      │── send ─────────>│                   │
      │  [1,0,0]         ● [1,1,0]           │
      │                   │  read(x)          │
      │                   │                   │
      │                   │── send ──────────>│
      │                   │  [1,1,0]          ● [1,1,1]
      │                   │                   │  process()
      │                   │                   │
      ● [2,0,0]          │                   │
      │  write(x=8)      │                   │
      │                   │                   │
      │                   │                   ● [1,1,2]
      │                   │                   │  write(x=3)
      │                   │                   │

    Now compare the last two events:

    Node A: [2,0,0]  -- write(x=8)
    Node C: [1,1,2]  -- write(x=3)

    Is [2,0,0] <= [1,1,2]?  No, because 2 > 1 in position 0
    Is [1,1,2] <= [2,0,0]?  No, because 1 > 0 in position 1, and 2 > 0 in position 2

    Result: CONCURRENT -- these writes conflict!
    The system must resolve this (last-writer-wins, merge, ask the user, etc.)
    ```

    ### Why Concurrency Detection Matters

    Consider a shopping cart system where two users (on different servers) both modify the
    same cart simultaneously:

    ```
    CONFLICT DETECTION IN A SHOPPING CART

    Server A                          Server B
    VC: [1,0]                         VC: [0,1]
     │                                 │
     ● [2,0] add(milk)                │
     │                                 ● [0,2] add(eggs)
     │                                 │
     └──── Both vectors arrive at ────┘
           coordination node

    Compare [2,0] vs [0,2]:
      Position 0: 2 > 0  (A ahead)
      Position 1: 0 < 2  (B ahead)

    CONCURRENT! Neither happened before the other.

    Resolution options:
    1. Merge both: cart = {milk, eggs}     <-- DynamoDB approach
    2. Last-writer-wins by wall clock      <-- Cassandra approach
    3. Return both to client to resolve    <-- Riak approach
    ```

    ### Scaling Problem

    Vector size grows with the number of nodes -- 1,000 nodes means 1,000 integers per
    event. Practical mitigations include **pruning** old entries (risks false concurrency),
    tracking vectors **server-side only** (not per-client), and using **dotted version
    vectors** that handle client operations more efficiently.

    !!! note "Vector Clocks vs Version Vectors"
        Strictly speaking, Amazon DynamoDB and Riak use **version vectors** (also called
        dotted version vectors), not classical vector clocks. Version vectors track the
        version of a data item across replicas, while vector clocks track the causal history
        of a process. In practice, the terms are often used interchangeably, but the
        distinction matters for correctness in some edge cases involving concurrent writes
        from the same client.

=== "Hybrid Logical Clocks"

    ## Hybrid Logical Clocks (HLC)

    Hybrid Logical Clocks, introduced by Kulkarni et al. in 2014, combine the best of
    physical and logical clocks. The key insight: use physical time when possible (because
    humans understand it and it correlates with real-world time), but fall back to logical
    counters to maintain causality when physical clocks are too coarse or have drifted.

    An HLC timestamp has three components packed into a single 64-bit value:

    ```
    HLC TIMESTAMP STRUCTURE

    ┌──────────────────────────────────────────────────────┐
    │             64-bit HLC Timestamp                     │
    ├────────────────────────┬──────────────┬──────────────┤
    │   Physical time (pt)   │ Logical (l)  │  Node ID     │
    │   (48 bits)            │ (12 bits)    │  (4 bits)    │
    └────────────────────────┴──────────────┴──────────────┘

    pt: Largest physical time seen so far
    l:  Logical counter for events at the same physical time
    ```

    ### How HLC Works

    ```
    HLC ALGORITHM

    On local event or send:
      pt_new = max(local_pt, physical_clock())
      if pt_new == local_pt:
          l = l + 1            // Same physical time, increment logical
      else:
          l = 0                // New physical time, reset logical
      local_pt = pt_new
      timestamp = (local_pt, l)

    On receive(msg_pt, msg_l):
      pt_new = max(local_pt, msg_pt, physical_clock())
      if pt_new == local_pt == msg_pt:
          l = max(l, msg_l) + 1
      elif pt_new == local_pt:
          l = l + 1
      elif pt_new == msg_pt:
          l = msg_l + 1
      else:
          l = 0                // physical_clock() is newest, reset logical
      local_pt = pt_new
      timestamp = (local_pt, l)
    ```

    ### HLC Properties

    | Property | Lamport | Vector Clock | HLC |
    |---|---|---|---|
    | Captures causality | Partial | Full | Partial (same as Lamport) |
    | Detects concurrency | No | Yes | No |
    | Timestamp size | O(1) | O(N) | O(1) |
    | Close to wall-clock time | No | No | Yes |
    | Useful for snapshots | No | Impractical | Yes |

    The critical advantage of HLC is that timestamps are always within a bounded distance
    of real physical time (bounded by the maximum clock drift in the system). This means you
    can use HLC timestamps for snapshot reads -- "give me all data as of this timestamp" --
    something impossible with pure Lamport clocks. CockroachDB uses HLCs for exactly this
    purpose.

    !!! tip "Why Not Just Use Lamport Clocks?"
        Lamport timestamps bear no relationship to real time. If a Lamport clock reads
        1,000,000, you have no idea if that corresponds to 10 AM or 3 PM. HLCs embed
        physical time, so a timestamp of (2026-02-17T10:00:00, 3) tells you the event
        happened at roughly 10:00 AM with the third logical tick at that physical time.
        This is invaluable for debugging, TTL expiration, and snapshot isolation.

=== "TrueTime"

    ## Google TrueTime: Bounded Uncertainty

    Google took a radically different approach with TrueTime, introduced alongside the
    Spanner database in 2012. Instead of giving up on physical time, Google invested in
    hardware to make physical time reliable enough for global ordering -- then exposed the
    remaining uncertainty explicitly in the API.

    ### The TrueTime API

    TrueTime does not return a single timestamp. It returns an **interval**:

    ```
    TrueTime API

    TT.now() returns [earliest, latest]

    Example: TT.now() = [10:00:00.001, 10:00:00.007]

    Meaning: The actual time is GUARANTEED to be somewhere
             in this interval. The uncertainty is 6ms.

    TT.after(t)  --> true if t is definitely in the past
    TT.before(t) --> true if t is definitely in the future
    ```

    ### Hardware Architecture

    Each Google data center has a set of **time masters**. Most time masters have GPS
    receivers that synchronize to atomic clocks on GPS satellites. A few have local atomic
    clocks (rubidium or cesium) as a safeguard against GPS failures (jamming, antenna
    issues, spoofing). Servers in the data center poll multiple time masters and use a
    variant of Marzullo's algorithm to compute a tight confidence interval.

    ```
    TRUETIME INFRASTRUCTURE

    ┌──────────────────────────────────────────────┐
    │               Data Center                    │
    │                                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │GPS Master│  │GPS Master│  │Atomic Clk│  │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
    │       └──────────┬───┴──────────────┘        │
    │    ┌─────────────┼─────────────┐             │
    │  ┌─┴──┐       ┌─┴──┐       ┌─┴──┐          │
    │  │Srv1│       │Srv2│       │Srv3│  ...      │
    │  │ε=4ms│      │ε=3ms│      │ε=5ms│          │
    │  └────┘       └────┘       └────┘           │
    └──────────────────────────────────────────────┘

    Each server polls multiple masters, computes uncertainty (ε)
    Typical ε: ~1ms after sync, grows with drift, ~4ms average
    ```

    ### How Spanner Uses TrueTime for Ordering

    Spanner assigns each transaction a timestamp and enforces a simple rule: **if
    transaction T1 committed before transaction T2 started, then T1's timestamp must be
    less than T2's timestamp.** This is called **external consistency** (equivalent to
    linearizability for transactions).

    To achieve this, after committing a transaction at timestamp `s`, the leader **waits**
    until TrueTime guarantees that `s` is in the past:

    ```
    COMMIT-WAIT PROTOCOL

    Transaction T1 commits:
      1. Acquire locks
      2. s = TT.now().latest         // Pick timestamp at top of uncertainty
      3. Commit with timestamp s
      4. WAIT until TT.after(s)      // Wait until s is definitely in the past
      5. Release locks, reply to client

    Wait duration = 2 * ε  (typically ~7-8 ms)

    WHY THIS WORKS:

    Time ──────────────────────────────────────────>

    T1 commits at s:
         [───── ε ─────s───── ε ─────]
                       ^              ^
                  s assigned     TT.after(s) = true
                                (safe to release)

    T2 starts after T1 replies:
                                     [── ε ──t── ε ──]
                                              ^
                                         T2 gets timestamp t

    Since T2 started after TT.after(s), we know t > s.
    External consistency guaranteed!
    ```

    !!! warning "The Cost of Certainty"
        The commit-wait adds ~7-8 ms of latency to every transaction. Google considers this
        an acceptable price for global consistency. The alternative -- distributed consensus
        protocols without TrueTime -- would require cross-region round-trips that take
        100+ ms. TrueTime's hardware investment converts a distributed coordination problem
        into a local wait, which is dramatically faster.

    ### Why Others Cannot Easily Replicate TrueTime

    TrueTime requires dedicated GPS receivers and atomic clocks in every data center.
    AWS and Azure offer GPS-backed time services but expose standard NTP interfaces, not
    bounded-uncertainty APIs. CockroachDB and YugabyteDB use HLCs instead, accepting
    that clock uncertainty is unbounded but small in practice.

=== "Happens-Before"

    ## The Happens-Before Relationship

    The **happens-before** relation (denoted ->) is the formal foundation for reasoning
    about event ordering in distributed systems. Defined by Lamport in 1978, it captures
    the minimum ordering that any correct system must respect.

    ### Formal Definition

    Event A **happens before** event B (written A -> B) if any of these hold:

    1. **Same node, program order**: A and B occur on the same node, and A comes before B
       in the local execution sequence
    2. **Message passing**: A is the sending of a message, and B is the receipt of that
       same message
    3. **Transitivity**: There exists an event C such that A -> C and C -> B

    If neither A -> B nor B -> A, the events are **concurrent** (written A || B).

    ```
    HAPPENS-BEFORE EXAMPLES

    Node 1              Node 2              Node 3
      │                   │                   │
      ● a                 │                   │
      │                   │                   │
      ● b                 ● d                 │
      │                   │                   │
      │── msg m1 ───────>● e                 │
      │                   │                   │
      ● c                 │── msg m2 ───────>● g
      │                   │                   │
      │                   ● f                 ● h
      │                   │                   │

    HAPPENS-BEFORE RELATIONS:
    a -> b  (same node, program order)
    b -> e  (message m1: b is before send, e is receive)
    a -> e  (transitivity: a -> b -> e)
    e -> g  (message m2: e is before send, g is receive)
    a -> g  (transitivity: a -> b -> e -> g)
    g -> h  (same node)
    d -> e  (same node)

    CONCURRENT PAIRS (no causal path connects them):
    a || d,   c || e,   c || f,   c || g,   b || d,   f || h
    ```

    ### Practical Implications

    The happens-before relation tells you the **minimum set of ordering constraints** a
    system must enforce. Events that are concurrent can safely be processed in any order
    (or in parallel) without violating correctness. This has direct implications:

    - **Databases**: Two concurrent writes to the same key represent a true conflict that
      must be detected (vector clocks, CRDTs) or prevented (locks, consensus)
    - **Message queues**: Messages with happens-before must be delivered in order; concurrent
      messages can be delivered in any order, enabling parallelism
    - **Debugging**: Tracing the happens-before chain reveals whether an anomaly is a
      genuine concurrency issue or a bug in the ordering mechanism

    !!! info "Causal Consistency"
        A system provides **causal consistency** if it respects all happens-before
        relationships: if A -> B, then every node sees A before B. Concurrent events may be
        seen in different orders by different nodes. Causal consistency is weaker than
        linearizability but stronger than eventual consistency, and it can be achieved
        without the latency cost of consensus protocols.

    ### The Spectrum of Consistency Through the Lens of Ordering

    ```
    CONSISTENCY MODELS AND ORDERING GUARANTEES

    Strongest ──────────────────────────────── Weakest

    Linearizability    Sequential     Causal        Eventual
    │                  │              │             │
    │ Total order      │ Total order  │ Partial     │ No ordering
    │ matching real    │ (some valid  │ order       │ guarantee
    │ time             │ sequential   │ (respects   │
    │                  │ interleaving)│ happens-    │
    │ Requires:        │              │ before)     │ Requires:
    │ TrueTime or      │ Requires:    │             │ Nothing
    │ consensus        │ Single       │ Requires:   │
    │                  │ leader       │ Vector/     │
    │ Example:         │              │ Lamport     │ Example:
    │ Spanner          │ Example:     │ clocks      │ DNS cache
    │                  │ ZooKeeper    │             │
    │                  │              │ Example:    │
    │                  │              │ COPS, Riak  │
    └──────────────────┴──────────────┴─────────────┘
    ```

=== "Real-World Systems"

    ## How Production Systems Handle Time

    ### Google Spanner: TrueTime for Global Consistency

    **Scale**: 10+ million servers, spans continents, sub-10ms cross-continent consistency

    Spanner is the only production database that provides external consistency (also called
    strict serializability) at global scale. The key enabler is TrueTime's bounded
    uncertainty.

    | Aspect | Detail |
    |---|---|
    | Clock source | GPS receivers + atomic clocks in every data center |
    | Uncertainty | ~4 ms average, ~7 ms worst case |
    | Ordering mechanism | Commit-wait: pause for 2*epsilon after assigning timestamp |
    | Trade-off | Adds ~7ms latency per transaction for global consistency |
    | Why it works | Hardware makes clock uncertainty small and bounded |

    **Key insight**: Google realized that investing in hardware (GPS + atomic clocks) to
    reduce clock uncertainty was cheaper and simpler than using software-only approaches
    (like consensus for every timestamp) that add much higher latency.

    ### Amazon DynamoDB: Version Vectors for Conflict Detection

    **Scale**: Tens of millions of requests per second, single-digit millisecond latency

    DynamoDB prioritizes availability over consistency (AP in CAP). When network partitions
    occur, replicas accept writes independently and reconcile later. Version vectors
    (similar to vector clocks) detect when writes conflict.

    ```
    DYNAMODB CONFLICT RESOLUTION FLOW

    Client writes key "cart-123" to two replicas during partition:

    Replica A                     Replica B
    VC: [2, 1]                    VC: [1, 2]
    value: {milk, bread}          value: {milk, eggs}
        │                             │
        └────── Partition heals ──────┘
                     │
                     v
              Compare vectors:
              [2,1] vs [1,2] --> CONCURRENT
                     │
                     v
              Return BOTH versions to client
              Client merges: {milk, bread, eggs}
              Client writes merged result back
    ```

    | Aspect | Detail |
    |---|---|
    | Clock source | None for ordering (pure logical approach) |
    | Uncertainty | N/A -- does not rely on physical time for ordering |
    | Ordering mechanism | Version vectors detect concurrent writes |
    | Trade-off | Clients must handle conflicts; eventual consistency |
    | Why it works | Shopping carts and similar data merge naturally |

    !!! note "DynamoDB's Evolution"
        Modern DynamoDB has moved away from client-side conflict resolution for most use
        cases. The current default is last-writer-wins using physical timestamps, with
        strong consistency available as an option (at the cost of routing reads to the
        leader replica). The vector clock approach described in the original Dynamo paper
        (2007) was found to be too complex for most application developers.

    ### CockroachDB: Hybrid Logical Clocks

    **Scale**: Multi-region deployments, serializable transactions

    CockroachDB cannot rely on custom hardware like Google, so it uses HLCs to get
    close-to-wall-clock timestamps with causal ordering guarantees.

    | Aspect | Detail |
    |---|---|
    | Clock source | NTP-synchronized physical clocks + logical counter |
    | Uncertainty | Configurable maximum offset (default: 500ms) |
    | Ordering mechanism | HLC timestamps + read uncertainty intervals |
    | Trade-off | Must handle "uncertain" reads; clock skew causes retries |
    | Why it works | NTP is good enough for most deployments |

    CockroachDB handles clock skew with **uncertainty intervals**: when a read encounters
    a value with a timestamp in its uncertainty window, it retries at a higher timestamp.
    If clocks are well-synchronized (low skew), retries are rare. If skew is high,
    performance degrades but correctness is preserved.

    ```
    COCKROACHDB UNCERTAINTY INTERVAL

    Node A reads key "x" at HLC time t=100, max_offset=10

    Uncertainty window: [100, 110]

        90     95    100    105    110    115
    ────┼──────┼──────┼──────┼──────┼──────┼────>
                      ^                    ^
                 read time            uncertainty
                                      boundary

    If "x" has a write at t=107:
      107 is within [100, 110] --> UNCERTAIN
      Node A cannot tell if this write happened
      before or after the read in real time.

      Solution: Retry the read at t=108 (above the uncertain write).
      Now the write at 107 is definitely visible.
    ```

=== "Comparison"

    ## Clock Types at a Glance

    | Property | Physical (NTP) | Lamport | Vector Clock | HLC | TrueTime |
    |---|---|---|---|---|---|
    | Tracks real time | Yes (approx.) | No | No | Yes (approx.) | Yes (bounded) |
    | Captures causality | No | Partial | Full | Partial | No (uses wait) |
    | Detects concurrency | No | No | Yes | No | No |
    | Timestamp size | 64 bits | 64 bits | O(N) * 64 bits | 64 bits | 128 bits |
    | Requires special HW | No | No | No | No | Yes (GPS + atomic) |
    | Accuracy | 1-100 ms | N/A | N/A | ~NTP accuracy | ~4 ms |
    | Suitable for ordering | Unreliable | Causal order | Causal + concurrent | Causal order | Total order |

    ### Trade-Off Summary

    ```
    TRADE-OFFS AT A GLANCE

                        Simple                           Complex
                        ┌─────────────────────────────────────┐
    Mechanism           │ NTP    Lamport    HLC    VC    TT   │
                        └─────────────────────────────────────┘

                        Weak                             Strong
                        ┌─────────────────────────────────────┐
    Ordering guarantee  │ NTP    Lamport    HLC    VC    TT   │
                        └─────────────────────────────────────┘

                        Low                               High
                        ┌─────────────────────────────────────┐
    Overhead per event  │ Lamport  HLC    NTP    VC    TT     │
                        └─────────────────────────────────────┘

                        None                         GPS+Atomic
                        ┌─────────────────────────────────────┐
    Hardware required   │ Lamport VC HLC NTP          TT      │
                        └─────────────────────────────────────┘
    ```

=== "Decision Guide"

    ## When to Use What

    ```
    DECISION TREE: CHOOSING A CLOCK MECHANISM

    Start here
        │
        ▼
    Do you need to detect concurrent writes?
        │
        ├── YES ──> Vector Clocks (or CRDTs)
        │           Examples: shopping carts, collaborative editing
        │           Used by: Riak, (original) DynamoDB
        │
        └── NO
             │
             ▼
        Do you need globally consistent ordering?
             │
             ├── YES
             │    │
             │    ▼
             │   Can you invest in GPS + atomic clock hardware?
             │    │
             │    ├── YES ──> TrueTime (Google Spanner approach)
             │    │           Provides: external consistency at global scale
             │    │           Latency cost: ~7ms commit-wait
             │    │
             │    └── NO ──> Hybrid Logical Clocks + Consensus
             │               Examples: CockroachDB, YugabyteDB
             │               Trade-off: Clock skew causes read retries
             │
             └── NO
                  │
                  ▼
             Do you need timestamps that correlate with wall-clock time?
                  │
                  ├── YES ──> Hybrid Logical Clocks
                  │           Good for: snapshot reads, TTL, debugging
                  │           Used by: CockroachDB, MongoDB
                  │
                  └── NO ──> Lamport Timestamps
                             Simplest option, lowest overhead
                             Good for: log ordering, message sequencing
                             Used by: many internal protocols
    ```

    ### Quick Reference: Common Scenarios

    | Scenario | Recommended Approach | Why |
    |---|---|---|
    | Global SQL database | TrueTime or HLC + consensus | Need serializable transactions across regions |
    | Multi-region key-value store (AP) | Vector clocks / version vectors | Must detect and resolve conflicts |
    | Event sourcing / log ordering | Lamport timestamps | Only need causal ordering of events |
    | Distributed cache invalidation | Physical clocks (NTP) | Approximate ordering is sufficient |
    | Financial transactions | TrueTime or consensus-based | Strict ordering required for audit trail |
    | Collaborative editing | Vector clocks + CRDTs | Must merge concurrent edits without data loss |
    | Snapshot isolation in NewSQL | HLC | Need wall-clock-correlated consistent snapshots |
    | Microservice request tracing | HLC or Lamport + trace ID | Causal ordering for debugging, wall-clock for display |

    !!! tip "The Pragmatic Default"
        If you are building a new distributed system and are unsure which clock to use,
        **Hybrid Logical Clocks** are a strong default. They provide causal ordering,
        correlate with wall-clock time, have O(1) timestamp size, and require no special
        hardware. You only need vector clocks if you must detect concurrency, and you only
        need TrueTime if you need global external consistency with minimal latency.

---

## Further Reading

- Leslie Lamport, "Time, Clocks, and the Ordering of Events in a Distributed System" (1978)
- Colin Fidge and Friedemann Mattern, Vector Clocks (1988, independently)
- Kulkarni et al., "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases" (2014) -- HLC paper
- Corbett et al., "Spanner: Google's Globally-Distributed Database" (2012) -- TrueTime
- DeCandia et al., "Dynamo: Amazon's Highly Available Key-value Store" (2007) -- Vector clocks in practice
