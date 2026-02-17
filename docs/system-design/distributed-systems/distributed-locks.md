# Distributed Locks

In a single-process application, a mutex or synchronized block prevents two threads
from modifying shared state simultaneously. In a distributed system, the problem is
harder: how do you ensure that only one process across multiple servers can access a
resource at a time, when the network between them is unreliable and any server might
crash while holding the lock?

Distributed locks are essential for operations that must not execute concurrently:
processing a payment exactly once, preventing two users from booking the last hotel
room, or ensuring that a scheduled job runs on only one node in a cluster. Get it
wrong, and you get double charges, oversold inventory, or corrupted data.

---

## When You Actually Need a Distributed Lock

Before reaching for a distributed lock, consider whether you truly need one. Locks add
latency, complexity, and failure modes.

```
DO YOU NEED A DISTRIBUTED LOCK?

Is the operation idempotent?
├── YES → You may not need a lock. Idempotent operations
│         are safe to retry and duplicate.
│
└── NO → Is correctness at stake (money, inventory, data)?
    ├── NO → Consider "last write wins" or optimistic concurrency.
    │         Occasional duplicates may be acceptable.
    │
    └── YES → You likely need a distributed lock (or fencing).
              Read on.
```

**Alternatives to locks:**

- **Idempotency keys** — assign a unique ID to each operation; reject duplicates
- **Optimistic concurrency** — version stamps; retry on conflict
- **Single-writer design** — partition data so only one process ever writes to a given partition
- **Queue-based serialization** — funnel work through a single-consumer queue

---

=== "The Lock Problem"

    ## What Makes Distributed Locks Hard

    A correct distributed lock must satisfy three properties:

    1. **Mutual exclusion** — at most one client holds the lock at any time
    2. **Deadlock freedom** — even if a client crashes while holding the lock, the lock
       is eventually released (via TTL expiration)
    3. **Fault tolerance** — the lock service remains available if some nodes fail

    The danger is **false mutual exclusion** — two clients both believe they hold the
    lock, leading to concurrent access to the protected resource.

    ```
    THE FUNDAMENTAL DANGER

    Client A acquires lock with TTL = 10 seconds
    Client A starts long operation...
    ┌─────────────────────────────────────────┐
    │ Time 0s:  Client A acquires lock        │
    │ Time 5s:  Client A processing...        │
    │ Time 10s: LOCK EXPIRES (TTL)            │
    │ Time 11s: Client B acquires lock        │ ← lock is now B's
    │ Time 12s: Client A finishes, writes     │ ← A still thinks it has lock!
    │           BOTH A and B wrote to the      │
    │           protected resource.            │
    └─────────────────────────────────────────┘
    ```

    This happens because of **GC pauses, network delays, and clock drift**. Client A
    may pause for garbage collection, resume, and believe it still holds the lock even
    though the TTL expired. This is the central challenge of distributed locking.

=== "Redis-Based Locks"

    ## Redis Single-Instance Lock

    The simplest approach: use Redis `SET key value NX PX timeout` to atomically set a
    key only if it does not exist, with an expiration time.

    ```
    ACQUIRE:
    SET my-lock <unique-id> NX PX 30000
    │           │            │  │
    │           │            │  └─ expires in 30 seconds (deadlock freedom)
    │           │            └──── only set if not exists (mutual exclusion)
    │           └───────────────── unique value to prevent releasing others' locks
    └───────────────────────────── the lock key

    RELEASE (must be atomic — use Lua script):
    if redis.call("GET", key) == unique_id then
        redis.call("DEL", key)
    end
    ```

    **Why the unique ID matters:** Without it, Client A could accidentally release
    Client B's lock. If A's lock expired and B acquired it, A's `DEL` would release
    B's lock. The unique ID ensures a client can only release its own lock.

    **Limitation:** A single Redis instance is a single point of failure. If Redis
    crashes, the lock is lost. Redis replication doesn't help — replication is
    asynchronous, so a lock acquired on the primary may not exist on the replica if the
    primary crashes before replicating.

    ### Redlock Algorithm

    Martin Kleppmann and Salvatore Sanfilippo (Redis creator) had a famous debate about
    whether Redis can provide correct distributed locks. The Redlock algorithm is Redis's
    answer: acquire the lock on a **majority** of independent Redis instances.

    ```
    REDLOCK: 5 independent Redis instances

    Client tries to acquire lock on ALL 5:
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Redis 1 │ │Redis 2 │ │Redis 3 │ │Redis 4 │ │Redis 5 │
    │  ✅     │ │  ✅     │ │  ❌     │ │  ✅     │ │  ✅     │
    └────────┘ └────────┘ └────────┘ └────────┘ └────────┘

    Got 4/5 (majority) → Lock acquired ✅
    Must acquire majority within a time limit.
    Lock validity = TTL - time_spent_acquiring.

    To release: send DEL to all 5 instances.
    ```

    **Steps:**
    1. Get current time
    2. Try to acquire the lock on all N instances sequentially, with a short timeout
    3. If acquired on majority (N/2 + 1) and total acquisition time < lock TTL, the
       lock is valid
    4. Lock validity time = initial TTL - elapsed acquisition time
    5. If majority not reached, release the lock on all instances

    **The Redlock debate:** Martin Kleppmann argued Redlock is fundamentally flawed
    because it depends on timing assumptions (clock synchronization) that don't hold in
    distributed systems. If a process pauses (GC, page faults) after acquiring the lock,
    the TTL may expire without the process knowing. Sanfilippo countered that in
    practice, these pauses are rare and detectable. The consensus in the industry is:

    - For **efficiency locks** (preventing duplicate work): Redlock is fine
    - For **correctness locks** (preventing data corruption): use fencing tokens with a
      consensus-based system like ZooKeeper or etcd

=== "Consensus-Based Locks"

    ## ZooKeeper Locks

    ZooKeeper provides distributed locking through **ephemeral sequential nodes**. When
    a client disconnects (crashes), its ephemeral nodes are automatically deleted,
    releasing the lock.

    ```
    ZOOKEEPER LOCK RECIPE

    To acquire lock on resource "my-resource":

    1. Create ephemeral sequential node:
       /locks/my-resource/lock-0000000001  ← Client A
       /locks/my-resource/lock-0000000002  ← Client B
       /locks/my-resource/lock-0000000003  ← Client C

    2. Get all children, sort by sequence number

    3. If YOUR node is the LOWEST → you have the lock
       Client A has lock-0001 (lowest) → holds lock

    4. If not lowest → watch the node JUST BEFORE yours
       Client B watches lock-0001
       Client C watches lock-0002

    5. When watched node deleted → check again if you're lowest
       Client A crashes → lock-0001 deleted
       Client B notified → lock-0002 is now lowest → B holds lock
    ```

    **Why this is correct:** Ephemeral nodes are tied to the client session. If the
    client crashes or loses its connection, ZooKeeper's session timeout (typically
    10-30 seconds) expires and the node is deleted automatically. No TTL guessing needed.

    **The herd effect:** A naive implementation where all waiters watch the lock holder
    creates a "thundering herd" when the lock is released — all waiters wake up
    simultaneously. The sequential recipe avoids this: each waiter only watches its
    immediate predecessor, so only one waiter wakes up when the lock is released.

    ## etcd Locks

    etcd provides similar semantics through its lease mechanism and revision-based
    ordering. Kubernetes uses etcd for all its coordination, including leader election
    for controllers.

    ```
    ETCD LOCK MECHANISM

    1. Create a lease with TTL
    2. Put a key with that lease attached
    3. Lowest creation revision wins the lock
    4. Others watch and wait
    5. Lease expiration = automatic lock release

    Key advantage: etcd's Raft consensus ensures the lock
    state is replicated to a majority before acknowledging,
    making it resilient to node failures.
    ```

    | Property | Redis (Redlock) | ZooKeeper | etcd |
    |---|---|---|---|
    | Consistency model | Approximate (timing-based) | Strong (ZAB consensus) | Strong (Raft consensus) |
    | Auto-release mechanism | TTL expiration | Ephemeral node + session | Lease expiration |
    | Failure detection | TTL-based | Session heartbeat | Lease keepalive |
    | Performance | Fastest (~1ms) | Medium (~5-10ms) | Medium (~5-10ms) |
    | Correctness guarantee | Efficiency-level | Correctness-level | Correctness-level |
    | Operational complexity | Low | High (ZK ensemble) | Medium |

=== "Fencing Tokens"

    ## Fencing Tokens: The Real Solution

    Fencing tokens solve the fundamental problem of lock expiration during long
    operations. Every time a lock is acquired, the lock service issues a
    **monotonically increasing token** (a number that only goes up). The protected
    resource rejects any request with a token lower than the highest it has seen.

    ```
    FENCING TOKEN IN ACTION

    Time 0:  Client A acquires lock, gets token #34
    Time 5:  Client A sends write with token #34 → accepted
    Time 10: Lock expires (TTL)
    Time 11: Client B acquires lock, gets token #35
    Time 12: Client B sends write with token #35 → accepted
    Time 13: Client A (delayed by GC pause) sends write with token #34
             → REJECTED (34 < 35, stale token)

    The resource itself enforces ordering, regardless of
    whether the lock is correctly held.

    Without fencing:
    Time 13: Client A writes → ACCEPTED → data corruption

    With fencing:
    Time 13: Client A writes → REJECTED → safety preserved
    ```

    **This is the key insight:** The lock is not the sole mechanism ensuring safety.
    The fencing token creates a second line of defense at the resource itself. Even if
    the lock fails (due to GC pauses, network delays, clock skew), the resource rejects
    stale operations.

    **Implementation:** ZooKeeper's sequential node numbers naturally serve as fencing
    tokens. etcd's revision numbers serve the same purpose. For Redis, you must
    implement fencing separately — which is one reason Kleppmann argues Redis locks are
    insufficient for correctness-critical applications.

---

## Real-World Usage

### Stripe: Idempotency Over Locks

Stripe processes millions of payments and deliberately avoids distributed locks for
most operations. Instead, they use **idempotency keys**: every API request includes a
unique key, and the system guarantees that processing the same key twice produces the
same result. This eliminates the need for locks in most payment flows.

For the cases where they do need mutual exclusion (like preventing double-spend on a
single account), they use database-level serializable transactions rather than external
distributed locks.

### Google Chubby

Google built Chubby, a distributed lock service based on Paxos consensus, specifically
for coarse-grained locking. It's used internally for:

- Leader election for GFS master, Bigtable tablet servers
- Configuration storage that all nodes must agree on
- Namespace management for distributed services

Chubby inspired Apache ZooKeeper, which provides similar capabilities to the
open-source community.

### Kubernetes Leader Election

Kubernetes controllers (scheduler, controller-manager) use etcd-based leader election
to ensure only one instance processes work at a time. The leader acquires a lease in
etcd and renews it periodically. If the leader crashes, the lease expires and another
instance becomes leader — typically within 15-30 seconds.

---

## Key Takeaways

1. **Avoid distributed locks when possible.** Idempotent operations, single-writer
   partitioning, and optimistic concurrency are simpler and more scalable alternatives.

2. **Redis locks are fine for efficiency, not correctness.** If duplicate work wastes
   resources but doesn't corrupt data, a Redis lock is simple and fast. If incorrect
   behavior causes real harm (double charges, lost data), use a consensus-based system.

3. **Fencing tokens are the real safety mechanism.** Even a perfect lock can be
   undermined by GC pauses and network delays. Fencing tokens let the protected
   resource itself reject stale operations, regardless of lock state.

4. **TTLs must be generous but not infinite.** Too short and locks expire during normal
   operations. Too long and crashed clients hold locks for extended periods. Typical
   values: 10-30 seconds for most operations, with renewal for long-running tasks.

5. **The Kleppmann-Sanfilippo debate matters.** Understanding it helps you choose the
   right tool. For efficiency locks (deduplicate cron jobs) → Redis. For correctness
   locks (financial transactions) → ZooKeeper/etcd with fencing tokens.

---

## Related Topics

- [Consensus Algorithms](consensus.md) — Raft and Paxos that power correct lock services
- [Distributed Transactions](distributed-transactions.md) — coordinating multi-service operations
- [Consistent Hashing](consistent-hashing.md) — partitioning data to reduce lock contention
- [Fault Tolerance](../reliability/fault-tolerance.md) — handling lock service failures gracefully
