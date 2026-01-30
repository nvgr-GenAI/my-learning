# CAP Theorem

**Master the fundamental trade-off in distributed systems** | 🎯 Core Concepts | ⚖️ CP vs AP | 💼 Interview Ready

## Quick Reference

**The CAP theorem** - You can only guarantee 2 out of 3 in a distributed system:

| Property | Definition | When Chosen | Example Systems |
|----------|-----------|-------------|-----------------|
| **C**onsistency | All nodes see same data | Financial, inventory | MongoDB, Redis Cluster, HBase |
| **A**vailability | System always responds | Social, content | Cassandra, DynamoDB, Riak |
| **P**artition Tolerance | Works despite network splits | Required in distributed | All distributed systems |

**Key Insight:** Network partitions will happen, so you're really choosing between **C** or **A** during partition events.

---

=== "🎯 Understanding CAP"

    ## What is CAP Theorem?

    **Brewer's theorem** states that a distributed system can simultaneously provide only **two out of three** guarantees:

    ```mermaid
    graph TB
        subgraph "The CAP Triangle"
        C[Consistency<br/>All nodes see same data<br/>at the same time]
        A[Availability<br/>Every request gets<br/>a response]
        P[Partition Tolerance<br/>System works despite<br/>network failures]

        C -.->|Pick 2| A
        A -.->|Pick 2| P
        P -.->|Pick 2| C
        end

        subgraph "Reality: Choose One"
        CP[CP Systems<br/>┌─────────────┐<br/>│ MongoDB     │<br/>│ HBase       │<br/>│ Redis       │<br/>└─────────────┘]

        AP[AP Systems<br/>┌─────────────┐<br/>│ Cassandra   │<br/>│ DynamoDB    │<br/>│ Riak        │<br/>└─────────────┘]
        end
    ```

    **The Reality:**
    - Network partitions **will happen** in distributed systems
    - You can't avoid partition tolerance
    - Therefore: Choose between **Consistency** or **Availability** during partition events
    - It's not "pick any two" - it's "pick C or A" (P is mandatory)

    ---

    ## The Three Properties

    === "C - Consistency"

        **All nodes see the same data at the same time**

        ### What It Means

        | Aspect | Description |
        |--------|-------------|
        | **Reads** | Every read returns the most recent write |
        | **Replicas** | All replicas have identical data |
        | **No Stale Data** | No old or conflicting information exists |
        | **Linearizability** | Operations appear to execute atomically |

        ### Visual Example

        ```
        Strong Consistency (CP):
        ────────────────────────────────
        Time 0: Client writes X=1 to Server A
        Time 1: Server A replicates to Server B ✓
        Time 2: Server A replicates to Server C ✓
        Time 3: Client reads X from any server → Gets 1
                ✅ All servers synchronized before read

        Eventual Consistency (AP):
        ────────────────────────────────
        Time 0: Client writes X=1 to Server A
        Time 1: Client reads X from Server B → Gets 0 (stale!)
        Time 2: Replication catches up...
        Time 3: Client reads X from Server B → Gets 1
                ⏰ Eventually consistent
        ```

        ### Real-World Scenarios

        **Banking System:**
        ```python
        # Strong Consistency Required
        def transfer_money(from_account, to_account, amount):
            with database.transaction():  # ACID transaction
                # Read with lock
                from_balance = db.read_with_lock(from_account)
                to_balance = db.read_with_lock(to_account)

                # Validate
                if from_balance < amount:
                    raise InsufficientFunds()

                # Update atomically
                db.update(from_account, from_balance - amount)
                db.update(to_account, to_balance + amount)

                # Commit (all-or-nothing)
                db.commit()

        # If network partition:
        # → System returns error (unavailable)
        # → Does NOT show wrong balance
        ```

        **Why Consistency Matters:**
        - Money lost = unacceptable
        - Duplicate charges = legal issues
        - Regulatory compliance requires accuracy

        **Use Cases:**
        - Financial transactions (banking, payments)
        - Inventory management (prevent overselling)
        - Booking systems (airline seats, hotel rooms)
        - Configuration management (distributed locks)

    === "A - Availability"

        **Every request receives a response (success or failure)**

        ### What It Means

        | Aspect | Description |
        |--------|-------------|
        | **Always Responds** | No request goes unanswered |
        | **No Timeouts** | System doesn't hang indefinitely |
        | **Partial Data OK** | Better to return something than nothing |
        | **Fault Tolerance** | Works even if nodes fail |

        ### Visual Example

        ```
        High Availability (AP):
        ────────────────────────────────
        User Request → Load Balancer
                       ├─> Server A (alive) ✓
                       ├─> Server B (dead) ✗
                       └─> Server C (alive) ✓

        Result: Request served by A or C
                System remains available

        Low Availability (CP during partition):
        ────────────────────────────────
        User Request → Load Balancer
                       ├─> Server A (partitioned)
                       └─> Server B (partitioned)

        Result: Cannot guarantee consistency
                → System returns 503 error
                → Chooses correctness over availability
        ```

        ### Real-World Scenarios

        **Social Media Feed:**
        ```python
        # Availability Prioritized
        def get_user_feed(user_id):
            try:
                # Try primary database
                posts = primary_db.get_feed(user_id)
                return posts
            except DatabaseUnavailable:
                try:
                    # Fallback to replica (might be stale)
                    posts = replica_db.get_feed(user_id)
                    return posts
                except ReplicaUnavailable:
                    # Last resort: return cached data
                    posts = cache.get_stale(f"feed:{user_id}")
                    if posts:
                        return posts
                    # Even if empty, return something
                    return []

        # System always responds
        # Data might be stale by 30 seconds
        # Better than showing error page
        ```

        **Why Availability Matters:**
        - User engagement requires responsiveness
        - Error pages = users leave
        - Revenue loss from downtime
        - Competitive advantage

        **Use Cases:**
        - Social media platforms (Facebook, Twitter)
        - Content delivery (YouTube, Netflix)
        - Search engines (Google, Bing)
        - Shopping carts (Amazon, eBay)

    === "P - Partition Tolerance"

        **System continues to operate despite network failures**

        ### What It Means

        | Aspect | Description |
        |--------|-------------|
        | **Network Splits** | Nodes can't communicate |
        | **Message Loss** | Packets dropped or delayed |
        | **Split Brain** | Different parts see different state |
        | **Mandatory** | Can't be avoided in distributed systems |

        ### Visual Example

        ```
        Normal Operation:
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Node A  │◄───►│ Node B  │◄───►│ Node C  │
        │ (US)    │     │ (EU)    │     │ (Asia)  │
        └─────────┘     └─────────┘     └─────────┘
        All nodes communicate freely

        Network Partition:
        ┌─────────┐     ╳╳╳╳╳╳╳╳╳     ┌─────────┐
        │ Node A  │     ╳ Cable  ╳     │ Node C  │
        │ (US)    │     ╳  Cut   ╳     │ (Asia)  │
        └─────────┘     ╳╳╳╳╳╳╳╳╳     └─────────┘
             │                              │
             │         ┌─────────┐         │
             └────────►│ Node B  │◄────────┘
                       │ (EU)    │
                       └─────────┘

        Now must choose:
        - CP: A and C become unavailable
        - AP: A and C accept writes (may conflict)
        ```

        ### Network Partition Scenarios

        | Cause | Frequency | Duration | Impact |
        |-------|-----------|----------|--------|
        | **Data center outage** | Rare | Hours | Entire region down |
        | **Network switch failure** | Uncommon | Minutes | Partial connectivity loss |
        | **Undersea cable cut** | Rare | Days | Inter-continental partition |
        | **DNS issues** | Common | Seconds-Minutes | Service discovery fails |
        | **Firewall rules** | Common | Variable | Selective connectivity loss |

        ### Real-World Scenarios

        **Multi-Region Deployment:**
        ```python
        # Handling Partition Tolerance
        class DistributedDatabase:
            def __init__(self, nodes):
                self.nodes = nodes  # [US-East, US-West, EU, Asia]
                self.replication_factor = 3

            def write(self, key, value):
                # Try to write to majority of nodes
                successful_writes = 0
                required_writes = (len(self.nodes) // 2) + 1

                for node in self.nodes:
                    try:
                        node.write(key, value, timeout=100ms)
                        successful_writes += 1

                        if successful_writes >= required_writes:
                            return True  # Quorum achieved
                    except NetworkTimeout:
                        continue  # Try next node

                # Could not reach quorum
                if self.mode == "CP":
                    raise UnavailableError("Cannot guarantee consistency")
                else:  # AP mode
                    # Accept write anyway (eventual consistency)
                    return True
        ```

        **Why Partition Tolerance is Mandatory:**
        - Networks are unreliable (always)
        - Physics limits (speed of light, distance)
        - Hardware failures inevitable
        - Can't prevent partitions, only handle them

        **Common Partition Causes:**
        - Data center network failures
        - Router/switch malfunctions
        - Physical cable damage
        - DDoS attacks
        - Configuration errors

=== "⚖️ CP vs AP"

    ## The Great Divide

    **During a network partition, you must choose:**

    ### Comparison Table

    | Aspect | CP Systems | AP Systems |
    |--------|-----------|-----------|
    | **Philosophy** | Correctness over availability | Availability over perfect consistency |
    | **During Partition** | Returns errors, becomes unavailable | Continues serving (possibly stale data) |
    | **Consistency** | Strong (linearizable) | Eventual (converges over time) |
    | **Latency** | Higher (wait for consensus) | Lower (local reads/writes) |
    | **Complexity** | Lower (simpler model) | Higher (conflict resolution) |
    | **User Experience** | May see errors | Always works (may see stale data) |

    ---

    ## CP Systems: Consistency + Partition Tolerance

    **"Better to be right than available"**

    === "How CP Works"

        ### Behavior During Partition

        ```
        Before Partition:
        Client → Node A (primary) → Replicate → Node B, C ✓

        During Partition:
        Client → Node A ╳ Cannot reach Node B, C

        CP Response:
        ❌ Reject write (cannot guarantee consistency)
        ❌ System becomes unavailable in affected region
        ✅ Data remains consistent across reachable nodes

        After Partition Heals:
        ✅ Synchronize data
        ✅ Resume normal operations
        ```

        ### Implementation Strategies

        **Quorum-Based Writes:**
        ```python
        class CPDatabase:
            def __init__(self, nodes, replication_factor=3):
                self.nodes = nodes
                self.W = (replication_factor // 2) + 1  # Write quorum
                self.R = (replication_factor // 2) + 1  # Read quorum
                # W + R > N ensures consistency

            def write(self, key, value):
                """
                Must write to majority of nodes
                If can't reach majority → FAIL
                """
                successful = 0

                for node in self.nodes[:self.replication_factor]:
                    try:
                        node.write(key, value, timeout=1000ms)
                        successful += 1
                    except NetworkError:
                        pass  # Node unreachable

                if successful >= self.W:
                    return True  # Quorum achieved
                else:
                    raise ConsistencyError("Cannot guarantee consistency")

            def read(self, key):
                """
                Must read from majority of nodes
                Return most recent version
                """
                responses = []

                for node in self.nodes[:self.replication_factor]:
                    try:
                        value, version = node.read(key, timeout=1000ms)
                        responses.append((value, version))
                    except NetworkError:
                        pass

                if len(responses) >= self.R:
                    # Return value with highest version
                    return max(responses, key=lambda x: x[1])[0]
                else:
                    raise ConsistencyError("Cannot guarantee consistency")
        ```

        **Two-Phase Commit (2PC):**
        ```python
        def two_phase_commit(transaction):
            """
            Phase 1: Prepare (vote)
            Phase 2: Commit or Abort
            """
            # Phase 1: Ask all nodes if they can commit
            votes = []
            for node in nodes:
                try:
                    vote = node.prepare(transaction)
                    votes.append(vote)
                except NetworkError:
                    votes.append(False)  # No response = No vote

            # All nodes must vote YES
            if all(votes):
                # Phase 2: Commit
                for node in nodes:
                    try:
                        node.commit(transaction)
                    except NetworkError:
                        # Inconsistent state! Need recovery
                        handle_failure(node)
                return True
            else:
                # Phase 2: Abort
                for node in nodes:
                    node.abort(transaction)
                return False
        ```

    === "CP Systems"

        ### Example Technologies

        | System | Use Case | Consistency Model | When Unavailable |
        |--------|----------|------------------|------------------|
        | **MongoDB** | Document store | Majority read/write | Cannot reach majority |
        | **Redis Cluster** | In-memory cache | Primary-replica | Primary unavailable |
        | **HBase** | Column store | Strong consistency | Region server down |
        | **ZooKeeper** | Coordination service | Linearizable | No quorum |
        | **Etcd** | Configuration | Raft consensus | Leader election |

        ### Real-World Example: Banking

        ```python
        class BankingSystem:
            """
            CP System: Cannot allow inconsistent balances
            """
            def transfer(self, from_account, to_account, amount):
                # Start distributed transaction
                transaction_id = generate_id()

                try:
                    # Phase 1: Lock accounts on all replicas
                    with distributed_lock([from_account, to_account]):
                        # Read current balances (with locks)
                        from_balance = self.read_with_quorum(from_account)
                        to_balance = self.read_with_quorum(to_account)

                        # Validate
                        if from_balance < amount:
                            raise InsufficientFunds()

                        # Phase 2: Update all replicas
                        new_from = from_balance - amount
                        new_to = to_balance + amount

                        # Write to quorum
                        self.write_with_quorum(from_account, new_from)
                        self.write_with_quorum(to_account, new_to)

                        # Commit transaction
                        return {
                            "status": "success",
                            "transaction_id": transaction_id
                        }

                except QuorumNotReached:
                    # Cannot guarantee consistency
                    return {
                        "status": "error",
                        "message": "Service temporarily unavailable"
                    }
                    # Better to fail than allow wrong balance
        ```

        **Why This Works:**
        - ✅ Money never lost or duplicated
        - ✅ All replicas have same balance
        - ✅ Atomic operations (all-or-nothing)
        - ❌ Service unavailable during partition
        - ❌ Higher latency (wait for quorum)

    === "When to Choose CP"

        ### Perfect For

        | Use Case | Why CP | Consequence of Inconsistency |
        |----------|--------|------------------------------|
        | **Banking** | Money accuracy critical | Lost funds, legal issues |
        | **Inventory** | Can't oversell products | Customer disappointment, refunds |
        | **Booking** | Double-booking unacceptable | Service disruption, refunds |
        | **Distributed Locks** | Mutual exclusion required | Data corruption |
        | **Configuration** | All services need same config | System malfunction |

        ### Decision Checklist

        Choose CP if:
        - [ ] Data correctness is more important than availability
        - [ ] Users understand "try again later" messages
        - [ ] Inconsistency has serious consequences
        - [ ] Transactions must be atomic
        - [ ] Regulatory compliance requires accuracy

        **Red Flags for CP:**
        - ❌ Users expect 99.99% availability
        - ❌ Real-time user experience required
        - ❌ Global user base (high partition risk)
        - ❌ Stale data is acceptable

    ---

    ## AP Systems: Availability + Partition Tolerance

    **"Better to be available than perfectly consistent"**

    === "How AP Works"

        ### Behavior During Partition

        ```
        Before Partition:
        Client → Any Node → Eventually replicates to all

        During Partition:
        Client A → Node A (US)  → Writes locally ✓
        Client B → Node C (Asia) → Writes locally ✓

        AP Response:
        ✅ Both writes succeed immediately
        ✅ System remains available in all regions
        ⚠️  Data temporarily inconsistent

        After Partition Heals:
        🔄 Merge data from both sides
        🔧 Resolve conflicts (last-write-wins, vector clocks, etc.)
        ✅ Eventually consistent
        ```

        ### Implementation Strategies

        **Last-Write-Wins (LWW):**
        ```python
        class APDatabase:
            def write(self, key, value):
                """
                Write to local node immediately
                Replicate asynchronously
                """
                timestamp = time.time()

                # Write locally (fast!)
                self.local_node.write(key, value, timestamp)

                # Replicate in background
                async_replicate(key, value, timestamp)

                return True  # Always succeeds

            def read(self, key):
                """
                Read from nearest node
                May return stale data
                """
                return self.nearest_node.read(key)

            def resolve_conflict(self, key, values_with_timestamps):
                """
                If multiple versions exist, use latest timestamp
                """
                latest = max(values_with_timestamps,
                           key=lambda x: x.timestamp)
                return latest.value
        ```

        **Vector Clocks (Causality Tracking):**
        ```python
        class VectorClock:
            """
            Track causality to resolve conflicts
            """
            def __init__(self, node_id, num_nodes):
                self.node_id = node_id
                self.clock = [0] * num_nodes

            def increment(self):
                """Increment local clock on write"""
                self.clock[self.node_id] += 1

            def merge(self, other_clock):
                """Merge clocks from another node"""
                for i in range(len(self.clock)):
                    self.clock[i] = max(self.clock[i], other_clock[i])

            def happens_before(self, other_clock):
                """Check if this event happened before another"""
                return (all(self.clock[i] <= other_clock[i]
                          for i in range(len(self.clock))) and
                       any(self.clock[i] < other_clock[i]
                          for i in range(len(self.clock))))

            def concurrent(self, other_clock):
                """Check if events are concurrent (conflicting)"""
                return not (self.happens_before(other_clock) or
                          other_clock.happens_before(self.clock))

        # Usage
        def handle_write(key, value):
            vc = vector_clock.increment()
            store(key, value, vc)

        def handle_read(key):
            versions = get_all_versions(key)

            # Filter out outdated versions
            current = [v for v in versions
                      if not any(v.vc.happens_before(other.vc)
                               for other in versions)]

            if len(current) == 1:
                return current[0]  # No conflict
            else:
                return resolve_siblings(current)  # Conflict!
        ```

    === "AP Systems"

        ### Example Technologies

        | System | Use Case | Conflict Resolution | Consistency Window |
        |--------|----------|--------------------|--------------------|
        | **Cassandra** | Wide-column store | Last-write-wins | Seconds to minutes |
        | **DynamoDB** | Key-value store | Vector clocks | Milliseconds |
        | **Riak** | Distributed KV | Sibling resolution | Seconds |
        | **CouchDB** | Document store | Multi-version concurrency | Minutes |
        | **Voldemort** | Distributed cache | Application-defined | Seconds |

        ### Real-World Example: Social Media

        ```python
        class SocialMediaFeed:
            """
            AP System: Availability matters more than perfect consistency
            """
            def create_post(self, user_id, content):
                """
                Write to local datacenter immediately
                Replicate asynchronously to others
                """
                post_id = generate_id()
                timestamp = time.time()

                # Write to local node (fast!)
                self.local_db.write({
                    "post_id": post_id,
                    "user_id": user_id,
                    "content": content,
                    "timestamp": timestamp
                })

                # Async replication (don't wait)
                replication_queue.enqueue({
                    "operation": "create_post",
                    "data": {
                        "post_id": post_id,
                        "user_id": user_id,
                        "content": content,
                        "timestamp": timestamp
                    },
                    "target_regions": ["US", "EU", "Asia"]
                })

                # Return immediately
                return {"post_id": post_id, "status": "published"}

            def get_feed(self, user_id):
                """
                Read from nearest datacenter
                Might show slightly stale posts
                """
                try:
                    # Try nearest replica
                    posts = self.nearest_replica.get_feed(user_id, limit=50)
                    return posts
                except ReplicaDown:
                    # Fallback to any available replica
                    posts = self.any_replica.get_feed(user_id, limit=50)
                    return posts
                    # Always returns something

            def like_post(self, post_id, user_id):
                """
                Increment like counter locally
                Eventual consistency for counts
                """
                # Local increment (fast)
                self.local_db.increment("likes:{}".format(post_id))

                # Replicate asynchronously
                # Different regions might show different counts temporarily
                # Eventually converges to correct count

                return {"status": "liked"}
        ```

        **Why This Works:**
        - ✅ Always available (never returns error)
        - ✅ Fast response times (local writes)
        - ✅ Global scalability
        - ⚠️  Post might appear on your feed before friend's feed (acceptable)
        - ⚠️  Like counts might differ by a few across regions (acceptable)

    === "When to Choose AP"

        ### Perfect For

        | Use Case | Why AP | Consequence of Unavailability |
        |----------|--------|-------------------------------|
        | **Social Media** | User engagement critical | Users leave, revenue loss |
        | **Content Delivery** | Global reach required | Poor user experience |
        | **Shopping Carts** | Abandoned carts OK | Lost sales |
        | **User Profiles** | Stale data acceptable | Minimal impact |
        | **Analytics** | Approximate counts OK | Slight inaccuracy acceptable |

        ### Decision Checklist

        Choose AP if:
        - [ ] Availability is more important than perfect consistency
        - [ ] Stale data is tolerable (seconds to minutes)
        - [ ] Global user base across regions
        - [ ] Real-time user experience required
        - [ ] Can handle conflict resolution

        **Red Flags for AP:**
        - ❌ Financial transactions
        - ❌ Data accuracy is critical
        - ❌ Regulatory compliance requires strong consistency
        - ❌ Conflicts are hard to resolve

=== "🎯 Real-World Examples"

    ## Case Studies

    === "Banking (CP)"

        **Scenario:** Global bank with customers worldwide

        ### Requirements

        | Requirement | Priority | CAP Choice |
        |-------------|----------|------------|
        | Account balance accuracy | Critical | → CP |
        | No double spending | Critical | → CP |
        | Fast transfers | Important | Trade-off |
        | 24/7 availability | Important | Trade-off |

        ### Architecture

        ```
        ┌──────────────────────────────────────────┐
        │           Load Balancer                  │
        └──────────────┬───────────────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
        ┌─────▼─────┐     ┌────▼──────┐
        │ Primary DB│────►│ Replica 1 │
        │  (Master) │     │(Read-only)│
        └───────────┘     └───────────┘
              │
              │ Synchronous Replication
              │
        ┌─────▼─────┐
        │ Replica 2 │
        │(Read-only)│
        └───────────┘

        CP Configuration:
        - Writes: Must succeed on Primary + 1 Replica
        - Reads: Can use any replica (slightly stale OK)
        - Partition: Reject writes if cannot reach quorum
        ```

        ### Implementation

        ```python
        class BankTransferService:
            def transfer(self, from_acct, to_acct, amount):
                # Start 2-phase commit
                transaction = {
                    "from": from_acct,
                    "to": to_acct,
                    "amount": amount,
                    "timestamp": time.time()
                }

                try:
                    # Phase 1: Prepare
                    primary_vote = primary_db.prepare(transaction)
                    replica_votes = [r.prepare(transaction)
                                    for r in replicas]

                    if not (primary_vote and any(replica_votes)):
                        raise QuorumFailed()

                    # Phase 2: Commit
                    primary_db.commit(transaction)
                    for replica in replicas:
                        try:
                            replica.commit(transaction)
                        except NetworkError:
                            # Continue anyway (eventual consistency for reads)
                            log_replication_failure(replica)

                    return {"status": "success"}

                except QuorumFailed:
                    # Abort transaction
                    primary_db.abort(transaction)
                    return {
                        "status": "error",
                        "message": "Service temporarily unavailable. Please try again."
                    }
        ```

        ### Trade-offs Accepted

        | Trade-off | Accepted | Reason |
        |-----------|----------|--------|
        | **Higher latency** | Yes | 100-200ms OK for accuracy |
        | **Occasional unavailability** | Yes | Users understand banks need maintenance |
        | **Complex implementation** | Yes | Worth it for correctness |
        | **Higher cost** | Yes | Cannot compromise on money |

    === "Social Media (AP)"

        **Scenario:** Global social network with 1B+ users

        ### Requirements

        | Requirement | Priority | CAP Choice |
        |-------------|----------|------------|
        | Always available | Critical | → AP |
        | Fast feed loads | Critical | → AP |
        | Perfect consistency | Low | Trade-off |
        | Real-time updates | Important | Trade-off |

        ### Architecture

        ```
        ┌───────────────────────────────────────────┐
        │        Global Traffic Manager             │
        │     (Route to nearest datacenter)         │
        └────┬───────────────┬──────────────┬───────┘
             │               │              │
        ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
        │   US    │    │   EU    │   │  Asia   │
        │ Region  │◄──►│ Region  │◄──►│ Region  │
        └─────────┘    └─────────┘   └─────────┘

        Each Region:
        ┌────────────────────────────────────┐
        │  App Servers (many)                │
        │  Database Cluster (replicated)     │
        │  Cache Layer (Redis)               │
        │  Message Queue (async replication) │
        └────────────────────────────────────┘

        AP Configuration:
        - Writes: Succeed on local datacenter
        - Reads: From nearest datacenter
        - Replication: Asynchronous across regions
        - Conflict: Last-write-wins
        ```

        ### Implementation

        ```python
        class SocialFeedService:
            def __init__(self, local_region):
                self.local_region = local_region
                self.local_db = get_db(local_region)
                self.replicas = get_all_regions_except(local_region)

            def create_post(self, user_id, content):
                # Write to local datacenter ONLY (fast!)
                post = {
                    "id": generate_id(),
                    "user_id": user_id,
                    "content": content,
                    "timestamp": time.time(),
                    "region": self.local_region
                }

                # Local write (5-10ms)
                self.local_db.write(post)

                # Queue for async replication
                replication_queue.enqueue(post)
                # User in US sees post immediately
                # User in Asia sees post in 1-30 seconds

                return post

            def get_feed(self, user_id, limit=50):
                # Read from local datacenter (fast!)
                # Might miss very recent posts from other regions

                try:
                    posts = self.local_db.get_feed(user_id, limit)
                    return posts
                except DatabaseError:
                    # Fallback to cache
                    cached_posts = cache.get(f"feed:{user_id}")
                    if cached_posts:
                        return cached_posts
                    # Last resort: return empty (still available!)
                    return []

            async def replicate_to_other_regions():
                """Background job: replicate writes"""
                while True:
                    batch = replication_queue.dequeue(batch_size=1000)

                    for region in self.replicas:
                        try:
                            region.bulk_write(batch)
                        except NetworkError:
                            # Retry later, but don't block
                            replication_queue.enqueue_failed(batch, region)
        ```

        ### Trade-offs Accepted

        | Trade-off | Accepted | Reason |
        |-----------|----------|--------|
        | **Eventual consistency** | Yes | Post appearing 10s late is OK |
        | **Duplicate posts** | Yes | Rare, can deduplicate |
        | **Like count discrepancy** | Yes | Approximate counts acceptable |
        | **Higher complexity** | Yes | Worth it for availability |

    === "E-Commerce (Hybrid)"

        **Scenario:** Online retailer with different consistency needs per feature

        ### Mixed Strategy

        | Feature | CAP Choice | Reason |
        |---------|------------|--------|
        | **Product Catalog** | AP | Stale prices/descriptions OK |
        | **Inventory Count** | CP | Cannot oversell products |
        | **Shopping Cart** | AP | Cart contents can be eventual |
        | **Checkout/Payment** | CP | Money must be accurate |
        | **User Reviews** | AP | Slight delays acceptable |
        | **Recommendations** | AP | Personalization can lag |

        ### Architecture

        ```python
        class EcommerceSystem:
            def __init__(self):
                # Different services, different consistency models
                self.catalog = APService()      # Availability
                self.inventory = CPService()    # Consistency
                self.cart = APService()         # Availability
                self.payment = CPService()      # Consistency

            def add_to_cart(self, user_id, product_id):
                """
                AP: Cart can be eventually consistent
                """
                # Write to local datacenter (fast)
                self.cart.add_item(user_id, product_id)
                return {"status": "added"}

            def checkout(self, user_id):
                """
                CP: Inventory check must be consistent
                """
                cart_items = self.cart.get_items(user_id)

                try:
                    # Check inventory with strong consistency
                    for item in cart_items:
                        available = self.inventory.check_and_reserve(
                            item.product_id,
                            item.quantity
                        )
                        if not available:
                            return {
                                "status": "error",
                                "message": f"{item.name} out of stock"
                            }

                    # Process payment with strong consistency
                    payment_result = self.payment.charge(
                        user_id,
                        calculate_total(cart_items)
                    )

                    if payment_result.success:
                        # Fulfill order
                        return {"status": "success"}
                    else:
                        # Release inventory
                        self.inventory.release_all(cart_items)
                        return {"status": "payment_failed"}

                except QuorumNotReached:
                    return {
                        "status": "error",
                        "message": "Checkout temporarily unavailable"
                    }
        ```

        **Why Hybrid Works:**
        - Browse products: Fast (AP)
        - Add to cart: Fast (AP)
        - Checkout: Accurate (CP), acceptable if slightly slower
        - Payment: Accurate (CP), users expect wait

=== "💡 Interview Tips"

    ## How to Discuss CAP in Interviews

    ### Common Questions

    **Q1: "Explain the CAP theorem"**

    **Good Answer:**
    ```
    "The CAP theorem states that in a distributed system experiencing
    network partitions, you must choose between Consistency and Availability.

    - Consistency (C): All nodes see the same data simultaneously
    - Availability (A): System always responds to requests
    - Partition Tolerance (P): Works despite network failures

    Since network partitions will inevitably occur in distributed systems,
    partition tolerance is mandatory. Therefore, you're really choosing
    between consistency (CP) or availability (AP) during partition events.

    It's not a binary choice - different parts of your system can make
    different trade-offs based on business requirements."
    ```

    **What interviewer likes:**
    - ✅ Mentions it's about behavior during partitions
    - ✅ Explains P is mandatory in distributed systems
    - ✅ Notes it's a spectrum, not binary
    - ✅ Ties to business requirements

    ---

    **Q2: "Would you choose CP or AP for [system]?"**

    **Answer Framework:**
    ```
    1. Clarify requirements:
       "Let me ask: What's the business impact of stale data?
        What happens if the system becomes unavailable?"

    2. Analyze the use case:
       For Banking (→ CP):
       - Stale balance = money lost = unacceptable
       - Downtime = users wait = acceptable
       - Implementation: Quorum writes, strong consistency

       For Social Feed (→ AP):
       - Stale posts = see post 10s late = acceptable
       - Downtime = users leave = unacceptable
       - Implementation: Eventual consistency, async replication

    3. Consider hybrid:
       "For complex systems, we might use both:
        - Critical operations: CP (payments, inventory)
        - User-facing features: AP (catalog, recommendations)"
    ```

    ---

    **Q3: "How do you handle eventual consistency?"**

    **Good Answer:**
    ```
    "Eventual consistency means the system will converge to a consistent
    state given enough time without writes. Here's how to implement it:

    1. Conflict Detection:
       - Vector clocks to track causality
       - Version numbers for each write
       - Timestamps (with clock sync)

    2. Conflict Resolution:
       - Last-write-wins (use timestamp)
       - Application-defined (user chooses)
       - CRDTs (conflict-free replicated data types)
       - Merge function (combine values)

    3. Example:
       Shopping cart using last-write-wins:
       - User adds item in US datacenter (timestamp: 1000)
       - User adds item in EU datacenter (timestamp: 1002)
       - Merge: EU write wins (later timestamp)
       - Result: Cart has both items (acceptable)

    The key is choosing a resolution strategy appropriate for your domain."
    ```

    ---

    ### Red Flags to Avoid

    | Red Flag | Why It's Wrong | Better Approach |
    |----------|---------------|-----------------|
    | "You can have all three" | Violates CAP theorem | Explain the trade-off |
    | "Just use strong consistency" | Ignores availability needs | Ask about requirements |
    | "Eventual consistency means eventually" | Too vague | Specify time window (seconds/minutes) |
    | "Always choose CP for accuracy" | Oversimplification | Depends on business impact |
    | "Network partitions are rare" | Wrong assumption | They're inevitable |
    | "CA systems exist" | Contradicts CAP | Single-node only (not distributed) |

    ---

    ### Interview Cheat Sheet

    **Memorize These:**

    **CP Systems (Consistency + Partition Tolerance):**
    - MongoDB, HBase, Redis Cluster, ZooKeeper, Etcd
    - Use cases: Banking, inventory, bookings, distributed locks
    - Behavior: Returns errors during partition (unavailable)

    **AP Systems (Availability + Partition Tolerance):**
    - Cassandra, DynamoDB, Riak, CouchDB, Voldemort
    - Use cases: Social media, content delivery, carts, analytics
    - Behavior: Continues serving (eventual consistency)

    **Quorum Formula:**
    - W + R > N ensures strong consistency
    - W = write quorum, R = read quorum, N = replicas
    - Example: N=5, W=3, R=3 → 3+3=6 > 5 ✓

    **Consistency Levels:**
    - Strong: Read sees latest write (CP)
    - Eventual: Reads converge over time (AP)
    - Causal: Causally related writes ordered (AP variant)

    ---

    ### Practice Problems

    **Design These with CAP Considerations:**

    1. **Distributed Cache (Redis)**
       - Question: CP or AP?
       - Answer: Depends! Single master = CP, Cluster with replicas = tunable

    2. **Global E-commerce Site**
       - Question: How to handle inventory?
       - Answer: CP for inventory, AP for browsing

    3. **Messaging App (WhatsApp)**
       - Question: Message delivery guarantees?
       - Answer: AP for availability, eventual consistency for delivery

    4. **Banking System**
       - Question: Account balance consistency?
       - Answer: Strong consistency (CP) for transfers, AP for balance queries acceptable

=== "⚠️ Common Mistakes"

    ## Misconceptions About CAP

    | Mistake | Reality | Why It Matters |
    |---------|---------|----------------|
    | **"Pick any 2 of 3"** | Pick C or A (P is mandatory) | Misunderstanding fundamentals |
    | **"CA systems exist in distributed"** | Only single-node is CA | All distributed systems must handle partitions |
    | **"Eventual = slow"** | Usually seconds, not hours | Sets wrong expectations |
    | **"One size fits all"** | Different features need different choices | Over-simplified architecture |
    | **"Strong consistency = CP"** | Strong consistency requires CP | Confusing cause and effect |
    | **"Network partitions are theoretical"** | They happen daily | Unprepared for reality |
    | **"CAP applies to single node"** | Only for distributed systems | Misapplying the theorem |
    | **"Availability means uptime"** | Availability = responds (even with stale data) | Confusing terms |

    ---

    ## Design Pitfalls

    | Pitfall | Problem | Solution |
    |---------|---------|----------|
    | **Ignoring partition tolerance** | System fails during network issues | Always design for partitions |
    | **Not defining consistency level** | Unclear system behavior | Explicitly choose strong/eventual |
    | **Assuming low partition rate** | Unprepared for reality | Test with chaos engineering |
    | **No conflict resolution** | AP system breaks on conflicts | Define resolution strategy upfront |
    | **Synchronous cross-region** | High latency, poor UX | Use async replication |
    | **No monitoring** | Can't detect consistency issues | Track replication lag, conflicts |

    ---

    ## Interview Mistakes

    **Don't Say:**
    - ❌ "We'll use blockchain for consistency" (buzzword without understanding)
    - ❌ "Just cache everything" (doesn't solve consistency)
    - ❌ "Network is reliable these days" (networks always fail)
    - ❌ "MongoDB is always CP" (depends on configuration)

    **Do Say:**
    - ✅ "For this use case, I'd choose [CP/AP] because [business reason]"
    - ✅ "Different parts of the system can make different trade-offs"
    - ✅ "We need to define SLAs for consistency and availability"
    - ✅ "I'd implement [specific strategy] for conflict resolution"

---

## 🎯 Key Takeaways

**The 10 Rules of CAP:**

1. **Network partitions will happen** - Design for them, don't hope they won't occur

2. **You must choose C or A** - During partitions, pick consistency or availability (not both)

3. **Context matters** - Banking needs CP, social media needs AP

4. **It's a spectrum** - Eventual consistency has many variants (seconds vs minutes vs hours)

5. **Different features, different choices** - Catalog (AP) + Inventory (CP) = Hybrid

6. **Test for partitions** - Use chaos engineering to verify behavior

7. **Define SLAs** - Consistency window, availability target, partition frequency

8. **Conflict resolution is hard** - Plan your strategy before going AP

9. **Monitor replication** - Track lag, detect split-brain, alert on issues

10. **Document your choice** - Explain why you chose CP or AP for each component

---

## 📚 Further Reading

**Master these related concepts:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **PACELC** | CAP extension (Latency trade-off) | [Data Consistency →](data-consistency.md) |
| **Consistency Models** | Strong, eventual, causal | [Data Consistency →](data-consistency.md) |
| **Consensus Algorithms** | Raft, Paxos implementation | [Distributed Systems →](../distributed-systems/consensus.md) |
| **Replication** | How to sync data across nodes | [Database Replication →](../databases/replication.md) |

**Practice with real systems:**
- [Distributed Cache (Redis)](../problems/design-redis.md) - CP or AP?
- [URL Shortener](../problems/url-shortener.md) - Consistency needs?
- [Chat System](../problems/whatsapp.md) - Message delivery guarantees?

---

**Master CAP and you'll make better architectural decisions! 🚀**
