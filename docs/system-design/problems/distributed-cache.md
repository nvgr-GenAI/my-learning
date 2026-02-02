# Design Distributed Cache (Redis/Memcached)

A high-performance, distributed in-memory caching system that stores key-value pairs with sub-millisecond latency, supporting data structures, persistence, replication, and automatic failover.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M ops/sec, TB-scale data, <1ms p99 latency, 99.99% availability |
| **Key Challenges** | Consistent hashing, replication strategies, eviction policies, data structure implementation |
| **Core Concepts** | Hash ring, virtual nodes, LRU/LFU/LFU eviction, sharding, master-replica topology |
| **Companies** | Meta, Google, Amazon, Netflix, Uber, Twitter, DoorDash, Stripe |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **SET/GET** | Store and retrieve key-value pairs | P0 (Must have) |
    | **DELETE** | Remove keys from cache | P0 (Must have) |
    | **TTL (Time To Live)** | Auto-expire keys after specified time | P0 (Must have) |
    | **Data Structures** | Strings, lists, sets, sorted sets, hashes | P0 (Must have) |
    | **Atomic Operations** | INCR, DECR, compare-and-set | P1 (Should have) |
    | **Pub/Sub** | Message broadcasting for real-time events | P1 (Should have) |
    | **Batch Operations** | MGET, MSET for multiple keys | P1 (Should have) |
    | **Transactions** | MULTI/EXEC for atomic command groups | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - SQL queries (use database instead)
    - Complex analytics (use data warehouse)
    - Full database replacement (cache is complementary)
    - Disk-only storage (caches are memory-first)
    - Strong consistency (eventual consistency acceptable)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (GET)** | < 1ms p99 | In-memory cache must be faster than disk |
    | **Latency (SET)** | < 2ms p99 | Writes involve replication, slightly slower |
    | **Availability** | 99.99% uptime | Near-constant availability for hot data |
    | **Throughput** | 10M ops/sec | Handle massive read-heavy workloads |
    | **Consistency** | Eventual consistency | Brief replication lag acceptable (< 100ms) |
    | **Durability** | Optional (AOF/RDB) | Trade-off: performance vs. persistence |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Operations per second:
    - Read operations (GET): 8M ops/sec (80% reads)
    - Write operations (SET/DELETE): 2M ops/sec (20% writes)
    - Total QPS: 10M ops/sec
    - Peak QPS: 3x average = 30M ops/sec (Black Friday, flash sales)

    Read/Write ratio: 4:1 (read-heavy typical for cache)

    Cache hit rate: 95% (5% cache misses go to database)
    - Effective reads served: 8M × 0.95 = 7.6M ops/sec
    - Database queries (cache miss): 8M × 0.05 = 400K ops/sec

    Network traffic:
    - Average value size: 1 KB
    - Read bandwidth: 8M ops/sec × 1 KB = 8 GB/sec = 64 Gbps
    - Write bandwidth: 2M ops/sec × 1 KB = 2 GB/sec = 16 Gbps
    - Total: 80 Gbps (requires high-bandwidth network)
    ```

    ### Storage Estimates

    ```
    Cache data:
    - Total keys: 1 billion (1B keys)
    - Average key size: 50 bytes
    - Average value size: 1 KB
    - Metadata per entry: 100 bytes (TTL, LRU pointers, flags)

    Per-entry storage: 50 bytes (key) + 1 KB (value) + 100 bytes (metadata) = 1.15 KB

    Total storage:
    - Data: 1B × 1.15 KB = 1.15 TB
    - Replication factor: 3x (for availability)
    - Total with replication: 1.15 TB × 3 = 3.45 TB

    Memory overhead (fragmentation, internal structures):
    - Redis overhead: 20-30% of data size
    - Effective storage: 1.15 TB × 1.3 = 1.5 TB per master
    - With replication: 1.5 TB × 3 = 4.5 TB total

    Partitioning:
    - 100 nodes (shards) × 15 GB RAM per node = 1.5 TB capacity
    - Each shard has 2 replicas: 300 total nodes
    ```

    ### Bandwidth Estimates

    ```
    Ingress (writes):
    - 2M SET ops/sec × 1 KB = 2 GB/sec ≈ 16 Gbps
    - With replication (3x): 16 Gbps × 3 = 48 Gbps

    Egress (reads):
    - 8M GET ops/sec × 1 KB = 8 GB/sec ≈ 64 Gbps

    Total bandwidth: 48 Gbps (ingress) + 64 Gbps (egress) = 112 Gbps
    - Per node (100 shards): 1.12 Gbps (manageable with 10 Gbps NICs)
    ```

    ### CPU Estimates

    ```
    Per-operation cost:
    - Simple GET/SET: 1-2 μs CPU time
    - Complex operations (ZADD, sorted sets): 5-10 μs
    - Hash operations: 2-5 μs

    Total CPU:
    - 10M ops/sec × 2 μs = 20 CPU-seconds per second
    - 100 shards: 0.2 CPU-seconds per shard
    - With overhead: 0.5 CPU cores per shard (single-threaded Redis)

    Recommendation: 2-4 CPU cores per node (allows headroom)
    ```

    ### Memory Estimates (Per Node)

    ```
    Data per shard:
    - 1.5 TB / 100 shards = 15 GB per shard
    - With overhead: 15 GB × 1.3 = 19.5 GB

    Additional memory:
    - Redis internal structures: 2 GB
    - OS cache: 2 GB
    - Total RAM per node: 19.5 + 2 + 2 ≈ 24 GB

    Recommendation: 32 GB RAM per node (25% headroom)
    ```

    ---

    ## Key Assumptions

    1. Read-heavy workload (80/20 read/write ratio)
    2. Cache hit rate of 95% (well-tuned cache)
    3. Average value size of 1 KB (mix of small and large objects)
    4. Eventual consistency acceptable (replication lag < 100ms)
    5. Majority of keys have TTL set (automatic cleanup)
    6. Hot keys (top 1%) account for 80% of traffic (Zipf distribution)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Consistent hashing:** Evenly distribute keys across nodes, minimize redistribution on scaling
    2. **Master-replica topology:** High availability through replication (async)
    3. **Client-side sharding:** Smart clients route requests directly to correct shard
    4. **Memory-first:** All data in RAM for sub-millisecond latency
    5. **Eviction policies:** Automatic removal of least-used data (LRU/LFU)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            App1[Application 1]
            App2[Application 2]
            App3[Application 3]
        end

        subgraph "Smart Client (Consistent Hashing)"
            Client[Cache Client<br/>Hash ring<br/>Routing logic]
        end

        subgraph "Cache Cluster (Sharded)"
            subgraph "Shard 1 (0-99)"
                Master1[Master Node 1<br/>Redis<br/>Slots: 0-99]
                Replica1A[Replica 1A]
                Replica1B[Replica 1B]
            end

            subgraph "Shard 2 (100-199)"
                Master2[Master Node 2<br/>Redis<br/>Slots: 100-199]
                Replica2A[Replica 2A]
                Replica2B[Replica 2B]
            end

            subgraph "Shard N (16200-16383)"
                MasterN[Master Node N<br/>Redis<br/>Slots: 16200-16383]
                ReplicaNA[Replica NA]
                ReplicaNB[Replica NB]
            end
        end

        subgraph "Coordination & Monitoring"
            Sentinel[Redis Sentinel<br/>Health checks<br/>Failover]
            Metrics[Monitoring<br/>Prometheus/Grafana]
            Config[Configuration<br/>Cluster topology]
        end

        subgraph "Persistence (Optional)"
            AOF[(AOF Log<br/>Append-only file)]
            RDB[(RDB Snapshot<br/>Point-in-time)]
        end

        App1 --> Client
        App2 --> Client
        App3 --> Client

        Client --> Master1
        Client --> Master2
        Client --> MasterN

        Master1 -.->|Async replication| Replica1A
        Master1 -.->|Async replication| Replica1B
        Master2 -.->|Async replication| Replica2A
        Master2 -.->|Async replication| Replica2B
        MasterN -.->|Async replication| ReplicaNA
        MasterN -.->|Async replication| ReplicaNB

        Sentinel --> Master1
        Sentinel --> Master2
        Sentinel --> MasterN
        Sentinel --> Replica1A
        Sentinel --> Replica2A
        Sentinel --> ReplicaNA

        Master1 --> AOF
        Master1 --> RDB
        Master2 --> AOF
        Master2 --> RDB

        Metrics --> Master1
        Metrics --> Master2
        Metrics --> MasterN

        style Client fill:#e1f5ff
        style Master1 fill:#ffe1e1
        style Master2 fill:#ffe1e1
        style MasterN fill:#ffe1e1
        style Replica1A fill:#fff4e1
        style Replica2A fill:#fff4e1
        style ReplicaNA fill:#fff4e1
        style Sentinel fill:#e8f5e9
        style AOF fill:#f3e5f5
        style RDB fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Consistent Hashing** | Minimize key redistribution when adding/removing nodes (only 1/N keys move) | Modulo hashing (all keys move on resize), range sharding (hot spots) |
    | **Master-Replica Topology** | High availability (automatic failover), read scalability | Multi-master (complex conflict resolution), no replication (downtime on failure) |
    | **Client-Side Sharding** | Eliminate proxy layer (lower latency), direct routing | Proxy-based (Twemproxy adds 1-2ms), server-side routing (Redis Cluster complexity) |
    | **In-Memory Storage** | Sub-millisecond latency (RAM is 100x faster than SSD) | SSD-backed cache (10-100ms latency), disk-only (defeats purpose of cache) |
    | **LRU/LFU Eviction** | Automatic memory management, keep hot data | Manual eviction (application complexity), FIFO (evicts useful data) |
    | **Redis Sentinel** | Automatic failover (< 30s downtime), health monitoring | Manual failover (slow, error-prone), no monitoring (blind to failures) |

    **Key Trade-off:** We chose **availability over consistency**. Async replication means recent writes may be lost during failover (< 1 second of data), but system remains available.

    ---

    ## API Design

    ### 1. SET (Store Key-Value)

    **Request:**
    ```bash
    SET user:12345 '{"name":"John","email":"john@example.com"}' EX 3600
    ```

    **Parameters:**
    - `key`: String identifier (e.g., "user:12345")
    - `value`: Any data type (string, JSON, binary)
    - `EX seconds`: Optional TTL in seconds
    - `PX milliseconds`: Optional TTL in milliseconds
    - `NX`: Only set if key doesn't exist (set-if-not-exists)
    - `XX`: Only set if key exists (update only)

    **Response:**
    ```
    OK
    ```

    **Design Notes:**

    - Atomic operation (no partial writes)
    - Returns immediately after writing to master (before replication)
    - TTL auto-expires key (frees memory automatically)
    - NX option for distributed locking

    ---

    ### 2. GET (Retrieve Value)

    **Request:**
    ```bash
    GET user:12345
    ```

    **Response:**
    ```json
    {"name":"John","email":"john@example.com"}
    ```

    **Design Notes:**

    - Sub-millisecond latency (data in RAM)
    - Returns null if key doesn't exist or expired
    - Can read from master or replica (eventual consistency)

    ---

    ### 3. DELETE (Remove Key)

    **Request:**
    ```bash
    DEL user:12345
    ```

    **Response:**
    ```
    (integer) 1  # Number of keys deleted
    ```

    ---

    ### 4. INCR/DECR (Atomic Counter)

    **Request:**
    ```bash
    INCR page_views:homepage
    INCRBY cart:12345:total 50
    DECR inventory:item789
    ```

    **Response:**
    ```
    (integer) 1001  # New value after increment
    ```

    **Use Cases:**
    - Rate limiting (count requests per user)
    - Inventory management (decrement stock)
    - Analytics (page view counters)

    ---

    ### 5. Data Structures

    **Lists (LPUSH, RPUSH, LRANGE):**
    ```bash
    LPUSH notifications:user123 "New message from Alice"
    LRANGE notifications:user123 0 10  # Get first 10 notifications
    ```

    **Sets (SADD, SMEMBERS, SISMEMBER):**
    ```bash
    SADD online_users:room5 "user123"
    SMEMBERS online_users:room5  # Get all online users in room
    ```

    **Sorted Sets (ZADD, ZRANGE, ZRANK):**
    ```bash
    ZADD leaderboard 9500 "player123"
    ZREVRANGE leaderboard 0 9 WITHSCORES  # Top 10 players
    ```

    **Hashes (HSET, HGET, HGETALL):**
    ```bash
    HSET user:12345 name "John" email "john@example.com"
    HGETALL user:12345
    ```

    ---

    ### 6. Pub/Sub (Real-time Messaging)

    **Publisher:**
    ```bash
    PUBLISH chat:room5 "Hello everyone!"
    ```

    **Subscriber:**
    ```bash
    SUBSCRIBE chat:room5
    # Blocks and receives messages in real-time
    ```

    **Use Cases:**
    - Chat applications (message broadcasting)
    - Real-time notifications (new order alerts)
    - Cache invalidation (notify all servers of cache clear)

    ---

    ## Database Schema

    ### Internal Data Structures

    **Redis uses internal encoding for efficiency:**

    ```c
    // Key-Value entry (Redis Object)
    typedef struct redisObject {
        unsigned type:4;        // REDIS_STRING, REDIS_LIST, REDIS_SET, etc.
        unsigned encoding:4;    // RAW, INT, ZIPLIST, HASHTABLE, etc.
        unsigned lru:24;        // LRU timestamp (for eviction)
        int refcount;           // Reference count (garbage collection)
        void *ptr;              // Pointer to actual data
    } robj;

    // Hash table for main dictionary
    typedef struct dictEntry {
        void *key;              // Key (string)
        union {
            void *val;
            uint64_t u64;
            int64_t s64;
            double d;
        } v;                    // Value (polymorphic)
        struct dictEntry *next; // Collision chain (linked list)
    } dictEntry;

    // LRU eviction metadata
    typedef struct {
        long long lru_clock;    // Global LRU clock
        long long memory_used;  // Current memory usage
        long long maxmemory;    // Maximum allowed memory
    } evictionState;
    ```

    **Why hash table:**

    - **O(1) average lookups:** GET/SET in constant time
    - **Dynamic resizing:** Incremental rehashing (no blocking)
    - **Low collision rate:** MurmurHash for even distribution

    ---

    ## Data Flow Diagrams

    ### Write Path (SET Operation)

    ```mermaid
    sequenceDiagram
        participant Client
        participant SmartClient
        participant Master
        participant Replica1
        participant Replica2

        Client->>SmartClient: SET user:12345 "data"
        SmartClient->>SmartClient: Hash key (CRC16)<br/>Determine shard
        SmartClient->>Master: SET user:12345 "data"

        Master->>Master: 1. Write to memory (hash table)
        Master->>Master: 2. Write to AOF log (optional)
        Master->>Master: 3. Update LRU metadata
        Master-->>SmartClient: OK (< 1ms)

        Master--)Replica1: Async replication (propagate)
        Master--)Replica2: Async replication (propagate)

        SmartClient-->>Client: OK
    ```

    **Flow Explanation:**

    1. **Hash key** - CRC16(key) % 16384 to determine shard (slot)
    2. **Write to master** - Update in-memory hash table (< 1ms)
    3. **Return immediately** - Don't wait for replication (async)
    4. **Replicate async** - Propagate to replicas (< 100ms lag)
    5. **Optional persistence** - Append to AOF log for durability

    ---

    ### Read Path (GET Operation)

    ```mermaid
    sequenceDiagram
        participant Client
        participant SmartClient
        participant Master
        participant Replica

        Client->>SmartClient: GET user:12345
        SmartClient->>SmartClient: Hash key (CRC16)<br/>Determine shard

        alt Read from Master (default)
            SmartClient->>Master: GET user:12345
            Master->>Master: 1. Lookup in hash table<br/>2. Update LRU timestamp
            Master-->>SmartClient: Value (< 1ms)
        else Read from Replica (optional, scale reads)
            SmartClient->>Replica: GET user:12345
            Replica->>Replica: Lookup in hash table
            Replica-->>SmartClient: Value (may be slightly stale)
        end

        SmartClient-->>Client: Value
    ```

    **Flow Explanation:**

    1. **Hash key** - Determine which shard owns the key
    2. **Read from master or replica** - Master for latest data, replica for scalability
    3. **Lookup in hash table** - O(1) average lookup time
    4. **Update LRU** - Track access time for eviction policy

    ---

    ### Failover Flow (Master Failure)

    ```mermaid
    sequenceDiagram
        participant Master
        participant Replica1
        participant Replica2
        participant Sentinel
        participant Client

        Master->>Master: ❌ Crash (hardware failure)

        Sentinel->>Master: Health check PING
        Master--xSentinel: No response (timeout)

        Sentinel->>Sentinel: Detect failure<br/>Wait for quorum (3 Sentinels)

        Sentinel->>Replica1: Promote to master
        Replica1->>Replica1: SLAVEOF NO ONE<br/>Become master

        Sentinel->>Replica2: SLAVEOF new_master_ip
        Replica2->>Replica1: Start replicating

        Sentinel->>Client: Update topology<br/>new_master: Replica1

        Client->>Replica1: SET user:12345 "data"
        Replica1-->>Client: OK (system recovered)
    ```

    **Failover Time:**

    - **Detection:** 5-10 seconds (health check interval)
    - **Promotion:** 1-2 seconds (replica becomes master)
    - **Total downtime:** 10-15 seconds

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical distributed cache subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Consistent Hashing** | How to distribute keys evenly and minimize redistribution? | Hash ring with virtual nodes |
    | **Eviction Policies** | How to decide which keys to remove when memory is full? | LRU, LFU, or TTL-based eviction |
    | **Replication Strategy** | How to ensure high availability without sacrificing performance? | Async master-replica with Sentinel failover |
    | **Data Structures** | How to implement lists, sets, sorted sets efficiently in memory? | Specialized encodings (ziplist, intset, skiplist) |

    ---

    === "🔄 Consistent Hashing"

        ## The Challenge

        **Problem:** Distribute 1 billion keys across 100 nodes. When adding/removing nodes, minimize key redistribution (expensive).

        **Naive approach (modulo hashing):**

        ```python
        node = hash(key) % num_nodes  # e.g., node = hash("user:12345") % 100
        ```

        **Problem:** Adding 1 node (100 → 101) causes 99% of keys to change nodes!

        ```
        Before: hash("user:12345") % 100 = 42 → Node 42
        After:  hash("user:12345") % 101 = 53 → Node 53 ❌ (moved!)
        ```

        ---

        ## Consistent Hashing Solution

        **Key Idea:** Map both nodes and keys to a circular hash ring (0 to 2^32 - 1). Each key belongs to the first node clockwise.

        **Algorithm:**

        ```python
        class ConsistentHash:
            """Consistent hashing with virtual nodes"""

            def __init__(self, nodes: List[str], virtual_nodes: int = 150):
                """
                Initialize hash ring

                Args:
                    nodes: List of node identifiers (e.g., ["node1", "node2"])
                    virtual_nodes: Number of virtual nodes per physical node
                """
                self.virtual_nodes = virtual_nodes
                self.ring = {}  # hash -> node_id
                self.sorted_keys = []  # Sorted hash values

                for node in nodes:
                    self.add_node(node)

            def add_node(self, node: str):
                """Add node to hash ring with virtual nodes"""
                for i in range(self.virtual_nodes):
                    # Create virtual node identifier
                    virtual_key = f"{node}:vnode{i}"

                    # Hash virtual node to ring position
                    hash_val = self._hash(virtual_key)

                    # Add to ring
                    self.ring[hash_val] = node
                    self.sorted_keys.append(hash_val)

                # Keep ring sorted for binary search
                self.sorted_keys.sort()

            def remove_node(self, node: str):
                """Remove node from hash ring"""
                for i in range(self.virtual_nodes):
                    virtual_key = f"{node}:vnode{i}"
                    hash_val = self._hash(virtual_key)

                    del self.ring[hash_val]
                    self.sorted_keys.remove(hash_val)

            def get_node(self, key: str) -> str:
                """
                Get node responsible for key

                Args:
                    key: Cache key (e.g., "user:12345")

                Returns:
                    Node identifier
                """
                if not self.ring:
                    return None

                # Hash the key
                hash_val = self._hash(key)

                # Find first node clockwise (binary search)
                idx = bisect.bisect_right(self.sorted_keys, hash_val)

                # Wrap around to beginning if past end
                if idx == len(self.sorted_keys):
                    idx = 0

                node_hash = self.sorted_keys[idx]
                return self.ring[node_hash]

            def _hash(self, key: str) -> int:
                """
                Hash function (CRC32 or MD5)

                Returns:
                    Integer hash value (0 to 2^32 - 1)
                """
                return crc32(key.encode()) & 0xFFFFFFFF

        # Usage
        ch = ConsistentHash(["node1", "node2", "node3"], virtual_nodes=150)

        # Distribute keys
        print(ch.get_node("user:12345"))  # node2
        print(ch.get_node("product:789"))  # node1

        # Add node (only 1/4 of keys move)
        ch.add_node("node4")
        print(ch.get_node("user:12345"))  # Still node2 (didn't move!)
        ```

        ---

        ## Virtual Nodes Explained

        **Problem:** With only 3 physical nodes, distribution is uneven (some nodes get 40%, others get 20%).

        **Solution:** Each physical node creates 100-200 virtual nodes on the ring.

        **Example:**

        ```
        Physical: node1, node2, node3

        Virtual:
        - node1:vnode0 → hash(node1:vnode0) = 1000
        - node1:vnode1 → hash(node1:vnode1) = 2500
        - node1:vnode2 → hash(node1:vnode2) = 4000
        - ...
        - node2:vnode0 → hash(node2:vnode0) = 1200
        - node2:vnode1 → hash(node2:vnode1) = 3000
        - ...
        ```

        **Benefits:**

        - **Even distribution:** 150 virtual nodes per physical node = 99% load balance
        - **Smooth redistribution:** When adding node4, keys spread evenly from all 3 existing nodes
        - **Fault tolerance:** If node2 fails, its keys distribute across node1 and node3

        ---

        ## Trade-offs

        | Approach | Key Redistribution | Load Balance | Complexity |
        |----------|-------------------|--------------|------------|
        | **Modulo hashing** | 99% (all keys) | Perfect (1/N) | O(1) - simple |
        | **Consistent hashing (no virtual nodes)** | 1/N (optimal) | Poor (varies 20-40%) | O(log N) - moderate |
        | **Consistent hashing (with virtual nodes)** | 1/N (optimal) | Excellent (< 1% variance) | O(log N) - moderate |

        **Production choice:** Consistent hashing with 150 virtual nodes per physical node.

    === "🗑️ Eviction Policies"

        ## The Challenge

        **Problem:** Cache memory is full (e.g., 15 GB). New SET requires evicting old data. Which key to remove?

        **Requirements:**

        - **Maximize hit rate:** Keep frequently accessed keys
        - **Fast decision:** Eviction must be O(1), not O(N)
        - **Memory efficient:** Eviction metadata should be minimal

        ---

        ## Eviction Algorithms

        ### 1. LRU (Least Recently Used)

        **Strategy:** Evict the key that hasn't been accessed for the longest time.

        **Implementation (Doubly-Linked List + Hash Map):**

        ```python
        class LRUCache:
            """LRU cache with O(1) get and set"""

            def __init__(self, capacity: int):
                self.capacity = capacity
                self.cache = {}  # key -> node
                self.head = Node(0, 0)  # Dummy head (most recent)
                self.tail = Node(0, 0)  # Dummy tail (least recent)
                self.head.next = self.tail
                self.tail.prev = self.head

            def get(self, key: str) -> any:
                """
                Get value and move to front (mark as recently used)

                Time: O(1)
                """
                if key not in self.cache:
                    return None

                node = self.cache[key]

                # Move to front (most recently used)
                self._remove(node)
                self._add_to_front(node)

                return node.value

            def set(self, key: str, value: any):
                """
                Set value, evict LRU if capacity exceeded

                Time: O(1)
                """
                if key in self.cache:
                    # Update existing key
                    node = self.cache[key]
                    node.value = value
                    self._remove(node)
                    self._add_to_front(node)
                else:
                    # New key
                    if len(self.cache) >= self.capacity:
                        # Evict LRU (tail)
                        lru_node = self.tail.prev
                        self._remove(lru_node)
                        del self.cache[lru_node.key]

                    # Add new node
                    new_node = Node(key, value)
                    self.cache[key] = new_node
                    self._add_to_front(new_node)

            def _remove(self, node):
                """Remove node from doubly-linked list"""
                prev_node = node.prev
                next_node = node.next
                prev_node.next = next_node
                next_node.prev = prev_node

            def _add_to_front(self, node):
                """Add node to front (most recent)"""
                node.next = self.head.next
                node.prev = self.head
                self.head.next.prev = node
                self.head.next = node

        class Node:
            def __init__(self, key, value):
                self.key = key
                self.value = value
                self.prev = None
                self.next = None
        ```

        **Pros:**

        - Simple to implement
        - Works well for temporal locality (recently accessed = likely accessed again)
        - O(1) get and set

        **Cons:**

        - Doesn't handle frequency well (one-time popular items evict frequently-used items)
        - Cache pollution from sequential scans

        ---

        ### 2. LFU (Least Frequently Used)

        **Strategy:** Evict the key with the lowest access count.

        **Implementation (Min-Heap + Hash Map):**

        ```python
        from collections import defaultdict
        import heapq

        class LFUCache:
            """LFU cache with O(1) get and O(log N) set"""

            def __init__(self, capacity: int):
                self.capacity = capacity
                self.cache = {}  # key -> (value, freq)
                self.freq_map = defaultdict(set)  # freq -> set of keys
                self.min_freq = 0

            def get(self, key: str) -> any:
                """
                Get value and increment frequency

                Time: O(1) average
                """
                if key not in self.cache:
                    return None

                value, freq = self.cache[key]

                # Update frequency
                self.freq_map[freq].remove(key)
                if not self.freq_map[freq] and freq == self.min_freq:
                    self.min_freq += 1

                new_freq = freq + 1
                self.cache[key] = (value, new_freq)
                self.freq_map[new_freq].add(key)

                return value

            def set(self, key: str, value: any):
                """
                Set value, evict LFU if capacity exceeded

                Time: O(1) average
                """
                if self.capacity <= 0:
                    return

                if key in self.cache:
                    # Update existing key
                    _, freq = self.cache[key]
                    self.cache[key] = (value, freq)
                    self.get(key)  # Update frequency
                else:
                    # Evict if at capacity
                    if len(self.cache) >= self.capacity:
                        # Evict key with min frequency
                        evict_key = self.freq_map[self.min_freq].pop()
                        del self.cache[evict_key]

                    # Add new key
                    self.cache[key] = (value, 1)
                    self.freq_map[1].add(key)
                    self.min_freq = 1
        ```

        **Pros:**

        - Better for frequency-based access patterns
        - Retains popular items even if not recently accessed

        **Cons:**

        - More complex than LRU
        - Historical bias (old popular items stay forever)
        - Cold start problem (new items have low frequency)

        ---

        ### 3. Redis Approximation: Sampling-Based LRU

        **Problem:** Maintaining perfect LRU for 1 billion keys is expensive (memory overhead for doubly-linked list).

        **Redis approach:** Sample 5 random keys, evict the one with oldest LRU timestamp.

        ```c
        // Redis eviction (simplified)
        void evict_lru_keys(redisDb *db) {
            int keys_to_sample = 5;
            long long min_idle_time = LLONG_MAX;
            robj *evict_key = NULL;

            // Sample random keys
            for (int i = 0; i < keys_to_sample; i++) {
                dictEntry *de = dictGetRandomKey(db->dict);
                robj *key = dictGetKey(de);
                robj *obj = dictGetVal(de);

                // Calculate idle time
                long long idle = estimateObjectIdleTime(obj);

                // Track minimum
                if (idle < min_idle_time) {
                    min_idle_time = idle;
                    evict_key = key;
                }
            }

            // Delete least recently used
            if (evict_key) {
                dbDelete(db, evict_key);
            }
        }
        ```

        **Accuracy:** 95% as good as perfect LRU, with 1/10th the memory overhead.

        ---

        ## Comparison

        | Policy | Best For | Hit Rate | Memory Overhead | Complexity |
        |--------|----------|----------|-----------------|------------|
        | **LRU** | Temporal locality (recent access) | 85-90% | Moderate (pointers) | O(1) |
        | **LFU** | Frequency-based access | 90-95% | High (frequency counters) | O(log N) |
        | **TTL** | Time-bound data (sessions, tokens) | N/A | Low (timestamp only) | O(1) |
        | **Random** | No pattern (baseline) | 70-75% | None | O(1) |

        **Production choice:** Redis uses **approximated LRU** (sample 5 keys) for 95% accuracy with minimal overhead.

    === "🔁 Replication Strategy"

        ## The Challenge

        **Problem:** Single-node cache has no redundancy. Hardware failure = 100% cache miss rate (database overload).

        **Requirements:**

        - **High availability:** Automatic failover in < 30 seconds
        - **Data durability:** Minimize data loss during failover
        - **Read scalability:** Distribute read traffic across replicas
        - **Low latency:** Replication shouldn't slow down writes

        ---

        ## Master-Replica Architecture

        **Topology:**

        ```
        Master Node (write + read)
        ├── Replica 1 (read-only)
        ├── Replica 2 (read-only)
        └── Replica 3 (read-only)
        ```

        **Replication Flow:**

        ```python
        class ReplicationManager:
            """Manage master-replica replication"""

            def __init__(self, master_host: str, replicas: List[str]):
                self.master = redis.Redis(host=master_host)
                self.replicas = [redis.Redis(host=r) for r in replicas]
                self.replication_offset = 0  # Track replication progress

            def write(self, key: str, value: str) -> bool:
                """
                Write to master, replicate asynchronously

                Args:
                    key: Cache key
                    value: Cache value

                Returns:
                    True if write succeeded on master
                """
                try:
                    # Write to master (blocks until complete)
                    self.master.set(key, value)

                    # Trigger async replication (non-blocking)
                    self._replicate_async(key, value)

                    return True
                except redis.RedisError as e:
                    logger.error(f"Write failed: {e}")
                    return False

            def _replicate_async(self, key: str, value: str):
                """
                Asynchronous replication to replicas

                Master sends replication log to replicas in background
                """
                # Redis handles this internally via replication backlog
                # Master sends write commands to replicas
                # Replicas apply commands in same order
                pass

            def read(self, key: str, prefer_replica: bool = True) -> str:
                """
                Read from replica (scale reads) or master (latest data)

                Args:
                    key: Cache key
                    prefer_replica: Read from replica if True (may be stale)

                Returns:
                    Value or None if key doesn't exist
                """
                if prefer_replica and self.replicas:
                    # Read from random replica (load balance)
                    replica = random.choice(self.replicas)
                    return replica.get(key)
                else:
                    # Read from master (guaranteed fresh)
                    return self.master.get(key)
        ```

        ---

        ## Replication Lag

        **Problem:** Async replication = replicas lag behind master (50-100ms typical).

        **Scenario:**

        ```
        T0: Client writes key="user:123" to master
        T1: Client reads key="user:123" from replica ❌ Not replicated yet!
        T2: Replication completes (100ms later)
        ```

        **Solutions:**

        1. **Read from master after write** (consistency over performance)
        2. **Sticky sessions** (same client always reads from same replica)
        3. **Eventual consistency** (acceptable for most cache use cases)

        ```python
        def write_then_read(key: str, value: str) -> str:
            """Write to master, read from master (no lag)"""
            self.master.set(key, value)
            return self.master.get(key)  # Read from master (consistent)

        def write_then_read_replica(key: str, value: str) -> str:
            """Write to master, read from replica (may be stale)"""
            self.master.set(key, value)
            time.sleep(0.1)  # Wait for replication (hacky!)
            return self.replicas[0].get(key)
        ```

        ---

        ## Automatic Failover (Redis Sentinel)

        **Sentinel monitors master health and promotes replica on failure.**

        **Architecture:**

        ```
        Sentinel 1 (quorum voter)
        Sentinel 2 (quorum voter)
        Sentinel 3 (quorum voter)
            |
            v
        Master (healthy?)
        Replica 1 (standby)
        Replica 2 (standby)
        ```

        **Failover Process:**

        ```python
        class RedisSentinel:
            """Simplified Sentinel failover logic"""

            def __init__(self, sentinels: List[str], master_name: str):
                self.sentinels = sentinels
                self.master_name = master_name
                self.quorum = len(sentinels) // 2 + 1  # Majority

            def monitor_master(self):
                """
                Continuously monitor master health

                Failover steps:
                1. Detect master failure (PING timeout)
                2. Reach quorum (majority of Sentinels agree)
                3. Promote best replica to master
                4. Update clients with new master IP
                """
                while True:
                    # Ping master every 1 second
                    if not self._ping_master():
                        logger.warning("Master is down!")

                        # Wait for quorum
                        if self._reach_quorum():
                            logger.info("Quorum reached. Starting failover...")
                            self._promote_replica()
                    time.sleep(1)

            def _ping_master(self) -> bool:
                """Check if master is responsive"""
                try:
                    response = self.master.ping()
                    return response == True
                except redis.RedisError:
                    return False

            def _reach_quorum(self) -> bool:
                """
                Check if majority of Sentinels agree master is down

                Returns:
                    True if >= quorum Sentinels vote yes
                """
                votes = 0
                for sentinel in self.sentinels:
                    if sentinel.is_master_down():
                        votes += 1

                return votes >= self.quorum

            def _promote_replica(self):
                """
                Promote best replica to master

                Selection criteria:
                1. Highest replication offset (most up-to-date)
                2. Lowest priority (manual override)
                3. Lexicographical order of run ID (tiebreaker)
                """
                # Get all replicas
                replicas = self.master.sentinel_slaves(self.master_name)

                # Sort by replication offset (most data)
                best_replica = max(replicas, key=lambda r: r['offset'])

                # Promote to master
                logger.info(f"Promoting {best_replica['ip']} to master")
                self._send_command(best_replica, "SLAVEOF NO ONE")

                # Update other replicas to follow new master
                for replica in replicas:
                    if replica != best_replica:
                        self._send_command(replica, f"SLAVEOF {best_replica['ip']} 6379")

                # Notify clients of new master
                self._update_clients(best_replica['ip'])
        ```

        **Failover Time:**

        - **Detection:** 5-10 seconds (PING timeout)
        - **Quorum:** 1-2 seconds (Sentinels vote)
        - **Promotion:** 1-2 seconds (SLAVEOF NO ONE)
        - **Total:** 10-15 seconds downtime

        ---

        ## Trade-offs

        | Strategy | Availability | Consistency | Complexity |
        |----------|--------------|-------------|------------|
        | **No replication** | Low (single point of failure) | Strong (one copy) | Simple |
        | **Async replication (Redis default)** | High (automatic failover) | Eventual (lag < 100ms) | Moderate |
        | **Sync replication (WAIT command)** | Medium (can block on replica failure) | Strong (no lag) | High (slower writes) |
        | **Chain replication** | High (multiple replicas) | Strong (sync) | High (complex) |

        **Production choice:** **Async replication with Sentinel** (99.99% availability, eventual consistency acceptable).

    === "📊 Data Structures"

        ## The Challenge

        **Problem:** Implement lists, sets, sorted sets in memory with minimal overhead.

        **Requirements:**

        - **Space efficient:** Minimize memory per element
        - **Fast operations:** O(1) or O(log N) for common ops
        - **Polymorphic:** Support both small and large collections

        ---

        ## String Encoding

        **Redis optimizes string storage based on content:**

        ```c
        // Redis string encodings
        #define OBJ_ENCODING_RAW 0        // Normal string (> 44 bytes)
        #define OBJ_ENCODING_INT 1        // Integer (stored as long)
        #define OBJ_ENCODING_EMBSTR 2     // Embedded string (<= 44 bytes)

        robj *createStringObject(char *ptr, size_t len) {
            // Try to encode as integer
            long long value;
            if (len <= 20 && string2ll(ptr, len, &value)) {
                return createObject(OBJ_STRING, OBJ_ENCODING_INT, (void*)value);
            }

            // Small string: embed in object (avoid extra allocation)
            if (len <= 44) {
                return createEmbeddedStringObject(ptr, len);
            }

            // Large string: separate allocation
            return createRawStringObject(ptr, len);
        }
        ```

        **Memory savings:**

        - Integer "12345" → 8 bytes (not 5 bytes + overhead)
        - Small string "user:123" → 24 bytes (embedded, no pointer)
        - Large string → 24 bytes (object) + N bytes (string)

        ---

        ## List Encoding (Ziplist vs Linked List)

        **Small lists (<512 elements, <64 bytes per entry): Ziplist (compact array)**

        ```c
        // Ziplist: contiguous memory block
        // Layout: [zlbytes][zltail][zllen][entry1][entry2]...[zlend]
        //
        // Example: LPUSH mylist "foo" "bar" "baz"
        // Memory: [16][12][3]["baz"]["bar"]["foo"][255]
        // Total: ~50 bytes (vs. 200+ bytes for linked list)

        unsigned char *ziplistPush(unsigned char *zl, unsigned char *s, unsigned int slen, int where) {
            // Resize ziplist to fit new entry
            unsigned char *p = ziplistResize(zl, curlen + reqlen);

            // Shift elements if inserting at head
            if (where == ZIPLIST_HEAD) {
                memmove(p + reqlen, p, curlen);
            }

            // Write entry
            p = ziplistInsert(p, s, slen);
            return zl;
        }
        ```

        **Large lists (>512 elements): Quicklist (hybrid of ziplist + linked list)**

        ```c
        // Quicklist: linked list of ziplists
        // Each node contains a ziplist (64KB max)
        //
        // Layout:
        // Node 1 (ziplist 1) <-> Node 2 (ziplist 2) <-> Node 3 (ziplist 3)

        typedef struct quicklistNode {
            struct quicklistNode *prev;
            struct quicklistNode *next;
            unsigned char *zl;          // Ziplist
            unsigned int sz;            // Ziplist size
            unsigned int count : 16;    // Number of items
        } quicklistNode;

        typedef struct quicklist {
            quicklistNode *head;
            quicklistNode *tail;
            unsigned long count;        // Total items
        } quicklist;
        ```

        **Benefits:**

        - Ziplist: 5x memory savings for small lists
        - Quicklist: O(1) head/tail operations, O(N) for middle (acceptable)

        ---

        ## Set Encoding (Intset vs Hash Table)

        **Small integer sets (<512 elements, all integers): Intset (sorted array)**

        ```c
        // Intset: sorted array of integers (binary search)
        typedef struct intset {
            uint32_t encoding;  // INT16, INT32, or INT64
            uint32_t length;    // Number of elements
            int8_t contents[];  // Flexible array
        } intset;

        // Example: SADD myset 1 2 3 4 5
        // Memory: [4][5][1][2][3][4][5]
        // Total: 24 bytes (vs. 200+ bytes for hash table)

        uint8_t intsetSearch(intset *is, int64_t value, uint32_t *pos) {
            // Binary search (O(log N))
            if (is->length == 0) return 0;

            int min = 0, max = is->length - 1;
            while (min <= max) {
                int mid = (min + max) / 2;
                int64_t cur = _intsetGet(is, mid);

                if (value > cur) {
                    min = mid + 1;
                } else if (value < cur) {
                    max = mid - 1;
                } else {
                    *pos = mid;
                    return 1;  // Found
                }
            }
            *pos = min;
            return 0;  // Not found
        }
        ```

        **Large sets: Hash Table**

        ```c
        // Standard hash table with chaining
        dictEntry *entry = dictFind(set->dict, "element");
        if (entry) {
            // Element exists
        }
        ```

        ---

        ## Sorted Set (Skip List + Hash Table)

        **Sorted set requires both:**

        - Fast score-based lookup (ZRANGE by score)
        - Fast member lookup (ZSCORE by member)

        **Solution: Skip list (probabilistic balanced tree) + hash table**

        ```c
        typedef struct zskiplistNode {
            sds ele;                        // Member (string)
            double score;                   // Score (for sorting)
            struct zskiplistNode *backward; // Previous node
            struct zskiplistLevel {
                struct zskiplistNode *forward;
                unsigned long span;         // Rank (for ZRANK)
            } level[];                      // Flexible array (multi-level)
        } zskiplistNode;

        typedef struct zskiplist {
            struct zskiplistNode *header, *tail;
            unsigned long length;
            int level;
        } zskiplist;

        typedef struct zset {
            dict *dict;          // Member -> score (O(1) lookup)
            zskiplist *zsl;      // Skip list (O(log N) range queries)
        } zset;

        // Example: ZADD leaderboard 100 "Alice" 200 "Bob" 150 "Charlie"
        //
        // Skip list (sorted by score):
        // Level 3:  Header ---------------------------------> NULL
        // Level 2:  Header ---------> 150 -----------------> NULL
        // Level 1:  Header --> 100 -> 150 -> 200 ----------> NULL
        //           (Alice)   (Charlie)  (Bob)
        //
        // Hash table:
        // "Alice" -> 100
        // "Bob" -> 200
        // "Charlie" -> 150
        ```

        **Operations:**

        | Operation | Time | Description |
        |-----------|------|-------------|
        | `ZADD key score member` | O(log N) | Insert into skip list and hash table |
        | `ZSCORE key member` | O(1) | Lookup in hash table |
        | `ZRANGE key start stop` | O(log N + M) | Traverse skip list (M = range size) |
        | `ZRANK key member` | O(log N) | Sum spans in skip list |

        **Why skip list over balanced tree (AVL/Red-Black)?**

        - Simpler implementation (no rotations)
        - Better cache locality (fewer pointers)
        - Easier to implement range queries
        - Similar performance (O(log N) average)

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling distributed cache from 10K to 10M ops/sec.

    **Scaling challenges at 10M ops/sec:**

    - **Memory:** 1.5 TB per cluster (100 shards × 15 GB each)
    - **Network:** 80 Gbps total (800 Mbps per shard)
    - **CPU:** Single-threaded Redis (need parallelism)
    - **Replication lag:** Async replication (< 100ms target)

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Single-threaded Redis** | ✅ Yes | 100 shards (parallel processing), 6.0+ has I/O threads |
    | **Network bandwidth** | ✅ Yes | 10 Gbps NICs, pipelining, compression |
    | **Memory capacity** | ✅ Yes | 100 shards × 32 GB RAM, eviction policies |
    | **Replication lag** | 🟡 Approaching | Async replication, Redis 6.0+ uses diskless replication |
    | **Failover time** | ❌ No | Sentinel (< 30s failover), acceptable |

    ---

    ## Performance Optimization

    ### 1. Pipelining (Batch Requests)

    **Problem:** Network round-trip time (RTT) dominates latency. Each request = 1 RTT (e.g., 0.5ms).

    ```
    Without pipelining (10 requests):
    Client -> Server: GET key1 (0.5ms RTT)
    Client -> Server: GET key2 (0.5ms RTT)
    ...
    Total: 10 × 0.5ms = 5ms
    ```

    **Solution:** Send multiple commands in one round trip.

    ```python
    import redis

    # Without pipelining (slow)
    r = redis.Redis()
    for i in range(10000):
        r.get(f"key:{i}")  # 10,000 RTTs!

    # With pipelining (fast)
    pipe = r.pipeline()
    for i in range(10000):
        pipe.get(f"key:{i}")  # Queue command
    results = pipe.execute()  # Single RTT!
    ```

    **Performance:**

    - Without: 10,000 requests × 0.5ms = 5 seconds
    - With: 1 RTT = 0.5ms (10,000x faster!)

    ---

    ### 2. Connection Pooling

    **Problem:** Creating TCP connection for every request is expensive (3-way handshake).

    ```python
    # Bad: new connection per request
    for i in range(1000):
        r = redis.Redis(host='localhost')
        r.get(f"key:{i}")
        r.close()  # Close connection (wasteful!)

    # Good: reuse connections
    pool = redis.ConnectionPool(host='localhost', max_connections=50)
    r = redis.Redis(connection_pool=pool)

    for i in range(1000):
        r.get(f"key:{i}")  # Reuse existing connection
    ```

    **Benefits:**

    - 10x lower latency (no handshake)
    - Higher throughput (fewer syscalls)

    ---

    ### 3. Compression (Large Values)

    **Problem:** 1 MB values consume excessive network bandwidth.

    ```python
    import zlib

    # Compress large values before caching
    def set_compressed(key: str, value: str):
        compressed = zlib.compress(value.encode())
        r.set(key, compressed)

    def get_compressed(key: str) -> str:
        compressed = r.get(key)
        if compressed:
            return zlib.decompress(compressed).decode()
        return None

    # Example: 1 MB JSON document
    large_doc = json.dumps({"data": "..." * 100000})

    # Without compression: 1 MB
    r.set("doc", large_doc)  # 1 MB network transfer

    # With compression: 100 KB (10x smaller)
    set_compressed("doc", large_doc)  # 100 KB network transfer
    ```

    **Trade-off:**

    - CPU cost: 5-10 μs compression time
    - Network savings: 10x smaller (1 MB → 100 KB)
    - Worth it for values > 10 KB

    ---

    ### 4. Hot Key Mitigation

    **Problem:** Single key receives 1M ops/sec (hot celebrity user). Single master can't handle it.

    **Solution 1: Local cache (application-level)**

    ```python
    from functools import lru_cache
    import time

    # Local LRU cache (in-process memory)
    @lru_cache(maxsize=10000)
    def get_user_cached(user_id: str, timestamp: int):
        """
        Cached for 1 second (timestamp = current_second)

        Args:
            user_id: User ID
            timestamp: Current timestamp (second granularity)
        """
        return r.get(f"user:{user_id}")

    # Usage
    current_second = int(time.time())
    user = get_user_cached("celebrity_user", current_second)
    # Next 999,999 requests in same second served from local cache!
    ```

    **Solution 2: Read replicas (distribute reads)**

    ```python
    # Read from replicas (5 replicas = 5x read capacity)
    replicas = [
        redis.Redis(host='replica1'),
        redis.Redis(host='replica2'),
        redis.Redis(host='replica3'),
        redis.Redis(host='replica4'),
        redis.Redis(host='replica5'),
    ]

    def get_from_replica(key: str):
        """Round-robin across replicas"""
        replica = replicas[hash(key) % len(replicas)]
        return replica.get(key)
    ```

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold | Tool |
    |--------|--------|-----------------|------|
    | **GET Latency (P99)** | < 1ms | > 5ms | Redis SLOWLOG, Prometheus |
    | **SET Latency (P99)** | < 2ms | > 10ms | Redis SLOWLOG |
    | **Cache Hit Rate** | > 95% | < 90% | Redis INFO stats |
    | **Evictions/sec** | < 100/sec | > 1000/sec | Redis INFO stats (evicted_keys) |
    | **Memory Usage** | < 80% | > 90% | Redis INFO memory |
    | **Replication Lag** | < 100ms | > 1 second | Redis INFO replication (master_repl_offset) |
    | **Connection Count** | < 1000/node | > 5000/node | Redis INFO clients |

    **Example Prometheus query:**

    ```promql
    # P99 latency
    histogram_quantile(0.99, rate(redis_command_duration_seconds_bucket[5m]))

    # Cache hit rate
    rate(redis_keyspace_hits_total[5m]) /
    (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))

    # Evictions per second
    rate(redis_evicted_keys_total[5m])
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 10M ops/sec:**

    | Component | Cost |
    |-----------|------|
    | **100 master nodes** | $14,400 (100 × r5.2xlarge @ $0.504/hr × 720hr) |
    | **200 replica nodes** | $28,800 (200 × r5.2xlarge) |
    | **Sentinel nodes (9)** | $648 (9 × t3.small) |
    | **Network transfer** | $4,500 (50 TB egress @ $0.09/GB) |
    | **Total** | **$48,348/month** |

    **Optimization:**

    - Use reserved instances (40% discount): $29,000/month
    - Spot instances for non-critical replicas: $25,000/month
    - Compress large values (reduce network cost): $22,000/month

    ---

    ## Disaster Recovery

    **Backup strategies:**

    1. **RDB snapshots** (point-in-time backup)
    2. **AOF logs** (replay all writes)
    3. **Cross-region replication** (geo-redundancy)

    ```bash
    # RDB snapshot (every 5 minutes)
    redis-cli BGSAVE

    # AOF log (every second)
    redis-cli CONFIG SET appendonly yes
    redis-cli CONFIG SET appendfsync everysec

    # Restore from backup
    # 1. Copy RDB file to Redis data dir
    cp dump.rdb /var/lib/redis/
    # 2. Restart Redis
    systemctl restart redis
    ```

    **Recovery time:**

    - RDB: 5-10 minutes (depends on data size)
    - AOF: 10-30 minutes (replay all commands)

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Consistent hashing with virtual nodes:** Even distribution, minimal redistribution
    2. **Master-replica with async replication:** High availability (99.99%), eventual consistency
    3. **LRU eviction (approximated):** Keep hot data, 95% accuracy with low overhead
    4. **Client-side sharding:** Direct routing, no proxy latency
    5. **Specialized encodings:** 5x memory savings (ziplist, intset, embstr)
    6. **Redis Sentinel:** Automatic failover (< 30s downtime)

    ---

    ## Interview Tips

    ✅ **Start with consistent hashing** - Core concept for distributed systems

    ✅ **Discuss eviction policies** - LRU vs LFU trade-offs

    ✅ **Explain replication lag** - Async vs sync replication

    ✅ **Mention Redis data structures** - Skip list for sorted sets, ziplist for small lists

    ✅ **Hot key problem** - Local cache, read replicas

    ✅ **Monitoring is critical** - Cache hit rate, eviction rate, latency

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle hot keys?"** | Local application cache (1 second TTL), read from replicas, shard by sub-key |
    | **"How to ensure data consistency?"** | Eventual consistency (async replication), read from master for critical data |
    | **"What happens during failover?"** | Sentinel detects failure (10s), promotes replica (5s), < 1 second data loss |
    | **"How to evict keys when memory is full?"** | LRU (approximated by sampling 5 keys), LFU for frequency-based, TTL for time-bound |
    | **"Why Redis over Memcached?"** | Data structures (lists, sets, sorted sets), persistence (AOF/RDB), replication |
    | **"How to scale to 100M ops/sec?"** | 10x more shards (1000 shards), pipelining, connection pooling, local cache |

    ---

    ## Real-World Examples

    **Twitter:**
    - 20+ TB of cached data
    - Redis for timelines, user sessions, trending topics
    - 1M ops/sec per cluster

    **Instagram:**
    - Redis for feed generation (fan-out on write)
    - Consistent hashing for photo metadata
    - 5M ops/sec across 100 shards

    **Uber:**
    - Real-time location tracking (geospatial queries)
    - Redis Sorted Sets for leaderboards (driver ratings)
    - 10M ops/sec during peak hours

    **Netflix:**
    - EVCache (Memcached-based)
    - 1 trillion requests/day
    - 100+ clusters globally

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Meta, Google, Amazon, Netflix, Uber, Twitter, DoorDash, Stripe

---

*Master this problem and you'll be ready for: Any system requiring caching, rate limiting, session storage, real-time features*
