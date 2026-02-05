# Design a Key-Value Store

Design a distributed key-value store like Redis or Memcached that supports basic CRUD operations (Create, Read, Update, Delete) with high performance and availability.

**Difficulty:** 🟢 Easy | **Frequency:** ⭐⭐⭐ Medium | **Time:** 30-40 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1M keys, 10K QPS, sub-millisecond latency |
| **Key Challenges** | Fast lookups, memory management, data persistence, eviction policies |
| **Core Concepts** | Hash table, LRU cache, memory optimization, replication |
| **Companies** | All companies (foundational problem) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **PUT** | Store key-value pair | P0 (Must have) |
    | **GET** | Retrieve value by key | P0 (Must have) |
    | **DELETE** | Remove key-value pair | P0 (Must have) |
    | **TTL** | Time-to-live for keys | P1 (Should have) |
    | **Key Expiration** | Auto-delete expired keys | P1 (Should have) |

    **Explicitly Out of Scope:**

    - Complex data structures (lists, sets, sorted sets)
    - Transactions
    - Pub/Sub messaging
    - Replication (for easy version)
    - Persistence to disk

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 1ms p99 | In-memory operations must be fast |
    | **Availability** | 99.9% | Cache should always be available |
    | **Scalability** | 10K QPS per node | Handle high request load |
    | **Memory Efficiency** | Bounded memory | Must evict when full |
    | **Consistency** | Strong consistency | Single node = no consistency issues |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    QPS: 10,000 requests/second
    - GET: 8,000 QPS (80% reads)
    - PUT: 1,500 QPS (15% writes)
    - DELETE: 500 QPS (5% deletes)

    Peak QPS: 3x average = 30,000 QPS
    ```

    ### Storage Estimates

    ```
    Total keys: 1 million
    Average key size: 50 bytes
    Average value size: 1 KB

    Total storage:
    - Keys: 1M × 50 bytes = 50 MB
    - Values: 1M × 1 KB = 1 GB
    - Overhead (pointers, metadata): 200 MB
    - Total: ~1.25 GB

    With safety margin: 2 GB memory
    ```

    ### Bandwidth Estimates

    ```
    GET bandwidth:
    - 8,000 GET/sec × 1 KB = 8 MB/sec ≈ 64 Mbps

    PUT bandwidth:
    - 1,500 PUT/sec × 1 KB = 1.5 MB/sec ≈ 12 Mbps

    Total bandwidth: ~76 Mbps
    ```

    ---

    ## Key Assumptions

    1. Single server (not distributed for easy version)
    2. In-memory storage only
    3. Simple string keys and values
    4. No complex data types
    5. LRU eviction when memory full

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **In-memory storage** - All data in RAM for speed
    2. **Hash table backbone** - O(1) average lookups
    3. **LRU eviction** - Remove least recently used items when full
    4. **Simple API** - GET, PUT, DELETE operations
    5. **Thread-safe** - Handle concurrent requests

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client1[Client 1]
            Client2[Client 2]
            Client3[Client N]
        end

        subgraph "Key-Value Store Server"
            API[API Interface<br/>GET/PUT/DELETE]
            Cache[In-Memory Cache<br/>Hash Map]
            LRU[LRU Eviction<br/>Doubly Linked List]
            Expiry[Expiry Manager<br/>Background Thread]
        end

        Client1 --> API
        Client2 --> API
        Client3 --> API

        API --> Cache
        Cache --> LRU
        Expiry --> Cache

        style Cache fill:#fff4e1
        style LRU fill:#e1f5ff
    ```

    ---

    ## API Design

    ### 1. PUT (Create/Update)

    **Request:**
    ```http
    PUT /api/v1/cache
    Content-Type: application/json

    {
      "key": "user:123",
      "value": "John Doe",
      "ttl": 3600
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "success": true,
      "message": "Key stored successfully"
    }
    ```

    ---

    ### 2. GET (Read)

    **Request:**
    ```http
    GET /api/v1/cache/user:123
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "key": "user:123",
      "value": "John Doe",
      "ttl_remaining": 3540
    }
    ```

    **Response (Not Found):**
    ```http
    HTTP/1.1 404 Not Found
    Content-Type: application/json

    {
      "error": "Key not found"
    }
    ```

    ---

    ### 3. DELETE (Remove)

    **Request:**
    ```http
    DELETE /api/v1/cache/user:123
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "success": true,
      "message": "Key deleted successfully"
    }
    ```

    ---

    ## Core Data Structure

    ### Hash Map + Doubly Linked List (LRU)

    ```python
    class Node:
        """Node in doubly linked list"""
        def __init__(self, key, value, expiry=None):
            self.key = key
            self.value = value
            self.expiry = expiry
            self.prev = None
            self.next = None

    class KeyValueStore:
        """In-memory key-value store with LRU eviction"""

        def __init__(self, capacity: int):
            self.capacity = capacity
            self.cache = {}  # key -> Node
            # Doubly linked list for LRU
            self.head = Node(None, None)  # Dummy head
            self.tail = Node(None, None)  # Dummy tail
            self.head.next = self.tail
            self.tail.prev = self.head
            self.lock = threading.Lock()

        def get(self, key: str) -> str:
            """
            Get value by key

            Returns:
                Value if key exists and not expired, None otherwise
            """
            with self.lock:
                if key not in self.cache:
                    return None

                node = self.cache[key]

                # Check expiry
                if node.expiry and time.time() > node.expiry:
                    self._remove_node(node)
                    del self.cache[key]
                    return None

                # Move to front (most recently used)
                self._move_to_front(node)

                return node.value

        def put(self, key: str, value: str, ttl: int = None):
            """
            Store key-value pair

            Args:
                key: Key string
                value: Value string
                ttl: Time-to-live in seconds (optional)
            """
            with self.lock:
                expiry = time.time() + ttl if ttl else None

                if key in self.cache:
                    # Update existing key
                    node = self.cache[key]
                    node.value = value
                    node.expiry = expiry
                    self._move_to_front(node)
                else:
                    # Add new key
                    if len(self.cache) >= self.capacity:
                        # Evict LRU item
                        lru_node = self.tail.prev
                        self._remove_node(lru_node)
                        del self.cache[lru_node.key]

                    # Create new node
                    node = Node(key, value, expiry)
                    self.cache[key] = node
                    self._add_to_front(node)

        def delete(self, key: str) -> bool:
            """
            Delete key-value pair

            Returns:
                True if key existed and was deleted, False otherwise
            """
            with self.lock:
                if key not in self.cache:
                    return False

                node = self.cache[key]
                self._remove_node(node)
                del self.cache[key]
                return True

        def _add_to_front(self, node: Node):
            """Add node right after head (most recently used)"""
            node.next = self.head.next
            node.prev = self.head
            self.head.next.prev = node
            self.head.next = node

        def _remove_node(self, node: Node):
            """Remove node from linked list"""
            node.prev.next = node.next
            node.next.prev = node.prev

        def _move_to_front(self, node: Node):
            """Move existing node to front"""
            self._remove_node(node)
            self._add_to_front(node)
    ```

    ---

    ## Data Flow

    ### GET Operation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Cache
        participant LRU

        Client->>API: GET key
        API->>Cache: Lookup key in hash map

        alt Key exists
            Cache-->>API: Return node
            API->>API: Check expiry
            alt Not expired
                API->>LRU: Move to front (MRU)
                API-->>Client: 200 OK (value)
            else Expired
                API->>Cache: Remove key
                API-->>Client: 404 Not Found
            end
        else Key not found
            Cache-->>API: Key not found
            API-->>Client: 404 Not Found
        end
    ```

    ### PUT Operation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Cache
        participant LRU

        Client->>API: PUT key, value, ttl
        API->>Cache: Check if key exists

        alt Key exists
            Cache-->>API: Return existing node
            API->>API: Update value, expiry
            API->>LRU: Move to front (MRU)
        else Key not found
            alt Cache full
                API->>LRU: Get LRU node (tail)
                API->>Cache: Delete LRU key
            end
            API->>API: Create new node
            API->>Cache: Add to hash map
            API->>LRU: Add to front (MRU)
        end

        API-->>Client: 200 OK
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. Hash Function

    **Good hash function properties:**
    - Deterministic
    - Uniform distribution
    - Fast to compute

    ```python
    def hash_key(key: str, table_size: int) -> int:
        """Simple hash function"""
        hash_value = 0
        for char in key:
            hash_value = (hash_value * 31 + ord(char)) % table_size
        return hash_value
    ```

    ### 2. Expiry Management

    **Two approaches:**

    **Lazy expiration (on access):**
    - Check expiry when key is accessed
    - Simple, no background work
    - Expired keys may sit in memory

    **Active expiration (background thread):**
    ```python
    def expire_keys_worker(store: KeyValueStore):
        """Background thread to actively expire keys"""
        while True:
            time.sleep(1)  # Check every second

            now = time.time()
            expired_keys = []

            with store.lock:
                for key, node in store.cache.items():
                    if node.expiry and now > node.expiry:
                        expired_keys.append(key)

            # Delete expired keys
            for key in expired_keys:
                store.delete(key)
    ```

    ### 3. LRU Eviction Policy

    **Why LRU?**
    - Simple to implement
    - Good hit rate in practice
    - O(1) operations (with hash map + linked list)

    **Alternatives:**
    - **LFU (Least Frequently Used):** Better for some workloads, more complex
    - **Random eviction:** Simpler but worse hit rate
    - **FIFO:** Doesn't account for access patterns

    ### 4. Thread Safety

    **Concurrency concerns:**
    - Multiple clients accessing simultaneously
    - Read-write conflicts
    - Linked list corruption

    **Solution:** Use locks (mutex)
    - Lock during entire operation
    - Simple but may bottleneck at high QPS

    **Better solution (advanced):** Sharding
    - Multiple independent caches
    - Hash key to determine shard
    - Reduces lock contention

=== "⚡ Step 4: Scale & Optimize"

    ## Performance Optimization

    ### 1. Memory Optimization

    **Techniques:**
    - Use compact data structures
    - Compress large values
    - Set appropriate capacity limits

    ### 2. Sharding for Concurrency

    ```python
    class ShardedKeyValueStore:
        """Sharded KV store to reduce lock contention"""

        def __init__(self, capacity: int, num_shards: int = 16):
            self.num_shards = num_shards
            self.shards = [
                KeyValueStore(capacity // num_shards)
                for _ in range(num_shards)
            ]

        def _get_shard(self, key: str) -> KeyValueStore:
            """Determine which shard handles this key"""
            shard_id = hash(key) % self.num_shards
            return self.shards[shard_id]

        def get(self, key: str) -> str:
            return self._get_shard(key).get(key)

        def put(self, key: str, value: str, ttl: int = None):
            self._get_shard(key).put(key, value, ttl)

        def delete(self, key: str) -> bool:
            return self._get_shard(key).delete(key)
    ```

    **Benefits:**
    - 16 shards = 16x concurrency
    - Each shard has its own lock
    - Near-linear scalability

    ### 3. Monitoring Metrics

    | Metric | Target | Why It Matters |
    |--------|--------|----------------|
    | **Hit Rate** | > 80% | Cache effectiveness |
    | **Latency (p99)** | < 1ms | User experience |
    | **Memory Usage** | < 90% | Avoid thrashing |
    | **Eviction Rate** | < 10% | Capacity planning |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Hash map for storage** - O(1) lookups
    2. **Doubly linked list for LRU** - O(1) eviction
    3. **Lazy + active expiration** - Hybrid approach
    4. **Thread-safe with locks** - Simple concurrency
    5. **In-memory only** - Maximum performance

    ## Interview Tips

    ✅ **Start simple** - Single node, in-memory
    ✅ **Explain data structures** - Hash map + linked list
    ✅ **Discuss eviction** - LRU algorithm
    ✅ **Consider expiration** - TTL handling
    ✅ **Address concurrency** - Thread safety

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle cache misses?"** | Return null, let client handle (fetch from DB, etc.) |
    | **"What if memory is full?"** | LRU eviction removes least recently used item |
    | **"How to make it distributed?"** | Consistent hashing, replication (harder problem) |
    | **"How to persist data?"** | Write-ahead log, snapshots (like Redis) |

---

**Difficulty:** 🟢 Easy | **Interview Time:** 30-40 minutes | **Companies:** All companies

---

*This is a foundational problem that helps understand caching, data structures, and memory management.*
