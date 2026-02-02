# Design Message Queue (Kafka)

A distributed streaming platform that enables publish-subscribe messaging with high throughput, fault tolerance, and ordering guarantees. Used for building real-time data pipelines and streaming applications.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1 trillion messages/day, millions of topics, TB/sec throughput, petabytes of data |
| **Key Challenges** | Ordering guarantees, partitioning strategy, consumer groups, exactly-once delivery, replication |
| **Core Concepts** | Partitions, consumer groups, offset management, replication, log compaction, zero-copy |
| **Companies** | LinkedIn (Kafka), Confluent, AWS (Kinesis/MSK), Google (Pub/Sub), Apache Pulsar, RabbitMQ |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Publish Messages** | Producers publish messages to topics | P0 (Must have) |
    | **Subscribe Messages** | Consumers subscribe to topics and consume messages | P0 (Must have) |
    | **Topic Management** | Create/delete topics with partitions | P0 (Must have) |
    | **Consumer Groups** | Multiple consumers coordinate to process partitions | P0 (Must have) |
    | **Offset Management** | Track consumer position in each partition | P0 (Must have) |
    | **Retention Policy** | Time-based and size-based retention | P0 (Must have) |
    | **Replication** | Data replicated across multiple brokers | P0 (Must have) |
    | **Partitioning** | Messages distributed across partitions by key | P1 (Should have) |
    | **Log Compaction** | Retain only latest value per key | P1 (Should have) |
    | **Dead Letter Queue** | Handle failed messages | P2 (Nice to have) |
    | **Schema Registry** | Manage message schemas | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Message transformation (use Kafka Streams separately)
    - Complex routing rules (use dedicated router)
    - Built-in ML/analytics (use external tools)
    - Message priority queues
    - Transactional outbox pattern implementation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Throughput** | Millions of messages/sec | High-volume data pipelines |
    | **Latency (Publish)** | < 10ms p95 | Fast event ingestion |
    | **Latency (Consume)** | < 100ms p95 | Near real-time processing |
    | **Availability** | 99.99% uptime | Critical infrastructure component |
    | **Durability** | No data loss (replicated) | Messages must survive broker failures |
    | **Ordering** | Per-partition ordering | Guarantee order within partition |
    | **Scalability** | Horizontal scaling (add brokers) | Handle growing data volumes |
    | **Delivery Semantics** | At-least-once (default), exactly-once (optional) | Configurable based on use case |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Messages: 1 trillion (1T)
    Topics: 10,000 topics
    Partitions per topic: 100 (average)
    Total partitions: 1 million

    Write throughput:
    - Messages/sec: 1T / 86,400 = ~11.6M messages/sec
    - Peak QPS: 3x average = ~35M messages/sec (during events)

    Read throughput (assume 3 consumers per partition):
    - Messages/sec: 11.6M × 3 = ~35M messages/sec
    - Total events processed: 3T reads/day

    Average message size: 1 KB (includes key, value, headers)

    Consumer lag:
    - Target lag: < 1 minute for real-time consumers
    - Batch consumers: lag up to 1 hour acceptable
    ```

    ### Storage Estimates

    ```
    Message storage:
    - Messages/day: 1T
    - Average size: 1 KB
    - Daily ingress: 1T × 1 KB = 1 PB/day

    Retention (7 days default):
    - Storage: 1 PB × 7 = 7 PB
    - With replication (3x): 7 PB × 3 = 21 PB

    Long-term retention (90 days for some topics):
    - Critical topics: 20% × 90 days = 18 PB
    - With replication: 18 PB × 3 = 54 PB

    Log segments:
    - Segment size: 1 GB (default)
    - Segments/day: 1 PB / 1 GB = 1 million segments
    - Index overhead: 10 MB per segment = 10 TB

    Total storage: 21 PB (standard) + 54 PB (long retention) + 10 TB (index) ≈ 75 PB
    ```

    ### Bandwidth Estimates

    ```
    Write bandwidth:
    - 11.6M messages/sec × 1 KB = 11.6 GB/sec ≈ 93 Gbps
    - Peak: 35M × 1 KB = 35 GB/sec ≈ 280 Gbps

    Read bandwidth (3x due to consumer groups):
    - 35M messages/sec × 1 KB = 35 GB/sec ≈ 280 Gbps
    - Peak: 105 GB/sec ≈ 840 Gbps

    Replication bandwidth (within cluster):
    - 11.6 GB/sec × 2 (leader to 2 replicas) = 23.2 GB/sec ≈ 186 Gbps

    Total ingress: ~93 Gbps (write) + 186 Gbps (replication) = 279 Gbps
    Total egress: ~280 Gbps (read)

    Network requirements: 10-40 Gbps per broker
    ```

    ### Memory Estimates (Per Broker)

    ```
    Page cache (critical for performance):
    - Kafka relies heavily on OS page cache
    - Per broker: 256 GB RAM (most for page cache)
    - 100 brokers × 256 GB = 25.6 TB total

    Metadata:
    - Topic metadata: 10K topics × 100 partitions × 1 KB = 1 GB
    - Producer state: 100K producers × 10 KB = 1 GB
    - Consumer group metadata: 10K groups × 1 MB = 10 GB
    - Per broker: ~12 GB

    In-flight requests:
    - 10K concurrent requests × 100 KB = 1 GB

    Total per broker: 256 GB (page cache) + 12 GB (metadata) + 1 GB (requests) ≈ 269 GB
    ```

    ---

    ## Key Assumptions

    1. Average message size is 1 KB (mix of small and large messages)
    2. 1 trillion messages per day (high-volume scenario)
    3. 7-day retention for most topics, 90-day for critical topics
    4. Replication factor of 3 for durability
    5. At-least-once delivery as default (exactly-once opt-in)
    6. Consumers use consumer groups for parallel processing
    7. Partitioning by key maintains ordering per key

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Distributed commit log:** Immutable, append-only log structure
    2. **Partitioning for parallelism:** Scale by adding partitions
    3. **Consumer groups:** Multiple consumers coordinate processing
    4. **Replication for durability:** Leader-follower replication
    5. **Zero-copy transfer:** Efficient data movement using sendfile()
    6. **Sequential I/O:** Disk writes are sequential (fast)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Producer Layer"
            P1[Producer 1<br/>Order Service]
            P2[Producer 2<br/>Payment Service]
            P3[Producer 3<br/>User Service]
        end

        subgraph "Kafka Cluster"
            ZK[ZooKeeper/KRaft<br/>Cluster coordination]

            subgraph "Broker 1"
                B1_L1[Partition 0<br/>Leader]
                B1_F2[Partition 1<br/>Follower]
                B1_F3[Partition 2<br/>Follower]
            end

            subgraph "Broker 2"
                B2_F1[Partition 0<br/>Follower]
                B2_L2[Partition 1<br/>Leader]
                B2_F3[Partition 2<br/>Follower]
            end

            subgraph "Broker 3"
                B3_F1[Partition 0<br/>Follower]
                B3_F2[Partition 1<br/>Follower]
                B3_L3[Partition 2<br/>Leader]
            end
        end

        subgraph "Consumer Layer"
            subgraph "Consumer Group 1"
                C1_1[Consumer 1<br/>Process P0]
                C1_2[Consumer 2<br/>Process P1]
                C1_3[Consumer 3<br/>Process P2]
            end

            subgraph "Consumer Group 2"
                C2_1[Consumer 1<br/>Process P0,P1,P2]
            end
        end

        subgraph "Storage"
            Disk1[(Disk 1<br/>Log segments)]
            Disk2[(Disk 2<br/>Log segments)]
            Disk3[(Disk 3<br/>Log segments)]
        end

        subgraph "Monitoring"
            Monitor[Prometheus<br/>Metrics]
            Schema[Schema Registry<br/>Avro/Protobuf]
        end

        P1 --> B1_L1
        P2 --> B2_L2
        P3 --> B3_L3

        ZK -.->|Metadata| B1_L1
        ZK -.->|Metadata| B2_L2
        ZK -.->|Metadata| B3_L3

        B1_L1 -->|Replicate| B2_F1
        B1_L1 -->|Replicate| B3_F1

        B2_L2 -->|Replicate| B1_F2
        B2_L2 -->|Replicate| B3_F2

        B3_L3 -->|Replicate| B1_F3
        B3_L3 -->|Replicate| B2_F3

        B1_L1 --> Disk1
        B2_L2 --> Disk2
        B3_L3 --> Disk3

        B1_L1 --> C1_1
        B2_L2 --> C1_2
        B3_L3 --> C1_3

        B1_L1 --> C2_1
        B2_L2 --> C2_1
        B3_L3 --> C2_1

        B1_L1 --> Monitor
        B2_L2 --> Monitor
        B3_L3 --> Monitor

        P1 -.->|Schema validation| Schema
        C1_1 -.->|Schema validation| Schema

        style B1_L1 fill:#90EE90
        style B2_L2 fill:#90EE90
        style B3_L3 fill:#90EE90
        style B2_F1 fill:#FFB6C1
        style B3_F1 fill:#FFB6C1
        style B1_F2 fill:#FFB6C1
        style B3_F2 fill:#FFB6C1
        style B1_F3 fill:#FFB6C1
        style B2_F3 fill:#FFB6C1
        style ZK fill:#e1f5ff
        style Disk1 fill:#ffe1e1
        style Disk2 fill:#ffe1e1
        style Disk3 fill:#ffe1e1
        style Monitor fill:#fff4e1
        style Schema fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Partitions** | Parallelism, horizontal scaling, ordering within partition | Single queue (no parallelism), sharding (complex) |
    | **Replication** | Durability, fault tolerance, high availability | RAID (single point of failure), external backup (slow recovery) |
    | **Consumer Groups** | Load balancing, automatic rebalancing, fault tolerance | Manual assignment (no auto-recovery), competing consumers (no coordination) |
    | **ZooKeeper/KRaft** | Cluster coordination, leader election, metadata | Custom consensus (complex), external DB (network overhead) |
    | **Sequential I/O** | High throughput (500 MB/sec on disk), predictable performance | Random I/O (10x slower), in-memory only (expensive, volatile) |
    | **Zero-copy** | Eliminate CPU overhead (sendfile syscall), 2-4x throughput | User-space copying (slow), compression only (CPU intensive) |

    **Key Trade-off:** We chose **availability and throughput over strong consistency**. Messages may be delivered more than once, but system remains available and fast.

    ---

    ## API Design

    ### 1. Publish Message

    **Request:**
    ```http
    POST /api/v1/topics/orders/messages
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "key": "order_12345",           // Optional (for partitioning)
      "value": {                       // Message payload
        "order_id": "12345",
        "user_id": "user_789",
        "amount": 99.99,
        "status": "pending"
      },
      "headers": {                     // Optional metadata
        "correlation_id": "req_abc123",
        "source": "order-service"
      },
      "partition": null,               // Optional (auto if null)
      "timestamp": null                // Optional (server time if null)
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "topic": "orders",
      "partition": 5,
      "offset": 1234567890,
      "timestamp": "2026-02-02T10:30:00.123Z",
      "key": "order_12345"
    }
    ```

    **Design Notes:**

    - Return immediately after leader write (don't wait for replication by default)
    - Partitioning: hash(key) % num_partitions
    - Batching: producers batch messages for efficiency (linger.ms config)
    - Idempotence: optional producer ID prevents duplicates

    ---

    ### 2. Consume Messages

    **Request:**
    ```http
    GET /api/v1/topics/orders/messages?group_id=order-processor&timeout=30000
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "messages": [
        {
          "topic": "orders",
          "partition": 5,
          "offset": 1234567890,
          "timestamp": "2026-02-02T10:30:00.123Z",
          "key": "order_12345",
          "value": {
            "order_id": "12345",
            "user_id": "user_789",
            "amount": 99.99,
            "status": "pending"
          },
          "headers": {
            "correlation_id": "req_abc123",
            "source": "order-service"
          }
        },
        // ... more messages
      ],
      "next_offsets": {
        "5": 1234567900,
        "6": 9876543210
      }
    }
    ```

    **Design Notes:**

    - Long polling: blocks until messages available or timeout
    - Consumer group coordination: auto-assign partitions
    - Offset commit: manual or auto (configurable)
    - Max poll records: limit batch size (max.poll.records)

    ---

    ### 3. Commit Offset

    **Request:**
    ```http
    POST /api/v1/consumer-groups/order-processor/offsets
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "topic": "orders",
      "partitions": [
        {
          "partition": 5,
          "offset": 1234567900,
          "metadata": "processed batch 123"
        },
        {
          "partition": 6,
          "offset": 9876543210
        }
      ]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "committed": [
        {
          "partition": 5,
          "offset": 1234567900
        },
        {
          "partition": 6,
          "offset": 9876543210
        }
      ]
    }
    ```

    **Design Notes:**

    - Store offsets in internal __consumer_offsets topic
    - Commit after processing (at-least-once)
    - Commit before processing (at-most-once)
    - Transactional commit (exactly-once)

    ---

    ### 4. Create Topic

    **Request:**
    ```http
    POST /api/v1/topics
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "name": "orders",
      "num_partitions": 100,
      "replication_factor": 3,
      "config": {
        "retention.ms": "604800000",        // 7 days
        "retention.bytes": "1073741824",    // 1 GB per partition
        "cleanup.policy": "delete",         // or "compact"
        "compression.type": "snappy",       // or "gzip", "lz4", "zstd"
        "min.insync.replicas": "2"          // durability guarantee
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "topic": "orders",
      "num_partitions": 100,
      "replication_factor": 3,
      "partitions": [
        {
          "partition": 0,
          "leader": "broker-1",
          "replicas": ["broker-1", "broker-2", "broker-3"],
          "isr": ["broker-1", "broker-2", "broker-3"]
        },
        // ... 99 more partitions
      ]
    }
    ```

    ---

    ## Data Model

    ### Message Format (On Disk)

    ```
    Message structure (v2 format):
    ┌─────────────────────────────────────────────────┐
    │ Offset: 8 bytes (int64)                         │
    ├─────────────────────────────────────────────────┤
    │ Message Size: 4 bytes (int32)                   │
    ├─────────────────────────────────────────────────┤
    │ CRC: 4 bytes (checksum)                         │
    ├─────────────────────────────────────────────────┤
    │ Magic Byte: 1 byte (version)                    │
    ├─────────────────────────────────────────────────┤
    │ Attributes: 1 byte (compression, timestamp type)│
    ├─────────────────────────────────────────────────┤
    │ Timestamp: 8 bytes (int64)                      │
    ├─────────────────────────────────────────────────┤
    │ Key Length: 4 bytes (int32)                     │
    ├─────────────────────────────────────────────────┤
    │ Key: variable (bytes)                           │
    ├─────────────────────────────────────────────────┤
    │ Value Length: 4 bytes (int32)                   │
    ├─────────────────────────────────────────────────┤
    │ Value: variable (bytes)                         │
    ├─────────────────────────────────────────────────┤
    │ Headers Count: 4 bytes (int32)                  │
    ├─────────────────────────────────────────────────┤
    │ Headers: variable (key-value pairs)             │
    └─────────────────────────────────────────────────┘

    Overhead: ~34 bytes + key + value + headers
    ```

    ---

    ### Partition Log Structure

    ```
    Topic: orders, Partition: 5

    /kafka-data/orders-5/
    ├── 00000000000000000000.log         (segment 0, offset 0-999999)
    ├── 00000000000000000000.index       (offset index for segment 0)
    ├── 00000000000000000000.timeindex   (timestamp index for segment 0)
    ├── 00000000000001000000.log         (segment 1, offset 1000000-1999999)
    ├── 00000000000001000000.index
    ├── 00000000000001000000.timeindex
    ├── 00000000000002000000.log         (active segment)
    ├── 00000000000002000000.index
    ├── 00000000000002000000.timeindex
    └── leader-epoch-checkpoint          (leader epoch history)

    Segment naming: {base_offset}.log
    Segment size: 1 GB (default)
    Index entry: every 4 KB of data
    ```

    ---

    ### Consumer Group Metadata

    ```json
    {
      "group_id": "order-processor",
      "state": "Stable",                // Empty, PreparingRebalance, CompletingRebalance, Stable, Dead
      "protocol_type": "consumer",
      "protocol": "range",              // range, roundrobin, sticky, cooperative-sticky
      "leader": "consumer-1",
      "members": [
        {
          "member_id": "consumer-1-uuid",
          "client_id": "consumer-1",
          "client_host": "10.0.1.5",
          "session_timeout": 30000,
          "rebalance_timeout": 300000,
          "assignment": {
            "orders": [0, 1, 2, 33, 34, 35, 66, 67, 68]  // 9 partitions
          }
        },
        {
          "member_id": "consumer-2-uuid",
          "client_id": "consumer-2",
          "client_host": "10.0.1.6",
          "assignment": {
            "orders": [3, 4, 5, 36, 37, 38, 69, 70, 71]
          }
        }
        // ... more members
      ],
      "generation": 42,                 // Rebalance generation
      "offsets": {
        "orders": {
          "0": 1234567890,
          "1": 9876543210,
          // ... offset per partition
        }
      }
    }
    ```

    ---

    ## Data Flow Diagrams

    ### Message Publish Flow

    ```mermaid
    sequenceDiagram
        participant Producer
        participant Broker_Leader
        participant Broker_Follower1
        participant Broker_Follower2
        participant Disk
        participant ZK as ZooKeeper

        Producer->>Broker_Leader: 1. Send message batch (topic, partition, key, value)
        Broker_Leader->>Broker_Leader: 2. Validate, assign offset
        Broker_Leader->>Disk: 3. Append to log (sequential write)

        par Replication
            Broker_Leader->>Broker_Follower1: 4. Replicate message
            Broker_Leader->>Broker_Follower2: 4. Replicate message
        end

        Broker_Follower1->>Disk: 5. Append to log
        Broker_Follower2->>Disk: 5. Append to log

        Broker_Follower1->>Broker_Leader: 6. ACK (in-sync)
        Broker_Follower2->>Broker_Leader: 6. ACK (in-sync)

        Broker_Leader->>Broker_Leader: 7. Update high watermark
        Broker_Leader-->>Producer: 8. ACK (offset, partition)

        Note over Producer,Broker_Leader: acks=1: ACK after leader write<br/>acks=all: ACK after all ISR replicas write
    ```

    **Flow Explanation:**

    1. **Producer batching** - Batch messages for efficiency (16 KB default)
    2. **Leader validation** - Check schema, assign monotonic offset
    3. **Sequential write** - Append to active segment (500+ MB/sec)
    4. **Replication** - Send to all replicas in parallel
    5. **Follower write** - Append to local log
    6. **ISR tracking** - Only in-sync replicas ACK
    7. **High watermark** - Last offset replicated to all ISR
    8. **Producer ACK** - Return offset to producer

    ---

    ### Message Consume Flow

    ```mermaid
    sequenceDiagram
        participant Consumer
        participant Coordinator
        participant Broker_Leader
        participant Disk
        participant PageCache

        Consumer->>Coordinator: 1. Join group (group_id, topics)
        Coordinator->>Coordinator: 2. Trigger rebalance
        Coordinator->>Consumer: 3. Assign partitions [P0, P5, P10]

        Consumer->>Broker_Leader: 4. Fetch request (partition, offset, max_bytes)

        alt Data in page cache (hot path)
            Broker_Leader->>PageCache: 5. Read from OS page cache
            PageCache-->>Broker_Leader: Messages
        else Cache miss (cold path)
            Broker_Leader->>Disk: 5. Read from disk
            Disk->>PageCache: Load into cache
            PageCache-->>Broker_Leader: Messages
        end

        Broker_Leader->>Broker_Leader: 6. Zero-copy sendfile()
        Broker_Leader-->>Consumer: 7. Message batch

        Consumer->>Consumer: 8. Process messages
        Consumer->>Coordinator: 9. Commit offset (async)

        Note over Consumer,Broker_Leader: Zero-copy: no user-space copying<br/>Page cache: 80%+ hit rate
    ```

    **Flow Explanation:**

    1. **Group join** - Consumer registers with coordinator
    2. **Rebalance** - Partition assignment across consumers
    3. **Assignment** - Consumer receives partition list
    4. **Fetch request** - Pull messages from leader
    5. **Page cache** - OS caches recent segments (hot data)
    6. **Zero-copy** - sendfile() transfers data kernel→socket
    7. **Batch delivery** - Return up to max.poll.records
    8. **Processing** - Consumer business logic
    9. **Offset commit** - Track progress

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Kafka subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Partitioning Strategy** | How to distribute messages for parallelism and ordering? | Hash-based partitioning with key-based ordering |
    | **Consumer Groups** | How to coordinate multiple consumers? | Group coordinator + rebalance protocol |
    | **Replication & Durability** | How to prevent data loss? | ISR-based replication with configurable acks |
    | **Exactly-Once Delivery** | How to guarantee no duplicates? | Idempotent producer + transactional API |

    ---

    === "🔀 Partitioning Strategy"

        ## The Challenge

        **Problem:** Distribute 11.6M messages/sec across cluster while maintaining ordering guarantees.

        **Naive approach:** Single queue for topic. **Doesn't scale** (bottleneck, no parallelism).

        **Requirements:**

        - **Parallelism:** Multiple consumers process simultaneously
        - **Ordering:** Preserve order for related messages (same user, same order)
        - **Scalability:** Add partitions without downtime
        - **Load balancing:** Distribute messages evenly

        ---

        ## Partitioning Implementation

        ```python
        import hashlib
        from typing import Optional, List

        class Partitioner:
            """Determine which partition a message belongs to"""

            def __init__(self, num_partitions: int):
                self.num_partitions = num_partitions

            def get_partition(
                self,
                topic: str,
                key: Optional[bytes],
                value: bytes,
                cluster_metadata: dict
            ) -> int:
                """
                Calculate partition for message

                Args:
                    topic: Topic name
                    key: Message key (for partitioning)
                    value: Message value
                    cluster_metadata: Available partitions

                Returns:
                    Partition number (0 to num_partitions-1)
                """
                available_partitions = cluster_metadata.get('available_partitions', [])

                if key is None:
                    # No key: round-robin across available partitions
                    return self._round_robin_partition(available_partitions)
                else:
                    # Key specified: hash-based partitioning (guarantees ordering per key)
                    return self._hash_partition(key, available_partitions)

            def _hash_partition(self, key: bytes, partitions: List[int]) -> int:
                """
                Hash-based partitioning (default Kafka strategy)

                Ensures:
                - Same key always goes to same partition (ordering)
                - Even distribution across partitions
                """
                # MurmurHash2 (Kafka's default)
                hash_value = self._murmur2_hash(key)

                # Modulo to get partition
                partition_idx = abs(hash_value) % len(partitions)
                return partitions[partition_idx]

            def _murmur2_hash(self, key: bytes) -> int:
                """
                MurmurHash2 implementation (Kafka default)

                Fast, non-cryptographic hash with good distribution
                """
                m = 0x5bd1e995
                seed = 0x9747b28c

                length = len(key)
                h = seed ^ length

                # Process 4 bytes at a time
                for i in range(0, length - 3, 4):
                    k = (key[i] & 0xff) + \
                        ((key[i+1] & 0xff) << 8) + \
                        ((key[i+2] & 0xff) << 16) + \
                        ((key[i+3] & 0xff) << 24)

                    k = (k * m) & 0xffffffff
                    k ^= k >> 24
                    k = (k * m) & 0xffffffff

                    h = (h * m) & 0xffffffff
                    h ^= k

                # Handle remaining bytes
                remaining = length % 4
                if remaining >= 3:
                    h ^= (key[length-3] & 0xff) << 16
                if remaining >= 2:
                    h ^= (key[length-2] & 0xff) << 8
                if remaining >= 1:
                    h ^= key[length-1] & 0xff
                    h = (h * m) & 0xffffffff

                # Final mix
                h ^= h >> 13
                h = (h * m) & 0xffffffff
                h ^= h >> 15

                return h

            def _round_robin_partition(self, partitions: List[int]) -> int:
                """
                Round-robin partitioning (no key)

                No ordering guarantee, but even distribution
                """
                # In practice, use sticky partitioning (batch to same partition)
                # to improve batching efficiency
                import random
                return random.choice(partitions)
        ```

        ---

        ## Partition Rebalancing

        **Challenge:** Add partitions to existing topic without disrupting ordering.

        **Problem:** hash(key) % 10 != hash(key) % 15 (if we add 5 partitions)

        **Solution:**

        ```python
        class PartitionExpander:
            """Handle partition expansion without breaking ordering"""

            def expand_partitions(
                self,
                topic: str,
                current_partitions: int,
                new_partitions: int
            ):
                """
                Expand partitions while maintaining ordering

                Strategy:
                1. Only ADD partitions (never reduce)
                2. Existing keys stay in same partition
                3. New keys distributed across all partitions

                Note: Kafka doesn't automatically rebalance existing data.
                Ordering preserved because same key always hashes to same partition.
                """
                if new_partitions < current_partitions:
                    raise ValueError("Cannot reduce partitions (breaks ordering)")

                # Create new partitions
                for partition_id in range(current_partitions, new_partitions):
                    self._create_partition(topic, partition_id)

                logger.info(f"Expanded {topic} from {current_partitions} to {new_partitions} partitions")

                # Note: Existing messages stay in original partitions
                # New messages with same key go to same partition (hash unchanged)

            def _create_partition(self, topic: str, partition_id: int):
                """Create new partition on cluster"""
                # 1. Select brokers for replicas (spread across racks)
                # 2. Create log directory
                # 3. Assign leader and followers
                # 4. Update topic metadata
                pass
        ```

        **Trade-off:** Cannot reduce partitions once created (breaks ordering guarantees).

        ---

        ## Partitioning Strategies

        | Strategy | Use Case | Pros | Cons |
        |----------|----------|------|------|
        | **Key-based hash** | Ordering per key (user_id, order_id) | Ordering guarantee, even distribution | Hot keys cause skew |
        | **Round-robin** | No ordering needed | Perfect distribution | No ordering |
        | **Custom partitioner** | Special logic (geo, priority) | Flexible | Complex, can cause skew |
        | **Sticky partitioning** | Better batching (null key) | Improved throughput | No ordering |

        ---

        ## Handling Hot Partitions

        **Problem:** One key has 100x more messages (celebrity user, popular product).

        **Solutions:**

        ```python
        class HotKeyPartitioner:
            """Handle hot keys that overwhelm single partition"""

            def __init__(self, num_partitions: int, hot_keys: set):
                self.num_partitions = num_partitions
                self.hot_keys = hot_keys  # Known hot keys

            def get_partition(self, key: bytes, value: bytes) -> int:
                """
                Partition with hot key handling

                For hot keys:
                - Add suffix to key (user_123_0, user_123_1, ...)
                - Distribute across multiple partitions
                - Consumer reassembles in-order using timestamp
                """
                if key in self.hot_keys:
                    # Split hot key across multiple partitions
                    suffix = self._get_hot_key_suffix(value)
                    modified_key = key + suffix
                    return self._hash_partition(modified_key)
                else:
                    # Regular partitioning
                    return self._hash_partition(key)

            def _get_hot_key_suffix(self, value: bytes) -> bytes:
                """
                Determine suffix for hot key

                Options:
                1. Sequential counter (user_123_0, user_123_1, ...)
                2. Hash of message content
                3. Timestamp-based
                """
                # Use message hash to distribute
                content_hash = hashlib.md5(value).hexdigest()[:4]
                return f"_{content_hash}".encode()
        ```

        **Trade-off:** Lose strict ordering for hot keys, but prevent partition overload.

    === "👥 Consumer Groups"

        ## The Challenge

        **Problem:** Coordinate multiple consumers to process partitions in parallel, with automatic failover.

        **Requirements:**

        - **Load balancing:** Distribute partitions across consumers
        - **Fault tolerance:** Reassign partitions if consumer fails
        - **No duplicate processing:** Each partition assigned to exactly one consumer
        - **Dynamic scaling:** Add/remove consumers without downtime

        ---

        ## Consumer Group Coordinator

        ```python
        import time
        from enum import Enum
        from typing import Dict, List, Set

        class GroupState(Enum):
            EMPTY = "Empty"                              # No members
            PREPARING_REBALANCE = "PreparingRebalance"   # Members joining
            COMPLETING_REBALANCE = "CompletingRebalance" # Assignments sent
            STABLE = "Stable"                            # Normal operation
            DEAD = "Dead"                                # Group deleted

        class ConsumerGroupCoordinator:
            """
            Manage consumer group membership and partition assignment

            Responsibilities:
            1. Track group members
            2. Trigger rebalances
            3. Distribute partition assignments
            4. Store offset commits
            """

            def __init__(self):
                self.groups: Dict[str, ConsumerGroup] = {}
                self.heartbeat_timeout = 30000  # 30 seconds
                self.session_timeout = 300000   # 5 minutes

            def join_group(
                self,
                group_id: str,
                member_id: str,
                client_id: str,
                topics: List[str],
                session_timeout: int
            ) -> dict:
                """
                Consumer joins group

                Returns:
                    {
                        'member_id': 'uuid',
                        'generation_id': 1,
                        'leader': True/False,
                        'assignment': {...}
                    }
                """
                group = self._get_or_create_group(group_id)

                # Add member to group
                member = ConsumerMember(
                    member_id=member_id or self._generate_member_id(),
                    client_id=client_id,
                    subscribed_topics=topics,
                    session_timeout=session_timeout
                )

                group.add_member(member)

                # Trigger rebalance
                if group.state == GroupState.STABLE:
                    logger.info(f"New member {member_id} joined, triggering rebalance")
                    self._trigger_rebalance(group)

                # Wait for rebalance to complete
                assignment = self._wait_for_assignment(group, member.member_id)

                return {
                    'member_id': member.member_id,
                    'generation_id': group.generation_id,
                    'leader': member.member_id == group.leader_id,
                    'assignment': assignment
                }

            def _trigger_rebalance(self, group: 'ConsumerGroup'):
                """
                Initiate partition rebalance

                Steps:
                1. Move to PREPARING_REBALANCE state
                2. Wait for all members to rejoin
                3. Calculate partition assignment
                4. Send assignments to members
                5. Move to STABLE state
                """
                group.state = GroupState.PREPARING_REBALANCE
                group.generation_id += 1

                logger.info(f"Rebalancing group {group.group_id}, generation {group.generation_id}")

                # Notify all members to rejoin (via heartbeat response)
                for member in group.members.values():
                    member.awaiting_join = True

                # Wait for all members to rejoin (timeout: session_timeout)
                start_time = time.time()
                while not self._all_members_rejoined(group):
                    if (time.time() - start_time) * 1000 > self.session_timeout:
                        # Remove members that didn't rejoin
                        self._remove_inactive_members(group)
                        break
                    time.sleep(0.1)

                # Calculate partition assignment
                assignment = self._assign_partitions(group)

                # Move to COMPLETING_REBALANCE
                group.state = GroupState.COMPLETING_REBALANCE
                group.assignment = assignment

                # Wait for all members to acknowledge
                self._wait_for_sync(group)

                # Move to STABLE
                group.state = GroupState.STABLE
                logger.info(f"Group {group.group_id} rebalance complete")

            def _assign_partitions(self, group: 'ConsumerGroup') -> Dict[str, List[int]]:
                """
                Assign partitions to consumers

                Strategies:
                1. Range: Divide partitions by range (P0-P33, P34-P66, P67-P99)
                2. RoundRobin: Distribute round-robin across consumers
                3. Sticky: Minimize reassignments during rebalance
                4. CooperativeSticky: Incremental rebalancing (no stop-the-world)
                """
                strategy = group.protocol  # 'range', 'roundrobin', 'sticky'

                if strategy == 'range':
                    return self._range_assignment(group)
                elif strategy == 'roundrobin':
                    return self._roundrobin_assignment(group)
                elif strategy == 'sticky':
                    return self._sticky_assignment(group)
                else:
                    raise ValueError(f"Unknown assignment strategy: {strategy}")

            def _range_assignment(self, group: 'ConsumerGroup') -> Dict[str, List[int]]:
                """
                Range assignment strategy (Kafka default)

                Example: 10 partitions, 3 consumers
                - Consumer 0: P0, P1, P2, P3
                - Consumer 1: P4, P5, P6
                - Consumer 2: P7, P8, P9

                Pros: Simple, predictable
                Cons: Uneven distribution if partitions not divisible
                """
                assignment = {}
                members = list(group.members.keys())
                members.sort()  # Stable ordering

                # Get all partitions for subscribed topics
                all_partitions = []
                for topic in group.subscribed_topics:
                    num_partitions = self._get_partition_count(topic)
                    all_partitions.extend([
                        (topic, partition_id)
                        for partition_id in range(num_partitions)
                    ])

                # Divide partitions into ranges
                partitions_per_consumer = len(all_partitions) // len(members)
                extra_partitions = len(all_partitions) % len(members)

                partition_idx = 0
                for i, member_id in enumerate(members):
                    # First 'extra_partitions' consumers get one extra partition
                    count = partitions_per_consumer + (1 if i < extra_partitions else 0)

                    assignment[member_id] = all_partitions[partition_idx:partition_idx + count]
                    partition_idx += count

                return assignment

            def _sticky_assignment(self, group: 'ConsumerGroup') -> Dict[str, List[int]]:
                """
                Sticky assignment strategy

                Goals:
                1. Balance partitions across consumers
                2. Minimize partition movement during rebalance
                3. Preserve existing assignments where possible

                Use case: Reduce rebalance overhead, maintain local state
                """
                # Get previous assignment
                previous_assignment = group.previous_assignment or {}

                # Identify partitions that need reassignment
                all_partitions = self._get_all_partitions(group.subscribed_topics)
                assigned_partitions = set()
                new_assignment = {}

                # Keep existing assignments for active members
                for member_id in group.members.keys():
                    if member_id in previous_assignment:
                        # Keep existing partitions
                        existing = previous_assignment[member_id]
                        new_assignment[member_id] = existing
                        assigned_partitions.update(existing)

                # Assign unassigned partitions
                unassigned = set(all_partitions) - assigned_partitions
                members = list(group.members.keys())
                member_idx = 0

                for partition in unassigned:
                    member_id = members[member_idx % len(members)]
                    if member_id not in new_assignment:
                        new_assignment[member_id] = []
                    new_assignment[member_id].append(partition)
                    member_idx += 1

                return new_assignment

            def heartbeat(self, group_id: str, member_id: str) -> dict:
                """
                Consumer sends periodic heartbeat

                Returns:
                    {
                        'rebalance_needed': True/False,
                        'generation_id': 1
                    }
                """
                group = self.groups.get(group_id)
                if not group:
                    return {'error': 'GROUP_NOT_FOUND'}

                member = group.members.get(member_id)
                if not member:
                    return {'error': 'UNKNOWN_MEMBER_ID'}

                # Update last heartbeat
                member.last_heartbeat = time.time()

                # Check if rebalance needed
                rebalance_needed = group.state != GroupState.STABLE

                return {
                    'rebalance_needed': rebalance_needed,
                    'generation_id': group.generation_id
                }

            def leave_group(self, group_id: str, member_id: str):
                """Consumer explicitly leaves group"""
                group = self.groups.get(group_id)
                if group:
                    group.remove_member(member_id)
                    logger.info(f"Member {member_id} left group {group_id}")

                    # Trigger rebalance
                    if group.members:
                        self._trigger_rebalance(group)
                    else:
                        group.state = GroupState.EMPTY

        class ConsumerGroup:
            """Represents a consumer group"""

            def __init__(self, group_id: str):
                self.group_id = group_id
                self.members: Dict[str, 'ConsumerMember'] = {}
                self.state = GroupState.EMPTY
                self.generation_id = 0
                self.leader_id = None
                self.protocol = 'range'  # Assignment strategy
                self.subscribed_topics: Set[str] = set()
                self.assignment: Dict[str, List[tuple]] = {}
                self.previous_assignment: Dict[str, List[tuple]] = {}

            def add_member(self, member: 'ConsumerMember'):
                """Add member to group"""
                self.members[member.member_id] = member
                self.subscribed_topics.update(member.subscribed_topics)

                # First member becomes leader
                if not self.leader_id:
                    self.leader_id = member.member_id

            def remove_member(self, member_id: str):
                """Remove member from group"""
                if member_id in self.members:
                    del self.members[member_id]

                # Elect new leader if needed
                if member_id == self.leader_id:
                    self.leader_id = next(iter(self.members.keys()), None)

        class ConsumerMember:
            """Represents a consumer in a group"""

            def __init__(
                self,
                member_id: str,
                client_id: str,
                subscribed_topics: List[str],
                session_timeout: int
            ):
                self.member_id = member_id
                self.client_id = client_id
                self.subscribed_topics = subscribed_topics
                self.session_timeout = session_timeout
                self.last_heartbeat = time.time()
                self.awaiting_join = False
        ```

        ---

        ## Rebalance Protocol

        **Steps:**

        ```
        1. Consumer joins group
           └─> Coordinator: PREPARING_REBALANCE

        2. All members receive REBALANCE_IN_PROGRESS
           └─> Stop fetching, commit offsets

        3. Members rejoin group
           └─> Send subscribed topics

        4. Coordinator assigns partitions
           └─> Use strategy (range/roundrobin/sticky)

        5. Coordinator: COMPLETING_REBALANCE
           └─> Send assignments to members

        6. Members sync and start fetching
           └─> Coordinator: STABLE

        7. Normal operation
           └─> Periodic heartbeats (every 3 seconds)
        ```

        **Rebalance triggers:**

        - New consumer joins
        - Consumer leaves/crashes (heartbeat timeout)
        - Topic metadata changes (partitions added)
        - Consumer manually unsubscribes

        ---

        ## Static Membership (KIP-345)

        **Problem:** Frequent rebalances cause processing disruption.

        **Solution:** Static group membership (Kafka 2.3+)

        ```python
        # Consumer config
        config = {
            'group.id': 'order-processor',
            'group.instance.id': 'consumer-1',  # Static member ID
            'session.timeout.ms': 300000        # 5 minutes (higher)
        }

        # Benefits:
        # - No rebalance on consumer restart (within session timeout)
        # - Partitions reassigned to same consumer
        # - Preserves local state (RocksDB, caches)
        ```

    === "🔄 Replication & Durability"

        ## The Challenge

        **Problem:** Ensure zero data loss even if brokers fail.

        **Requirements:**

        - **Durability:** Messages survive broker failures
        - **Availability:** System remains writable during failures
        - **Performance:** Replication doesn't slow down writes
        - **Consistency:** All replicas eventually converge

        ---

        ## Replication Architecture

        ```python
        from typing import List, Set
        from dataclasses import dataclass
        from enum import Enum

        class ReplicaState(Enum):
            ONLINE = "Online"
            OFFLINE = "Offline"
            CATCHING_UP = "CatchingUp"

        @dataclass
        class Replica:
            """Represents a partition replica"""
            broker_id: int
            partition_id: int
            is_leader: bool
            state: ReplicaState
            log_end_offset: int      # Last offset in log
            high_watermark: int      # Last committed offset (visible to consumers)
            lag: int                 # log_end_offset - high_watermark

        class ReplicationManager:
            """
            Manage partition replication

            Key concepts:
            - Leader: Handles all reads/writes
            - Follower: Replicates from leader
            - ISR: In-Sync Replicas (caught up with leader)
            - High Watermark: Last offset replicated to all ISR
            """

            def __init__(self, replication_factor: int = 3):
                self.replication_factor = replication_factor
                self.min_insync_replicas = 2  # Minimum ISR for durability
                self.replica_lag_max_messages = 4000  # Max lag to stay in ISR

            def replicate_message(
                self,
                partition_id: int,
                message: bytes,
                leader: Replica,
                followers: List[Replica]
            ) -> bool:
                """
                Replicate message to followers

                Steps:
                1. Leader appends to log
                2. Leader sends to followers
                3. Followers append to log and ACK
                4. Leader updates high watermark
                5. Leader ACKs to producer (based on acks config)

                Returns:
                    True if successfully replicated
                """
                # 1. Leader appends to log
                offset = leader.log_end_offset
                self._append_to_log(leader, message, offset)
                leader.log_end_offset += 1

                # 2. Send to all followers (parallel)
                acks_received = 0
                for follower in followers:
                    if follower.state == ReplicaState.ONLINE:
                        success = self._send_to_replica(follower, message, offset)
                        if success:
                            acks_received += 1

                # 3. Update ISR based on lag
                isr = self._update_isr(leader, followers)

                # 4. Update high watermark (min offset across ISR)
                min_isr_offset = min([r.log_end_offset for r in isr])
                leader.high_watermark = min_isr_offset

                # 5. Check if we can ACK to producer
                return self._can_ack_producer(leader, isr)

            def _send_to_replica(
                self,
                replica: Replica,
                message: bytes,
                offset: int
            ) -> bool:
                """
                Send message to follower replica

                Follower sends fetch request, leader responds with new messages
                """
                try:
                    # In real Kafka: follower pulls via fetch request
                    # Leader responds with messages starting from follower's offset
                    self._append_to_log(replica, message, offset)
                    replica.log_end_offset += 1
                    return True
                except Exception as e:
                    logger.error(f"Failed to replicate to {replica.broker_id}: {e}")
                    replica.state = ReplicaState.OFFLINE
                    return False

            def _update_isr(
                self,
                leader: Replica,
                followers: List[Replica]
            ) -> List[Replica]:
                """
                Update In-Sync Replica set

                ISR criteria:
                1. Replica is online
                2. Lag < replica.lag.max.messages (default: 4000)
                3. Last fetch within replica.lag.time.max.ms (default: 30s)
                """
                isr = [leader]  # Leader always in ISR

                for follower in followers:
                    lag = leader.log_end_offset - follower.log_end_offset

                    if follower.state == ReplicaState.ONLINE and \
                       lag < self.replica_lag_max_messages:
                        isr.append(follower)
                        follower.state = ReplicaState.ONLINE
                    else:
                        logger.warning(f"Replica {follower.broker_id} removed from ISR (lag: {lag})")
                        follower.state = ReplicaState.CATCHING_UP

                return isr

            def _can_ack_producer(
                self,
                leader: Replica,
                isr: List[Replica]
            ) -> bool:
                """
                Check if producer ACK conditions met

                acks config:
                - acks=0: No ACK (fire and forget)
                - acks=1: ACK after leader write
                - acks=all: ACK after all ISR replicas write
                """
                # For acks=all, need min.insync.replicas
                if len(isr) < self.min_insync_replicas:
                    logger.error(f"Not enough ISR replicas: {len(isr)} < {self.min_insync_replicas}")
                    return False

                return True

            def handle_leader_failure(
                self,
                partition_id: int,
                replicas: List[Replica]
            ) -> Replica:
                """
                Elect new leader when current leader fails

                Election strategy:
                1. Prefer replicas in ISR (no data loss)
                2. Choose replica with highest log_end_offset
                3. If no ISR available, choose unclean leader (data loss)

                Returns:
                    New leader replica
                """
                # Get ISR replicas
                isr_replicas = [r for r in replicas if r.state == ReplicaState.ONLINE]

                if isr_replicas:
                    # Clean leader election (no data loss)
                    new_leader = max(isr_replicas, key=lambda r: r.log_end_offset)
                    logger.info(f"Elected ISR replica {new_leader.broker_id} as leader")
                else:
                    # Unclean leader election (potential data loss)
                    logger.warning("No ISR replicas available, performing unclean election")
                    new_leader = max(replicas, key=lambda r: r.log_end_offset)

                new_leader.is_leader = True
                return new_leader

            def _append_to_log(self, replica: Replica, message: bytes, offset: int):
                """Append message to replica's log"""
                # In real Kafka: append to active segment file
                # Update index, update metrics
                pass
        ```

        ---

        ## Durability Guarantees

        | Config | Durability | Latency | Use Case |
        |--------|-----------|---------|----------|
        | **acks=0** | No guarantee | Lowest (<1ms) | Metrics, logs (OK to lose) |
        | **acks=1** | Leader only | Low (~5ms) | General use (rare data loss) |
        | **acks=all** | All ISR | Higher (~10ms) | Financial, critical data |

        **Combined with:**

        ```python
        # Producer config for maximum durability
        config = {
            'acks': 'all',                    # Wait for all ISR
            'retries': 2147483647,            # Retry forever
            'max.in.flight.requests.per.connection': 5,
            'enable.idempotence': True,       # No duplicates
            'compression.type': 'snappy'      # Compress for efficiency
        }

        # Broker config
        broker_config = {
            'min.insync.replicas': 2,         # At least 2 replicas
            'unclean.leader.election.enable': False,  # No data loss
            'log.flush.interval.messages': 10000,     # Flush every 10K messages
        }
        ```

        ---

        ## Handling Failures

        **Scenario 1: Follower failure**

        ```
        Before:
        Leader (B1): [msg1, msg2, msg3] ISR: [B1, B2, B3]
        Follower (B2): [msg1, msg2, msg3]
        Follower (B3): [msg1, msg2, msg3]

        After B3 fails:
        Leader (B1): [msg1, msg2, msg3, msg4] ISR: [B1, B2]
        Follower (B2): [msg1, msg2, msg3, msg4]
        Follower (B3): OFFLINE

        When B3 recovers:
        - Catches up from offset 3
        - Rejoins ISR when caught up
        ```

        **Scenario 2: Leader failure**

        ```
        Before:
        Leader (B1): [msg1, msg2, msg3] ISR: [B1, B2, B3]
        Follower (B2): [msg1, msg2, msg3]
        Follower (B3): [msg1, msg2]

        After B1 fails:
        1. Controller detects failure
        2. Elects new leader from ISR (B2)
        3. B2 becomes leader
        4. B3 truncates to B2's offset (loses msg3 if not replicated)

        New state:
        Leader (B2): [msg1, msg2, msg3, msg4] ISR: [B2, B3]
        Follower (B3): [msg1, msg2, msg3, msg4]
        Follower (B1): OFFLINE (when recovered, catches up)
        ```

        **Scenario 3: Network partition**

        ```
        Partition: {B1} | {B2, B3}

        With min.insync.replicas=2:
        - B1 (leader, ISR=[B1]): CANNOT accept writes (< 2 ISR)
        - B2, B3: Elect new leader (B2)

        System remains available (writes go to B2)
        When partition heals, B1 rejoins as follower
        ```

    === "✅ Exactly-Once Delivery"

        ## The Challenge

        **Problem:** Guarantee each message is processed exactly once, even with failures and retries.

        **Challenges:**

        - **Producer retries:** Network timeout causes duplicate send
        - **Consumer crashes:** Reprocess messages after restart
        - **Offset commits:** Commit before/after processing?

        ---

        ## Delivery Semantics

        | Semantic | Implementation | Trade-off |
        |----------|---------------|-----------|
        | **At-most-once** | Commit offset before processing | Fast, but may lose messages |
        | **At-least-once** | Commit offset after processing | May duplicate, but no loss |
        | **Exactly-once** | Transactional producer + consumer | Slower, but guaranteed |

        ---

        ## Idempotent Producer

        ```python
        class IdempotentProducer:
            """
            Prevent duplicate messages from producer retries

            Key concepts:
            - Producer ID (PID): Unique per producer instance
            - Sequence number: Incremented per partition
            - Broker tracks (PID, partition, sequence) to detect duplicates
            """

            def __init__(self, broker_connection):
                self.broker = broker_connection
                self.producer_id = None      # Assigned by broker
                self.producer_epoch = 0      # Incremented on failures
                self.sequence_numbers = {}   # partition -> sequence

                # Enable idempotence
                self.enable_idempotence = True
                self.max_in_flight = 5       # Max 5 inflight requests

            def initialize_producer_id(self):
                """
                Get producer ID from broker

                Called on producer startup
                """
                response = self.broker.init_producer_id_request(
                    transactional_id=None,  # None for non-transactional
                    transaction_timeout_ms=60000
                )

                self.producer_id = response['producer_id']
                self.producer_epoch = response['producer_epoch']

                logger.info(f"Initialized producer ID: {self.producer_id}, epoch: {self.producer_epoch}")

            def send(self, topic: str, partition: int, message: bytes) -> dict:
                """
                Send message with idempotence guarantee

                Broker deduplicates based on (PID, partition, sequence)
                """
                # Get next sequence number for partition
                if partition not in self.sequence_numbers:
                    self.sequence_numbers[partition] = 0

                sequence = self.sequence_numbers[partition]

                # Send to broker
                try:
                    response = self.broker.produce_request(
                        topic=topic,
                        partition=partition,
                        message=message,
                        producer_id=self.producer_id,
                        producer_epoch=self.producer_epoch,
                        sequence=sequence
                    )

                    if response['error'] is None:
                        # Success, increment sequence
                        self.sequence_numbers[partition] += 1
                        return response
                    elif response['error'] == 'DUPLICATE_SEQUENCE_NUMBER':
                        # Broker already has this message
                        logger.info(f"Duplicate detected for sequence {sequence}")
                        self.sequence_numbers[partition] += 1
                        return response  # Treat as success
                    else:
                        raise Exception(f"Send failed: {response['error']}")

                except NetworkTimeout:
                    # Retry with same sequence number (idempotence)
                    logger.warning(f"Timeout, retrying with sequence {sequence}")
                    return self.send(topic, partition, message)  # Retry
        ```

        ---

        ## Transactional Producer

        ```python
        class TransactionalProducer:
            """
            Exactly-once across multiple partitions/topics

            Use case:
            - Read from topic A, process, write to topic B
            - Guarantee: message processed exactly once

            Implementation:
            - Transaction coordinator tracks state
            - Two-phase commit protocol
            - Atomic writes across partitions
            """

            def __init__(self, broker_connection, transactional_id: str):
                self.broker = broker_connection
                self.transactional_id = transactional_id
                self.producer_id = None
                self.producer_epoch = 0
                self.sequence_numbers = {}
                self.current_transaction = None

            def init_transactions(self):
                """Initialize transactional producer"""
                response = self.broker.init_producer_id_request(
                    transactional_id=self.transactional_id,
                    transaction_timeout_ms=60000
                )

                self.producer_id = response['producer_id']
                self.producer_epoch = response['producer_epoch']

                logger.info(f"Initialized transactional producer: {self.transactional_id}")

            def begin_transaction(self):
                """Start new transaction"""
                if self.current_transaction:
                    raise Exception("Transaction already in progress")

                self.current_transaction = Transaction(
                    transactional_id=self.transactional_id,
                    producer_id=self.producer_id,
                    producer_epoch=self.producer_epoch
                )

                logger.info("Transaction started")

            def send_in_transaction(
                self,
                topic: str,
                partition: int,
                message: bytes
            ):
                """Send message as part of transaction"""
                if not self.current_transaction:
                    raise Exception("No active transaction")

                # Add to transaction
                self.current_transaction.add_partition(topic, partition)

                # Send to broker (not visible to consumers yet)
                sequence = self.sequence_numbers.get(partition, 0)
                self.broker.produce_request(
                    topic=topic,
                    partition=partition,
                    message=message,
                    producer_id=self.producer_id,
                    producer_epoch=self.producer_epoch,
                    sequence=sequence,
                    transactional_id=self.transactional_id
                )

                self.sequence_numbers[partition] = sequence + 1

            def send_offsets_to_transaction(
                self,
                offsets: dict,
                consumer_group_id: str
            ):
                """
                Add offset commits to transaction

                Enables read-process-write exactly-once
                """
                if not self.current_transaction:
                    raise Exception("No active transaction")

                self.current_transaction.add_offsets(offsets, consumer_group_id)

                # Send to transaction coordinator
                self.broker.add_offsets_to_txn_request(
                    transactional_id=self.transactional_id,
                    producer_id=self.producer_id,
                    producer_epoch=self.producer_epoch,
                    group_id=consumer_group_id
                )

            def commit_transaction(self):
                """
                Commit transaction (two-phase commit)

                Phase 1: Prepare
                - Send PREPARE to all partition leaders
                - Write transaction markers to log

                Phase 2: Commit
                - Send COMMIT to all partition leaders
                - Mark messages as visible to consumers
                - Commit offsets
                """
                if not self.current_transaction:
                    raise Exception("No active transaction")

                try:
                    # Phase 1: Prepare
                    self.broker.end_txn_request(
                        transactional_id=self.transactional_id,
                        producer_id=self.producer_id,
                        producer_epoch=self.producer_epoch,
                        committed=True
                    )

                    # Phase 2: Transaction coordinator writes markers
                    # Messages become visible atomically

                    logger.info("Transaction committed")
                    self.current_transaction = None

                except Exception as e:
                    logger.error(f"Commit failed: {e}")
                    self.abort_transaction()
                    raise

            def abort_transaction(self):
                """Abort transaction (rollback)"""
                if not self.current_transaction:
                    return

                self.broker.end_txn_request(
                    transactional_id=self.transactional_id,
                    producer_id=self.producer_id,
                    producer_epoch=self.producer_epoch,
                    committed=False
                )

                logger.info("Transaction aborted")
                self.current_transaction = None

        class Transaction:
            """Represents an active transaction"""

            def __init__(self, transactional_id: str, producer_id: int, producer_epoch: int):
                self.transactional_id = transactional_id
                self.producer_id = producer_id
                self.producer_epoch = producer_epoch
                self.partitions = set()  # (topic, partition)
                self.offsets = {}        # For offset commits

            def add_partition(self, topic: str, partition: int):
                """Add partition to transaction"""
                self.partitions.add((topic, partition))

            def add_offsets(self, offsets: dict, group_id: str):
                """Add offset commits to transaction"""
                self.offsets = {
                    'group_id': group_id,
                    'offsets': offsets
                }
        ```

        ---

        ## Exactly-Once Read-Process-Write

        ```python
        def process_orders_exactly_once():
            """
            Example: Read orders, process, write to notifications

            Guarantee: Each order processed exactly once
            """
            # Create transactional producer
            producer = TransactionalProducer(
                broker,
                transactional_id='order-processor-1'
            )
            producer.init_transactions()

            # Create consumer
            consumer = KafkaConsumer(
                'orders',
                group_id='order-processor',
                enable_auto_commit=False,  # Manual commit
                isolation_level='read_committed'  # Only read committed messages
            )

            while True:
                # Poll messages
                messages = consumer.poll(timeout_ms=1000)

                if not messages:
                    continue

                # Begin transaction
                producer.begin_transaction()

                try:
                    for topic_partition, records in messages.items():
                        for record in records:
                            # Process message
                            notification = process_order(record.value)

                            # Write output to notification topic
                            producer.send_in_transaction(
                                topic='notifications',
                                partition=hash(notification['user_id']) % 10,
                                message=notification
                            )

                        # Add offset commits to transaction
                        producer.send_offsets_to_transaction(
                            offsets={
                                topic_partition: {
                                    'offset': records[-1].offset + 1,
                                    'metadata': ''
                                }
                            },
                            consumer_group_id='order-processor'
                        )

                    # Commit transaction (atomic)
                    producer.commit_transaction()

                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    producer.abort_transaction()
        ```

        **Guarantees:**

        1. **Atomicity:** All writes + offset commit succeed or none
        2. **Idempotence:** Retries don't create duplicates
        3. **Isolation:** Consumers only see committed messages

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Kafka from 1M to 1 trillion messages/day.

    **Scaling challenges at 1T messages/day:**

    - **Write throughput:** 11.6M messages/sec (35M peak)
    - **Read throughput:** 35M messages/sec (3 consumer groups)
    - **Storage:** 75 PB (7-day retention + replication)
    - **Network:** 279 Gbps ingress, 280 Gbps egress

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Disk I/O** | ✅ Yes | Sequential writes (500+ MB/sec), NVMe SSDs, RAID 10 |
    | **Network** | ✅ Yes | 40 Gbps NICs, dedicated replication network, compression |
    | **Page cache** | ✅ Yes | 256 GB RAM per broker, OS tuning (vm.swappiness=1) |
    | **CPU** | 🟡 Moderate | Compression/decompression, batching to reduce syscalls |
    | **ZooKeeper** | 🟡 Approaching | Migrate to KRaft (no ZooKeeper), reduce metadata writes |

    ---

    ## Cluster Sizing

    **For 1T messages/day:**

    ```
    Throughput: 11.6M msg/sec × 1 KB = 11.6 GB/sec
    Replication: 11.6 GB/sec × 2 (leader to 2 followers) = 23.2 GB/sec
    Total ingress: 11.6 GB/sec + 23.2 GB/sec = 34.8 GB/sec

    Per broker throughput: 500 MB/sec (sequential write)
    Brokers needed: 34.8 GB/sec / 0.5 GB/sec = 70 brokers (min)

    Storage per broker: 75 PB / 100 brokers = 750 TB
    Disk config: 10x 10 TB NVMe (RAID 10) = 50 TB usable per broker

    Memory per broker: 256 GB (mostly page cache)
    CPU: 32 cores (compression, handling requests)
    Network: 40 Gbps NIC

    Total cluster: 100 brokers
    - EC2: i4i.8xlarge (32 vCPU, 256 GB RAM, 7.5 TB NVMe × 2)
    - Storage: 100 brokers × 50 TB = 5 PB raw (10 PB with RAID)
    ```

    ---

    ## Performance Optimizations

    ### 1. Zero-Copy Transfer

    ```python
    # Traditional copy (4 context switches, 4 copies)
    # 1. Read file to kernel buffer
    # 2. Copy to user-space buffer
    # 3. Copy to socket buffer
    # 4. DMA to NIC

    # Zero-copy with sendfile() (2 context switches, 0 copies)
    import os

    def send_messages_zero_copy(socket_fd: int, file_path: str, offset: int, size: int):
        """
        Use sendfile() for zero-copy transfer

        Bypasses user space entirely:
        file descriptor -> socket descriptor
        """
        file_fd = os.open(file_path, os.O_RDONLY)

        try:
            # sendfile(out_fd, in_fd, offset, count)
            # Kernel directly transfers from file to socket
            sent = os.sendfile(socket_fd, file_fd, offset, size)
            return sent
        finally:
            os.close(file_fd)

    # Kafka uses this for consumer fetch requests
    # 2-4x throughput improvement (no CPU copying)
    ```

    ---

    ### 2. Batching

    ```python
    class BatchProducer:
        """Batch messages for efficiency"""

        def __init__(self, broker):
            self.broker = broker
            self.batch_size = 16384      # 16 KB
            self.linger_ms = 10          # Wait 10ms for more messages
            self.buffer = []
            self.buffer_size = 0

        def send(self, message: bytes):
            """
            Add message to batch

            Send when:
            1. Batch size reached (16 KB)
            2. Linger time elapsed (10 ms)
            3. Producer flush() called
            """
            self.buffer.append(message)
            self.buffer_size += len(message)

            # Send if batch full
            if self.buffer_size >= self.batch_size:
                self.flush()

        def flush(self):
            """Send batched messages"""
            if not self.buffer:
                return

            # Send entire batch in single request
            self.broker.send_batch(self.buffer)

            # Clear buffer
            self.buffer.clear()
            self.buffer_size = 0

    # Benefits:
    # - Reduce network overhead (1 request vs 100)
    # - Better compression (compress batch, not individual messages)
    # - Higher throughput (100K msg/sec -> 1M msg/sec)
    ```

    ---

    ### 3. Compression

    ```python
    # Compression algorithms

    compression_stats = {
        'none': {
            'compression_ratio': 1.0,
            'cpu_overhead': 'None',
            'latency': 'Lowest',
            'use_case': 'Already compressed data (images, video)'
        },
        'gzip': {
            'compression_ratio': 3.5,   # 3.5x reduction
            'cpu_overhead': 'High',
            'latency': '+50ms',
            'use_case': 'Low throughput, high compression needed'
        },
        'snappy': {
            'compression_ratio': 2.0,   # 2x reduction
            'cpu_overhead': 'Low',
            'latency': '+5ms',
            'use_case': 'Balanced (Kafka default)'
        },
        'lz4': {
            'compression_ratio': 2.2,   # 2.2x reduction
            'cpu_overhead': 'Very Low',
            'latency': '+3ms',
            'use_case': 'High throughput, low latency'
        },
        'zstd': {
            'compression_ratio': 3.0,   # 3x reduction
            'cpu_overhead': 'Medium',
            'latency': '+15ms',
            'use_case': 'Best compression/speed trade-off (Kafka 2.1+)'
        }
    }

    # Recommendation: snappy or zstd
    # - snappy: 2x compression, minimal latency
    # - zstd: 3x compression, moderate latency
    ```

    ---

    ### 4. Page Cache Tuning

    ```bash
    # OS configuration for Kafka

    # Disable swap (Kafka relies on page cache)
    vm.swappiness = 1

    # Increase page cache size
    # Kafka doesn't use heap (relies on OS page cache)
    # Allocate most RAM to page cache

    # File system
    # Use XFS (better than ext4 for large files)
    mkfs.xfs /dev/nvme0n1

    # Mount options
    mount -o noatime,nodiratime /dev/nvme0n1 /kafka-data
    # noatime: Don't update access time (faster)

    # Increase file descriptors
    ulimit -n 100000

    # Network tuning
    net.core.rmem_max = 134217728  # 128 MB receive buffer
    net.core.wmem_max = 134217728  # 128 MB send buffer
    net.ipv4.tcp_rmem = 4096 87380 134217728
    net.ipv4.tcp_wmem = 4096 65536 134217728
    ```

    ---

    ### 5. Partitioning Strategy

    ```python
    # Optimal partition count

    def calculate_optimal_partitions(
        target_throughput_mb_per_sec: float,
        partition_throughput_mb_per_sec: float = 10.0  # Conservative
    ) -> int:
        """
        Calculate optimal partition count

        Factors:
        1. Throughput: More partitions = higher throughput
        2. Parallelism: Max consumers = num partitions
        3. Overhead: Too many partitions = metadata overhead

        Rule of thumb:
        - Single partition: ~10 MB/sec
        - Total partitions: target_throughput / 10
        """
        partitions = int(target_throughput_mb_per_sec / partition_throughput_mb_per_sec)

        # Clamp to reasonable range
        return max(10, min(partitions, 1000))

    # Example: 1 GB/sec throughput
    # Partitions: 1000 MB/sec / 10 MB/sec = 100 partitions

    # Benefits:
    # - 100 consumers can process in parallel
    # - Each partition ~10 MB/sec (manageable)
    # - Can scale to 10x throughput (add brokers + partitions)
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 1T messages/day:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (brokers)** | $350,000 (100 × i4i.8xlarge @ $3,500/mo) |
    | **EBS storage** | $0 (use instance storage) |
    | **Network (inter-AZ)** | $62,500 (replication traffic) |
    | **Data transfer out** | $25,000 (consumers pulling data) |
    | **ZooKeeper cluster** | $10,000 (5 × m5.xlarge) |
    | **CloudWatch** | $5,000 (metrics) |
    | **Total** | **$452,500/month** |

    **Cost optimizations:**

    - Use instance storage (NVMe) instead of EBS
    - Single-AZ deployment (if acceptable)
    - Compress messages (2-3x storage reduction)
    - Tiered storage (AWS S3 for old data)
    - Reserved instances (40% discount)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Producer Latency (P95)** | < 10ms | > 50ms |
    | **Consumer Lag** | < 1000 messages | > 100K messages |
    | **ISR Shrink Rate** | 0 | > 0 (replica falling behind) |
    | **Under-replicated Partitions** | 0 | > 0 (replication failure) |
    | **Network Throughput** | < 30 Gbps | > 35 Gbps (nearing limit) |
    | **Disk Utilization** | < 70% | > 80% |
    | **Request Queue Size** | < 100 | > 500 (broker overloaded) |

    ---

    ## KRaft Migration

    **ZooKeeper limitations:**

    - Limited scalability (metadata bottleneck)
    - Complex operational overhead
    - Slow metadata propagation

    **KRaft (Kafka Raft):**

    - Native consensus (no ZooKeeper dependency)
    - Faster metadata propagation (< 1s)
    - Simpler architecture

    ```yaml
    # KRaft configuration (Kafka 3.3+)
    process.roles=broker,controller
    node.id=1
    controller.quorum.voters=1@broker1:9093,2@broker2:9093,3@broker3:9093

    # Benefits:
    # - Single process (not broker + ZooKeeper)
    # - Faster partition creation (100x faster)
    # - Support for millions of partitions
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Partitioning for parallelism:** Hash-based partitioning preserves ordering per key
    2. **Consumer groups for coordination:** Automatic rebalancing, fault tolerance
    3. **ISR-based replication:** Durability without sacrificing availability
    4. **Sequential I/O:** 500+ MB/sec throughput on commodity hardware
    5. **Zero-copy transfer:** Kernel-level optimization for high throughput
    6. **Page cache optimization:** Rely on OS caching, minimal heap usage
    7. **Exactly-once semantics:** Transactional API for critical use cases

    ---

    ## Interview Tips

    ✅ **Emphasize partitioning strategy** - Key to scalability and ordering

    ✅ **Discuss consumer group rebalancing** - Complex coordination protocol

    ✅ **Explain replication trade-offs** - acks=1 vs acks=all vs acks=0

    ✅ **Highlight zero-copy optimization** - Unique Kafka performance feature

    ✅ **Cover exactly-once semantics** - Advanced topic, shows depth

    ✅ **Mention KRaft migration** - Modern Kafka (ZooKeeper removal)

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How does Kafka achieve high throughput?"** | Sequential I/O, zero-copy, batching, page cache, compression |
    | **"How to guarantee ordering?"** | Partitioning by key, single consumer per partition |
    | **"How to handle consumer lag?"** | Add consumers (up to partition count), optimize processing, scale partitions |
    | **"Kafka vs RabbitMQ vs SQS?"** | Kafka: high throughput, ordering, replay. RabbitMQ: routing, priority. SQS: managed, simple. |
    | **"How to achieve exactly-once?"** | Idempotent producer + transactional API + read_committed isolation |
    | **"How to handle hot partitions?"** | Add suffix to key, redistribute, or use sticky assignment |
    | **"What happens if all ISR fail?"** | Unclean leader election (data loss) or wait for ISR (availability loss) |

    ---

    ## Design Variations

    ### AWS Kinesis

    **Differences:**

    - **Shards instead of partitions** (similar concept)
    - **Fully managed** (no broker management)
    - **Auto-scaling** (but expensive)
    - **7-day max retention** (Kafka: unlimited with tiered storage)

    ### Google Pub/Sub

    **Differences:**

    - **No partitions** (automatic scaling)
    - **Pull or push delivery**
    - **At-least-once delivery** (no exactly-once)
    - **Message ordering per key** (similar to Kafka)

    ### Apache Pulsar

    **Differences:**

    - **Separate storage layer** (BookKeeper)
    - **Multi-tenancy built-in**
    - **Tiered storage native**
    - **Geo-replication** (cross-datacenter)

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** LinkedIn (Kafka), Confluent, AWS, Google, Apache, Uber, Netflix

---

*Master this problem and you'll be ready for: Stream processing systems, Event-driven architectures, Real-time data pipelines, Microservices communication*
