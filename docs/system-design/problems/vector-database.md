# Design Vector Database (Pinecone/Weaviate)

A high-performance, distributed vector database system that stores and indexes high-dimensional vectors, enabling fast similarity search using Approximate Nearest Neighbor (ANN) algorithms for AI/ML applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | Billions of vectors, 100K+ QPS, <100ms p99 latency, 99.99% availability |
| **Key Challenges** | ANN indexing (HNSW, IVF), similarity search optimization, sharding high-dimensional data, filtering with metadata |
| **Core Concepts** | Vector embeddings, cosine similarity, HNSW graphs, product quantization, hybrid search, GPU acceleration |
| **Companies** | Pinecone, Weaviate, Qdrant, Milvus, Chroma, OpenAI, Anthropic, Google, Meta |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Insert Vectors** | Store high-dimensional vectors (128-4096 dimensions) with metadata | P0 (Must have) |
    | **Similarity Search** | Find k-nearest neighbors using cosine, dot product, or Euclidean distance | P0 (Must have) |
    | **Metadata Filtering** | Search with filters (e.g., date range, category, tags) | P0 (Must have) |
    | **Update Vectors** | Modify existing vectors or metadata | P0 (Must have) |
    | **Delete Vectors** | Remove vectors from index | P0 (Must have) |
    | **Batch Operations** | Bulk insert/update/delete for efficiency | P1 (Should have) |
    | **Hybrid Search** | Combine vector similarity with keyword search (BM25) | P1 (Should have) |
    | **Namespaces/Collections** | Logical isolation of vector datasets | P1 (Should have) |
    | **Index Optimization** | Background index building and optimization | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - SQL-style joins (use application layer)
    - ACID transactions (eventual consistency acceptable)
    - Full-text search analytics (use Elasticsearch instead)
    - Training embeddings (use separate ML service)
    - Real-time clustering (use batch processing)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Search)** | < 100ms p99 | Fast enough for real-time applications (chatbots, search) |
    | **Latency (Insert)** | < 50ms p99 | Writes less critical than reads, but should be fast |
    | **Availability** | 99.99% uptime | Critical for production AI applications |
    | **Throughput** | 100K+ QPS | Handle high-traffic AI applications |
    | **Recall** | > 95% @ k=10 | ANN should return nearly same results as exact search |
    | **Scalability** | Billions of vectors | Support large-scale embeddings (entire web corpus) |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Operations per second:
    - Read operations (Search): 80K QPS (80% reads)
    - Write operations (Insert/Update/Delete): 20K QPS (20% writes)
    - Total QPS: 100K QPS
    - Peak QPS: 3x average = 300K QPS (during traffic spikes)

    Read/Write ratio: 4:1 (read-heavy typical for vector search)

    Search parameters:
    - Average k (top-k results): 10
    - Average filter complexity: 2 metadata conditions
    - Average query vector dimension: 1536 (OpenAI text-embedding-3-small)

    Network traffic:
    - Query size: 1536 dimensions × 4 bytes (float32) = 6 KB
    - Response size: 10 results × (8 bytes ID + 4 bytes score + 200 bytes metadata) = 2 KB
    - Read bandwidth: 80K QPS × (6 KB + 2 KB) = 640 MB/sec = 5.12 Gbps
    - Write bandwidth: 20K QPS × 6 KB = 120 MB/sec = 0.96 Gbps
    - Total: 6 Gbps (manageable with 10 Gbps NICs)
    ```

    ### Storage Estimates

    ```
    Vector data:
    - Total vectors: 5 billion (5B vectors)
    - Dimensions per vector: 1536 (OpenAI embeddings)
    - Bytes per dimension: 4 bytes (float32)
    - Metadata per vector: 500 bytes (JSON: title, content, date, tags)

    Per-vector storage:
    - Raw vector: 1536 × 4 bytes = 6 KB
    - Quantized vector (PQ): 128 bytes (48x compression)
    - Metadata: 500 bytes
    - HNSW index overhead: 200 bytes (graph edges, levels)
    - Total per vector: 128 + 500 + 200 = 828 bytes

    Total storage:
    - Data: 5B × 828 bytes = 4.14 TB
    - Replication factor: 3x (for availability)
    - Total with replication: 4.14 TB × 3 = 12.42 TB

    Partitioning:
    - 200 shards × 20 GB per shard = 4 TB capacity
    - Each shard has 2 replicas: 600 total nodes
    - Per shard: 25M vectors (5B / 200)
    ```

    ### Bandwidth Estimates

    ```
    Ingress (writes):
    - 20K inserts/sec × 6 KB = 120 MB/sec ≈ 1 Gbps
    - With replication (3x): 1 Gbps × 3 = 3 Gbps

    Egress (reads):
    - 80K searches/sec × 8 KB (query + response) = 640 MB/sec ≈ 5 Gbps

    Total bandwidth: 3 Gbps (ingress) + 5 Gbps (egress) = 8 Gbps
    - Per shard (200 shards): 40 Mbps (very manageable)
    ```

    ### CPU Estimates

    ```
    Per-search cost (HNSW):
    - Distance calculations: 200 candidates × 1536 dimensions = 307K operations
    - CPU time: ~2 ms (optimized with SIMD)

    Per-insert cost:
    - HNSW insertion: ~10 ms (building graph connections)
    - Quantization: ~1 ms

    Total CPU:
    - Searches: 80K QPS × 2 ms = 160 CPU-seconds per second
    - Inserts: 20K QPS × 11 ms = 220 CPU-seconds per second
    - Total: 380 CPU-seconds per second
    - 200 shards: 1.9 CPU-seconds per shard
    - With overhead: 4 CPU cores per shard (allows 2x headroom)

    Recommendation: 8 CPU cores per node (allows GPU acceleration)
    ```

    ### Memory Estimates (Per Node)

    ```
    Index data per shard:
    - 25M vectors × 828 bytes = 20.7 GB
    - HNSW graph (in-memory): 25M × 200 bytes = 5 GB
    - Metadata index: 2 GB
    - Query cache: 1 GB
    - Total RAM per node: 20.7 + 5 + 2 + 1 ≈ 29 GB

    Recommendation: 64 GB RAM per node (allows headroom for sorting, filtering)
    ```

    ---

    ## Key Assumptions

    1. Read-heavy workload (80/20 read/write ratio)
    2. Vector dimensions: 1536 (typical for modern embeddings)
    3. ANN recall target: 95% (acceptable trade-off for speed)
    4. Product quantization: 48x compression (1536 dims → 32 bytes)
    5. Metadata filters on 30% of queries (affects performance)
    6. Hot vectors (top 10%) account for 80% of traffic (Zipf distribution)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **ANN indexing:** Use HNSW for fast approximate search (vs. exact search)
    2. **Product quantization:** Compress vectors 48x to reduce memory footprint
    3. **Horizontal sharding:** Partition vectors by ID across nodes
    4. **Metadata co-location:** Store metadata with vectors for fast filtering
    5. **Hybrid indexes:** Combine HNSW (vector) with inverted index (keywords)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            App1[Application/LLM]
            App2[RAG System]
            App3[Recommendation Engine]
        end

        subgraph "API Gateway"
            Gateway[Load Balancer<br/>Query Router<br/>Rate Limiter]
        end

        subgraph "Query Processing Layer"
            QP1[Query Processor 1<br/>Parse & validate<br/>Embedding generation]
            QP2[Query Processor 2<br/>Filter optimization<br/>Result ranking]
            QP3[Query Processor N<br/>Batch processing]
        end

        subgraph "Index Management"
            IM[Index Manager<br/>HNSW builder<br/>PQ quantizer<br/>Shard router]
        end

        subgraph "Vector Storage Cluster (Sharded)"
            subgraph "Shard 1 (0-25M)"
                Master1[Master Node 1<br/>HNSW Index<br/>Vectors: 0-25M]
                Replica1A[Replica 1A<br/>Hot standby]
                Replica1B[Replica 1B<br/>Read replica]
            end

            subgraph "Shard 2 (25M-50M)"
                Master2[Master Node 2<br/>HNSW Index<br/>Vectors: 25M-50M]
                Replica2A[Replica 2A<br/>Hot standby]
                Replica2B[Replica 2B<br/>Read replica]
            end

            subgraph "Shard N (4.975B-5B)"
                MasterN[Master Node N<br/>HNSW Index<br/>Vectors: 4.975B-5B]
                ReplicaNA[Replica NA<br/>Hot standby]
                ReplicaNB[Replica NB<br/>Read replica]
            end
        end

        subgraph "Metadata Store"
            Meta[(PostgreSQL<br/>Filters, tags, dates<br/>Secondary indexes)]
        end

        subgraph "Object Storage"
            S3[(S3/GCS<br/>Raw vectors<br/>Index snapshots<br/>Backups)]
        end

        subgraph "Monitoring & Coordination"
            Metrics[Monitoring<br/>Prometheus/Grafana<br/>Recall metrics]
            Coord[ZooKeeper/etcd<br/>Cluster topology<br/>Shard mapping]
        end

        App1 --> Gateway
        App2 --> Gateway
        App3 --> Gateway

        Gateway --> QP1
        Gateway --> QP2
        Gateway --> QP3

        QP1 --> IM
        QP2 --> IM
        QP3 --> IM

        IM --> Master1
        IM --> Master2
        IM --> MasterN

        Master1 -.->|Async replication| Replica1A
        Master1 -.->|Async replication| Replica1B
        Master2 -.->|Async replication| Replica2A
        Master2 -.->|Async replication| Replica2B
        MasterN -.->|Async replication| ReplicaNA
        MasterN -.->|Async replication| ReplicaNB

        Master1 <--> Meta
        Master2 <--> Meta
        MasterN <--> Meta

        Master1 --> S3
        Master2 --> S3
        MasterN --> S3

        Metrics --> Master1
        Metrics --> Master2
        Metrics --> MasterN

        Coord --> IM
        Coord --> Master1
        Coord --> Master2

        style Gateway fill:#e1f5ff
        style Master1 fill:#ffe1e1
        style Master2 fill:#ffe1e1
        style MasterN fill:#ffe1e1
        style Replica1A fill:#fff4e1
        style Replica2A fill:#fff4e1
        style ReplicaNA fill:#fff4e1
        style Meta fill:#e8f5e9
        style S3 fill:#f3e5f5
        style IM fill:#fce4ec
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **HNSW Index** | Best ANN algorithm for high recall (95%+) and speed (<100ms) | IVF (lower recall), Annoy (slower inserts), ScaNN (more complex) |
    | **Product Quantization** | 48x compression (1536 dims → 32 bytes) with <1% recall loss | Binary quantization (lower recall), no compression (high memory cost) |
    | **Sharding by Vector ID** | Even distribution, simple routing, no hotspots | Sharding by LSH (complex), range sharding (imbalanced load) |
    | **Metadata Co-location** | Fast filtering (no joins), single query to each shard | Separate metadata store (requires joins, adds latency) |
    | **PostgreSQL for Metadata** | Rich filtering (date ranges, tags), B-tree indexes, SQL queries | MongoDB (weaker filtering), DynamoDB (complex queries expensive) |
    | **S3 for Backups** | Durable storage for index snapshots, cost-effective for cold data | Local disk (no redundancy), expensive block storage |

    **Key Trade-off:** We chose **speed over exact accuracy**. ANN with 95% recall is 100x faster than exact search, acceptable for most AI applications.

    ---

    ## API Design

    ### 1. Insert Vector

    **Request:**
    ```http
    POST /vectors/insert
    Content-Type: application/json

    {
      "vectors": [
        {
          "id": "vec_123",
          "values": [0.1, 0.2, ..., 0.5],  // 1536 dimensions
          "metadata": {
            "title": "Introduction to AI",
            "content": "Artificial intelligence is...",
            "date": "2024-01-15",
            "category": "technology",
            "tags": ["AI", "machine learning"]
          }
        }
      ],
      "namespace": "knowledge_base"
    }
    ```

    **Response:**
    ```json
    {
      "inserted_count": 1,
      "ids": ["vec_123"],
      "duration_ms": 45
    }
    ```

    **Design Notes:**

    - Atomic batch inserts (up to 1000 vectors per request)
    - Async index building (returns immediately)
    - ID auto-generated if not provided (UUID v7)
    - Validates dimension count matches index configuration

    ---

    ### 2. Search (Similarity Query)

    **Request:**
    ```http
    POST /vectors/search
    Content-Type: application/json

    {
      "vector": [0.1, 0.2, ..., 0.5],  // Query vector
      "top_k": 10,                      // Return top 10 results
      "namespace": "knowledge_base",
      "filter": {
        "category": {"$eq": "technology"},
        "date": {"$gte": "2024-01-01"},
        "tags": {"$in": ["AI", "ML"]}
      },
      "include_metadata": true,
      "include_values": false           // Don't return vectors (save bandwidth)
    }
    ```

    **Response:**
    ```json
    {
      "results": [
        {
          "id": "vec_123",
          "score": 0.95,                 // Cosine similarity
          "metadata": {
            "title": "Introduction to AI",
            "category": "technology",
            "date": "2024-01-15"
          }
        },
        ...
      ],
      "duration_ms": 87
    }
    ```

    **Design Notes:**

    - Distance metric: cosine similarity (default), dot product, Euclidean
    - Filtering BEFORE ANN search (post-filtering less efficient)
    - Score normalization: [0, 1] for cosine, [-1, 1] for dot product
    - Pagination for results > 100

    ---

    ### 3. Update Vector

    **Request:**
    ```http
    POST /vectors/update
    Content-Type: application/json

    {
      "id": "vec_123",
      "values": [0.2, 0.3, ..., 0.6],   // New vector (optional)
      "metadata": {                      // Partial update (merge)
        "tags": ["AI", "deep learning", "NLP"]
      },
      "namespace": "knowledge_base"
    }
    ```

    **Response:**
    ```json
    {
      "updated": true,
      "duration_ms": 35
    }
    ```

    **Design Notes:**

    - Updates trigger re-indexing (expensive for HNSW)
    - Metadata-only updates are fast (no re-indexing)
    - Optimistic locking with version field (optional)

    ---

    ### 4. Delete Vector

    **Request:**
    ```http
    POST /vectors/delete
    Content-Type: application/json

    {
      "ids": ["vec_123", "vec_456"],
      "namespace": "knowledge_base"
    }
    ```

    **Response:**
    ```json
    {
      "deleted_count": 2,
      "duration_ms": 20
    }
    ```

    **Design Notes:**

    - Soft delete (mark as deleted, lazy removal)
    - Background compaction rebuilds index (remove tombstones)
    - Hard delete for compliance (GDPR) forces immediate removal

    ---

    ### 5. Hybrid Search (Vector + Keyword)

    **Request:**
    ```http
    POST /vectors/hybrid-search
    Content-Type: application/json

    {
      "vector": [0.1, 0.2, ..., 0.5],   // Semantic search
      "keywords": "artificial intelligence machine learning",  // Keyword search
      "top_k": 10,
      "alpha": 0.7,                     // Weight: 0.7 vector + 0.3 keyword
      "namespace": "knowledge_base"
    }
    ```

    **Response:**
    ```json
    {
      "results": [
        {
          "id": "vec_789",
          "score": 0.89,                 // Combined score
          "vector_score": 0.92,          // Semantic similarity
          "keyword_score": 0.85,         // BM25 score
          "metadata": {...}
        },
        ...
      ]
    }
    ```

    **Design Notes:**

    - BM25 for keyword search (inverted index)
    - Score fusion: `final_score = alpha * vector_score + (1 - alpha) * bm25_score`
    - Runs two searches in parallel (vector + keyword), merges results

    ---

    ## Database Schema

    ### Vector Index Structure (HNSW)

    ```python
    # HNSW node representation
    class HNSWNode:
        """
        Node in HNSW graph (Hierarchical Navigable Small World)
        """
        vector_id: str              # Unique vector ID
        vector: np.ndarray          # Quantized vector (128 bytes)
        metadata: dict              # JSON metadata

        # HNSW graph structure
        layers: List[int]           # Layer assignments [0, 1, 3]
        neighbors: Dict[int, List[str]]  # layer -> [neighbor_ids]
        # Example: {0: ["vec_2", "vec_5"], 1: ["vec_3"], 3: ["vec_1"]}

        # Index metadata
        insertion_timestamp: int    # For ordering
        deleted: bool               # Soft delete flag

    # Index configuration
    class VectorIndex:
        """
        Vector index configuration and statistics
        """
        namespace: str              # Logical isolation
        dimension: int              # Vector dimensions (e.g., 1536)
        metric: str                 # "cosine", "dot", "euclidean"

        # HNSW parameters
        m: int = 16                 # Max connections per layer
        ef_construction: int = 200  # Candidate list size during build
        ef_search: int = 100        # Candidate list size during search

        # Quantization
        use_pq: bool = True         # Product quantization enabled
        pq_subspaces: int = 48      # Number of subspaces
        pq_bits: int = 8            # Bits per subspace (256 centroids)

        # Statistics
        vector_count: int           # Total vectors
        index_size_bytes: int       # Memory usage
        last_optimized: datetime    # Last optimization timestamp
    ```

    ### PostgreSQL Metadata Schema

    ```sql
    -- Metadata table (for filtering)
    CREATE TABLE vector_metadata (
        vector_id VARCHAR(255) PRIMARY KEY,
        namespace VARCHAR(100) NOT NULL,

        -- Metadata fields (JSONB for flexibility)
        metadata JSONB NOT NULL,

        -- Commonly filtered fields (extracted for indexing)
        category VARCHAR(100),
        date DATE,
        tags TEXT[],

        -- Timestamps
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        -- Index for fast lookups
        CONSTRAINT fk_namespace FOREIGN KEY (namespace) REFERENCES namespaces(name)
    );

    -- Indexes for fast filtering
    CREATE INDEX idx_metadata_namespace ON vector_metadata(namespace);
    CREATE INDEX idx_metadata_category ON vector_metadata(category);
    CREATE INDEX idx_metadata_date ON vector_metadata(date);
    CREATE INDEX idx_metadata_tags ON vector_metadata USING GIN(tags);
    CREATE INDEX idx_metadata_json ON vector_metadata USING GIN(metadata);

    -- Namespaces table
    CREATE TABLE namespaces (
        name VARCHAR(100) PRIMARY KEY,
        dimension INT NOT NULL,
        metric VARCHAR(20) NOT NULL,
        vector_count INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW()
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Insert Path

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway
        participant IndexManager
        participant Shard1
        participant Shard2
        participant PG as PostgreSQL
        participant S3

        Client->>Gateway: POST /vectors/insert<br/>[vec_123, vec_456]
        Gateway->>Gateway: Validate dimensions<br/>Rate limit check
        Gateway->>IndexManager: Route batch

        IndexManager->>IndexManager: Partition by ID hash<br/>vec_123 → Shard1<br/>vec_456 → Shard2

        par Parallel Insert
            IndexManager->>Shard1: Insert vec_123
            Shard1->>Shard1: 1. Quantize vector (PQ)<br/>2. Insert into HNSW graph<br/>3. Update graph edges
            Shard1->>PG: Insert metadata (async)
            Shard1-->>IndexManager: OK (45ms)
        and
            IndexManager->>Shard2: Insert vec_456
            Shard2->>Shard2: 1. Quantize vector<br/>2. Insert into HNSW
            Shard2->>PG: Insert metadata (async)
            Shard2-->>IndexManager: OK (42ms)
        end

        IndexManager-->>Gateway: Inserted: 2 vectors
        Gateway-->>Client: 200 OK<br/>{"inserted_count": 2}

        Note over Shard1,S3: Background: Snapshot to S3 (every 5 min)
        Shard1--)S3: Upload index snapshot
    ```

    ---

    ### Search Path (with Filtering)

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway
        participant QP as Query Processor
        participant PG as PostgreSQL
        participant Shard1
        participant Shard2

        Client->>Gateway: POST /vectors/search<br/>vector + filter
        Gateway->>QP: Parse query

        QP->>QP: 1. Extract filter conditions<br/>2. Determine shards to query

        alt Pre-filtering (Small Result Set)
            QP->>PG: SELECT vector_id<br/>WHERE category='tech'<br/>AND date >= '2024-01-01'
            PG-->>QP: [vec_123, vec_789, ...]<br/>(500 IDs matching filter)

            par Query Shards (Filtered)
                QP->>Shard1: Search in [vec_123, vec_456]<br/>(HNSW with allow-list)
                Shard1->>Shard1: ANN search (HNSW)<br/>Only consider allowed IDs
                Shard1-->>QP: Top-5 results (score, ID)
            and
                QP->>Shard2: Search in [vec_789, vec_1011]
                Shard2->>Shard2: ANN search (filtered)
                Shard2-->>QP: Top-5 results
            end
        else Post-filtering (Large Result Set)
            par Query All Shards
                QP->>Shard1: Search (no filter)
                Shard1-->>QP: Top-20 candidates
            and
                QP->>Shard2: Search (no filter)
                Shard2-->>QP: Top-20 candidates
            end

            QP->>PG: Get metadata for [40 candidates]
            PG-->>QP: Metadata with filter fields
            QP->>QP: Filter results<br/>Apply category/date filters
        end

        QP->>QP: Merge & re-rank results<br/>Sort by score<br/>Take top-10

        QP-->>Gateway: Top-10 results
        Gateway-->>Client: 200 OK<br/>{"results": [...], "duration_ms": 87}
    ```

    **Flow Explanation:**

    1. **Pre-filtering**: If filter is selective (<1K results), fetch IDs from PostgreSQL first, then search
    2. **Post-filtering**: If filter is broad (>10K results), search all shards, filter results after
    3. **Shard pruning**: Skip shards that don't contain filtered IDs (future optimization)

    ---

    ### Update Path (Re-indexing)

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway
        participant Shard
        participant HNSW as HNSW Index
        participant PG as PostgreSQL

        Client->>Gateway: POST /vectors/update<br/>vec_123 + new_vector
        Gateway->>Shard: Update request

        alt Vector Changed (Re-index Required)
            Shard->>HNSW: 1. Remove old node<br/>(disconnect edges)
            HNSW-->>Shard: Removed from graph

            Shard->>Shard: 2. Quantize new vector
            Shard->>HNSW: 3. Insert new node<br/>(rebuild edges)
            HNSW-->>Shard: Inserted (12ms)

            Shard->>PG: Update metadata
            PG-->>Shard: OK

            Shard-->>Gateway: Updated (35ms total)
        else Metadata Only (No Re-index)
            Shard->>PG: Update metadata
            PG-->>Shard: OK (5ms)
            Shard-->>Gateway: Updated (5ms)
        end

        Gateway-->>Client: 200 OK
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical vector database subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **HNSW Algorithm** | How to search billions of vectors in <100ms? | Hierarchical graph with greedy routing |
    | **Product Quantization** | How to compress vectors 48x without losing recall? | Cluster subspaces, store centroid IDs |
    | **Filtering Strategies** | How to combine vector search with metadata filters? | Pre-filtering vs. post-filtering heuristics |
    | **Sharding & Replication** | How to scale to billions of vectors? | Hash-based sharding, async replication |

    ---

    === "🔄 HNSW Algorithm"

        ## The Challenge

        **Problem:** Exact nearest neighbor search is O(N) - checking 5 billion vectors takes minutes.

        **Naive approach (linear scan):**

        ```python
        def exact_search(query_vector, all_vectors, k=10):
            """Brute force: check every vector"""
            scores = []
            for i, vec in enumerate(all_vectors):  # 5 billion iterations!
                score = cosine_similarity(query_vector, vec)
                scores.append((score, i))

            scores.sort(reverse=True)
            return scores[:k]  # Takes 5+ minutes for 5B vectors!
        ```

        **Solution:** HNSW (Hierarchical Navigable Small World) - O(log N) ANN search.

        ---

        ## HNSW Structure

        **Key Idea:** Build a multi-layer graph where upper layers are "highways" for fast traversal, lower layers are "local roads" for precision.

        **Layers:**

        ```
        Layer 3 (sparse):      A ------------------- G
                               |                     |
        Layer 2 (medium):      A ------- D --------- G
                               |         |           |
        Layer 1 (dense):       A -- B -- D -- E -- F G
                               |    |    |    |    | |
        Layer 0 (full graph):  A-B-C-D-E-F-G-H-I-J-K-L-M
        ```

        **Properties:**

        - **Layer 0**: All vectors connected (densely connected graph)
        - **Layer L**: Exponentially fewer nodes (probability = 1/2^L)
        - **Connections per layer**: M = 16 (configurable)
        - **Entry point**: Top layer (start search from sparse layer)

        ---

        ## HNSW Insertion Algorithm

        ```python
        import numpy as np
        from typing import List, Dict, Set
        import random

        class HNSWIndex:
            """
            HNSW (Hierarchical Navigable Small World) index implementation
            """

            def __init__(self, dimension: int, m: int = 16, ef_construction: int = 200):
                """
                Initialize HNSW index

                Args:
                    dimension: Vector dimensions
                    m: Max connections per layer (typical: 16)
                    ef_construction: Candidate list size during build (typical: 200)
                """
                self.dimension = dimension
                self.m = m  # Max neighbors per layer
                self.m_max = m  # Layer 0 can have more connections
                self.m_max_0 = m * 2  # Layer 0 max connections
                self.ef_construction = ef_construction
                self.ml = 1 / np.log(2)  # Layer assignment multiplier

                # Storage
                self.vectors = {}  # id -> vector
                self.layers = {}   # id -> max_layer
                self.graph = {}    # id -> {layer: [neighbor_ids]}
                self.entry_point = None  # Top-layer entry point

            def insert(self, vector_id: str, vector: np.ndarray):
                """
                Insert vector into HNSW graph

                Time complexity: O(log N) average

                Args:
                    vector_id: Unique ID
                    vector: Dense vector (dimension: self.dimension)
                """
                # Store vector
                self.vectors[vector_id] = vector

                # Assign random layer (exponential distribution)
                max_layer = self._assign_layer()
                self.layers[vector_id] = max_layer

                # Initialize graph structure
                self.graph[vector_id] = {layer: [] for layer in range(max_layer + 1)}

                if self.entry_point is None:
                    # First vector becomes entry point
                    self.entry_point = vector_id
                    return

                # Find nearest neighbors at each layer
                nearest = [self.entry_point]

                # Search from top layer down to target layer
                for layer in range(len(self.graph[self.entry_point]) - 1, max_layer, -1):
                    # Greedy search in current layer
                    nearest = self._search_layer(vector, nearest, 1, layer)

                # Insert and connect at each layer (target layer to 0)
                for layer in range(max_layer, -1, -1):
                    # Find ef_construction nearest neighbors
                    candidates = self._search_layer(
                        vector, nearest, self.ef_construction, layer
                    )

                    # Select M best neighbors (heuristic)
                    m = self.m if layer > 0 else self.m_max_0
                    neighbors = self._select_neighbors(vector, candidates, m)

                    # Add bidirectional links
                    for neighbor_id in neighbors:
                        self.graph[vector_id][layer].append(neighbor_id)
                        self.graph[neighbor_id][layer].append(vector_id)

                        # Prune neighbor's connections if too many
                        max_conn = self.m if layer > 0 else self.m_max_0
                        if len(self.graph[neighbor_id][layer]) > max_conn:
                            # Keep M best connections
                            neighbor_vec = self.vectors[neighbor_id]
                            neighbor_conns = self.graph[neighbor_id][layer]
                            pruned = self._select_neighbors(
                                neighbor_vec,
                                neighbor_conns,
                                max_conn
                            )
                            self.graph[neighbor_id][layer] = pruned

                    nearest = candidates

                # Update entry point if new vector is at higher layer
                if max_layer > self.layers[self.entry_point]:
                    self.entry_point = vector_id

            def _assign_layer(self) -> int:
                """
                Assign random layer using exponential distribution

                Higher layers have exponentially fewer nodes

                Returns:
                    Layer number (0 = bottom/dense, N = top/sparse)
                """
                return int(-np.log(random.uniform(0, 1)) * self.ml)

            def _search_layer(
                self,
                query: np.ndarray,
                entry_points: List[str],
                k: int,
                layer: int
            ) -> List[str]:
                """
                Greedy search in single layer

                Args:
                    query: Query vector
                    entry_points: Starting nodes
                    k: Number of nearest neighbors to find
                    layer: Layer to search in

                Returns:
                    List of k nearest neighbor IDs
                """
                visited = set(entry_points)
                candidates = []  # Min-heap (distance, id)
                results = []     # Max-heap (distance, id)

                # Initialize with entry points
                for ep_id in entry_points:
                    dist = self._distance(query, self.vectors[ep_id])
                    candidates.append((dist, ep_id))
                    results.append((-dist, ep_id))  # Negative for max-heap

                # Greedy search
                while candidates:
                    candidates.sort()  # Sort by distance (min-heap)
                    current_dist, current_id = candidates.pop(0)

                    # Stop if current is farther than k-th result
                    if results:
                        results.sort()  # Sort by negative distance (max-heap)
                        furthest_dist = -results[0][0]
                        if current_dist > furthest_dist:
                            break

                    # Explore neighbors
                    for neighbor_id in self.graph[current_id].get(layer, []):
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            dist = self._distance(query, self.vectors[neighbor_id])

                            # Add to candidates and results
                            if len(results) < k or dist < -results[0][0]:
                                candidates.append((dist, neighbor_id))
                                results.append((-dist, neighbor_id))

                                # Keep only k best results
                                if len(results) > k:
                                    results.sort()
                                    results.pop(0)  # Remove worst

                # Extract IDs from results
                return [result_id for (neg_dist, result_id) in results]

            def _select_neighbors(
                self,
                vector: np.ndarray,
                candidates: List[str],
                m: int
            ) -> List[str]:
                """
                Select M best neighbors using heuristic

                Prefer diverse neighbors (avoid clustering)

                Args:
                    vector: Reference vector
                    candidates: Candidate neighbor IDs
                    m: Number of neighbors to select

                Returns:
                    List of M best neighbor IDs
                """
                # Simple heuristic: select M nearest
                distances = [
                    (self._distance(vector, self.vectors[cand_id]), cand_id)
                    for cand_id in candidates
                ]
                distances.sort()
                return [cand_id for (dist, cand_id) in distances[:m]]

            def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
                """
                Compute distance (smaller = more similar)

                Using cosine distance: 1 - cosine_similarity
                """
                return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ```

        ---

        ## HNSW Search Algorithm

        ```python
            def search(
                self,
                query: np.ndarray,
                k: int = 10,
                ef_search: int = 100
            ) -> List[tuple]:
                """
                Search for k nearest neighbors

                Time complexity: O(log N) average

                Args:
                    query: Query vector
                    k: Number of results to return
                    ef_search: Candidate list size (higher = better recall)

                Returns:
                    List of (score, vector_id) tuples
                """
                if self.entry_point is None:
                    return []

                # Start from top layer, traverse down
                nearest = [self.entry_point]

                # Greedy search in upper layers (fast traversal)
                for layer in range(len(self.graph[self.entry_point]) - 1, 0, -1):
                    nearest = self._search_layer(query, nearest, 1, layer)

                # Detailed search in layer 0 (find ef_search candidates)
                candidates = self._search_layer(query, nearest, ef_search, layer=0)

                # Compute distances and return top-k
                results = []
                for cand_id in candidates:
                    dist = self._distance(query, self.vectors[cand_id])
                    similarity = 1.0 - dist  # Convert distance to similarity
                    results.append((similarity, cand_id))

                results.sort(reverse=True)  # Sort by similarity (descending)
                return results[:k]
        ```

        ---

        ## HNSW Performance Analysis

        **Time Complexity:**

        | Operation | Time | Explanation |
        |-----------|------|-------------|
        | **Insert** | O(log N) | Search log N layers, connect M neighbors |
        | **Search** | O(log N) | Traverse log N layers, explore M neighbors per layer |
        | **Delete** | O(M × log N) | Disconnect edges, update neighbors |
        | **Memory** | O(N × M) | N nodes, M edges per node per layer |

        **Parameter Tuning:**

        | Parameter | Value | Impact |
        |-----------|-------|--------|
        | **M** (connections) | 16 | Higher M = better recall, more memory (16-64 typical) |
        | **ef_construction** | 200 | Higher = better index quality, slower inserts (100-500) |
        | **ef_search** | 100 | Higher = better recall, slower search (50-500) |

        **Recall vs. Speed Trade-off:**

        ```
        ef_search = 50:  85% recall, 20ms search
        ef_search = 100: 95% recall, 50ms search
        ef_search = 200: 98% recall, 100ms search
        ef_search = 500: 99.5% recall, 300ms search
        ```

        **Production choice:** `M=16, ef_construction=200, ef_search=100` (95% recall, <100ms search)

    === "🗜️ Product Quantization"

        ## The Challenge

        **Problem:** Storing 5 billion vectors × 1536 dimensions × 4 bytes = 30 TB (too expensive!).

        **Requirements:**

        - Compress vectors 48x (1536 dims × 4 bytes = 6 KB → 128 bytes)
        - Maintain 95%+ recall (minimal accuracy loss)
        - Fast distance computation (still <100ms search)

        ---

        ## Product Quantization (PQ) Algorithm

        **Key Idea:** Split vector into subspaces, quantize each subspace independently using k-means clustering.

        **Example: 1536-dim vector with 48 subspaces**

        ```
        Original vector (1536 dims × 4 bytes = 6 KB):
        [0.1, 0.2, 0.3, ..., 0.5]  (1536 float32 values)

        Split into 48 subspaces (32 dims each):
        Subspace 1: [0.1, 0.2, ..., 0.31]  (dims 0-31)
        Subspace 2: [0.32, 0.33, ..., 0.63] (dims 32-63)
        ...
        Subspace 48: [0.481, 0.482, ..., 0.5] (dims 1504-1535)

        Quantize each subspace (k-means with 256 clusters):
        - Find nearest centroid (0-255) for each subspace
        - Store centroid ID (1 byte) instead of 32 floats (128 bytes)

        Compressed vector (48 bytes):
        [127, 203, 45, ..., 89]  (48 uint8 values)

        With overhead (metadata): 128 bytes total
        Compression: 6 KB / 128 bytes = 48x
        ```

        ---

        ## PQ Training Algorithm

        ```python
        import numpy as np
        from sklearn.cluster import KMeans

        class ProductQuantizer:
            """
            Product Quantization for vector compression
            """

            def __init__(self, dimension: int, n_subspaces: int = 48, n_clusters: int = 256):
                """
                Initialize PQ quantizer

                Args:
                    dimension: Vector dimensions (e.g., 1536)
                    n_subspaces: Number of subspaces (e.g., 48)
                    n_clusters: Clusters per subspace (256 = 1 byte per subspace)
                """
                self.dimension = dimension
                self.n_subspaces = n_subspaces
                self.n_clusters = n_clusters
                self.subspace_dim = dimension // n_subspaces

                # Codebooks: n_subspaces × n_clusters × subspace_dim
                # Example: 48 × 256 × 32 = 393,216 float32 values (1.5 MB total)
                self.codebooks = None

            def train(self, vectors: np.ndarray, n_iterations: int = 20):
                """
                Train PQ codebooks using k-means clustering

                Args:
                    vectors: Training vectors (N × dimension)
                    n_iterations: k-means iterations
                """
                n_vectors = vectors.shape[0]
                self.codebooks = np.zeros(
                    (self.n_subspaces, self.n_clusters, self.subspace_dim)
                )

                print(f"Training PQ on {n_vectors} vectors...")

                # Train k-means for each subspace independently
                for i in range(self.n_subspaces):
                    start_dim = i * self.subspace_dim
                    end_dim = (i + 1) * self.subspace_dim

                    # Extract subspace vectors
                    subspace_vectors = vectors[:, start_dim:end_dim]

                    # Train k-means (256 clusters)
                    print(f"Subspace {i+1}/{self.n_subspaces}: k-means clustering...")
                    kmeans = KMeans(
                        n_clusters=self.n_clusters,
                        max_iter=n_iterations,
                        n_init=1,
                        random_state=42
                    )
                    kmeans.fit(subspace_vectors)

                    # Store centroids as codebook
                    self.codebooks[i] = kmeans.cluster_centers_

                print("PQ training complete!")

            def encode(self, vector: np.ndarray) -> np.ndarray:
                """
                Encode vector into PQ codes

                Args:
                    vector: Dense vector (dimension,)

                Returns:
                    PQ codes (n_subspaces,) - uint8 array
                """
                codes = np.zeros(self.n_subspaces, dtype=np.uint8)

                for i in range(self.n_subspaces):
                    start_dim = i * self.subspace_dim
                    end_dim = (i + 1) * self.subspace_dim

                    # Extract subspace
                    subspace = vector[start_dim:end_dim]

                    # Find nearest centroid
                    centroids = self.codebooks[i]
                    distances = np.linalg.norm(centroids - subspace, axis=1)
                    codes[i] = np.argmin(distances)  # 0-255

                return codes

            def decode(self, codes: np.ndarray) -> np.ndarray:
                """
                Decode PQ codes back to approximate vector

                Args:
                    codes: PQ codes (n_subspaces,)

                Returns:
                    Reconstructed vector (dimension,)
                """
                vector = np.zeros(self.dimension)

                for i in range(self.n_subspaces):
                    start_dim = i * self.subspace_dim
                    end_dim = (i + 1) * self.subspace_dim

                    # Lookup centroid from codebook
                    centroid_id = codes[i]
                    centroid = self.codebooks[i][centroid_id]

                    # Reconstruct subspace
                    vector[start_dim:end_dim] = centroid

                return vector

            def compute_distance(
                self,
                query: np.ndarray,
                codes: np.ndarray
            ) -> float:
                """
                Compute distance using asymmetric distance computation (ADC)

                Faster than decoding + computing distance

                Args:
                    query: Query vector (dimension,)
                    codes: PQ codes (n_subspaces,)

                Returns:
                    Approximate distance
                """
                distance = 0.0

                for i in range(self.n_subspaces):
                    start_dim = i * self.subspace_dim
                    end_dim = (i + 1) * self.subspace_dim

                    # Query subspace
                    query_subspace = query[start_dim:end_dim]

                    # Centroid subspace
                    centroid_id = codes[i]
                    centroid_subspace = self.codebooks[i][centroid_id]

                    # Accumulate squared distance
                    distance += np.sum((query_subspace - centroid_subspace) ** 2)

                return np.sqrt(distance)


        # Example usage
        dimension = 1536
        n_vectors = 10000

        # Generate random training vectors
        training_vectors = np.random.randn(n_vectors, dimension).astype(np.float32)

        # Normalize (for cosine similarity)
        training_vectors /= np.linalg.norm(training_vectors, axis=1, keepdims=True)

        # Train PQ
        pq = ProductQuantizer(dimension=dimension, n_subspaces=48, n_clusters=256)
        pq.train(training_vectors)

        # Encode a vector
        test_vector = training_vectors[0]
        codes = pq.encode(test_vector)
        print(f"Original size: {test_vector.nbytes} bytes")  # 6,144 bytes
        print(f"Compressed size: {codes.nbytes} bytes")      # 48 bytes
        print(f"Compression ratio: {test_vector.nbytes / codes.nbytes:.1f}x")  # 128x

        # Decode and measure error
        reconstructed = pq.decode(codes)
        error = np.linalg.norm(test_vector - reconstructed)
        print(f"Reconstruction error: {error:.4f}")  # ~0.15 (small error)

        # Compute distance (fast, no decoding)
        query = np.random.randn(dimension).astype(np.float32)
        query /= np.linalg.norm(query)
        distance = pq.compute_distance(query, codes)
        print(f"Distance: {distance:.4f}")
        ```

        ---

        ## Asymmetric Distance Computation (ADC)

        **Key optimization:** Don't decode quantized vectors during search!

        **Standard approach (slow):**

        ```python
        # 1. Decode all vectors (expensive!)
        for codes in all_codes:
            reconstructed = pq.decode(codes)  # 48 lookups
            distance = np.linalg.norm(query - reconstructed)
        ```

        **ADC approach (10x faster):**

        ```python
        # Precompute query-centroid distances (once per query)
        distance_table = np.zeros((n_subspaces, n_clusters))
        for i in range(n_subspaces):
            for j in range(n_clusters):
                query_subspace = query[i*32:(i+1)*32]
                centroid = codebooks[i][j]
                distance_table[i, j] = np.linalg.norm(query_subspace - centroid)

        # Compute distance using lookup table (fast!)
        for codes in all_codes:
            distance = 0
            for i in range(n_subspaces):
                distance += distance_table[i, codes[i]]  # Just a lookup!
        ```

        **Performance:**

        - Precompute: 48 subspaces × 256 centroids × 32 dims = 393K ops (once per query)
        - Distance per vector: 48 lookups + 48 additions (vs. 1536 multiplications)
        - Speedup: 10-20x faster than exact distance

        ---

        ## PQ Trade-offs

        | Configuration | Compression | Recall | Memory per Vector |
        |---------------|-------------|--------|-------------------|
        | **No PQ** | 1x | 100% | 6 KB (1536 × 4 bytes) |
        | **PQ: 48 subspaces, 256 clusters** | 48x | 95-97% | 128 bytes (48 bytes + overhead) |
        | **PQ: 96 subspaces, 256 clusters** | 32x | 97-98% | 192 bytes (96 bytes + overhead) |
        | **Binary quantization (1-bit)** | 256x | 80-85% | 24 bytes (1536 / 8) |

        **Production choice:** 48 subspaces, 256 clusters (95%+ recall, 48x compression)

    === "🔍 Filtering Strategies"

        ## The Challenge

        **Problem:** Combine vector similarity with metadata filters efficiently.

        **Example query:**

        ```
        Find 10 most similar documents to query vector
        WHERE category = 'technology'
          AND date >= '2024-01-01'
          AND tags CONTAINS 'AI'
        ```

        **Naive approach (post-filtering):**

        1. Search HNSW for 10 nearest neighbors (ignoring filters)
        2. Filter results by metadata
        3. Problem: May return < 10 results if filtered out!

        **Better approach:** Pre-filtering or hybrid filtering

        ---

        ## Filtering Strategies

        ### 1. Post-Filtering (Simple, Low Selectivity)

        **When to use:** Filter is broad (returns >50% of vectors)

        **Algorithm:**

        ```python
        def post_filter_search(
            query_vector: np.ndarray,
            k: int,
            filters: dict
        ) -> List[tuple]:
            """
            Post-filtering: search first, filter after

            Best when filter is non-selective (broad)

            Args:
                query_vector: Query vector
                k: Desired result count
                filters: Metadata filters {"category": "tech", "date": "2024-01-01"}

            Returns:
                List of (score, id, metadata) tuples
            """
            # Search HNSW for more candidates than needed
            # Fetch 5x more to account for filtering
            over_request = k * 5  # Heuristic: request 5x more

            candidates = hnsw_index.search(query_vector, k=over_request)

            # Fetch metadata and filter
            results = []
            for score, vector_id in candidates:
                metadata = get_metadata(vector_id)

                # Apply filters
                if self._match_filters(metadata, filters):
                    results.append((score, vector_id, metadata))

                    if len(results) >= k:
                        break

            return results

        def _match_filters(self, metadata: dict, filters: dict) -> bool:
            """Check if metadata matches all filters"""
            for key, condition in filters.items():
                if key not in metadata:
                    return False

                # Handle different operators
                if isinstance(condition, dict):
                    if "$eq" in condition and metadata[key] != condition["$eq"]:
                        return False
                    if "$gte" in condition and metadata[key] < condition["$gte"]:
                        return False
                    if "$in" in condition and metadata[key] not in condition["$in"]:
                        return False
                else:
                    # Simple equality
                    if metadata[key] != condition:
                        return False

            return True
        ```

        **Pros:**

        - Simple to implement
        - No index modifications
        - Works with any filter

        **Cons:**

        - May return < k results
        - Wastes computation on filtered-out vectors
        - Poor for selective filters (<10% match)

        ---

        ### 2. Pre-Filtering (Complex, High Selectivity)

        **When to use:** Filter is selective (returns <10% of vectors)

        **Algorithm:**

        ```python
        def pre_filter_search(
            query_vector: np.ndarray,
            k: int,
            filters: dict
        ) -> List[tuple]:
            """
            Pre-filtering: filter first, search within allowed set

            Best when filter is very selective (narrow)

            Args:
                query_vector: Query vector
                k: Desired result count
                filters: Metadata filters

            Returns:
                List of (score, id, metadata) tuples
            """
            # Step 1: Query PostgreSQL for IDs matching filter
            allowed_ids = self._query_metadata_store(filters)

            print(f"Filter matched {len(allowed_ids)} vectors")

            # Step 2: Search HNSW within allowed set only
            results = self._filtered_hnsw_search(
                query_vector,
                k,
                allowed_ids
            )

            return results

        def _query_metadata_store(self, filters: dict) -> Set[str]:
            """
            Query PostgreSQL for matching IDs

            Returns:
                Set of vector IDs matching filters
            """
            # Build SQL query
            conditions = []
            params = []

            if "category" in filters:
                conditions.append("category = %s")
                params.append(filters["category"])

            if "date" in filters and "$gte" in filters["date"]:
                conditions.append("date >= %s")
                params.append(filters["date"]["$gte"])

            if "tags" in filters and "$in" in filters["tags"]:
                conditions.append("tags && %s")  # Array overlap
                params.append(filters["tags"]["$in"])

            where_clause = " AND ".join(conditions)
            query = f"SELECT vector_id FROM vector_metadata WHERE {where_clause}"

            # Execute query (indexed, fast)
            cursor = pg_connection.execute(query, params)
            return {row[0] for row in cursor.fetchall()}

        def _filtered_hnsw_search(
            self,
            query_vector: np.ndarray,
            k: int,
            allowed_ids: Set[str]
        ) -> List[tuple]:
            """
            Modified HNSW search that only considers allowed IDs

            Args:
                query_vector: Query vector
                k: Result count
                allowed_ids: Set of allowed vector IDs

            Returns:
                Top-k results within allowed set
            """
            # Modified search that skips disallowed nodes
            results = []
            visited = set()
            candidates = [(0, self.entry_point)]

            while candidates and len(results) < k * 10:  # Explore more
                candidates.sort()
                dist, node_id = candidates.pop(0)

                if node_id in visited:
                    continue
                visited.add(node_id)

                # Check if allowed
                if node_id in allowed_ids:
                    results.append((1 - dist, node_id))  # Convert to similarity

                # Explore neighbors (even if current is disallowed)
                for neighbor_id in self.graph[node_id].get(0, []):
                    if neighbor_id not in visited:
                        neighbor_dist = self._distance(
                            query_vector,
                            self.vectors[neighbor_id]
                        )
                        candidates.append((neighbor_dist, neighbor_id))

            # Sort and return top-k
            results.sort(reverse=True)
            return results[:k]
        ```

        **Pros:**

        - Efficient for selective filters
        - Always returns k results (if available)
        - No wasted computation on filtered-out vectors

        **Cons:**

        - Requires PostgreSQL query (adds latency)
        - HNSW graph exploration less efficient (sparse allowed set)
        - Complex implementation

        ---

        ### 3. Hybrid Strategy (Adaptive)

        **Best approach:** Choose pre/post-filtering based on filter selectivity.

        ```python
        def adaptive_filter_search(
            query_vector: np.ndarray,
            k: int,
            filters: dict
        ) -> List[tuple]:
            """
            Adaptive filtering: choose strategy based on selectivity

            Args:
                query_vector: Query vector
                k: Desired result count
                filters: Metadata filters

            Returns:
                List of (score, id, metadata) tuples
            """
            # Estimate filter selectivity (fast count query)
            selectivity = self._estimate_selectivity(filters)

            print(f"Filter selectivity: {selectivity:.1%}")

            if selectivity < 0.1:
                # Very selective (< 10%): pre-filter
                print("Using pre-filtering strategy")
                return self.pre_filter_search(query_vector, k, filters)

            elif selectivity > 0.5:
                # Broad filter (> 50%): post-filter
                print("Using post-filtering strategy")
                return self.post_filter_search(query_vector, k, filters)

            else:
                # Medium selectivity (10-50%): hybrid
                print("Using hybrid strategy")
                return self._hybrid_search(query_vector, k, filters)

        def _estimate_selectivity(self, filters: dict) -> float:
            """
            Estimate what fraction of vectors match filter

            Uses PostgreSQL EXPLAIN or cached statistics

            Returns:
                Selectivity (0.0 to 1.0)
            """
            # Build count query
            conditions = []
            params = []

            if "category" in filters:
                conditions.append("category = %s")
                params.append(filters["category"])

            if "date" in filters and "$gte" in filters["date"]:
                conditions.append("date >= %s")
                params.append(filters["date"]["$gte"])

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT COUNT(*) FROM vector_metadata WHERE {where_clause}"

            # Execute count (indexed, fast)
            cursor = pg_connection.execute(query, params)
            match_count = cursor.fetchone()[0]

            # Selectivity = matches / total
            total_count = self.vector_count
            return match_count / total_count

        def _hybrid_search(
            self,
            query_vector: np.ndarray,
            k: int,
            filters: dict
        ) -> List[tuple]:
            """
            Hybrid: partially filter, then search

            Good for medium selectivity (10-50%)
            """
            # Fetch subset of matching IDs (e.g., 10K IDs)
            allowed_ids = self._query_metadata_store(filters)

            if len(allowed_ids) < k:
                # Too few matches, return all
                return self._filtered_hnsw_search(query_vector, len(allowed_ids), allowed_ids)

            # Search with soft filtering (prefer allowed, but explore others)
            return self._soft_filtered_search(query_vector, k, allowed_ids)
        ```

        **Performance Comparison:**

        | Strategy | Selectivity | Latency | Recall |
        |----------|------------|---------|--------|
        | **Post-filtering** | >50% | 50ms | 95% (may return <k) |
        | **Pre-filtering** | <10% | 80ms | 95% (always k results) |
        | **Hybrid** | 10-50% | 65ms | 95% |
        | **Adaptive** | Any | 50-80ms | 95% (chooses best) |

        **Production choice:** Adaptive filtering (automatically chooses best strategy)

    === "🌐 Sharding & Replication"

        ## The Challenge

        **Problem:** Single node can't store 5 billion vectors (4 TB data, 100K QPS).

        **Requirements:**

        - Distribute vectors across 200 nodes (25M vectors each)
        - High availability (replicate each shard 3x)
        - Even load distribution (no hotspots)
        - Fast routing (client knows which shard to query)

        ---

        ## Sharding Strategy

        **Approach: Hash-based sharding by vector ID**

        ```python
        import hashlib
        from typing import List

        class VectorShardRouter:
            """
            Routes vector operations to correct shard using consistent hashing
            """

            def __init__(self, n_shards: int = 200):
                """
                Initialize shard router

                Args:
                    n_shards: Number of shards (partitions)
                """
                self.n_shards = n_shards
                self.shard_nodes = {}  # shard_id -> [master, replica1, replica2]

                # Initialize shard topology
                for shard_id in range(n_shards):
                    self.shard_nodes[shard_id] = {
                        "master": f"node-{shard_id}-master",
                        "replicas": [
                            f"node-{shard_id}-replica-1",
                            f"node-{shard_id}-replica-2"
                        ]
                    }

            def get_shard_id(self, vector_id: str) -> int:
                """
                Compute shard ID from vector ID using hash

                Args:
                    vector_id: Vector identifier

                Returns:
                    Shard ID (0 to n_shards-1)
                """
                # Hash vector ID to shard
                hash_bytes = hashlib.md5(vector_id.encode()).digest()
                hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
                return hash_int % self.n_shards

            def route_insert(self, vector_id: str) -> str:
                """
                Route insert to master node

                Args:
                    vector_id: Vector ID

                Returns:
                    Master node address
                """
                shard_id = self.get_shard_id(vector_id)
                return self.shard_nodes[shard_id]["master"]

            def route_search(
                self,
                query_vector: np.ndarray,
                k: int,
                filters: dict = None
            ) -> List[tuple]:
                """
                Route search to all shards (scatter-gather)

                Args:
                    query_vector: Query vector
                    k: Desired result count
                    filters: Optional metadata filters

                Returns:
                    Top-k results from all shards combined
                """
                # Query all shards in parallel
                shard_results = []

                # Use thread pool for parallel queries
                with ThreadPoolExecutor(max_workers=self.n_shards) as executor:
                    futures = []

                    for shard_id in range(self.n_shards):
                        # Route to replica (load balance reads)
                        node = self._select_replica(shard_id)

                        # Submit search request
                        future = executor.submit(
                            self._search_shard,
                            node,
                            query_vector,
                            k,
                            filters
                        )
                        futures.append(future)

                    # Gather results
                    for future in futures:
                        shard_results.extend(future.result())

                # Merge and re-rank results from all shards
                shard_results.sort(key=lambda x: x[0], reverse=True)  # Sort by score
                return shard_results[:k]  # Return top-k globally

            def _select_replica(self, shard_id: int) -> str:
                """
                Select replica for read (load balancing)

                Strategy: Round-robin or least-loaded

                Args:
                    shard_id: Shard ID

                Returns:
                    Node address (replica)
                """
                # Simple round-robin (can use least-loaded for better balance)
                replicas = self.shard_nodes[shard_id]["replicas"]
                return replicas[random.randint(0, len(replicas) - 1)]

            def _search_shard(
                self,
                node: str,
                query_vector: np.ndarray,
                k: int,
                filters: dict
            ) -> List[tuple]:
                """
                Execute search on single shard

                Args:
                    node: Node address
                    query_vector: Query vector
                    k: Result count
                    filters: Metadata filters

                Returns:
                    Top-k results from this shard
                """
                # Send HTTP request to shard node
                response = requests.post(
                    f"http://{node}:8000/search",
                    json={
                        "vector": query_vector.tolist(),
                        "k": k,
                        "filters": filters
                    },
                    timeout=1.0  # 1 second timeout
                )

                if response.status_code == 200:
                    return response.json()["results"]
                else:
                    # Handle failure (return empty, retry, or skip)
                    print(f"Shard {node} failed: {response.status_code}")
                    return []
        ```

        ---

        ## Replication Strategy

        **Master-Replica Topology:**

        ```
        Shard 1:
        ├── Master (writes + reads)
        ├── Replica 1 (reads, hot standby)
        └── Replica 2 (reads, hot standby)

        Shard 2:
        ├── Master (writes + reads)
        ├── Replica 1 (reads, hot standby)
        └── Replica 2 (reads, hot standby)
        ```

        **Replication Flow:**

        ```python
        class ReplicationManager:
            """
            Manage async replication from master to replicas
            """

            def __init__(self, master_node: str, replica_nodes: List[str]):
                self.master_node = master_node
                self.replica_nodes = replica_nodes
                self.replication_log = []  # Write-ahead log

            def replicate_insert(
                self,
                vector_id: str,
                vector: np.ndarray,
                metadata: dict
            ):
                """
                Replicate insert to all replicas

                Args:
                    vector_id: Vector ID
                    vector: Vector data
                    metadata: Metadata
                """
                # Write to master first (synchronous)
                self._write_master(vector_id, vector, metadata)

                # Replicate to replicas (asynchronous)
                self._replicate_async(vector_id, vector, metadata)

            def _write_master(
                self,
                vector_id: str,
                vector: np.ndarray,
                metadata: dict
            ):
                """Write to master node (blocks until complete)"""
                # Insert into HNSW index
                hnsw_index.insert(vector_id, vector)

                # Write to metadata store
                metadata_store.insert(vector_id, metadata)

                # Append to replication log
                self.replication_log.append({
                    "op": "insert",
                    "vector_id": vector_id,
                    "vector": vector,
                    "metadata": metadata,
                    "timestamp": time.time()
                })

            def _replicate_async(
                self,
                vector_id: str,
                vector: np.ndarray,
                metadata: dict
            ):
                """
                Asynchronously replicate to replicas

                Uses replication log (streaming or batch)
                """
                # Send to replicas in background
                for replica in self.replica_nodes:
                    threading.Thread(
                        target=self._send_to_replica,
                        args=(replica, vector_id, vector, metadata)
                    ).start()

            def _send_to_replica(
                self,
                replica: str,
                vector_id: str,
                vector: np.ndarray,
                metadata: dict
            ):
                """Send insert to replica node"""
                try:
                    response = requests.post(
                        f"http://{replica}:8000/replicate",
                        json={
                            "vector_id": vector_id,
                            "vector": vector.tolist(),
                            "metadata": metadata
                        },
                        timeout=2.0
                    )
                    if response.status_code != 200:
                        print(f"Replication to {replica} failed")
                except Exception as e:
                    print(f"Replication error: {e}")
        ```

        **Replication Lag:**

        - Typical lag: 50-200ms (async replication)
        - Can read from master for consistency
        - Eventual consistency acceptable for most use cases

        ---

        ## Failover Strategy

        **Automatic failover using health checks:**

        ```python
        class FailoverManager:
            """
            Monitor master health and promote replica on failure
            """

            def __init__(self, shard_id: int):
                self.shard_id = shard_id
                self.master = shard_nodes[shard_id]["master"]
                self.replicas = shard_nodes[shard_id]["replicas"]

            def monitor(self):
                """
                Continuously monitor master health
                """
                while True:
                    if not self._ping_master():
                        print(f"Master {self.master} is down! Starting failover...")
                        self._promote_replica()

                    time.sleep(5)  # Check every 5 seconds

            def _ping_master(self) -> bool:
                """Check if master is responsive"""
                try:
                    response = requests.get(
                        f"http://{self.master}:8000/health",
                        timeout=1.0
                    )
                    return response.status_code == 200
                except Exception:
                    return False

            def _promote_replica(self):
                """
                Promote replica to master

                Steps:
                1. Select best replica (most up-to-date)
                2. Promote to master
                3. Update routing table
                4. Notify clients
                """
                # Select replica with lowest lag
                best_replica = self.replicas[0]  # Simplified

                # Promote to master
                print(f"Promoting {best_replica} to master")
                requests.post(
                    f"http://{best_replica}:8000/promote",
                    json={"role": "master"}
                )

                # Update routing table
                shard_nodes[self.shard_id]["master"] = best_replica

                # Update other replicas to follow new master
                for replica in self.replicas[1:]:
                    requests.post(
                        f"http://{replica}:8000/set_master",
                        json={"master": best_replica}
                    )
        ```

        **Failover Time:**

        - Detection: 5-10 seconds (health check interval)
        - Promotion: 2-5 seconds (replica becomes master)
        - Total downtime: 10-15 seconds

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling vector database from 1K to 100K QPS, 1M to 5B vectors.

    **Scaling challenges at 100K QPS, 5B vectors:**

    - **Memory:** 4 TB total (200 shards × 20 GB each)
    - **Network:** 8 Gbps total (40 Mbps per shard)
    - **CPU:** 800 cores (200 shards × 4 cores each)
    - **Latency:** <100ms p99 (including scatter-gather)

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Distance computation** | ✅ Yes | SIMD (AVX-512), GPU acceleration, PQ compression |
    | **Scatter-gather latency** | ✅ Yes | Query caching, hot shard optimization, async queries |
    | **Index building** | 🟡 Approaching | Background indexing, batch inserts, delta indexes |
    | **Network bandwidth** | ❌ No | 40 Mbps per shard (well below 1 Gbps NIC) |
    | **Metadata filtering** | 🟡 Approaching | Indexed columns, PostgreSQL tuning, denormalization |

    ---

    ## Performance Optimization

    ### 1. SIMD Acceleration (Distance Computation)

    **Problem:** Computing cosine similarity for 1536-dim vectors is CPU-intensive.

    ```python
    # Naive implementation (slow)
    def cosine_similarity_naive(v1, v2):
        """Scalar implementation: ~100 μs"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm1 * norm2)

    # SIMD implementation (fast)
    def cosine_similarity_simd(v1, v2):
        """
        Vectorized using NumPy (SIMD under the hood)

        Time: ~5 μs (20x faster!)
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Even faster: precomputed norms
    def cosine_similarity_optimized(v1, v2, norm1, norm2):
        """
        Precompute norms (store with vectors)

        Time: ~2 μs (50x faster than naive!)
        """
        return np.dot(v1, v2) / (norm1 * norm2)
    ```

    **SIMD libraries:**

    - NumPy (uses BLAS/MKL with AVX-512)
    - Faiss (Facebook's optimized vector search library)
    - HNSW-lib (C++ with SIMD intrinsics)

    ---

    ### 2. GPU Acceleration (Batch Search)

    **Problem:** Batch searches (100 queries/batch) can be parallelized on GPU.

    ```python
    import cupy as cp  # GPU-accelerated NumPy

    class GPUVectorSearch:
        """
        GPU-accelerated vector search using CUDA
        """

        def __init__(self, vectors: np.ndarray):
            """
            Initialize GPU search

            Args:
                vectors: All vectors (N × dimension) on GPU
            """
            # Transfer vectors to GPU memory
            self.vectors_gpu = cp.array(vectors, dtype=cp.float32)
            self.norms_gpu = cp.linalg.norm(self.vectors_gpu, axis=1)

        def batch_search(
            self,
            queries: np.ndarray,
            k: int = 10
        ) -> np.ndarray:
            """
            Batch search on GPU

            Args:
                queries: Query vectors (batch_size × dimension)
                k: Top-k results per query

            Returns:
                Top-k IDs for each query (batch_size × k)
            """
            # Transfer queries to GPU
            queries_gpu = cp.array(queries, dtype=cp.float32)
            query_norms = cp.linalg.norm(queries_gpu, axis=1, keepdims=True)

            # Compute all pairwise similarities (batch_size × N)
            # Uses GPU matrix multiplication (10-100x faster)
            similarities = cp.dot(queries_gpu, self.vectors_gpu.T)
            similarities /= (query_norms * self.norms_gpu)

            # Get top-k for each query
            top_k_indices = cp.argsort(similarities, axis=1)[:, -k:][:, ::-1]

            # Transfer results back to CPU
            return cp.asnumpy(top_k_indices)

    # Usage
    n_vectors = 25_000_000  # 25M vectors per shard
    dimension = 1536
    batch_size = 100

    # CPU: 100 queries × 50ms each = 5 seconds
    # GPU: 100 queries in batch = 200ms (25x faster!)
    ```

    **GPU Performance:**

    - CPU (single query): 50ms
    - CPU (100 queries, sequential): 5 seconds
    - GPU (100 queries, batch): 200ms (25x faster)
    - Cost: NVIDIA A100 GPU ($2/hour vs. $0.50/hour for 16 CPU cores)

    ---

    ### 3. Query Caching (Hot Vectors)

    **Problem:** Top 10% of queries account for 80% of traffic (Zipf distribution).

    ```python
    from functools import lru_cache
    import hashlib

    class CachedVectorSearch:
        """
        Vector search with query result caching
        """

        def __init__(self, hnsw_index, cache_size: int = 10000):
            self.hnsw_index = hnsw_index
            self.cache_size = cache_size

        def search(
            self,
            query_vector: np.ndarray,
            k: int = 10
        ) -> List[tuple]:
            """
            Search with caching

            Cache hit: 1ms (memory lookup)
            Cache miss: 50ms (HNSW search)
            """
            # Compute cache key (hash of query vector)
            cache_key = self._hash_vector(query_vector)

            # Check cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # Cache miss: perform search
            results = self.hnsw_index.search(query_vector, k)

            # Store in cache (TTL: 60 seconds)
            self._set_cache(cache_key, results, ttl=60)

            return results

        def _hash_vector(self, vector: np.ndarray) -> str:
            """
            Compute hash of vector for cache key

            Quantize to reduce precision (increase cache hits)
            """
            # Round to 2 decimal places (allows small variations)
            quantized = np.round(vector, decimals=2)
            vector_bytes = quantized.tobytes()
            return hashlib.md5(vector_bytes).hexdigest()

        @lru_cache(maxsize=10000)
        def _get_from_cache(self, cache_key: str):
            """Get from LRU cache"""
            return None  # Placeholder

        def _set_cache(self, cache_key: str, results, ttl: int):
            """Set in cache with TTL"""
            # Use Redis or in-memory LRU cache
            pass
    ```

    **Cache Performance:**

    - Cache hit rate: 30-40% (for hot queries)
    - Cache hit latency: 1ms (vs. 50ms for miss)
    - Effective latency: 0.4 × 1ms + 0.6 × 50ms = 30.4ms (40% improvement)

    ---

    ### 4. Index Optimization (Background Compaction)

    **Problem:** Updates and deletes fragment HNSW index (performance degrades over time).

    ```python
    class IndexOptimizer:
        """
        Background index optimization and compaction
        """

        def __init__(self, hnsw_index):
            self.hnsw_index = hnsw_index
            self.last_optimization = time.time()

        def should_optimize(self) -> bool:
            """
            Check if optimization is needed

            Criteria:
            - Deleted vectors > 10% of total
            - 24 hours since last optimization
            - Fragmentation > 20%
            """
            deleted_ratio = self.hnsw_index.deleted_count / self.hnsw_index.vector_count
            time_since_last = time.time() - self.last_optimization

            return (
                deleted_ratio > 0.1 or
                time_since_last > 24 * 3600 or
                self.hnsw_index.fragmentation > 0.2
            )

        def optimize(self):
            """
            Optimize index (background task)

            Steps:
            1. Build new index from scratch (active vectors only)
            2. Swap old index with new index (atomic)
            3. Delete old index

            Time: 5-10 minutes for 25M vectors
            """
            print("Starting index optimization...")

            # Build new index
            new_index = HNSWIndex(dimension=self.hnsw_index.dimension)

            # Copy active vectors (skip deleted)
            for vector_id, vector in self.hnsw_index.vectors.items():
                if not self.hnsw_index.is_deleted(vector_id):
                    new_index.insert(vector_id, vector)

            # Atomic swap
            old_index = self.hnsw_index
            self.hnsw_index = new_index

            # Delete old index
            del old_index

            self.last_optimization = time.time()
            print("Index optimization complete!")
    ```

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold | Tool |
    |--------|--------|-----------------|------|
    | **Search Latency (P99)** | < 100ms | > 200ms | Prometheus, Grafana |
    | **Insert Latency (P99)** | < 50ms | > 100ms | Prometheus |
    | **Recall @ k=10** | > 95% | < 90% | Custom benchmark |
    | **Cache Hit Rate** | > 30% | < 20% | Redis INFO |
    | **Shard CPU Usage** | < 70% | > 90% | Node exporter |
    | **Replication Lag** | < 200ms | > 1 second | Custom metric |
    | **Index Fragmentation** | < 10% | > 20% | Custom metric |

    **Example Prometheus query:**

    ```promql
    # P99 search latency
    histogram_quantile(0.99, rate(vector_search_duration_seconds_bucket[5m]))

    # Recall metric (custom)
    vector_search_recall_at_k{k="10"}

    # Cache hit rate
    rate(vector_search_cache_hits_total[5m]) /
    (rate(vector_search_cache_hits_total[5m]) + rate(vector_search_cache_misses_total[5m]))
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 100K QPS, 5B vectors:**

    | Component | Cost |
    |-----------|------|
    | **200 master nodes** | $14,400 (200 × c6i.2xlarge @ $0.10/hr × 720hr) |
    | **400 replica nodes** | $28,800 (400 × c6i.2xlarge) |
    | **PostgreSQL (metadata)** | $1,440 (RDS db.r6i.2xlarge @ $2/hr) |
    | **S3 (backups)** | $500 (10 TB @ $0.023/GB) |
    | **Network transfer** | $1,800 (20 TB egress @ $0.09/GB) |
    | **Total** | **$46,940/month** |

    **Optimization:**

    - Use reserved instances (40% discount): $28,164/month
    - Use spot instances for replicas (70% discount): $21,882/month
    - Compress PQ further (96 subspaces): $19,098/month (less memory)

    ---

    ## Disaster Recovery

    **Backup strategies:**

    1. **Index snapshots** (every 5 minutes to S3)
    2. **Replication log** (replay inserts/updates)
    3. **Cross-region replication** (geo-redundancy)

    ```bash
    # Snapshot index to S3 (background task)
    python snapshot_index.py --shard-id 1 --output s3://backups/shard-1/

    # Restore from snapshot
    python restore_index.py --shard-id 1 --input s3://backups/shard-1/latest.hnsw

    # Cross-region replication
    aws s3 sync s3://backups-us-east-1/ s3://backups-eu-west-1/ --delete
    ```

    **Recovery time:**

    - Index snapshot restore: 2-5 minutes (load from S3)
    - Replication log replay: 5-10 minutes (rebuild index)
    - Cross-region failover: 10-15 minutes (promote replica cluster)

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **HNSW for ANN search:** 95% recall, O(log N) search time, best-in-class performance
    2. **Product quantization (PQ):** 48x compression, <1% recall loss, 10x faster distance computation
    3. **Hash-based sharding:** Even distribution, simple routing, no hotspots
    4. **Adaptive filtering:** Choose pre/post-filtering based on selectivity
    5. **Master-replica replication:** High availability (99.99%), async replication (<200ms lag)
    6. **Hybrid search:** Combine vector (HNSW) with keyword (BM25) for better results

    ---

    ## Interview Tips

    ✅ **Start with HNSW** - Core ANN algorithm, explain hierarchical graph structure

    ✅ **Discuss PQ compression** - 48x memory savings, explain subspace clustering

    ✅ **Explain filtering strategies** - Pre vs. post-filtering trade-offs

    ✅ **Mention sharding** - Hash-based partitioning, scatter-gather queries

    ✅ **Cover replication** - Async replication for availability, eventual consistency

    ✅ **Discuss recall vs. speed** - ANN is 100x faster than exact search, 95% recall acceptable

    ✅ **GPU acceleration** - 25x faster for batch queries, cost-effective for high QPS

    ✅ **Hybrid search** - Combine vector + keyword for best results (alpha blending)

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"Why HNSW over other ANN algorithms?"** | HNSW has best recall (95%+), fast inserts (O(log N)), good for dynamic data. IVF has slower inserts, Annoy has lower recall. |
    | **"How to handle updates?"** | Updates require re-indexing (expensive). Use soft deletes + background compaction. Metadata-only updates are fast. |
    | **"How does product quantization work?"** | Split vector into subspaces (48), cluster each subspace (256 centroids), store centroid IDs (1 byte each). 48x compression. |
    | **"How to filter with metadata?"** | Pre-filtering (selective filters, query PostgreSQL first), post-filtering (broad filters, search then filter), adaptive (choose based on selectivity). |
    | **"How to scale to 1 trillion vectors?"** | Increase shards (10,000+ shards), use hierarchical sharding (shard groups), improve PQ compression (128x), use distributed index. |
    | **"Why scatter-gather for search?"** | Vectors distributed across shards, must query all shards to find global top-k. Parallelize queries (100ms latency). |
    | **"How to ensure high availability?"** | Master-replica replication (3x), automatic failover (10-15s downtime), health checks (5s interval), cross-region replication. |
    | **"Hybrid search vs. pure vector search?"** | Hybrid combines semantic (vector) and lexical (keyword) search. Better for queries with specific terms. Use alpha blending (0.7 vector + 0.3 keyword). |

    ---

    ## Real-World Examples

    **Pinecone:**
    - Managed vector database (SaaS)
    - Uses HNSW + custom optimizations
    - 2B+ vectors, <100ms p99 latency
    - $70M+ funding, major AI companies as customers

    **Weaviate:**
    - Open-source vector database
    - HNSW index, hybrid search (vector + BM25)
    - Used by Nvidia, Red Hat, Stack Overflow
    - 10M+ downloads

    **OpenAI:**
    - Uses Pinecone for ChatGPT retrieval
    - Billions of embeddings (text-embedding-ada-002)
    - Powers ChatGPT plugins, RAG applications

    **Anthropic:**
    - Vector search for long context (200K tokens)
    - Custom HNSW implementation
    - Retrieval-augmented generation (RAG)

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Pinecone, Weaviate, Qdrant, Milvus, Chroma, OpenAI, Anthropic, Google, Meta

---

*Master this problem and you'll be ready for: AI/ML infrastructure, search systems, recommendation engines, RAG applications, embedding-based retrieval*
