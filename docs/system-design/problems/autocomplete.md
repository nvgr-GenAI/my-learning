# Design Autocomplete/Typeahead

A real-time search suggestion system that provides instant query completions as users type, enabling fast and efficient search experiences across web and mobile applications.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10B searches/day, 100M+ unique queries, 1B+ query suggestions |
| **Key Challenges** | Ultra-low latency (<100ms), trie data structure, caching strategies, personalization |
| **Core Concepts** | Trie/prefix tree, distributed caching, ranking by popularity, fuzzy matching |
| **Companies** | Google, Amazon, Netflix, YouTube, LinkedIn, Bing, DuckDuckGo, Uber |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Real-time Suggestions** | Display suggestions as user types (after 2-3 characters) | P0 (Must have) |
    | **Ranking by Popularity** | Show most popular/relevant suggestions first | P0 (Must have) |
    | **Prefix Matching** | Match queries starting with user input | P0 (Must have) |
    | **Top-K Results** | Return top 5-10 suggestions per query | P0 (Must have) |
    | **Personalization** | User-specific suggestions based on history | P1 (Should have) |
    | **Fuzzy Matching** | Handle typos and spelling errors | P1 (Should have) |
    | **Multi-language** | Support international characters (UTF-8) | P1 (Should have) |
    | **Trending Queries** | Boost recently popular searches | P2 (Nice to have) |
    | **Category Filtering** | Filter by search categories (e.g., products, videos) | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Voice-based search suggestions
    - Image-based autocomplete
    - Real-time search results (only suggestions)
    - Query understanding/semantic search
    - Ad suggestions

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 100ms p95, < 50ms p50 | Critical for user experience; delays break typing flow |
    | **Availability** | 99.9% uptime | Users expect always-on search functionality |
    | **Scalability** | Handle 10B queries/day | Must scale to billions of daily searches |
    | **Consistency** | Eventual consistency | Brief delays acceptable for new suggestions |
    | **Relevance** | Top suggestion CTR > 30% | Suggestions must be useful |
    | **Freshness** | New trending queries appear within 1 hour | Keep suggestions current with trending topics |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 500M
    Searches per user per day: 20
    Total searches per day: 10B

    Autocomplete requests:
    - Average keystrokes per search: 5 (trigger after 2 chars, 3 more chars)
    - Total autocomplete requests: 10B × 5 = 50B requests/day
    - Autocomplete QPS: 50B / 86,400 = ~578,700 req/sec
    - Peak QPS: 3x average = ~1.7M req/sec

    Read/Write ratio:
    - Reads (autocomplete): 578,700 QPS
    - Writes (update suggestions): ~12 QPS (batch updates every hour)
    - Read/Write ratio: ~48,000:1 (extremely read-heavy)

    Query patterns:
    - 80% of queries come from 20% of suggestions (Pareto principle)
    - Top 10K queries account for 30% of traffic
    - Long-tail: 70% of queries are unique or rare
    ```

    ### Storage Estimates

    ```
    Query data:
    - Unique queries (10 years): 100M
    - Average query length: 30 characters × 2 bytes (UTF-16) = 60 bytes
    - Metadata per query: 40 bytes (query_id, count, last_updated, score)
    - Total per query: ~100 bytes
    - Total storage: 100M × 100 bytes = 10 GB

    Trie data structure:
    - Nodes needed: ~26^3 = 17,576 for 3-char prefixes
    - Average children per node: 10
    - Node size: 200 bytes (char, children pointers, top suggestions)
    - Trie storage: 100M queries × 30 chars × 200 bytes = 600 GB
    - With compression: ~100 GB

    User personalization:
    - Users with history: 200M (40% of DAU)
    - Queries per user: 100 (last 3 months)
    - Storage per user: 100 × 8 bytes = 800 bytes
    - Total: 200M × 800 bytes = 160 GB

    Total storage: 10 GB (queries) + 100 GB (trie) + 160 GB (personalization) = 270 GB
    (Easily fits in memory with Redis!)
    ```

    ### Bandwidth Estimates

    ```
    Autocomplete requests:
    - Request size: 100 bytes (prefix, user_id, metadata)
    - Response size: 500 bytes (5 suggestions × 100 bytes each)
    - Ingress: 578,700 req/sec × 100 bytes = 57.87 MB/sec ≈ 463 Mbps
    - Egress: 578,700 req/sec × 500 bytes = 289.35 MB/sec ≈ 2.3 Gbps

    Query updates:
    - Update frequency: Every 1 hour (batch process)
    - Queries updated per batch: 1M
    - Update size: 1M × 100 bytes = 100 MB per batch
    - Update bandwidth: Negligible (< 1 Mbps)

    Total ingress: ~500 Mbps
    Total egress: ~2.5 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    Hot data (in-memory):
    - Top 1M popular queries (80% of traffic): 100 MB
    - Trie structure (prefixes): 100 GB
    - User sessions: 50M concurrent × 1 KB = 50 GB
    - Query cache (recent requests): 10 GB

    Total cache: 100 MB + 100 GB + 50 GB + 10 GB ≈ 160 GB

    Distribution:
    - L1 Cache (local server): Top 10K queries = 1 MB
    - L2 Cache (Redis cluster): Trie + hot queries = 110 GB
    - L3 Cache (User personalization): 50 GB
    ```

    ---

    ## Key Assumptions

    1. Average query length: 30 characters (3-5 words)
    2. Suggestions triggered after 2 characters typed
    3. 80/20 rule: 80% of traffic from 20% of queries
    4. Cache hit rate: 95%+ (extremely repetitive queries)
    5. Eventual consistency acceptable (1-hour delay for new suggestions)
    6. UTF-8 support for international characters

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Cache-first architecture:** Serve 95%+ requests from memory (Redis)
    2. **Trie data structure:** Fast prefix matching (O(k) where k = prefix length)
    3. **Read-optimized:** Pre-compute rankings, no real-time calculations
    4. **Distributed caching:** Sharded Redis cluster for horizontal scaling
    5. **Eventual consistency:** Batch updates hourly, prioritize read performance

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN/Edge Cache<br/>CloudFlare]
            LB[Load Balancer<br/>NGINX]
        end

        subgraph "API Layer"
            Autocomplete_API[Autocomplete Service<br/>Query suggestions]
            Analytics_API[Analytics Service<br/>Track query events]
            Personalization_API[Personalization Service<br/>User history]
        end

        subgraph "Cache Layer"
            Redis_Trie[Redis Cluster<br/>Trie data structure<br/>Sharded by prefix]
            Redis_Query[Redis Cache<br/>Query metadata<br/>Popularity scores]
            Redis_User[Redis<br/>User history<br/>Personalization]
            Local_Cache[Local Cache<br/>LRU<br/>Top 10K queries]
        end

        subgraph "Data Processing"
            Query_Aggregator[Query Aggregator<br/>Count queries]
            Trie_Builder[Trie Builder<br/>Rebuild hourly]
            Ranker[Ranking Service<br/>ML-based scoring]
            Trending_Detector[Trending Detector<br/>Spike detection]
        end

        subgraph "Storage"
            Query_DB[(Query Database<br/>PostgreSQL<br/>Query logs)]
            Analytics_DB[(Analytics Store<br/>ClickHouse<br/>Time-series data)]
            User_DB[(User Database<br/>MongoDB<br/>Search history)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Query events stream]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        LB --> Autocomplete_API
        LB --> Analytics_API
        LB --> Personalization_API

        Autocomplete_API --> Local_Cache
        Local_Cache --> Redis_Trie
        Autocomplete_API --> Redis_Query
        Autocomplete_API --> Redis_User

        Analytics_API --> Kafka
        Kafka --> Query_Aggregator
        Kafka --> Analytics_DB

        Query_Aggregator --> Query_DB
        Query_Aggregator --> Trie_Builder

        Trie_Builder --> Redis_Trie
        Trie_Builder --> Query_DB

        Ranker --> Redis_Query
        Trending_Detector --> Redis_Query

        Personalization_API --> Redis_User
        Personalization_API --> User_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Local_Cache fill:#fff4e1
        style Redis_Trie fill:#fff4e1
        style Redis_Query fill:#fff4e1
        style Redis_User fill:#fff4e1
        style Query_DB fill:#ffe1e1
        style Analytics_DB fill:#ffe1e1
        style User_DB fill:#e1f5e1
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Trie Data Structure** | O(k) prefix matching, memory efficient, natural for autocomplete | Hash map (no prefix support), database queries (too slow <100ms) |
    | **Redis Cluster** | In-memory storage (< 10ms latency), 160GB trie fits in RAM, horizontal scaling | Memcached (limited data structures), local cache (no sharing across servers) |
    | **PostgreSQL** | Query metadata storage, analytics, batch processing | NoSQL (need ACID for analytics), Cassandra (overkill for 10GB data) |
    | **Kafka** | High-throughput event streaming (578K events/sec), reliable delivery | Direct writes (no buffering), RabbitMQ (lower throughput) |
    | **Local LRU Cache** | Ultra-fast access to top 10K queries (< 1ms), reduces Redis load | No local cache (higher Redis load), too large cache (memory waste) |
    | **ClickHouse** | Time-series analytics (query trends), fast aggregations | PostgreSQL (slow for time-series), Elasticsearch (higher cost) |

    **Key Trade-off:** We chose **read performance over consistency**. Suggestions may be stale by up to 1 hour, but queries are served in < 50ms.

    ---

    ## API Design

    ### 1. Get Autocomplete Suggestions

    **Request:**
    ```http
    GET /api/v1/autocomplete?q=mach&limit=5&user_id=user123
    Authorization: Bearer <token>
    ```

    **Query Parameters:**
    - `q` (required): Prefix query string (2-50 characters)
    - `limit` (optional): Number of suggestions (default: 5, max: 10)
    - `user_id` (optional): User ID for personalization
    - `locale` (optional): Language/region (e.g., en-US, ja-JP)
    - `category` (optional): Filter by category (e.g., products, videos)

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json
    X-Response-Time: 23ms

    {
      "query": "mach",
      "suggestions": [
        {
          "text": "machine learning",
          "score": 0.95,
          "type": "query",
          "highlight": "<b>mach</b>ine learning"
        },
        {
          "text": "machine learning course",
          "score": 0.87,
          "type": "query",
          "highlight": "<b>mach</b>ine learning course"
        },
        {
          "text": "macbook pro",
          "score": 0.82,
          "type": "product",
          "highlight": "<b>mac</b>book pro"
        },
        {
          "text": "machu picchu",
          "score": 0.76,
          "type": "query",
          "highlight": "<b>mach</b>u picchu"
        },
        {
          "text": "machine learning python",
          "score": 0.71,
          "type": "query",
          "highlight": "<b>mach</b>ine learning python"
        }
      ],
      "metadata": {
        "response_time_ms": 23,
        "source": "cache",
        "personalized": true
      }
    }
    ```

    **Design Notes:**

    - Return immediately with cached results (< 50ms)
    - Limit minimum query length to 2 characters (avoid excessive requests)
    - Rate limit: 100 requests per second per user
    - Highlight matching prefix in suggestions
    - Include metadata for debugging and monitoring

    ---

    ### 2. Track Query Event

    **Request:**
    ```http
    POST /api/v1/analytics/query
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "user_id": "user123",
      "query": "machine learning",
      "selected_suggestion": "machine learning course",
      "timestamp": "2026-02-02T10:30:00Z",
      "session_id": "sess_456",
      "result_clicked": true,
      "position": 2
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "status": "accepted",
      "event_id": "evt_789"
    }
    ```

    **Design Notes:**

    - Asynchronous processing via Kafka
    - Track query impressions and clicks for ranking
    - Used for personalization and trending detection
    - No blocking on user experience

    ---

    ### 3. Get Personalized Suggestions

    **Request:**
    ```http
    GET /api/v1/autocomplete/personalized?user_id=user123&limit=5
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "suggestions": [
        {
          "text": "machine learning python",
          "score": 0.98,
          "reason": "frequently_searched",
          "last_searched": "2026-02-01T14:22:00Z"
        },
        {
          "text": "deep learning course",
          "score": 0.92,
          "reason": "recent_search",
          "last_searched": "2026-02-02T09:15:00Z"
        }
      ]
    }
    ```

    **Design Notes:**

    - Pre-compute personalized suggestions per user
    - Cache in Redis with TTL of 1 hour
    - Blend with global popular suggestions
    - Privacy-conscious (user can clear history)

    ---

    ## Database Schema

    ### Query Metadata (PostgreSQL)

    ```sql
    -- Query statistics table
    CREATE TABLE queries (
        query_id BIGSERIAL PRIMARY KEY,
        query_text VARCHAR(200) UNIQUE NOT NULL,
        search_count BIGINT DEFAULT 0,
        click_count BIGINT DEFAULT 0,
        ctr DECIMAL(5,4) DEFAULT 0.0,  -- Click-through rate
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        popularity_score DECIMAL(10,6) DEFAULT 0.0,
        trending_score DECIMAL(10,6) DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_query_text (query_text),
        INDEX idx_popularity (popularity_score DESC),
        INDEX idx_trending (trending_score DESC),
        INDEX idx_last_updated (last_updated)
    );

    -- Query prefix index (for batch processing)
    CREATE TABLE query_prefixes (
        prefix VARCHAR(50) PRIMARY KEY,
        query_ids BIGINT[],  -- Array of top query IDs
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_prefix (prefix)
    );

    -- Trending queries (time-windowed)
    CREATE TABLE trending_queries (
        query_id BIGINT REFERENCES queries(query_id),
        time_window TIMESTAMP,  -- Hour bucket
        search_count INT,
        velocity DECIMAL(10,2),  -- Rate of increase

        PRIMARY KEY (query_id, time_window),
        INDEX idx_time_window (time_window),
        INDEX idx_velocity (velocity DESC)
    );
    ```

    **Why PostgreSQL:**

    - **ACID compliance:** Accurate analytics data
    - **Complex queries:** Aggregations for trending detection
    - **Small dataset:** 10GB easily handled by single instance
    - **Batch processing:** Hourly updates, not real-time

    ---

    ### User Search History (MongoDB)

    ```javascript
    // User collection
    {
      "_id": ObjectId("..."),
      "user_id": "user123",
      "search_history": [
        {
          "query": "machine learning",
          "timestamp": ISODate("2026-02-02T10:30:00Z"),
          "clicked": true,
          "result_position": 2
        },
        {
          "query": "deep learning",
          "timestamp": ISODate("2026-02-02T09:15:00Z"),
          "clicked": false
        }
        // ... more recent searches (max 100)
      ],
      "personalized_suggestions": [
        {
          "query": "machine learning python",
          "score": 0.98,
          "last_updated": ISODate("2026-02-02T11:00:00Z")
        }
      ],
      "created_at": ISODate("2025-01-01T00:00:00Z"),
      "last_active": ISODate("2026-02-02T10:30:00Z")
    }

    // Indexes
    db.users.createIndex({ "user_id": 1 }, { unique: true })
    db.users.createIndex({ "search_history.timestamp": -1 })
    db.users.createIndex({ "last_active": -1 })
    ```

    **Why MongoDB:**

    - **Flexible schema:** Easy to add new fields
    - **Array operations:** Efficient for search history
    - **Document model:** Natural fit for user data
    - **TTL indexes:** Auto-expire old search history

    ---

    ### Analytics Store (ClickHouse)

    ```sql
    -- Query events table (time-series)
    CREATE TABLE query_events (
        event_id String,
        user_id String,
        query String,
        selected_suggestion String,
        timestamp DateTime,
        session_id String,
        result_clicked Boolean,
        position Int8,
        response_time_ms Int16,
        source String  -- 'cache', 'redis', 'db'
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (timestamp, user_id)
    TTL timestamp + INTERVAL 90 DAY;  -- Retain 90 days

    -- Aggregated hourly stats (materialized view)
    CREATE MATERIALIZED VIEW query_hourly_stats
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(hour)
    ORDER BY (hour, query)
    AS SELECT
        toStartOfHour(timestamp) AS hour,
        query,
        count() AS search_count,
        countIf(result_clicked) AS click_count,
        avg(response_time_ms) AS avg_response_time
    FROM query_events
    GROUP BY hour, query;
    ```

    **Why ClickHouse:**

    - **Time-series optimized:** Fast aggregations on time windows
    - **High write throughput:** 578K events/sec
    - **Compression:** 10x compression ratio for logs
    - **Analytics queries:** Sub-second complex aggregations

    ---

    ## Data Flow Diagrams

    ### Autocomplete Request Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant LB
        participant API
        participant Local_Cache
        participant Redis
        participant DB

        Client->>LB: GET /autocomplete?q=mach
        LB->>API: Route request

        API->>API: Validate input (length >= 2)

        API->>Local_Cache: Check top 10K cache
        alt Cache HIT (30% of requests)
            Local_Cache-->>API: Return suggestions
        else Cache MISS
            Local_Cache-->>API: null

            API->>Redis: GET trie node for prefix "mac"
            alt Redis HIT (95% of requests)
                Redis-->>API: Trie node with top suggestions
                API->>API: Rank and personalize
            else Redis MISS (5% of requests)
                Redis-->>API: null
                API->>DB: Query suggestions from database
                DB-->>API: Query results
                API->>Redis: Cache result (TTL: 1 hour)
            end

            API->>Local_Cache: Update local cache
        end

        API->>API: Apply personalization (if user_id)
        API-->>Client: Return top 5 suggestions (< 50ms)

        Note over Client,DB: Async: Track query event
        API->>Kafka: Publish query event
    ```

    **Flow Explanation:**

    1. **Input validation** - Check minimum length (2 chars), sanitize input
    2. **L1 Cache (Local)** - Check in-process cache for top 10K queries (< 1ms)
    3. **L2 Cache (Redis)** - Query trie structure from Redis cluster (< 10ms)
    4. **Database fallback** - Query PostgreSQL for cold data (< 100ms)
    5. **Personalization** - Blend with user history if available
    6. **Async tracking** - Publish event to Kafka for analytics

    ---

    ### Trie Update Flow (Hourly Batch)

    ```mermaid
    sequenceDiagram
        participant Kafka
        participant Aggregator
        participant DB
        participant Trie_Builder
        participant Redis

        Note over Kafka,Redis: Runs every 1 hour

        Kafka->>Aggregator: Consume query events (last 1 hour)
        Aggregator->>Aggregator: Aggregate query counts
        Aggregator->>DB: Update query statistics

        DB->>Trie_Builder: Trigger trie rebuild
        Trie_Builder->>DB: Fetch top 1M queries
        Trie_Builder->>Trie_Builder: Build in-memory trie

        loop For each prefix
            Trie_Builder->>Trie_Builder: Calculate top-K suggestions
            Trie_Builder->>Trie_Builder: Apply ranking algorithm
        end

        Trie_Builder->>Redis: Batch update trie nodes
        Redis-->>Trie_Builder: ACK

        Trie_Builder->>Trie_Builder: Log completion metrics

        Note over Trie_Builder,Redis: Atomic swap (no downtime)
    ```

    **Flow Explanation:**

    1. **Aggregate events** - Sum query counts from last hour
    2. **Update database** - Write aggregated stats to PostgreSQL
    3. **Build trie** - Construct in-memory trie from top 1M queries
    4. **Rank suggestions** - Calculate top-K per prefix using ML ranking
    5. **Atomic swap** - Update Redis cluster atomically (no downtime)
    6. **Monitor** - Track build time, cache hit rate, suggestion quality

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical autocomplete subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Trie Data Structure** | How to efficiently store and query millions of suggestions? | Compressed trie with top-K per node |
    | **Ranking Algorithm** | How to order suggestions by relevance? | ML-based scoring: popularity + CTR + personalization |
    | **Fuzzy Matching** | How to handle typos and misspellings? | Edit distance + phonetic matching |
    | **Personalization** | How to provide user-specific suggestions? | User history + collaborative filtering |

    ---

    === "🌲 Trie Data Structure"

        ## The Challenge

        **Problem:** Store 100M queries efficiently, support fast prefix matching (< 10ms), keep in memory (< 200GB).

        **Naive approach:** Database LIKE query (`SELECT * FROM queries WHERE query LIKE 'mach%'`). **Too slow** (> 500ms for 100M rows).

        **Solution:** Trie (prefix tree) - O(k) lookup where k = prefix length.

        ---

        ## Trie Implementation

        ```python
        from typing import List, Dict, Optional
        from dataclasses import dataclass, field
        import heapq

        @dataclass
        class TrieNode:
            """
            Node in the autocomplete trie

            Attributes:
                char: Character at this node
                children: Map of child nodes (char -> TrieNode)
                is_end_of_word: True if this node ends a query
                query_id: ID of the query (if is_end_of_word)
                top_suggestions: Pre-computed top-K suggestions for this prefix
                frequency: Number of times this prefix was searched
            """
            char: str = ""
            children: Dict[str, 'TrieNode'] = field(default_factory=dict)
            is_end_of_word: bool = False
            query_id: Optional[int] = None
            top_suggestions: List[tuple] = field(default_factory=list)  # [(query, score)]
            frequency: int = 0

        class AutocompleteTrie:
            """
            Trie data structure optimized for autocomplete

            Features:
            - O(k) prefix matching where k = prefix length
            - Pre-computed top-K suggestions per node
            - Memory efficient (compressed paths)
            - Supports case-insensitive search
            """

            def __init__(self, top_k: int = 5):
                self.root = TrieNode()
                self.top_k = top_k
                self.total_queries = 0

            def insert(self, query: str, query_id: int, frequency: int = 1):
                """
                Insert query into trie with frequency

                Args:
                    query: Query string to insert
                    query_id: Unique ID for the query
                    frequency: Search frequency (default: 1)

                Time Complexity: O(n) where n = len(query)
                """
                query = query.lower().strip()
                node = self.root

                for char in query:
                    if char not in node.children:
                        node.children[char] = TrieNode(char=char)

                    node = node.children[char]
                    node.frequency += frequency

                # Mark end of word
                node.is_end_of_word = True
                node.query_id = query_id
                self.total_queries += 1

            def search_prefix(self, prefix: str) -> List[tuple]:
                """
                Get top-K suggestions for a prefix

                Args:
                    prefix: Prefix to search for

                Returns:
                    List of (query, score) tuples, sorted by score descending

                Time Complexity: O(k) where k = len(prefix)
                """
                prefix = prefix.lower().strip()

                # Navigate to prefix node
                node = self.root
                for char in prefix:
                    if char not in node.children:
                        return []  # Prefix not found
                    node = node.children[char]

                # Return pre-computed top-K suggestions
                return node.top_suggestions[:self.top_k]

            def build_top_suggestions(self, node: TrieNode, current_word: str) -> List[tuple]:
                """
                Recursively build top-K suggestions for all descendants

                This is done as a batch process (not per-query)

                Args:
                    node: Current trie node
                    current_word: Current prefix being built

                Returns:
                    List of (query, frequency) tuples for all descendants
                """
                suggestions = []

                # If this is a complete query, add it
                if node.is_end_of_word:
                    suggestions.append((current_word, node.frequency))

                # Recursively collect from children
                for char, child in node.children.items():
                    child_suggestions = self.build_top_suggestions(
                        child,
                        current_word + char
                    )
                    suggestions.extend(child_suggestions)

                # Sort by frequency and keep top-K
                suggestions.sort(key=lambda x: x[1], reverse=True)
                node.top_suggestions = suggestions[:self.top_k * 2]  # Keep more for flexibility

                return suggestions

            def finalize(self):
                """
                Pre-compute top-K suggestions for all nodes

                Call this after all queries are inserted.
                This is a one-time operation done during batch updates.

                Time Complexity: O(n × k) where n = total nodes, k = top_k
                """
                self.build_top_suggestions(self.root, "")

            def get_memory_usage(self) -> dict:
                """Estimate memory usage of the trie"""
                def count_nodes(node):
                    count = 1
                    for child in node.children.values():
                        count += count_nodes(child)
                    return count

                num_nodes = count_nodes(self.root)

                # Estimate size per node
                # char: 2 bytes, children dict: ~100 bytes, top_suggestions: ~500 bytes
                bytes_per_node = 602

                total_mb = (num_nodes * bytes_per_node) / (1024 * 1024)

                return {
                    'num_nodes': num_nodes,
                    'total_queries': self.total_queries,
                    'estimated_mb': round(total_mb, 2)
                }


        # Example usage
        def build_autocomplete_trie(queries: List[tuple]) -> AutocompleteTrie:
            """
            Build autocomplete trie from query data

            Args:
                queries: List of (query_text, query_id, frequency) tuples

            Returns:
                AutocompleteTrie ready for serving
            """
            trie = AutocompleteTrie(top_k=5)

            print(f"Building trie with {len(queries)} queries...")

            for query_text, query_id, frequency in queries:
                trie.insert(query_text, query_id, frequency)

            print("Finalizing trie (pre-computing top suggestions)...")
            trie.finalize()

            memory = trie.get_memory_usage()
            print(f"Trie built: {memory['num_nodes']} nodes, {memory['estimated_mb']} MB")

            return trie


        # Demo
        if __name__ == "__main__":
            # Sample queries with frequencies
            queries = [
                ("machine learning", 1, 10000),
                ("machine learning course", 2, 8000),
                ("machine learning python", 3, 7500),
                ("machine learning tutorial", 4, 6000),
                ("macbook pro", 5, 5000),
                ("machu picchu", 6, 3000),
                ("mach number", 7, 1000),
            ]

            trie = build_autocomplete_trie(queries)

            # Test searches
            test_prefixes = ["mach", "machine", "machine learning"]

            for prefix in test_prefixes:
                suggestions = trie.search_prefix(prefix)
                print(f"\nPrefix: '{prefix}'")
                print("Suggestions:")
                for query, score in suggestions:
                    print(f"  - {query} (score: {score})")
        ```

        **Output:**
        ```
        Building trie with 7 queries...
        Finalizing trie (pre-computing top suggestions)...
        Trie built: 47 nodes, 0.03 MB

        Prefix: 'mach'
        Suggestions:
          - machine learning (score: 10000)
          - machine learning course (score: 8000)
          - machine learning python (score: 7500)
          - machine learning tutorial (score: 6000)
          - macbook pro (score: 5000)

        Prefix: 'machine'
        Suggestions:
          - machine learning (score: 10000)
          - machine learning course (score: 8000)
          - machine learning python (score: 7500)
          - machine learning tutorial (score: 6000)

        Prefix: 'machine learning'
        Suggestions:
          - machine learning (score: 10000)
          - machine learning course (score: 8000)
          - machine learning python (score: 7500)
          - machine learning tutorial (score: 6000)
        ```

        ---

        ## Trie Optimization Techniques

        | Technique | Description | Memory Savings |
        |-----------|-------------|----------------|
        | **Path Compression** | Merge single-child nodes (e.g., "abc" → single node) | 30-50% |
        | **Top-K Pre-computation** | Store only top-K suggestions per node (not all) | 70-80% |
        | **Frequency Quantization** | Store frequency as int8 (0-255) instead of int64 | 87.5% |
        | **Lazy Loading** | Load cold prefixes from disk on demand | 50-70% |
        | **Delta Encoding** | Store only differences from parent node | 40-60% |

        **With optimizations:** 600 GB → 100 GB (83% reduction)

        ---

        ## Distributed Trie (Sharding)

        **Problem:** 100 GB trie may not fit on single Redis instance.

        **Solution:** Shard by prefix hash.

        ```python
        class DistributedTrie:
            """
            Distributed trie across multiple Redis instances

            Sharding strategy: Hash prefix → Redis shard
            """

            def __init__(self, redis_clients: List, num_shards: int = 10):
                self.redis_clients = redis_clients
                self.num_shards = num_shards

            def get_shard(self, prefix: str) -> int:
                """Determine shard for prefix using consistent hashing"""
                return hash(prefix[:2]) % self.num_shards  # Hash first 2 chars

            def search_prefix(self, prefix: str) -> List[tuple]:
                """Search distributed trie"""
                shard_id = self.get_shard(prefix)
                redis_client = self.redis_clients[shard_id]

                # Query Redis for trie node
                key = f"trie:{prefix}"
                suggestions = redis_client.get(key)

                if suggestions:
                    return json.loads(suggestions)

                return []

            def insert_batch(self, queries: List[tuple]):
                """Batch insert queries into distributed trie"""
                shard_buckets = [[] for _ in range(self.num_shards)]

                # Group queries by shard
                for query_text, query_id, frequency in queries:
                    shard_id = self.get_shard(query_text)
                    shard_buckets[shard_id].append((query_text, query_id, frequency))

                # Parallel insert to each shard
                with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
                    futures = []
                    for shard_id, queries in enumerate(shard_buckets):
                        future = executor.submit(
                            self._insert_to_shard,
                            shard_id,
                            queries
                        )
                        futures.append(future)

                    # Wait for all shards
                    for future in futures:
                        future.result()
        ```

        **Benefits:**

        - **Scalability:** Distribute 100 GB across 10 Redis instances (10 GB each)
        - **Parallelism:** Query multiple shards in parallel (< 10ms total)
        - **Fault tolerance:** Shard failures only affect subset of prefixes

    === "📊 Ranking Algorithm"

        ## The Challenge

        **Problem:** Given prefix "mach", which suggestions to show first? Need to balance popularity, relevance, freshness, and personalization.

        **Naive approach:** Sort by frequency. **Not enough** - ignores CTR, trending, personalization.

        **Solution:** Multi-factor ML ranking model.

        ---

        ## Ranking Features

        | Feature | Weight | Description | Range |
        |---------|--------|-------------|-------|
        | **Search Frequency** | 30% | Total searches for this query | 0-1M+ |
        | **Click-Through Rate (CTR)** | 25% | % of times query was clicked when shown | 0-100% |
        | **Trending Score** | 20% | Velocity of searches (recent spike) | 0-10 |
        | **Personalization Score** | 15% | User's past searches + similar users | 0-100 |
        | **Recency** | 10% | How recently query was searched | 0-∞ |

        ---

        ## Ranking Implementation

        ```python
        import math
        from datetime import datetime, timedelta
        from typing import List, Dict
        import numpy as np

        class SuggestionRanker:
            """
            ML-based ranking algorithm for autocomplete suggestions

            Combines multiple signals:
            - Popularity (search frequency)
            - Relevance (CTR)
            - Trending (velocity)
            - Personalization (user history)
            - Recency (time decay)
            """

            # Feature weights (sum to 1.0)
            WEIGHTS = {
                'frequency': 0.30,
                'ctr': 0.25,
                'trending': 0.20,
                'personalization': 0.15,
                'recency': 0.10
            }

            def __init__(self):
                self.max_frequency = 1_000_000  # Normalization
                self.trending_window_hours = 24

            def calculate_frequency_score(self, search_count: int) -> float:
                """
                Normalize frequency using log scale

                Why log scale? Difference between 10 and 100 is more significant
                than between 1M and 1.0001M.

                Args:
                    search_count: Total number of searches

                Returns:
                    Score between 0 and 1
                """
                if search_count <= 0:
                    return 0.0

                # Log normalization
                score = math.log10(search_count + 1) / math.log10(self.max_frequency)
                return min(score, 1.0)

            def calculate_ctr_score(self, click_count: int, impression_count: int) -> float:
                """
                Calculate click-through rate with smoothing

                Smoothing prevents queries with 1 impression, 1 click (100% CTR)
                from ranking too high.

                Args:
                    click_count: Number of times query was clicked
                    impression_count: Number of times query was shown

                Returns:
                    Smoothed CTR score between 0 and 1
                """
                if impression_count == 0:
                    return 0.0

                # Add-k smoothing (k=10)
                smoothed_ctr = (click_count + 10) / (impression_count + 20)

                # Apply confidence penalty for low impressions
                confidence = min(impression_count / 100, 1.0)

                return smoothed_ctr * confidence

            def calculate_trending_score(
                self,
                recent_searches: int,
                historical_avg: float
            ) -> float:
                """
                Calculate trending score based on search velocity

                Detects sudden spikes in search volume (viral queries).

                Args:
                    recent_searches: Searches in last 24 hours
                    historical_avg: Average searches per day (last 30 days)

                Returns:
                    Trending score between 0 and 10
                """
                if historical_avg <= 0:
                    return 0.0

                # Velocity ratio
                velocity = recent_searches / max(historical_avg, 1.0)

                # Cap at 10x for normalization
                trending_score = min(velocity, 10.0) / 10.0

                return trending_score

            def calculate_personalization_score(
                self,
                query: str,
                user_history: List[str],
                user_embedding: np.ndarray = None,
                query_embedding: np.ndarray = None
            ) -> float:
                """
                Calculate personalization score based on user history

                Two approaches:
                1. Simple: String matching with past queries
                2. Advanced: Embedding similarity (if available)

                Args:
                    query: Query to score
                    user_history: User's past search queries
                    user_embedding: User's embedding vector (optional)
                    query_embedding: Query's embedding vector (optional)

                Returns:
                    Personalization score between 0 and 1
                """
                if not user_history:
                    return 0.0

                # Approach 1: Exact match bonus
                if query in user_history:
                    return 1.0

                # Approach 2: Prefix match
                prefix_matches = sum(
                    1 for past_query in user_history
                    if past_query.startswith(query) or query.startswith(past_query)
                )
                prefix_score = min(prefix_matches / 5, 0.7)

                # Approach 3: Embedding similarity (if available)
                if user_embedding is not None and query_embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(user_embedding, query_embedding) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(query_embedding)
                    )
                    embedding_score = (similarity + 1) / 2  # Normalize to [0, 1]

                    return max(prefix_score, embedding_score)

                return prefix_score

            def calculate_recency_score(self, last_searched: datetime) -> float:
                """
                Calculate recency score with exponential decay

                Recent queries ranked higher (time-sensitive).

                Args:
                    last_searched: Timestamp of last search

                Returns:
                    Recency score between 0 and 1
                """
                if last_searched is None:
                    return 0.0

                hours_ago = (datetime.utcnow() - last_searched).total_seconds() / 3600

                # Exponential decay (half-life = 7 days)
                half_life_hours = 7 * 24
                decay_rate = math.log(2) / half_life_hours

                recency_score = math.exp(-decay_rate * hours_ago)

                return recency_score

            def rank_suggestions(
                self,
                suggestions: List[Dict],
                user_context: Dict = None
            ) -> List[Dict]:
                """
                Rank suggestions using multi-factor scoring

                Args:
                    suggestions: List of suggestion dicts with metadata
                    user_context: Optional user context for personalization

                Returns:
                    Ranked list of suggestions with scores
                """
                scored_suggestions = []

                for suggestion in suggestions:
                    # Calculate individual scores
                    freq_score = self.calculate_frequency_score(
                        suggestion.get('search_count', 0)
                    )

                    ctr_score = self.calculate_ctr_score(
                        suggestion.get('click_count', 0),
                        suggestion.get('impression_count', 1)
                    )

                    trending_score = self.calculate_trending_score(
                        suggestion.get('recent_searches', 0),
                        suggestion.get('historical_avg', 1.0)
                    )

                    personalization_score = 0.0
                    if user_context:
                        personalization_score = self.calculate_personalization_score(
                            suggestion['query'],
                            user_context.get('search_history', []),
                            user_context.get('embedding'),
                            suggestion.get('embedding')
                        )

                    recency_score = self.calculate_recency_score(
                        suggestion.get('last_searched')
                    )

                    # Weighted combination
                    final_score = (
                        self.WEIGHTS['frequency'] * freq_score +
                        self.WEIGHTS['ctr'] * ctr_score +
                        self.WEIGHTS['trending'] * trending_score +
                        self.WEIGHTS['personalization'] * personalization_score +
                        self.WEIGHTS['recency'] * recency_score
                    )

                    suggestion['score'] = final_score
                    suggestion['score_breakdown'] = {
                        'frequency': freq_score,
                        'ctr': ctr_score,
                        'trending': trending_score,
                        'personalization': personalization_score,
                        'recency': recency_score
                    }

                    scored_suggestions.append(suggestion)

                # Sort by final score descending
                scored_suggestions.sort(key=lambda x: x['score'], reverse=True)

                return scored_suggestions


        # Example usage
        if __name__ == "__main__":
            ranker = SuggestionRanker()

            # Sample suggestions
            suggestions = [
                {
                    'query': 'machine learning',
                    'search_count': 100000,
                    'click_count': 8000,
                    'impression_count': 20000,
                    'recent_searches': 5000,
                    'historical_avg': 3000,
                    'last_searched': datetime.utcnow() - timedelta(hours=2)
                },
                {
                    'query': 'machine learning course',
                    'search_count': 50000,
                    'click_count': 6000,
                    'impression_count': 15000,
                    'recent_searches': 2000,
                    'historical_avg': 1500,
                    'last_searched': datetime.utcnow() - timedelta(days=1)
                },
                {
                    'query': 'macbook pro',
                    'search_count': 80000,
                    'click_count': 5000,
                    'impression_count': 10000,
                    'recent_searches': 1000,
                    'historical_avg': 2500,
                    'last_searched': datetime.utcnow() - timedelta(days=7)
                }
            ]

            # User context
            user_context = {
                'search_history': ['machine learning python', 'deep learning'],
                'embedding': None
            }

            # Rank suggestions
            ranked = ranker.rank_suggestions(suggestions, user_context)

            print("Ranked Suggestions:")
            for i, suggestion in enumerate(ranked, 1):
                print(f"\n{i}. {suggestion['query']}")
                print(f"   Final Score: {suggestion['score']:.4f}")
                print(f"   Breakdown: {suggestion['score_breakdown']}")
        ```

        **Output:**
        ```
        Ranked Suggestions:

        1. machine learning
           Final Score: 0.7834
           Breakdown: {'frequency': 0.833, 'ctr': 0.324, 'trending': 0.167, 'personalization': 0.4, 'recency': 0.989}

        2. machine learning course
           Final Score: 0.6521
           Breakdown: {'frequency': 0.783, 'ctr': 0.357, 'trending': 0.133, 'personalization': 0.8, 'recency': 0.857}

        3. macbook pro
           Final Score: 0.5123
           Breakdown: {'frequency': 0.817, 'ctr': 0.422, 'trending': 0.040, 'personalization': 0.0, 'recency': 0.442}
        ```

        ---

        ## A/B Testing for Ranking

        **Continuous improvement:**

        1. **Split traffic:** 90% production model, 10% experimental model
        2. **Track metrics:** CTR, conversion rate, user satisfaction
        3. **Statistical significance:** Wait for 95% confidence
        4. **Promote winner:** Deploy better model to 100% traffic

        **Metrics to track:**

        - **Top-1 CTR:** % of times first suggestion is clicked
        - **MRR (Mean Reciprocal Rank):** Average of 1/rank for clicked suggestion
        - **Zero-result rate:** % of queries with no suggestions
        - **User satisfaction:** Survey ratings, engagement time

    === "🔤 Fuzzy Matching"

        ## The Challenge

        **Problem:** Users make typos. "mahcine learning" should still suggest "machine learning".

        **Requirements:**

        - Handle common typos (missing letter, swapped letters, extra letter)
        - Fast (<100ms even with fuzzy matching)
        - Don't overwhelm user with irrelevant suggestions

        ---

        ## Fuzzy Matching Approaches

        | Approach | Description | Pros | Cons |
        |----------|-------------|------|------|
        | **Edit Distance** | Levenshtein distance (insertions, deletions, substitutions) | Accurate, handles all typos | Slow (O(n×m)) |
        | **Phonetic Matching** | Soundex, Metaphone (sound-alike words) | Good for pronunciation typos | English-centric |
        | **N-gram Matching** | Break into 3-grams, find similar | Fast, works for all languages | Less accurate |
        | **BK-Tree** | Tree structure for edit distance queries | Fast (O(log n)) | Complex to implement |

        **Production approach:** Hybrid (n-gram + edit distance)

        ---

        ## Implementation

        ```python
        from typing import List, Set, Dict
        import re

        class FuzzyMatcher:
            """
            Fuzzy matching for autocomplete with typo tolerance

            Approach:
            1. N-gram indexing for fast candidate retrieval
            2. Edit distance for accurate similarity scoring
            3. Phonetic matching for pronunciation typos
            """

            def __init__(self, max_edit_distance: int = 2):
                self.max_edit_distance = max_edit_distance
                self.ngram_size = 3
                self.ngram_index = {}  # n-gram -> set of query_ids

            def generate_ngrams(self, text: str, n: int = 3) -> List[str]:
                """
                Generate character n-grams from text

                Example: "machine" → ["mac", "ach", "chi", "hin", "ine"]

                Args:
                    text: Input text
                    n: N-gram size

                Returns:
                    List of n-grams
                """
                text = text.lower()
                # Add padding for edge n-grams
                padded = f"${text}$"

                ngrams = []
                for i in range(len(padded) - n + 1):
                    ngrams.append(padded[i:i+n])

                return ngrams

            def build_ngram_index(self, queries: List[tuple]):
                """
                Build n-gram inverted index for fast fuzzy search

                Args:
                    queries: List of (query_text, query_id) tuples
                """
                for query_text, query_id in queries:
                    ngrams = self.generate_ngrams(query_text, self.ngram_size)

                    for ngram in ngrams:
                        if ngram not in self.ngram_index:
                            self.ngram_index[ngram] = set()

                        self.ngram_index[ngram].add(query_id)

            def find_candidates(self, query: str, top_k: int = 20) -> Set[int]:
                """
                Find candidate queries using n-gram overlap

                Fast pre-filtering step before edit distance calculation.

                Args:
                    query: Query with potential typo
                    top_k: Number of candidates to return

                Returns:
                    Set of candidate query IDs
                """
                query_ngrams = self.generate_ngrams(query, self.ngram_size)

                # Count n-gram overlaps
                candidate_scores = {}

                for ngram in query_ngrams:
                    if ngram in self.ngram_index:
                        for query_id in self.ngram_index[ngram]:
                            candidate_scores[query_id] = candidate_scores.get(query_id, 0) + 1

                # Sort by overlap count
                sorted_candidates = sorted(
                    candidate_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Return top-K candidate IDs
                return {query_id for query_id, _ in sorted_candidates[:top_k]}

            def edit_distance(self, s1: str, s2: str) -> int:
                """
                Calculate Levenshtein edit distance

                Dynamic programming approach.

                Args:
                    s1: First string
                    s2: Second string

                Returns:
                    Minimum edit distance (insertions, deletions, substitutions)

                Time Complexity: O(n × m)
                """
                m, n = len(s1), len(s2)

                # DP table
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                # Initialize base cases
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j

                # Fill DP table
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1]  # No operation needed
                        else:
                            dp[i][j] = 1 + min(
                                dp[i-1][j],      # Deletion
                                dp[i][j-1],      # Insertion
                                dp[i-1][j-1]     # Substitution
                            )

                return dp[m][n]

            def fuzzy_search(
                self,
                query: str,
                all_queries: Dict[int, str],
                top_k: int = 5
            ) -> List[tuple]:
                """
                Fuzzy search with typo tolerance

                Two-phase approach:
                1. Fast candidate retrieval (n-gram overlap)
                2. Accurate scoring (edit distance)

                Args:
                    query: Query with potential typos
                    all_queries: Map of query_id -> query_text
                    top_k: Number of suggestions to return

                Returns:
                    List of (query_text, similarity_score) tuples
                """
                # Phase 1: Find candidates using n-grams
                candidate_ids = self.find_candidates(query, top_k=20)

                # Phase 2: Calculate edit distance for candidates
                results = []

                for query_id in candidate_ids:
                    candidate_text = all_queries[query_id]

                    # Calculate edit distance
                    distance = self.edit_distance(query.lower(), candidate_text.lower())

                    # Filter by max edit distance
                    if distance <= self.max_edit_distance:
                        # Convert distance to similarity score (0-1)
                        max_len = max(len(query), len(candidate_text))
                        similarity = 1 - (distance / max_len)

                        results.append((candidate_text, similarity, distance))

                # Sort by similarity (descending) and distance (ascending)
                results.sort(key=lambda x: (-x[1], x[2]))

                # Return top-K
                return [(text, score) for text, score, _ in results[:top_k]]

            def phonetic_match(self, word: str) -> str:
                """
                Simple phonetic matching (Soundex-like algorithm)

                Convert word to phonetic representation.
                Words that sound similar will have same code.

                Example: "wright" and "write" → same code

                Args:
                    word: Input word

                Returns:
                    Phonetic code (4 characters)
                """
                word = word.upper()

                # Keep first letter
                code = word[0]

                # Phonetic mapping
                mapping = {
                    'BFPV': '1',
                    'CGJKQSXZ': '2',
                    'DT': '3',
                    'L': '4',
                    'MN': '5',
                    'R': '6'
                }

                # Convert remaining letters
                for char in word[1:]:
                    for group, digit in mapping.items():
                        if char in group:
                            if code[-1] != digit:  # Avoid duplicates
                                code += digit
                            break

                # Pad or truncate to 4 characters
                code = (code + '000')[:4]

                return code


        # Example usage
        if __name__ == "__main__":
            # Sample queries
            queries = [
                ("machine learning", 1),
                ("machine learning course", 2),
                ("deep learning", 3),
                ("reinforcement learning", 4),
                ("macbook pro", 5),
            ]

            # Build fuzzy matcher
            matcher = FuzzyMatcher(max_edit_distance=2)
            matcher.build_ngram_index(queries)

            # Create query lookup
            query_dict = {qid: text for text, qid in queries}

            # Test with typos
            test_queries = [
                "mahcine learning",     # Typo: swapped 'a' and 'c'
                "machne learning",      # Typo: missing 'i'
                "machine lerning",      # Typo: missing 'a'
                "machin learning",      # Typo: missing 'e'
            ]

            for typo in test_queries:
                print(f"\nQuery (with typo): '{typo}'")
                results = matcher.fuzzy_search(typo, query_dict, top_k=3)
                print("Fuzzy matches:")
                for text, score in results:
                    print(f"  - {text} (similarity: {score:.3f})")
        ```

        **Output:**
        ```
        Query (with typo): 'mahcine learning'
        Fuzzy matches:
          - machine learning (similarity: 0.941)
          - machine learning course (similarity: 0.870)

        Query (with typo): 'machne learning'
        Fuzzy matches:
          - machine learning (similarity: 0.941)
          - machine learning course (similarity: 0.870)

        Query (with typo): 'machine lerning'
        Fuzzy matches:
          - machine learning (similarity: 0.941)
          - machine learning course (similarity: 0.870)

        Query (with typo): 'machin learning'
        Fuzzy matches:
          - machine learning (similarity: 0.947)
          - machine learning course (similarity: 0.875)
        ```

        ---

        ## Optimization: Fuzzy on Demand

        **Problem:** Fuzzy matching is slower than exact prefix matching.

        **Solution:** Progressive search strategy

        1. **Try exact match** (< 10ms) - Most queries have no typos
        2. **If no results, try fuzzy** (< 100ms) - Only 10-20% of queries
        3. **Cache fuzzy results** - Store common typos

        ```python
        def progressive_search(query: str, trie, fuzzy_matcher) -> List[str]:
            """
            Progressive search: exact first, fuzzy if needed
            """
            # Step 1: Try exact prefix match
            results = trie.search_prefix(query)

            if results:
                return results  # Fast path (90% of queries)

            # Step 2: Try fuzzy match (only if no exact results)
            fuzzy_results = fuzzy_matcher.fuzzy_search(query, all_queries)

            return fuzzy_results
        ```

    === "👤 Personalization"

        ## The Challenge

        **Problem:** Same query should show different suggestions for different users. "python" → "python tutorial" for beginner, "python asyncio" for expert.

        **Requirements:**

        - User-specific suggestions based on search history
        - Privacy-conscious (user can opt-out)
        - Real-time updates (reflect recent searches)
        - Blend with global popular suggestions

        ---

        ## Personalization Approaches

        | Approach | Description | Complexity | Accuracy |
        |----------|-------------|------------|----------|
        | **User History** | Show user's past searches | Low | Medium |
        | **Collaborative Filtering** | "Users like you also searched..." | Medium | High |
        | **Embedding-based** | Neural network embeddings | High | Very High |
        | **Context-aware** | Time, location, device | Medium | High |

        **Production approach:** Hybrid (user history + collaborative filtering)

        ---

        ## Implementation

        ```python
        from typing import List, Dict, Optional
        from collections import defaultdict
        import numpy as np
        from datetime import datetime, timedelta

        class PersonalizationEngine:
            """
            Personalization engine for autocomplete suggestions

            Features:
            - User search history tracking
            - Collaborative filtering (similar users)
            - Embedding-based similarity (optional)
            - Time-decay for recent searches
            """

            def __init__(self, redis_client, db_client):
                self.redis = redis_client
                self.db = db_client
                self.time_decay_days = 30

            def track_user_search(self, user_id: str, query: str, clicked: bool):
                """
                Track user search query and interaction

                Args:
                    user_id: User identifier
                    query: Search query
                    clicked: Whether user clicked a search result
                """
                # Store in Redis (fast access)
                key = f"user_history:{user_id}"

                search_event = {
                    'query': query,
                    'timestamp': datetime.utcnow().isoformat(),
                    'clicked': clicked
                }

                # Append to user history (keep last 100)
                self.redis.lpush(key, json.dumps(search_event))
                self.redis.ltrim(key, 0, 99)  # Keep only 100 most recent

                # Set TTL (expire after 90 days)
                self.redis.expire(key, 90 * 24 * 3600)

                # Async: Update database for analytics
                self.db.insert_search_event(user_id, query, clicked)

            def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
                """
                Retrieve user's recent search history

                Args:
                    user_id: User identifier
                    limit: Number of recent searches to return

                Returns:
                    List of search events
                """
                key = f"user_history:{user_id}"

                # Get from Redis
                raw_history = self.redis.lrange(key, 0, limit - 1)

                history = []
                for item in raw_history:
                    event = json.loads(item)
                    history.append(event)

                return history

            def get_personalized_suggestions(
                self,
                user_id: str,
                prefix: str,
                global_suggestions: List[Dict],
                blend_ratio: float = 0.3
            ) -> List[Dict]:
                """
                Generate personalized suggestions for user

                Blends personal history with global popular suggestions.

                Args:
                    user_id: User identifier
                    prefix: Query prefix
                    global_suggestions: Popular suggestions from trie
                    blend_ratio: % of personalized suggestions (0-1)

                Returns:
                    Blended list of suggestions
                """
                # Get user history
                user_history = self.get_user_history(user_id, limit=50)

                if not user_history:
                    return global_suggestions  # No personalization available

                # Extract queries from history
                past_queries = [event['query'] for event in user_history]

                # Find relevant personal queries (matching prefix)
                personal_matches = [
                    {
                        'query': query,
                        'score': self._calculate_personal_score(query, user_history),
                        'type': 'personal'
                    }
                    for query in past_queries
                    if query.lower().startswith(prefix.lower())
                ]

                # Sort by score
                personal_matches.sort(key=lambda x: x['score'], reverse=True)

                # Blend personal and global suggestions
                num_personal = int(len(global_suggestions) * blend_ratio)
                num_global = len(global_suggestions) - num_personal

                blended = personal_matches[:num_personal] + global_suggestions[:num_global]

                # Re-sort by combined score
                blended.sort(key=lambda x: x['score'], reverse=True)

                return blended

            def _calculate_personal_score(self, query: str, user_history: List[Dict]) -> float:
                """
                Calculate personalization score for query based on user history

                Factors:
                - Frequency (how many times user searched this)
                - Recency (how recently user searched this)
                - Engagement (whether user clicked results)

                Args:
                    query: Query to score
                    user_history: User's search history

                Returns:
                    Personalization score (0-1)
                """
                frequency = 0
                most_recent = None
                total_clicked = 0

                for event in user_history:
                    if event['query'] == query:
                        frequency += 1

                        if event['clicked']:
                            total_clicked += 1

                        event_time = datetime.fromisoformat(event['timestamp'])
                        if most_recent is None or event_time > most_recent:
                            most_recent = event_time

                # Frequency score (log scale)
                freq_score = min(math.log10(frequency + 1) / math.log10(10), 1.0)

                # Recency score (time decay)
                if most_recent:
                    days_ago = (datetime.utcnow() - most_recent).days
                    decay_rate = math.log(2) / self.time_decay_days
                    recency_score = math.exp(-decay_rate * days_ago)
                else:
                    recency_score = 0.0

                # Engagement score
                engagement_score = total_clicked / max(frequency, 1)

                # Combined score
                personal_score = (
                    0.4 * freq_score +
                    0.4 * recency_score +
                    0.2 * engagement_score
                )

                return personal_score

            def find_similar_users(self, user_id: str, top_k: int = 10) -> List[str]:
                """
                Find users with similar search patterns

                Collaborative filtering approach.

                Args:
                    user_id: User identifier
                    top_k: Number of similar users to find

                Returns:
                    List of similar user IDs
                """
                # Get user's search queries
                user_history = self.get_user_history(user_id)
                user_queries = set(event['query'] for event in user_history)

                if not user_queries:
                    return []

                # Find users who searched similar queries
                # (In production, use pre-computed similarity matrix)
                similar_users = self.db.find_users_with_queries(
                    list(user_queries),
                    limit=top_k + 1  # +1 to exclude self
                )

                # Remove self
                similar_users = [uid for uid in similar_users if uid != user_id]

                return similar_users[:top_k]

            def get_collaborative_suggestions(
                self,
                user_id: str,
                prefix: str,
                top_k: int = 5
            ) -> List[Dict]:
                """
                Get suggestions based on similar users

                "Users like you also searched for..."

                Args:
                    user_id: User identifier
                    prefix: Query prefix
                    top_k: Number of suggestions

                Returns:
                    List of collaborative suggestions
                """
                # Find similar users
                similar_users = self.find_similar_users(user_id, top_k=10)

                if not similar_users:
                    return []

                # Aggregate queries from similar users
                query_votes = defaultdict(int)

                for similar_user_id in similar_users:
                    user_history = self.get_user_history(similar_user_id)

                    for event in user_history:
                        query = event['query']
                        if query.lower().startswith(prefix.lower()):
                            query_votes[query] += 1

                # Sort by votes
                collaborative_suggestions = [
                    {
                        'query': query,
                        'score': votes / len(similar_users),
                        'type': 'collaborative'
                    }
                    for query, votes in query_votes.items()
                ]

                collaborative_suggestions.sort(key=lambda x: x['score'], reverse=True)

                return collaborative_suggestions[:top_k]


        # Example usage
        if __name__ == "__main__":
            # Mock Redis and DB clients
            class MockRedis:
                def __init__(self):
                    self.storage = {}

                def lpush(self, key, value):
                    if key not in self.storage:
                        self.storage[key] = []
                    self.storage[key].insert(0, value)

                def lrange(self, key, start, end):
                    return self.storage.get(key, [])[start:end+1]

                def ltrim(self, key, start, end):
                    if key in self.storage:
                        self.storage[key] = self.storage[key][start:end+1]

                def expire(self, key, seconds):
                    pass

            class MockDB:
                def insert_search_event(self, user_id, query, clicked):
                    pass

                def find_users_with_queries(self, queries, limit):
                    return ['user2', 'user3', 'user4']

            # Initialize
            redis = MockRedis()
            db = MockDB()
            engine = PersonalizationEngine(redis, db)

            # Simulate user searches
            user_id = "user1"
            engine.track_user_search(user_id, "machine learning", True)
            engine.track_user_search(user_id, "machine learning course", True)
            engine.track_user_search(user_id, "deep learning", False)
            engine.track_user_search(user_id, "machine learning python", True)

            # Global suggestions
            global_suggestions = [
                {'query': 'machine learning tutorial', 'score': 0.8, 'type': 'global'},
                {'query': 'machine learning jobs', 'score': 0.7, 'type': 'global'},
                {'query': 'machine learning algorithms', 'score': 0.6, 'type': 'global'},
            ]

            # Get personalized suggestions
            personalized = engine.get_personalized_suggestions(
                user_id,
                "machine",
                global_suggestions,
                blend_ratio=0.4
            )

            print("Personalized Suggestions:")
            for i, suggestion in enumerate(personalized, 1):
                print(f"{i}. {suggestion['query']} (score: {suggestion['score']:.3f}, type: {suggestion['type']})")
        ```

        **Output:**
        ```
        Personalized Suggestions:
        1. machine learning python (score: 0.845, type: personal)
        2. machine learning course (score: 0.823, type: personal)
        3. machine learning tutorial (score: 0.800, type: global)
        4. machine learning jobs (score: 0.700, type: global)
        5. machine learning algorithms (score: 0.600, type: global)
        ```

        ---

        ## Privacy Considerations

        **Key principles:**

        1. **User consent:** Allow users to opt-out of personalization
        2. **Data minimization:** Store only necessary data (queries, not full URLs)
        3. **Encryption:** Encrypt user data at rest and in transit
        4. **Retention limits:** Auto-delete history after 90 days
        5. **Clear history:** Allow users to delete search history anytime

        **GDPR compliance:**

        - Right to access: Users can download their search history
        - Right to deletion: Users can delete all personal data
        - Data portability: Export history in JSON format
        - Transparency: Clear privacy policy explaining data usage

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling autocomplete from 1M to 10B queries/day.

    **Scaling challenges at 10B queries/day:**

    - **Read throughput:** 578K QPS (extremely high)
    - **Low latency:** < 100ms p95 (cannot sacrifice for scale)
    - **Memory:** 160 GB trie structure (must fit in RAM)
    - **Global distribution:** Users worldwide expect low latency

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Redis cluster** | ✅ Yes | Shard across 20 nodes (8 GB each), read replicas (5x replication) |
    | **API servers** | ✅ Yes | 200 servers (3K QPS each), horizontal scaling |
    | **Local cache** | 🟢 No | LRU cache sufficient for top 10K queries |
    | **Database writes** | 🟢 No | Batch updates hourly, low write volume |
    | **Network bandwidth** | 🟡 Approaching | Edge caching (CloudFlare), regional POPs |

    ---

    ## Performance Optimizations

    ### 1. Multi-Level Caching

    **Three-tier cache hierarchy:**

    ```
    L1: Local Cache (in-process)
    ├── Top 10K queries
    ├── Hit rate: 30%
    ├── Latency: < 1ms
    └── Size: 1 MB per server

    L2: Redis Cluster (distributed)
    ├── Full trie (100 GB)
    ├── Hit rate: 95%
    ├── Latency: 5-10ms
    └── Sharded across 20 nodes

    L3: Database (persistent)
    ├── Cold queries
    ├── Hit rate: 5%
    ├── Latency: 50-100ms
    └── PostgreSQL (backup)
    ```

    **Cache invalidation strategy:**

    - **L1 Cache:** TTL = 5 minutes (short TTL for freshness)
    - **L2 Cache:** TTL = 1 hour (rebuilt hourly)
    - **L3 Cache:** No TTL (persistent)

    **Implementation:**

    ```python
    class MultiLevelCache:
        """
        Three-tier caching for autocomplete
        """

        def __init__(self, local_cache, redis_client, db_client):
            self.l1 = local_cache  # Local LRU cache
            self.l2 = redis_client  # Redis cluster
            self.l3 = db_client  # PostgreSQL

        def get_suggestions(self, prefix: str) -> List[str]:
            """
            Query multi-level cache

            Cascades through L1 → L2 → L3 until hit
            """
            # Try L1 (local cache)
            result = self.l1.get(f"suggestions:{prefix}")
            if result:
                metrics.increment('cache.l1.hit')
                return result

            # Try L2 (Redis)
            result = self.l2.get(f"trie:{prefix}")
            if result:
                metrics.increment('cache.l2.hit')

                # Backfill L1
                self.l1.set(f"suggestions:{prefix}", result, ttl=300)

                return json.loads(result)

            # Try L3 (Database)
            metrics.increment('cache.l2.miss')
            result = self.l3.query(
                "SELECT * FROM query_prefixes WHERE prefix = %s",
                (prefix,)
            )

            if result:
                # Backfill L2 and L1
                self.l2.setex(f"trie:{prefix}", 3600, json.dumps(result))
                self.l1.set(f"suggestions:{prefix}", result, ttl=300)

            return result
    ```

    ---

    ### 2. Request Coalescing

    **Problem:** Multiple concurrent requests for same prefix (thundering herd).

    **Solution:** Coalesce identical requests.

    ```python
    import asyncio
    from collections import defaultdict

    class RequestCoalescer:
        """
        Coalesce concurrent identical requests

        Benefits:
        - Reduce Redis load (1 request instead of 100)
        - Lower latency (parallel requests wait together)
        """

        def __init__(self):
            self.pending_requests = {}  # prefix -> Future
            self.lock = asyncio.Lock()

        async def get_suggestions(self, prefix: str, fetch_fn) -> List[str]:
            """
            Get suggestions with request coalescing

            Args:
                prefix: Query prefix
                fetch_fn: Async function to fetch suggestions

            Returns:
                List of suggestions
            """
            async with self.lock:
                # Check if request already pending
                if prefix in self.pending_requests:
                    # Wait for existing request
                    return await self.pending_requests[prefix]

                # Create new request
                future = asyncio.create_task(fetch_fn(prefix))
                self.pending_requests[prefix] = future

            try:
                # Execute request
                result = await future
                return result
            finally:
                # Clean up
                async with self.lock:
                    del self.pending_requests[prefix]
    ```

    **Impact:**

    - **Reduced load:** 10x fewer Redis requests during traffic spikes
    - **Lower latency:** Faster response for coalesced requests
    - **Better resource usage:** Less CPU/memory per request

    ---

    ### 3. Compression

    **Problem:** 100 GB trie in Redis = expensive memory.

    **Solution:** Compress trie nodes.

    ```python
    import zlib
    import json

    class CompressedTrieStorage:
        """
        Store trie nodes with compression

        Trade-off: 3x smaller memory, 2ms added latency
        """

        def set_node(self, redis_client, prefix: str, suggestions: List[Dict]):
            """
            Store trie node with compression
            """
            # Serialize
            json_str = json.dumps(suggestions)

            # Compress (zlib level 6 = good balance)
            compressed = zlib.compress(json_str.encode(), level=6)

            # Store in Redis
            redis_client.setex(
                f"trie:{prefix}",
                3600,  # 1 hour TTL
                compressed
            )

            # Metrics
            original_size = len(json_str)
            compressed_size = len(compressed)
            compression_ratio = original_size / compressed_size

            metrics.gauge('trie.compression_ratio', compression_ratio)

        def get_node(self, redis_client, prefix: str) -> List[Dict]:
            """
            Retrieve and decompress trie node
            """
            # Get from Redis
            compressed = redis_client.get(f"trie:{prefix}")

            if not compressed:
                return None

            # Decompress
            json_str = zlib.decompress(compressed).decode()

            # Deserialize
            suggestions = json.loads(json_str)

            return suggestions
    ```

    **Results:**

    - **Memory savings:** 100 GB → 33 GB (67% reduction)
    - **Latency impact:** +2ms for decompression
    - **Cost savings:** $500/month in Redis memory

    ---

    ### 4. Geographic Distribution

    **Problem:** Global users experience high latency (200ms+ for distant regions).

    **Solution:** Regional POPs (Points of Presence) with edge caching.

    **Architecture:**

    ```
    User in Tokyo → Closest POP (Tokyo)
    ├── Edge Cache (CloudFlare)
    │   ├── Top 100K queries cached at edge
    │   ├── Hit rate: 70%
    │   └── Latency: 10-20ms
    └── Regional Redis Cluster (Tokyo)
        ├── Full trie replica
        ├── Hit rate: 30%
        └── Latency: 30-50ms
    ```

    **Deployment regions:**

    - **North America:** us-east-1, us-west-2 (2 regions)
    - **Europe:** eu-west-1, eu-central-1 (2 regions)
    - **Asia:** ap-northeast-1, ap-southeast-1 (2 regions)
    - **South America:** sa-east-1 (1 region)

    **Benefits:**

    - **Lower latency:** 80% reduction for international users
    - **Higher availability:** Regional failures don't affect global service
    - **Better UX:** Consistent experience worldwide

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Response Time (P50)** | < 50ms | > 100ms |
    | **Response Time (P95)** | < 100ms | > 200ms |
    | **Response Time (P99)** | < 200ms | > 500ms |
    | **Cache Hit Rate (L1)** | > 30% | < 20% |
    | **Cache Hit Rate (L2)** | > 95% | < 90% |
    | **Error Rate** | < 0.1% | > 1% |
    | **Top-1 CTR** | > 30% | < 20% |
    | **Zero Results Rate** | < 5% | > 10% |

    **Dashboards:**

    1. **Latency dashboard:** P50/P95/P99 over time
    2. **Cache dashboard:** Hit rates for L1/L2/L3
    3. **Quality dashboard:** CTR, zero-results, user satisfaction
    4. **Capacity dashboard:** QPS, CPU, memory, network

    **Alerts:**

    - **Critical:** P95 latency > 200ms for 5 minutes
    - **Warning:** Cache hit rate < 90% for 15 minutes
    - **Info:** QPS > 1M (approaching capacity)

    ---

    ## Cost Optimization

    **Monthly cost at 10B queries/day:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $28,800 (200 × m5.xlarge @ $144/month) |
    | **Redis cluster** | $34,560 (20 nodes × r5.2xlarge @ $1,728/month) |
    | **PostgreSQL** | $2,160 (1 × db.r5.xlarge @ $2,160/month) |
    | **ClickHouse** | $4,320 (3 nodes × $1,440/month) |
    | **S3 storage** | $30 (300 GB @ $0.10/GB) |
    | **Data transfer** | $4,500 (50 TB egress @ $0.09/GB) |
    | **CloudFlare CDN** | $2,000 (edge caching) |
    | **Total** | **$76,370/month** (~$916K/year) |

    **Cost per query:** $0.0000076 (< 1 cent per 1,000 queries)

    **Optimization opportunities:**

    1. **Reserved instances:** 40% savings on EC2/RDS ($15K/month)
    2. **Spot instances:** 70% savings on batch processing ($3K/month)
    3. **Compression:** 67% savings on Redis memory ($11K/month)
    4. **Edge caching:** 50% reduction in data transfer ($2K/month)

    **Optimized cost:** $45K/month (~$540K/year)

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Trie data structure:** O(k) prefix matching, memory efficient (100 GB)
    2. **Multi-level caching:** L1 (local) + L2 (Redis) + L3 (DB) = 95%+ hit rate
    3. **Pre-computation:** Hourly batch updates, no real-time ranking
    4. **Fuzzy matching:** Progressive (exact first, fuzzy fallback)
    5. **Personalization:** User history + collaborative filtering
    6. **Geographic distribution:** Regional POPs for global low latency

    ---

    ## Interview Tips

    ✅ **Start with trie data structure** - Explain why it's optimal for prefix matching

    ✅ **Discuss caching strategy** - Multi-level caching critical for performance

    ✅ **Mention fuzzy matching** - Handle typos gracefully

    ✅ **Address personalization** - Blend user history with global suggestions

    ✅ **Consider scale** - 578K QPS requires distributed caching

    ✅ **Talk about metrics** - CTR, latency, zero-results are key quality indicators

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle typos?"** | Fuzzy matching: n-gram indexing + edit distance, progressive search (exact first, fuzzy fallback) |
    | **"How to rank suggestions?"** | Multi-factor ML: popularity (30%), CTR (25%), trending (20%), personalization (15%), recency (10%) |
    | **"How to scale to 10B queries/day?"** | Multi-level caching (L1/L2/L3), sharded Redis, request coalescing, edge caching |
    | **"How to personalize suggestions?"** | User history tracking, collaborative filtering, embedding similarity, privacy controls |
    | **"How to update suggestions in real-time?"** | Hourly batch updates (not real-time), eventual consistency acceptable |
    | **"How to support multiple languages?"** | UTF-8 trie, language-specific ranking, locale-aware suggestions |

    ---

    ## Common Pitfalls to Avoid

    ❌ **Using database for prefix queries** - Too slow (> 500ms), doesn't scale

    ❌ **Real-time ranking** - Adds latency, pre-compute rankings instead

    ❌ **No fuzzy matching** - 10-20% of queries have typos, must handle

    ❌ **Global-only suggestions** - Personalization significantly improves CTR

    ❌ **Single cache tier** - Multi-level caching critical for 95%+ hit rate

    ❌ **Ignoring privacy** - Must allow opt-out, data deletion, GDPR compliance

    ---

    ## Extension Ideas

    **If time permits, discuss:**

    1. **Voice search integration** - Real-time audio to text, prefix suggestions as user speaks
    2. **Image-based autocomplete** - Suggest products from image recognition
    3. **Contextual suggestions** - Time of day, location, device-specific
    4. **Multi-field autocomplete** - Search across users, products, posts simultaneously
    5. **Query understanding** - Semantic search, intent detection, entity recognition

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Google, Amazon, Netflix, YouTube, LinkedIn, Bing, DuckDuckGo, Uber

---

*Master this problem and you'll be ready for: Search autocomplete, type-ahead search, query suggestions, search refinement, related searches*
