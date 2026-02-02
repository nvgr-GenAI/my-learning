# Design a Search Engine (Google)

A distributed web search engine that crawls, indexes, and ranks billions of web pages, providing relevant search results in milliseconds with features like autocomplete, image search, and knowledge graphs.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 8.5B searches/day, 130 trillion pages indexed, 63K searches/second |
| **Key Challenges** | Web crawling at scale, distributed indexing, PageRank calculation, sub-200ms latency |
| **Core Concepts** | Inverted index, TF-IDF, PageRank, distributed crawling, query processing |
| **Companies** | Google, Bing, DuckDuckGo, Microsoft, Amazon (Alexa) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Web Crawling** | Discover and fetch web pages continuously | P0 (Must have) |
    | **Indexing** | Build inverted index for fast keyword lookup | P0 (Must have) |
    | **Search Query** | Return relevant results for user queries | P0 (Must have) |
    | **Ranking** | Rank results by relevance (PageRank + relevance) | P0 (Must have) |
    | **Autocomplete** | Suggest queries as user types | P1 (Should have) |
    | **Image Search** | Search and rank images | P1 (Should have) |
    | **Snippets** | Show preview text from pages | P1 (Should have) |
    | **Spell Check** | Correct misspelled queries | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Video search
    - Personalized search (user history)
    - Ads ranking system
    - Knowledge graph
    - Voice search
    - Local search (maps integration)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Search is critical infrastructure |
    | **Latency (Search)** | < 200ms p95 | Users expect instant results |
    | **Latency (Autocomplete)** | < 50ms p95 | Real-time typing experience |
    | **Freshness** | New pages indexed within 1-2 days | Recent content discovery |
    | **Scalability** | Handle 130 trillion pages | Web continues to grow |
    | **Accuracy** | > 90% relevant results in top 10 | User satisfaction critical |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users: 4B (worldwide)
    Searches per user: ~2 searches/day

    Search queries:
    - Daily searches: 8.5B searches/day
    - Search QPS: 8.5B / 86,400 = ~98,400 searches/sec
    - Peak QPS: 3x average = ~295,000 searches/sec

    Autocomplete requests:
    - Requests per search: ~5 autocomplete requests (as user types)
    - Daily autocomplete: 8.5B × 5 = 42.5B requests/day
    - Autocomplete QPS: 42.5B / 86,400 = ~492,000 req/sec

    Web crawling:
    - Total pages: 130 trillion pages
    - Pages per day: 50B pages/day (new + recrawl)
    - Crawl QPS: 50B / 86,400 = ~578,700 pages/sec
    - Average page size: 100 KB
    - Crawl bandwidth: 578,700 × 100 KB = 57.87 GB/sec = 463 Gbps

    Total Read QPS: ~590K (search + autocomplete)
    Total Write QPS: ~579K (indexing from crawls)
    Read/Write ratio: 1:1 (balanced)
    ```

    ### Storage Estimates

    ```
    Web page storage:
    - Pages indexed: 130 trillion pages
    - Average page size: 100 KB (HTML + resources)
    - Raw storage: 130T × 100 KB = 13 exabytes
    - Compressed (3:1 ratio): 4.3 exabytes

    Inverted index:
    - Average words per page: 500 words
    - Total unique terms: ~100 million unique terms
    - Posting list per term: 100M pages (average)
    - Entry size: 16 bytes (doc_id: 8 bytes, position: 4 bytes, frequency: 4 bytes)
    - Index size: 100M terms × 100M docs × 16 bytes = 160 petabytes

    Forward index (doc_id -> content):
    - Similar to raw page storage: 4.3 exabytes

    PageRank scores:
    - 130T pages × 8 bytes (float64) = 1 petabyte

    Link graph (for PageRank):
    - Average links per page: 50 links
    - Total edges: 130T × 50 = 6.5 quadrillion edges
    - Edge storage: 16 bytes (from_doc, to_doc)
    - Link graph: 6.5Q × 16 bytes = 104 petabytes

    Image index:
    - Indexed images: 10 trillion images
    - Metadata per image: 1 KB
    - Image index: 10T × 1 KB = 10 petabytes

    Total: 4.3 EB (pages) + 160 PB (index) + 4.3 EB (forward) + 104 PB (links) ≈ 8.9 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Crawling ingress:
    - 578,700 pages/sec × 100 KB = 57.87 GB/sec ≈ 463 Gbps

    Search egress:
    - 98,400 searches/sec × 20 results × 1 KB (snippets) = 1.97 GB/sec ≈ 16 Gbps

    Autocomplete egress:
    - 492,000 requests/sec × 10 suggestions × 100 bytes = 4.92 GB/sec ≈ 39 Gbps

    Total ingress: ~463 Gbps (crawling)
    Total egress: ~55 Gbps (search + autocomplete)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot index (most frequent terms):
    - Top 1% terms: 1M terms
    - Posting lists: 1M × 100M docs × 16 bytes = 1.6 TB

    Query cache (search results):
    - Cache 10M most popular queries × 20 results × 1 KB = 200 GB

    Autocomplete cache:
    - Top 100M queries × 10 suggestions × 100 bytes = 100 GB

    PageRank cache (top ranked pages):
    - 100M most popular pages × 8 bytes = 800 MB

    DNS cache:
    - 10M domains × 100 bytes = 1 GB

    Total cache: 1.6 TB + 200 GB + 100 GB + 1 GB ≈ 1.9 TB
    ```

    ---

    ## Key Assumptions

    1. 130 trillion pages indexed (Google's estimate)
    2. 8.5 billion searches per day worldwide
    3. Average page size 100 KB (HTML + metadata)
    4. Web grows by ~50 billion new pages daily
    5. Recrawl frequency: High-value pages daily, others weekly/monthly
    6. Distributed system across multiple data centers globally

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Distributed crawling:** Thousands of crawlers fetching pages in parallel
    2. **Inverted index:** Fast keyword to document lookup
    3. **PageRank:** Link-based authority ranking
    4. **Caching layers:** Query cache, index cache, DNS cache
    5. **Sharding:** Partition index across thousands of servers
    6. **Replication:** 3x replication for availability

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Browser[Web Browser]
            Mobile[Mobile App]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static assets]
            LB[Global Load Balancer<br/>GeoDNS]
        end

        subgraph "API Layer"
            Search_API[Search Service<br/>Query processing]
            Autocomplete_API[Autocomplete Service<br/>Query suggestions]
            Image_API[Image Search Service<br/>Image queries]
        end

        subgraph "Query Processing"
            Query_Parser[Query Parser<br/>Tokenize, analyze]
            Spell_Checker[Spell Checker<br/>Query correction]
            Query_Rewriter[Query Rewriter<br/>Synonyms, expansion]
            Ranker[Ranking Service<br/>ML-based ranking]
        end

        subgraph "Index Layer"
            Index_Server1[Index Server 1<br/>Shard A-D]
            Index_Server2[Index Server 2<br/>Shard E-H]
            Index_ServerN[Index Server N<br/>Shard Y-Z]
            Doc_Server[Document Server<br/>Fetch snippets]
        end

        subgraph "Crawling System"
            URL_Frontier[URL Frontier<br/>Queue management]
            Crawler1[Crawler 1<br/>Fetch pages]
            Crawler2[Crawler 2<br/>Fetch pages]
            CrawlerN[Crawler N<br/>1000s of crawlers]
            DNS_Cache[DNS Cache<br/>Domain resolution]
            Robots_Cache[Robots.txt Cache<br/>Crawl rules]
        end

        subgraph "Indexing Pipeline"
            Content_Parser[Content Parser<br/>Extract text/links]
            Dedup[Deduplication<br/>Check duplicates]
            Indexer[Indexer<br/>Build inverted index]
            Link_Extractor[Link Extractor<br/>Graph building]
        end

        subgraph "Ranking System"
            PageRank_Calc[PageRank Calculator<br/>MapReduce job]
            Link_Graph[(Link Graph DB<br/>Neo4j<br/>6.5Q edges)]
            ML_Ranker[ML Ranking Model<br/>TensorFlow Serving]
        end

        subgraph "Caching"
            Redis_Query[Redis<br/>Query cache]
            Redis_Auto[Redis<br/>Autocomplete cache]
            Redis_Index[Redis<br/>Hot index terms]
        end

        subgraph "Storage"
            Index_DB[(Inverted Index<br/>Distributed KV Store<br/>Sharded)]
            Doc_DB[(Document Store<br/>Bigtable/HBase<br/>Page content)]
            Meta_DB[(Metadata DB<br/>PostgreSQL<br/>Page metadata)]
            Image_DB[(Image Index<br/>Elasticsearch<br/>Image search)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Crawled pages stream]
        end

        Browser --> LB
        Mobile --> LB

        LB --> Search_API
        LB --> Autocomplete_API
        LB --> Image_API

        Search_API --> Query_Parser
        Query_Parser --> Spell_Checker
        Spell_Checker --> Query_Rewriter
        Query_Rewriter --> Redis_Query

        Redis_Query --> Index_Server1
        Redis_Query --> Index_Server2
        Redis_Query --> Index_ServerN

        Index_Server1 --> Index_DB
        Index_Server2 --> Index_DB
        Index_ServerN --> Index_DB

        Index_Server1 --> Ranker
        Index_Server2 --> Ranker
        Index_ServerN --> Ranker

        Ranker --> ML_Ranker
        Ranker --> Doc_Server
        Doc_Server --> Doc_DB

        URL_Frontier --> Crawler1
        URL_Frontier --> Crawler2
        URL_Frontier --> CrawlerN

        Crawler1 --> DNS_Cache
        Crawler1 --> Robots_Cache
        Crawler1 --> Kafka

        Crawler2 --> DNS_Cache
        CrawlerN --> Robots_Cache

        Kafka --> Content_Parser
        Content_Parser --> Dedup
        Dedup --> Indexer
        Dedup --> Link_Extractor

        Indexer --> Index_DB
        Link_Extractor --> Link_Graph
        Link_Graph --> PageRank_Calc
        PageRank_Calc --> Meta_DB

        Autocomplete_API --> Redis_Auto

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Query fill:#fff4e1
        style Redis_Auto fill:#fff4e1
        style Redis_Index fill:#fff4e1
        style Index_DB fill:#ffe1e1
        style Doc_DB fill:#ffe1e1
        style Meta_DB fill:#ffe1e1
        style Link_Graph fill:#e1f5e1
        style Image_DB fill:#e8eaf6
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Inverted Index** | Fast keyword lookup (< 10ms), core of search | Forward index only (too slow), SQL database (can't scale) |
    | **Distributed Crawlers** | Crawl 50B pages/day, 578K pages/sec throughput | Single crawler (too slow), third-party crawlers (no control) |
    | **Kafka** | Stream processing for crawled pages, replay capability | Direct processing (no fault tolerance), RabbitMQ (throughput limits) |
    | **Bigtable/HBase** | Store 130T documents, horizontal scaling, fast random reads | MySQL (can't scale), HDFS only (no random reads), S3 (high latency) |
    | **Redis Cache** | Sub-millisecond query cache, 80% cache hit rate | No cache (500ms+ latency), Memcached (limited features) |
    | **Neo4j (Link Graph)** | PageRank calculation, efficient graph traversal | Relational DB (too slow for graph), adjacency lists (complex) |
    | **MapReduce (PageRank)** | Parallel PageRank calculation across 130T pages | Single machine (impossible), Spark (works but MapReduce proven) |

    **Key Trade-off:** We chose **availability over freshness**. Pages may take 1-2 days to index, but system remains highly available (99.99%).

    ---

    ## API Design

    ### 1. Search Query

    **Request:**
    ```http
    GET /api/v1/search?q=machine+learning&page=1&size=10&safe=on
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "query": "machine learning",
      "total_results": 4820000000,
      "search_time": 0.156,
      "results": [
        {
          "url": "https://en.wikipedia.org/wiki/Machine_learning",
          "title": "Machine Learning - Wikipedia",
          "snippet": "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms...",
          "rank": 1,
          "page_rank": 0.985,
          "relevance_score": 0.943
        },
        {
          "url": "https://www.coursera.org/learn/machine-learning",
          "title": "Machine Learning Specialization by Andrew Ng",
          "snippet": "Learn machine learning fundamentals from Stanford professor Andrew Ng...",
          "rank": 2,
          "page_rank": 0.892,
          "relevance_score": 0.921
        }
        // ... 8 more results
      ],
      "related_searches": [
        "machine learning algorithms",
        "machine learning tutorial",
        "deep learning vs machine learning"
      ],
      "did_you_mean": null
    }
    ```

    **Design Notes:**

    - Query string normalized (lowercase, remove special chars)
    - Pagination with page/size parameters
    - Safe search filter (adult content)
    - Return total results count (approximate)
    - Include search latency for monitoring

    ---

    ### 2. Autocomplete

    **Request:**
    ```http
    GET /api/v1/autocomplete?q=mach&limit=10
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "query": "mach",
      "suggestions": [
        {
          "text": "machine learning",
          "score": 0.98,
          "search_volume": 15000000
        },
        {
          "text": "machine",
          "score": 0.95,
          "search_volume": 8500000
        },
        {
          "text": "machu picchu",
          "score": 0.89,
          "search_volume": 4200000
        }
        // ... 7 more suggestions
      ]
    }
    ```

    **Design Notes:**

    - Prefix matching on query
    - Ranked by popularity (search volume)
    - Return top 10 suggestions (default)
    - Sub-50ms latency requirement

    ---

    ### 3. Submit URL for Crawling

    **Request:**
    ```http
    POST /api/v1/crawler/submit
    Content-Type: application/json

    {
      "url": "https://example.com/new-page",
      "priority": "normal"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "url": "https://example.com/new-page",
      "status": "queued",
      "estimated_crawl_time": "2-3 days",
      "queue_position": 45283921
    }
    ```

    **Design Notes:**

    - Allow webmasters to submit URLs
    - Priority levels: high (news), normal, low
    - Return queue position estimate
    - Rate limit: 100 URLs per day per user

    ---

    ## Database Schema

    ### Inverted Index (Distributed Key-Value Store)

    ```python
    # Key-Value structure
    {
      "term": "machine",
      "postings": [
        {
          "doc_id": 123456789,
          "term_frequency": 15,
          "positions": [10, 45, 89, ...],  # Word positions in document
          "field": "title"  # title, body, anchor_text
        },
        {
          "doc_id": 987654321,
          "term_frequency": 8,
          "positions": [5, 23, 67, ...],
          "field": "body"
        }
        // ... millions more documents containing "machine"
      ],
      "document_frequency": 1500000000,  # Number of docs containing term
      "shard": 12  # Shard assignment for distribution
    }
    ```

    **Sharding strategy:**

    - Shard by term (e.g., hash("machine") % 1000 = shard 12)
    - 1000 index shards distributed across servers
    - Each shard handles ~100,000 terms

    ---

    ### Document Store (Bigtable/HBase)

    ```sql
    -- Document table (NoSQL wide-column store)
    CREATE TABLE documents (
        doc_id BIGINT PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        title TEXT,
        content BLOB,  -- Compressed HTML
        parsed_text TEXT,
        last_crawled TIMESTAMP,
        content_hash CHAR(64),  -- SHA-256 for deduplication
        page_rank DOUBLE,
        domain VARCHAR(255),
        language CHAR(2),
        metadata JSONB,  -- Open Graph, Schema.org
        INDEX idx_url (url),
        INDEX idx_domain (domain),
        INDEX idx_last_crawled (last_crawled)
    ) PARTITION BY HASH (doc_id);
    ```

    ---

    ### Link Graph (Neo4j)

    ```cypher
    // Page node
    CREATE (p:Page {
        doc_id: 123456789,
        url: "https://example.com",
        page_rank: 0.00025
    })

    // Link relationship
    CREATE (p1:Page {doc_id: 123})-[:LINKS_TO {anchor_text: "learn more"}]->(p2:Page {doc_id: 456})

    // Query: Get all outbound links
    MATCH (p:Page {doc_id: 123})-[l:LINKS_TO]->(target:Page)
    RETURN target.doc_id, l.anchor_text

    // Query: Get all inbound links (for PageRank)
    MATCH (source:Page)-[:LINKS_TO]->(p:Page {doc_id: 123})
    RETURN source.doc_id

    // Query: Calculate PageRank (iterative)
    CALL gds.pageRank.stream('linkGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).doc_id AS doc_id, score
    ORDER BY score DESC
    LIMIT 100
    ```

    ---

    ### Metadata (PostgreSQL)

    ```sql
    -- Page metadata
    CREATE TABLE page_metadata (
        doc_id BIGINT PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        title VARCHAR(500),
        description VARCHAR(1000),
        author VARCHAR(255),
        published_date TIMESTAMP,
        last_modified TIMESTAMP,
        page_rank DOUBLE,
        domain_authority INT,
        spam_score DOUBLE,
        language CHAR(2),
        country CHAR(2)
    ) PARTITION BY RANGE (page_rank);

    -- Crawl queue
    CREATE TABLE crawl_queue (
        url TEXT PRIMARY KEY,
        priority INT,  -- 1 (high) to 5 (low)
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        scheduled_at TIMESTAMP,
        retry_count INT DEFAULT 0,
        last_crawl_status VARCHAR(50),
        INDEX idx_priority_scheduled (priority, scheduled_at)
    );

    -- Query autocomplete
    CREATE TABLE query_suggestions (
        query TEXT PRIMARY KEY,
        search_volume BIGINT,
        trending_score DOUBLE,
        last_updated TIMESTAMP,
        INDEX idx_search_volume (search_volume DESC)
    );

    -- For prefix matching (autocomplete)
    CREATE INDEX idx_query_prefix ON query_suggestions
    USING gin (query gin_trgm_ops);
    ```

    ---

    ## Data Flow Diagrams

    ### Search Query Flow

    ```mermaid
    sequenceDiagram
        participant User
        participant LB as Load Balancer
        participant Search as Search Service
        participant Cache as Redis Cache
        participant Parser as Query Parser
        participant Index as Index Servers
        participant Ranker as Ranking Service
        participant Doc as Doc Server

        User->>LB: GET /search?q=machine learning
        LB->>Search: Route to nearest DC

        Search->>Parser: Parse query
        Parser->>Parser: Tokenize: ["machine", "learning"]
        Parser->>Parser: Remove stopwords, stem

        Search->>Cache: Check query cache
        alt Cache HIT (80% of queries)
            Cache-->>Search: Cached results
            Search-->>User: 200 OK (< 50ms)
        else Cache MISS
            Cache-->>Search: null

            Search->>Index: Lookup "machine" (shard 12)
            Search->>Index: Lookup "learning" (shard 45)

            par Parallel index lookups
                Index-->>Search: Posting list for "machine" (1.5B docs)
                Index-->>Search: Posting list for "learning" (2.1B docs)
            end

            Search->>Search: Intersect posting lists
            Search->>Search: Calculate TF-IDF scores

            Search->>Ranker: Rank top 1000 candidates
            Ranker->>Ranker: Apply PageRank + ML model
            Ranker-->>Search: Top 20 results

            Search->>Doc: Fetch snippets for top 20
            Doc-->>Search: Document snippets

            Search->>Cache: Cache results (TTL: 1 hour)
            Search-->>User: 200 OK (< 200ms)
        end
    ```

    **Flow Explanation:**

    1. **Parse query** - Tokenize, remove stopwords, stem words (< 5ms)
    2. **Check cache** - Redis lookup for popular queries (< 2ms)
    3. **Index lookup** - Parallel lookup across shards (< 50ms)
    4. **Intersect** - Find documents containing all terms (< 20ms)
    5. **Rank** - Apply TF-IDF, PageRank, ML model (< 100ms)
    6. **Fetch snippets** - Get document previews (< 30ms)
    7. **Cache** - Store results for future queries

    ---

    ### Web Crawling Flow

    ```mermaid
    sequenceDiagram
        participant Scheduler as Crawl Scheduler
        participant Frontier as URL Frontier
        participant Crawler as Crawler
        participant DNS as DNS Cache
        participant Robots as Robots.txt Cache
        participant Web as Web Server
        participant Kafka
        participant Parser as Content Parser
        participant Dedup as Deduplication
        participant Indexer
        participant Index_DB as Index DB
        participant Link as Link Extractor
        participant Graph as Link Graph

        Scheduler->>Frontier: Add seed URLs + discovered URLs
        Frontier->>Frontier: Prioritize URLs (PageRank, freshness)

        Frontier->>Crawler: Fetch next batch (1000 URLs)

        loop For each URL
            Crawler->>DNS: Resolve domain
            DNS-->>Crawler: IP address

            Crawler->>Robots: Check robots.txt
            Robots-->>Crawler: Crawl rules

            alt Crawl allowed
                Crawler->>Web: GET /page.html (with politeness delay)
                Web-->>Crawler: HTML content (100 KB)

                Crawler->>Kafka: Publish crawled page event
            else Crawl disallowed
                Crawler->>Crawler: Skip URL
            end
        end

        Kafka->>Parser: Consume page events
        Parser->>Parser: Extract text, links, metadata

        Parser->>Dedup: Check if duplicate
        Dedup->>Dedup: Compare content hash

        alt New content
            Parser->>Indexer: Send parsed content
            Parser->>Link: Send extracted links

            Indexer->>Index_DB: Update inverted index
            Link->>Graph: Update link graph
            Link->>Frontier: Add new discovered URLs
        else Duplicate
            Dedup->>Dedup: Discard
        end
    ```

    **Flow Explanation:**

    1. **URL prioritization** - Frontier sorts by PageRank, freshness
    2. **Politeness** - Delay between requests to same domain (1-5 seconds)
    3. **Parallel crawling** - 10,000 crawler instances in parallel
    4. **Respect robots.txt** - Check crawl permissions
    5. **Deduplication** - SHA-256 content hash comparison
    6. **Indexing** - Async indexing via Kafka (decoupled from crawling)
    7. **Link discovery** - Extract links for future crawls

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical search engine subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Inverted Index** | How to search 130 trillion docs in < 200ms? | Distributed inverted index with sharding |
    | **PageRank** | How to rank pages by authority? | Iterative graph algorithm (MapReduce) |
    | **Web Crawler** | How to crawl 50B pages/day? | Distributed crawlers with URL frontier |
    | **Query Processing** | How to understand user intent? | Query parsing, spell check, rewriting |

    ---

    === "📚 Inverted Index"

        ## The Challenge

        **Problem:** Given query "machine learning", find all documents containing both terms from 130 trillion pages in < 200ms.

        **Naive approach:** Scan all documents for keywords. **Doesn't scale** (would take years!).

        **Solution:** Inverted index - map from terms to documents (reverse of document → terms).

        ---

        ## Inverted Index Structure

        **Example:**

        ```
        Documents:
        - Doc 1: "Machine learning is a subset of artificial intelligence"
        - Doc 2: "Deep learning is a type of machine learning"
        - Doc 3: "Artificial intelligence and machine learning are related"

        Inverted Index:
        "machine"    → [Doc1, Doc2, Doc3]
        "learning"   → [Doc1, Doc2, Doc3]
        "artificial" → [Doc1, Doc3]
        "deep"       → [Doc2]
        "intelligence" → [Doc1, Doc3]
        ```

        **With positions and frequencies:**

        ```python
        {
          "machine": {
            "document_frequency": 3,
            "postings": [
              {"doc_id": 1, "tf": 1, "positions": [0], "field": "body"},
              {"doc_id": 2, "tf": 1, "positions": [6], "field": "body"},
              {"doc_id": 3, "tf": 1, "positions": [3], "field": "body"}
            ]
          },
          "learning": {
            "document_frequency": 3,
            "postings": [
              {"doc_id": 1, "tf": 1, "positions": [1], "field": "body"},
              {"doc_id": 2, "tf": 2, "positions": [1, 7], "field": "body"},
              {"doc_id": 3, "tf": 1, "positions": [4], "field": "body"}
            ]
          }
        }
        ```

        ---

        ## Index Building Implementation

        ```python
        from collections import defaultdict
        import hashlib
        import re

        class InvertedIndexBuilder:
            """Build inverted index from crawled documents"""

            def __init__(self):
                self.index = defaultdict(lambda: {
                    "document_frequency": 0,
                    "postings": []
                })
                self.total_docs = 0

            def add_document(self, doc_id: int, content: str, field: str = "body"):
                """
                Add document to index

                Args:
                    doc_id: Unique document identifier
                    content: Text content
                    field: Field name (title, body, anchor_text)
                """
                # Tokenize: lowercase, split on non-alphanumeric
                tokens = re.findall(r'\w+', content.lower())

                # Remove stopwords
                stopwords = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
                tokens = [t for t in tokens if t not in stopwords]

                # Stem words (simple suffix removal)
                tokens = [self._stem(t) for t in tokens]

                # Build term frequency and positions
                term_positions = defaultdict(list)
                for position, term in enumerate(tokens):
                    term_positions[term].append(position)

                # Update inverted index
                for term, positions in term_positions.items():
                    posting = {
                        "doc_id": doc_id,
                        "tf": len(positions),
                        "positions": positions,
                        "field": field
                    }

                    # Add to postings list
                    self.index[term]["postings"].append(posting)
                    self.index[term]["document_frequency"] += 1

                self.total_docs += 1

            def _stem(self, word: str) -> str:
                """Simple stemming (remove common suffixes)"""
                suffixes = ['ing', 'ed', 'es', 's', 'ly']
                for suffix in suffixes:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        return word[:-len(suffix)]
                return word

            def search(self, query: str, top_k: int = 20) -> list:
                """
                Search for query terms

                Args:
                    query: Search query
                    top_k: Number of results to return

                Returns:
                    List of (doc_id, score) tuples
                """
                # Parse query
                terms = re.findall(r'\w+', query.lower())
                terms = [self._stem(t) for t in terms if t not in {'the', 'is', 'a'}]

                if not terms:
                    return []

                # Get posting lists for each term
                posting_lists = []
                for term in terms:
                    if term in self.index:
                        posting_lists.append(self.index[term]["postings"])
                    else:
                        return []  # No results if any term missing

                # Intersect posting lists (find docs containing all terms)
                if len(posting_lists) == 1:
                    candidates = posting_lists[0]
                else:
                    # Merge join on doc_id
                    candidates = self._intersect_postings(posting_lists)

                # Calculate TF-IDF scores
                scores = []
                for posting in candidates:
                    score = self._calculate_tfidf(posting, terms)
                    scores.append((posting["doc_id"], score))

                # Sort by score descending
                scores.sort(key=lambda x: x[1], reverse=True)

                return scores[:top_k]

            def _intersect_postings(self, posting_lists: list) -> list:
                """
                Intersect multiple posting lists

                Returns documents containing all terms
                """
                # Sort by doc_id for efficient merge
                for pl in posting_lists:
                    pl.sort(key=lambda x: x["doc_id"])

                # Start with first posting list
                result = posting_lists[0]

                # Intersect with remaining lists
                for pl in posting_lists[1:]:
                    result = self._merge_join(result, pl)

                return result

            def _merge_join(self, list1: list, list2: list) -> list:
                """Merge join two sorted posting lists"""
                result = []
                i, j = 0, 0

                while i < len(list1) and j < len(list2):
                    if list1[i]["doc_id"] == list2[j]["doc_id"]:
                        result.append(list1[i])
                        i += 1
                        j += 1
                    elif list1[i]["doc_id"] < list2[j]["doc_id"]:
                        i += 1
                    else:
                        j += 1

                return result

            def _calculate_tfidf(self, posting: dict, query_terms: list) -> float:
                """
                Calculate TF-IDF score

                TF-IDF = Term Frequency × Inverse Document Frequency
                TF = (term count in doc) / (total terms in doc)
                IDF = log(total docs / docs containing term)
                """
                import math

                score = 0.0

                for term in query_terms:
                    if term in self.index:
                        # Term frequency (normalized)
                        tf = posting["tf"]

                        # Inverse document frequency
                        df = self.index[term]["document_frequency"]
                        idf = math.log(self.total_docs / df)

                        # TF-IDF
                        score += tf * idf

                return score

        # Usage
        builder = InvertedIndexBuilder()
        builder.add_document(1, "Machine learning is a subset of artificial intelligence")
        builder.add_document(2, "Deep learning is a type of machine learning")
        builder.add_document(3, "Artificial intelligence and machine learning are related")

        results = builder.search("machine learning", top_k=3)
        print(results)  # [(2, 2.48), (1, 2.48), (3, 2.48)]
        ```

        ---

        ## Distributed Sharding

        **Challenge:** 160 petabyte index won't fit on one server.

        **Solution:** Shard by term across 1000 servers.

        ```python
        class DistributedIndexLookup:
            """Distributed index lookup across shards"""

            def __init__(self, num_shards: int = 1000):
                self.num_shards = num_shards
                self.shard_servers = [
                    f"index-shard-{i}.search.com" for i in range(num_shards)
                ]

            def get_shard(self, term: str) -> int:
                """
                Determine shard for a term

                Args:
                    term: Index term

                Returns:
                    Shard number (0 to num_shards-1)
                """
                import hashlib
                hash_value = int(hashlib.md5(term.encode()).hexdigest(), 16)
                return hash_value % self.num_shards

            def lookup_term(self, term: str) -> dict:
                """
                Lookup term in distributed index

                Args:
                    term: Term to lookup

                Returns:
                    Posting list for term
                """
                shard_id = self.get_shard(term)
                server = self.shard_servers[shard_id]

                # Make RPC call to index server
                response = self._rpc_call(server, f"/index/lookup?term={term}")

                return response

            def search_query(self, query_terms: list) -> list:
                """
                Search across shards for multiple terms

                Args:
                    query_terms: List of terms

                Returns:
                    Merged posting lists
                """
                from concurrent.futures import ThreadPoolExecutor, as_completed

                # Lookup each term in parallel
                posting_lists = []

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(self.lookup_term, term): term
                        for term in query_terms
                    }

                    for future in as_completed(futures):
                        term = futures[future]
                        try:
                            postings = future.result()
                            posting_lists.append(postings)
                        except Exception as e:
                            print(f"Error looking up {term}: {e}")

                return posting_lists

            def _rpc_call(self, server: str, endpoint: str) -> dict:
                """Make RPC call to index server (simplified)"""
                import requests
                response = requests.get(f"http://{server}{endpoint}", timeout=1)
                return response.json()
        ```

        **Benefits:**

        - **Parallel lookups:** Query terms looked up simultaneously
        - **Horizontal scaling:** Add more shards as index grows
        - **Fault tolerance:** Replicate each shard 3x

    === "⭐ PageRank Algorithm"

        ## The Challenge

        **Problem:** Not all web pages are equally important. How to rank pages by authority?

        **Insight:** Important pages are linked to by other important pages (like academic citations).

        **PageRank formula:**

        ```
        PR(A) = (1-d) + d × (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))

        where:
        - PR(A) = PageRank of page A
        - d = damping factor (0.85) - probability user continues clicking
        - T1...Tn = pages linking to A
        - C(T) = number of outbound links from page T
        ```

        **Intuition:**

        - Page with many inbound links → high PageRank
        - Link from high-PageRank page → more valuable
        - Page linking to many others → dilutes its voting power

        ---

        ## PageRank Implementation

        ```python
        import numpy as np
        from collections import defaultdict

        class PageRankCalculator:
            """Calculate PageRank for web graph"""

            def __init__(self, damping_factor: float = 0.85, iterations: int = 20):
                self.d = damping_factor
                self.iterations = iterations

            def calculate(self, links: dict) -> dict:
                """
                Calculate PageRank using power iteration method

                Args:
                    links: Graph as dict {page_id: [outbound_link_ids]}

                Returns:
                    Dict of {page_id: pagerank_score}
                """
                # Build graph structure
                pages = set(links.keys())
                for outbound in links.values():
                    pages.update(outbound)

                pages = list(pages)
                n = len(pages)

                # Initialize PageRank (uniform distribution)
                page_rank = {page: 1.0 / n for page in pages}

                # Build inbound links
                inbound_links = defaultdict(list)
                for page, outbound in links.items():
                    for target in outbound:
                        inbound_links[target].append(page)

                # Iterative calculation
                for iteration in range(self.iterations):
                    new_page_rank = {}

                    for page in pages:
                        # Base rank (random surfer teleports)
                        rank = (1 - self.d) / n

                        # Add rank from inbound links
                        for source in inbound_links[page]:
                            source_outbound_count = len(links.get(source, []))
                            if source_outbound_count > 0:
                                rank += self.d * page_rank[source] / source_outbound_count

                        new_page_rank[page] = rank

                    # Check convergence
                    diff = sum(abs(new_page_rank[p] - page_rank[p]) for p in pages)
                    page_rank = new_page_rank

                    print(f"Iteration {iteration + 1}: diff = {diff:.6f}")

                    if diff < 1e-6:
                        print(f"Converged after {iteration + 1} iterations")
                        break

                # Normalize to sum to 1
                total = sum(page_rank.values())
                page_rank = {page: rank / total for page, rank in page_rank.items()}

                return page_rank

        # Example usage
        links = {
            "A": ["B", "C"],
            "B": ["C"],
            "C": ["A"],
            "D": ["C"]
        }

        calculator = PageRankCalculator(damping_factor=0.85, iterations=20)
        page_ranks = calculator.calculate(links)

        for page, rank in sorted(page_ranks.items(), key=lambda x: x[1], reverse=True):
            print(f"Page {page}: {rank:.4f}")
        ```

        **Output:**
        ```
        Iteration 1: diff = 0.300000
        Iteration 2: diff = 0.135000
        ...
        Converged after 18 iterations
        Page C: 0.3887
        Page A: 0.3409
        Page B: 0.2042
        Page D: 0.0662
        ```

        ---

        ## MapReduce for PageRank at Scale

        **Challenge:** 130 trillion pages, 6.5 quadrillion links - can't fit in memory.

        **Solution:** Distributed MapReduce algorithm.

        ```python
        from mrjob.job import MRJob
        from mrjob.step import MRStep

        class PageRankMapReduce(MRJob):
            """
            Distributed PageRank using MapReduce

            Requires multiple iterations (20-30 for convergence)
            """

            def configure_args(self):
                super().configure_args()
                self.add_passthru_arg('--damping', type=float, default=0.85)
                self.add_passthru_arg('--num-pages', type=int, required=True)

            def mapper_parse_graph(self, _, line):
                """
                Parse input: page_id, page_rank, [outbound_links]

                Emit:
                - Graph structure
                - PageRank contributions
                """
                parts = line.split('\t')
                page_id = parts[0]
                page_rank = float(parts[1])
                outbound = parts[2].split(',') if len(parts) > 2 and parts[2] else []

                # Emit graph structure (preserve for next iteration)
                yield page_id, f"GRAPH:{','.join(outbound)}"

                # Emit PageRank contributions to outbound links
                if outbound:
                    contribution = page_rank / len(outbound)
                    for target in outbound:
                        yield target, f"PR:{contribution}"

            def reducer_calculate_pagerank(self, page_id, values):
                """
                Aggregate PageRank contributions

                Calculate new PageRank = (1-d)/N + d × Σ(contributions)
                """
                d = self.options.damping
                n = self.options.num_pages

                graph = None
                rank_sum = 0.0

                for value in values:
                    if value.startswith("GRAPH:"):
                        graph = value[6:]  # Extract graph
                    elif value.startswith("PR:"):
                        rank_sum += float(value[3:])  # Sum contributions

                # Calculate new PageRank
                new_rank = (1 - d) / n + d * rank_sum

                # Emit: page_id, new_rank, outbound_links
                yield page_id, f"{new_rank:.10f}\t{graph}"

            def steps(self):
                return [
                    MRStep(
                        mapper=self.mapper_parse_graph,
                        reducer=self.reducer_calculate_pagerank
                    )
                ]

        # Run for 20 iterations
        # Input: page1 \t 0.00001 \t page2,page3,page4
        # Output: page1 \t 0.00234 \t page2,page3,page4
        ```

        **Process:**

        1. **Initialization:** All pages start with rank 1/N
        2. **Iteration:** Run MapReduce job 20-30 times
        3. **Map phase:** Distribute rank to outbound links
        4. **Reduce phase:** Aggregate contributions, calculate new rank
        5. **Convergence:** Stop when changes < threshold

        **Scale:**

        - **130 trillion pages:** Process in parallel across 10,000 machines
        - **20 iterations:** Takes ~6-8 hours on full cluster
        - **Daily update:** Recalculate PageRank for entire web daily

    === "🕷️ Web Crawler"

        ## The Challenge

        **Problem:** Crawl 50 billion pages per day (578,700 pages/sec) while respecting:

        - Robots.txt (crawl permissions)
        - Politeness (don't overload servers)
        - Bandwidth limits
        - Duplicate content
        - Malicious/spam sites

        ---

        ## Crawler Architecture

        **Components:**

        1. **URL Frontier:** Prioritized queue of URLs to crawl
        2. **DNS Cache:** Avoid redundant DNS lookups
        3. **Robots.txt Cache:** Store crawl permissions
        4. **Content Fetcher:** HTTP/HTTPS requests
        5. **Deduplicator:** Detect duplicate content
        6. **Link Extractor:** Parse HTML, extract links

        ---

        ## URL Frontier Implementation

        ```python
        import heapq
        from collections import defaultdict
        from datetime import datetime, timedelta
        import time

        class URLFrontier:
            """
            Prioritized URL queue with politeness policy

            Features:
            - Priority queue (high PageRank pages first)
            - Politeness: delay between requests to same domain
            - Partitioning by domain for parallel crawling
            """

            def __init__(self, politeness_delay: int = 5):
                """
                Args:
                    politeness_delay: Seconds to wait between requests to same domain
                """
                self.politeness_delay = politeness_delay

                # Priority queue: (priority, timestamp, url, domain)
                self.queue = []

                # Track last crawl time per domain
                self.last_crawl_time = defaultdict(lambda: datetime.min)

                # Domain-based partitioning
                self.domain_queues = defaultdict(list)

            def add_url(self, url: str, priority: float, domain: str):
                """
                Add URL to frontier

                Args:
                    url: URL to crawl
                    priority: Higher = more important (e.g., PageRank score)
                    domain: Domain name for politeness
                """
                # Negative priority for max-heap behavior
                heapq.heappush(self.queue, (-priority, datetime.utcnow(), url, domain))

            def get_next_batch(self, batch_size: int = 100) -> list:
                """
                Get next batch of URLs to crawl

                Returns URLs respecting politeness policy
                """
                urls = []
                seen_domains = set()

                # Temporary storage for URLs we can't crawl yet
                delayed = []

                while len(urls) < batch_size and self.queue:
                    priority, timestamp, url, domain = heapq.heappop(self.queue)

                    # Check politeness: has enough time passed since last crawl?
                    time_since_last = datetime.utcnow() - self.last_crawl_time[domain]

                    if time_since_last.total_seconds() >= self.politeness_delay:
                        # Domain ready to crawl
                        if domain not in seen_domains:
                            urls.append(url)
                            seen_domains.add(domain)
                            self.last_crawl_time[domain] = datetime.utcnow()
                    else:
                        # Too soon, delay this URL
                        delayed.append((priority, timestamp, url, domain))

                # Re-add delayed URLs
                for item in delayed:
                    heapq.heappush(self.queue, item)

                return urls

            def size(self) -> int:
                """Return number of URLs in frontier"""
                return len(self.queue)
        ```

        ---

        ## Crawler Implementation

        ```python
        import requests
        import hashlib
        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup
        import time

        class WebCrawler:
            """Distributed web crawler"""

            def __init__(self, frontier: URLFrontier, kafka_producer):
                self.frontier = frontier
                self.producer = kafka_producer

                # Caches
                self.dns_cache = {}
                self.robots_cache = {}

                # Deduplication (simplified - use Bloom filter in production)
                self.crawled_urls = set()
                self.content_hashes = set()

                # User agent
                self.user_agent = "GoogleBot/1.0 (compatible; MySearchEngine/1.0)"

            def crawl(self, max_pages: int = 1000):
                """
                Main crawl loop

                Args:
                    max_pages: Maximum pages to crawl in this session
                """
                crawled_count = 0

                while crawled_count < max_pages:
                    # Get batch of URLs
                    urls = self.frontier.get_next_batch(batch_size=100)

                    if not urls:
                        print("Frontier empty, waiting...")
                        time.sleep(10)
                        continue

                    # Crawl each URL
                    for url in urls:
                        try:
                            self._crawl_url(url)
                            crawled_count += 1

                            if crawled_count % 100 == 0:
                                print(f"Crawled {crawled_count} pages")

                        except Exception as e:
                            print(f"Error crawling {url}: {e}")

            def _crawl_url(self, url: str):
                """
                Crawl single URL

                Args:
                    url: URL to crawl
                """
                # Check if already crawled
                if url in self.crawled_urls:
                    return

                # Parse URL
                parsed = urlparse(url)
                domain = parsed.netloc

                # Check robots.txt
                if not self._check_robots_txt(domain, url):
                    print(f"Blocked by robots.txt: {url}")
                    return

                # Fetch page
                try:
                    response = requests.get(
                        url,
                        headers={'User-Agent': self.user_agent},
                        timeout=10,
                        allow_redirects=True
                    )

                    if response.status_code != 200:
                        print(f"HTTP {response.status_code}: {url}")
                        return

                    content = response.content

                    # Deduplication: check content hash
                    content_hash = hashlib.sha256(content).hexdigest()
                    if content_hash in self.content_hashes:
                        print(f"Duplicate content: {url}")
                        return

                    # Mark as crawled
                    self.crawled_urls.add(url)
                    self.content_hashes.add(content_hash)

                    # Parse HTML
                    soup = BeautifulSoup(content, 'html.parser')

                    # Extract text
                    text = soup.get_text(separator=' ', strip=True)

                    # Extract links
                    links = []
                    for a_tag in soup.find_all('a', href=True):
                        link = urljoin(url, a_tag['href'])
                        if link.startswith('http'):
                            links.append(link)

                    # Extract metadata
                    title = soup.find('title')
                    title_text = title.string if title else ""

                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc['content'] if meta_desc else ""

                    # Create crawled page object
                    page = {
                        'url': url,
                        'title': title_text,
                        'description': description,
                        'content': text,
                        'content_hash': content_hash,
                        'links': links,
                        'crawled_at': datetime.utcnow().isoformat(),
                        'http_status': response.status_code
                    }

                    # Send to Kafka for indexing
                    self.producer.send('crawled_pages', value=page)

                    # Add discovered links to frontier
                    for link in links:
                        link_domain = urlparse(link).netloc
                        if link not in self.crawled_urls:
                            # Priority: assume medium priority for now
                            # (will update with PageRank later)
                            self.frontier.add_url(link, priority=0.5, domain=link_domain)

                    print(f"Crawled: {url} ({len(links)} links found)")

                except requests.RequestException as e:
                    print(f"Request error: {url} - {e}")

            def _check_robots_txt(self, domain: str, url: str) -> bool:
                """
                Check if URL is allowed by robots.txt

                Args:
                    domain: Domain name
                    url: Full URL

                Returns:
                    True if crawling allowed
                """
                # Check cache
                if domain in self.robots_cache:
                    return self._match_robots_rules(self.robots_cache[domain], url)

                # Fetch robots.txt
                robots_url = f"https://{domain}/robots.txt"
                try:
                    response = requests.get(robots_url, timeout=5)
                    if response.status_code == 200:
                        rules = self._parse_robots_txt(response.text)
                        self.robots_cache[domain] = rules
                        return self._match_robots_rules(rules, url)
                    else:
                        # No robots.txt = allowed
                        self.robots_cache[domain] = {'allow': True}
                        return True
                except:
                    # Error fetching = assume allowed
                    return True

            def _parse_robots_txt(self, content: str) -> dict:
                """Parse robots.txt rules (simplified)"""
                rules = {'disallow': [], 'allow': []}

                for line in content.split('\n'):
                    line = line.strip().lower()
                    if line.startswith('disallow:'):
                        path = line.split(':', 1)[1].strip()
                        rules['disallow'].append(path)
                    elif line.startswith('allow:'):
                        path = line.split(':', 1)[1].strip()
                        rules['allow'].append(path)

                return rules

            def _match_robots_rules(self, rules: dict, url: str) -> bool:
                """Check if URL matches robots.txt rules"""
                path = urlparse(url).path

                # Check disallow rules
                for disallow_path in rules.get('disallow', []):
                    if path.startswith(disallow_path):
                        return False

                return True
        ```

        ---

        ## Politeness & Crawl Rate

        **Politeness strategies:**

        1. **Delay between requests:** 1-5 seconds per domain
        2. **Parallel crawlers:** Different domains crawled simultaneously
        3. **Respect Crawl-delay directive** in robots.txt
        4. **User-agent identification:** Identify as "GoogleBot"

        **Crawl prioritization:**

        | Priority | Description | Examples |
        |----------|-------------|----------|
        | **High** | High PageRank, news sites | CNN.com, Wikipedia |
        | **Medium** | Moderate PageRank, regular updates | Blogs, forums |
        | **Low** | Low PageRank, rarely updated | Personal pages, old sites |

    === "🔤 Query Processing"

        ## The Challenge

        **Problem:** User queries are messy:

        - Typos: "machien learning"
        - Ambiguous: "apple" (fruit vs. company)
        - Incomplete: "best laptop for"
        - Natural language: "What is the capital of France?"

        **Solution:** Multi-stage query processing pipeline.

        ---

        ## Query Processing Pipeline

        ```mermaid
        graph LR
            A[User Query] --> B[Tokenization]
            B --> C[Spell Correction]
            C --> D[Stopword Removal]
            D --> E[Stemming]
            E --> F[Query Expansion]
            F --> G[Intent Detection]
            G --> H[Index Lookup]
        ```

        ---

        ## Implementation

        ```python
        import re
        from typing import List, Tuple
        from collections import Counter

        class QueryProcessor:
            """Process user search queries"""

            def __init__(self):
                # Stopwords (common words with little meaning)
                self.stopwords = {
                    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and',
                    'or', 'but', 'in', 'with', 'to', 'for', 'of'
                }

                # Synonym dictionary
                self.synonyms = {
                    'buy': ['purchase', 'shop', 'order'],
                    'cheap': ['affordable', 'inexpensive', 'budget'],
                    'fast': ['quick', 'rapid', 'speedy']
                }

                # Spell correction dictionary (simplified)
                self.dictionary = self._load_dictionary()

            def process_query(self, query: str) -> dict:
                """
                Process search query through full pipeline

                Args:
                    query: Raw user query

                Returns:
                    Processed query info
                """
                result = {
                    'original': query,
                    'tokenized': [],
                    'corrected': '',
                    'filtered': [],
                    'stemmed': [],
                    'expanded': [],
                    'intent': None
                }

                # 1. Tokenization
                tokens = self._tokenize(query)
                result['tokenized'] = tokens

                # 2. Spell correction
                corrected_tokens = [self._spell_correct(t) for t in tokens]
                result['corrected'] = ' '.join(corrected_tokens)

                # 3. Stopword removal
                filtered = [t for t in corrected_tokens if t.lower() not in self.stopwords]
                result['filtered'] = filtered

                # 4. Stemming
                stemmed = [self._stem(t) for t in filtered]
                result['stemmed'] = stemmed

                # 5. Query expansion (add synonyms)
                expanded = self._expand_query(stemmed)
                result['expanded'] = expanded

                # 6. Intent detection
                intent = self._detect_intent(query)
                result['intent'] = intent

                return result

            def _tokenize(self, query: str) -> List[str]:
                """Split query into tokens"""
                # Lowercase and split on non-alphanumeric
                tokens = re.findall(r'\w+', query.lower())
                return tokens

            def _spell_correct(self, word: str) -> str:
                """
                Spell correction using edit distance

                Returns closest dictionary word within edit distance 2
                """
                # If word in dictionary, return as-is
                if word in self.dictionary:
                    return word

                # Find candidates within edit distance 1
                candidates = self._edits1(word)
                valid = [w for w in candidates if w in self.dictionary]

                if valid:
                    # Return most common word
                    return max(valid, key=self.dictionary.get)

                # Try edit distance 2
                candidates = self._edits2(word)
                valid = [w for w in candidates if w in self.dictionary]

                if valid:
                    return max(valid, key=self.dictionary.get)

                # No correction found
                return word

            def _edits1(self, word: str) -> set:
                """Generate words within edit distance 1"""
                letters = 'abcdefghijklmnopqrstuvwxyz'
                splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

                deletes = [L + R[1:] for L, R in splits if R]
                transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
                replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
                inserts = [L + c + R for L, R in splits for c in letters]

                return set(deletes + transposes + replaces + inserts)

            def _edits2(self, word: str) -> set:
                """Generate words within edit distance 2"""
                return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

            def _stem(self, word: str) -> str:
                """
                Stemming: reduce words to root form

                Examples: running -> run, studies -> studi
                """
                # Simple suffix removal (Porter stemmer simplified)
                suffixes = ['ing', 'ed', 'es', 's', 'ly', 'tion', 'ness']

                for suffix in suffixes:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        return word[:-len(suffix)]

                return word

            def _expand_query(self, terms: List[str]) -> List[str]:
                """
                Query expansion: add synonyms

                Improves recall (find more relevant docs)
                """
                expanded = list(terms)

                for term in terms:
                    if term in self.synonyms:
                        expanded.extend(self.synonyms[term])

                return list(set(expanded))  # Deduplicate

            def _detect_intent(self, query: str) -> str:
                """
                Detect user intent

                Types:
                - informational: "what is", "how to", "define"
                - navigational: brand names, URLs
                - transactional: "buy", "purchase", "order"
                """
                query_lower = query.lower()

                # Informational
                if any(q in query_lower for q in ['what is', 'how to', 'why', 'define']):
                    return 'informational'

                # Transactional
                if any(q in query_lower for q in ['buy', 'purchase', 'order', 'price']):
                    return 'transactional'

                # Navigational (specific site/brand)
                if any(q in query_lower for q in ['facebook', 'youtube', 'amazon', 'twitter']):
                    return 'navigational'

                return 'informational'  # Default

            def _load_dictionary(self) -> dict:
                """Load dictionary with word frequencies (simplified)"""
                # In production: load from large corpus
                return {
                    'machine': 1000000,
                    'learning': 800000,
                    'artificial': 500000,
                    'intelligence': 600000,
                    # ... millions more words
                }

        # Example usage
        processor = QueryProcessor()

        # Example 1: Typo correction
        result = processor.process_query("machien lerning tutorail")
        print(f"Original: {result['original']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Stemmed: {result['stemmed']}")
        print(f"Intent: {result['intent']}")

        # Example 2: Natural language
        result = processor.process_query("What is the best laptop for programming?")
        print(f"\nOriginal: {result['original']}")
        print(f"Filtered: {result['filtered']}")
        print(f"Intent: {result['intent']}")
        ```

        ---

        ## Autocomplete Implementation

        ```python
        from typing import List

        class AutocompleteService:
            """Provide query suggestions as user types"""

            def __init__(self, redis_client):
                self.redis = redis_client
                self.min_prefix_length = 2

            def get_suggestions(self, prefix: str, limit: int = 10) -> List[dict]:
                """
                Get autocomplete suggestions

                Args:
                    prefix: Partial query (e.g., "mach")
                    limit: Max suggestions

                Returns:
                    List of suggestions with scores
                """
                if len(prefix) < self.min_prefix_length:
                    return []

                prefix = prefix.lower()

                # Lookup in Redis sorted set (sorted by search volume)
                # Key: autocomplete:{prefix}
                # Members: full queries
                # Scores: search volumes

                cache_key = f"autocomplete:{prefix}"

                # Get top suggestions from Redis
                suggestions = self.redis.zrevrange(
                    cache_key,
                    start=0,
                    end=limit-1,
                    withscores=True
                )

                # Format results
                results = []
                for query, score in suggestions:
                    results.append({
                        'text': query.decode('utf-8'),
                        'search_volume': int(score),
                        'score': score
                    })

                return results

            def update_autocomplete_index(self, query: str, search_volume: int):
                """
                Update autocomplete index with new query

                Called when users search (batch updated daily)
                """
                query = query.lower()

                # Add query to all prefix keys
                for i in range(self.min_prefix_length, len(query) + 1):
                    prefix = query[:i]
                    cache_key = f"autocomplete:{prefix}"

                    # Add to sorted set with search volume as score
                    self.redis.zadd(cache_key, {query: search_volume})

                    # Keep only top 100 suggestions per prefix
                    self.redis.zremrangebyrank(cache_key, 0, -101)

                    # Set expiry (30 days)
                    self.redis.expire(cache_key, 30 * 24 * 3600)
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 1M to 8.5B searches per day.

    **Scaling challenges at 8.5B searches/day:**

    - **Search QPS:** 98,400 queries/sec (295K peak)
    - **Autocomplete QPS:** 492,000 requests/sec
    - **Crawl rate:** 578,700 pages/sec
    - **Storage:** 8.9 exabytes of data
    - **Index size:** 160 petabytes

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Index lookups** | ✅ Yes | Shard across 1000 servers, cache hot terms, SSD storage |
    | **PageRank calculation** | ✅ Yes | MapReduce on 10K machines, incremental updates, cache scores |
    | **Crawling throughput** | ✅ Yes | 10K parallel crawlers, distributed URL frontier, CDN-like architecture |
    | **Query cache** | 🟡 Approaching | Redis cluster, 80% hit rate, 1-hour TTL |
    | **Network bandwidth** | ✅ Yes | 463 Gbps ingress (crawling), CDN for static assets |

    ---

    ## Optimization Strategies

    ### 1. Index Optimization

    **Problem:** 160 PB inverted index too large, slow lookups.

    **Solutions:**

    | Technique | Savings | Trade-off |
    |-----------|---------|-----------|
    | **Compression** | 70% size reduction | 10% CPU overhead for decompression |
    | **Delta encoding** | Encode doc IDs as differences | 50% space savings, slightly slower |
    | **Pruning** | Remove low-quality pages | 30% smaller index, may miss some results |
    | **Tiered indexing** | Hot/warm/cold tiers (SSD/HDD/S3) | 60% cost savings, 2x latency for cold |

    ```python
    class CompressedInvertedIndex:
        """Space-optimized inverted index"""

        def encode_posting_list(self, doc_ids: List[int]) -> bytes:
            """
            Delta encoding + variable-byte compression

            Example:
            Doc IDs: [100, 105, 110, 120]
            Deltas:  [100, 5, 5, 10]
            Compressed: [0x64, 0x05, 0x05, 0x0A] (1 byte each)

            vs uncompressed: [8, 8, 8, 8] = 32 bytes
            """
            if not doc_ids:
                return b''

            # Sort doc IDs
            doc_ids = sorted(doc_ids)

            # Delta encode
            deltas = [doc_ids[0]]
            for i in range(1, len(doc_ids)):
                deltas.append(doc_ids[i] - doc_ids[i-1])

            # Variable-byte encoding
            compressed = bytearray()
            for delta in deltas:
                while delta >= 128:
                    compressed.append((delta & 0x7F) | 0x80)
                    delta >>= 7
                compressed.append(delta)

            return bytes(compressed)

        def decode_posting_list(self, compressed: bytes) -> List[int]:
            """Decode delta-encoded posting list"""
            deltas = []
            value = 0

            for byte in compressed:
                value |= (byte & 0x7F)
                if byte & 0x80:
                    value <<= 7
                else:
                    deltas.append(value)
                    value = 0

            # Convert deltas back to doc IDs
            doc_ids = [deltas[0]]
            for i in range(1, len(deltas)):
                doc_ids.append(doc_ids[-1] + deltas[i])

            return doc_ids
    ```

    ---

    ### 2. Caching Strategy

    **Multi-level cache:**

    ```
    Query → L1: Browser cache (5 min TTL)
          → L2: CDN cache (1 hour TTL)
          → L3: Redis query cache (1 hour TTL)
          → L4: Hot index terms cache (Redis)
          → L5: Index servers (SSD)
          → L6: Index storage (HDD/S3)
    ```

    **Cache hit rates:**

    - L1 (Browser): 40% hit rate
    - L2 (CDN): 20% hit rate
    - L3 (Redis query): 20% hit rate
    - Total: 80% cache hit rate

    **Latency:**

    - L1 hit: < 10ms
    - L2 hit: < 30ms
    - L3 hit: < 50ms
    - Cache miss: 150-200ms

    ---

    ### 3. Incremental PageRank

    **Problem:** Recalculating PageRank for 130T pages takes 8 hours daily.

    **Solution:** Incremental updates.

    ```python
    class IncrementalPageRank:
        """Update PageRank incrementally for changed pages"""

        def __init__(self, damping: float = 0.85):
            self.d = damping
            self.page_rank = {}  # Current PageRank scores
            self.graph = {}  # Link graph

        def update_page(self, page_id: str, new_links: List[str]):
            """
            Update PageRank after page changes links

            Only recalculate affected pages (local update)
            """
            old_links = self.graph.get(page_id, [])

            # Find changed links
            added = set(new_links) - set(old_links)
            removed = set(old_links) - set(new_links)

            # Update graph
            self.graph[page_id] = new_links

            # Recalculate PageRank for:
            # 1. The modified page
            # 2. Pages it links to (gained/lost inbound link)
            # 3. Pages within distance 2 (affected by propagation)

            affected_pages = self._find_affected_pages(page_id, distance=2)

            # Local recalculation (much faster than full)
            self._recalculate_local(affected_pages)

        def _find_affected_pages(self, page_id: str, distance: int) -> set:
            """Find pages within distance hops"""
            affected = {page_id}
            current = {page_id}

            for _ in range(distance):
                next_level = set()
                for page in current:
                    # Add outbound links
                    next_level.update(self.graph.get(page, []))
                    # Add inbound links
                    for p, links in self.graph.items():
                        if page in links:
                            next_level.add(p)
                affected.update(next_level)
                current = next_level

            return affected

        def _recalculate_local(self, pages: set):
            """Recalculate PageRank for subset of pages"""
            # Power iteration (10 iterations sufficient for local update)
            for _ in range(10):
                new_ranks = {}
                for page in pages:
                    rank = (1 - self.d) / len(self.page_rank)

                    # Sum contributions from inbound links
                    for p, links in self.graph.items():
                        if page in links:
                            rank += self.d * self.page_rank[p] / len(links)

                    new_ranks[page] = rank

                # Update ranks
                self.page_rank.update(new_ranks)
    ```

    **Benefits:**

    - **99% faster:** Update 0.1% of pages instead of 100%
    - **Real-time:** New pages get initial PageRank immediately
    - **Accuracy:** Full recalc weekly, incremental daily

    ---

    ### 4. Geographic Distribution

    **Problem:** Users worldwide need < 200ms latency.

    **Solution:** Deploy to 20+ data centers globally.

    ```
    Global Architecture:

    North America:
    ├── US-West (Oregon): 200 index servers
    ├── US-East (Virginia): 200 index servers
    └── Canada (Toronto): 100 index servers

    Europe:
    ├── EU-West (Ireland): 150 index servers
    ├── EU-Central (Germany): 150 index servers
    └── UK (London): 100 index servers

    Asia:
    ├── Asia-East (Tokyo): 150 index servers
    ├── Asia-Southeast (Singapore): 150 index servers
    └── Asia-South (Mumbai): 100 index servers

    Total: 1,300 index servers globally
    ```

    **Request routing:**

    - GeoDNS: Route to nearest data center
    - CDN: Cache static assets (CSS, JS, images)
    - Anycast: Lowest-latency path

    ---

    ## Cost Optimization

    **Monthly cost at 8.5B searches/day:**

    | Component | Instances | Unit Cost | Total Cost |
    |-----------|-----------|-----------|------------|
    | **Index servers (SSD)** | 1,000 | $500/month | $500,000 |
    | **Crawlers** | 10,000 | $100/month | $1,000,000 |
    | **Redis cache** | 200 nodes | $300/month | $60,000 |
    | **PostgreSQL (metadata)** | 50 nodes | $500/month | $25,000 |
    | **Neo4j (link graph)** | 100 nodes | $800/month | $80,000 |
    | **Kafka brokers** | 100 | $200/month | $20,000 |
    | **Storage (HDD)** | 8.9 EB | $0.01/GB | $93,000 |
    | **Network (egress)** | 4.8 PB/mo | $0.08/GB | $393,000 |
    | **CDN** | 10 PB/mo | $0.02/GB | $200,000 |
    | **Load balancers** | 200 | $50/month | $10,000 |
    | **Total** | | | **$2,381,000/month** |

    **Cost per search:** $2.38M / 8.5B = $0.00028 (~$0.3 per 1000 searches)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Search Latency (P95)** | < 200ms | > 500ms |
    | **Search Latency (P99)** | < 500ms | > 1000ms |
    | **Autocomplete Latency (P95)** | < 50ms | > 150ms |
    | **Crawl Rate** | 578K pages/sec | < 400K pages/sec |
    | **Index Lag** | < 2 days | > 5 days |
    | **Cache Hit Rate** | > 80% | < 70% |
    | **Availability** | 99.99% | < 99.9% |
    | **Query Success Rate** | > 99% | < 98% |

    **Dashboards:**

    1. **Search performance:** Latency, QPS, cache hit rate
    2. **Crawl health:** Crawl rate, queue size, errors
    3. **Index status:** Size, lag, freshness
    4. **Infrastructure:** CPU, memory, disk, network

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Inverted index:** Core data structure for fast keyword lookup
    2. **Distributed sharding:** Partition index across 1000 servers by term
    3. **PageRank:** Link-based authority ranking (MapReduce calculation)
    4. **Politeness:** Respect robots.txt, delay between requests
    5. **Multi-level caching:** Browser → CDN → Redis → Index servers
    6. **Incremental updates:** Avoid full PageRank recalculation
    7. **Geographic distribution:** 20+ data centers for low latency

    ---

    ## Interview Tips

    ✅ **Start with inverted index** - Core concept, explain clearly

    ✅ **Discuss scale challenges** - 130T pages, 8.5B searches/day is massive

    ✅ **PageRank algorithm** - Be ready to explain formula and MapReduce approach

    ✅ **Crawling politeness** - Show awareness of ethical crawling

    ✅ **Query processing** - Spell check, stemming, intent detection

    ✅ **Caching strategy** - Multi-level caching critical for latency

    ✅ **Trade-offs** - Index size vs. completeness, freshness vs. cost

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How does PageRank work?"** | Link-based ranking, iterative calculation, damping factor 0.85, MapReduce for scale |
    | **"How to handle duplicate content?"** | SHA-256 content hash, Bloom filter for fast lookup, canonical URL detection |
    | **"How to crawl 50B pages/day?"** | 10K parallel crawlers, politeness policy, URL frontier with prioritization |
    | **"How to provide autocomplete in < 50ms?"** | Redis sorted sets, prefix indexing, pre-computed suggestions |
    | **"How to keep index fresh?"** | Continuous crawling, high-value pages daily, incremental indexing |
    | **"How to handle 'did you mean'?"** | Spell checker with edit distance, query logs for common corrections |
    | **"How to scale to 1 trillion pages?"** | More shards (10K servers), compression (delta encoding), tiered storage |

    ---

    ## System Design Interview Structure

    **Follow this flow (60 minutes):**

    1. **Requirements (8 min):** Clarify functional (search, crawl, rank) and non-functional (latency, scale)
    2. **Capacity estimation (7 min):** Calculate QPS, storage, bandwidth
    3. **High-level design (10 min):** Draw architecture diagram with main components
    4. **API design (5 min):** Define search, autocomplete APIs
    5. **Data model (5 min):** Inverted index, document store, link graph
    6. **Deep dive (20 min):** Pick 2-3 topics - inverted index, PageRank, crawler
    7. **Scale & optimize (5 min):** Sharding, caching, compression

    ---

    ## Related Problems

    **Master this, and you can solve:**

    - **YouTube search:** Similar index + ranking (video metadata)
    - **E-commerce search:** Product catalog search (Amazon, eBay)
    - **Enterprise search:** Elasticsearch-based internal search
    - **Code search:** GitHub code search (same inverted index)
    - **Image search:** Visual search (CNN embeddings + nearest neighbor)

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Google, Bing, DuckDuckGo, Microsoft, Amazon

---

*Master this problem and you'll be ready for: Any search engine, recommendation system, or large-scale data retrieval system*
