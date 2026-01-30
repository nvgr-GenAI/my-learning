# Design a URL Shortener (like bit.ly, TinyURL)

A URL shortening service creates short aliases for long URLs. When users access the short URL, they are redirected to the original long URL.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M URLs/day creation, 10B redirects/day (100:1 read/write ratio) |
| **Key Challenges** | Short code generation, caching strategy, handling massive scale |
| **Core Concepts** | Base62 encoding, multi-layer caching, database sharding, analytics |
| **Companies** | Amazon, Google, Meta, Microsoft, Uber, Twitter (t.co) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **URL Shortening** | Convert long URL to short URL | P0 (Must have) |
    | **URL Redirection** | Redirect short URL to original URL | P0 (Must have) |
    | **Custom Aliases** | Users can choose custom short URLs | P1 (Should have) |
    | **Expiration** | URLs expire after specified time | P1 (Should have) |
    | **Analytics** | Track click counts, locations, devices | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - User authentication/accounts
    - URL preview/thumbnails
    - QR code generation
    - Link management dashboard

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users expect links to always work |
    | **Latency** | < 100ms for redirection | Fast redirects for good UX |
    | **Read/Write Ratio** | 100:1 | Heavy read, light write workload |
    | **Scalability** | Millions of URLs per day | Must handle viral content |
    | **Durability** | No data loss | Links should never break |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily URL creations: 100M URLs/day
    Daily redirections: 10B redirections/day (100:1 ratio)

    QPS calculations:
    - Write QPS: 100M / (24 * 3600) = ~1,160 writes/sec
    - Read QPS: 10B / (24 * 3600) = ~116,000 reads/sec
    - Peak QPS: 3x average = ~350,000 reads/sec
    ```

    ### Storage Estimates

    ```
    URL storage per record:
    - Short URL: 7 bytes (7 characters)
    - Long URL: 500 bytes (average)
    - Metadata: 100 bytes (created_at, expires_at, user_id)
    - Total per URL: ~600 bytes

    For 5 years of data:
    - Total URLs: 100M * 365 * 5 = 182.5B URLs
    - Storage needed: 182.5B * 600 bytes = ~110 TB

    With 20% buffer: ~130 TB
    ```

    ### Bandwidth Estimates

    ```
    Write bandwidth:
    - 1,160 writes/sec * 600 bytes = ~0.7 MB/sec

    Read bandwidth:
    - 116,000 reads/sec * 600 bytes = ~70 MB/sec
    ```

    ### Memory Estimates (Caching)

    ```
    Cache 20% of daily traffic (Pareto principle):
    - Daily reads: 10B
    - Cache 20%: 2B URLs
    - Memory needed: 2B * 600 bytes = 1.2 TB

    With realistic caching (only hot URLs):
    - Cache top 1% = 100M URLs
    - Memory: 100M * 600 bytes = ~60 GB
    ```

    ---

    ## Key Assumptions

    1. URL length: Average 500 characters, max 2048
    2. Short URL length: 7 characters (62^7 = 3.5 trillion combinations)
    3. URLs don't expire by default (or 5-year expiration)
    4. No authentication required for basic shortening

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Stateless API servers** - Any server can handle any request, enables horizontal scaling
    2. **Read-optimized** - Heavy caching for 100:1 read/write ratio
    3. **Separation of concerns** - URL storage separate from analytics
    4. **Eventual consistency** - Acceptable for non-critical redirects, prioritize availability

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            WebApp[Web Application]
            MobileApp[Mobile App]
            API_Client[API Client]
        end

        subgraph "Load Balancing"
            LB[Load Balancer<br/>nginx/HAProxy]
        end

        subgraph "Application Layer"
            API1[API Server 1<br/>Write & Read]
            API2[API Server 2<br/>Write & Read]
            API3[API Server 3<br/>Write & Read]
        end

        subgraph "Caching Layer"
            Redis[(Redis Cache<br/>Hot URLs)]
        end

        subgraph "Data Layer"
            Primary[(Primary DB<br/>MySQL/PostgreSQL)]
            Replica1[(Read Replica 1)]
            Replica2[(Read Replica 2)]
        end

        subgraph "Storage"
            ObjectStore[Object Storage<br/>Analytics Data]
        end

        WebApp --> LB
        MobileApp --> LB
        API_Client --> LB

        LB --> API1
        LB --> API2
        LB --> API3

        API1 --> Redis
        API2 --> Redis
        API3 --> Redis

        API1 --> Primary
        API2 --> Primary
        API3 --> Primary

        API1 --> Replica1
        API2 --> Replica1
        API3 --> Replica2

        Primary --> Replica1
        Primary --> Replica2

        API1 -.Analytics.-> ObjectStore
        API2 -.Analytics.-> ObjectStore
        API3 -.Analytics.-> ObjectStore

        style LB fill:#e1f5ff
        style Redis fill:#fff4e1
        style Primary fill:#ffe1e1
        style Replica1 fill:#ffe1e1
        style Replica2 fill:#ffe1e1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Load Balancer** | Distribute traffic across multiple API servers for high availability and scalability | Single server (doesn't scale) |
    | **Multiple API Servers** | Horizontal scaling, no single point of failure, handle peak traffic | Vertical scaling (limited, expensive) |
    | **Redis Cache** | Serve 95%+ of reads from memory (< 1ms), reduce database load by 100x | No cache (database overload), CDN only (no centralized invalidation) |
    | **Primary Database** | Single source of truth for URL mappings, ACID guarantees for writes | NoSQL (unnecessary complexity for simple key-value) |
    | **Read Replicas** | Handle massive read traffic (116K QPS), geographic distribution | Single DB (can't handle load), full sharding (premature) |
    | **Object Storage** | Cost-effective storage for high-volume analytics data | Database (expensive, slow for analytics queries) |

    **Key Trade-off:** We chose **availability over strong consistency** (AP in CAP theorem). Brief inconsistency between cache and database is acceptable because:


    - User impact is minimal (may get 404 for few seconds)
    - Redirects are not mission-critical (unlike payments)
    - Performance gain is massive (1ms vs 50ms latency)

    ---

    ## API Design

    ### 1. Create Short URL

    **Request:**
    ```http
    POST /api/v1/shorten
    Content-Type: application/json

    {
      "long_url": "https://example.com/very/long/url/path?param=value",
      "custom_alias": "my-link",     // Optional
      "expiration_date": "2026-12-31" // Optional
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "short_url": "https://short.ly/abc123",
      "long_url": "https://example.com/very/long/url/path?param=value",
      "created_at": "2026-01-29T10:30:00Z",
      "expires_at": "2026-12-31T23:59:59Z"
    }
    ```

    **Design Notes:**

    - `POST` instead of `GET` - Creating a resource (not idempotent)
    - Return `201 Created` with location - RESTful best practice
    - Include both short and long URL - Client confirmation
    - Optional custom_alias - Power users, branded links
    - Optional expiration - Temporary campaigns, security

    ---

    ### 2. Redirect to Long URL

    **Request:**
    ```http
    GET /{short_code}
    ```

    **Response:**
    ```http
    HTTP/1.1 302 Found
    Location: https://example.com/very/long/url/path?param=value
    ```

    **Design Notes:**

    - `302 Found` (temporary redirect) - Allows URL tracking on every request
    - Alternative `301 Moved Permanently` - Browsers cache, no tracking
    - Short path (`/abc123` not `/api/v1/redirect/abc123`) - Minimal characters
    - No auth required - Public access by design

    ---

    ### 3. Get URL Analytics (Optional)

    **Request:**
    ```http
    GET /api/v1/analytics/{short_code}
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "short_code": "abc123",
      "total_clicks": 15420,
      "created_at": "2026-01-15T10:30:00Z",
      "last_accessed": "2026-01-29T14:22:00Z"
    }
    ```

    ---

    ## Database Schema

    ### Option 1: Single Table (Simpler)

    ```sql
    CREATE TABLE urls (
        id BIGSERIAL PRIMARY KEY,
        short_code VARCHAR(10) UNIQUE NOT NULL,
        long_url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        click_count BIGINT DEFAULT 0,
        INDEX idx_short_code (short_code),
        INDEX idx_created_at (created_at)
    );
    ```

    **Use when:** Small to medium scale (< 10M URLs), simple requirements

    **Pros:** Simple, easy to maintain, good enough for most cases

    **Cons:** Writes on every click (for click_count), analytics queries slow down main DB

    ---

    ### Option 2: Separate Analytics Table (Better for scale)

    ```sql
    -- Main URL mapping table
    CREATE TABLE urls (
        id BIGSERIAL PRIMARY KEY,
        short_code VARCHAR(10) UNIQUE NOT NULL,
        long_url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        INDEX idx_short_code (short_code),
        INDEX idx_expires_at (expires_at)
    );

    -- Separate analytics table (can be in different DB)
    CREATE TABLE url_analytics (
        id BIGSERIAL PRIMARY KEY,
        short_code VARCHAR(10) NOT NULL,
        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ip_address VARCHAR(45),
        user_agent TEXT,
        referer TEXT,
        INDEX idx_short_code (short_code),
        INDEX idx_accessed_at (accessed_at)
    );
    ```

    **Use when:** High scale (100M+ URLs), detailed analytics needed

    **Pros:** Write isolation, can use different databases (OLTP for URLs, OLAP for analytics), better query performance

    **Cons:** More complex, eventually consistent counts

    **Recommended:** Option 2 for production systems

    ---

    ## Data Flow Diagrams

    ### Write Flow (Create Short URL)

    ```mermaid
    sequenceDiagram
        participant Client
        participant LB as Load Balancer
        participant API as API Server
        participant Cache as Redis
        participant DB as Database

        Client->>LB: POST /api/v1/shorten<br/>{long_url}
        LB->>API: Route request

        API->>API: Generate short_code (random/hash)
        API->>DB: Check if short_code exists

        alt Code exists (collision)
            DB-->>API: Code exists
            API->>API: Regenerate short_code
            API->>DB: Check again
        end

        DB-->>API: Code available
        API->>DB: INSERT INTO urls
        DB-->>API: Success

        API->>Cache: SET short_code → long_url (TTL: 24h)
        Cache-->>API: OK

        API-->>Client: 201 Created<br/>{short_url, long_url}
    ```

    **Flow Explanation:**

    1. **Load balancer routes** to any available API server (round-robin or least connections)
    2. **Generate short code** using chosen algorithm (random, hash, or sequential)
    3. **Collision check** in database (rare with 62^7 space, but must handle)
    4. **Insert into database** for durability and uniqueness constraint
    5. **Pre-populate cache** with new mapping (write-through caching)
    6. **Return immediately** to user (don't wait for replication)

    **Latency:** ~50-100ms (dominated by database write)

    ---

    ### Read Flow (Redirect)

    ```mermaid
    sequenceDiagram
        participant Client
        participant LB as Load Balancer
        participant API as API Server
        participant Cache as Redis
        participant DB as Read Replica
        participant Analytics as Analytics Queue

        Client->>LB: GET /abc123
        LB->>API: Route request

        API->>Cache: GET abc123

        alt Cache HIT (95-99% of requests)
            Cache-->>API: long_url (< 1ms)
        else Cache MISS (1-5% of requests)
            Cache-->>API: null
            API->>DB: SELECT long_url WHERE short_code='abc123'
            DB-->>API: long_url (5-10ms)
            API->>Cache: SET abc123 → long_url (TTL: 24h)
        end

        API->>Analytics: Log click event (async)
        API-->>Client: 302 Redirect<br/>Location: long_url

        Note over Analytics: Process async for analytics
    ```

    **Flow Explanation:**

    1. **Check cache first** (Redis in-memory, < 1ms latency)
    2. **Cache hit (95%+)** - Return immediately, users get sub-millisecond redirect
    3. **Cache miss (rare)** - Query read replica, populate cache for future requests
    4. **Log analytics asynchronously** - Don't wait, don't slow down redirect
    5. **Return 302 redirect** - Browser follows to long URL

    **Latency:**
    - Cache hit: < 10ms total (mostly network)
    - Cache miss: 20-50ms (database query)

    **Why this is fast:** 95%+ of requests served from memory, no disk I/O

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section covers four critical components that make URL shortening work at scale. Each requires careful design to balance performance, reliability, and cost.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **URL Generation** | How to create unique, short codes? | Random generation with collision handling |
    | **Caching** | How to serve 116K redirects/sec? | Multi-layer caching with 95%+ hit rate |
    | **Rate Limiting** | How to prevent abuse? | Distributed sliding window algorithm |
    | **Analytics** | How to track without slowing redirects? | Async batch processing |

    ---

    === "🔑 URL Generation"

        ## The Challenge

        We need to generate **short, unique identifiers** for billions of URLs. The algorithm must be:

        - **Fast** - Generate codes in < 1ms (1,160 QPS for writes)
        - **Unique** - No collisions (or handle them gracefully)
        - **Short** - 6-7 characters to keep URLs minimal
        - **Scalable** - Work across distributed systems
        - **Unpredictable** - Prevent guessing of other URLs (security)

        **Key insight:** With 62 characters (a-z, A-Z, 0-9) and length 7, we have 62^7 = **3.5 trillion** possible codes. This is enough for 96 years at 100M URLs/day.

        ---

        ## Approach 1: Hash-Based Generation

        **Concept:** Use a cryptographic hash function (MD5, SHA-256) on the long URL, then encode and truncate to desired length.

        **How it works:**

        1. Hash the long URL → Get fixed-size output (e.g., 128-bit MD5)
        2. Encode in base64 → Make URL-safe
        3. Take first N characters → Get short code

        ```python
        import hashlib
        import base64

        def generate_short_code_hash(long_url: str, length: int = 7) -> str:
            """
            Generate short code using hash function
            Pros: Deterministic (same URL → same code)
            Cons: Collisions possible, need to check DB
            """
            # Create hash of URL
            hash_object = hashlib.md5(long_url.encode())
            hash_digest = hash_object.digest()

            # Convert to base64 and take first N characters
            base64_str = base64.urlsafe_b64encode(hash_digest).decode('utf-8')
            short_code = base64_str[:length].rstrip('=')

            return short_code

        # Example usage
        url = "https://example.com/very/long/url"
        code = generate_short_code_hash(url)  # Returns: "6aHR0cH"
        ```

        **Collision handling:**
        ```python
        def create_short_url_with_collision_handling(long_url: str) -> str:
            """Handle collisions by appending counter"""
            base_code = generate_short_code_hash(long_url)
            short_code = base_code
            counter = 0

            while db.exists(short_code):
                # Collision detected, append counter
                counter += 1
                short_code = f"{base_code}{counter}"

                if counter > 100:  # Safety limit
                    # Fallback to random generation
                    short_code = generate_random_code()
                    break

            return short_code
        ```

        **Pros:**

        - Deterministic: Same long URL always gets same short code
        - No central coordination needed
        - Good for deduplication

        **Cons:**

        - Collision probability increases with truncation
        - Must check database for uniqueness
        - Birthday paradox: Collisions more likely than expected

        **Use case:** Systems where same URL should always map to same short code

        ---

        ## Approach 2: Base62 Encoding of Auto-Increment ID

        **Concept:** Use database auto-increment ID, convert to base62 for short representation.

        **How it works:**

        1. Insert URL into database → Get auto-increment ID (e.g., 125,000,000)
        2. Convert ID to base62 → Get short string (e.g., "3D7Vb")
        3. Use base62 string as short code

        ```python
        import string

        class Base62Encoder:
            """
            Convert numeric ID to base62 string
            Pros: Guaranteed unique, no collisions
            Cons: Sequential (predictable), reveals total count
            """

            ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
            BASE = len(ALPHABET)  # 62

            @classmethod
            def encode(cls, num: int) -> str:
                """Convert number to base62 string"""
                if num == 0:
                    return cls.ALPHABET[0]

                result = []
                while num:
                    num, remainder = divmod(num, cls.BASE)
                    result.append(cls.ALPHABET[remainder])

                return ''.join(reversed(result))

            @classmethod
            def decode(cls, code: str) -> int:
                """Convert base62 string to number"""
                num = 0
                for char in code:
                    num = num * cls.BASE + cls.ALPHABET.index(char)
                return num

        # Example usage
        encoder = Base62Encoder()

        # ID from database auto-increment
        url_id = 125_000_000
        short_code = encoder.encode(url_id)  # Returns: "3D7Vb"

        # Decode back
        decoded_id = encoder.decode("3D7Vb")  # Returns: 125000000
        ```

        **Capacity calculation:**
        ```python
        # With 7 characters using base62:
        capacity = 62 ** 7  # = 3,521,614,606,208 (3.5 trillion URLs)

        # How many years at 100M URLs/day?
        years = (62 ** 7) / (100_000_000 * 365)  # = ~96 years
        ```

        **Pros:**

        - **Zero collisions** - Auto-increment guarantees uniqueness
        - **Simple** - Straightforward implementation
        - **Efficient** - Fast encoding/decoding

        **Cons:**

        - **Sequential** - Predictable pattern (security risk)
        - **Reveals scale** - Anyone can estimate total URLs
        - **Coordination** - Need centralized ID generator (bottleneck in distributed systems)

        **Use case:** Internal tools, admin panels where predictability is acceptable

        ---

        ## Approach 3: Random Generation (Recommended)

        **Concept:** Generate random strings from base62 character set, handle rare collisions.

        **How it works:**

        1. Randomly select 7 characters from [a-z, A-Z, 0-9]
        2. Check database if code already exists
        3. If collision (very rare), generate new code
        4. Insert into database

        ```python
        import random
        import string

        class RandomCodeGenerator:
            """
            Generate random short codes
            Pros: Unpredictable, evenly distributed
            Cons: Need collision checking (rare at scale)
            """

            ALPHABET = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
            CODE_LENGTH = 7

            @classmethod
            def generate(cls, length: int = CODE_LENGTH) -> str:
                """Generate random short code"""
                return ''.join(random.choices(cls.ALPHABET, k=length))

            @classmethod
            def generate_with_retry(cls, max_retries: int = 3) -> str:
                """Generate code with collision retry"""
                for attempt in range(max_retries):
                    code = cls.generate()

                    # Check if exists in DB
                    if not db.exists(code):
                        return code

                # If all retries fail (very rare), increase length
                return cls.generate(length=cls.CODE_LENGTH + 1)

        # Collision probability calculation
        def collision_probability(n_codes: int, alphabet_size: int = 62, length: int = 7):
            """
            Calculate probability of collision
            Using birthday paradox approximation: P ≈ n^2 / (2 * N)
            where N = total possible codes (62^7)
            """
            total_combinations = alphabet_size ** length
            probability = (n_codes ** 2) / (2 * total_combinations)
            return probability

        # At 1 billion URLs:
        prob = collision_probability(1_000_000_000)  # ≈ 0.014% (very low)
        ```

        **Collision Analysis:**

        | Total URLs | Collision Probability | Expected Collisions |
        |------------|----------------------|-------------------|
        | 1 million | 0.000014% | ~0.14 |
        | 10 million | 0.0014% | ~14 |
        | 100 million | 0.14% | ~1,400 |
        | 1 billion | 0.014% | ~140,000 |

        Even at 1 billion URLs, collision probability is < 0.02%. With retry logic, this is negligible.

        **Pros:**

        - **Unpredictable** - Cannot guess other URLs (better security)
        - **Distributed** - No central coordination needed
        - **Scalable** - Works across multiple servers
        - **Simple** - Easy to implement and understand

        **Cons:**

        - **Database check required** - Must verify uniqueness
        - **Rare collisions** - Need retry logic (< 0.02% at 1B URLs)

        **Use case:** **Production systems (Recommended)** - Best balance of simplicity, security, and scalability

        ---

        ## Comparison Summary

        | Approach | Uniqueness | Predictability | Scalability | Complexity | Best For |
        |----------|-----------|---------------|------------|-----------|----------|
        | **Hash-Based** | Collisions likely | Medium | High | Medium | Same URL → same code needed |
        | **Base62 ID** | Guaranteed | Predictable | Low (needs coordination) | Low | Internal tools, small scale |
        | **Random** | Very high (0.014% collision at 1B) | Unpredictable | High | Low | **Production (Recommended)** |

    === "⚡ Caching Strategy"

        ## Why Caching is Critical

        **The Problem:** At 116,000 redirects/sec, hitting the database for every request would:

        - **Overload database** - Even with 10 replicas, each handles 11.6K QPS (near limit)
        - **High latency** - Database queries: 10-50ms, unacceptable for redirects
        - **Expensive** - Database hardware costs scale linearly with load
        - **Single point of failure** - Database becomes bottleneck

        **The Solution:** Multi-layer caching to achieve **95%+ cache hit rate**, reducing database load by **20-100x**.

        **Target:** < 10ms p50 latency, < 50ms p99 latency

        ---

        ## Caching Architecture

        **Three-tier caching strategy:**

        1. **Local in-memory cache** (API server) - Fastest, limited capacity
        2. **Distributed Redis cache** (shared) - Fast, large capacity
        3. **Database** (fallback) - Slow, source of truth

        **Cache hierarchy visualization:**

        ```
        Request → Local Cache (< 1ms, 10K URLs)
                     ↓ miss
                  Redis (1-5ms, 100M URLs)
                     ↓ miss
                  Database (10-50ms, all URLs)
        ```

        **Key metrics:**

        - Local cache hit rate: 20-30% (hot URLs accessed frequently by same server)
        - Redis hit rate: 65-75% (remaining requests)
        - **Combined hit rate: 95%+** (only 5% hit database)

        ---

        ## Implementation

        ### Multi-Layer Cache

        ```python
        class URLCache:
            """Multi-layer caching for URL lookups"""

            def __init__(self):
                self.local_cache = {}  # In-memory LRU cache (per server)
                self.redis = RedisClient()  # Distributed cache
                self.db = Database()  # Source of truth

            def get_long_url(self, short_code: str) -> str:
                """Get long_url with multi-layer caching"""

                # Layer 1: Local in-memory cache (< 1ms)
                if short_code in self.local_cache:
                    return self.local_cache[short_code]

                # Layer 2: Redis cache (1-5ms)
                long_url = self.redis.get(f"url:{short_code}")
                if long_url:
                    # Populate local cache for future requests
                    self.local_cache[short_code] = long_url
                    return long_url

                # Layer 3: Database (10-50ms) - Cache miss
                result = self.db.query(
                    "SELECT long_url FROM urls WHERE short_code = %s",
                    (short_code,)
                )

                if result:
                    long_url = result['long_url']

                    # Populate both caches (write-back)
                    self.redis.setex(f"url:{short_code}", 86400, long_url)  # 24h TTL
                    self.local_cache[short_code] = long_url

                    return long_url

                return None  # URL not found
        ```

        **Flow explanation:**

        1. **Check local first** - Fastest, no network round-trip
        2. **Fall back to Redis** - Shared across all servers, large capacity
        3. **Query database only if needed** - Populate caches for future requests
        4. **TTL 24 hours** - Balance freshness vs hit rate

        ---

        ## Cache Eviction Strategy

        **Challenge:** Cache memory is limited. What do we keep?

        **Solution: Least Recently Used (LRU)**

        - Keep frequently accessed URLs in cache
        - Evict URLs not accessed recently
        - Natural fit for 80/20 rule (20% of URLs get 80% of traffic)

        **Redis configuration:**
        ```python
        redis_config = {
            'maxmemory': '60gb',  # Based on capacity estimate
            'maxmemory-policy': 'allkeys-lru',  # Evict least recently used
            'maxmemory-samples': 5  # Sample size for LRU
        }
        ```

        **Why LRU works well for URL shortening:**

        - Viral content gets cached and stays cached (frequent access)
        - Old links naturally drop out (infrequent access)
        - No manual cache invalidation needed for most cases

        ---

        ## Cache Warming Strategy

        **Problem:** After server restart, cache is empty ("cold start")

        **Impact:**

        - First requests slow (all database hits)
        - Database spike can cause cascading failures
        - Poor user experience

        **Solution: Pre-load hot URLs on startup**

        ```python
        def warm_cache_on_startup():
            """Pre-load hot URLs into cache on server start"""

            # Get top 1M most accessed URLs from last 7 days
            hot_urls = db.query("""
                SELECT short_code, long_url
                FROM urls u
                JOIN (
                    SELECT short_code, COUNT(*) as clicks
                    FROM url_analytics
                    WHERE accessed_at > NOW() - INTERVAL '7 days'
                    GROUP BY short_code
                    ORDER BY clicks DESC
                    LIMIT 1000000
                ) a ON u.short_code = a.short_code
            """)

            # Load into Redis using pipeline for efficiency
            pipeline = redis.pipeline()
            for url in hot_urls:
                pipeline.setex(
                    f"url:{url['short_code']}",
                    86400,  # 24h TTL
                    url['long_url']
                )
            pipeline.execute()

            print(f"Warmed cache with {len(hot_urls)} URLs")
        ```

        **When to warm:**

        - On server startup
        - After cache flush
        - During low-traffic periods (proactive refresh)

        **Benefits:**

        - 95%+ hit rate from first request
        - No database spike
        - Consistent performance

        ---

        ## Cache Invalidation

        **The famous quote:** "There are only two hard things in Computer Science: cache invalidation and naming things." — Phil Karlton

        **When to invalidate:**

        | Scenario | Strategy | Reason |
        |----------|----------|--------|
        | **URL updated** | Invalidate immediately | Rare, user-triggered |
        | **URL deleted** | Invalidate immediately | Rare, must return 404 |
        | **URL expired** | Lazy deletion (TTL-based) | Happens automatically |
        | **New URL created** | Write-through | Populate cache immediately |

        **Write-through caching (for new URLs):**
        ```python
        def create_short_url(long_url: str) -> str:
            # Generate short code
            short_code = generate_short_code()

            # Write to database
            db.insert("INSERT INTO urls (short_code, long_url) VALUES (%s, %s)",
                     (short_code, long_url))

            # Immediately cache (write-through)
            redis.setex(f"url:{short_code}", 86400, long_url)

            return short_code
        ```

        **Explicit invalidation (for updates/deletes):**
        ```python
        def delete_url(short_code: str):
            # Delete from database
            db.delete("DELETE FROM urls WHERE short_code = %s", (short_code,))

            # Invalidate cache
            redis.delete(f"url:{short_code}")
        ```

        ---

        ## Cache Performance Metrics

        **Target metrics for monitoring:**

        | Metric | Target | Alert Threshold | Impact |
        |--------|--------|----------------|--------|
        | **Hit Rate** | > 95% | < 90% | High DB load |
        | **Latency (hit)** | < 5ms | > 10ms | Slow redirects |
        | **Latency (miss)** | < 50ms | > 100ms | Poor UX |
        | **Memory Usage** | < 80% | > 90% | Evictions increase |
        | **Eviction Rate** | < 1% of requests | > 5% | Hit rate drops |

        **What to monitor:**

        - Cache hit/miss ratio per minute
        - P50, P95, P99 latency for cache hits vs misses
        - Memory usage and eviction rate
        - Cache availability (is Redis up?)

    === "🚦 Rate Limiting"

        ## Why Rate Limiting Matters

        **The Problem:** Without rate limiting, malicious users can:

        - **Create spam URLs** - Millions of random short URLs flooding the system
        - **Abuse redirects** - DDoS attack by hitting redirects repeatedly
        - **Exhaust resources** - Fill up database and cache with garbage
        - **Cost money** - Unnecessary infrastructure costs

        **The Goal:** Allow legitimate users while blocking abuse.

        **Requirements:**

        - **Per-IP limiting** - 10 URL creations per minute per IP
        - **Distributed** - Work across multiple API servers
        - **Fast** - < 1ms overhead per request
        - **Accurate** - Minimize false positives (blocking legitimate users)

        ---

        ## Rate Limiting Algorithms

        ### Token Bucket Algorithm

        **Concept:** Each user has a bucket with N tokens. Each request consumes 1 token. Tokens refill at constant rate.

        **Analogy:** Like a water bucket with a hole at bottom. Water (tokens) drips in at constant rate. Requests pour water out.

        **Characteristics:**

        - **Burst handling** - Allows short bursts up to bucket size
        - **Smooth rate** - Over time, rate is constant
        - **Simple** - Easy to implement and understand

        **Example:** Bucket size = 10, refill rate = 1/sec

        - User can make 10 requests instantly (burst)
        - Then limited to 1 request/sec
        - After 60 seconds of no activity, bucket refills to 10

        ---

        ### Sliding Window Counter (Recommended)

        **Concept:** Track request timestamps in a time window that "slides" with current time.

        **How it works:**

        1. Store each request timestamp in Redis sorted set
        2. Remove timestamps older than window (e.g., 60 seconds)
        3. Count remaining timestamps
        4. Allow request if count < limit

        **Why it's better:**

        - **More accurate** - Counts actual requests in last N seconds
        - **No burst issues** - Can't bypass by timing requests
        - **Distributed-friendly** - Redis sorted sets handle concurrency

        ```python
        from datetime import datetime

        class RateLimiter:
            """Sliding window rate limiter using Redis"""

            def __init__(self, redis_client):
                self.redis = redis_client

            def is_allowed(
                self,
                ip_address: str,
                max_requests: int = 10,
                window_seconds: int = 60
            ) -> bool:
                """
                Check if request is allowed under rate limit

                Args:
                    ip_address: User's IP address (identifier)
                    max_requests: Maximum requests allowed in window
                    window_seconds: Time window in seconds

                Returns:
                    True if request allowed, False if rate limited
                """
                key = f"ratelimit:{ip_address}"
                now = datetime.utcnow().timestamp()
                window_start = now - window_seconds

                # Remove old entries outside the window
                self.redis.zremrangebyscore(key, 0, window_start)

                # Count requests in current window
                request_count = self.redis.zcard(key)

                if request_count >= max_requests:
                    return False  # Rate limit exceeded

                # Add current request with timestamp as score
                self.redis.zadd(key, {now: now})

                # Set expiration to prevent memory leak
                self.redis.expire(key, window_seconds)

                return True  # Request allowed
        ```

        **Usage in API endpoint:**
        ```python
        @app.post("/api/v1/shorten")
        def create_short_url(request: Request):
            client_ip = request.client.host

            # Check rate limit
            if not rate_limiter.is_allowed(
                ip_address=client_ip,
                max_requests=10,
                window_seconds=60
            ):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Max 10 requests per minute."
                )

            # Process request...
            long_url = request.json()['long_url']
            short_url = create_short_url(long_url)

            return {"short_url": short_url}
        ```

        **Response headers (best practice):**
        ```http
        HTTP/1.1 429 Too Many Requests
        X-RateLimit-Limit: 10
        X-RateLimit-Remaining: 0
        X-RateLimit-Reset: 1643712000
        Retry-After: 45

        {"error": "Rate limit exceeded. Try again in 45 seconds."}
        ```

        ---

        ## Distributed Rate Limiting Challenges

        **Problem:** With multiple API servers, how do we track limits consistently?

        **Challenge 1: Race Conditions**

        - Two servers check count simultaneously
        - Both see count < limit
        - Both allow request
        - Limit exceeded by 1

        **Solution:** Redis atomic operations

        - `ZADD` + `ZCARD` are atomic
        - Redis single-threaded execution prevents races
        - Consistent across all servers

        **Challenge 2: Redis as Single Point of Failure**

        - If Redis down, rate limiting fails
        - Options: Fail open (allow all) or fail closed (reject all)

        **Solution: Fail open with local fallback**
        ```python
        def is_allowed_with_fallback(ip_address: str, max_requests: int, window_seconds: int) -> bool:
            try:
                # Try distributed rate limiting (Redis)
                return is_allowed_redis(ip_address, max_requests, window_seconds)
            except RedisConnectionError:
                # Fallback to local in-memory limiting
                # Less accurate but better than no limiting
                return is_allowed_local(ip_address, max_requests, window_seconds)
        ```

        ---

        ## Rate Limiting Strategy by Use Case

        | Use Case | Limit | Window | Reason |
        |----------|-------|--------|--------|
        | **URL Creation** | 10 requests | 1 minute | Prevent spam, allow burst |
        | **Redirects** | 1000 requests | 1 minute | Allow high traffic, detect DDoS |
        | **Analytics API** | 100 requests | 1 hour | Expensive queries |
        | **Custom Alias** | 3 requests | 1 minute | Prevent keyword squatting |

        **Tiered limits (advanced):**

        - **Free tier:** 10 URLs/minute
        - **Paid tier:** 100 URLs/minute
        - **Enterprise:** No limit (or very high)

    === "📊 Analytics Collection"

        ## The Challenge

        **The Problem:** We need to track every redirect for analytics, but:

        - **Volume** - 10 billion redirects per day = 115K writes/sec
        - **Latency** - Cannot slow down redirects (target: < 10ms)
        - **Data size** - Each event ~200 bytes × 10B = 2TB per day
        - **Queries** - Analytics queries are slow (aggregations, joins)

        **The Goal:** Collect detailed analytics without impacting redirect performance.

        **Key Insight:** Analytics can be **eventually consistent**. It's okay if counts are delayed by 5-10 seconds.

        ---

        ## Architecture: Async Processing

        **Pattern: Fire-and-forget with message queue**

        ```
        Request → API Server → [Redirect immediately (< 10ms)]
                      ↓ async
                Event Queue → Background Worker → Analytics DB
                (in-memory)     (batch process)      (OLAP)
        ```

        **Flow:**

        1. **Redirect immediately** - Don't wait for analytics write
        2. **Queue event** - Add to in-memory queue (< 1ms)
        3. **Background worker** - Process queue in batches
        4. **Batch write** - Insert 1000 events at once (efficient)

        **Why this works:**

        - User gets redirect fast (not blocked by analytics)
        - Batching reduces database load by 1000x
        - Eventually consistent counts (users don't notice 5-sec delay)

        ---

        ## Implementation

        ### Async Event Collection

        ```python
        from dataclasses import dataclass
        from datetime import datetime
        import asyncio

        @dataclass
        class ClickEvent:
            """Click event data structure"""
            short_code: str
            accessed_at: datetime
            ip_address: str
            user_agent: str
            referer: str
            country: str  # From IP geolocation
            device: str   # Mobile/Desktop/Tablet

        class AnalyticsCollector:
            """Async analytics collection to avoid blocking redirects"""

            def __init__(self):
                self.queue = asyncio.Queue(maxsize=100000)  # Buffer 100K events
                self.batch_size = 1000  # Write 1000 events at once
                self.flush_interval = 5  # Flush every 5 seconds

            async def log_click(self, event: ClickEvent):
                """
                Add click event to queue (non-blocking)

                This method returns immediately, analytics written later.
                If queue full, oldest events are dropped (graceful degradation).
                """
                try:
                    await self.queue.put(event, timeout=0.001)  # 1ms timeout
                except asyncio.QueueFull:
                    # Queue full, drop event (or send to backup queue)
                    logger.warning(f"Analytics queue full, dropping event for {event.short_code}")

            async def flush_worker(self):
                """
                Background worker to batch-write analytics

                Runs continuously, collecting events and flushing to database.
                Two flush triggers: batch full OR timeout reached.
                """
                batch = []

                while True:
                    try:
                        # Wait for event with timeout
                        event = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=self.flush_interval
                        )
                        batch.append(event)

                        # Flush when batch is full
                        if len(batch) >= self.batch_size:
                            await self._flush_batch(batch)
                            batch = []

                    except asyncio.TimeoutError:
                        # Timeout reached - flush whatever we have
                        if batch:
                            await self._flush_batch(batch)
                            batch = []

            async def _flush_batch(self, batch: list):
                """
                Write batch to database efficiently

                Using executemany() is 100x faster than individual inserts.
                """
                values = [
                    (
                        e.short_code,
                        e.accessed_at,
                        e.ip_address,
                        e.user_agent,
                        e.referer,
                        e.country,
                        e.device
                    )
                    for e in batch
                ]

                try:
                    await db.executemany(
                        """INSERT INTO url_analytics
                           (short_code, accessed_at, ip_address, user_agent, referer, country, device)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        values
                    )
                    logger.info(f"Flushed {len(batch)} analytics events to database")
                except DatabaseError as e:
                    # Database error - send to dead letter queue for retry
                    logger.error(f"Failed to flush analytics: {e}")
                    await dead_letter_queue.send(batch)
        ```

        ---

        ### Integration with Redirect Endpoint

        ```python
        @app.get("/{short_code}")
        async def redirect_url(short_code: str, request: Request):
            """
            Redirect endpoint with non-blocking analytics

            Critical: Return redirect immediately, log analytics async.
            """

            # 1. Get long URL from cache (fast: < 10ms)
            long_url = cache.get_long_url(short_code)

            if not long_url:
                raise HTTPException(status_code=404, detail="URL not found")

            # 2. Log analytics asynchronously (DON'T AWAIT)
            asyncio.create_task(
                analytics.log_click(ClickEvent(
                    short_code=short_code,
                    accessed_at=datetime.utcnow(),
                    ip_address=request.client.host,
                    user_agent=request.headers.get('user-agent', ''),
                    referer=request.headers.get('referer', ''),
                    country=geolocate_ip(request.client.host),
                    device=detect_device(request.headers.get('user-agent', ''))
                ))
            )

            # 3. Return redirect immediately (don't wait for analytics)
            return RedirectResponse(url=long_url, status_code=302)
        ```

        **Key points:**

        - `asyncio.create_task()` - Schedules task but doesn't wait
        - Redirect returns in < 10ms
        - Analytics written 5-10 seconds later
        - If analytics fails, redirect still works

        ---

        ## Analytics Database Design

        **Use separate database for analytics:**

        | Reason | Explanation |
        |--------|-------------|
        | **Different workload** | OLTP (URL storage) vs OLAP (analytics) |
        | **Query patterns** | Point lookups vs aggregations |
        | **Write volume** | 1,160 writes/sec vs 115,000 writes/sec (100x more) |
        | **Query latency** | Sub-ms vs seconds (analytics queries are slow) |
        | **Storage** | 130 TB for URLs vs 2 TB per day for analytics |

        **Options:**

        1. **Time-series database** (recommended)
           - InfluxDB, TimescaleDB
           - Optimized for time-based queries
           - Automatic data retention (drop old data)
           - Good for: "Clicks per hour in last 7 days"

        2. **Columnar database**
           - ClickHouse, Amazon Redshift
           - Fast aggregations on large datasets
           - Excellent compression (10-50x)
           - Good for: "Top 100 URLs by country"

        3. **Regular SQL with partitioning**
           - PostgreSQL with partitioning by date
           - Cheaper, simpler
           - Good enough for small-medium scale

        **Schema (time-series optimized):**
        ```sql
        -- Partitioned by day for easy deletion of old data
        CREATE TABLE url_analytics (
            id BIGSERIAL,
            short_code VARCHAR(10) NOT NULL,
            accessed_at TIMESTAMP NOT NULL,
            ip_address VARCHAR(45),
            user_agent TEXT,
            referer TEXT,
            country VARCHAR(2),  -- Country code
            device VARCHAR(20),  -- Mobile/Desktop/Tablet
            INDEX idx_short_code_time (short_code, accessed_at),
            INDEX idx_time (accessed_at)
        ) PARTITION BY RANGE (accessed_at);

        -- Partitions (one per day, easy to drop old ones)
        CREATE TABLE url_analytics_2026_01_29
            PARTITION OF url_analytics
            FOR VALUES FROM ('2026-01-29') TO ('2026-01-30');
        ```

        ---

        ## Analytics Queries (Examples)

        **Get total clicks for a URL:**
        ```sql
        SELECT COUNT(*) as total_clicks
        FROM url_analytics
        WHERE short_code = 'abc123';
        ```

        **Get clicks over time (last 7 days):**
        ```sql
        SELECT
            DATE_TRUNC('hour', accessed_at) as hour,
            COUNT(*) as clicks
        FROM url_analytics
        WHERE short_code = 'abc123'
          AND accessed_at > NOW() - INTERVAL '7 days'
        GROUP BY hour
        ORDER BY hour;
        ```

        **Top countries:**
        ```sql
        SELECT
            country,
            COUNT(*) as clicks
        FROM url_analytics
        WHERE short_code = 'abc123'
        GROUP BY country
        ORDER BY clicks DESC
        LIMIT 10;
        ```

        **Device breakdown:**
        ```sql
        SELECT
            device,
            COUNT(*) as clicks,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM url_analytics
        WHERE short_code = 'abc123'
        GROUP BY device;
        ```

        ---

        ## Data Retention Strategy

        **Problem:** 2 TB of analytics data per day = 730 TB per year. Can't keep forever.

        **Solution: Age-based retention**

        | Age | Storage | Reason |
        |-----|---------|--------|
        | 0-30 days | Full detail | Recent data, frequent queries |
        | 31-90 days | Hourly aggregates | Reduce by 60x (3,600 seconds/hour) |
        | 91-365 days | Daily aggregates | Reduce by 1,440x |
        | 1+ years | Monthly aggregates | Historical trends only |

        **Implementation:**
        ```sql
        -- Delete raw data older than 90 days
        DROP TABLE url_analytics_2025_10_01;  -- Instant, just drop partition

        -- Pre-aggregate for older data
        CREATE TABLE url_analytics_hourly AS
        SELECT
            short_code,
            DATE_TRUNC('hour', accessed_at) as hour,
            COUNT(*) as clicks,
            COUNT(DISTINCT ip_address) as unique_visitors,
            array_agg(DISTINCT country) as countries
        FROM url_analytics
        WHERE accessed_at BETWEEN '2025-10-01' AND '2025-12-31'
        GROUP BY short_code, hour;
        ```

        **Storage savings:**

        - Full detail: 2 TB/day × 90 days = 180 TB
        - Hourly: 2 TB/60 × 275 days = 9 TB
        - Daily: 2 TB/1440 × 365 days = 0.5 TB
        - **Total: ~190 TB** vs 730 TB (74% savings)

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    This section covers how to scale from prototype (1K requests/sec) to production (350K requests/sec peak). We'll identify bottlenecks and apply targeted optimizations.

    **Scaling philosophy:**

    1. **Measure first** - Don't optimize blindly
    2. **Target bottlenecks** - Focus on the slowest component
    3. **Scale horizontally** - Add more machines, not bigger machines
    4. **Cache aggressively** - Memory is cheap, databases are expensive

    ---

    ## Bottleneck Identification

    | Component | Current Capacity | Required Capacity | Bottleneck? | Solution |
    |-----------|-----------------|------------------|------------|----------|
    | **API Servers** | 10K req/sec per server | 350K req/sec | ✅ Easily scalable | Add more servers (horizontal scaling) |
    | **Load Balancer** | 500K req/sec | 350K req/sec | ❌ Not a bottleneck | Single LB sufficient, can add backup |
    | **Redis Cache** | 100K req/sec per instance | 350K req/sec | ❌ Not a bottleneck | 4 instances in cluster, can scale to 10+ |
    | **Database (reads)** | 10K req/sec per replica | 17.5K req/sec (5% of 350K) | 🟡 Approaching limit | Add 1-2 more read replicas |
    | **Database (writes)** | 20K writes/sec | 1,160 writes/sec | ❌ Not a bottleneck | Single primary sufficient |
    | **Network** | 1 Gbps | 70 MB/sec = 560 Mbps | ❌ Not a bottleneck | Current network sufficient |

    **Critical bottleneck: Database reads**

    Even with 95% cache hit rate, we still hit database 17.5K times/sec. A single PostgreSQL instance handles ~10K QPS max.

    **Solution:** Add read replicas and improve cache hit rate to 99%.

    ---

    ## Scaling Strategy: Horizontal vs Vertical

    ### Horizontal Scaling (Add More Servers)

    **What it means:** Add more machines of the same size.

    **Example:** 10 servers handling 1K QPS each = 10K QPS total

    **When to use:**

    - Stateless components (API servers, cache nodes)
    - Read-heavy workloads (database read replicas)
    - Near-linear scalability possible

    **Benefits:**

    - **Cost-effective** - Use commodity hardware
    - **Fault tolerance** - Lose one server, 90% capacity remains
    - **Flexibility** - Add/remove servers dynamically

    **Challenges:**

    - **Coordination** - Need load balancer, service discovery
    - **State management** - Can't store state locally
    - **Consistency** - Multiple nodes can have stale data

    ---

    ### Vertical Scaling (Bigger Servers)

    **What it means:** Replace server with bigger machine (more CPU/RAM/disk).

    **Example:** Replace 4-core server with 16-core server

    **When to use:**

    - Single-writer systems (database primary)
    - Memory-intensive workloads (cache nodes)
    - Quick fix before re-architecting

    **Benefits:**

    - **Simple** - No code changes needed
    - **No coordination** - Still single machine
    - **Strong consistency** - Single source of truth

    **Challenges:**

    - **Limits** - Maximum machine size (96 cores, 768 GB RAM)
    - **Expensive** - Cost not linear (16-core is 3x price of 4-core, not 4x)
    - **Downtime** - Must restart to upgrade
    - **Single point of failure** - If machine dies, 100% capacity lost

    ---

    ## Database Scaling Strategy

    ### Phase 1: Read Replicas (Up to 1M read QPS)

    **Concept:** Create read-only copies of primary database. Writes go to primary, reads distributed across replicas.

    ```mermaid
    graph TB
        subgraph "Write Path"
            API_W[API Servers<br/>1,160 writes/sec]
            Primary[(Primary DB<br/>Writes Only)]
        end

        subgraph "Read Path"
            API_R[API Servers<br/>17,500 reads/sec<br/>5% of 350K]
            Replica1[(Read Replica 1<br/>5,800 reads/sec)]
            Replica2[(Read Replica 2<br/>5,800 reads/sec)]
            Replica3[(Read Replica 3<br/>5,800 reads/sec)]
        end

        API_W --> Primary
        Primary -.Replication<br/>Async.-> Replica1
        Primary -.Replication<br/>Async.-> Replica2
        Primary -.Replication<br/>Async.-> Replica3

        API_R --> Replica1
        API_R --> Replica2
        API_R --> Replica3

        style Primary fill:#ff6b6b
        style Replica1 fill:#4ecdc4
        style Replica2 fill:#4ecdc4
        style Replica3 fill:#4ecdc4
    ```

    **How many replicas needed?**
    ```
    Read QPS = 350,000 × 5% (cache miss rate) = 17,500 reads/sec
    Replica capacity = 10,000 reads/sec
    Replicas needed = 17,500 / 10,000 = 2 replicas (add 3 for safety)
    ```

    **Replication lag:** Async replication causes 10-100ms delay. Acceptable for URL shortening (eventual consistency).

    **Load balancing strategy:** Round-robin or least connections across replicas.

    ---

    ### Phase 2: Sharding (Beyond 1M read QPS)

    **When read replicas aren't enough:** At very high scale, even 100 replicas can't keep up. Time to shard.

    **Concept:** Split data across multiple databases by key (e.g., short_code). Each shard is independent database cluster.

    **Sharding strategy: Consistent hashing by short_code prefix**

    ```python
    def get_shard_id(short_code: str, num_shards: int = 64) -> int:
        """
        Determine which shard contains this short_code

        Using first 2 characters ensures even distribution
        (62^2 = 3,844 possible values across 64 shards)
        """
        return hash(short_code[:2]) % num_shards

    # Shard configuration
    SHARDS = {
        0: {
            'primary': 'db-shard-00-primary.us-east-1',
            'replicas': ['db-shard-00-replica-1', 'db-shard-00-replica-2']
        },
        1: {
            'primary': 'db-shard-01-primary.us-east-1',
            'replicas': ['db-shard-01-replica-1', 'db-shard-01-replica-2']
        },
        # ... 62 more shards
        63: {
            'primary': 'db-shard-63-primary.us-west-2',
            'replicas': ['db-shard-63-replica-1', 'db-shard-63-replica-2']
        }
    }
    ```

    **Benefits:**
    - **Massive scale** - 64 shards × 10K QPS = 640K QPS capacity
    - **Isolation** - One shard failure doesn't affect others
    - **Geographic distribution** - Place shards near users

    **Challenges:**
    - **Complexity** - Must route every query to correct shard
    - **Cross-shard queries** - "List all URLs" becomes hard
    - **Rebalancing** - Adding shards requires data migration

    **When to shard:** Only when read replicas exhausted (> 1M read QPS). Premature sharding adds unnecessary complexity.

    ---

    ## Performance Optimizations

    ### Database Optimizations

    **1. Partitioning by creation date**

    **Problem:** Deleting expired URLs is slow (full table scan).

    **Solution:** Partition table by month. Drop entire partition instead of DELETE query.

    ```sql
    -- Partitioning setup
    CREATE TABLE urls (
        id BIGSERIAL,
        short_code VARCHAR(10) NOT NULL,
        long_url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    ) PARTITION BY RANGE (created_at);

    -- Monthly partitions (create programmatically)
    CREATE TABLE urls_2026_01 PARTITION OF urls
        FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

    CREATE TABLE urls_2026_02 PARTITION OF urls
        FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

    -- Cleanup expired URLs (instant, no DELETE query)
    DROP TABLE urls_2024_01;  -- Drop 2-year-old partition
    ```

    **Benefits:**

    - Deletion is instant (DROP TABLE vs DELETE scan)
    - Queries faster (scan only relevant partitions)
    - Storage reclaimed immediately

    ---

    **2. Connection Pooling**

    **Problem:** Opening database connection takes 50-100ms. At 10K QPS, that's 10,000 connections!

    **Solution:** Reuse connections with connection pool.

    ```python
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    # Connection pool configuration
    engine = create_engine(
        'postgresql://user:pass@db-host/urlshortener',
        poolclass=QueuePool,
        pool_size=20,          # Keep 20 connections per API server
        max_overflow=10,       # Allow 10 extra under load
        pool_timeout=30,       # Wait 30s for connection
        pool_recycle=3600,     # Recycle connections every hour (prevent stale)
        pool_pre_ping=True     # Check connection health before using
    )
    ```

    **Benefits:**

    - **No connection overhead** - Reuse existing connections
    - **Limit connections** - Prevent database overload (max 30 per server)
    - **Automatic recovery** - Recycle stale connections

    **Calculation:**

    - 50 API servers × 30 connections = 1,500 total connections
    - Database limit: 10,000 connections (well within limit)

    ---

    **3. Indexing Strategy**

    **Critical indexes:**
    ```sql
    -- Primary lookup index (most important)
    CREATE UNIQUE INDEX idx_short_code ON urls(short_code);

    -- Cleanup index for expired URLs
    CREATE INDEX idx_expires_at ON urls(expires_at)
    WHERE expires_at IS NOT NULL;

    -- Optional: Find URLs by long_url (deduplication)
    CREATE INDEX idx_long_url_hash ON urls(MD5(long_url));
    ```

    **Index maintenance:**

    - `REINDEX` weekly to prevent bloat
    - `VACUUM ANALYZE` daily for query planner statistics
    - Monitor index usage: `SELECT * FROM pg_stat_user_indexes`

    ---

    ## Trade-offs Discussion

    **Every decision has trade-offs. Here are the key ones:**

    ### Short Code Generation

    | Decision | Pros | Cons | When to Choose |
    |----------|------|------|----------------|
    | **Hash-based** | Deterministic, same URL → same code | Collisions likely, need DB check | Same URL needs same code (deduplication) |
    | **Base62 ID** | Zero collisions, simple | Predictable, reveals count | Internal tools, small scale |
    | **Random** | Unpredictable, scalable | Rare collisions (0.014% at 1B) | **Production systems** |

    **Our choice:** Random generation - Best balance of simplicity, security, and scalability.

    ---

    ### Cache Strategy

    | Decision | Pros | Cons | When to Choose |
    |----------|------|------|----------------|
    | **Aggressive caching (1h+ TTL)** | Fewer DB hits, faster | Stale data possible | High read/write ratio (ours: 100:1) |
    | **Conservative caching (5min TTL)** | Fresh data | More DB hits | Frequently updated URLs |

    **Our choice:** Aggressive caching (24h TTL) - Reads are 100x writes, acceptable staleness.

    ---

    ### Consistency Model

    | Decision | Pros | Cons | When to Choose |
    |----------|------|------|----------------|
    | **Strong consistency** | No stale redirects | Higher latency, less available | Critical links (payments) |
    | **Eventual consistency** | Lower latency, high availability | Brief inconsistency possible | **General use (ours)** |

    **Our choice:** Eventual consistency - Brief 404 acceptable, performance critical.

    ---

    ### Analytics Processing

    | Decision | Pros | Cons | When to Choose |
    |----------|------|------|----------------|
    | **Real-time** | Instant insights | High write load, expensive | Business-critical metrics |
    | **Batch (5min delay)** | Efficient, scalable | Delayed insights | **General analytics (ours)** |

    **Our choice:** Batch processing - Users tolerate 5-sec delay, saves 100x database load.

    ---

    ## Monitoring & Alerting

    **What to monitor:**

    | Metric | Target | Alert Threshold | Why It Matters |
    |--------|--------|-----------------|----------------|
    | **Redirect Latency (P50)** | < 10ms | > 50ms | User experience |
    | **Redirect Latency (P99)** | < 50ms | > 200ms | Tail latency affects some users |
    | **Cache Hit Rate** | > 95% | < 90% | Low hit rate = database overload |
    | **Error Rate** | < 0.1% | > 1% | System health |
    | **404 Rate** | < 1% | > 5% | Broken links or misconfiguration |
    | **DB Connection Pool** | < 80% | > 85% | Connection exhaustion |
    | **CPU Usage** | < 60% | > 80% | Need to scale API servers |
    | **Memory Usage** | < 75% | > 85% | Cache evictions or memory leak |

    **Alert priorities:**

    - **P0 Critical:** Error rate > 1%, redirect latency P99 > 500ms
      - Page ops team immediately
      - Expect resolution in < 15 minutes

    - **P1 High:** Cache hit rate < 80%, DB pool > 80%
      - Alert on-call engineer
      - Investigate within 1 hour

    - **P2 Medium:** 404 rate > 5%, latency P99 > 200ms
      - Create ticket for next day
      - Investigate within 24 hours

    **Monitoring tools:**

    - **Metrics:** Prometheus, DataDog, CloudWatch
    - **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana)
    - **Tracing:** Jaeger, Zipkin for distributed tracing
    - **Uptime:** Pingdom, UptimeRobot for external monitoring

=== "📝 Summary & Tips"

    ## Architecture Summary

    **Core Components:**

    | Component | Purpose | Technology | Quantity |
    |-----------|---------|------------|----------|
    | **Load Balancer** | Distribute traffic, health checks | nginx, HAProxy, AWS ELB | 1 (+ backup) |
    | **API Servers** | Stateless, horizontally scalable | Python/Flask, Go, Node.js | 20-50 servers |
    | **Redis Cache** | 95%+ cache hit rate, 24h TTL | Redis Cluster | 4-10 nodes |
    | **Database Primary** | Single writer, ACID guarantees | PostgreSQL, MySQL | 1 primary |
    | **Database Replicas** | Read scaling, geographic distribution | PostgreSQL, MySQL | 3-10 replicas |
    | **Object Storage** | Cost-effective analytics storage | Amazon S3, GCS | 1 bucket |

    ---

    ## Capacity Handled

    | Metric | Capacity | Headroom |
    |--------|----------|----------|
    | **URL creations** | 100M URLs/day (~1,160 QPS) | 10x capacity (database can handle 20K writes/sec) |
    | **Redirections** | 10B redirections/day (~116K QPS) | 3x capacity (350K peak QPS) |
    | **Storage** | 130 TB for 5 years | With compression: ~50 TB |
    | **Cache** | 60 GB for hot URLs | 100M URLs cached |
    | **Latency** | < 10ms p50, < 50ms p99 | 95%+ cache hit rate |

    ---

    ## Key Design Decisions

    1. **Random short code generation** - Best balance of simplicity, security, and collision avoidance
    2. **Aggressive caching (24h TTL)** - 95%+ hit rate reduces DB load by 20-100x
    3. **Read replicas for scaling** - Handle 100:1 read/write ratio without sharding
    4. **Async analytics collection** - Don't slow down redirects for tracking
    5. **Eventual consistency** - Prioritize availability and performance over strong consistency
    6. **Horizontal scaling** - Stateless API servers enable near-linear scaling

    ---

    ## Interview Tips

    ### What Interviewers Look For

    ✅ **Start with requirements** - Functional, non-functional, capacity estimation
    ✅ **Discuss trade-offs** - Hash vs random vs sequential, caching strategies
    ✅ **Identify bottlenecks** - Database reads, cache capacity, network
    ✅ **Propose scaling solutions** - Read replicas, sharding, CDN
    ✅ **Consider edge cases** - Collisions, expired URLs, rate limiting, abuse prevention
    ✅ **Think about monitoring** - What metrics to track, when to alert

    ---

    ### Common Follow-up Questions

    | Question | Key Points to Cover |
    |----------|-------------------|
    | **"How do you handle short code collisions?"** | Retry with new code, very low probability (0.014% at 1B URLs), exponential backoff |
    | **"What if cache fails completely?"** | Graceful degradation to database, circuit breaker pattern, local fallback cache |
    | **"How would you prevent abuse/spam?"** | Rate limiting by IP (10/min), CAPTCHA for suspicious IPs, API keys for power users |
    | **"How add analytics without slowing redirects?"** | Async queue processing, batch writes (1000 events at once), separate analytics DB |
    | **"How support custom aliases?"** | Check uniqueness, reserve namespace (e.g., '/admin'), input validation, rate limit stricter |
    | **"How to handle hot URLs (viral content)?"** | Already handled by cache, consider CDN for static pages, multiple cache layers |
    | **"What about security?"** | URL validation (prevent malicious redirects), rate limiting, HTTPS, XSS protection |

    ---

    ### Things to Mention

    **CAP Theorem Trade-offs:**

    - We chose AP (Availability + Partition Tolerance) over C (Consistency)
    - Eventual consistency acceptable for URL shortening
    - Brief 404 better than system down

    **Database Strategy:**

    - Start with read replicas (simpler)
    - Shard only when necessary (> 1M QPS)
    - Partition by date for efficient cleanup

    **Monitoring:**

    - Track cache hit rate, latency percentiles, error rates
    - Alert on degradation (< 90% hit rate, > 200ms P99)
    - Dashboard for real-time visibility

    **Security:**

    - Validate URLs (prevent redirects to malicious sites)
    - Rate limiting (prevent spam, DDoS)
    - HTTPS everywhere (prevent man-in-the-middle)

    ---

    ## Related Problems

    | Problem | Similarity | Key Differences |
    |---------|------------|-----------------|
    | **Pastebin** | Very similar - store text, return short code | Store text content instead of URLs, syntax highlighting, larger payloads |
    | **TinyURL with Analytics** | Extension of this problem | Add detailed analytics dashboard, A/B testing, user accounts |
    | **QR Code Generator** | Related - also uses short codes | Generate QR codes, image storage, multiple image formats |
    | **Instagram/Twitter** | Uses URL shortening internally | More complex social features, feeds, timelines, user graphs |
    | **File Upload Service** | Similar short identifier concept | Handle large files, chunking, resumable uploads, virus scanning |

    ---

    ## Next Steps

    **After mastering URL Shortener:**

    1. **Pastebin** - Similar design with text storage and expiration
    2. **Rate Limiter** - Deep dive into distributed rate limiting algorithms
    3. **Twitter Feed** - Advanced fan-out patterns and timeline generation
    4. **Video Streaming** - CDN, encoding, adaptive bitrate streaming

    **Practice variations:**

    - Add user authentication and URL management dashboard
    - Implement detailed analytics with charts and reports
    - Add link expiration and automatic cleanup jobs
    - Support QR code generation for URLs
    - Implement A/B testing for redirect destinations
    - Add custom domains (e.g., brand.com/abc123)

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Amazon, Google, Meta, Microsoft, Uber, Twitter

---

*This template applies to similar problems: Pastebin, File Upload Service, Image Hosting Service*
