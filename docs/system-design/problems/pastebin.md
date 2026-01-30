# Design Pastebin (like pastebin.com, GitHub Gist)

A text sharing service that allows users to store and share text snippets. Users can paste text, get a unique URL, and share it with others.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M pastes/day, 500M reads/day (10:1 read/write ratio) |
| **Key Challenges** | Text storage optimization, expiration handling, syntax highlighting |
| **Core Concepts** | Object storage, content deduplication, CDN, text compression |
| **Companies** | Amazon, Google, Microsoft, Dropbox, GitHub |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Create Paste** | User pastes text, gets unique URL | P0 (Must have) |
    | **Read Paste** | Anyone with URL can view paste | P0 (Must have) |
    | **Expiration** | Pastes expire after specified time (1h, 1d, 1w, never) | P0 (Must have) |
    | **Syntax Highlighting** | Detect language, apply syntax highlighting | P1 (Should have) |
    | **Custom Aliases** | Users can choose custom short URLs | P2 (Nice to have) |
    | **Edit/Delete** | Paste creator can edit or delete | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - User authentication (optional feature)
    - Comments on pastes
    - Forking/versioning
    - Real-time collaboration
    - Paste analytics

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users expect pastes to be accessible 24/7 |
    | **Latency** | < 200ms for read | Fast loading for good UX |
    | **Durability** | No data loss | Pastes should never disappear unexpectedly |
    | **Scalability** | Millions of pastes per day | Handle viral content |
    | **Storage Efficiency** | Compress text | Save storage costs |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily paste creations: 50M pastes/day
    Daily reads: 500M reads/day (10:1 ratio)

    QPS calculations:
    - Write QPS: 50M / (24 * 3600) = ~580 writes/sec
    - Read QPS: 500M / (24 * 3600) = ~5,800 reads/sec
    - Peak QPS: 3x average = ~17,400 reads/sec
    ```

    ### Storage Estimates

    ```
    Paste size assumptions:
    - Average paste size: 10 KB (most are code snippets, logs)
    - Metadata: 500 bytes (paste_id, created_at, expires_at, language, etc.)
    - Total per paste: ~10.5 KB

    For 5 years of data:
    - Daily: 50M pastes × 10.5 KB = 525 GB/day
    - Yearly: 525 GB × 365 = 192 TB/year
    - 5 years: 192 TB × 5 = 960 TB

    With compression (3:1 ratio for text):
    - Actual storage: 960 TB / 3 = ~320 TB

    With 20% buffer: ~400 TB
    ```

    ### Bandwidth Estimates

    ```
    Write bandwidth:
    - 580 writes/sec × 10 KB = ~5.8 MB/sec

    Read bandwidth:
    - 5,800 reads/sec × 10 KB = ~58 MB/sec
    ```

    ### Memory Estimates (Caching)

    ```
    Cache 20% of daily reads (hot pastes):
    - Daily reads: 500M
    - Cache 20%: 100M pastes
    - Memory: 100M × 10 KB = ~1 TB

    Realistic caching (top 1% hot pastes):
    - Cache 5M pastes
    - Memory: 5M × 10 KB = ~50 GB
    ```

    ---

    ## Key Assumptions

    1. Average paste size: 10 KB (range: 100 bytes to 10 MB)
    2. Short URL length: 8 characters (62^8 = 218 trillion combinations)
    3. Default expiration: 30 days (configurable)
    4. Read-heavy workload: 10:1 read/write ratio
    5. Text compression: 3:1 ratio on average

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Object storage for text** - Store paste content in S3/Blob storage for cost efficiency
    2. **Metadata in database** - Fast lookups for paste metadata (paste_id → storage_key)
    3. **Aggressive caching** - CDN + Redis for hot pastes
    4. **Async expiration** - Background jobs to delete expired pastes

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Browser]
            Mobile[Mobile App]
            API_Client[API/CLI]
        end

        subgraph "CDN Layer"
            CDN[CDN<br/>CloudFront/Cloudflare<br/>Cache static content]
        end

        subgraph "Load Balancing"
            LB[Load Balancer<br/>nginx]
        end

        subgraph "Application Layer"
            API1[API Server 1]
            API2[API Server 2]
            API3[API Server 3]
        end

        subgraph "Caching Layer"
            Redis[(Redis Cache<br/>Metadata + Hot Pastes)]
        end

        subgraph "Data Layer"
            Primary[(Primary DB<br/>Metadata)]
            Replica1[(Read Replica 1)]
            Replica2[(Read Replica 2)]
        end

        subgraph "Storage Layer"
            S3[Object Storage<br/>S3/Azure Blob<br/>Paste Content]
        end

        subgraph "Background Jobs"
            Expiration[Expiration Worker<br/>Delete expired pastes]
        end

        Web --> CDN
        Mobile --> CDN
        API_Client --> LB

        CDN --> LB

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

        API1 --> S3
        API2 --> S3
        API3 --> S3

        Expiration --> Primary
        Expiration --> S3

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis fill:#fff4e1
        style Primary fill:#ffe1e1
        style Replica1 fill:#ffe1e1
        style Replica2 fill:#ffe1e1
        style S3 fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **CDN** | Cache static pastes closer to users, reduce origin load by 60-80% | Direct server access (higher latency, more load) |
    | **Object Storage (S3)** | Cost-effective storage (~$0.023/GB vs ~$0.10/GB for database), unlimited scale | Database BLOBs (expensive, slow, limited scale) |
    | **Redis Cache** | Fast metadata lookups (< 1ms), cache hot paste content | No cache (slower reads, database overload) |
    | **Separate Metadata DB** | Fast paste_id lookups, manage expiration efficiently | Store everything in S3 (slow lookups, no indexing) |
    | **Background Workers** | Delete expired pastes without blocking user requests | Lazy deletion on read (unpredictable latency spikes) |

    **Key Trade-off:** We chose **cost over latency** for cold pastes. Reading from S3 (50-100ms) is slower than database (10-20ms), but 90% cheaper at scale. Hot pastes are cached anyway.

    ---

    ## API Design

    ### 1. Create Paste

    **Request:**
    ```http
    POST /api/v1/pastes
    Content-Type: application/json

    {
      "content": "print('Hello, World!')",
      "language": "python",              // Optional, auto-detect if not provided
      "expiration": "24h",                // Options: 1h, 24h, 7d, 30d, never
      "custom_alias": "hello-world"      // Optional
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "paste_id": "abc12345",
      "url": "https://pastebin.com/abc12345",
      "created_at": "2026-01-29T10:30:00Z",
      "expires_at": "2026-01-30T10:30:00Z",
      "language": "python",
      "size": 24
    }
    ```

    **Design Notes:**

    - Auto-detect language if not provided (using libraries like Pygments, linguist)
    - Return `201 Created` with paste URL
    - Accept various expiration formats: "1h", "24h", "7d", "30d", "never"
    - Reject pastes > 10 MB (configurable limit)

    ---

    ### 2. Read Paste

    **Request:**
    ```http
    GET /api/v1/pastes/{paste_id}
    Accept: application/json
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "paste_id": "abc12345",
      "content": "print('Hello, World!')",
      "language": "python",
      "created_at": "2026-01-29T10:30:00Z",
      "expires_at": "2026-01-30T10:30:00Z",
      "size": 24
    }
    ```

    **Design Notes:**

    - Return `404 Not Found` if paste expired or doesn't exist
    - Support `Accept: text/plain` for raw text
    - Support `Accept: text/html` for syntax-highlighted HTML

    ---

    ### 3. Get Raw Content

    **Request:**
    ```http
    GET /raw/{paste_id}
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: text/plain

    print('Hello, World!')
    ```

    **Design Notes:**

    - Return plain text without metadata
    - Useful for `curl` users and programmatic access
    - Set `Content-Disposition: attachment` for downloads

    ---

    ## Database Schema

    ### Metadata Table (PostgreSQL/MySQL)

    ```sql
    CREATE TABLE pastes (
        id BIGSERIAL PRIMARY KEY,
        paste_id VARCHAR(10) UNIQUE NOT NULL,
        storage_key VARCHAR(255) NOT NULL,  -- S3 key: pastes/{year}/{month}/{paste_id}
        language VARCHAR(50),
        size_bytes INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        views_count BIGINT DEFAULT 0,
        INDEX idx_paste_id (paste_id),
        INDEX idx_expires_at (expires_at)
    ) PARTITION BY RANGE (created_at);

    -- Partitions (one per month for easy cleanup)
    CREATE TABLE pastes_2026_01
        PARTITION OF pastes
        FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
    ```

    **Why separate metadata from content:**

    - Fast lookups (indexed paste_id)
    - Cheap storage (metadata ~500 bytes, content ~10 KB)
    - Easy expiration management (query by expires_at)
    - Content stored in S3 (cheaper, scalable)

    ---

    ## Data Flow Diagrams

    ### Write Flow (Create Paste)

    ```mermaid
    sequenceDiagram
        participant Client
        participant LB as Load Balancer
        participant API as API Server
        participant DB as Database
        participant S3 as Object Storage
        participant Redis as Redis Cache

        Client->>LB: POST /api/v1/pastes<br/>{content, language}
        LB->>API: Route request

        API->>API: Generate paste_id (random 8 chars)
        API->>API: Compress content (gzip)
        API->>API: Detect language if not provided

        API->>S3: PUT pastes/2026/01/abc12345<br/>(compressed content)
        S3-->>API: Success (storage_key)

        API->>DB: INSERT INTO pastes<br/>(paste_id, storage_key, language, size, expires_at)
        DB-->>API: Success

        API->>Redis: SETEX paste:abc12345 3600<br/>(cache metadata + content)
        Redis-->>API: OK

        API-->>Client: 201 Created<br/>{paste_id, url, expires_at}
    ```

    **Flow Explanation:**

    1. **Generate unique paste_id** - Random 8-character string (62^8 combinations)
    2. **Compress content** - gzip compression (3:1 ratio for text, saves storage costs)
    3. **Detect language** - Auto-detect if not provided (using regex patterns, file extensions)
    4. **Store in S3** - Content stored with key `pastes/{year}/{month}/{paste_id}`
    5. **Save metadata in database** - Fast lookups, expiration tracking
    6. **Cache in Redis** - Pre-populate cache for immediate reads
    7. **Return immediately** - Don't wait for replication or cache propagation

    **Latency:** ~150-200ms (dominated by S3 write + database insert)

    ---

    ### Read Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant CDN
        participant LB as Load Balancer
        participant API as API Server
        participant Redis as Redis Cache
        participant DB as Database
        participant S3 as Object Storage

        Client->>CDN: GET /abc12345

        alt CDN Cache HIT (60-70% of requests)
            CDN-->>Client: HTML page (< 50ms)
        else CDN Cache MISS
            CDN->>LB: Forward request
            LB->>API: Route request

            API->>Redis: GET paste:abc12345

            alt Redis Cache HIT (25-30% of requests)
                Redis-->>API: {content, metadata}
            else Redis Cache MISS (5-10% of requests)
                Redis-->>API: null

                API->>DB: SELECT * FROM pastes WHERE paste_id='abc12345'
                DB-->>API: {storage_key, language, expires_at}

                alt Paste Expired
                    API-->>Client: 404 Not Found
                end

                API->>S3: GET pastes/2026/01/abc12345
                S3-->>API: content (compressed)

                API->>API: Decompress content

                API->>Redis: SETEX paste:abc12345 3600<br/>(cache for 1 hour)
            end

            API-->>CDN: HTML page (with syntax highlighting)
            CDN->>CDN: Cache for 1 hour
            CDN-->>Client: HTML page
        end
    ```

    **Flow Explanation:**

    1. **CDN check first** - 60-70% of requests served from CDN edge (< 50ms)
    2. **Redis cache** - 25-30% of requests hit Redis (< 10ms)
    3. **Database lookup** - Get metadata (storage_key, expiration)
    4. **Check expiration** - Return 404 if paste expired
    5. **S3 fetch** - Retrieve content from object storage (50-100ms)
    6. **Decompress** - gzip decompression (< 1ms for typical paste)
    7. **Cache result** - Store in Redis for future reads
    8. **Return to CDN** - CDN caches for subsequent requests

    **Latency:**
    - CDN hit: < 50ms (60-70% of requests)
    - Redis hit: < 20ms (25-30% of requests)
    - S3 fetch: 100-200ms (5-10% of requests, cold pastes)

    **Why this is efficient:** 90-95% of requests served from cache (CDN + Redis), only 5-10% hit S3.

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section covers four critical components for building Pastebin at scale.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Text Storage** | How to store billions of pastes cost-effectively? | S3 + compression + smart partitioning |
    | **Expiration** | How to delete 50M pastes/day efficiently? | Background workers + partitioning + lazy deletion |
    | **Content Deduplication** | How to avoid storing duplicates? | Content hashing + reference counting |
    | **Syntax Highlighting** | How to highlight code without slowing reads? | Pre-render + cache + client-side library |

    ---

    === "💾 Text Storage Strategy"

        ## The Challenge

        **Problem:** Storing 960 TB of text over 5 years. Database storage costs ~$0.10/GB/month = $96,000/month. Too expensive!

        **Solution:** Use object storage (S3) at ~$0.023/GB/month = $22,080/month. Save 75%!

        **Requirements:**

        - **Cost-effective** - Minimize storage costs
        - **Durable** - 99.999999999% durability (S3 standard)
        - **Scalable** - No capacity planning, auto-scales
        - **Fast reads** - < 100ms p99 latency

        ---

        ## Storage Architecture

        **Two-tier storage:**

        1. **Metadata in database** - Fast lookups, expiration tracking
        2. **Content in S3** - Cost-effective, unlimited scale

        **Why separate:**

        | Aspect | Database | Object Storage |
        |--------|----------|----------------|
        | **Cost** | $0.10/GB/month | $0.023/GB/month (4.3x cheaper) |
        | **Lookup speed** | 10-20ms | 50-100ms |
        | **Scale** | Limited (TBs) | Unlimited (PBs) |
        | **Use case** | Fast indexed lookups | Bulk blob storage |

        ---

        ## S3 Key Structure

        **Hierarchical key design:**

        ```
        pastes/{year}/{month}/{paste_id}

        Examples:
        pastes/2026/01/abc12345
        pastes/2026/01/xyz67890
        pastes/2026/02/def54321
        ```

        **Why this structure:**

        - **Even distribution** - Prevent hot partitions (S3 partitions by prefix)
        - **Easy cleanup** - Delete entire month: `aws s3 rm s3://bucket/pastes/2024/01/ --recursive`
        - **Cost optimization** - Move old pastes to Glacier: `pastes/2024/*`
        - **Debugging** - Easy to find pastes by date

        **Partition considerations:**

        ```python
        # DON'T: All pastes in one prefix (hot partition)
        bad_key = f"pastes/{paste_id}"  # All writes go to same partition

        # DO: Shard by date (even distribution)
        good_key = f"pastes/{year}/{month}/{paste_id}"  # Distributed writes
        ```

        ---

        ## Text Compression

        **Problem:** Text is highly compressible. Storing uncompressed wastes storage.

        **Solution:** gzip compression before storing in S3.

        ```python
        import gzip
        import boto3

        class PasteStorage:
            """Handle paste storage in S3 with compression"""

            def __init__(self, s3_client, bucket_name):
                self.s3 = s3_client
                self.bucket = bucket_name

            def store_paste(self, paste_id: str, content: str) -> str:
                """
                Store paste content in S3 with compression

                Returns:
                    storage_key: S3 key where content is stored
                """
                # Compress content
                compressed = gzip.compress(content.encode('utf-8'))

                # Generate S3 key with date partitioning
                now = datetime.utcnow()
                storage_key = f"pastes/{now.year}/{now.month:02d}/{paste_id}"

                # Upload to S3
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=storage_key,
                    Body=compressed,
                    ContentType='text/plain',
                    ContentEncoding='gzip',
                    Metadata={
                        'original_size': str(len(content)),
                        'compressed_size': str(len(compressed))
                    }
                )

                compression_ratio = len(content) / len(compressed)
                print(f"Compressed {len(content)} bytes to {len(compressed)} bytes "
                      f"(ratio: {compression_ratio:.2f}:1)")

                return storage_key

            def retrieve_paste(self, storage_key: str) -> str:
                """
                Retrieve and decompress paste content from S3

                Returns:
                    content: Original paste content
                """
                # Download from S3
                response = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=storage_key
                )

                # Decompress
                compressed = response['Body'].read()
                content = gzip.decompress(compressed).decode('utf-8')

                return content
        ```

        **Compression analysis:**

        | Content Type | Typical Size | Compressed Size | Ratio | Savings |
        |-------------|--------------|----------------|-------|---------|
        | **Code snippets** | 5 KB | 1.5 KB | 3.3:1 | 70% |
        | **JSON/XML** | 10 KB | 2 KB | 5:1 | 80% |
        | **Logs** | 20 KB | 4 KB | 5:1 | 80% |
        | **Plain text** | 5 KB | 2.5 KB | 2:1 | 50% |
        | **Already compressed** | 10 KB | 10 KB | 1:1 | 0% |

        **Average savings:** ~70% storage reduction

        ---

        ## Storage Lifecycle Policy

        **Problem:** Old pastes (> 1 year) rarely accessed. Paying $0.023/GB for storage we barely use.

        **Solution:** Move old pastes to cheaper storage tiers.

        **S3 Storage Tiers:**

        | Tier | Cost | Retrieval Time | Use Case |
        |------|------|---------------|----------|
        | **S3 Standard** | $0.023/GB/month | < 100ms | Hot pastes (< 30 days old) |
        | **S3 IA** | $0.0125/GB/month | < 100ms | Warm pastes (30-90 days) |
        | **S3 Glacier** | $0.004/GB/month | 3-5 hours | Cold pastes (> 90 days) |
        | **S3 Glacier Deep Archive** | $0.00099/GB/month | 12 hours | Archive (> 1 year) |

        **Lifecycle policy (S3 configuration):**

        ```json
        {
          "Rules": [
            {
              "Id": "MoveOldPastesToIA",
              "Status": "Enabled",
              "Prefix": "pastes/",
              "Transitions": [
                {
                  "Days": 30,
                  "StorageClass": "STANDARD_IA"
                },
                {
                  "Days": 90,
                  "StorageClass": "GLACIER"
                },
                {
                  "Days": 365,
                  "StorageClass": "DEEP_ARCHIVE"
                }
              ]
            }
          ]
        }
        ```

        **Cost savings:**

        ```
        Without lifecycle:
        - 960 TB × $0.023/GB = $22,080/month

        With lifecycle (assuming 50% > 90 days old):
        - 480 TB × $0.023/GB (Standard) = $11,040/month
        - 480 TB × $0.004/GB (Glacier) = $1,920/month
        - Total: $12,960/month
        - Savings: $9,120/month (41% reduction)
        ```

    === "⏰ Expiration Handling"

        ## The Challenge

        **Problem:** With 50M pastes/day and various expiration times, we need to delete ~10M expired pastes/day efficiently.

        **Requirements:**

        - **Don't slow down reads** - Expiration check shouldn't add latency
        - **Clean up storage** - Delete from both database and S3
        - **Batch efficiently** - Don't delete one-by-one (too slow)
        - **Handle failures** - Retry failed deletions

        ---

        ## Expiration Strategy

        **Hybrid approach: Lazy deletion + Background workers**

        ### 1. Lazy Deletion (on read)

        **How it works:** Check expiration when paste is accessed.

        ```python
        def get_paste(paste_id: str):
            """Get paste with expiration check"""

            # Get metadata from cache or database
            metadata = cache.get(f"paste:{paste_id}") or db.get_paste(paste_id)

            if not metadata:
                return None  # Paste doesn't exist

            # Check if expired
            if metadata['expires_at'] and datetime.utcnow() > metadata['expires_at']:
                # Paste expired - return 404
                # Delete asynchronously (don't block request)
                async_delete_paste(paste_id)
                return None

            # Paste valid, retrieve content
            content = get_paste_content(metadata['storage_key'])
            return content
        ```

        **Pros:**

        - No latency impact (check is < 1ms)
        - No background infrastructure needed
        - Works for rarely accessed pastes

        **Cons:**

        - Unused pastes never deleted (waste storage)
        - Database bloat (expired records remain)

        ---

        ### 2. Background Expiration Worker

        **How it works:** Periodic job deletes expired pastes in batches.

        ```python
        class ExpirationWorker:
            """Background worker to delete expired pastes"""

            def __init__(self, db, s3_client, batch_size=1000):
                self.db = db
                self.s3 = s3_client
                self.batch_size = batch_size

            def run(self):
                """Run expiration cleanup job"""
                while True:
                    try:
                        # Find expired pastes (batch)
                        expired_pastes = self.find_expired_pastes()

                        if expired_pastes:
                            self.delete_batch(expired_pastes)

                        # Sleep for 1 minute
                        time.sleep(60)

                    except Exception as e:
                        logger.error(f"Expiration worker error: {e}")
                        time.sleep(60)

            def find_expired_pastes(self) -> List[dict]:
                """
                Find expired pastes using indexed query

                Returns up to batch_size expired pastes
                """
                query = """
                    SELECT paste_id, storage_key
                    FROM pastes
                    WHERE expires_at IS NOT NULL
                      AND expires_at < NOW()
                    ORDER BY expires_at ASC
                    LIMIT %s
                """
                return self.db.query(query, (self.batch_size,))

            def delete_batch(self, expired_pastes: List[dict]):
                """
                Delete batch of expired pastes from DB and S3

                Uses parallel deletion for performance
                """
                paste_ids = [p['paste_id'] for p in expired_pastes]
                storage_keys = [p['storage_key'] for p in expired_pastes]

                # Delete from database (batch)
                delete_query = "DELETE FROM pastes WHERE paste_id = ANY(%s)"
                self.db.execute(delete_query, (paste_ids,))

                # Delete from S3 (batch - up to 1000 per call)
                self.s3.delete_objects(
                    Bucket='pastebin-content',
                    Delete={
                        'Objects': [{'Key': key} for key in storage_keys],
                        'Quiet': True
                    }
                )

                # Invalidate cache
                for paste_id in paste_ids:
                    cache.delete(f"paste:{paste_id}")

                logger.info(f"Deleted {len(expired_pastes)} expired pastes")
        ```

        **Job scheduling:**

        ```python
        # Run every minute
        schedule.every(1).minutes.do(expiration_worker.run)

        # Or use cron: */1 * * * * /usr/bin/python expiration_worker.py
        ```

        **Pros:**

        - Proactive cleanup (don't wait for access)
        - Batch deletion (efficient, 1000 at a time)
        - Reduces database bloat
        - Recovers storage quickly

        **Cons:**

        - Requires background infrastructure
        - Adds system complexity

        ---

        ## Partition-Based Cleanup (Advanced)

        **For very old pastes:** Drop entire database partitions instead of row-by-row deletion.

        ```sql
        -- Create monthly partitions
        CREATE TABLE pastes_2024_01 PARTITION OF pastes
            FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

        -- Drop entire partition (instant, vs DELETE which scans all rows)
        DROP TABLE pastes_2024_01;

        -- Also delete S3 prefix
        aws s3 rm s3://pastebin-content/pastes/2024/01/ --recursive
        ```

        **Benefits:**

        - **Instant deletion** - DROP TABLE is instant, DELETE scans all rows
        - **Reclaim storage immediately** - No VACUUM needed
        - **Batch S3 deletion** - Delete entire prefix in one command

        ---

        ## Expiration Monitoring

        **Metrics to track:**

        | Metric | Target | Alert Threshold |
        |--------|--------|-----------------|
        | **Expired pastes count** | < 1M | > 5M (cleanup falling behind) |
        | **Deletion rate** | > 10K/min | < 5K/min (worker slow) |
        | **Storage reclaimed** | > 100 GB/day | < 50 GB/day (not cleaning enough) |
        | **Failed deletions** | < 0.1% | > 1% (S3/DB errors) |

    === "🔄 Content Deduplication"

        ## The Challenge

        **Problem:** Popular code snippets (e.g., "Hello World", common configs) are pasted thousands of times. Storing duplicates wastes storage.

        **Opportunity:** If 10% of pastes are duplicates, we save ~96 TB (10% of 960 TB).

        **Requirements:**

        - **Detect duplicates** - Identify identical content
        - **Share storage** - Multiple paste_ids point to same content
        - **Maintain independence** - Each paste can have different expiration
        - **Fast lookup** - Don't slow down paste creation

        ---

        ## Deduplication Strategy

        **Content-addressable storage: Use content hash as storage key**

        ### How It Works

        1. **Hash content** - SHA-256 hash of paste content
        2. **Check if exists** - Look up hash in database
        3. **If exists** - Reuse existing storage, increment reference count
        4. **If new** - Store content, set reference count = 1

        ---

        ## Implementation

        ```python
        import hashlib

        class DeduplicatingPasteStorage:
            """Paste storage with content deduplication"""

            def __init__(self, db, s3_client):
                self.db = db
                self.s3 = s3_client

            def create_paste(self, content: str, language: str, expires_at: datetime) -> str:
                """
                Create paste with deduplication

                Returns:
                    paste_id: Unique identifier for this paste
                """
                # Generate content hash
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

                # Check if content already exists
                existing = self.db.get_content_by_hash(content_hash)

                if existing:
                    # Content exists - reuse storage
                    storage_key = existing['storage_key']

                    # Increment reference count
                    self.db.increment_ref_count(content_hash)

                    logger.info(f"Duplicate content found (hash: {content_hash[:8]}...), "
                                f"reusing storage: {storage_key}")
                else:
                    # New content - store in S3
                    storage_key = self.store_in_s3(content)

                    # Record content with ref_count = 1
                    self.db.insert_content(
                        content_hash=content_hash,
                        storage_key=storage_key,
                        size=len(content),
                        ref_count=1
                    )

                # Create paste metadata (always unique)
                paste_id = generate_paste_id()
                self.db.insert_paste(
                    paste_id=paste_id,
                    content_hash=content_hash,
                    language=language,
                    expires_at=expires_at
                )

                return paste_id

            def delete_paste(self, paste_id: str):
                """
                Delete paste with reference counting

                Only deletes S3 content when ref_count reaches 0
                """
                # Get paste metadata
                paste = self.db.get_paste(paste_id)
                content_hash = paste['content_hash']

                # Delete paste metadata
                self.db.delete_paste(paste_id)

                # Decrement reference count
                new_count = self.db.decrement_ref_count(content_hash)

                # If no more references, delete content from S3
                if new_count == 0:
                    content = self.db.get_content_by_hash(content_hash)
                    self.s3.delete_object(
                        Bucket='pastebin-content',
                        Key=content['storage_key']
                    )
                    self.db.delete_content(content_hash)
                    logger.info(f"Deleted content {content_hash[:8]}... from S3 (ref_count = 0)")
                else:
                    logger.info(f"Content {content_hash[:8]}... still has {new_count} references")
        ```

        ---

        ## Database Schema (with deduplication)

        ```sql
        -- Content table (deduplicated storage)
        CREATE TABLE contents (
            content_hash VARCHAR(64) PRIMARY KEY,  -- SHA-256 hash
            storage_key VARCHAR(255) NOT NULL,
            size_bytes INT NOT NULL,
            ref_count INT DEFAULT 1,               -- Number of pastes using this content
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ref_count (ref_count)
        );

        -- Pastes table (metadata only)
        CREATE TABLE pastes (
            id BIGSERIAL PRIMARY KEY,
            paste_id VARCHAR(10) UNIQUE NOT NULL,
            content_hash VARCHAR(64) NOT NULL,     -- Foreign key to contents
            language VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            views_count BIGINT DEFAULT 0,
            INDEX idx_paste_id (paste_id),
            INDEX idx_expires_at (expires_at),
            FOREIGN KEY (content_hash) REFERENCES contents(content_hash)
        );
        ```

        ---

        ## Deduplication Analysis

        **Storage savings:**

        ```python
        # Example: "Hello World" in Python pasted 10,000 times
        duplicate_content = "print('Hello, World!')"  # 24 bytes

        # Without deduplication:
        storage_without = 10000 * 24 bytes = 240 KB

        # With deduplication:
        storage_with = 1 * 24 bytes = 24 bytes
        savings = 99.99%
        ```

        **Real-world deduplication rates:**

        | Paste Type | Dedup Rate | Savings |
        |------------|-----------|---------|
        | **Code templates** | 30-40% | 300-400 GB saved |
        | **Config files** | 20-30% | 200-300 GB saved |
        | **Error logs** | 40-50% | 400-500 GB saved |
        | **General pastes** | 10-15% | 100-150 GB saved |
        | **Overall average** | ~15% | ~150 GB saved |

        **Trade-offs:**

        | Aspect | Pro | Con |
        |--------|-----|-----|
        | **Storage** | Save 10-15% storage (~150 GB) | Extra database table (contents) |
        | **Complexity** | Transparent to users | Reference counting logic |
        | **Performance** | Faster writes (skip S3 if duplicate) | Extra hash computation (< 1ms) |
        | **Deletion** | Must track ref_count | Can't delete content until ref_count = 0 |

        **Recommendation:** Enable deduplication if storage costs > $10K/month. Otherwise, added complexity may not be worth 15% savings.

    === "🎨 Syntax Highlighting"

        ## The Challenge

        **Problem:** Syntax highlighting is expensive (regex parsing, tokenization). Doing it on every read adds 50-100ms latency.

        **Goal:** Provide syntax highlighting without slowing down paste reads.

        **Options:**

        1. **Server-side rendering** - Pre-render HTML with highlighting
        2. **Client-side rendering** - Send raw text, highlight in browser
        3. **Hybrid** - Pre-render + cache, fallback to client-side

        ---

        ## Strategy: Hybrid Approach

        **Pre-render on paste creation, cache result, fallback to client-side for cache misses**

        ### How It Works

        1. **On paste creation:** Render syntax-highlighted HTML, cache it
        2. **On paste read:** Serve cached HTML (fast)
        3. **On cache miss:** Send raw text, let browser highlight (slower but works)

        ---

        ## Implementation

        ### Server-Side Rendering (on create)

        ```python
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.formatters import HtmlFormatter

        class SyntaxHighlighter:
            """Syntax highlighting with caching"""

            def __init__(self, cache):
                self.cache = cache

            def highlight_code(self, content: str, language: str = None) -> dict:
                """
                Generate syntax-highlighted HTML

                Returns:
                    {
                        'html': '<div class="highlight">...',
                        'language': 'python',
                        'css': '.highlight { background: #f8f8f8; }'
                    }
                """
                try:
                    # Get lexer (language parser)
                    if language:
                        lexer = get_lexer_by_name(language, stripall=True)
                    else:
                        # Auto-detect language
                        lexer = guess_lexer(content)
                        language = lexer.name.lower()

                    # Generate HTML
                    formatter = HtmlFormatter(
                        style='github',
                        linenos='table',  # Line numbers
                        cssclass='highlight'
                    )
                    html = highlight(content, lexer, formatter)

                    # Get CSS (cache this, it's the same for all pastes)
                    css = formatter.get_style_defs('.highlight')

                    return {
                        'html': html,
                        'language': language,
                        'css': css
                    }

                except Exception as e:
                    # Fallback: no highlighting
                    logger.warning(f"Syntax highlighting failed: {e}")
                    return {
                        'html': f'<pre>{content}</pre>',
                        'language': 'text',
                        'css': ''
                    }

            def get_highlighted_paste(self, paste_id: str, content: str, language: str) -> str:
                """
                Get highlighted HTML (from cache or generate)

                Returns:
                    html: Syntax-highlighted HTML
                """
                # Check cache
                cache_key = f"paste:html:{paste_id}"
                cached_html = self.cache.get(cache_key)

                if cached_html:
                    return cached_html

                # Cache miss - generate highlighting
                result = self.highlight_code(content, language)
                html = result['html']

                # Cache for 24 hours
                self.cache.setex(cache_key, 86400, html)

                return html
        ```

        ### Client-Side Rendering (fallback)

        ```html
        <!-- If server-side rendering unavailable, use Prism.js -->
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism.css" rel="stylesheet" />
        </head>
        <body>
            <pre><code class="language-python">
                print('Hello, World!')
            </code></pre>

            <script src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.js"></script>
        </body>
        </html>
        ```

        **Prism.js benefits:**

        - **Fast** - Highlights in < 10ms in browser
        - **Small** - ~2 KB gzipped
        - **200+ languages** - Covers most use cases
        - **No server load** - Offload work to client

        ---

        ## Language Detection

        **Problem:** Users often don't specify language. Auto-detection needed.

        **Approach: Heuristic-based detection**

        ```python
        def detect_language(content: str, filename: str = None) -> str:
            """
            Auto-detect programming language

            Uses multiple signals:
            1. File extension (if provided)
            2. Shebang line (#!/usr/bin/python)
            3. Content patterns (keywords, syntax)
            """

            # 1. Check file extension
            if filename:
                ext_map = {
                    '.py': 'python',
                    '.js': 'javascript',
                    '.java': 'java',
                    '.cpp': 'cpp',
                    '.c': 'c',
                    '.go': 'go',
                    '.rs': 'rust',
                    '.rb': 'ruby',
                    '.php': 'php',
                    '.sh': 'bash',
                    # ... 100+ extensions
                }
                ext = os.path.splitext(filename)[1]
                if ext in ext_map:
                    return ext_map[ext]

            # 2. Check shebang
            if content.startswith('#!'):
                first_line = content.split('\n')[0]
                if 'python' in first_line:
                    return 'python'
                elif 'node' in first_line or 'nodejs' in first_line:
                    return 'javascript'
                elif 'bash' in first_line or 'sh' in first_line:
                    return 'bash'

            # 3. Pattern matching (keywords, syntax)
            patterns = {
                'python': [r'\bdef\s+\w+\(', r'\bimport\s+\w+', r'\bclass\s+\w+:'],
                'javascript': [r'\bfunction\s+\w+\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*='],
                'java': [r'\bpublic\s+class\s+\w+', r'\bpublic\s+static\s+void\s+main'],
                'go': [r'\bpackage\s+main', r'\bfunc\s+\w+\(', r'\bimport\s+\('],
                # ... more patterns
            }

            for lang, lang_patterns in patterns.items():
                for pattern in lang_patterns:
                    if re.search(pattern, content):
                        return lang

            # 4. Fallback: use Pygments guess_lexer
            try:
                lexer = guess_lexer(content)
                return lexer.name.lower()
            except:
                return 'text'  # Default to plain text
        ```

        **Accuracy:**

        - File extension: 95% accurate
        - Shebang: 100% accurate (when present)
        - Pattern matching: 80-90% accurate
        - Pygments guess_lexer: 70-85% accurate
        - **Overall: ~85% accurate**

        ---

        ## Performance Comparison

        | Approach | Latency | Cache Hit Rate | Cost |
        |----------|---------|---------------|------|
        | **Server-side (cached)** | 5-10ms | 95% | CPU cost (on create) |
        | **Server-side (uncached)** | 50-100ms | N/A | High CPU cost |
        | **Client-side (Prism.js)** | 10-20ms | N/A (CDN cached) | Zero server cost |

        **Recommendation:** Use server-side with caching. Fallback to client-side if cache miss or old paste.

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    This section covers scaling from prototype (1K req/sec) to production (17.4K req/sec peak).

    **Scaling philosophy:**

    1. **Measure first** - Profile to find bottlenecks
    2. **Cache aggressively** - CDN + Redis for hot pastes
    3. **Optimize storage** - Compression + lifecycle policies
    4. **Horizontal scaling** - Add more API servers

    ---

    ## Bottleneck Identification

    | Component | Current Capacity | Required Capacity | Bottleneck? | Solution |
    |-----------|-----------------|------------------|------------|----------|
    | **API Servers** | 1K req/sec per server | 17.4K req/sec | ✅ Easily scalable | Add 20-30 servers |
    | **Load Balancer** | 100K req/sec | 17.4K req/sec | ❌ Not a bottleneck | Single LB sufficient |
    | **Redis Cache** | 50K req/sec per instance | 17.4K req/sec | ❌ Not a bottleneck | 2-3 instances sufficient |
    | **Database** | 10K read QPS | 870 QPS (5% miss rate) | ❌ Not a bottleneck | 1-2 read replicas sufficient |
    | **S3** | Unlimited | Any | ❌ Not a bottleneck | Auto-scales |
    | **CDN** | 100K req/sec | 12K req/sec (70%) | ❌ Not a bottleneck | CloudFront handles this easily |

    **No critical bottlenecks at 17.4K QPS scale.** System can handle 3x peak (50K QPS) with current architecture.

    ---

    ## CDN Optimization

    **Goal:** Serve 70-80% of requests from CDN edge locations (< 50ms latency)

    ### CDN Configuration

    ```javascript
    // CloudFront cache behavior
    {
      "PathPattern": "/pastes/*",
      "TargetOriginId": "pastebin-origin",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": ["GET", "HEAD"],
      "CachedMethods": ["GET", "HEAD"],
      "Compress": true,
      "MinTTL": 0,
      "DefaultTTL": 3600,      // Cache for 1 hour
      "MaxTTL": 86400,         // Max 24 hours
      "ForwardedValues": {
        "QueryString": false,
        "Headers": ["Accept", "Accept-Encoding"],
        "Cookies": {
          "Forward": "none"
        }
      }
    }
    ```

    **Cache control headers (from API):**

    ```python
    @app.get("/pastes/{paste_id}")
    def get_paste(paste_id: str):
        paste = fetch_paste(paste_id)

        # Set cache headers
        headers = {
            'Cache-Control': 'public, max-age=3600, s-maxage=86400',
            'ETag': f'"{paste["paste_id"]}-{paste["created_at"]}"',
            'Vary': 'Accept-Encoding'
        }

        return Response(paste['html'], headers=headers)
    ```

    **Cache invalidation (when paste deleted):**

    ```python
    def delete_paste(paste_id: str):
        # Delete from database and S3
        db.delete_paste(paste_id)
        s3.delete_object(storage_key)

        # Invalidate CDN cache
        cloudfront.create_invalidation(
            DistributionId='E1234567890ABC',
            InvalidationBatch={
                'Paths': {
                    'Quantity': 1,
                    'Items': [f'/pastes/{paste_id}']
                },
                'CallerReference': str(time.time())
            }
        )
    ```

    **CDN benefits:**

    - **Latency:** 50ms (CDN edge) vs 200ms (origin)
    - **Origin load:** Reduced by 70-80%
    - **Cost:** $0.085/GB (CDN) vs $0.10/GB (origin bandwidth)
    - **Global reach:** 200+ edge locations worldwide

    ---

    ## Database Optimization

    ### Connection Pooling

    ```python
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    # Efficient connection pool
    engine = create_engine(
        'postgresql://user:pass@db-host/pastebin',
        poolclass=QueuePool,
        pool_size=20,           # Keep 20 connections per API server
        max_overflow=10,        # Allow 10 extra under load
        pool_timeout=30,
        pool_recycle=3600,      # Recycle connections every hour
        pool_pre_ping=True      # Check connection health
    )
    ```

    ### Indexing Strategy

    ```sql
    -- Critical indexes
    CREATE UNIQUE INDEX idx_paste_id ON pastes(paste_id);
    CREATE INDEX idx_expires_at ON pastes(expires_at) WHERE expires_at IS NOT NULL;
    CREATE INDEX idx_content_hash ON pastes(content_hash);  -- For deduplication

    -- Monitoring index usage
    SELECT
        schemaname,
        tablename,
        indexname,
        idx_scan,              -- Times index was scanned
        idx_tup_read,          -- Tuples read from index
        idx_tup_fetch          -- Tuples fetched using index
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    ORDER BY idx_scan DESC;
    ```

    ### Query Optimization

    ```sql
    -- BAD: Full table scan
    SELECT * FROM pastes WHERE created_at > NOW() - INTERVAL '7 days';

    -- GOOD: Use partition pruning
    SELECT * FROM pastes_2026_01 WHERE created_at > '2026-01-22';

    -- BAD: Slow count
    SELECT COUNT(*) FROM pastes WHERE expires_at < NOW();

    -- GOOD: Approximate count (fast)
    SELECT reltuples::BIGINT AS estimate
    FROM pg_class
    WHERE relname = 'pastes';
    ```

    ---

    ## Storage Optimization

    ### Text Compression (detailed)

    ```python
    import gzip
    import zlib

    # Compression comparison
    content = "print('Hello, World!')" * 100  # 2,400 bytes

    # gzip (default)
    gzip_compressed = gzip.compress(content.encode())
    gzip_ratio = len(content) / len(gzip_compressed)
    # Size: 43 bytes, Ratio: 55.8:1

    # zlib (slightly better compression)
    zlib_compressed = zlib.compress(content.encode(), level=9)
    zlib_ratio = len(content) / len(zlib_compressed)
    # Size: 39 bytes, Ratio: 61.5:1

    # Decompression speed
    import time

    start = time.time()
    gzip.decompress(gzip_compressed)
    gzip_time = time.time() - start  # ~0.05ms

    start = time.time()
    zlib.decompress(zlib_compressed)
    zlib_time = time.time() - start  # ~0.04ms
    ```

    **Recommendation:** Use gzip (better compatibility, S3 native support)

    ---

    ## Monitoring & Alerting

    **Key metrics to track:**

    | Metric | Target | Alert Threshold | Why It Matters |
    |--------|--------|-----------------|----------------|
    | **Read Latency (P50)** | < 50ms | > 100ms | User experience |
    | **Read Latency (P99)** | < 200ms | > 500ms | Tail latency affects some users |
    | **CDN Hit Rate** | > 70% | < 60% | Low hit rate = high origin load |
    | **Redis Hit Rate** | > 90% | < 80% | Cache warming issues |
    | **S3 Latency** | < 100ms | > 200ms | Storage performance degradation |
    | **Expiration Lag** | < 1M expired pastes | > 5M | Cleanup falling behind |
    | **Error Rate** | < 0.1% | > 1% | System health |
    | **Storage Used** | Monitoring | 80% capacity | Need to scale storage |

    **Dashboards:**

    - **Real-time:** Grafana dashboard with 1-min granularity
    - **Historical:** Weekly/monthly trends for capacity planning
    - **Alerting:** PagerDuty for P0, Slack for P1/P2

    ---

    ## Cost Optimization

    **Monthly cost breakdown (at 50M pastes/day):**

    | Component | Cost | Optimization |
    |-----------|------|-------------|
    | **S3 Storage** | $12,960 (320 TB × $0.023/GB × 1.75 lifecycle factor) | Use lifecycle policies, compression |
    | **S3 Requests** | $250 (50M PUT + 500M GET) | Cache aggressively, batch operations |
    | **CDN** | $8,500 (100 TB transfer × $0.085/GB) | Already optimized |
    | **EC2 (API servers)** | $4,320 (30 × m5.large × $6/day) | Use spot instances (60% savings) |
    | **RDS** | $2,160 (db.m5.2xlarge × $72/day) | Use read replicas, not bigger instance |
    | **ElastiCache (Redis)** | $1,080 (3 × cache.m5.large × $12/day) | Right-sized for workload |
    | **Data transfer** | $1,000 | Use CloudFront, compress responses |
    | **Total** | **$30,270/month** | |

    **Optimization opportunities:**

    - **Use spot instances:** Save $2,592/month (60% on EC2)
    - **Reserved instances:** Save $650/month (15% on RDS)
    - **S3 Intelligent-Tiering:** Save $1,944/month (automatic lifecycle)
    - **CDN optimization:** Already efficient
    - **Potential savings:** $5,186/month (17% reduction)
    - **Optimized cost:** $25,084/month

=== "📝 Summary & Tips"

    ## Architecture Summary

    **Core Components:**

    | Component | Purpose | Technology | Quantity |
    |-----------|---------|------------|----------|
    | **CDN** | Cache pastes close to users, 70% hit rate | CloudFront, Cloudflare | 1 distribution |
    | **Load Balancer** | Distribute traffic, health checks | nginx, AWS ALB | 1 (+ backup) |
    | **API Servers** | Stateless, horizontally scalable | Python/Flask, Go | 20-30 servers |
    | **Redis Cache** | Metadata + hot pastes, 90% hit rate | Redis Cluster | 3 nodes |
    | **Database** | Metadata storage, expiration tracking | PostgreSQL | 1 primary + 2 replicas |
    | **Object Storage** | Cost-effective paste content storage | S3, Azure Blob | 1 bucket |
    | **Background Workers** | Delete expired pastes, cleanup | Python scripts | 2-3 workers |

    ---

    ## Capacity Handled

    | Metric | Capacity | Headroom |
    |--------|----------|----------|
    | **Paste creations** | 50M pastes/day (~580 QPS) | 10x capacity (S3 unlimited) |
    | **Paste reads** | 500M reads/day (~5.8K QPS) | 5x capacity (CDN scales easily) |
    | **Storage** | 320 TB (compressed, with lifecycle) | Cost-optimized with S3 tiers |
    | **Cache** | 50 GB for hot pastes | 5M pastes cached, 90% hit rate |
    | **Latency** | < 50ms p50 (CDN), < 200ms p99 | 70% CDN hit, 25% Redis hit |

    ---

    ## Key Design Decisions

    1. **Object storage for content** - 75% cost savings vs database ($22K vs $96K/month)
    2. **Aggressive caching (CDN + Redis)** - 95% of reads served from cache
    3. **Text compression (gzip)** - 70% storage reduction (3:1 ratio)
    4. **Content deduplication** - 15% additional savings for common pastes
    5. **Background expiration workers** - Proactive cleanup, batch deletion
    6. **Lifecycle policies** - Move old pastes to cheaper storage (Glacier)

    ---

    ## Interview Tips

    ### What Interviewers Look For

    ✅ **Start with requirements** - Functional, non-functional, capacity estimation

    ✅ **Discuss storage trade-offs** - Database vs object storage, cost vs latency

    ✅ **Address expiration** - Lazy deletion vs background workers

    ✅ **Consider optimizations** - Compression, deduplication, CDN

    ✅ **Think about scale** - Partitioning, caching strategy, cost

    ✅ **Mention monitoring** - Key metrics, alerting thresholds

    ---

    ### Common Follow-up Questions

    | Question | Key Points to Cover |
    |----------|-------------------|
    | **"Why use S3 instead of database?"** | 75% cost savings ($0.023 vs $0.10/GB), unlimited scale, durability. Trade-off: 50ms slower for cold reads |
    | **"How do you handle paste expiration?"** | Hybrid: lazy deletion (on read) + background workers (batch cleanup). Partition-based for old data (DROP TABLE) |
    | **"What if S3 goes down?"** | Graceful degradation: serve cached pastes, return 503 for new pastes. S3 has 99.99% SLA, rarely down |
    | **"How to prevent spam/abuse?"** | Rate limiting (10 pastes/min per IP), CAPTCHA for suspicious IPs, size limits (10 MB), content filtering |
    | **"How to add user accounts?"** | Add users table, paste ownership, authentication. Enable edit/delete, private pastes, usage quotas |
    | **"How to handle large pastes (100 MB)?"** | Chunked upload, streaming from S3, consider separate storage tier (Glacier for archives) |
    | **"How to make pastes searchable?"** | Add Elasticsearch, index content + metadata. Trade-off: high cost, complexity. Most pastes aren't searched |

    ---

    ### Things to Mention

    **Storage Strategy:**

    - Object storage (S3) for cost efficiency
    - Compression (gzip) for 3:1 savings
    - Lifecycle policies for old pastes (Glacier)
    - Deduplication for common content (15% savings)

    **Caching Strategy:**

    - Multi-layer: CDN (70%) + Redis (25%) + S3 (5%)
    - Cache invalidation on delete
    - CDN for global reach, Redis for hot metadata

    **Expiration:**

    - Hybrid: lazy deletion + background workers
    - Batch deletion (1000 at a time)
    - Partition-based cleanup (DROP TABLE)

    **Monitoring:**

    - Track latency percentiles, cache hit rates
    - Alert on expiration lag, error rates
    - Cost monitoring for storage growth

    ---

    ## Related Problems

    | Problem | Similarity | Key Differences |
    |---------|------------|-----------------|
    | **URL Shortener** | Very similar - short IDs, expiration, caching | Smaller content (URL vs text), higher read ratio (100:1 vs 10:1) |
    | **GitHub Gist** | Extension of Pastebin | Version control (git), multiple files per gist, syntax highlighting, comments |
    | **Image Upload Service** | Similar storage strategy | Binary storage (images vs text), CDN critical, thumbnails, metadata extraction |
    | **File Upload Service** | Related - storage + expiration | Larger files (GBs), chunked upload, resumable uploads, virus scanning |
    | **Google Docs** | Similar text storage | Real-time collaboration (CRDT), rich formatting, editing, permissions |

    ---

    ## Next Steps

    **After mastering Pastebin:**

    1. **URL Shortener** - Simpler version with URL storage and redirects
    2. **Image Upload Service** - Binary storage, CDN, thumbnails
    3. **Google Docs** - Real-time collaboration, CRDT, operational transforms
    4. **File Upload Service** - Large files, chunked upload, resumable uploads

    **Practice variations:**

    - Add user authentication and paste management
    - Implement paste forking and versioning
    - Add collaborative editing (real-time)
    - Support multiple files per paste (like GitHub Gist)
    - Add paste search (Elasticsearch integration)
    - Implement paste templates (boilerplates)

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Amazon, Google, Microsoft, Dropbox, GitHub

---

*This template applies to similar problems: URL Shortener, Image Upload Service, File Upload Service*
