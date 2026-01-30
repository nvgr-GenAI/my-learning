# Design Instagram

A photo-sharing social network where users can upload photos, follow other users, and view a personalized feed of photos from people they follow.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 500M daily active users, 100M photos/day, 4.5B likes/day |
| **Key Challenges** | Feed generation, image storage, real-time notifications, global scale |
| **Core Concepts** | News feed algorithm, CDN, sharding, fan-out on write, object storage |
| **Companies** | Meta, Google, Amazon, Netflix, Twitter, TikTok |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Upload Photos** | Users can upload photos with captions | P0 (Must have) |
    | **View Feed** | Personalized feed of photos from followed users | P0 (Must have) |
    | **Follow/Unfollow** | Users can follow/unfollow others | P0 (Must have) |
    | **Like Photos** | Users can like photos | P0 (Must have) |
    | **Comments** | Users can comment on photos | P1 (Should have) |
    | **Notifications** | Notify users of likes, comments, follows | P1 (Should have) |
    | **Search** | Search users, hashtags | P2 (Nice to have) |
    | **Stories** | 24-hour ephemeral content | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Direct messaging
    - Video uploads (focus on photos)
    - Live streaming
    - Shopping/e-commerce
    - Reels/short-form video

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Social network, users expect 24/7 access |
    | **Latency (Feed)** | < 200ms p95 | Fast feed loading critical for engagement |
    | **Latency (Upload)** | < 3s | Acceptable delay for upload processing |
    | **Consistency** | Eventual consistency | Brief inconsistency acceptable (likes may lag) |
    | **Scalability** | Billions of users | Must scale globally |
    | **Durability** | No photo loss | Photos are permanent, critical user data |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 500M
    Monthly Active Users (MAU): 1B

    Photo uploads:
    - Assumption: 20% of DAU upload photos
    - Photos per upload: 1-10 (average 2)
    - Daily uploads: 500M × 20% × 2 = 200M photos/day
    - Upload QPS: 200M / 86,400 = ~2,300 uploads/sec

    Feed views:
    - Assumption: Each DAU views feed 10 times/day
    - Daily feed views: 500M × 10 = 5B views/day
    - Feed QPS: 5B / 86,400 = ~58K feed requests/sec

    Likes:
    - Assumption: Average 15 likes per DAU per day
    - Daily likes: 500M × 15 = 7.5B likes/day
    - Like QPS: 7.5B / 86,400 = ~87K likes/sec

    Total Read QPS: ~145K (feed + profile views)
    Total Write QPS: ~90K (uploads + likes + comments)
    ```

    ### Storage Estimates

    ```
    Photo storage:
    - Original: ~2 MB per photo (average, JPEG compressed)
    - Thumbnails: 200 KB, 50 KB, 10 KB (3 sizes)
    - Total per photo: 2 MB + 260 KB ≈ 2.3 MB

    For 10 years:
    - Photos: 200M/day × 365 × 10 = 730B photos
    - Storage: 730B × 2.3 MB = 1,679 PB (1.7 exabytes)

    Metadata storage:
    - Per photo: ~2 KB (photo_id, user_id, caption, timestamp, location)
    - Total: 730B × 2 KB = 1.46 PB

    User data:
    - 1B users × 10 KB (profile, settings, followers) = 10 TB

    Total: ~1,680 PB (photos) + 1.5 PB (metadata) + 10 TB (users) ≈ 1.68 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Upload bandwidth:
    - 2,300 uploads/sec × 2 MB = 4.6 GB/sec = 37 Gbps

    Feed bandwidth (download):
    - Assumption: 20 photos per feed load
    - 58K feed requests/sec × 20 photos × 200 KB (thumbnail) = 232 GB/sec = 1.86 Tbps

    Total ingress: ~37 Gbps
    Total egress: ~1.86 Tbps (CDN critical for this scale)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot photos (last 24 hours):
    - Photos: 200M photos × 200 KB (thumbnail) = 40 TB
    - Metadata: 200M × 2 KB = 400 GB
    - Total: ~40 TB for hot photo cache

    User sessions:
    - Active sessions: 50M concurrent users
    - Session data: 50M × 10 KB = 500 GB

    Feed cache:
    - Cache feeds for 10M most active users
    - 10M × 100 photos × 100 bytes (photo_id, metadata) = 100 GB

    Total cache: 40 TB (photos) + 500 GB (sessions) + 100 GB (feeds) ≈ 41 TB
    ```

    ---

    ## Key Assumptions

    1. Focus on core features: upload, feed, follow, like
    2. 500M DAU, 1B MAU (50% daily engagement)
    3. Read-heavy: 145K read QPS vs 90K write QPS (~1.6:1 ratio)
    4. Average photo size: 2 MB original, multiple thumbnails
    5. Eventual consistency acceptable (likes, follower counts can lag)
    6. Global user base, need CDN and multi-region deployment

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separate image storage from metadata** - Photos in object storage (S3/CDN), metadata in database
    2. **Fan-out on write** - Pre-compute feeds when photo uploaded (push model)
    3. **Graph database for social graph** - Efficient follower/following queries
    4. **Eventual consistency** - Prioritize availability over strong consistency
    5. **CDN for global reach** - Serve images from edge locations worldwide

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App<br/>iOS/Android]
            Web[Web Browser]
        end

        subgraph "CDN & Load Balancing"
            CDN[CDN<br/>CloudFront<br/>Image delivery]
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Upload_API[Upload Service<br/>Photo upload]
            Feed_API[Feed Service<br/>Timeline generation]
            Social_API[Social Service<br/>Follow/like]
            Notification_API[Notification Service<br/>Push notifications]
        end

        subgraph "Data Processing"
            Image_Proc[Image Processing<br/>Resize, compress]
            Feed_Worker[Feed Fanout Worker<br/>Pre-compute feeds]
            Analytics[Analytics Pipeline<br/>Kafka/Spark]
        end

        subgraph "Caching"
            Redis_Feed[Redis<br/>Feed cache]
            Redis_Photo[Redis<br/>Photo metadata]
            Redis_User[Redis<br/>User sessions]
        end

        subgraph "Storage"
            Photo_DB[(Photo Metadata DB<br/>Cassandra<br/>Sharded)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Graph_DB[(Social Graph DB<br/>Neo4j/Graph)]
            S3[Object Storage<br/>S3/Blob<br/>Photos]
        end

        subgraph "Message Queue"
            Queue[Message Queue<br/>Kafka/RabbitMQ]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        CDN --> S3

        LB --> Upload_API
        LB --> Feed_API
        LB --> Social_API
        LB --> Notification_API

        Upload_API --> Queue
        Upload_API --> Photo_DB
        Upload_API --> S3

        Queue --> Image_Proc
        Queue --> Feed_Worker
        Queue --> Analytics

        Image_Proc --> S3
        Feed_Worker --> Redis_Feed
        Feed_Worker --> Photo_DB
        Feed_Worker --> Graph_DB

        Feed_API --> Redis_Feed
        Feed_API --> Photo_DB
        Social_API --> Graph_DB
        Social_API --> Redis_User

        Feed_API --> User_DB
        Social_API --> User_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Feed fill:#fff4e1
        style Redis_Photo fill:#fff4e1
        style Redis_User fill:#fff4e1
        style Photo_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Graph_DB fill:#e1f5e1
        style S3 fill:#f3e5f5
        style Queue fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **CDN** | Global image delivery, < 100ms latency worldwide | Direct S3 (slow, expensive egress $0.09/GB) |
    | **Object Storage (S3)** | Unlimited photo storage, durable (99.999999999%) | Database BLOBs (expensive, doesn't scale) |
    | **Cassandra (Photo DB)** | Write-heavy workload (200M photos/day), horizontal scaling | MySQL (can't handle write volume, single leader bottleneck) |
    | **Graph Database** | Fast follower queries (2-3 hops), social graph traversal | SQL joins (too slow for complex queries) |
    | **Message Queue (Kafka)** | Decouple upload from feed generation, reliable async processing | Synchronous processing (slow uploads, no retry) |
    | **Redis Cache** | Fast feed reads (< 10ms), reduce database load | No cache (database can't handle 145K read QPS) |

    **Key Trade-off:** We chose **eventual consistency** over strong consistency. Newsfeed may lag by a few seconds, but system remains available and fast. Users tolerate brief delays for likes/followers.

    ---

    ## API Design

    ### 1. Upload Photo

    **Request:**
    ```http
    POST /api/v1/photos
    Content-Type: multipart/form-data
    Authorization: Bearer <token>

    photo: <binary_data>
    caption: "Beautiful sunset 🌅"
    location: "San Francisco, CA"
    tags: ["#sunset", "#nature"]
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "photo_id": "7f8a3b2c",
      "user_id": "user123",
      "url": "https://cdn.instagram.com/p/7f8a3b2c",
      "thumbnail_urls": {
        "small": "https://cdn.instagram.com/p/7f8a3b2c/t",
        "medium": "https://cdn.instagram.com/p/7f8a3b2c/m",
        "large": "https://cdn.instagram.com/p/7f8a3b2c/l"
      },
      "created_at": "2026-01-29T10:30:00Z",
      "status": "processing"
    }
    ```

    **Design Notes:**

    - Async processing: return immediately, process image in background
    - Generate multiple thumbnails: small (150x150), medium (640x640), large (1080x1080)
    - Upload to S3, then trigger fan-out to followers' feeds
    - Rate limit: 100 uploads per hour per user

    ---

    ### 2. Get Feed (Timeline)

    **Request:**
    ```http
    GET /api/v1/feed?page=1&count=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "photos": [
        {
          "photo_id": "7f8a3b2c",
          "user_id": "user456",
          "username": "john_doe",
          "profile_pic": "https://cdn.instagram.com/u/user456/profile.jpg",
          "url": "https://cdn.instagram.com/p/7f8a3b2c/l",
          "caption": "Beautiful sunset 🌅",
          "created_at": "2026-01-29T10:30:00Z",
          "likes_count": 1542,
          "comments_count": 87,
          "liked_by_user": false
        },
        // ... 19 more photos
      ],
      "next_page": 2,
      "has_more": true
    }
    ```

    **Design Notes:**

    - Pre-computed feed (fan-out on write)
    - Paginated: 20 photos per page
    - Include metadata: username, profile pic, counts
    - Cache feed in Redis for fast access

    ---

    ### 3. Like Photo

    **Request:**
    ```http
    POST /api/v1/photos/{photo_id}/like
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "photo_id": "7f8a3b2c",
      "likes_count": 1543,
      "liked_by_user": true
    }
    ```

    **Design Notes:**

    - Async processing: increment count in background
    - Send notification to photo owner
    - Handle duplicate likes (idempotent)

    ---

    ### 4. Follow User

    **Request:**
    ```http
    POST /api/v1/users/{user_id}/follow
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "user_id": "user456",
      "following": true,
      "follower_count": 1234,
      "following_count": 567
    }
    ```

    **Design Notes:**

    - Update social graph database
    - Send notification to followed user
    - Trigger feed backfill for new follower

    ---

    ## Database Schema

    ### Photo Metadata (Cassandra)

    ```sql
    -- Photos table (Cassandra)
    CREATE TABLE photos (
        photo_id UUID PRIMARY KEY,
        user_id UUID,
        caption TEXT,
        location TEXT,
        tags LIST<TEXT>,
        created_at TIMESTAMP,
        likes_count COUNTER,
        comments_count COUNTER,
        storage_path TEXT,  -- S3 key
        thumbnail_paths MAP<TEXT, TEXT>,  -- {small: s3_key, medium: s3_key}
        -- Partition by photo_id for even distribution
    );

    -- User photos index (for profile view)
    CREATE TABLE user_photos (
        user_id UUID,
        created_at TIMESTAMP,
        photo_id UUID,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Photo feed (fan-out on write)
    CREATE TABLE user_feed (
        user_id UUID,
        created_at TIMESTAMP,
        photo_id UUID,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);
    ```

    **Why Cassandra:**

    - **Write-heavy:** 200M photos/day + 7.5B likes/day
    - **Horizontal scaling:** Add nodes without downtime
    - **No single point of failure:** Multi-master replication
    - **Time-series friendly:** Efficient range queries on created_at

    ---

    ### User Data (PostgreSQL)

    ```sql
    -- Users table (PostgreSQL, sharded by user_id)
    CREATE TABLE users (
        user_id UUID PRIMARY KEY,
        username VARCHAR(30) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        full_name VARCHAR(100),
        bio TEXT,
        profile_pic_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        follower_count INT DEFAULT 0,
        following_count INT DEFAULT 0,
        photo_count INT DEFAULT 0,
        INDEX idx_username (username),
        INDEX idx_email (email)
    ) PARTITION BY HASH (user_id);
    ```

    **Why PostgreSQL (sharded):**

    - **User data is relational:** Authentication, profile, settings
    - **ACID guarantees:** Critical for auth, prevent duplicate usernames
    - **Proven at scale:** Instagram uses PostgreSQL sharded 1000+ ways

    ---

    ### Social Graph (Neo4j or custom graph store)

    ```cypher
    // Neo4j schema
    CREATE (u:User {user_id: 'user123', username: 'john_doe'})
    CREATE (u2:User {user_id: 'user456', username: 'jane_smith'})
    CREATE (u)-[:FOLLOWS {created_at: 1643712000}]->(u2)

    // Query: Get followers
    MATCH (follower:User)-[:FOLLOWS]->(user:User {user_id: 'user123'})
    RETURN follower.user_id, follower.username

    // Query: Get following
    MATCH (user:User {user_id: 'user123'})-[:FOLLOWS]->(following:User)
    RETURN following.user_id, following.username

    // Query: Check if follows
    MATCH (u1:User {user_id: 'user123'})-[:FOLLOWS]->(u2:User {user_id: 'user456'})
    RETURN COUNT(*) > 0 as follows
    ```

    **Why graph database:**

    - **Fast graph traversals:** Get followers of followers (2-3 hops) in < 10ms
    - **Social graph queries:** "People you may know" (mutual friends)
    - **Path finding:** "How are you connected to X?"

    ---

    ## Data Flow Diagrams

    ### Photo Upload Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Upload_API
        participant S3
        participant Photo_DB
        participant Queue
        participant Image_Worker
        participant Feed_Worker

        Client->>Upload_API: POST /api/v1/photos<br/>(multipart upload)
        Upload_API->>Upload_API: Validate auth, rate limit
        Upload_API->>S3: Upload original photo
        S3-->>Upload_API: S3 URL

        Upload_API->>Photo_DB: INSERT photo metadata
        Photo_DB-->>Upload_API: Success

        Upload_API->>Queue: Publish photo_uploaded event
        Upload_API-->>Client: 201 Created (photo_id, status: processing)

        Queue->>Image_Worker: Process image
        Image_Worker->>Image_Worker: Resize (3 thumbnails)
        Image_Worker->>S3: Upload thumbnails
        Image_Worker->>Photo_DB: Update thumbnail URLs

        Queue->>Feed_Worker: Fan-out to followers
        Feed_Worker->>Graph_DB: Get followers (10K followers)
        Feed_Worker->>Feed_Worker: Batch followers (1000 per batch)
        loop For each batch
            Feed_Worker->>Photo_DB: INSERT into user_feed (1000 users)
        end

        Feed_Worker->>Queue: Send notifications (new photo)
    ```

    **Flow Explanation:**

    1. **Upload original** - Store in S3, return immediately (don't wait for processing)
    2. **Save metadata** - Photo info in Cassandra for fast queries
    3. **Async image processing** - Resize to 3 thumbnail sizes in background
    4. **Fan-out to followers** - Add photo to each follower's feed (pre-compute)
    5. **Send notifications** - Push notifications to followers (async)

    **Latency:**
    - User sees response: < 2s (upload + metadata save)
    - Image processing: 5-10s (background)
    - Feed fan-out: 10-30s (background, depends on follower count)

    ---

    ### Feed Generation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Feed_API
        participant Redis
        participant Photo_DB
        participant Graph_DB

        Client->>Feed_API: GET /api/v1/feed
        Feed_API->>Feed_API: Authenticate user

        Feed_API->>Redis: GET feed:user123
        alt Cache HIT (90% of requests)
            Redis-->>Feed_API: Cached feed (photo_ids)
            Feed_API->>Photo_DB: Batch get photo metadata
            Photo_DB-->>Feed_API: Photo details
        else Cache MISS (10% of requests)
            Redis-->>Feed_API: null

            Feed_API->>Photo_DB: SELECT from user_feed WHERE user_id=user123<br/>ORDER BY created_at DESC LIMIT 100
            Photo_DB-->>Feed_API: Photo IDs

            Feed_API->>Photo_DB: Batch get photo metadata
            Photo_DB-->>Feed_API: Photo details

            Feed_API->>Redis: SET feed:user123 (TTL: 5min)
        end

        Feed_API->>Feed_API: Rank photos (ML algorithm)
        Feed_API-->>Client: 200 OK (20 photos)
    ```

    **Flow Explanation:**

    1. **Check Redis cache** - 90% of feeds served from cache (< 10ms)
    2. **Cache miss** - Query pre-computed feed from Cassandra (user_feed table)
    3. **Batch fetch metadata** - Get photo details, user info in one query
    4. **Rank photos** - Apply ML ranking (engagement-based)
    5. **Return feed** - 20 photos per page, paginated

    **Latency:**
    - Cache hit: < 50ms
    - Cache miss: 100-200ms (database query)

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Instagram subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **News Feed Generation** | How to show relevant photos to 500M users? | Fan-out on write + ML ranking |
    | **Image Storage** | How to store 1.68 exabytes of photos? | S3 + CDN + multiple thumbnails |
    | **Sharding Strategy** | How to distribute data across servers? | Consistent hashing + geographic sharding |
    | **Notifications** | How to notify 500M users instantly? | Push notifications + WebSocket + APNs/FCM |

    ---

    === "📰 News Feed Generation"

        ## The Challenge

        **Problem:** Generate personalized feeds for 500M users showing photos from their followed users (~150 follows per user average). Must be fast (< 200ms) and relevant.

        **Naive approach:** On feed request, query all followed users' photos, sort by time. **Doesn't scale** (150 queries per feed load!)

        **Solution:** Pre-compute feeds using **fan-out on write** (push model).

        ---

        ## Feed Generation Strategies

        ### Strategy 1: Fan-Out on Write (Push Model)

        **Concept:** When user posts photo, push it to all followers' feeds immediately.

        **How it works:**

        1. User uploads photo
        2. Get list of followers (e.g., 10K followers)
        3. Write photo_id to each follower's feed table
        4. When follower requests feed, read from pre-computed feed table

        **Implementation:**

        ```python
        class FeedFanoutWorker:
            """Fan-out new photos to followers' feeds"""

            def __init__(self, graph_db, photo_db):
                self.graph = graph_db
                self.photo_db = photo_db

            def fanout_photo(self, photo_id: str, user_id: str):
                """
                Push photo to all followers' feeds

                Args:
                    photo_id: ID of newly uploaded photo
                    user_id: ID of user who uploaded photo
                """
                # Get all followers
                followers = self.graph.get_followers(user_id)  # e.g., 10K followers

                logger.info(f"Fanning out photo {photo_id} to {len(followers)} followers")

                # Batch write to feeds (1000 at a time)
                batch_size = 1000
                for i in range(0, len(followers), batch_size):
                    batch = followers[i:i+batch_size]

                    # Batch insert into user_feed table
                    self.photo_db.batch_insert(
                        table='user_feed',
                        data=[
                            {
                                'user_id': follower_id,
                                'created_at': datetime.utcnow(),
                                'photo_id': photo_id
                            }
                            for follower_id in batch
                        ]
                    )

                    logger.info(f"Written batch {i//batch_size + 1}/{len(followers)//batch_size}")

                logger.info(f"Fanout complete for photo {photo_id}")
        ```

        **Pros:**

        - **Fast reads:** Feed already computed, just read from table (< 50ms)
        - **Simple read logic:** No complex queries or aggregation

        **Cons:**

        - **Slow writes:** For celebrity with 100M followers, takes minutes to fan-out
        - **Storage overhead:** Each photo stored in 10K+ feed tables (high write volume)
        - **Wasted work:** Inactive users get feed updates they'll never read

        **Use case:** Good for most users (< 100K followers). Instagram uses this for regular users.

        ---

        ### Strategy 2: Fan-Out on Read (Pull Model)

        **Concept:** When user requests feed, query all followed users' photos and merge.

        **How it works:**

        1. User requests feed
        2. Get list of followed users (e.g., 150 followed)
        3. Query recent photos from each followed user
        4. Merge and sort by timestamp
        5. Return top N photos

        **Implementation:**

        ```python
        class FeedPullService:
            """Generate feed on-demand by pulling from followed users"""

            def __init__(self, graph_db, photo_db, cache):
                self.graph = graph_db
                self.photo_db = photo_db
                self.cache = cache

            def generate_feed(self, user_id: str, count: int = 20) -> List[dict]:
                """
                Generate feed by pulling from followed users

                Args:
                    user_id: User requesting feed
                    count: Number of photos to return

                Returns:
                    List of photo objects, sorted by created_at
                """
                # Get followed users
                following = self.graph.get_following(user_id)  # e.g., 150 users

                # Query recent photos from each followed user (parallel)
                all_photos = []
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = [
                        executor.submit(self.get_user_photos, followed_id, limit=100)
                        for followed_id in following
                    ]

                    for future in as_completed(futures):
                        photos = future.result()
                        all_photos.extend(photos)

                # Sort by created_at and take top N
                all_photos.sort(key=lambda p: p['created_at'], reverse=True)
                return all_photos[:count]

            def get_user_photos(self, user_id: str, limit: int) -> List[dict]:
                """Get recent photos from user (with caching)"""
                cache_key = f"user_photos:{user_id}"
                cached = self.cache.get(cache_key)

                if cached:
                    return cached

                # Query database
                photos = self.photo_db.query(
                    "SELECT * FROM user_photos WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                    (user_id, limit)
                )

                # Cache for 1 minute
                self.cache.setex(cache_key, 60, photos)
                return photos
        ```

        **Pros:**

        - **Fast writes:** Just save photo once, no fan-out needed
        - **No wasted work:** Only generate feed for active users
        - **Always fresh:** No stale data, always latest photos

        **Cons:**

        - **Slow reads:** Must query 150 users, merge, sort (200-500ms even with caching)
        - **High read load:** Database hit on every feed request
        - **Complex:** Parallel queries, merge logic, ranking

        **Use case:** Good for celebrities with millions of followers. Twitter uses this for high-follower accounts.

        ---

        ### Strategy 3: Hybrid (Instagram's Approach)

        **Concept:** Use fan-out on write for most users, fan-out on read for celebrities.

        **How it works:**

        - **Regular users (< 100K followers):** Fan-out on write
        - **Celebrities (> 100K followers):** Fan-out on read (pull on demand)
        - **Mixed approach:** Pre-compute feed with regular users' photos, fetch celebrity photos on-demand

        **Implementation:**

        ```python
        class HybridFeedService:
            """Hybrid feed generation strategy"""

            CELEBRITY_THRESHOLD = 100_000  # Followers

            def __init__(self, graph_db, photo_db, cache):
                self.graph = graph_db
                self.photo_db = photo_db
                self.cache = cache

            def generate_feed(self, user_id: str, count: int = 20) -> List[dict]:
                """
                Generate feed using hybrid strategy

                Pre-computed feed + celebrity pulls
                """
                following = self.graph.get_following(user_id)

                # Separate into regular and celebrity accounts
                regular_users = []
                celebrities = []

                for followed_id in following:
                    follower_count = self.graph.get_follower_count(followed_id)
                    if follower_count > self.CELEBRITY_THRESHOLD:
                        celebrities.append(followed_id)
                    else:
                        regular_users.append(followed_id)

                # Get pre-computed feed (regular users only)
                precomputed = self.photo_db.query(
                    "SELECT * FROM user_feed WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
                    (user_id,)
                )

                # Pull celebrity photos on-demand
                celebrity_photos = []
                if celebrities:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [
                            executor.submit(self.get_user_photos, celeb_id, limit=20)
                            for celeb_id in celebrities
                        ]

                        for future in as_completed(futures):
                            celebrity_photos.extend(future.result())

                # Merge and sort
                all_photos = precomputed + celebrity_photos
                all_photos.sort(key=lambda p: p['created_at'], reverse=True)

                # Rank using ML model
                ranked_photos = self.rank_photos(all_photos, user_id)

                return ranked_photos[:count]

            def rank_photos(self, photos: List[dict], user_id: str) -> List[dict]:
                """
                Rank photos using ML model (engagement prediction)

                Factors:
                - Recency (newer photos ranked higher)
                - Engagement (likes, comments)
                - Relationship strength (how often user interacts with poster)
                - Content type (photo vs carousel vs video)
                """
                # Simplified scoring
                for photo in photos:
                    score = 0

                    # Recency (decay over time)
                    age_hours = (datetime.utcnow() - photo['created_at']).total_seconds() / 3600
                    score += 100 / (1 + age_hours/24)  # Decay over days

                    # Engagement
                    score += photo['likes_count'] * 0.1
                    score += photo['comments_count'] * 0.5

                    # Relationship strength (cache user interactions)
                    interaction_score = self.get_interaction_score(user_id, photo['user_id'])
                    score += interaction_score * 10

                    photo['score'] = score

                # Sort by score
                photos.sort(key=lambda p: p['score'], reverse=True)
                return photos
        ```

        **Pros:**

        - **Best of both:** Fast reads for most cases, handles celebrities gracefully
        - **Scalable:** No single bottleneck
        - **Cost-effective:** Don't fan-out to 100M followers

        **Cons:**

        - **Complex:** Two code paths to maintain
        - **Tuning required:** What's the threshold? (Instagram uses ~100K)

        **Recommendation:** Use hybrid approach for Instagram scale. This is what Instagram actually does.

        ---

        ## Feed Ranking (ML Model)

        **Problem:** Chronological feed not optimal. Users want to see most engaging content first.

        **Solution:** Machine learning ranking model predicts engagement probability.

        **Features used:**

        | Feature | Description | Weight |
        |---------|-------------|--------|
        | **Recency** | How new is the photo? | High |
        | **Poster engagement** | How often does user like poster's content? | Very High |
        | **Content engagement** | Likes/comments on this photo | Medium |
        | **Content type** | Photo, carousel, video | Medium |
        | **Viewing time** | Time spent on similar content | High |
        | **Hashtags** | User's hashtag interests | Low |

        **Model:** Gradient boosted trees (XGBoost) predicting probability of engagement (like, comment, share).

        **Training:**

        - **Positive examples:** Photos user liked/commented on
        - **Negative examples:** Photos user saw but didn't engage with
        - **Labels:** Binary (engaged=1, not engaged=0)
        - **Re-train:** Daily with fresh data

        **Serving:**

        - **Real-time:** Score photos at feed request time (< 50ms for 100 photos)
        - **Batch:** Pre-compute scores for top 1000 photos per user, refresh hourly

    === "🖼️ Image Storage & CDN"

        ## The Challenge

        **Problem:** Store 1.68 exabytes of photos, serve them globally with < 100ms latency, handle 1.86 Tbps of traffic.

        **Requirements:**

        - **Durability:** 99.999999999% (11 nines) - photos must never be lost
        - **Availability:** 99.99% - photos accessible 24/7
        - **Low latency:** < 100ms p95 worldwide
        - **Cost-effective:** Minimize storage and bandwidth costs

        ---

        ## Storage Architecture

        **Three-tier storage:**

        1. **S3 (origin)** - Store original and thumbnails
        2. **CDN (edge)** - Serve photos from 200+ edge locations worldwide
        3. **Redis (hot cache)** - Cache metadata and most recent photos

        ---

        ## Image Processing Pipeline

        **On upload:**

        1. **Upload original** - User uploads 2-5 MB photo
        2. **Store original** - Save to S3 (archival, rarely accessed)
        3. **Generate thumbnails** - Resize to multiple sizes:
           - **Large:** 1080×1080 (feed, profile)
           - **Medium:** 640×640 (grid view)
           - **Small:** 150×150 (notifications, comments)
        4. **Compress** - JPEG quality 85%, WebP for supported clients (30% smaller)
        5. **Upload thumbnails** - Store in S3
        6. **Invalidate CDN** - (not needed, new photo = new URL)

        **Implementation:**

        ```python
        from PIL import Image
        import io
        import boto3

        class ImageProcessor:
            """Process and store images with multiple thumbnails"""

            THUMBNAIL_SIZES = {
                'small': (150, 150),
                'medium': (640, 640),
                'large': (1080, 1080)
            }

            def __init__(self, s3_client):
                self.s3 = s3_client
                self.bucket = 'instagram-photos'

            def process_upload(self, photo_data: bytes, photo_id: str, user_id: str) -> dict:
                """
                Process uploaded photo and generate thumbnails

                Returns:
                    {
                        'original': 's3://bucket/photos/original/user123/photo456.jpg',
                        'thumbnails': {
                            'small': 's3://bucket/photos/small/user123/photo456.jpg',
                            'medium': '...',
                            'large': '...'
                        }
                    }
                """
                # Load image
                image = Image.open(io.BytesIO(photo_data))

                # Store original
                original_key = f"photos/original/{user_id}/{photo_id}.jpg"
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=original_key,
                    Body=photo_data,
                    ContentType='image/jpeg',
                    StorageClass='STANDARD_IA'  # Infrequent access (cheaper)
                )

                # Generate and store thumbnails
                thumbnail_keys = {}
                for size_name, (width, height) in self.THUMBNAIL_SIZES.items():
                    # Resize with aspect ratio preservation
                    thumbnail = self.resize_image(image, width, height)

                    # Compress to JPEG
                    output = io.BytesIO()
                    thumbnail.save(output, format='JPEG', quality=85, optimize=True)
                    output.seek(0)

                    # Upload to S3
                    thumbnail_key = f"photos/{size_name}/{user_id}/{photo_id}.jpg"
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=thumbnail_key,
                        Body=output.getvalue(),
                        ContentType='image/jpeg',
                        StorageClass='STANDARD',  # Frequently accessed
                        CacheControl='public, max-age=31536000'  # Cache for 1 year
                    )

                    thumbnail_keys[size_name] = f"s3://{self.bucket}/{thumbnail_key}"

                return {
                    'original': f"s3://{self.bucket}/{original_key}",
                    'thumbnails': thumbnail_keys
                }

            def resize_image(self, image: Image, max_width: int, max_height: int) -> Image:
                """
                Resize image preserving aspect ratio

                Args:
                    image: PIL Image object
                    max_width: Maximum width
                    max_height: Maximum height

                Returns:
                    Resized PIL Image
                """
                # Calculate aspect ratio
                width, height = image.size
                aspect_ratio = width / height

                # Determine new dimensions
                if width > height:
                    # Landscape
                    new_width = min(width, max_width)
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Portrait
                    new_height = min(height, max_height)
                    new_width = int(new_height * aspect_ratio)

                # Resize with high-quality Lanczos filter
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return resized
        ```

        ---

        ## CDN Configuration

        **CDN provider:** CloudFront (AWS), Cloudflare, or Fastly

        **Configuration:**

        ```javascript
        // CloudFront distribution config
        {
          "Origins": [
            {
              "Id": "instagram-s3-origin",
              "DomainName": "instagram-photos.s3.amazonaws.com",
              "S3OriginConfig": {
                "OriginAccessIdentity": "origin-access-identity/cloudfront/E1234567890"
              }
            }
          ],
          "CacheBehaviors": [
            {
              "PathPattern": "/photos/*",
              "TargetOriginId": "instagram-s3-origin",
              "ViewerProtocolPolicy": "https-only",
              "Compress": true,
              "MinTTL": 86400,          // 1 day
              "DefaultTTL": 2592000,    // 30 days
              "MaxTTL": 31536000,       // 1 year
              "AllowedMethods": ["GET", "HEAD"],
              "CachedMethods": ["GET", "HEAD"],
              "ForwardedValues": {
                "QueryString": false,
                "Headers": ["Accept", "Accept-Encoding"],
                "Cookies": {
                  "Forward": "none"
                }
              }
            }
          ],
          "PriceClass": "PriceClass_All",  // Global edge locations
          "GeoRestriction": {
            "RestrictionType": "none"
          }
        }
        ```

        **Benefits:**

        - **Low latency:** 50-100ms worldwide vs 300-500ms from origin
        - **Cost savings:** CDN egress $0.085/GB vs S3 egress $0.09/GB
        - **Origin protection:** 80% of traffic served from CDN, not S3
        - **DDoS protection:** CDN absorbs attack traffic

        ---

        ## Storage Optimization

        ### Compression

        **JPEG vs WebP:**

        | Format | Size | Quality | Browser Support |
        |--------|------|---------|-----------------|
        | **JPEG (quality 85)** | 200 KB | Good | 100% |
        | **WebP (quality 85)** | 140 KB | Better | 95% (not IE11) |
        | **Savings** | 30% | Same perceived quality | Use content negotiation |

        **Implementation:**

        ```python
        def serve_photo(request, photo_id):
            """Serve photo in optimal format based on client support"""

            # Check Accept header for WebP support
            accept = request.headers.get('Accept', '')
            supports_webp = 'image/webp' in accept

            # Get photo URL
            if supports_webp:
                photo_url = f"https://cdn.instagram.com/photos/{photo_id}.webp"
            else:
                photo_url = f"https://cdn.instagram.com/photos/{photo_id}.jpg"

            return {
                'url': photo_url,
                'format': 'webp' if supports_webp else 'jpeg'
            }
        ```

        ### Deduplication

        **Problem:** Users upload same photo multiple times (re-shares, duplicates).

        **Solution:** Content-based deduplication using perceptual hash.

        ```python
        import imagehash
        from PIL import Image

        def deduplicate_photo(photo_data: bytes) -> tuple:
            """
            Check if photo already exists using perceptual hash

            Returns:
                (is_duplicate, existing_photo_id)
            """
            # Generate perceptual hash
            image = Image.open(io.BytesIO(photo_data))
            phash = str(imagehash.phash(image))

            # Check if hash exists in database
            existing = db.query(
                "SELECT photo_id FROM photo_hashes WHERE phash = %s",
                (phash,)
            )

            if existing:
                return (True, existing['photo_id'])

            # Not a duplicate
            return (False, None)
        ```

        **Savings:** 5-10% storage reduction (common memes, screenshots)

    === "🔀 Sharding Strategy"

        ## The Challenge

        **Problem:** 1B users, 730B photos, 7.5B likes/day. Single database can't handle this scale.

        **Solution:** Shard data across multiple databases.

        **Requirements:**

        - **Even distribution:** No hot shards
        - **Minimal cross-shard queries:** Most queries hit single shard
        - **Easy to add shards:** Re-sharding shouldn't cause downtime

        ---

        ## Sharding Approaches

        ### 1. Shard by User ID (Recommended)

        **Concept:** User data and their photos on same shard.

        **Shard key:** `user_id`

        **Distribution:** `shard_id = hash(user_id) % num_shards`

        **Schema distribution:**

        - **User data:** Shard by user_id
        - **User photos:** Shard by user_id (co-located with user)
        - **Photo metadata:** Shard by photo_id (separate)
        - **Feed data:** Shard by user_id (feed owner)

        **Benefits:**

        - **User profile queries:** Single shard (user + their photos)
        - **Photo upload:** Single shard (insert user_photo)
        - **Feed generation:** Single shard (read user_feed table)

        **Challenges:**

        - **Photo queries:** Need photo_id → user_id mapping (extra lookup)
        - **Cross-user operations:** Follow/like require cross-shard queries

        ---

        ### 2. Shard by Photo ID

        **Concept:** Photos distributed across shards by photo_id.

        **Shard key:** `photo_id`

        **Distribution:** `shard_id = hash(photo_id) % num_shards`

        **Benefits:**

        - **Photo queries:** Single shard (direct photo lookup)
        - **Even distribution:** Photos evenly distributed

        **Challenges:**

        - **User profile queries:** Photos scattered across shards (slow)
        - **Feed generation:** Must query multiple shards

        **Not recommended for Instagram** (user-centric system).

        ---

        ## Instagram's Sharding Strategy

        **Hybrid approach:**

        1. **User data:** Shard by user_id (4096 shards)
        2. **Photos:** Shard by user_id (co-located with user)
        3. **Social graph:** Separate graph database (Neo4j, sharded by user_id)
        4. **Likes/Comments:** Shard by photo_id (separate cluster)

        **Implementation:**

        ```python
        class ShardRouter:
            """Route queries to appropriate database shard"""

            NUM_SHARDS = 4096

            def __init__(self, shard_configs):
                self.shards = {}
                for shard_id, config in shard_configs.items():
                    self.shards[shard_id] = DatabaseConnection(config)

            def get_user_shard(self, user_id: str) -> DatabaseConnection:
                """Get shard for user data"""
                shard_id = self._hash_to_shard(user_id)
                return self.shards[shard_id]

            def get_photo_shard(self, photo_id: str) -> DatabaseConnection:
                """Get shard for photo data (via user_id lookup)"""
                # Photos sharded by user_id, need to map photo_id -> user_id
                user_id = self._get_photo_owner(photo_id)
                return self.get_user_shard(user_id)

            def _hash_to_shard(self, key: str) -> int:
                """Consistent hash to determine shard"""
                return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.NUM_SHARDS

            def _get_photo_owner(self, photo_id: str) -> str:
                """
                Get user_id who owns photo

                This mapping stored in cache or separate lookup table
                """
                # Try cache first
                user_id = cache.get(f"photo_owner:{photo_id}")
                if user_id:
                    return user_id

                # Fallback: query mapping table (replicated to all shards)
                user_id = self.global_lookup(
                    "SELECT user_id FROM photo_owners WHERE photo_id = %s",
                    (photo_id,)
                )
                cache.setex(f"photo_owner:{photo_id}", 3600, user_id)
                return user_id
        ```

        ---

        ## Re-Sharding Strategy

        **Problem:** Need to add more shards as data grows. How to migrate without downtime?

        **Solution: Consistent hashing + gradual migration**

        **Steps:**

        1. **Add new shards** (e.g., 4096 → 8192 shards)
        2. **Dual-write period:** Write to both old and new shards
        3. **Background migration:** Move data shard-by-shard
        4. **Read from new shards:** Once migration complete
        5. **Decommission old shards**

        **No downtime, gradual transition.**

    === "🔔 Notifications System"

        ## The Challenge

        **Problem:** Send real-time notifications to 500M users for likes, comments, follows, mentions.

        **Scale:** 7.5B likes/day = 87K notifications/sec just for likes!

        **Requirements:**

        - **Real-time:** Notify within 1-2 seconds
        - **Reliable:** Don't miss notifications
        - **Deduplicated:** "John and 5 others liked your photo" (not 6 separate notifications)
        - **Multi-channel:** Push notifications, in-app, web push

        ---

        ## Notification Architecture

        ```mermaid
        graph TB
            subgraph "Trigger Sources"
                Like[Like Event]
                Comment[Comment Event]
                Follow[Follow Event]
            end

            subgraph "Notification Service"
                Queue[Kafka Queue]
                Processor[Notification Processor]
                Dedup[Deduplication Logic]
            end

            subgraph "Delivery Channels"
                APNs[Apple Push<br/>iOS]
                FCM[Firebase Cloud Messaging<br/>Android]
                WebPush[Web Push<br/>Browser]
                InApp[In-App<br/>Badge]
            end

            subgraph "Storage"
                Notification_DB[(Notification History<br/>Cassandra)]
                Redis_Dedup[(Redis<br/>Dedup tracking)]
            end

            Like --> Queue
            Comment --> Queue
            Follow --> Queue

            Queue --> Processor
            Processor --> Dedup
            Dedup --> Redis_Dedup

            Processor --> APNs
            Processor --> FCM
            Processor --> WebPush
            Processor --> InApp

            Processor --> Notification_DB
        ```

        ---

        ## Notification Processing

        **Flow:**

        1. **Event published:** User likes photo → Kafka event
        2. **Deduplication:** Check if similar notification sent recently
        3. **Aggregate:** "John and 5 others" instead of 6 notifications
        4. **Delivery:** Push to APNs/FCM/WebPush
        5. **Store:** Save to notification history

        **Implementation:**

        ```python
        class NotificationService:
            """Handle real-time notifications"""

            def __init__(self, kafka, redis, apns, fcm):
                self.kafka = kafka
                self.redis = redis
                self.apns = apns  # Apple Push Notification Service
                self.fcm = fcm    # Firebase Cloud Messaging

            def process_like_event(self, event: dict):
                """
                Process like event and send notification

                Event format:
                {
                    'photo_id': '7f8a3b2c',
                    'liked_by_user_id': 'user123',
                    'liked_by_username': 'john_doe',
                    'photo_owner_id': 'user456',
                    'timestamp': 1643712000
                }
                """
                photo_id = event['photo_id']
                photo_owner = event['photo_owner_id']
                liker_username = event['liked_by_username']

                # Don't notify if user likes their own photo
                if event['liked_by_user_id'] == photo_owner:
                    return

                # Deduplication key (aggregate likes within 1 minute)
                dedup_key = f"like_notif:{photo_owner}:{photo_id}:{int(time.time() / 60)}"

                # Check if notification already sent recently
                if self.redis.exists(dedup_key):
                    # Add to aggregation counter
                    self.redis.hincrby(dedup_key, 'count', 1)
                    self.redis.sadd(f"{dedup_key}:users", liker_username)
                    logger.info(f"Aggregating like notification for {photo_owner}")
                    return

                # First like in this window - send notification
                self.redis.setex(dedup_key, 60, 1)  # 1 minute window
                self.redis.sadd(f"{dedup_key}:users", liker_username)

                # Get additional likers (for "and N others")
                other_likers_count = self.redis.scard(f"{dedup_key}:users") - 1

                # Build notification message
                if other_likers_count == 0:
                    message = f"{liker_username} liked your photo"
                else:
                    message = f"{liker_username} and {other_likers_count} others liked your photo"

                # Send push notification
                self.send_push_notification(
                    user_id=photo_owner,
                    title="New Like",
                    message=message,
                    data={
                        'type': 'like',
                        'photo_id': photo_id,
                        'action_url': f'/photos/{photo_id}'
                    }
                )

                # Store in notification history
                self.save_notification(
                    user_id=photo_owner,
                    type='like',
                    message=message,
                    photo_id=photo_id
                )

            def send_push_notification(self, user_id: str, title: str, message: str, data: dict):
                """
                Send push notification to user's devices

                User may have multiple devices (iPhone, iPad, Android)
                """
                # Get user's device tokens
                devices = self.get_user_devices(user_id)

                for device in devices:
                    if device['platform'] == 'ios':
                        # Send to APNs
                        self.apns.send_notification(
                            device_token=device['token'],
                            alert={
                                'title': title,
                                'body': message
                            },
                            badge=self.get_unread_count(user_id),
                            sound='default',
                            data=data
                        )

                    elif device['platform'] == 'android':
                        # Send to FCM
                        self.fcm.send_notification(
                            device_token=device['token'],
                            notification={
                                'title': title,
                                'body': message,
                                'sound': 'default'
                            },
                            data=data
                        )

            def get_unread_count(self, user_id: str) -> int:
                """Get unread notification count (for badge)"""
                # Cache in Redis
                count = self.redis.get(f"unread_count:{user_id}")
                if count is not None:
                    return int(count)

                # Query database
                count = db.query(
                    "SELECT COUNT(*) FROM notifications WHERE user_id = %s AND read = false",
                    (user_id,)
                )['count']

                self.redis.setex(f"unread_count:{user_id}", 300, count)
                return count
        ```

        ---

        ## Notification Deduplication

        **Problem:** User gets 100 likes in 5 minutes → 100 separate notifications (annoying!)

        **Solution:** Aggregate within time window (1-5 minutes).

        **Strategies:**

        1. **Time-based:** Group notifications within 1-minute window
        2. **Count-based:** After 5 likes, send "5 people liked your photo"
        3. **Intelligent:** Use ML to determine optimal aggregation

        **Benefits:**

        - Reduce notification spam (100 → 1 notification)
        - Better user experience
        - Lower push notification costs

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Instagram from prototype to 500M DAU.

    **Scaling milestones:**

    - **1K DAU:** Single server
    - **100K DAU:** Add caching, CDN
    - **1M DAU:** Database read replicas, sharding
    - **10M DAU:** Multi-region, graph database
    - **100M+ DAU:** Advanced optimizations below

    ---

    ## Bottleneck Identification

    | Component | Bottleneck at 500M DAU? | Solution |
    |-----------|------------------------|----------|
    | **Feed generation** | ✅ Yes (145K QPS) | Pre-compute feeds (fan-out on write), Redis cache |
    | **Image serving** | ✅ Yes (1.86 Tbps) | CDN (CloudFront), serve 80% from edge |
    | **Database writes** | ✅ Yes (90K write QPS) | Cassandra sharding (write-optimized), 4096 shards |
    | **Database reads** | ✅ Yes (145K read QPS) | Redis cache (90% hit rate), read replicas |
    | **Notifications** | ✅ Yes (87K/sec) | Kafka queue, batch processing, deduplication |
    | **Social graph** | 🟡 Approaching limit | Graph database (Neo4j), cache follower lists |

    ---

    ## Multi-Region Architecture

    **Problem:** Users worldwide, single region = high latency for distant users.

    **Solution:** Multi-region deployment with geographic routing.

    **Regions:**

    - **US-East (Virginia):** North America, South America
    - **EU-West (Ireland):** Europe, Africa
    - **AP-Southeast (Singapore):** Asia, Australia

    **Architecture per region:**

    - Full stack deployment (API, cache, database)
    - Photos in regional S3 buckets
    - CDN serves from nearest edge (automatic)
    - Social graph replicated globally (eventual consistency)

    **Benefits:**

    - **Latency:** 50-100ms vs 300-500ms cross-region
    - **Compliance:** EU data stays in EU (GDPR)
    - **Availability:** Region failure doesn't affect others

    ---

    ## Cost Optimization

    **Monthly cost at 500M DAU:**

    | Component | Cost | Optimization |
    |-----------|------|-------------|
    | **S3 Storage** | $38,640 (1.68 PB × $0.023/GB) | Lifecycle policies (move to Glacier after 1 year) |
    | **CDN** | $85,000 (1,000 TB egress × $0.085/GB) | Already optimized, cache aggressively |
    | **EC2 (API)** | $43,200 (300 × m5.2xlarge × $6/day) | Use spot instances (60% savings) |
    | **Cassandra (managed)** | $108,000 (100 nodes × $36/day) | Use open-source Cassandra on EC2 (50% savings) |
    | **RDS (PostgreSQL)** | $21,600 (30 shards × $24/day) | Use read replicas, not bigger instances |
    | **ElastiCache (Redis)** | $32,400 (100 nodes × $10.80/day) | Right-sized for workload |
    | **Kafka** | $12,960 (20 nodes × $21.60/day) | Use Amazon MSK (managed, cheaper) |
    | **Data transfer** | $18,000 | Use CloudFront (reduces origin transfer) |
    | **Total** | **$359,800/month** | |

    **Optimizations:**

    - **Spot instances:** Save $25,920/month on EC2
    - **Self-hosted Cassandra:** Save $54,000/month
    - **Reserved instances:** Save $6,480/month on RDS
    - **S3 lifecycle:** Save $9,660/month (Glacier after 1 year)
    - **Potential savings:** $96,060/month (27%)
    - **Optimized cost:** $263,740/month

    ---

    ## Performance Optimizations

    ### Feed Generation Optimization

    **Problem:** Feed generation is CPU-intensive (ranking 100 photos).

    **Solutions:**

    1. **Pre-compute rankings:** Rank photos offline, serve cached results
    2. **Limit ranking:** Only rank top 1000 photos per user (not all)
    3. **Use simpler model:** LightGBM instead of deep learning (10x faster)

    ### Image Optimization

    **Problem:** Large images slow down feed loading.

    **Solutions:**

    1. **Lazy loading:** Load images as user scrolls, not all at once
    2. **Progressive JPEG:** Show low-res preview, then full image
    3. **WebP format:** 30% smaller than JPEG (save bandwidth)

    ### Database Optimization

    **Problem:** Hot shards (celebrity accounts).

    **Solutions:**

    1. **Read replicas for hot shards:** 10 replicas for top 1% users
    2. **Cache celebrity profiles:** Cache for 1 hour (high hit rate)
    3. **Rate limit writes:** Limit likes/comments per second per photo

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Feed Latency (P95)** | < 200ms | > 500ms |
    | **Image Load Time (P95)** | < 100ms | > 300ms |
    | **Upload Success Rate** | > 99.9% | < 99% |
    | **CDN Hit Rate** | > 80% | < 70% |
    | **Database Lag (Cassandra)** | < 100ms | > 500ms |
    | **Cache Hit Rate (Redis)** | > 90% | < 80% |
    | **Notification Delivery Rate** | > 99% | < 95% |

=== "📝 Summary & Tips"

    ## Architecture Summary

    **Core components:**

    | Component | Purpose | Scale |
    |-----------|---------|-------|
    | **CDN** | Global image delivery | 1.86 Tbps traffic, 80% cache hit |
    | **API Servers** | Stateless, horizontally scalable | 300 servers, 145K QPS |
    | **Cassandra** | Photo metadata, feeds | 100 nodes, 90K write QPS |
    | **PostgreSQL** | User data, auth | 30 shards, 4096 logical shards |
    | **Neo4j** | Social graph | Followers/following queries |
    | **Redis** | Caching | 100 nodes, 90% hit rate, 41 TB data |
    | **S3** | Photo storage | 1.68 exabytes over 10 years |
    | **Kafka** | Async processing | Feed fanout, notifications |

    ---

    ## Key Design Decisions

    1. **Fan-out on write (hybrid):** Pre-compute feeds for fast reads
    2. **CDN for images:** Serve 80% from edge, < 100ms latency worldwide
    3. **Cassandra for metadata:** Write-heavy workload, horizontal scaling
    4. **Graph database for social:** Fast follower queries, social graph traversal
    5. **Eventual consistency:** Prioritize availability, tolerate brief inconsistency
    6. **Multi-region:** Deploy in 3 regions for low latency worldwide

    ---

    ## Interview Tips

    ✅ **Start with scale numbers** - 500M DAU, 200M photos/day, 1.68 exabytes

    ✅ **Discuss feed generation** - Fan-out on write vs read, hybrid approach

    ✅ **Address image storage** - S3 + CDN, multiple thumbnails, WebP compression

    ✅ **Talk about sharding** - User-based sharding, consistent hashing

    ✅ **Consider notifications** - Real-time, deduplication, multi-channel

    ✅ **Mention trade-offs** - Availability vs consistency, cost vs latency

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle celebrities with 100M followers?"** | Hybrid feed: fan-out on read for celebrities, pre-compute for regular users |
    | **"How to make feed generation faster?"** | Cache pre-computed feeds, use simpler ranking model, limit to top 1000 photos |
    | **"What if CDN goes down?"** | Fallback to origin (S3), multiple CDN providers for redundancy |
    | **"How to prevent spam/abuse?"** | Rate limiting, ML-based spam detection, shadowban for violators |
    | **"How to handle data consistency across regions?"** | Eventual consistency acceptable, use Cassandra multi-region, conflict resolution |
    | **"How to scale database writes (90K QPS)?"** | Cassandra (write-optimized), 4096 shards, SSD storage, batch writes |

    ---

    ## Related Problems

    | Problem | Similarity | Key Differences |
    |---------|------------|-----------------|
    | **Twitter** | News feed, follow/follower, real-time | Text-based (not images), higher write rate (tweets) |
    | **TikTok** | Feed, follow, likes, comments | Video (not images), recommendation algorithm critical |
    | **Facebook** | News feed, social graph, photos | More complex (groups, pages, events), friendship bidirectional |
    | **Pinterest** | Images, feed, follow | Interest-based (not social), recommendation-heavy |
    | **YouTube** | Video, feed, subscriptions | Video encoding/streaming, longer content, ad insertion |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Meta, Google, Amazon, Netflix, Twitter, TikTok

---

*Master this problem and you'll be ready for: Twitter, TikTok, Facebook, Pinterest, YouTube*
