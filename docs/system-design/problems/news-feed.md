# Design News Feed (Facebook)

A personalized news feed system where users see posts from friends, pages, and groups in a ranked order optimized for engagement. The feed includes text posts, photos, videos, ads, and recommendations.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 2B daily active users, 100M posts/day, 50B feed requests/day |
| **Key Challenges** | ML-based ranking, personalization, multiple content types, real-time updates, ad insertion |
| **Core Concepts** | EdgeRank algorithm, ML ranking models, hybrid fan-out, content diversity, A/B testing |
| **Companies** | Facebook, LinkedIn, Reddit, Instagram, Medium, Pinterest |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Feed Generation** | Personalized feed showing posts from friends/pages/groups | P0 (Must have) |
    | **Post Creation** | Create text, photo, video posts with tagging | P0 (Must have) |
    | **Interactions** | Like, comment, share posts | P0 (Must have) |
    | **ML Ranking** | Rank posts by predicted engagement | P0 (Must have) |
    | **Real-time Updates** | New posts appear without refresh | P1 (Should have) |
    | **Filters** | Filter by friends only, pages only, chronological | P1 (Should have) |
    | **Ads Insertion** | Insert sponsored posts in feed | P1 (Should have) |
    | **Notifications** | Notify of comments, likes, mentions | P2 (Nice to have) |
    | **Stories** | Ephemeral content (24h) at top of feed | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Video calling/messaging
    - Marketplace
    - Gaming platform
    - Content moderation (ML/AI systems)
    - Live streaming infrastructure

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.95% uptime | Users expect consistent access, revenue depends on it |
    | **Latency (Feed Load)** | < 500ms p95 | Fast feed load critical for engagement |
    | **Latency (Post Creation)** | < 1s | Quick feedback encourages posting |
    | **Consistency** | Eventual consistency | Brief delays acceptable (posts may appear with lag) |
    | **Personalization** | User-specific ranked feed | Different users see different content order |
    | **Scalability** | Billions of posts per day | Must handle global scale and viral content |
    | **Content Diversity** | Prevent echo chambers | Show diverse content types and sources |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 2B
    Monthly Active Users (MAU): 3B

    Post creation:
    - Active posters: 20% of DAU = 400M users
    - Posts per active poster: ~0.25 posts/day (many users just consume)
    - Daily posts: 400M × 0.25 = 100M posts/day
    - Post QPS: 100M / 86,400 = ~1,160 posts/sec
    - Peak QPS: 3x average = ~3,480 posts/sec

    Feed requests:
    - Feed views per DAU: ~25 views/day
    - Daily feed requests: 2B × 25 = 50B requests/day
    - Feed QPS: 50B / 86,400 = ~578,700 req/sec
    - Peak QPS: 1.5x = ~868K req/sec

    Interactions:
    - Likes per DAU: ~10 likes/day
    - Comments per DAU: ~2 comments/day
    - Shares per DAU: ~1 share/day
    - Daily interactions: 2B × 13 = 26B interactions/day
    - Interaction QPS: 26B / 86,400 = ~300,900 int/sec

    Total Read QPS: ~880K (feed + profile + search)
    Total Write QPS: ~305K (posts + interactions)
    Read/Write ratio: 2.9:1 (read-heavy system)
    ```

    ### Storage Estimates

    ```
    Post storage:
    - Text post: 5 KB (content, metadata, user_id, timestamps)
    - Photo post: 5 KB + 2 MB (compressed photo)
    - Video post: 5 KB + 50 MB (compressed video)

    Post distribution:
    - Text: 40% (40M/day) = 40M × 5 KB = 200 GB/day
    - Photo: 45% (45M/day) = 45M × 2 MB = 90 TB/day
    - Video: 15% (15M/day) = 15M × 50 MB = 750 TB/day

    Daily total: ~840 TB/day
    For 10 years: 840 TB × 365 × 10 = 3.07 exabytes

    User data:
    - 3B users × 50 KB (profile, preferences, features) = 150 TB

    Social graph (friendships):
    - 3B users × 300 friends (avg) × 16 bytes = 14.4 TB

    ML features & embeddings:
    - User embeddings: 3B × 1 KB = 3 TB
    - Post embeddings: 100M/day × 2 KB × 30 days = 6 TB

    Total: 3.07 EB (posts/media) + 150 TB (users) + 14.4 TB (graph) + 9 TB (ML) ≈ 3.07 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Post ingress:
    - Text: 1,160 posts/sec × 5 KB = 5.8 MB/sec ≈ 46 Mbps
    - Media uploads: 60% × 1,160 = 696 media/sec × 10 MB (avg) = 6.96 GB/sec ≈ 56 Gbps

    Feed egress:
    - 578,700 feed/sec × 20 posts × 5 KB = 57.9 GB/sec ≈ 463 Gbps (text only)
    - Media downloads: 15x uploads = 840 Gbps (CDN critical)

    Total ingress: ~56 Gbps
    Total egress: ~1,303 Gbps (1.3 Tbps - CDN absolutely required)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot posts (last 24 hours):
    - Posts: 100M × 5 KB = 500 GB
    - Cache 30% hottest: 150 GB

    User sessions:
    - 200M concurrent users × 20 KB = 4 TB

    Feed cache (pre-generated):
    - 50M most active users × 100 post IDs × 8 bytes = 40 GB

    ML model serving:
    - Ranking models in memory: 50 GB
    - Feature stores: 200 GB

    Social graph cache:
    - Hot edges: 500M users × 300 friends × 8 bytes = 1.2 TB

    Total cache: 150 GB + 4 TB + 40 GB + 250 GB + 1.2 TB ≈ 5.6 TB
    ```

    ---

    ## Key Assumptions

    1. Users have average of 300 friends
    2. Only 20% of users actively post, 80% are content consumers
    3. Feed is highly personalized (different for every user)
    4. ML ranking is critical (not just chronological)
    5. Eventual consistency acceptable (feed may be seconds behind)
    6. Ad insertion required (15-20% of feed content)
    7. Content diversity important (avoid filter bubbles)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **ML-first ranking:** Every feed ranked by ML model predicting engagement
    2. **Hybrid fan-out:** Pre-compute + on-demand generation for personalization
    3. **Multi-content types:** Text, photo, video, ads, stories unified
    4. **A/B testing infrastructure:** Continuous experimentation for ranking
    5. **Content diversity:** Prevent echo chambers with diverse signals

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Media delivery]
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Feed_API[Feed Service<br/>Generate personalized feed]
            Post_API[Post Service<br/>Create/edit posts]
            Interaction_API[Interaction Service<br/>Like/comment/share]
            Ranking_API[Ranking Service<br/>ML model serving]
            Ad_API[Ad Service<br/>Ad selection & insertion]
        end

        subgraph "ML Layer"
            Ranker[ML Ranker<br/>Score posts for user]
            Feature_Store[Feature Store<br/>User/post features]
            Model_Training[Model Training<br/>Offline pipeline]
            AB_Testing[A/B Testing<br/>Experimentation]
        end

        subgraph "Data Processing"
            Fanout_Worker[Fanout Worker<br/>Distribute posts]
            Aggregation[Aggregation Worker<br/>Compute stats]
            ETL[ETL Pipeline<br/>Feature extraction]
            Stream_Processor[Stream Processor<br/>Real-time events]
        end

        subgraph "Caching"
            Redis_Feed[Redis<br/>Feed cache]
            Redis_Post[Redis<br/>Post cache]
            Redis_User[Redis<br/>User cache]
            Redis_Social[Redis<br/>Social graph]
        end

        subgraph "Storage"
            Post_DB[(Post DB<br/>Cassandra<br/>Sharded)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Graph_DB[(Social Graph<br/>Graph DB)]
            Analytics_DB[(Analytics<br/>ClickHouse)]
            S3[Object Storage<br/>S3<br/>Media files]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "Search & Discovery"
            Search[Elasticsearch<br/>Content search]
            Recommendation[Recommendation<br/>Content discovery]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        CDN --> S3

        LB --> Feed_API
        LB --> Post_API
        LB --> Interaction_API

        Feed_API --> Ranking_API
        Feed_API --> Redis_Feed
        Feed_API --> Post_DB
        Feed_API --> Ad_API

        Post_API --> Kafka
        Post_API --> Post_DB
        Post_API --> S3

        Interaction_API --> Kafka
        Interaction_API --> Post_DB

        Ranking_API --> Ranker
        Ranking_API --> Feature_Store
        Ranker --> Redis_Post
        Ranker --> AB_Testing

        Kafka --> Fanout_Worker
        Kafka --> Aggregation
        Kafka --> ETL
        Kafka --> Stream_Processor

        Fanout_Worker --> Redis_Feed
        Fanout_Worker --> Post_DB

        ETL --> Feature_Store
        ETL --> Analytics_DB

        Model_Training --> Ranker
        Model_Training --> Analytics_DB

        Aggregation --> Post_DB
        Stream_Processor --> Redis_Feed

        Feature_Store --> Redis_User
        Graph_DB --> Redis_Social

        Ad_API --> Post_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Feed fill:#fff4e1
        style Redis_Post fill:#fff4e1
        style Redis_User fill:#fff4e1
        style Redis_Social fill:#fff4e1
        style Post_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Graph_DB fill:#e1f5e1
        style Analytics_DB fill:#e8eaf6
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
        style Ranker fill:#fff9c4
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **ML Ranker** | Personalized ranking (each user different), maximize engagement | Chronological (poor engagement), simple scoring (not personalized) |
    | **Cassandra (Post DB)** | Write-optimized, handles 305K write QPS, time-series data | MySQL (write bottleneck), MongoDB (consistency issues) |
    | **Feature Store** | Low-latency feature serving (<10ms), real-time + batch features | Direct DB queries (too slow), cache only (no feature consistency) |
    | **Graph Database** | Fast friend queries, mutual friends, "people you may know" | SQL joins (too slow for 300M friends), custom solution (complex) |
    | **Kafka** | Event streaming for posts/interactions, replay capability | Direct processing (no reliability), RabbitMQ (throughput limit) |
    | **Redis Cache** | Feed caching (80% hit rate), sub-10ms reads | No cache (can't handle 880K read QPS), Memcached (limited features) |
    | **ClickHouse** | Fast analytics for ML training, OLAP queries | PostgreSQL (too slow for TB queries), Redshift (expensive) |

    **Key Trade-off:** We chose **personalization over consistency**. Different users may see same post at different times, but each gets optimized feed.

    ---

    ## API Design

    ### 1. Get News Feed

    **Request:**
    ```http
    GET /api/v1/feed?cursor=abc123&count=20&filter=all
    Authorization: Bearer <token>
    ```

    **Query Parameters:**
    - `cursor`: Pagination cursor (encrypted timestamp + offset)
    - `count`: Number of posts to return (default: 20, max: 50)
    - `filter`: "all" | "friends" | "pages" | "groups"

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "posts": [
        {
          "post_id": "post_123456",
          "author": {
            "user_id": "user_789",
            "name": "John Doe",
            "profile_pic": "https://cdn.fb.com/profiles/user_789.jpg"
          },
          "content": {
            "type": "photo",
            "text": "Beautiful sunset today!",
            "media": [
              {
                "type": "image",
                "url": "https://cdn.fb.com/photos/123.jpg",
                "width": 1920,
                "height": 1080
              }
            ]
          },
          "created_at": "2026-02-02T10:30:00Z",
          "stats": {
            "likes": 152,
            "comments": 23,
            "shares": 8
          },
          "user_interaction": {
            "liked": true,
            "commented": false,
            "shared": false
          },
          "ranking_score": 0.87,
          "reason": "Your friend John Doe posted this"
        },
        {
          "post_id": "post_123457",
          "is_sponsored": true,
          "sponsor": {
            "name": "Nike",
            "page_id": "page_456"
          },
          // ... ad content
        },
        // ... 18 more posts
      ],
      "next_cursor": "xyz789",
      "has_more": true,
      "metadata": {
        "ranking_version": "v4.2.1",
        "ab_test_group": "control_a",
        "generation_time_ms": 287
      }
    }
    ```

    **Design Notes:**

    - Encrypted cursor prevents manipulation
    - Ranking score included for debugging (not shown to users)
    - Ads marked with `is_sponsored: true`
    - Metadata for A/B testing and debugging
    - Personalized reason ("Your friend X posted...")

    ---

    ### 2. Create Post

    **Request:**
    ```http
    POST /api/v1/posts
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "content": {
        "text": "Just finished my morning run! 🏃",
        "type": "status"  // "status" | "photo" | "video" | "link"
      },
      "media_ids": ["media_123", "media_456"],  // Optional
      "privacy": "friends",  // "public" | "friends" | "friends_except" | "only_me"
      "tagged_users": ["user_456", "user_789"],  // Optional
      "location": {  // Optional
        "lat": 37.7749,
        "lng": -122.4194,
        "name": "San Francisco, CA"
      },
      "feeling": "excited"  // Optional
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "post_id": "post_123456",
      "user_id": "user123",
      "created_at": "2026-02-02T10:30:00Z",
      "status": "published",
      "visibility": {
        "privacy": "friends",
        "estimated_reach": 287  // Number of friends who'll see it
      }
    }
    ```

    **Design Notes:**

    - Return immediately after validation and storage (don't wait for fan-out)
    - Extract hashtags, mentions automatically
    - Rate limit: 100 posts per day per user
    - Validate media upload completion before accepting post

    ---

    ### 3. Interact with Post

    **Request (Like):**
    ```http
    POST /api/v1/posts/post_123456/like
    Authorization: Bearer <token>
    ```

    **Request (Comment):**
    ```http
    POST /api/v1/posts/post_123456/comments
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "text": "Amazing photo!",
      "reply_to": "comment_789"  // Optional, for nested replies
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "interaction_id": "interaction_123",
      "post_id": "post_123456",
      "type": "like",  // or "comment" or "share"
      "created_at": "2026-02-02T10:30:15Z",
      "updated_stats": {
        "likes": 153,
        "comments": 23,
        "shares": 8
      }
    }
    ```

    ---

    ## Database Schema

    ### Posts (Cassandra)

    ```sql
    -- Main posts table
    CREATE TABLE posts (
        post_id BIGINT PRIMARY KEY,
        user_id BIGINT,
        content_type TEXT,  -- 'status', 'photo', 'video', 'link'
        text_content TEXT,
        media_urls LIST<TEXT>,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        privacy TEXT,  -- 'public', 'friends', 'custom'
        location TEXT,
        tagged_users LIST<BIGINT>,
        likes_count COUNTER,
        comments_count COUNTER,
        shares_count COUNTER,
        is_deleted BOOLEAN,
        -- Partitioned by post_id for write distribution
    );

    -- User posts (for profile timeline)
    CREATE TABLE user_posts (
        user_id BIGINT,
        created_at TIMESTAMP,
        post_id BIGINT,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Feed candidate pool (pre-computed)
    CREATE TABLE feed_candidates (
        user_id BIGINT,
        post_id BIGINT,
        created_at TIMESTAMP,
        source_type TEXT,  -- 'friend', 'page', 'group', 'suggested'
        source_id BIGINT,
        PRIMARY KEY (user_id, created_at, post_id)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Interactions table
    CREATE TABLE post_interactions (
        post_id BIGINT,
        user_id BIGINT,
        interaction_type TEXT,  -- 'like', 'comment', 'share'
        created_at TIMESTAMP,
        interaction_data TEXT,  -- JSON with comment text, share caption, etc.
        PRIMARY KEY (post_id, interaction_type, user_id)
    );
    ```

    **Why Cassandra:**

    - **High write throughput:** 305K write QPS (posts + interactions)
    - **Time-series optimized:** Clustering by created_at for feed queries
    - **Linear scalability:** Add nodes for more capacity
    - **No single point of failure:** Multi-master architecture

    ---

    ### Users (PostgreSQL)

    ```sql
    -- Users table (sharded by user_id)
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        username VARCHAR(50) UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        profile_pic_url TEXT,
        cover_photo_url TEXT,
        bio TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_active_at TIMESTAMP,
        friend_count INT DEFAULT 0,
        follower_count INT DEFAULT 0,
        privacy_settings JSONB,
        preferences JSONB,  -- Feed preferences, notification settings
        INDEX idx_email (email),
        INDEX idx_username (username)
    ) PARTITION BY HASH (user_id);
    ```

    ---

    ### Social Graph (Graph Database)

    ```cypher
    // Friendship relationship (bidirectional)
    CREATE (u1:User {user_id: 123})-[:FRIENDS_WITH {since: 1643712000}]->(u2:User {user_id: 456})

    // Follow relationship (unidirectional, for pages/public figures)
    CREATE (u1:User {user_id: 123})-[:FOLLOWS {created_at: 1643712000}]->(p:Page {page_id: 789})

    // Query: Get friends
    MATCH (user:User {user_id: 123})-[:FRIENDS_WITH]->(friend:User)
    RETURN friend.user_id

    // Query: Get mutual friends
    MATCH (user:User {user_id: 123})-[:FRIENDS_WITH]->(mutual:User)-[:FRIENDS_WITH]->(other:User {user_id: 456})
    RETURN COUNT(mutual) as mutual_friends

    // Query: Friend recommendations (friends of friends)
    MATCH (user:User {user_id: 123})-[:FRIENDS_WITH]->(friend)-[:FRIENDS_WITH]->(fof:User)
    WHERE NOT (user)-[:FRIENDS_WITH]->(fof) AND fof.user_id <> 123
    RETURN fof.user_id, COUNT(*) as mutual_friends
    ORDER BY mutual_friends DESC
    LIMIT 10
    ```

    ---

    ## Data Flow Diagrams

    ### Post Creation & Fan-out Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Post_API
        participant Post_DB
        participant S3
        participant Kafka
        participant Fanout_Worker
        participant Feed_DB
        participant Notif_Service

        Client->>Post_API: POST /api/v1/posts
        Post_API->>Post_API: Validate, extract metadata

        alt Has media
            Post_API->>S3: Upload media
            S3-->>Post_API: media_urls
        end

        Post_API->>Post_DB: INSERT post
        Post_DB-->>Post_API: post_id

        Post_API->>Kafka: Publish post_created event
        Post_API-->>Client: 201 Created (post_id)

        Kafka->>Fanout_Worker: Process post_created
        Fanout_Worker->>Graph_DB: Get friends list
        Graph_DB-->>Fanout_Worker: friend_ids

        alt Regular user (<5000 friends)
            Fanout_Worker->>Feed_DB: Batch INSERT into feed_candidates
            Fanout_Worker->>Notif_Service: Notify online friends
        else Popular user (>5000 friends)
            Fanout_Worker->>Feed_DB: INSERT for active friends only
            Fanout_Worker->>Cache: Mark for hybrid retrieval
        end

        Fanout_Worker->>Kafka: Publish fanout_complete event
    ```

    **Flow Explanation:**

    1. **Validate & store** - Save post, upload media (< 500ms)
    2. **Publish event** - Kafka for async processing
    3. **Fan-out** - Write to friends' feed_candidates (hybrid approach)
    4. **Notify** - Real-time notifications to online friends
    5. **Feature extraction** - Async ML feature computation

    ---

    ### Feed Generation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Feed_API
        participant Redis
        participant Feed_DB
        participant Ranking_API
        participant Feature_Store
        participant Post_DB

        Client->>Feed_API: GET /api/v1/feed
        Feed_API->>Feed_API: Authenticate user

        Feed_API->>Redis: GET feed:user123:hash
        alt Cache HIT (70% of requests)
            Redis-->>Feed_API: Cached ranked post IDs
            Feed_API->>Post_DB: Batch get post details
            Post_DB-->>Feed_API: Post objects
        else Cache MISS (30% of requests)
            Redis-->>Feed_API: null

            Feed_API->>Feed_DB: Get feed_candidates (last 7 days)
            Feed_DB-->>Feed_API: ~500-1000 candidate posts

            Feed_API->>Ranking_API: Rank candidates for user
            Ranking_API->>Feature_Store: Get user features
            Ranking_API->>Feature_Store: Get post features
            Feature_Store-->>Ranking_API: Features

            Ranking_API->>Ranking_API: ML model inference (score each post)
            Ranking_API-->>Feed_API: Ranked post IDs with scores

            Feed_API->>Feed_API: Apply business logic (diversity, ads)
            Feed_API->>Post_DB: Batch get top 20 posts
            Post_DB-->>Feed_API: Post objects

            Feed_API->>Redis: SETEX feed:user123:hash (TTL: 120s)
        end

        Feed_API->>Feed_API: Insert ads (3-4 per 20 posts)
        Feed_API-->>Client: 200 OK (ranked feed)
    ```

    **Flow Explanation:**

    1. **Check cache** - Redis hit for 70% (returning users within 2 min)
    2. **Fetch candidates** - Get pool of 500-1000 recent posts
    3. **ML ranking** - Score each post for user (< 100ms)
    4. **Business logic** - Apply diversity, insert ads
    5. **Cache result** - Store ranked IDs for 2 minutes
    6. **Hydrate** - Batch fetch full post details

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section explores four critical News Feed subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **ML Ranking** | How to rank billions of posts for each user? | Two-stage ranking: candidate generation + ML scoring |
    | **Fan-out Strategy** | How to deliver posts to thousands of friends? | Hybrid fan-out with active user optimization |
    | **Content Diversity** | How to prevent filter bubbles? | Diversity signals in ranking + explicit injection |
    | **A/B Testing** | How to test ranking changes safely? | Multi-armed bandit + counterfactual evaluation |

    ---

    === "🤖 ML Ranking System"

        ## The Challenge

        **Problem:** Generate personalized feed for 2B users. Each user has 300 friends × 0.25 posts/day = 75 candidate posts. Need to rank these in < 200ms.

        **Naive approach:** Chronological feed. **Problem:** Low engagement (users miss important posts from close friends).

        **Facebook's EdgeRank Evolution:**

        1. **2009:** EdgeRank algorithm (edge weight × content type × time decay)
        2. **2013:** ML-based ranking (logistic regression)
        3. **2018:** Deep learning models (DLRM - Deep Learning Recommendation Model)
        4. **2023+:** Transformer-based ranking with multi-objective optimization

        ---

        ## Two-Stage Ranking Architecture

        **Stage 1: Candidate Generation (Fast)**
        - Goal: Reduce 500-1000 posts to top 200
        - Latency budget: 50ms
        - Method: Simple scoring (recency + engagement)

        **Stage 2: ML Ranking (Accurate)**
        - Goal: Rank top 200 posts by predicted engagement
        - Latency budget: 150ms
        - Method: Deep learning model with 100+ features

        ```
        500-1000 candidates → [Stage 1: Simple scoring] → Top 200
        → [Stage 2: ML model] → Top 50 ranked → [Business logic] → Top 20 final
        ```

        ---

        ## Stage 1: Candidate Generation

        ```python
        class CandidateGenerator:
            """Fast filtering of post candidates using simple scoring"""

            def __init__(self, feed_db, cache):
                self.db = feed_db
                self.cache = cache

            def generate_candidates(
                self,
                user_id: str,
                days_back: int = 7,
                target_count: int = 200
            ) -> List[dict]:
                """
                Generate candidate posts for user

                Args:
                    user_id: Target user
                    days_back: How far back to look
                    target_count: Number of candidates to return

                Returns:
                    List of candidate posts with simple scores
                """
                # Get all posts in user's feed (pre-computed via fan-out)
                candidates = self.db.get_feed_candidates(
                    user_id=user_id,
                    since=datetime.utcnow() - timedelta(days=days_back),
                    limit=1000
                )

                logger.info(f"Retrieved {len(candidates)} candidates for user {user_id}")

                # Apply fast filters
                candidates = self._apply_filters(user_id, candidates)

                # Simple scoring for fast ranking
                scored_candidates = []
                for post in candidates:
                    score = self._calculate_simple_score(user_id, post)
                    scored_candidates.append({
                        'post_id': post['post_id'],
                        'score': score,
                        'post': post
                    })

                # Sort by score and take top N
                scored_candidates.sort(key=lambda x: x['score'], reverse=True)

                return scored_candidates[:target_count]

            def _apply_filters(self, user_id: str, candidates: List[dict]) -> List[dict]:
                """Apply business logic filters"""
                filtered = []

                for post in candidates:
                    # Filter deleted posts
                    if post.get('is_deleted'):
                        continue

                    # Filter blocked users
                    if self._is_blocked(user_id, post['user_id']):
                        continue

                    # Filter by privacy settings
                    if not self._can_see_post(user_id, post):
                        continue

                    # Filter already seen (if user has seen in last 7 days)
                    if self._has_seen_recently(user_id, post['post_id']):
                        continue

                    filtered.append(post)

                return filtered

            def _calculate_simple_score(self, user_id: str, post: dict) -> float:
                """
                Calculate simple score for fast ranking

                Factors:
                - Recency (exponential decay)
                - Total engagement (likes + comments + shares)
                - Relationship strength (close friends boosted)
                """
                # Time decay (exponential)
                age_hours = (datetime.utcnow() - post['created_at']).total_seconds() / 3600
                time_score = math.exp(-0.1 * age_hours)  # Decay factor

                # Engagement score
                likes = post.get('likes_count', 0)
                comments = post.get('comments_count', 0)
                shares = post.get('shares_count', 0)
                engagement_score = math.log1p(likes + comments * 2 + shares * 3)

                # Relationship strength (cached)
                author_id = post['user_id']
                relationship_score = self.cache.get(f"rel:{user_id}:{author_id}") or 1.0

                # Combined score
                score = (
                    time_score * 0.3 +
                    engagement_score * 0.4 +
                    relationship_score * 0.3
                )

                return score

            def _is_blocked(self, user_id: str, author_id: str) -> bool:
                """Check if user has blocked author"""
                return self.cache.sismember(f"blocked:{user_id}", author_id)

            def _can_see_post(self, user_id: str, post: dict) -> bool:
                """Check privacy settings"""
                privacy = post.get('privacy', 'friends')

                if privacy == 'public':
                    return True
                elif privacy == 'friends':
                    return self.cache.sismember(f"friends:{post['user_id']}", user_id)
                else:
                    # Custom privacy - check detailed settings
                    return self._check_custom_privacy(user_id, post)

            def _has_seen_recently(self, user_id: str, post_id: str) -> bool:
                """Check if user has seen this post recently"""
                return self.cache.sismember(f"seen:{user_id}", post_id)
        ```

        ---

        ## Stage 2: ML Ranking Model

        **Model Architecture: DLRM (Deep Learning Recommendation Model)**

        ```python
        import torch
        import torch.nn as nn

        class NewsRankingModel(nn.Module):
            """
            Deep learning model for news feed ranking

            Architecture:
            - User features → Embedding → Dense layers
            - Post features → Embedding → Dense layers
            - Interaction features → Dense layers
            - Concat all → Final ranking score
            """

            def __init__(
                self,
                num_users: int,
                num_posts: int,
                embedding_dim: int = 128,
                hidden_dims: List[int] = [512, 256, 128]
            ):
                super().__init__()

                # Embeddings
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.author_embedding = nn.Embedding(num_users, embedding_dim)

                # User dense features
                self.user_dense = nn.Sequential(
                    nn.Linear(50, 128),  # 50 user features
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64)
                )

                # Post dense features
                self.post_dense = nn.Sequential(
                    nn.Linear(30, 128),  # 30 post features
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64)
                )

                # Interaction features
                self.interaction_dense = nn.Sequential(
                    nn.Linear(20, 64),  # 20 interaction features
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32)
                )

                # Combine all features
                total_features = embedding_dim * 2 + 64 + 64 + 32  # 416

                # Deep layers
                layers = []
                prev_dim = total_features
                for dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    prev_dim = dim

                # Output layer (multi-task)
                self.deep_layers = nn.Sequential(*layers)

                # Multi-task heads
                self.engagement_head = nn.Linear(hidden_dims[-1], 1)  # Will user engage?
                self.time_spent_head = nn.Linear(hidden_dims[-1], 1)  # Time spent prediction
                self.share_head = nn.Linear(hidden_dims[-1], 1)      # Will user share?

            def forward(
                self,
                user_ids: torch.Tensor,
                author_ids: torch.Tensor,
                user_features: torch.Tensor,
                post_features: torch.Tensor,
                interaction_features: torch.Tensor
            ):
                """
                Forward pass

                Args:
                    user_ids: [batch_size]
                    author_ids: [batch_size]
                    user_features: [batch_size, 50]
                    post_features: [batch_size, 30]
                    interaction_features: [batch_size, 20]

                Returns:
                    Dictionary with engagement, time_spent, share predictions
                """
                # Embeddings
                user_emb = self.user_embedding(user_ids)  # [batch_size, 128]
                author_emb = self.author_embedding(author_ids)

                # Dense features
                user_dense = self.user_dense(user_features)      # [batch_size, 64]
                post_dense = self.post_dense(post_features)      # [batch_size, 64]
                interaction_dense = self.interaction_dense(interaction_features)  # [batch_size, 32]

                # Concatenate all
                combined = torch.cat([
                    user_emb,
                    author_emb,
                    user_dense,
                    post_dense,
                    interaction_dense
                ], dim=1)  # [batch_size, 416]

                # Deep layers
                deep_out = self.deep_layers(combined)  # [batch_size, 128]

                # Multi-task predictions
                engagement_score = torch.sigmoid(self.engagement_head(deep_out))
                time_spent = self.time_spent_head(deep_out)
                share_score = torch.sigmoid(self.share_head(deep_out))

                return {
                    'engagement': engagement_score,
                    'time_spent': time_spent,
                    'share': share_score
                }
        ```

        ---

        ## Feature Engineering

        **User Features (50 features):**

        ```python
        def extract_user_features(user_id: str) -> np.ndarray:
            """
            Extract user features for ranking

            Categories:
            1. Demographics (age, gender, location)
            2. Activity patterns (post frequency, active hours)
            3. Engagement history (like rate, comment rate, share rate)
            4. Network features (friend count, network density)
            5. Content preferences (interests, page likes)
            """
            user = db.get_user(user_id)

            features = []

            # Demographics (5 features)
            features.extend([
                user['age'] / 100.0,  # Normalized
                1.0 if user['gender'] == 'male' else 0.0,
                hash(user['location']) % 1000 / 1000.0,  # Location hash
                user['account_age_days'] / 3650.0,  # Years on platform
                1.0 if user['is_verified'] else 0.0
            ])

            # Activity patterns (10 features)
            activity = get_user_activity_stats(user_id)
            features.extend([
                activity['posts_per_day'],
                activity['likes_per_day'],
                activity['comments_per_day'],
                activity['shares_per_day'],
                activity['sessions_per_day'],
                activity['avg_session_duration'] / 3600.0,  # Hours
                activity['weekend_activity_ratio'],
                activity['morning_activity_ratio'],
                activity['evening_activity_ratio'],
                activity['mobile_usage_ratio']
            ])

            # Engagement history (15 features)
            engagement = get_engagement_history(user_id)
            features.extend([
                engagement['like_rate'],  # % of posts liked
                engagement['comment_rate'],
                engagement['share_rate'],
                engagement['click_through_rate'],
                engagement['video_completion_rate'],
                engagement['avg_time_per_post'] / 60.0,  # Minutes
                engagement['photo_engagement_rate'],
                engagement['video_engagement_rate'],
                engagement['link_engagement_rate'],
                engagement['friend_post_engagement'],
                engagement['page_post_engagement'],
                engagement['ad_engagement_rate'],
                engagement['negative_feedback_rate'],
                engagement['hide_rate'],
                engagement['report_rate']
            ])

            # Network features (10 features)
            network = get_network_features(user_id)
            features.extend([
                math.log1p(network['friend_count']),
                math.log1p(network['page_likes']),
                math.log1p(network['group_memberships']),
                network['network_density'],
                network['avg_friend_activity'],
                network['influential_friends_ratio'],
                network['mutual_friends_avg'],
                network['network_diversity_score'],
                network['close_friends_count'] / max(network['friend_count'], 1),
                network['content_creator_ratio']
            ])

            # Content preferences (10 features)
            # Top interest categories (0-1 for each)
            interests = get_user_interests(user_id)
            interest_categories = [
                'sports', 'technology', 'entertainment',
                'politics', 'health', 'travel', 'food',
                'fashion', 'gaming', 'education'
            ]
            for category in interest_categories:
                features.append(interests.get(category, 0.0))

            return np.array(features, dtype=np.float32)
        ```

        **Post Features (30 features):**

        ```python
        def extract_post_features(post_id: str) -> np.ndarray:
            """
            Extract post features for ranking

            Categories:
            1. Content type (text, photo, video, link)
            2. Engagement signals (likes, comments, shares)
            3. Author features (popularity, authority)
            4. Temporal features (time of day, day of week)
            5. Content quality (length, media quality, sentiment)
            """
            post = db.get_post(post_id)
            author = db.get_user(post['user_id'])

            features = []

            # Content type (4 features - one-hot)
            content_types = ['status', 'photo', 'video', 'link']
            for ctype in content_types:
                features.append(1.0 if post['content_type'] == ctype else 0.0)

            # Engagement signals (6 features)
            features.extend([
                math.log1p(post['likes_count']),
                math.log1p(post['comments_count']),
                math.log1p(post['shares_count']),
                post['comments_count'] / max(post['likes_count'], 1),  # Comment rate
                post['shares_count'] / max(post['likes_count'], 1),    # Share rate
                post.get('negative_feedback_count', 0) / max(post['views_count'], 1)
            ])

            # Velocity (how fast is it gaining engagement)
            age_hours = (datetime.utcnow() - post['created_at']).total_seconds() / 3600
            features.extend([
                post['likes_count'] / max(age_hours, 1),  # Likes per hour
                post['comments_count'] / max(age_hours, 1),
                post['shares_count'] / max(age_hours, 1)
            ])

            # Author features (7 features)
            features.extend([
                math.log1p(author['friend_count']),
                math.log1p(author['follower_count']),
                1.0 if author['is_verified'] else 0.0,
                author['avg_post_engagement'],
                author['content_quality_score'],
                author['spam_score'],  # Lower is better
                author['authority_score']
            ])

            # Temporal features (4 features)
            created_time = post['created_at']
            features.extend([
                created_time.hour / 24.0,
                created_time.weekday() / 7.0,
                1.0 if created_time.weekday() < 5 else 0.0,  # Is weekday
                age_hours / 168.0  # Age in weeks
            ])

            # Content quality (6 features)
            features.extend([
                min(len(post.get('text_content', '')), 5000) / 5000.0,
                len(post.get('media_urls', [])),
                post.get('has_hashtags', 0.0),
                post.get('has_mentions', 0.0),
                post.get('has_location', 0.0),
                post.get('sentiment_score', 0.5)  # Positive sentiment
            ])

            return np.array(features, dtype=np.float32)
        ```

        **Interaction Features (20 features):**

        ```python
        def extract_interaction_features(user_id: str, post: dict) -> np.ndarray:
            """
            Extract user-post interaction features

            Categories:
            1. Relationship strength (how close are user and author)
            2. Historical interactions (past engagement with author)
            3. Content affinity (does user like this type of content)
            4. Social proof (how many friends engaged)
            """
            author_id = post['user_id']

            features = []

            # Relationship strength (6 features)
            relationship = get_relationship_strength(user_id, author_id)
            features.extend([
                1.0 if relationship['is_friend'] else 0.0,
                1.0 if relationship['is_close_friend'] else 0.0,
                1.0 if relationship['is_family'] else 0.0,
                relationship['interaction_frequency'],  # How often they interact
                relationship['message_frequency'],
                math.log1p(relationship['mutual_friends'])
            ])

            # Historical interactions with author (6 features)
            history = get_user_author_history(user_id, author_id)
            features.extend([
                history['like_rate'],  # % of author's posts user liked
                history['comment_rate'],
                history['share_rate'],
                history['profile_visits'],
                history['avg_time_on_posts'] / 60.0,
                history['last_interaction_days'] / 30.0
            ])

            # Content affinity (5 features)
            affinity = calculate_content_affinity(user_id, post)
            features.extend([
                affinity['content_type_affinity'],  # Does user like this type?
                affinity['topic_affinity'],         # Does user like this topic?
                affinity['time_affinity'],          # Right time for user?
                affinity['format_affinity'],        # Photo/video preference match
                affinity['length_affinity']         # Content length preference
            ])

            # Social proof (3 features)
            social = get_social_proof(user_id, post['post_id'])
            features.extend([
                social['friends_liked_count'],
                social['friends_commented_count'],
                social['friends_shared_count']
            ])

            return np.array(features, dtype=np.float32)
        ```

        ---

        ## Ranking Service

        ```python
        class RankingService:
            """ML-based ranking service for news feed"""

            def __init__(self, model_path: str, feature_store):
                self.model = self._load_model(model_path)
                self.feature_store = feature_store
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.model.eval()

            def rank_posts(
                self,
                user_id: str,
                candidate_posts: List[dict]
            ) -> List[dict]:
                """
                Rank candidate posts for user using ML model

                Args:
                    user_id: Target user
                    candidate_posts: List of candidate posts

                Returns:
                    Ranked list of posts with scores
                """
                if not candidate_posts:
                    return []

                # Extract features (batched)
                user_features = self.feature_store.get_user_features(user_id)

                batch_user_ids = []
                batch_author_ids = []
                batch_user_features = []
                batch_post_features = []
                batch_interaction_features = []

                for post in candidate_posts:
                    post_features = self.feature_store.get_post_features(post['post_id'])
                    interaction_features = extract_interaction_features(user_id, post)

                    batch_user_ids.append(hash(user_id) % 1000000)
                    batch_author_ids.append(hash(post['user_id']) % 1000000)
                    batch_user_features.append(user_features)
                    batch_post_features.append(post_features)
                    batch_interaction_features.append(interaction_features)

                # Convert to tensors
                user_ids_tensor = torch.tensor(batch_user_ids, dtype=torch.long).to(self.device)
                author_ids_tensor = torch.tensor(batch_author_ids, dtype=torch.long).to(self.device)
                user_features_tensor = torch.tensor(batch_user_features, dtype=torch.float32).to(self.device)
                post_features_tensor = torch.tensor(batch_post_features, dtype=torch.float32).to(self.device)
                interaction_features_tensor = torch.tensor(batch_interaction_features, dtype=torch.float32).to(self.device)

                # Model inference
                with torch.no_grad():
                    predictions = self.model(
                        user_ids_tensor,
                        author_ids_tensor,
                        user_features_tensor,
                        post_features_tensor,
                        interaction_features_tensor
                    )

                # Calculate final scores (weighted combination of predictions)
                engagement_scores = predictions['engagement'].cpu().numpy()
                time_spent_scores = predictions['time_spent'].cpu().numpy()
                share_scores = predictions['share'].cpu().numpy()

                # Weighted combination
                final_scores = (
                    engagement_scores * 0.5 +
                    time_spent_scores * 0.3 +
                    share_scores * 0.2
                )

                # Combine posts with scores
                ranked_posts = []
                for i, post in enumerate(candidate_posts):
                    ranked_posts.append({
                        'post_id': post['post_id'],
                        'post': post,
                        'score': float(final_scores[i]),
                        'engagement_prob': float(engagement_scores[i]),
                        'predicted_time': float(time_spent_scores[i]),
                        'share_prob': float(share_scores[i])
                    })

                # Sort by score
                ranked_posts.sort(key=lambda x: x['score'], reverse=True)

                return ranked_posts

            def _load_model(self, model_path: str):
                """Load trained model"""
                model = NewsRankingModel(
                    num_users=3_000_000_000,  # 3B users
                    num_posts=100_000_000,    # Track last 100M posts
                    embedding_dim=128,
                    hidden_dims=[512, 256, 128]
                )
                model.load_state_dict(torch.load(model_path))
                return model
        ```

        ---

        ## Model Training Pipeline

        ```python
        class ModelTrainingPipeline:
            """
            Offline training pipeline for ranking model

            Training data: User-post interactions (likes, comments, shares, time spent)
            Label: Engagement score (weighted combination of interactions)
            """

            def __init__(self, analytics_db):
                self.db = analytics_db

            def prepare_training_data(self, days: int = 30) -> tuple:
                """
                Prepare training data from historical interactions

                Args:
                    days: How many days of data to use

                Returns:
                    (features, labels, sample_weights)
                """
                # Query interactions from ClickHouse
                query = """
                SELECT
                    user_id,
                    post_id,
                    author_id,
                    liked,
                    commented,
                    shared,
                    time_spent_seconds,
                    timestamp
                FROM feed_interactions
                WHERE timestamp >= now() - INTERVAL {days} DAY
                AND time_spent_seconds > 0
                """.format(days=days)

                interactions = self.db.query(query)

                logger.info(f"Retrieved {len(interactions)} interactions")

                # Extract features for each interaction
                features = {
                    'user_ids': [],
                    'author_ids': [],
                    'user_features': [],
                    'post_features': [],
                    'interaction_features': []
                }
                labels = []
                sample_weights = []

                for interaction in interactions:
                    # Extract features
                    user_features = extract_user_features(interaction['user_id'])
                    post_features = extract_post_features(interaction['post_id'])
                    interaction_features = extract_interaction_features(
                        interaction['user_id'],
                        {'user_id': interaction['author_id'], 'post_id': interaction['post_id']}
                    )

                    features['user_ids'].append(hash(interaction['user_id']) % 1000000)
                    features['author_ids'].append(hash(interaction['author_id']) % 1000000)
                    features['user_features'].append(user_features)
                    features['post_features'].append(post_features)
                    features['interaction_features'].append(interaction_features)

                    # Calculate engagement label
                    engagement = (
                        interaction['liked'] * 1.0 +
                        interaction['commented'] * 2.0 +
                        interaction['shared'] * 3.0 +
                        min(interaction['time_spent_seconds'], 300) / 300.0  # Normalize to [0,1]
                    ) / 7.0  # Normalize total score

                    labels.append(engagement)

                    # Weight samples (recent interactions weighted higher)
                    age_days = (datetime.utcnow() - interaction['timestamp']).days
                    weight = math.exp(-0.05 * age_days)
                    sample_weights.append(weight)

                return features, np.array(labels), np.array(sample_weights)

            def train(self, epochs: int = 10, batch_size: int = 1024):
                """Train ranking model"""
                # Prepare data
                features, labels, weights = self.prepare_training_data(days=30)

                # Create DataLoader
                dataset = FeedRankingDataset(features, labels, weights)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Initialize model
                model = NewsRankingModel(
                    num_users=3_000_000_000,
                    num_posts=100_000_000
                )

                # Multi-task loss
                criterion = MultiTaskLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Training loop
                for epoch in range(epochs):
                    total_loss = 0
                    for batch in dataloader:
                        optimizer.zero_grad()

                        predictions = model(
                            batch['user_ids'],
                            batch['author_ids'],
                            batch['user_features'],
                            batch['post_features'],
                            batch['interaction_features']
                        )

                        loss = criterion(predictions, batch['labels'], batch['weights'])
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

                # Save model
                torch.save(model.state_dict(), 'ranking_model.pth')
                logger.info("Model training complete")
        ```

    === "🌊 Fan-out Strategy"

        ## The Challenge

        **Problem:** When user creates post, notify their 300 friends. Active users post frequently, generating massive write load.

        **Naive approaches:**

        1. **Pure fan-out on write:** Write to all 300 friends' feeds immediately
           - **Problem:** Popular users (5000+ friends) create write hotspots

        2. **Pure fan-out on read:** Generate feed on-demand for each user
           - **Problem:** Feed generation too slow (500ms+)

        **Facebook's hybrid approach:** Combine both strategies based on user characteristics.

        ---

        ## Hybrid Fan-out Implementation

        ```python
        class HybridFanoutService:
            """
            Hybrid fan-out strategy for news feed

            Strategy:
            1. Regular users (<1000 friends): Full fan-out on write
            2. Popular users (1000-5000 friends): Partial fan-out (active users only)
            3. Very popular (>5000 friends): No fan-out (pull on read)
            """

            REGULAR_USER_THRESHOLD = 1000
            POPULAR_USER_THRESHOLD = 5000
            ACTIVE_USER_DAYS = 7  # User active in last 7 days

            def __init__(self, graph_db, feed_db, cache):
                self.graph = graph_db
                self.feed = feed_db
                self.cache = cache

            def fanout_post(self, post_id: str, author_id: str):
                """
                Fan-out post to friends using hybrid strategy

                Args:
                    post_id: New post ID
                    author_id: Post author
                """
                friend_count = self.graph.get_friend_count(author_id)

                logger.info(f"Fan-out for user {author_id} with {friend_count} friends")

                if friend_count == 0:
                    return

                if friend_count < self.REGULAR_USER_THRESHOLD:
                    # Regular user: full fan-out
                    self._full_fanout(post_id, author_id)
                elif friend_count < self.POPULAR_USER_THRESHOLD:
                    # Popular user: partial fan-out
                    self._partial_fanout(post_id, author_id)
                else:
                    # Very popular: no fan-out
                    self._no_fanout(post_id, author_id)

            def _full_fanout(self, post_id: str, author_id: str):
                """
                Full fan-out: write to all friends' feeds

                Used for: Regular users (<1000 friends)
                """
                friends = self.graph.get_all_friends(author_id)

                logger.info(f"Full fan-out to {len(friends)} friends")

                # Batch insert into feed_candidates
                batch_size = 1000
                for i in range(0, len(friends), batch_size):
                    batch = friends[i:i+batch_size]

                    self.feed.batch_insert_candidates([
                        {
                            'user_id': friend_id,
                            'post_id': post_id,
                            'created_at': datetime.utcnow(),
                            'source_type': 'friend',
                            'source_id': author_id
                        }
                        for friend_id in batch
                    ])

                    # Invalidate cache for these users
                    self._invalidate_feed_cache(batch)

            def _partial_fanout(self, post_id: str, author_id: str):
                """
                Partial fan-out: write only to active friends

                Used for: Popular users (1000-5000 friends)
                Strategy: Fan-out only to friends active in last 7 days
                """
                # Get active friends
                active_friends = self.graph.get_active_friends(
                    author_id,
                    since=datetime.utcnow() - timedelta(days=self.ACTIVE_USER_DAYS)
                )

                all_friends_count = self.graph.get_friend_count(author_id)

                logger.info(
                    f"Partial fan-out to {len(active_friends)}/{all_friends_count} friends"
                )

                # Fan-out to active friends
                if active_friends:
                    self.feed.batch_insert_candidates([
                        {
                            'user_id': friend_id,
                            'post_id': post_id,
                            'created_at': datetime.utcnow(),
                            'source_type': 'friend',
                            'source_id': author_id
                        }
                        for friend_id in active_friends
                    ])

                    self._invalidate_feed_cache(active_friends)

                # Mark author as "popular" for on-read retrieval
                self.cache.sadd('popular_users', author_id)
                self.cache.zadd(f'popular_posts:{author_id}', {post_id: time.time()})

            def _no_fanout(self, post_id: str, author_id: str):
                """
                No fan-out: just mark for on-read retrieval

                Used for: Very popular users (>5000 friends)
                Strategy: Don't fan-out at all. Friends will pull posts when viewing feed.
                """
                logger.info(f"No fan-out for very popular user {author_id}")

                # Mark author as "celebrity"
                self.cache.sadd('celebrity_users', author_id)

                # Add to author's post list (for on-read retrieval)
                self.cache.zadd(
                    f'user_posts:{author_id}',
                    {post_id: time.time()},
                    nx=True  # Only if not exists
                )

                # Keep only last 100 posts
                self.cache.zremrangebyrank(f'user_posts:{author_id}', 0, -101)

            def _invalidate_feed_cache(self, user_ids: List[str]):
                """Invalidate feed cache for users"""
                pipeline = self.cache.pipeline()
                for user_id in user_ids:
                    pipeline.delete(f'feed:{user_id}:*')
                pipeline.execute()
        ```

        ---

        ## Feed Generation with Hybrid Retrieval

        ```python
        class FeedGenerator:
            """Generate personalized feed using hybrid retrieval"""

            def __init__(self, graph_db, feed_db, post_db, cache):
                self.graph = graph_db
                self.feed = feed_db
                self.posts = post_db
                self.cache = cache

            def generate_feed(
                self,
                user_id: str,
                count: int = 20,
                days_back: int = 7
            ) -> List[dict]:
                """
                Generate personalized feed for user

                Combines:
                1. Pre-computed feed (from fan-out)
                2. On-demand pull (from popular/celebrity friends)

                Args:
                    user_id: Target user
                    count: Number of posts to return
                    days_back: How far back to look

                Returns:
                    List of candidate posts (before ranking)
                """
                candidates = []

                # Part 1: Get pre-computed candidates (from fan-out)
                precomputed = self.feed.get_feed_candidates(
                    user_id=user_id,
                    since=datetime.utcnow() - timedelta(days=days_back),
                    limit=500
                )
                candidates.extend(precomputed)

                logger.info(f"Retrieved {len(precomputed)} pre-computed candidates")

                # Part 2: Pull from popular/celebrity friends
                friends = self.graph.get_friends(user_id)
                popular_friends = []
                celebrity_friends = []

                for friend_id in friends:
                    if self.cache.sismember('celebrity_users', friend_id):
                        celebrity_friends.append(friend_id)
                    elif self.cache.sismember('popular_users', friend_id):
                        popular_friends.append(friend_id)

                # Pull posts from popular friends (last 7 days)
                for friend_id in popular_friends:
                    posts = self.cache.zrevrangebyscore(
                        f'popular_posts:{friend_id}',
                        max=time.time(),
                        min=time.time() - (days_back * 86400),
                        start=0,
                        num=20
                    )
                    for post_id in posts:
                        candidates.append({
                            'post_id': post_id,
                            'source_type': 'friend',
                            'source_id': friend_id
                        })

                logger.info(f"Pulled {len(popular_friends)} popular friends' posts")

                # Pull posts from celebrity friends (last 3 days only - fresher)
                for friend_id in celebrity_friends:
                    posts = self.cache.zrevrangebyscore(
                        f'user_posts:{friend_id}',
                        max=time.time(),
                        min=time.time() - (3 * 86400),  # Last 3 days only
                        start=0,
                        num=10
                    )
                    for post_id in posts:
                        candidates.append({
                            'post_id': post_id,
                            'source_type': 'friend',
                            'source_id': friend_id
                        })

                logger.info(f"Pulled {len(celebrity_friends)} celebrity friends' posts")

                # Part 3: Add page posts (user follows)
                page_posts = self._get_page_posts(user_id, days_back)
                candidates.extend(page_posts)

                logger.info(f"Added {len(page_posts)} page posts")

                # Part 4: Add group posts
                group_posts = self._get_group_posts(user_id, days_back)
                candidates.extend(group_posts)

                logger.info(f"Added {len(group_posts)} group posts")

                # Deduplicate by post_id
                seen = set()
                unique_candidates = []
                for candidate in candidates:
                    if candidate['post_id'] not in seen:
                        seen.add(candidate['post_id'])
                        unique_candidates.append(candidate)

                logger.info(
                    f"Total candidates: {len(unique_candidates)} (after deduplication)"
                )

                return unique_candidates

            def _get_page_posts(self, user_id: str, days_back: int) -> List[dict]:
                """Get posts from pages user follows"""
                pages = self.graph.get_followed_pages(user_id)

                candidates = []
                for page_id in pages[:50]:  # Limit to top 50 pages
                    posts = self.posts.get_page_posts(
                        page_id=page_id,
                        since=datetime.utcnow() - timedelta(days=days_back),
                        limit=5
                    )
                    for post in posts:
                        candidates.append({
                            'post_id': post['post_id'],
                            'source_type': 'page',
                            'source_id': page_id
                        })

                return candidates

            def _get_group_posts(self, user_id: str, days_back: int) -> List[dict]:
                """Get posts from groups user is member of"""
                groups = self.graph.get_user_groups(user_id)

                candidates = []
                for group_id in groups[:20]:  # Limit to top 20 groups
                    posts = self.posts.get_group_posts(
                        group_id=group_id,
                        since=datetime.utcnow() - timedelta(days=days_back),
                        limit=10
                    )
                    for post in posts:
                        candidates.append({
                            'post_id': post['post_id'],
                            'source_type': 'group',
                            'source_id': group_id
                        })

                return candidates
        ```

    === "🎨 Content Diversity"

        ## The Challenge

        **Problem:** Pure engagement-based ranking creates "filter bubbles" - users only see content similar to what they've engaged with before.

        **Issues with pure ML ranking:**

        1. Echo chambers (only see similar political views)
        2. Content type monotony (all photos, no text)
        3. Source concentration (only see posts from same 5 friends)
        4. Recency bias (miss important older posts)

        **Facebook's approach:** Explicitly inject diversity into ranking.

        ---

        ## Diversity Injection

        ```python
        class DiversityOptimizer:
            """
            Apply diversity constraints to ranked feed

            Ensures feed has:
            1. Content type diversity (mix of text, photo, video)
            2. Source diversity (posts from different friends, not just top 5)
            3. Temporal diversity (mix of recent and slightly older posts)
            4. Topic diversity (different topics, not just one interest)
            """

            def __init__(self):
                # Diversity targets (% of feed)
                self.content_type_targets = {
                    'status': 0.20,   # 20% text posts
                    'photo': 0.45,    # 45% photo posts
                    'video': 0.25,    # 25% video posts
                    'link': 0.10      # 10% link posts
                }

                # Maximum posts from single source
                self.max_posts_per_source = 3

                # Recency distribution
                self.recency_targets = {
                    'last_hour': 0.30,      # 30% from last hour
                    'last_6_hours': 0.40,   # 40% from 1-6 hours
                    'last_day': 0.20,       # 20% from 6-24 hours
                    'older': 0.10           # 10% from 1-7 days
                }

            def apply_diversity(
                self,
                ranked_posts: List[dict],
                target_count: int = 20
            ) -> List[dict]:
                """
                Apply diversity constraints to ranked posts

                Args:
                    ranked_posts: Posts ranked by ML model
                    target_count: Number of posts to return

                Returns:
                    Diversified list of posts
                """
                diversified = []

                # Track diversity metrics
                content_type_counts = defaultdict(int)
                source_counts = defaultdict(int)
                recency_counts = defaultdict(int)

                # Iterate through ranked posts and apply constraints
                for post in ranked_posts:
                    if len(diversified) >= target_count:
                        break

                    # Check content type diversity
                    content_type = post['post']['content_type']
                    current_ratio = content_type_counts[content_type] / max(len(diversified), 1)
                    target_ratio = self.content_type_targets.get(content_type, 0.25)

                    if current_ratio >= target_ratio + 0.1:  # Allow 10% variance
                        # Skip if we have too many of this type
                        continue

                    # Check source diversity
                    source_id = post['post']['user_id']
                    if source_counts[source_id] >= self.max_posts_per_source:
                        # Skip if too many posts from this source
                        continue

                    # Check recency diversity
                    age = datetime.utcnow() - post['post']['created_at']
                    recency_bucket = self._get_recency_bucket(age)
                    current_recency_ratio = recency_counts[recency_bucket] / max(len(diversified), 1)
                    target_recency_ratio = self.recency_targets[recency_bucket]

                    if current_recency_ratio >= target_recency_ratio + 0.1:
                        # Skip if we have too many from this recency bucket
                        continue

                    # Post passes all diversity checks - add it
                    diversified.append(post)
                    content_type_counts[content_type] += 1
                    source_counts[source_id] += 1
                    recency_counts[recency_bucket] += 1

                logger.info(
                    f"Diversity applied: {len(diversified)} posts from {len(ranked_posts)} candidates"
                )
                logger.info(f"Content type distribution: {dict(content_type_counts)}")
                logger.info(f"Recency distribution: {dict(recency_counts)}")

                return diversified

            def _get_recency_bucket(self, age: timedelta) -> str:
                """Categorize post age into recency bucket"""
                hours = age.total_seconds() / 3600

                if hours < 1:
                    return 'last_hour'
                elif hours < 6:
                    return 'last_6_hours'
                elif hours < 24:
                    return 'last_day'
                else:
                    return 'older'
        ```

        ---

        ## Topic Diversity

        ```python
        class TopicDiversifier:
            """
            Ensure feed shows diverse topics

            Uses post embeddings to measure topic similarity
            """

            def __init__(self, embedding_service):
                self.embeddings = embedding_service
                self.min_topic_similarity = 0.7  # Cosine similarity threshold

            def apply_topic_diversity(
                self,
                posts: List[dict],
                target_count: int = 20
            ) -> List[dict]:
                """
                Apply topic diversity - avoid too many posts on same topic

                Args:
                    posts: Candidate posts
                    target_count: Number of posts to return

                Returns:
                    Topic-diversified posts
                """
                if not posts:
                    return []

                diversified = []
                selected_embeddings = []

                for post in posts:
                    if len(diversified) >= target_count:
                        break

                    # Get post embedding
                    post_embedding = self.embeddings.get_post_embedding(post['post_id'])

                    # Check similarity with already selected posts
                    is_too_similar = False
                    for selected_emb in selected_embeddings:
                        similarity = self._cosine_similarity(post_embedding, selected_emb)
                        if similarity > self.min_topic_similarity:
                            is_too_similar = True
                            break

                    if not is_too_similar or len(diversified) < 5:
                        # Include post (or include anyway if we have <5 posts)
                        diversified.append(post)
                        selected_embeddings.append(post_embedding)

                logger.info(
                    f"Topic diversity: {len(diversified)} posts (filtered {len(posts) - len(diversified)})"
                )

                return diversified

            def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
                """Calculate cosine similarity between embeddings"""
                return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        ```

        ---

        ## Explicit Diversity Signals

        ```python
        class ExplicitDiversityInjector:
            """
            Explicitly inject diverse content into feed

            Types of injected content:
            1. "People you may know" suggestions
            2. "Pages you might like" recommendations
            3. Trending topics/posts
            4. Local events/news
            5. Memories (on this day X years ago)
            """

            def __init__(self, recommendation_service):
                self.recommendations = recommendation_service

            def inject_diverse_content(
                self,
                user_id: str,
                ranked_posts: List[dict],
                target_count: int = 20
            ) -> List[dict]:
                """
                Inject diverse content into feed

                Strategy: Insert 1-2 "discovery" posts per 20 posts

                Args:
                    user_id: Target user
                    ranked_posts: Already ranked posts
                    target_count: Total posts to return

                Returns:
                    Feed with injected diverse content
                """
                # Calculate injection positions (every 7-10 posts)
                injection_positions = [7, 15]

                # Get diverse content recommendations
                diverse_content = []

                # 1. Friend recommendations
                friend_suggestions = self.recommendations.get_friend_suggestions(
                    user_id,
                    limit=1
                )
                if friend_suggestions:
                    diverse_content.append({
                        'type': 'friend_suggestion',
                        'content': friend_suggestions[0]
                    })

                # 2. Page recommendations
                page_suggestions = self.recommendations.get_page_suggestions(
                    user_id,
                    limit=1
                )
                if page_suggestions:
                    diverse_content.append({
                        'type': 'page_suggestion',
                        'content': page_suggestions[0]
                    })

                # 3. Trending posts
                trending = self.recommendations.get_trending_posts(
                    region=self._get_user_region(user_id),
                    limit=1
                )
                if trending:
                    diverse_content.append({
                        'type': 'trending',
                        'content': trending[0]
                    })

                # Inject diverse content at specified positions
                final_feed = []
                diverse_idx = 0

                for i, post in enumerate(ranked_posts):
                    final_feed.append(post)

                    # Inject diverse content at injection positions
                    if i in injection_positions and diverse_idx < len(diverse_content):
                        final_feed.append(diverse_content[diverse_idx])
                        diverse_idx += 1

                logger.info(f"Injected {diverse_idx} diverse content items")

                return final_feed[:target_count]

            def _get_user_region(self, user_id: str) -> str:
                """Get user's region for localized content"""
                user = db.get_user(user_id)
                return user.get('location', 'US')
        ```

    === "🧪 A/B Testing Infrastructure"

        ## The Challenge

        **Problem:** News feed ranking is critical for engagement. Need to continuously test improvements while minimizing risk.

        **Requirements:**

        1. Test multiple ranking algorithms simultaneously
        2. Measure impact on key metrics (engagement, time spent, satisfaction)
        3. Ramp up winning variants gradually
        4. Automatic rollback if metrics degrade

        ---

        ## A/B Testing Framework

        ```python
        class ABTestFramework:
            """
            A/B testing framework for feed ranking

            Features:
            1. Multi-armed bandit for variant selection
            2. Stratified sampling (ensure even distribution)
            3. Metric tracking and statistical significance
            4. Automatic winner selection
            """

            def __init__(self, analytics_db, cache):
                self.analytics = analytics_db
                self.cache = cache

            def assign_variant(self, user_id: str, experiment_id: str) -> str:
                """
                Assign user to experiment variant

                Uses consistent hashing to ensure:
                1. Same user always gets same variant
                2. Even distribution across variants

                Args:
                    user_id: User to assign
                    experiment_id: Experiment ID

                Returns:
                    Variant name (e.g., 'control', 'treatment_a', 'treatment_b')
                """
                # Check if user already assigned
                cache_key = f"ab_test:{experiment_id}:{user_id}"
                cached_variant = self.cache.get(cache_key)
                if cached_variant:
                    return cached_variant

                # Get experiment config
                experiment = self._get_experiment_config(experiment_id)

                if not experiment or not experiment['is_active']:
                    return 'control'

                # Consistent hashing for assignment
                hash_value = int(hashlib.sha256(
                    f"{experiment_id}:{user_id}".encode()
                ).hexdigest(), 16)

                # Determine variant based on hash and traffic allocation
                cumulative = 0
                for variant_name, allocation in experiment['variants'].items():
                    cumulative += allocation
                    if (hash_value % 100) < cumulative:
                        # Cache assignment
                        self.cache.setex(
                            cache_key,
                            86400 * 30,  # 30 days
                            variant_name
                        )
                        return variant_name

                return 'control'

            def _get_experiment_config(self, experiment_id: str) -> dict:
                """Get experiment configuration"""
                # Example experiment config
                experiments = {
                    'ranking_v2': {
                        'is_active': True,
                        'variants': {
                            'control': 50,       # 50% of users
                            'treatment_a': 25,   # 25% new model
                            'treatment_b': 25    # 25% aggressive ranking
                        },
                        'metrics': [
                            'engagement_rate',
                            'time_spent',
                            'return_rate_7d'
                        ],
                        'start_date': '2026-02-01',
                        'end_date': '2026-02-14'
                    }
                }

                return experiments.get(experiment_id)
        ```

        ---

        ## Metric Tracking

        ```python
        class MetricTracker:
            """Track metrics for A/B test variants"""

            def __init__(self, analytics_db):
                self.analytics = analytics_db

            def track_feed_view(
                self,
                user_id: str,
                experiment_id: str,
                variant: str,
                posts_shown: List[str],
                metadata: dict
            ):
                """
                Track feed view event

                Args:
                    user_id: User who viewed feed
                    experiment_id: Experiment ID
                    variant: Assigned variant
                    posts_shown: List of post IDs shown
                    metadata: Additional metadata (generation_time, etc.)
                """
                event = {
                    'event_type': 'feed_view',
                    'user_id': user_id,
                    'experiment_id': experiment_id,
                    'variant': variant,
                    'timestamp': datetime.utcnow(),
                    'posts_shown': posts_shown,
                    'num_posts': len(posts_shown),
                    'generation_time_ms': metadata.get('generation_time_ms'),
                    'ranking_version': metadata.get('ranking_version')
                }

                # Write to analytics DB (ClickHouse for fast aggregation)
                self.analytics.insert('feed_events', event)

            def track_engagement(
                self,
                user_id: str,
                post_id: str,
                engagement_type: str,  # 'like', 'comment', 'share', 'click'
                experiment_id: str,
                variant: str,
                metadata: dict = None
            ):
                """
                Track engagement event

                Args:
                    user_id: User who engaged
                    post_id: Post that was engaged with
                    engagement_type: Type of engagement
                    experiment_id: Experiment ID
                    variant: Assigned variant
                    metadata: Additional metadata
                """
                event = {
                    'event_type': 'engagement',
                    'user_id': user_id,
                    'post_id': post_id,
                    'engagement_type': engagement_type,
                    'experiment_id': experiment_id,
                    'variant': variant,
                    'timestamp': datetime.utcnow(),
                    'time_spent_seconds': metadata.get('time_spent_seconds') if metadata else None
                }

                self.analytics.insert('engagement_events', event)

            def get_variant_metrics(
                self,
                experiment_id: str,
                start_date: datetime,
                end_date: datetime
            ) -> dict:
                """
                Get metrics for all variants in experiment

                Args:
                    experiment_id: Experiment ID
                    start_date: Start of analysis period
                    end_date: End of analysis period

                Returns:
                    Dictionary with metrics per variant
                """
                query = """
                SELECT
                    variant,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(*) as feed_views,
                    AVG(generation_time_ms) as avg_generation_time,

                    -- Engagement metrics
                    SUM(engagements) / COUNT(*) as engagement_rate,
                    SUM(likes) / COUNT(*) as like_rate,
                    SUM(comments) / COUNT(*) as comment_rate,
                    SUM(shares) / COUNT(*) as share_rate,

                    -- Time spent
                    AVG(total_time_spent) as avg_time_spent,

                    -- Negative signals
                    SUM(hides) / COUNT(*) as hide_rate,
                    SUM(reports) / COUNT(*) as report_rate
                FROM (
                    SELECT
                        fv.user_id,
                        fv.variant,
                        fv.generation_time_ms,
                        COUNT(e.engagement_id) as engagements,
                        SUM(CASE WHEN e.engagement_type = 'like' THEN 1 ELSE 0 END) as likes,
                        SUM(CASE WHEN e.engagement_type = 'comment' THEN 1 ELSE 0 END) as comments,
                        SUM(CASE WHEN e.engagement_type = 'share' THEN 1 ELSE 0 END) as shares,
                        SUM(CASE WHEN e.engagement_type = 'hide' THEN 1 ELSE 0 END) as hides,
                        SUM(CASE WHEN e.engagement_type = 'report' THEN 1 ELSE 0 END) as reports,
                        SUM(e.time_spent_seconds) as total_time_spent
                    FROM feed_events fv
                    LEFT JOIN engagement_events e
                        ON fv.user_id = e.user_id
                        AND e.timestamp BETWEEN fv.timestamp AND fv.timestamp + INTERVAL 1 HOUR
                    WHERE fv.experiment_id = '{experiment_id}'
                        AND fv.timestamp BETWEEN '{start_date}' AND '{end_date}'
                    GROUP BY fv.user_id, fv.variant, fv.generation_time_ms
                )
                GROUP BY variant
                """.format(
                    experiment_id=experiment_id,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )

                results = self.analytics.query(query)

                # Format results
                metrics = {}
                for row in results:
                    metrics[row['variant']] = {
                        'unique_users': row['unique_users'],
                        'feed_views': row['feed_views'],
                        'avg_generation_time': row['avg_generation_time'],
                        'engagement_rate': row['engagement_rate'],
                        'like_rate': row['like_rate'],
                        'comment_rate': row['comment_rate'],
                        'share_rate': row['share_rate'],
                        'avg_time_spent': row['avg_time_spent'],
                        'hide_rate': row['hide_rate'],
                        'report_rate': row['report_rate']
                    }

                return metrics
        ```

        ---

        ## Statistical Significance Testing

        ```python
        from scipy import stats

        class StatisticalAnalyzer:
            """Statistical analysis for A/B test results"""

            def __init__(self, significance_level: float = 0.05):
                self.alpha = significance_level

            def is_significant(
                self,
                control_metrics: dict,
                treatment_metrics: dict,
                metric_name: str
            ) -> dict:
                """
                Test if difference between variants is statistically significant

                Uses two-sample t-test for continuous metrics
                Uses chi-square test for rate metrics

                Args:
                    control_metrics: Metrics for control variant
                    treatment_metrics: Metrics for treatment variant
                    metric_name: Which metric to test

                Returns:
                    Dictionary with significance test results
                """
                control_value = control_metrics[metric_name]
                treatment_value = treatment_metrics[metric_name]

                # Calculate lift
                lift = (treatment_value - control_value) / control_value

                # Sample sizes
                n_control = control_metrics['unique_users']
                n_treatment = treatment_metrics['unique_users']

                # Perform appropriate test based on metric type
                if metric_name in ['engagement_rate', 'like_rate', 'comment_rate', 'share_rate']:
                    # Rate metric - use chi-square test
                    control_successes = int(control_value * n_control)
                    treatment_successes = int(treatment_value * n_treatment)

                    contingency_table = [
                        [control_successes, n_control - control_successes],
                        [treatment_successes, n_treatment - treatment_successes]
                    ]

                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                else:
                    # Continuous metric - use t-test
                    # Note: Need raw data for proper t-test, using approximation here
                    t_stat, p_value = stats.ttest_ind_from_stats(
                        mean1=control_value,
                        std1=control_value * 0.1,  # Approximation
                        nobs1=n_control,
                        mean2=treatment_value,
                        std2=treatment_value * 0.1,
                        nobs2=n_treatment
                    )

                is_sig = p_value < self.alpha

                return {
                    'metric': metric_name,
                    'control_value': control_value,
                    'treatment_value': treatment_value,
                    'lift': lift,
                    'lift_percent': lift * 100,
                    'p_value': p_value,
                    'is_significant': is_sig,
                    'confidence_level': (1 - self.alpha) * 100
                }
        ```

        ---

        ## Multi-Armed Bandit

        ```python
        class MultiArmedBandit:
            """
            Thompson Sampling for dynamic traffic allocation

            Automatically shift traffic to winning variants during experiment
            """

            def __init__(self):
                # Beta distribution parameters for each variant
                self.alpha = defaultdict(lambda: 1.0)  # Successes + 1
                self.beta = defaultdict(lambda: 1.0)   # Failures + 1

            def update(self, variant: str, success: bool):
                """
                Update bandit with observation

                Args:
                    variant: Variant name
                    success: Whether action was successful (e.g., user engaged)
                """
                if success:
                    self.alpha[variant] += 1
                else:
                    self.beta[variant] += 1

            def select_variant(self, available_variants: List[str]) -> str:
                """
                Select variant using Thompson Sampling

                Args:
                    available_variants: List of variant names

                Returns:
                    Selected variant
                """
                samples = {}

                for variant in available_variants:
                    # Sample from Beta distribution
                    samples[variant] = np.random.beta(
                        self.alpha[variant],
                        self.beta[variant]
                    )

                # Select variant with highest sample
                return max(samples, key=samples.get)

            def get_variant_probabilities(self, available_variants: List[str]) -> dict:
                """Get probability of each variant being best"""
                probabilities = {}

                for variant in available_variants:
                    alpha = self.alpha[variant]
                    beta = self.beta[variant]

                    # Mean of Beta distribution
                    probabilities[variant] = alpha / (alpha + beta)

                return probabilities
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling News Feed from 100M to 2B DAU.

    **Scaling challenges at 2B DAU:**

    - **Read throughput:** 880K read QPS (feed + profile + search)
    - **Write throughput:** 305K write QPS (posts + interactions)
    - **Storage:** 3.07 exabytes of data
    - **ML inference:** 578K ranking calls per second
    - **Cache capacity:** 5.6 TB of hot data

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **ML Ranker** | ✅ Yes | Model serving: 1000 GPU servers, batch inference, model quantization |
    | **Cassandra writes** | ✅ Yes | 1000 nodes, SSD storage, tuned consistency (ONE for writes) |
    | **Feed generation** | ✅ Yes | Aggressive caching (70% hit rate), pre-computation, Redis cluster (200 nodes) |
    | **Feature Store** | ✅ Yes | Distributed feature store (Feast), Redis + S3 backing |
    | **Graph DB** | 🟡 Moderate | Sharded by user_id, cached hot edges, 100 nodes |
    | **Kafka throughput** | 🟡 Approaching | 100 brokers, partitioning by user_id, compression |

    ---

    ## ML Inference Optimization

    **Challenge:** 578K ranking calls/sec × 200 candidates × 100 features = 11.5B feature lookups/sec

    **Solutions:**

    1. **Model quantization:** INT8 instead of FP32 (4x speedup, minimal accuracy loss)
    2. **Batch inference:** Process 100 users at once (GPU utilization)
    3. **Feature caching:** Cache user features for 5 minutes
    4. **Model distillation:** Smaller student model (80% accuracy, 10x faster)
    5. **GPU serving:** 1000 NVIDIA A100 GPUs

    ```python
    class OptimizedRankingService:
        """Optimized ML ranking with batching and caching"""

        def __init__(self, model_path: str):
            self.model = self._load_quantized_model(model_path)
            self.feature_cache = redis.Redis(host='redis-features')
            self.batch_size = 100
            self.batch_timeout_ms = 50  # Max wait for batch

        async def rank_posts_batch(
            self,
            requests: List[tuple]  # [(user_id, candidate_posts), ...]
        ) -> List[List[dict]]:
            """Batch ranking for multiple users"""
            # Fetch all features in parallel
            feature_tasks = []
            for user_id, posts in requests:
                feature_tasks.append(self._get_features_cached(user_id, posts))

            all_features = await asyncio.gather(*feature_tasks)

            # Batch inference on GPU
            with torch.no_grad():
                predictions = self.model.forward_batch(all_features)

            # Distribute results back to requests
            results = []
            for i, (user_id, posts) in enumerate(requests):
                ranked = self._combine_posts_with_scores(
                    posts,
                    predictions[i]
                )
                results.append(ranked)

            return results

        async def _get_features_cached(self, user_id: str, posts: List[dict]):
            """Get features with caching"""
            cache_key = f"features:{user_id}"
            cached = self.feature_cache.get(cache_key)

            if cached:
                return pickle.loads(cached)

            features = extract_all_features(user_id, posts)
            self.feature_cache.setex(
                cache_key,
                300,  # 5 minutes
                pickle.dumps(features)
            )
            return features
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 2B DAU:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $432,000 (3000 × m5.2xlarge) |
    | **GPU servers (ML)** | $1,200,000 (1000 × A100 instances) |
    | **Cassandra cluster** | $540,000 (1000 nodes × r5.4xlarge) |
    | **Redis cache** | $216,000 (200 nodes × r5.2xlarge) |
    | **Graph DB** | $108,000 (100 nodes) |
    | **Kafka cluster** | $54,000 (100 brokers) |
    | **ClickHouse analytics** | $108,000 (200 nodes) |
    | **S3 storage** | $69,000 (3.07 EB) |
    | **CDN** | $390,000 (4,500 TB egress) |
    | **Total** | **$3,117,000/month** (~$37M/year) |

    **Optimization strategies:**

    1. **Reserved instances:** 40% cost savings on EC2
    2. **Spot instances:** Use for batch ML training (70% savings)
    3. **S3 tiering:** Move old media to Glacier (90% savings)
    4. **CDN optimization:** Aggressive caching, WebP images

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Feed Latency (P95)** | < 500ms | > 1000ms |
    | **ML Ranking Latency (P95)** | < 150ms | > 300ms |
    | **Post Creation Latency (P95)** | < 1s | > 3s |
    | **Cache Hit Rate** | > 70% | < 60% |
    | **Engagement Rate** | > 8% | < 6% |
    | **Ranking Model AUC** | > 0.75 | < 0.70 |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **ML-first ranking:** Two-stage ranking (candidate gen + ML scoring)
    2. **Hybrid fan-out:** Full write for regular, partial for popular, on-read for celebrities
    3. **Content diversity:** Explicit diversity constraints to prevent filter bubbles
    4. **A/B testing infrastructure:** Multi-armed bandit for continuous experimentation
    5. **Feature store:** Centralized feature serving for ML models
    6. **Cassandra for posts:** Write-optimized, time-series queries
    7. **Aggressive caching:** 70% Redis hit rate for feeds

    ---

    ## Interview Tips

    ✅ **Emphasize ML complexity** - Ranking is the core challenge, not just storage

    ✅ **Discuss personalization** - Every user's feed is different

    ✅ **Content diversity is critical** - Prevent echo chambers

    ✅ **A/B testing infrastructure** - Continuous experimentation required

    ✅ **Scale of ML inference** - 578K ranking calls/sec is massive

    ✅ **Fan-out strategy evolution** - Explain why hybrid is better than pure

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How does ranking work?"** | Two-stage: candidate generation (500→200) + ML ranking (200→20). ML model predicts engagement, time spent, shares. |
    | **"How to prevent filter bubbles?"** | Diversity constraints: content type, source, topic, recency. Explicit injection of diverse content. |
    | **"How to handle popular users?"** | Hybrid fan-out: regular users get full write, popular users get partial, celebrities get on-read. |
    | **"How to test ranking changes?"** | A/B testing with multi-armed bandit. Track engagement, time spent, satisfaction. Statistical significance testing. |
    | **"What features for ML ranking?"** | User (50): demographics, activity, engagement history. Post (30): content type, engagement, author. Interaction (20): relationship, history. |
    | **"How to scale ML inference?"** | Batch inference, GPU serving, model quantization, feature caching, 1000 GPU servers. |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Facebook, LinkedIn, Reddit, Instagram, Medium, Pinterest

---

*Master this problem and you'll be ready for: LinkedIn Feed, Reddit Homepage, Instagram Feed, Pinterest Home Feed, TikTok For You*
