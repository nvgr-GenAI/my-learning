# Design Twitter

A microblogging platform where users can post tweets (140-280 characters), follow users, and view a real-time timeline of tweets from people they follow.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 400M daily active users, 500M tweets/day, 1B timeline requests/day |
| **Key Challenges** | Real-time timeline, tweet fan-out, trending topics, massive write throughput |
| **Core Concepts** | Fan-out on write/read hybrid, timeline generation, search, real-time updates |
| **Companies** | Twitter, Meta, Google, Amazon, LinkedIn, Reddit |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Post Tweet** | Users can post 280-character tweets | P0 (Must have) |
    | **View Timeline** | Home timeline showing tweets from followed users | P0 (Must have) |
    | **Follow/Unfollow** | Users can follow/unfollow others | P0 (Must have) |
    | **Like/Retweet** | Users can like and retweet tweets | P0 (Must have) |
    | **Reply** | Users can reply to tweets (threads) | P1 (Should have) |
    | **Search** | Search tweets, users, hashtags | P1 (Should have) |
    | **Trending Topics** | Real-time trending hashtags | P1 (Should have) |
    | **Notifications** | Notify of mentions, likes, retweets | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Direct messaging
    - Twitter Spaces (audio rooms)
    - Live video streaming
    - Ads system
    - Content moderation (ML models)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users expect near-constant access |
    | **Latency (Timeline)** | < 200ms p95 | Fast timeline loading critical for UX |
    | **Latency (Tweet Post)** | < 1s | Quick feedback for user engagement |
    | **Consistency** | Eventual consistency | Brief delays acceptable (tweets may not appear immediately) |
    | **Scalability** | Billions of tweets per day | Must handle viral events (Super Bowl, elections) |
    | **Real-time** | New tweets appear within 5 seconds | Users expect near-instant updates |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 400M
    Monthly Active Users (MAU): 800M

    Tweet creation:
    - Tweets per DAU: ~1.25 tweets/day (power users tweet more)
    - Daily tweets: 400M × 1.25 = 500M tweets/day
    - Tweet QPS: 500M / 86,400 = ~5,800 tweets/sec
    - Peak QPS: 3x average = ~17,400 tweets/sec (during events)

    Timeline views:
    - Timeline views per DAU: ~10 views/day
    - Daily timeline requests: 400M × 10 = 4B requests/day
    - Timeline QPS: 4B / 86,400 = ~46,300 req/sec

    Likes/Retweets:
    - Likes per DAU: ~20 likes/day
    - Daily likes: 400M × 20 = 8B likes/day
    - Like QPS: 8B / 86,400 = ~92,600 likes/sec

    Total Read QPS: ~139K (timeline + profile views + search)
    Total Write QPS: ~115K (tweets + likes + retweets + follows)
    Read/Write ratio: 1.2:1 (more balanced than Instagram)
    ```

    ### Storage Estimates

    ```
    Tweet storage:
    - Tweet content: 280 chars × 2 bytes (UTF-16) = 560 bytes
    - Metadata: 1 KB (tweet_id, user_id, timestamp, media_urls, counts)
    - Total per tweet: ~1.5 KB

    For 10 years:
    - Tweets: 500M/day × 365 × 10 = 1.825 trillion tweets
    - Storage: 1.825T × 1.5 KB = 2.74 PB

    Media storage (photos/videos):
    - 30% of tweets have media (~150M media/day)
    - Average size: 2 MB (compressed photo/video)
    - Daily: 150M × 2 MB = 300 TB/day
    - 10 years: 300 TB × 365 × 10 = 1,095 PB (1.1 exabytes)

    User data:
    - 800M users × 10 KB = 8 TB

    Social graph (followers/following):
    - 800M users × 200 followers (avg) × 16 bytes = 2.56 TB

    Total: 2.74 PB (tweets) + 1.1 EB (media) + 10 TB (users/graph) ≈ 1.1 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Tweet ingress:
    - 5,800 tweets/sec × 1.5 KB = 8.7 MB/sec ≈ 70 Mbps
    - Media uploads: 150M media/day = 1,736 media/sec × 2 MB = 3.47 GB/sec ≈ 28 Gbps

    Timeline egress:
    - 46,300 timeline/sec × 50 tweets × 1.5 KB = 3.47 GB/sec ≈ 28 Gbps (text only)
    - Media downloads: 10x more than uploads = 280 Gbps

    Total ingress: ~28 Gbps
    Total egress: ~308 Gbps (CDN critical)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot tweets (last 24 hours):
    - Tweets: 500M × 1.5 KB = 750 GB
    - Cache 20% hottest: 150 GB

    User sessions:
    - 40M concurrent users × 10 KB = 400 GB

    Timeline cache:
    - 10M most active users × 50 tweets × 1.5 KB = 750 GB

    Trending topics:
    - Real-time aggregations: 10 GB

    Total cache: 150 GB + 400 GB + 750 GB + 10 GB ≈ 1.3 TB
    ```

    ---

    ## Key Assumptions

    1. Average 280 characters per tweet (max limit)
    2. 400M DAU, ~50% daily engagement from MAU
    3. More balanced read/write ratio (1.2:1) compared to Instagram
    4. Real-time is critical (< 5 seconds for tweet propagation)
    5. Eventual consistency acceptable (brief delays tolerated)
    6. Celebrity tweets require special handling (millions of followers)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Hybrid fan-out:** Fan-out on write for regular users, on read for celebrities
    2. **Write-heavy system:** Optimize for high write throughput
    3. **Real-time updates:** WebSocket connections for instant tweet delivery
    4. **Eventual consistency:** Prioritize availability over strong consistency
    5. **Search-first design:** Full-text search critical for discovery

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
            Tweet_API[Tweet Service<br/>Post/delete tweets]
            Timeline_API[Timeline Service<br/>Home/user timeline]
            Social_API[Social Service<br/>Follow/unfollow]
            Search_API[Search Service<br/>Tweets/users/hashtags]
            Trending_API[Trending Service<br/>Real-time trends]
        end

        subgraph "Real-time Layer"
            WebSocket[WebSocket Server<br/>Real-time updates]
            Fanout[Fanout Service<br/>Deliver tweets to followers]
        end

        subgraph "Data Processing"
            Fanout_Worker[Fanout Worker<br/>Timeline updates]
            Trend_Worker[Trend Aggregator<br/>Trending topics]
            Search_Indexer[Search Indexer<br/>Elasticsearch]
            Analytics[Analytics Pipeline<br/>Kafka Streams]
        end

        subgraph "Caching"
            Redis_Timeline[Redis<br/>Timeline cache]
            Redis_Tweet[Redis<br/>Tweet cache]
            Redis_User[Redis<br/>User cache]
            Redis_Trend[Redis<br/>Trending topics]
        end

        subgraph "Storage"
            Tweet_DB[(Tweet DB<br/>Cassandra<br/>Sharded)]
            User_DB[(User DB<br/>PostgreSQL<br/>Sharded)]
            Graph_DB[(Social Graph<br/>Neo4j)]
            Search_DB[(Elasticsearch<br/>Full-text search)]
            S3[Object Storage<br/>S3<br/>Media files]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        CDN --> S3

        LB --> Tweet_API
        LB --> Timeline_API
        LB --> Social_API
        LB --> Search_API
        LB --> Trending_API
        LB --> WebSocket

        Tweet_API --> Kafka
        Tweet_API --> Tweet_DB
        Tweet_API --> S3

        Kafka --> Fanout_Worker
        Kafka --> Trend_Worker
        Kafka --> Search_Indexer
        Kafka --> Analytics

        Fanout_Worker --> Fanout
        Fanout_Worker --> Redis_Timeline
        Fanout_Worker --> Tweet_DB

        Timeline_API --> Redis_Timeline
        Timeline_API --> Tweet_DB
        Search_API --> Search_DB

        Trending_API --> Redis_Trend
        Trend_Worker --> Redis_Trend

        Social_API --> Graph_DB
        Social_API --> User_DB

        Fanout --> WebSocket

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Timeline fill:#fff4e1
        style Redis_Tweet fill:#fff4e1
        style Redis_User fill:#fff4e1
        style Redis_Trend fill:#fff4e1
        style Tweet_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Graph_DB fill:#e1f5e1
        style Search_DB fill:#e8eaf6
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Kafka** | High-throughput event streaming (115K write QPS), reliable delivery | RabbitMQ (can't handle throughput), direct fan-out (no retry/replay) |
    | **Cassandra (Tweet DB)** | Write-optimized, handles 115K write QPS, horizontal scaling | MySQL (write bottleneck), MongoDB (consistency issues at scale) |
    | **Elasticsearch** | Fast full-text search (<100ms), complex queries (hashtags, mentions) | Database LIKE queries (too slow), custom search (reinventing wheel) |
    | **Graph Database** | Fast follower queries, social graph traversal | SQL joins (too slow for millions of followers), custom graph (complex) |
    | **WebSocket** | Real-time tweet delivery (< 1s latency), bidirectional communication | Polling (wasteful, high latency), Server-Sent Events (one-way only) |
    | **Redis** | Timeline cache (< 10ms reads), hot tweet data | No cache (Cassandra can't handle 139K read QPS), Memcached (limited features) |

    **Key Trade-off:** We chose **write availability over consistency**. Tweets may not appear in all timelines simultaneously, but system remains available during failures.

    ---

    ## API Design

    ### 1. Post Tweet

    **Request:**
    ```http
    POST /api/v1/tweets
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "content": "Hello World! #firsttweet",
      "media_ids": ["media_123", "media_456"],  // Optional
      "reply_to": null,                         // Optional (for replies)
      "quote_tweet": null                       // Optional (for quotes)
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "tweet_id": "1234567890",
      "user_id": "user123",
      "username": "john_doe",
      "content": "Hello World! #firsttweet",
      "created_at": "2026-01-29T10:30:00Z",
      "likes_count": 0,
      "retweets_count": 0,
      "replies_count": 0,
      "media_urls": [
        "https://cdn.twitter.com/media/123.jpg",
        "https://cdn.twitter.com/media/456.jpg"
      ]
    }
    ```

    **Design Notes:**

    - Return immediately after saving tweet (don't wait for fan-out)
    - Extract hashtags and mentions automatically
    - Rate limit: 300 tweets per 3 hours per user
    - Validate content length (280 characters)

    ---

    ### 2. Get Home Timeline

    **Request:**
    ```http
    GET /api/v1/timeline/home?cursor=1234567890&count=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tweets": [
        {
          "tweet_id": "1234567890",
          "user": {
            "user_id": "user456",
            "username": "jane_smith",
            "profile_pic": "https://cdn.twitter.com/profiles/user456.jpg"
          },
          "content": "Just finished my morning run 🏃",
          "created_at": "2026-01-29T10:30:00Z",
          "likes_count": 152,
          "retweets_count": 23,
          "replies_count": 8,
          "liked_by_user": false,
          "retweeted_by_user": false
        },
        // ... 19 more tweets
      ],
      "next_cursor": "1234567800",
      "has_more": true
    }
    ```

    **Design Notes:**

    - Cursor-based pagination (not offset, better for real-time data)
    - Pre-computed timeline (fan-out on write for regular users)
    - Includes aggregated counts and user state
    - Real-time updates via WebSocket

    ---

    ### 3. Search Tweets

    **Request:**
    ```http
    GET /api/v1/search?q=machine+learning&type=recent&count=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tweets": [
        // ... tweet objects matching query
      ],
      "next_cursor": "...",
      "search_metadata": {
        "query": "machine learning",
        "result_type": "recent",
        "count": 20,
        "completed_in": 0.045
      }
    }
    ```

    **Design Notes:**

    - Elasticsearch for full-text search
    - Support filters: from:user, since:date, hashtag:tag
    - Real-time indexing (< 5 seconds for new tweets to be searchable)

    ---

    ## Database Schema

    ### Tweets (Cassandra)

    ```sql
    -- Tweets table
    CREATE TABLE tweets (
        tweet_id BIGINT PRIMARY KEY,
        user_id BIGINT,
        content TEXT,
        created_at TIMESTAMP,
        likes_count COUNTER,
        retweets_count COUNTER,
        replies_count COUNTER,
        media_urls LIST<TEXT>,
        hashtags LIST<TEXT>,
        mentions LIST<TEXT>,
        reply_to BIGINT,
        quote_tweet BIGINT,
        -- Efficient writes, distributed by tweet_id
    );

    -- User tweets (for profile timeline)
    CREATE TABLE user_tweets (
        user_id BIGINT,
        created_at TIMESTAMP,
        tweet_id BIGINT,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Home timeline (fan-out on write)
    CREATE TABLE home_timeline (
        user_id BIGINT,
        created_at TIMESTAMP,
        tweet_id BIGINT,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Likes (separate table for scalability)
    CREATE TABLE tweet_likes (
        tweet_id BIGINT,
        user_id BIGINT,
        created_at TIMESTAMP,
        PRIMARY KEY (tweet_id, user_id)
    );
    ```

    **Why Cassandra:**

    - **High write throughput:** 115K write QPS (tweets + likes + retweets)
    - **Linear scalability:** Add nodes without downtime
    - **No single point of failure:** Multi-master, every node can write
    - **Time-series optimized:** Clustering by created_at for fast timeline queries

    ---

    ### Users (PostgreSQL)

    ```sql
    -- Users table (sharded by user_id)
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        username VARCHAR(15) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        display_name VARCHAR(50),
        bio VARCHAR(160),
        profile_pic_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        follower_count INT DEFAULT 0,
        following_count INT DEFAULT 0,
        tweet_count INT DEFAULT 0,
        verified BOOLEAN DEFAULT FALSE,
        INDEX idx_username (username),
        INDEX idx_email (email)
    ) PARTITION BY HASH (user_id);
    ```

    ---

    ### Social Graph (Neo4j)

    ```cypher
    // Follow relationship
    CREATE (u1:User {user_id: 123})-[:FOLLOWS {created_at: 1643712000}]->(u2:User {user_id: 456})

    // Query: Get followers
    MATCH (follower:User)-[:FOLLOWS]->(user:User {user_id: 123})
    RETURN follower.user_id

    // Query: Get following
    MATCH (user:User {user_id: 123})-[:FOLLOWS]->(following:User)
    RETURN following.user_id

    // Query: Check if follows
    MATCH (u1:User {user_id: 123})-[:FOLLOWS]->(u2:User {user_id: 456})
    RETURN COUNT(*) > 0 as follows
    ```

    ---

    ## Data Flow Diagrams

    ### Tweet Post Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Tweet_API
        participant Tweet_DB
        participant Kafka
        participant Fanout_Worker
        participant Timeline_DB
        participant WebSocket

        Client->>Tweet_API: POST /api/v1/tweets
        Tweet_API->>Tweet_API: Validate, extract hashtags/mentions
        Tweet_API->>Tweet_DB: INSERT tweet
        Tweet_DB-->>Tweet_API: tweet_id

        Tweet_API->>Kafka: Publish tweet_created event
        Tweet_API-->>Client: 201 Created (tweet_id)

        Kafka->>Fanout_Worker: Process tweet_created
        Fanout_Worker->>Graph_DB: Get followers (e.g., 10K followers)

        alt Regular user (<100K followers)
            Fanout_Worker->>Timeline_DB: Batch INSERT into home_timeline (fan-out on write)
            Fanout_Worker->>WebSocket: Notify online followers
        else Celebrity (>100K followers)
            Fanout_Worker->>Timeline_DB: Skip fan-out (will pull on read)
            Fanout_Worker->>WebSocket: Notify subset of active followers
        end

        WebSocket->>Client: Real-time tweet notification
    ```

    **Flow Explanation:**

    1. **Save tweet** - Store in Cassandra, return immediately (< 100ms)
    2. **Publish event** - Kafka event for async processing
    3. **Fan-out to followers** - Write to followers' timelines (regular users only)
    4. **Real-time delivery** - WebSocket push to online followers
    5. **Search indexing** - Elasticsearch index (async, within 5 seconds)

    ---

    ### Timeline Generation Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Timeline_API
        participant Redis
        participant Timeline_DB
        participant Graph_DB
        participant Tweet_DB

        Client->>Timeline_API: GET /api/v1/timeline/home
        Timeline_API->>Timeline_API: Authenticate user

        Timeline_API->>Redis: GET timeline:user123
        alt Cache HIT (80% of requests)
            Redis-->>Timeline_API: Cached tweet IDs
            Timeline_API->>Tweet_DB: Batch get tweet details
            Tweet_DB-->>Timeline_API: Tweet objects
        else Cache MISS (20% of requests)
            Redis-->>Timeline_API: null

            Timeline_API->>Graph_DB: Get following list
            Graph_DB-->>Timeline_API: Following IDs

            alt Regular user feed (all following are regular users)
                Timeline_API->>Timeline_DB: SELECT from home_timeline
                Timeline_DB-->>Timeline_API: Pre-computed tweet IDs
            else Mixed feed (some celebrities in following)
                Timeline_API->>Timeline_DB: Get regular tweets (pre-computed)
                Timeline_API->>Tweet_DB: Pull celebrity tweets (on demand)
                Timeline_API->>Timeline_API: Merge and sort
            end

            Timeline_API->>Tweet_DB: Batch get tweet details
            Tweet_DB-->>Timeline_API: Tweet objects

            Timeline_API->>Redis: SET timeline:user123 (TTL: 60s)
        end

        Timeline_API->>Timeline_API: Rank tweets (ML algorithm)
        Timeline_API-->>Client: 200 OK (20 tweets)
    ```

    **Flow Explanation:**

    1. **Check cache** - Redis cache hit for 80% of requests (< 20ms)
    2. **Cache miss** - Query pre-computed timeline or hybrid approach
    3. **Fetch tweet details** - Batch query for efficiency
    4. **Rank tweets** - ML-based ranking (engagement prediction)
    5. **Cache result** - Store in Redis for 60 seconds

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Twitter subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Fan-out Strategy** | How to deliver tweets to millions of followers? | Hybrid: fan-out on write (regular) + on read (celebrity) |
    | **Real-time Updates** | How to show new tweets instantly? | WebSocket + long polling fallback |
    | **Trending Topics** | How to calculate trending hashtags in real-time? | Stream processing + time-decay algorithm |
    | **Search System** | How to search billions of tweets in <100ms? | Elasticsearch sharding + caching |

    ---

    === "📊 Fan-out Strategy"

        ## The Challenge

        **Problem:** When user posts tweet, deliver it to all followers. Celebrities have 100M+ followers!

        **Naive approach:** On timeline request, query tweets from all following. **Doesn't scale** (200 queries per timeline!).

        **Twitter's evolution:**

        1. **2010:** Fan-out on write (for everyone) - celebrity tweets took 5+ minutes to propagate
        2. **2012:** Fan-out on read (for celebrities) - timeline generation slow (500ms+)
        3. **2016+:** Hybrid approach (best of both)

        ---

        ## Hybrid Fan-out Implementation

        ```python
        class TweetFanoutService:
            """Hybrid fan-out: write for regular users, read for celebrities"""

            CELEBRITY_THRESHOLD = 100_000  # Followers
            ACTIVE_FOLLOWER_THRESHOLD = 10_000  # For partial fan-out

            def __init__(self, graph_db, timeline_db, websocket):
                self.graph = graph_db
                self.timeline = timeline_db
                self.ws = websocket

            def fanout_tweet(self, tweet_id: str, user_id: str):
                """
                Fan-out tweet to followers using hybrid strategy

                Args:
                    tweet_id: ID of new tweet
                    user_id: Tweet author
                """
                follower_count = self.graph.get_follower_count(user_id)

                if follower_count == 0:
                    logger.info(f"No followers to fanout for user {user_id}")
                    return

                if follower_count < self.CELEBRITY_THRESHOLD:
                    # Regular user: fan-out on write (full)
                    self._fanout_on_write(tweet_id, user_id)
                elif follower_count < 1_000_000:
                    # Mid-tier celebrity: partial fan-out (active followers only)
                    self._partial_fanout(tweet_id, user_id)
                else:
                    # Mega celebrity: no fan-out (pull on read)
                    self._fanout_on_read_marker(tweet_id, user_id)

            def _fanout_on_write(self, tweet_id: str, user_id: str):
                """
                Full fan-out: write tweet to all followers' timelines

                Used for: Regular users (< 100K followers)
                """
                followers = self.graph.get_all_followers(user_id)
                logger.info(f"Full fanout to {len(followers)} followers")

                # Batch write to timelines
                batch_size = 1000
                for i in range(0, len(followers), batch_size):
                    batch = followers[i:i+batch_size]

                    # Write to home_timeline table
                    self.timeline.batch_insert([
                        {
                            'user_id': follower_id,
                            'created_at': datetime.utcnow(),
                            'tweet_id': tweet_id
                        }
                        for follower_id in batch
                    ])

                    # Notify online followers via WebSocket
                    self._notify_online_followers(batch, tweet_id)

            def _partial_fanout(self, tweet_id: str, user_id: str):
                """
                Partial fan-out: write only to active followers

                Used for: Mid-tier celebrities (100K-1M followers)
                Strategy: Fan-out to recently active followers (logged in last 24h)
                """
                # Get active followers (logged in last 24h)
                active_followers = self.graph.get_active_followers(
                    user_id,
                    since=datetime.utcnow() - timedelta(days=1),
                    limit=self.ACTIVE_FOLLOWER_THRESHOLD
                )

                logger.info(f"Partial fanout to {len(active_followers)}/{self.graph.get_follower_count(user_id)} followers")

                # Fan-out to active followers only
                self._fanout_on_write(tweet_id, user_id)

                # Mark for hybrid retrieval
                self.timeline.set_celebrity_marker(user_id, tweet_id)

            def _fanout_on_read_marker(self, tweet_id: str, user_id: str):
                """
                No fan-out: mark for pull on read

                Used for: Mega celebrities (> 1M followers)
                Strategy: Don't fan-out. Followers pull tweets on timeline request.
                """
                logger.info(f"No fanout for mega celebrity {user_id}")

                # Just mark user as having new tweet
                self.timeline.set_celebrity_marker(user_id, tweet_id)

                # Notify subset of highly engaged followers
                top_followers = self.graph.get_top_followers(user_id, limit=10000)
                self._notify_online_followers(top_followers, tweet_id)

            def _notify_online_followers(self, follower_ids: List[str], tweet_id: str):
                """Push real-time notification to online followers"""
                for follower_id in follower_ids:
                    if self.ws.is_online(follower_id):
                        self.ws.send_notification(
                            follower_id,
                            {
                                'type': 'new_tweet',
                                'tweet_id': tweet_id
                            }
                        )
        ```

        ---

        ## Timeline Generation (Hybrid)

        ```python
        class TimelineService:
            """Generate timeline using hybrid approach"""

            def __init__(self, graph_db, timeline_db, tweet_db, cache):
                self.graph = graph_db
                self.timeline = timeline_db
                self.tweets = tweet_db
                self.cache = cache

            def get_home_timeline(self, user_id: str, count: int = 20) -> List[dict]:
                """
                Generate home timeline combining pre-computed and on-demand tweets

                Returns:
                    List of tweet objects, sorted by created_at
                """
                # Check cache
                cache_key = f"timeline:{user_id}"
                cached = self.cache.get(cache_key)
                if cached:
                    return self._hydrate_tweets(cached[:count])

                # Get following list
                following = self.graph.get_following(user_id)

                # Separate into regular users and celebrities
                regular_users = []
                celebrities = []

                for followed_id in following:
                    if self.timeline.is_celebrity(followed_id):
                        celebrities.append(followed_id)
                    else:
                        regular_users.append(followed_id)

                # Get pre-computed timeline (regular users)
                precomputed_tweets = self.timeline.get_home_timeline(
                    user_id,
                    limit=100
                )

                # Pull celebrity tweets on-demand
                celebrity_tweets = []
                if celebrities:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [
                            executor.submit(self.tweets.get_user_tweets, celeb_id, 20)
                            for celeb_id in celebrities
                        ]

                        for future in as_completed(futures):
                            celebrity_tweets.extend(future.result())

                # Merge and sort
                all_tweet_ids = precomputed_tweets + [t['tweet_id'] for t in celebrity_tweets]
                all_tweet_ids = list(set(all_tweet_ids))  # Deduplicate
                all_tweet_ids.sort(key=lambda tid: self._get_tweet_timestamp(tid), reverse=True)

                # Cache for 60 seconds
                self.cache.setex(cache_key, 60, all_tweet_ids[:100])

                # Hydrate with full tweet objects
                return self._hydrate_tweets(all_tweet_ids[:count])

            def _hydrate_tweets(self, tweet_ids: List[str]) -> List[dict]:
                """Batch fetch full tweet objects"""
                return self.tweets.batch_get(tweet_ids)
        ```

        ---

        ## Fan-out Trade-offs

        | Strategy | Pros | Cons | Use Case |
        |----------|------|------|----------|
        | **Fan-out on Write** | Fast reads (< 20ms), simple | Slow writes (seconds), wasted work (inactive users) | Regular users (< 100K followers) |
        | **Fan-out on Read** | Fast writes (< 100ms), fresh data | Slow reads (200-500ms), complex | Mega celebrities (> 1M followers) |
        | **Hybrid** | Best of both, handles all scales | Complex implementation, two code paths | **Production (Twitter's approach)** |

    === "⚡ Real-time Updates"

        ## The Challenge

        **Problem:** Show new tweets in timeline without page refresh. 400M concurrent users expect < 1s latency.

        **Requirements:**

        - **Real-time:** New tweets appear within 1 second
        - **Scalable:** 400M concurrent connections
        - **Reliable:** Don't miss tweets
        - **Efficient:** Minimize bandwidth and server load

        ---

        ## Real-time Architecture

        **Three-tier approach:**

        1. **WebSocket (primary):** Full-duplex, low latency (< 100ms)
        2. **Long polling (fallback):** For browsers not supporting WebSocket
        3. **Push notifications (mobile):** APNs/FCM for background updates

        ---

        ## WebSocket Implementation

        ```python
        import asyncio
        import websockets
        import redis.asyncio as redis

        class WebSocketServer:
            """Handle real-time tweet delivery via WebSocket"""

            def __init__(self):
                self.connections = {}  # user_id -> WebSocket connection
                self.redis = redis.Redis(host='redis-host')

            async def register(self, user_id: str, websocket):
                """Register new WebSocket connection"""
                self.connections[user_id] = websocket
                logger.info(f"User {user_id} connected. Total connections: {len(self.connections)}")

                # Subscribe to user's personal channel
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(f"timeline:{user_id}")

                try:
                    # Listen for messages
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            # Forward to WebSocket client
                            await websocket.send(message['data'])

                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"User {user_id} disconnected")
                finally:
                    del self.connections[user_id]
                    await pubsub.unsubscribe(f"timeline:{user_id}")

            async def notify_user(self, user_id: str, tweet_data: dict):
                """
                Send new tweet notification to user

                Args:
                    user_id: User to notify
                    tweet_data: Tweet object to send
                """
                # Publish to Redis pub/sub
                await self.redis.publish(
                    f"timeline:{user_id}",
                    json.dumps({
                        'type': 'new_tweet',
                        'tweet': tweet_data
                    })
                )

            async def notify_batch(self, user_ids: List[str], tweet_data: dict):
                """
                Batch notify multiple users

                More efficient than individual notifications
                """
                pipeline = self.redis.pipeline()
                for user_id in user_ids:
                    pipeline.publish(
                        f"timeline:{user_id}",
                        json.dumps({'type': 'new_tweet', 'tweet': tweet_data})
                    )
                await pipeline.execute()

        # WebSocket server
        async def handler(websocket, path):
            user_id = await authenticate_websocket(websocket)
            if user_id:
                await ws_server.register(user_id, websocket)

        ws_server = WebSocketServer()
        start_server = websockets.serve(handler, "0.0.0.0", 8765)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
        ```

        ---

        ## Connection Management

        **Challenge:** 400M concurrent connections = massive resource usage.

        **Solution: Connection pooling + horizontal scaling**

        **Architecture:**

        ```
        400M users
        ├── 400 WebSocket servers (1M connections each)
        │   ├── Server 1: users 0-1M
        │   ├── Server 2: users 1M-2M
        │   └── ...
        └── Redis Pub/Sub (routing layer)
            └── Publish to specific server handling user
        ```

        **Connection routing:**

        ```python
        def get_websocket_server(user_id: str) -> str:
            """Determine which WebSocket server handles this user"""
            server_id = hash(user_id) % 400
            return f"ws-server-{server_id}.twitter.com"
        ```

        **Benefits:**

        - **Scalable:** Add more WebSocket servers as needed
        - **Fault tolerant:** Server failure only affects 1M users
        - **Efficient:** 1M connections per server (proven at scale)

        ---

        ## Fallback: Long Polling

        **For browsers without WebSocket support:**

        ```python
        @app.get("/api/v1/timeline/poll")
        async def long_poll_timeline(user_id: str, last_tweet_id: str):
            """
            Long polling endpoint for timeline updates

            Blocks for up to 30 seconds waiting for new tweets
            """
            timeout = 30  # seconds
            poll_interval = 1  # second
            elapsed = 0

            while elapsed < timeout:
                # Check for new tweets
                new_tweets = db.query(
                    """SELECT * FROM home_timeline
                       WHERE user_id = %s AND tweet_id > %s
                       ORDER BY created_at DESC LIMIT 20""",
                    (user_id, last_tweet_id)
                )

                if new_tweets:
                    return {'tweets': new_tweets}

                # Wait before next poll
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            # Timeout - return empty
            return {'tweets': []}
        ```

        **Trade-offs:**

        - **Pros:** Works in all browsers, simple to implement
        - **Cons:** Higher latency (1-30s), more server load (constant polling)

    === "🔥 Trending Topics"

        ## The Challenge

        **Problem:** Calculate trending hashtags in real-time across 500M tweets/day.

        **Requirements:**

        - **Real-time:** Trending topics update every 1-5 minutes
        - **Accurate:** Detect sudden spikes (viral events)
        - **Localized:** Trending by country/region
        - **Spam-resistant:** Ignore bot-generated hashtags

        ---

        ## Trending Algorithm

        **Time-decay weighted count:**

        ```
        TrendScore(hashtag) = Σ(weight(t) × mentions(t))

        where:
        - t = time window (e.g., last 1 hour)
        - weight(t) = e^(-λ × age)  // Exponential decay
        - λ = decay constant (higher = faster decay)
        ```

        **Why time-decay:**

        - Recent mentions weighted higher (viral events)
        - Old trends naturally fade out
        - Detects sudden spikes (Super Bowl, elections)

        ---

        ## Implementation (Kafka Streams)

        ```python
        from kafka import KafkaConsumer, KafkaProducer
        from collections import defaultdict, Counter
        import time
        import math

        class TrendingCalculator:
            """Real-time trending topic calculation"""

            DECAY_CONSTANT = 0.0001  # Decay rate
            WINDOW_MINUTES = 60      # Consider last 60 minutes
            UPDATE_INTERVAL = 60     # Update trends every 60 seconds

            def __init__(self, kafka_brokers):
                self.consumer = KafkaConsumer(
                    'tweet_events',
                    bootstrap_servers=kafka_brokers,
                    auto_offset_reset='latest'
                )
                self.producer = KafkaProducer(bootstrap_servers=kafka_brokers)

                # In-memory state (or use Redis for distributed)
                self.hashtag_mentions = defaultdict(list)  # hashtag -> [(timestamp, weight)]
                self.last_update = time.time()

            def calculate_trending(self):
                """Main loop: consume tweets and calculate trends"""
                while True:
                    # Consume tweets
                    messages = self.consumer.poll(timeout_ms=1000)

                    for topic_partition, records in messages.items():
                        for record in records:
                            tweet = json.loads(record.value)
                            self._process_tweet(tweet)

                    # Update trends every minute
                    if time.time() - self.last_update > self.UPDATE_INTERVAL:
                        self._update_trending_topics()
                        self.last_update = time.time()

            def _process_tweet(self, tweet: dict):
                """Extract hashtags and update counts"""
                hashtags = tweet.get('hashtags', [])
                timestamp = tweet['created_at']

                for hashtag in hashtags:
                    # Add mention with timestamp
                    self.hashtag_mentions[hashtag].append((timestamp, 1.0))

            def _update_trending_topics(self):
                """Calculate trending scores and publish"""
                now = time.time()
                cutoff = now - (self.WINDOW_MINUTES * 60)

                trending_scores = {}

                for hashtag, mentions in self.hashtag_mentions.items():
                    # Remove old mentions (outside window)
                    mentions = [(ts, w) for ts, w in mentions if ts > cutoff]
                    self.hashtag_mentions[hashtag] = mentions

                    # Calculate score with time decay
                    score = 0
                    for timestamp, weight in mentions:
                        age = now - timestamp
                        decay_weight = math.exp(-self.DECAY_CONSTANT * age)
                        score += weight * decay_weight

                    if score > 0:
                        trending_scores[hashtag] = score

                # Get top 50 trending
                top_trending = sorted(
                    trending_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:50]

                # Publish to Kafka (for caching)
                self.producer.send(
                    'trending_topics',
                    value=json.dumps({
                        'timestamp': now,
                        'trending': [
                            {'hashtag': tag, 'score': score}
                            for tag, score in top_trending
                        ]
                    }).encode()
                )

                logger.info(f"Updated trending topics: {len(top_trending)} hashtags")
        ```

        ---

        ## Spam Detection

        **Problem:** Bots can manipulate trending by spamming hashtags.

        **Solutions:**

        1. **Rate limiting:** Max 10 tweets per minute per user
        2. **Duplicate detection:** Ignore identical tweets from same user
        3. **Bot detection:** ML model identifies bot accounts, lower their weight
        4. **Velocity checks:** Sudden spike from new accounts = suspicious

        ```python
        def calculate_user_weight(user_id: str) -> float:
            """
            Calculate user's contribution weight to trending

            Returns:
                Weight between 0.0 and 1.0
            """
            user = db.get_user(user_id)

            # Factors
            account_age_days = (datetime.utcnow() - user['created_at']).days
            is_verified = user['verified']
            follower_count = user['follower_count']
            following_ratio = user['following_count'] / max(follower_count, 1)

            # Calculate weight
            weight = 1.0

            # Penalty for new accounts
            if account_age_days < 30:
                weight *= 0.5

            # Boost for verified accounts
            if is_verified:
                weight *= 1.5

            # Penalty for suspicious following ratio (bots often follow many, have few followers)
            if following_ratio > 10:
                weight *= 0.3

            # Boost for popular accounts
            if follower_count > 10000:
                weight *= 1.2

            return max(0.1, min(weight, 2.0))  # Clamp between 0.1 and 2.0
        ```

    === "🔍 Search System"

        ## The Challenge

        **Problem:** Search 1.825 trillion tweets in < 100ms with complex queries (hashtags, mentions, date ranges).

        **Requirements:**

        - **Fast:** < 100ms p95 latency
        - **Comprehensive:** Full-text search, filters, ranking
        - **Real-time:** New tweets searchable within 5 seconds
        - **Scalable:** Handle 1M search QPS during events

        ---

        ## Elasticsearch Architecture

        **Sharding strategy:**

        ```
        1.825 trillion tweets
        ├── Shard by month (120 shards for 10 years)
        │   ├── Shard 2026-01: 15B tweets
        │   ├── Shard 2026-02: 15B tweets
        │   └── ...
        └── Each shard replicated 3x (for availability)
        ```

        **Benefits:**

        - **Time-based queries:** "tweets since:2026-01-01" only scans relevant shards
        - **Hot/cold data:** Recent shards on SSD, old shards on HDD
        - **Easy archival:** Delete old shards after retention period

        ---

        ## Indexing Pipeline

        ```python
        from elasticsearch import Elasticsearch, helpers

        class TweetSearchIndexer:
            """Index tweets in Elasticsearch for search"""

            def __init__(self, es_client):
                self.es = es_client

            def index_tweet(self, tweet: dict):
                """
                Index single tweet

                Args:
                    tweet: Tweet object with content, metadata
                """
                document = {
                    'tweet_id': tweet['tweet_id'],
                    'user_id': tweet['user_id'],
                    'username': tweet['username'],
                    'content': tweet['content'],
                    'created_at': tweet['created_at'],
                    'hashtags': tweet.get('hashtags', []),
                    'mentions': tweet.get('mentions', []),
                    'likes_count': tweet['likes_count'],
                    'retweets_count': tweet['retweets_count']
                }

                # Index with monthly routing
                index_name = f"tweets-{tweet['created_at'][:7]}"  # e.g., tweets-2026-01

                self.es.index(
                    index=index_name,
                    id=tweet['tweet_id'],
                    document=document
                )

            def bulk_index(self, tweets: List[dict]):
                """
                Bulk index multiple tweets (more efficient)

                Args:
                    tweets: List of tweet objects
                """
                actions = []
                for tweet in tweets:
                    actions.append({
                        '_index': f"tweets-{tweet['created_at'][:7]}",
                        '_id': tweet['tweet_id'],
                        '_source': {
                            'tweet_id': tweet['tweet_id'],
                            'user_id': tweet['user_id'],
                            'username': tweet['username'],
                            'content': tweet['content'],
                            'created_at': tweet['created_at'],
                            'hashtags': tweet.get('hashtags', []),
                            'mentions': tweet.get('mentions', []),
                            'likes_count': tweet['likes_count'],
                            'retweets_count': tweet['retweets_count']
                        }
                    })

                # Bulk index (1000 at a time)
                helpers.bulk(self.es, actions, chunk_size=1000)
                logger.info(f"Indexed {len(tweets)} tweets")

            def search_tweets(self, query: str, filters: dict = None, size: int = 20) -> List[dict]:
                """
                Search tweets with filters

                Args:
                    query: Search query string
                    filters: Optional filters (from:user, since:date, etc.)
                    size: Number of results

                Returns:
                    List of matching tweets
                """
                # Build Elasticsearch query
                must_clauses = []

                # Full-text search
                if query:
                    must_clauses.append({
                        'multi_match': {
                            'query': query,
                            'fields': ['content^2', 'hashtags', 'mentions'],  # content boosted 2x
                            'type': 'best_fields'
                        }
                    })

                # Apply filters
                filter_clauses = []

                if filters:
                    # from:username
                    if 'from' in filters:
                        filter_clauses.append({
                            'term': {'username': filters['from']}
                        })

                    # since:date
                    if 'since' in filters:
                        filter_clauses.append({
                            'range': {'created_at': {'gte': filters['since']}}
                        })

                    # to:date
                    if 'until' in filters:
                        filter_clauses.append({
                            'range': {'created_at': {'lte': filters['until']}}
                        })

                    # hashtag
                    if 'hashtag' in filters:
                        filter_clauses.append({
                            'term': {'hashtags': filters['hashtag']}
                        })

                # Construct full query
                search_body = {
                    'query': {
                        'bool': {
                            'must': must_clauses,
                            'filter': filter_clauses
                        }
                    },
                    'sort': [
                        {'created_at': {'order': 'desc'}}  # Most recent first
                    ],
                    'size': size
                }

                # Execute search
                response = self.es.search(index='tweets-*', body=search_body)

                # Extract results
                tweets = [hit['_source'] for hit in response['hits']['hits']]
                return tweets
        ```

        ---

        ## Search Ranking

        **Relevance score factors:**

        | Factor | Weight | Description |
        |--------|--------|-------------|
        | **Text relevance** | 40% | TF-IDF score for query match |
        | **Recency** | 30% | Recent tweets ranked higher (time decay) |
        | **Engagement** | 20% | Likes + retweets boost score |
        | **User authority** | 10% | Verified users, high follower count |

        **Implementation:**

        ```json
        {
          "query": {
            "function_score": {
              "query": { "match": { "content": "machine learning" } },
              "functions": [
                {
                  "gauss": {
                    "created_at": {
                      "origin": "now",
                      "scale": "7d",
                      "decay": 0.5
                    }
                  },
                  "weight": 3
                },
                {
                  "field_value_factor": {
                    "field": "likes_count",
                    "modifier": "log1p",
                    "factor": 0.1
                  },
                  "weight": 2
                }
              ],
              "score_mode": "sum",
              "boost_mode": "multiply"
            }
          }
        }
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Twitter from 1M to 400M DAU.

    **Scaling challenges at 400M DAU:**

    - **Write throughput:** 115K write QPS (tweets + likes + follows)
    - **Read throughput:** 139K read QPS (timelines + profiles + search)
    - **Storage:** 1.1 exabytes of data
    - **Real-time:** 400M WebSocket connections

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Cassandra writes** | ✅ Yes | Shard across 500 nodes, SSD storage, tuned compaction |
    | **Timeline generation** | ✅ Yes | Hybrid fan-out, Redis cache (80% hit rate), parallel queries |
    | **WebSocket servers** | ✅ Yes | 400 servers (1M connections each), Redis pub/sub routing |
    | **Elasticsearch** | ✅ Yes | 120 shards (by month), 3x replication, hot/cold data tiers |
    | **Kafka throughput** | 🟡 Approaching | 50 brokers, partitioning by user_id, compression |

    ---

    ## Cost Optimization

    **Monthly cost at 400M DAU:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $86,400 (600 × m5.2xlarge) |
    | **Cassandra cluster** | $216,000 (500 nodes) |
    | **Elasticsearch cluster** | $108,000 (300 nodes) |
    | **Redis cache** | $43,200 (200 nodes) |
    | **Kafka cluster** | $21,600 (50 brokers) |
    | **WebSocket servers** | $86,400 (400 servers) |
    | **S3 storage** | $25,300 (1.1 EB) |
    | **CDN** | $127,500 (1,500 TB egress) |
    | **Total** | **$714,400/month** |

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Timeline Latency (P95)** | < 200ms | > 500ms |
    | **Tweet Post Latency (P95)** | < 100ms | > 500ms |
    | **WebSocket Connection Success** | > 99% | < 95% |
    | **Cassandra Write Latency** | < 10ms | > 50ms |
    | **Cache Hit Rate** | > 80% | < 70% |
    | **Search Latency (P95)** | < 100ms | > 300ms |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Hybrid fan-out:** Write for regular users, read for celebrities
    2. **Cassandra for tweets:** Write-optimized, handles 115K write QPS
    3. **Elasticsearch for search:** Fast full-text search, complex queries
    4. **WebSocket for real-time:** 400M concurrent connections
    5. **Redis caching:** 80% cache hit rate for timelines
    6. **Eventual consistency:** Prioritize availability over consistency

    ---

    ## Interview Tips

    ✅ **Emphasize write throughput** - 115K write QPS is challenging

    ✅ **Discuss fan-out trade-offs** - Hybrid approach is key

    ✅ **Real-time is critical** - WebSocket architecture important

    ✅ **Search complexity** - Elasticsearch sharding strategy

    ✅ **Trending topics** - Time-decay algorithm, spam detection

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle celebrity tweets?"** | Hybrid fan-out: no pre-computation for >1M followers, pull on read |
    | **"How to show new tweets in real-time?"** | WebSocket with Redis pub/sub, long polling fallback |
    | **"How to calculate trending topics?"** | Kafka Streams, time-decay weighted count, spam detection |
    | **"How to search tweets?"** | Elasticsearch sharded by month, real-time indexing (<5s) |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Twitter, Meta, Google, Amazon, LinkedIn, Reddit

---

*Master this problem and you'll be ready for: Instagram, Facebook, LinkedIn, Reddit, TikTok*
