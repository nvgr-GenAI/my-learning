# Design Recommendation System (Netflix/Amazon)

A personalized recommendation system that suggests relevant items (movies, products, videos, music) to users based on their preferences, behavior, and similar users' patterns.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 200M daily active users, 10M items, 1B recommendations/day, 100M interactions/day |
| **Key Challenges** | Cold start problem, real-time personalization, diversity vs relevance, scalability, freshness |
| **Core Concepts** | Collaborative filtering, matrix factorization, deep learning (two-tower, transformers), A/B testing |
| **Companies** | Netflix, Amazon, YouTube, Spotify, TikTok, LinkedIn, Pinterest, Uber Eats |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Personalized Recommendations** | Suggest relevant items based on user history | P0 (Must have) |
    | **Similar Items** | "Users who liked X also liked Y" | P0 (Must have) |
    | **Trending/Popular** | Show trending items in each category | P0 (Must have) |
    | **Real-time Updates** | Recommendations update based on current session | P1 (Should have) |
    | **Diversity** | Mix of familiar and discovery content | P1 (Should have) |
    | **Explainability** | "Because you watched X" reasoning | P1 (Should have) |
    | **Category-specific** | Recommendations within genres/categories | P1 (Should have) |
    | **Search Integration** | Personalized search results ranking | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Content creation/upload system
    - Payment processing
    - Social features (sharing, comments)
    - Content moderation
    - Live streaming infrastructure

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 200ms p95 | Users expect instant page loads |
    | **Availability** | 99.9% uptime | Critical for user experience and revenue |
    | **Freshness** | < 1 hour for behavioral updates | New interactions should influence recommendations quickly |
    | **Relevance** | CTR > 5%, engagement rate > 30% | Must provide value to users |
    | **Diversity** | 30% exploration, 70% exploitation | Balance familiarity and discovery |
    | **Serendipity** | 10% surprising recommendations | Delight users with unexpected finds |
    | **Consistency** | Eventual consistency acceptable | Brief delays in recommendation updates tolerable |
    | **Scalability** | Support 10x traffic spikes | Handle viral content and marketing campaigns |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 200M
    Monthly Active Users (MAU): 400M

    Recommendations:
    - Avg recommendations per user session: 5 API calls
    - Sessions per DAU: ~3 sessions/day
    - Daily recommendation requests: 200M × 3 × 5 = 3B requests/day
    - Recommendation QPS: 3B / 86,400 = ~34,700 req/sec
    - Peak QPS: 3x average = ~104,000 req/sec

    User interactions (clicks, views, ratings):
    - Interactions per DAU: ~50 interactions/day (views, clicks, ratings)
    - Daily interactions: 200M × 50 = 10B interactions/day
    - Interaction QPS: 10B / 86,400 = ~115,700 events/sec

    Model training:
    - Batch training: Daily (full model retrain)
    - Incremental updates: Every 15 minutes
    - Online learning: Real-time for active users

    Read/Write ratio: 30:1 (mostly read-heavy for recommendations)
    ```

    ### Storage Estimates

    ```
    User profiles:
    - User features: 5 KB per user (demographics, preferences, history)
    - 400M users × 5 KB = 2 TB

    Item catalog:
    - Item features: 10 KB per item (metadata, embeddings, stats)
    - 10M items × 10 KB = 100 GB

    User-item interactions:
    - Interaction record: 50 bytes (user_id, item_id, interaction_type, timestamp, context)
    - 10 years: 10B interactions/day × 365 × 10 = 36.5 trillion interactions
    - Storage: 36.5T × 50 bytes = 1.825 PB

    User embeddings (for neural models):
    - Embedding dimension: 256 floats × 4 bytes = 1 KB per user
    - 400M users × 1 KB = 400 GB

    Item embeddings:
    - 256 floats × 4 bytes = 1 KB per item
    - 10M items × 1 KB = 10 GB

    Pre-computed recommendations (cache):
    - 50M active users × 100 recommendations × 20 bytes = 100 GB

    Feature store (real-time features):
    - User session features: 1 KB per active user
    - 20M concurrent users × 1 KB = 20 GB

    Total: 2 TB (users) + 100 GB (items) + 1.825 PB (interactions) + 400 GB (embeddings) + 130 GB (cache) ≈ 2 PB
    ```

    ### Bandwidth Estimates

    ```
    Recommendation API:
    - 34,700 req/sec × 5 KB response (20 recommendations with metadata) = 174 MB/sec ≈ 1.4 Gbps

    Interaction ingestion:
    - 115,700 events/sec × 200 bytes (enriched event) = 23 MB/sec ≈ 184 Mbps

    Model serving (embedding lookups):
    - 34,700 req/sec × 2 KB (user + item embeddings) = 69 MB/sec ≈ 552 Mbps

    Feature store queries:
    - 34,700 req/sec × 500 bytes (real-time features) = 17 MB/sec ≈ 136 Mbps

    Total ingress: ~184 Mbps
    Total egress: ~2 Gbps (mostly cached responses)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot recommendations:
    - 50M users × 100 recs × 20 bytes = 100 GB

    User embeddings (hot users):
    - 20M active users × 1 KB = 20 GB

    Item embeddings (all items, frequently accessed):
    - 10M items × 1 KB = 10 GB

    Feature cache:
    - User features: 20M × 2 KB = 40 GB
    - Item features: 1M hot items × 5 KB = 5 GB

    Model serving cache:
    - Pre-computed scores: 10 GB

    Total cache: 100 GB + 20 GB + 10 GB + 45 GB + 10 GB ≈ 185 GB
    ```

    ---

    ## Key Assumptions

    1. Average user views 50 items per day (videos, products, etc.)
    2. 10M items in catalog (growing at 1% monthly)
    3. User-item interaction matrix is extremely sparse (< 0.01% density)
    4. Cold start affects 20% of users (new or inactive)
    5. Real-time updates within 1 hour acceptable for most use cases
    6. Diversity and serendipity are critical for user satisfaction
    7. A/B testing framework required for continuous improvement

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Hybrid approach:** Combine collaborative filtering, content-based, and deep learning
    2. **Offline + Online:** Pre-compute recommendations, personalize in real-time
    3. **Multi-stage ranking:** Candidate generation → ranking → reranking → filtering
    4. **Feature engineering:** Rich features from user behavior, context, and content
    5. **Continuous learning:** A/B testing, online learning, model updates

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Static content]
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Rec_API[Recommendation API<br/>Get personalized recs]
            Track_API[Tracking API<br/>Log interactions]
            Search_API[Search API<br/>Personalized search]
            AB_API[A/B Test Service<br/>Experiment assignment]
        end

        subgraph "Recommendation Engine"
            Candidate_Gen[Candidate Generation<br/>Fast retrieval]
            Ranking[Ranking Service<br/>Score & sort]
            Reranking[Reranking Service<br/>Diversity, business rules]
            Filter[Filter Service<br/>Already seen, business rules]
        end

        subgraph "Model Serving"
            CF_Model[Collaborative Filtering<br/>User-item similarity]
            DNN_Model[Deep Learning Model<br/>Two-tower, transformers]
            Content_Model[Content-Based<br/>Item similarity]
            Context_Model[Context Model<br/>Time, device, location]
        end

        subgraph "Real-time Processing"
            Stream_Processor[Stream Processor<br/>Kafka Streams]
            Feature_Computer[Feature Computer<br/>Real-time features]
            Online_Learner[Online Learning<br/>Model updates]
        end

        subgraph "Offline Processing"
            Batch_Pipeline[Batch Pipeline<br/>Spark jobs]
            Model_Training[Model Training<br/>Daily retraining]
            Embedding_Gen[Embedding Generator<br/>User/item embeddings]
            Similarity[Similarity Computer<br/>Item-item, user-user]
        end

        subgraph "Caching"
            Redis_Rec[Redis<br/>Pre-computed recs]
            Redis_Feature[Redis<br/>Feature cache]
            Redis_Embed[Redis<br/>Embeddings cache]
        end

        subgraph "Storage"
            User_DB[(User Profile DB<br/>PostgreSQL)]
            Item_DB[(Item Catalog<br/>PostgreSQL)]
            Interaction_DB[(Interactions<br/>Cassandra)]
            Feature_Store[(Feature Store<br/>DynamoDB)]
            Vector_DB[(Vector DB<br/>Pinecone/Milvus)]
            Model_Store[Model Store<br/>S3]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "Analytics"
            Metrics[Metrics Service<br/>CTR, engagement]
            AB_Analysis[A/B Analysis<br/>Experiment results]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        LB --> Rec_API
        LB --> Track_API
        LB --> Search_API

        Rec_API --> AB_API
        Rec_API --> Candidate_Gen
        Track_API --> Kafka

        Candidate_Gen --> CF_Model
        Candidate_Gen --> DNN_Model
        Candidate_Gen --> Content_Model
        Candidate_Gen --> Vector_DB
        Candidate_Gen --> Redis_Rec

        Candidate_Gen --> Ranking
        Ranking --> Context_Model
        Ranking --> Feature_Computer
        Ranking --> Redis_Feature

        Ranking --> Reranking
        Reranking --> Filter
        Filter --> Rec_API

        Kafka --> Stream_Processor
        Stream_Processor --> Feature_Computer
        Stream_Processor --> Online_Learner
        Stream_Processor --> Interaction_DB
        Stream_Processor --> Feature_Store

        Kafka --> Batch_Pipeline
        Batch_Pipeline --> Model_Training
        Batch_Pipeline --> Embedding_Gen
        Batch_Pipeline --> Similarity

        Model_Training --> Model_Store
        Model_Training --> CF_Model
        Model_Training --> DNN_Model
        Model_Training --> Content_Model

        Embedding_Gen --> Vector_DB
        Embedding_Gen --> Redis_Embed

        Similarity --> Redis_Rec

        Track_API --> Metrics
        AB_API --> AB_Analysis

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Rec fill:#fff4e1
        style Redis_Feature fill:#fff4e1
        style Redis_Embed fill:#fff4e1
        style User_DB fill:#ffe1e1
        style Item_DB fill:#ffe1e1
        style Interaction_DB fill:#ffe1e1
        style Feature_Store fill:#e1f5e1
        style Vector_DB fill:#e8eaf6
        style Model_Store fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Cassandra (Interactions)** | Write-heavy (115K write QPS), time-series data, horizontal scaling | PostgreSQL (write bottleneck), MongoDB (consistency issues) |
    | **Vector Database** | Fast similarity search (< 10ms), billion-scale embeddings | Elasticsearch (not optimized for dense vectors), FAISS (not distributed) |
    | **Kafka** | High-throughput event streaming, reliable delivery, replay capability | RabbitMQ (throughput limit), direct streaming (no persistence) |
    | **Redis (Multi-layer)** | Fast caching (< 1ms), pre-computed recommendations, embeddings | Memcached (limited features), no cache (latency too high) |
    | **Spark** | Distributed batch processing, model training, feature engineering | Single-node (can't handle 36T interactions), MapReduce (too slow) |
    | **Feature Store** | Centralized feature management, online/offline consistency | Custom solution (complex), database (latency too high) |

    **Key Trade-off:** We chose **eventual consistency** over strong consistency. Recommendations may not reflect latest interactions immediately, but system remains available and fast.

    ---

    ## API Design

    ### 1. Get Personalized Recommendations

    **Request:**
    ```http
    POST /api/v1/recommendations
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "user_id": "user123",
      "context": {
        "page": "home",              // home, search, item_page, category
        "device": "mobile",           // mobile, web, tv
        "location": "US-CA-SF",
        "time": "2026-01-29T19:30:00Z"
      },
      "filters": {
        "category": "action",         // Optional: filter by category
        "exclude_seen": true,         // Don't show already viewed items
        "min_rating": 4.0             // Optional: quality threshold
      },
      "count": 20,
      "experiment_id": "exp_2024_01"  // A/B test assignment
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "recommendations": [
        {
          "item_id": "movie_456",
          "title": "Inception",
          "thumbnail_url": "https://cdn.example.com/inception.jpg",
          "score": 0.92,
          "reason": "Because you watched The Dark Knight",
          "category": "action",
          "metadata": {
            "rating": 4.8,
            "year": 2010,
            "duration": "148 min"
          }
        },
        // ... 19 more items
      ],
      "request_id": "rec_abc123",
      "model_version": "v2.3.1",
      "experiment_id": "exp_2024_01"
    }
    ```

    **Design Notes:**

    - Return pre-computed recommendations when available (< 20ms)
    - Personalize in real-time using context and recent behavior
    - Include explainability for user trust
    - Log request for A/B testing analysis
    - Support multiple recommendation strategies (exploration vs exploitation)

    ---

    ### 2. Track User Interactions

    **Request:**
    ```http
    POST /api/v1/track
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "user_id": "user123",
      "event_type": "view",  // view, click, rating, purchase, skip
      "item_id": "movie_456",
      "timestamp": "2026-01-29T19:35:00Z",
      "context": {
        "device": "mobile",
        "page": "home",
        "position": 3,          // Position in recommendation list
        "session_id": "sess_xyz",
        "request_id": "rec_abc123"  // Link to recommendation request
      },
      "metadata": {
        "rating": 4.5,          // For rating events
        "watch_duration": 120,  // For view events (seconds)
        "total_duration": 148   // Total item duration
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 202 Accepted
    Content-Type: application/json

    {
      "event_id": "evt_789",
      "status": "queued"
    }
    ```

    **Design Notes:**

    - Async processing (return immediately)
    - Rich context for feature engineering
    - Link interactions to recommendations for attribution
    - Support multiple event types with different schemas
    - Rate limiting to prevent abuse

    ---

    ### 3. Get Similar Items

    **Request:**
    ```http
    GET /api/v1/items/movie_456/similar?count=10
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "item_id": "movie_456",
      "similar_items": [
        {
          "item_id": "movie_789",
          "title": "The Matrix",
          "similarity_score": 0.89,
          "reason": "Similar genre and director"
        },
        // ... 9 more items
      ]
    }
    ```

    **Design Notes:**

    - Pre-computed item-item similarity
    - Fast lookup from vector database
    - Cached for hot items
    - Multiple similarity metrics (content, collaborative, hybrid)

    ---

    ## Database Schema

    ### User Profiles (PostgreSQL)

    ```sql
    -- Users table
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_active TIMESTAMP,
        demographics JSONB,  -- age, gender, location
        preferences JSONB,   -- favorite categories, languages
        INDEX idx_email (email),
        INDEX idx_last_active (last_active)
    );

    -- User embeddings (learned representations)
    CREATE TABLE user_embeddings (
        user_id BIGINT PRIMARY KEY,
        embedding VECTOR(256),  -- Using pgvector extension
        model_version VARCHAR(20),
        updated_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    ```

    ---

    ### Item Catalog (PostgreSQL)

    ```sql
    -- Items table
    CREATE TABLE items (
        item_id BIGINT PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        description TEXT,
        category VARCHAR(100),
        subcategory VARCHAR(100),
        tags TEXT[],  -- Array of tags
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB,  -- rating, year, duration, price, etc.
        is_active BOOLEAN DEFAULT TRUE,
        INDEX idx_category (category),
        INDEX idx_created_at (created_at),
        INDEX idx_tags USING GIN (tags)
    );

    -- Item embeddings
    CREATE TABLE item_embeddings (
        item_id BIGINT PRIMARY KEY,
        content_embedding VECTOR(256),  -- From content features
        collaborative_embedding VECTOR(256),  -- From user interactions
        model_version VARCHAR(20),
        updated_at TIMESTAMP,
        FOREIGN KEY (item_id) REFERENCES items(item_id)
    );

    -- Item statistics (for popularity, trending)
    CREATE TABLE item_stats (
        item_id BIGINT PRIMARY KEY,
        view_count_24h INT DEFAULT 0,
        view_count_7d INT DEFAULT 0,
        view_count_30d INT DEFAULT 0,
        click_count_24h INT DEFAULT 0,
        conversion_count_24h INT DEFAULT 0,
        avg_rating DECIMAL(3,2),
        rating_count INT,
        updated_at TIMESTAMP,
        FOREIGN KEY (item_id) REFERENCES items(item_id)
    );
    ```

    ---

    ### User-Item Interactions (Cassandra)

    ```sql
    -- Interactions table (write-heavy)
    CREATE TABLE interactions (
        user_id BIGINT,
        timestamp TIMESTAMP,
        item_id BIGINT,
        event_type TEXT,  -- view, click, rating, purchase, skip
        event_id UUID,
        context MAP<TEXT, TEXT>,  -- device, location, page, etc.
        metadata MAP<TEXT, TEXT>,  -- rating, watch_duration, etc.
        PRIMARY KEY (user_id, timestamp, event_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);

    -- User interaction history (for quick lookup)
    CREATE TABLE user_history (
        user_id BIGINT,
        item_id BIGINT,
        last_interaction TIMESTAMP,
        interaction_count INT,
        last_event_type TEXT,
        PRIMARY KEY (user_id, item_id)
    );

    -- Item interaction history
    CREATE TABLE item_history (
        item_id BIGINT,
        user_id BIGINT,
        timestamp TIMESTAMP,
        event_type TEXT,
        PRIMARY KEY (item_id, timestamp, user_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);
    ```

    ---

    ### Feature Store (DynamoDB)

    ```json
    // User features (real-time)
    {
      "user_id": "user123",
      "features": {
        "session_count_24h": 5,
        "items_viewed_24h": 42,
        "avg_watch_duration": 85.5,
        "favorite_categories": ["action", "sci-fi"],
        "active_hours": [19, 20, 21, 22],
        "device_preference": "mobile",
        "engagement_score": 0.78
      },
      "ttl": 1706558400,  // Auto-expire after 24 hours
      "updated_at": "2026-01-29T19:35:00Z"
    }

    // Item features (real-time)
    {
      "item_id": "movie_456",
      "features": {
        "view_velocity_1h": 1250,  // Trending indicator
        "ctr_24h": 0.12,
        "avg_watch_completion": 0.82,
        "popularity_score": 0.91,
        "recency_days": 5,
        "seasonal_score": 0.88
      },
      "ttl": 1706558400,
      "updated_at": "2026-01-29T19:35:00Z"
    }
    ```

    ---

    ## Data Flow Diagrams

    ### Recommendation Request Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Rec_API
        participant AB_Service
        participant Cache
        participant Candidate_Gen
        participant Ranking
        participant Feature_Store
        participant Vector_DB

        Client->>Rec_API: GET /recommendations (user_id, context)
        Rec_API->>AB_Service: Get experiment assignment
        AB_Service-->>Rec_API: Experiment config

        Rec_API->>Cache: GET cached_recs:user123
        alt Cache HIT (70% of requests)
            Cache-->>Rec_API: Pre-computed recommendations
            Rec_API->>Rec_API: Personalize with context
            Rec_API-->>Client: 200 OK (20 items, <50ms)
        else Cache MISS (30% of requests)
            Cache-->>Rec_API: null

            Rec_API->>Candidate_Gen: Generate candidates

            par Parallel candidate generation
                Candidate_Gen->>Vector_DB: ANN search (user embedding)
                Vector_DB-->>Candidate_Gen: 500 candidates (collaborative)

                Candidate_Gen->>Vector_DB: Content-based search
                Vector_DB-->>Candidate_Gen: 300 candidates (content)

                Candidate_Gen->>Cache: Get popular/trending
                Cache-->>Candidate_Gen: 200 candidates (trending)
            end

            Candidate_Gen->>Candidate_Gen: Merge & deduplicate (1000 candidates)
            Candidate_Gen->>Ranking: Score candidates

            Ranking->>Feature_Store: Batch get features (user + items)
            Feature_Store-->>Ranking: Real-time features

            Ranking->>Ranking: ML model scoring
            Ranking->>Ranking: Rerank (diversity, business rules)
            Ranking->>Ranking: Filter (already seen, quality)
            Ranking-->>Rec_API: Top 20 items

            Rec_API->>Cache: SET cached_recs:user123 (TTL: 5 min)
            Rec_API-->>Client: 200 OK (20 items, <200ms)
        end
    ```

    **Flow Explanation:**

    1. **A/B test assignment** - Determine experiment variant
    2. **Check cache** - 70% cache hit for active users
    3. **Candidate generation** - Fast retrieval from multiple sources (1000 candidates)
    4. **Ranking** - Score candidates using ML model with real-time features
    5. **Reranking** - Apply diversity, business rules, personalization
    6. **Filtering** - Remove already seen, low-quality items
    7. **Cache result** - Store for 5 minutes

    ---

    ### Interaction Tracking Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Track_API
        participant Kafka
        participant Stream_Processor
        participant Interaction_DB
        participant Feature_Store
        participant Online_Learner
        participant Batch_Pipeline

        Client->>Track_API: POST /track (event)
        Track_API->>Track_API: Validate & enrich event
        Track_API->>Kafka: Publish interaction event
        Track_API-->>Client: 202 Accepted (<10ms)

        Kafka->>Stream_Processor: Consume event

        par Real-time processing
            Stream_Processor->>Interaction_DB: Store interaction (async)

            Stream_Processor->>Feature_Store: Update user features
            Note over Feature_Store: Increment counters,<br/>update aggregates

            Stream_Processor->>Online_Learner: Update model (for active users)
            Note over Online_Learner: Incremental learning,<br/>adjust embeddings
        end

        Kafka->>Batch_Pipeline: Store in data lake (hourly)
        Note over Batch_Pipeline: For offline training,<br/>analytics, A/B analysis
    ```

    **Flow Explanation:**

    1. **Track API** - Accept event, validate, enrich with metadata
    2. **Kafka publish** - Reliable, async event streaming
    3. **Real-time processing** - Update features, online learning
    4. **Batch storage** - Data lake for offline training and analytics

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical recommendation system components.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Collaborative Filtering** | How to leverage user-user and item-item patterns? | Matrix factorization, ALS, embeddings |
    | **Deep Learning Models** | How to capture complex patterns? | Two-tower networks, transformers, attention |
    | **Feature Engineering** | What features drive relevance? | User, item, context, cross features |
    | **A/B Testing Framework** | How to measure and improve? | Experimentation platform, metrics, analysis |

    ---

    === "🤝 Collaborative Filtering"

        ## The Challenge

        **Problem:** Recommend items to users based on patterns from similar users and items. Handle sparse data (< 0.01% density).

        **Collaborative filtering types:**

        1. **User-based:** Find similar users, recommend what they liked
        2. **Item-based:** Find similar items to what user liked
        3. **Matrix factorization:** Learn latent factors for users and items

        ---

        ## Matrix Factorization (ALS)

        **Alternating Least Squares (ALS) for implicit feedback:**

        **Goal:** Decompose sparse user-item matrix into user and item factor matrices.

        ```
        R ≈ U × I^T

        where:
        - R: user-item interaction matrix (200M × 10M, sparse)
        - U: user factor matrix (200M × k)
        - I: item factor matrix (10M × k)
        - k: latent dimensions (e.g., 256)
        ```

        **Why ALS:**

        - Handles implicit feedback (views, clicks) better than SVD
        - Parallelizable (Spark MLlib implementation)
        - Confidence weighting (frequent interactions weighted higher)

        ---

        ## Implementation (PySpark MLlib)

        ```python
        from pyspark.ml.recommendation import ALS
        from pyspark.ml.evaluation import RegressionEvaluator
        from pyspark.sql import SparkSession

        class CollaborativeFilteringModel:
            """
            ALS-based collaborative filtering model

            Learns latent factors for users and items from implicit feedback
            """

            def __init__(self, spark_session, rank=256, max_iter=10):
                """
                Args:
                    spark_session: SparkSession instance
                    rank: Number of latent factors (embedding dimension)
                    max_iter: Number of ALS iterations
                """
                self.spark = spark_session
                self.rank = rank
                self.max_iter = max_iter
                self.model = None

            def prepare_data(self, interactions_df):
                """
                Prepare training data from interactions

                Args:
                    interactions_df: DataFrame with (user_id, item_id, event_type, timestamp)

                Returns:
                    DataFrame with (user_id, item_id, confidence)
                """
                # Calculate confidence from interaction frequency and type
                confidence_df = interactions_df.groupBy("user_id", "item_id").agg(
                    # Weighted confidence based on event type
                    sum(
                        when(col("event_type") == "purchase", 5)
                        .when(col("event_type") == "rating", 4)
                        .when(col("event_type") == "click", 2)
                        .when(col("event_type") == "view", 1)
                        .otherwise(0)
                    ).alias("confidence")
                )

                # Apply log transformation to confidence
                # confidence = 1 + α * log(1 + interaction_count)
                confidence_df = confidence_df.withColumn(
                    "confidence",
                    lit(1) + lit(2) * log1p(col("confidence"))
                )

                return confidence_df

            def train(self, training_data):
                """
                Train ALS model

                Args:
                    training_data: DataFrame with (user_id, item_id, confidence)
                """
                print(f"Training ALS model with rank={self.rank}")

                als = ALS(
                    rank=self.rank,
                    maxIter=self.max_iter,
                    regParam=0.01,  # L2 regularization
                    userCol="user_id",
                    itemCol="item_id",
                    ratingCol="confidence",
                    coldStartStrategy="drop",  # Drop NaN predictions
                    implicitPrefs=True,  # Implicit feedback
                    nonnegative=True,  # Non-negative factors
                    alpha=40  # Confidence scaling for implicit feedback
                )

                self.model = als.fit(training_data)
                print("ALS model training complete")

            def generate_user_embeddings(self):
                """
                Extract learned user embeddings

                Returns:
                    DataFrame with (user_id, embedding)
                """
                user_factors = self.model.userFactors
                return user_factors.select("id", "features").toDF("user_id", "embedding")

            def generate_item_embeddings(self):
                """
                Extract learned item embeddings

                Returns:
                    DataFrame with (item_id, embedding)
                """
                item_factors = self.model.itemFactors
                return item_factors.select("id", "features").toDF("item_id", "embedding")

            def recommend_for_users(self, user_ids, num_recommendations=100):
                """
                Generate top-N recommendations for given users

                Args:
                    user_ids: List of user IDs
                    num_recommendations: Number of recommendations per user

                Returns:
                    DataFrame with (user_id, recommendations)
                """
                users_df = self.spark.createDataFrame(
                    [(uid,) for uid in user_ids],
                    ["user_id"]
                )

                recommendations = self.model.recommendForUserSubset(
                    users_df,
                    num_recommendations
                )

                return recommendations

            def recommend_similar_items(self, item_ids, num_similar=20):
                """
                Find similar items using item embeddings

                Args:
                    item_ids: List of item IDs
                    num_similar: Number of similar items per item

                Returns:
                    DataFrame with (item_id, similar_items)
                """
                items_df = self.spark.createDataFrame(
                    [(iid,) for iid in item_ids],
                    ["item_id"]
                )

                similar_items = self.model.recommendForItemSubset(
                    items_df,
                    num_similar
                )

                return similar_items

            def compute_item_similarity(self, item_embedding_df):
                """
                Compute item-item cosine similarity matrix

                Used for "Users who liked X also liked Y" recommendations

                Args:
                    item_embedding_df: DataFrame with (item_id, embedding)

                Returns:
                    DataFrame with (item_id_1, item_id_2, similarity)
                """
                from pyspark.ml.feature import BucketedRandomProjectionLSH

                # Use LSH for approximate nearest neighbors (faster than brute force)
                lsh = BucketedRandomProjectionLSH(
                    inputCol="embedding",
                    outputCol="hashes",
                    bucketLength=2.0,
                    numHashTables=3
                )

                lsh_model = lsh.fit(item_embedding_df)

                # Find approximate nearest neighbors
                similarity_df = lsh_model.approxSimilarityJoin(
                    item_embedding_df,
                    item_embedding_df,
                    threshold=2.0,  # Distance threshold
                    distCol="distance"
                ).select(
                    col("datasetA.item_id").alias("item_id_1"),
                    col("datasetB.item_id").alias("item_id_2"),
                    (lit(1) / (lit(1) + col("distance"))).alias("similarity")
                ).filter(col("item_id_1") != col("item_id_2"))

                return similarity_df

            def evaluate(self, test_data):
                """
                Evaluate model performance

                Args:
                    test_data: DataFrame with (user_id, item_id, confidence)

                Returns:
                    RMSE score
                """
                predictions = self.model.transform(test_data)

                evaluator = RegressionEvaluator(
                    metricName="rmse",
                    labelCol="confidence",
                    predictionCol="prediction"
                )

                rmse = evaluator.evaluate(predictions)
                print(f"Test RMSE: {rmse}")

                return rmse
        ```

        ---

        ## Training Pipeline

        ```python
        # Daily batch job to retrain ALS model
        def train_als_model():
            """
            Daily job to retrain collaborative filtering model
            """
            spark = SparkSession.builder \
                .appName("ALS Training") \
                .config("spark.executor.memory", "32g") \
                .config("spark.driver.memory", "16g") \
                .getOrCreate()

            # Load interactions from last 90 days (recency matters)
            interactions_df = spark.read \
                .format("parquet") \
                .load("s3://data-lake/interactions/") \
                .filter(col("timestamp") >= date_sub(current_date(), 90))

            print(f"Loaded {interactions_df.count()} interactions")

            # Initialize model
            cf_model = CollaborativeFilteringModel(
                spark_session=spark,
                rank=256,
                max_iter=10
            )

            # Prepare training data
            training_data = cf_model.prepare_data(interactions_df)

            # Split train/test (80/20)
            train_df, test_df = training_data.randomSplit([0.8, 0.2], seed=42)

            # Train model
            cf_model.train(train_df)

            # Evaluate
            rmse = cf_model.evaluate(test_df)

            # Extract embeddings
            user_embeddings = cf_model.generate_user_embeddings()
            item_embeddings = cf_model.generate_item_embeddings()

            # Save embeddings to vector database
            save_to_vector_db(user_embeddings, "user_embeddings_cf")
            save_to_vector_db(item_embeddings, "item_embeddings_cf")

            # Pre-compute recommendations for all users
            all_users = user_embeddings.select("user_id").collect()
            user_ids = [row.user_id for row in all_users]

            # Batch process in chunks
            batch_size = 10000
            for i in range(0, len(user_ids), batch_size):
                batch = user_ids[i:i+batch_size]
                recs = cf_model.recommend_for_users(batch, num_recommendations=100)

                # Save to cache (Redis)
                save_to_cache(recs)

            print("ALS model training and deployment complete")

            spark.stop()
        ```

        ---

        ## Item-Item Similarity

        **For "Similar Items" recommendations:**

        ```python
        class ItemSimilarityService:
            """
            Pre-compute and serve item-item similarities
            """

            def __init__(self, redis_client):
                self.redis = redis_client

            def compute_similarities(self, item_embeddings_df):
                """
                Compute cosine similarity between all item pairs

                Args:
                    item_embeddings_df: Spark DataFrame with (item_id, embedding)
                """
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                # Collect embeddings (10M items × 256 dims = 2.5GB, fits in memory)
                items = item_embeddings_df.collect()

                item_ids = [row.item_id for row in items]
                embeddings = np.array([row.embedding for row in items])

                print(f"Computing similarities for {len(item_ids)} items")

                # Compute cosine similarity matrix
                similarity_matrix = cosine_similarity(embeddings)

                # For each item, store top 100 similar items
                for i, item_id in enumerate(item_ids):
                    # Get top 100 similar items (excluding self)
                    similar_indices = np.argsort(similarity_matrix[i])[-101:-1][::-1]
                    similar_items = [
                        {
                            'item_id': item_ids[idx],
                            'score': float(similarity_matrix[i][idx])
                        }
                        for idx in similar_indices
                    ]

                    # Store in Redis
                    self.redis.setex(
                        f"similar_items:{item_id}",
                        3600 * 24,  # 24 hour TTL
                        json.dumps(similar_items)
                    )

                print("Item similarities computed and cached")

            def get_similar_items(self, item_id, count=20):
                """
                Get similar items from cache

                Args:
                    item_id: Item ID
                    count: Number of similar items

                Returns:
                    List of similar items with scores
                """
                cached = self.redis.get(f"similar_items:{item_id}")
                if cached:
                    similar_items = json.loads(cached)
                    return similar_items[:count]

                return []
        ```

        ---

        ## Cold Start Problem

        **Challenge:** New users/items have no interaction history.

        **Solutions:**

        1. **New users:**
           - Show popular/trending items
           - Ask onboarding questions (favorite categories)
           - Use demographic features (age, location)
           - Rapid learning from first few interactions

        2. **New items:**
           - Content-based recommendations (metadata, tags)
           - Explore/exploit strategy (show to sample of users)
           - Boost new items temporarily (freshness bonus)

        ```python
        def handle_cold_start(user_id, item_id, interaction_count):
            """
            Adjust recommendation strategy for cold start scenarios

            Args:
                user_id: User ID
                item_id: Item ID
                interaction_count: Number of interactions

            Returns:
                Recommendation strategy
            """
            user_interactions = get_user_interaction_count(user_id)
            item_interactions = get_item_interaction_count(item_id)

            if user_interactions < 5:
                # New user: popularity + demographics
                return {
                    'strategy': 'cold_start_user',
                    'weights': {
                        'popularity': 0.6,
                        'demographic': 0.3,
                        'collaborative': 0.1
                    }
                }
            elif item_interactions < 10:
                # New item: content-based + exploration
                return {
                    'strategy': 'cold_start_item',
                    'weights': {
                        'content_based': 0.7,
                        'trending': 0.2,
                        'collaborative': 0.1
                    }
                }
            else:
                # Normal: collaborative dominant
                return {
                    'strategy': 'standard',
                    'weights': {
                        'collaborative': 0.7,
                        'content_based': 0.2,
                        'popularity': 0.1
                    }
                }
        ```

    === "🧠 Deep Learning Models"

        ## The Challenge

        **Problem:** Capture complex, non-linear patterns in user behavior. Traditional collaborative filtering limited to linear interactions.

        **Deep learning advantages:**

        - Learn hierarchical features automatically
        - Handle multi-modal data (text, images, metadata)
        - Capture sequential patterns (session-based recommendations)
        - Better performance on cold start (with side features)

        ---

        ## Two-Tower Neural Network

        **Architecture:** Separate towers for user and item embeddings, dot product for scoring.

        ```
        User Tower                Item Tower
        ↓                         ↓
        [User Features]           [Item Features]
        ↓                         ↓
        Dense(512) + ReLU         Dense(512) + ReLU
        ↓                         ↓
        Dense(256) + ReLU         Dense(256) + ReLU
        ↓                         ↓
        Dense(128) + Norm         Dense(128) + Norm
        ↓                         ↓
        User Embedding (128)      Item Embedding (128)
        ↓_________________________↓
                    ↓
            Dot Product Score
        ```

        **Why Two-Tower:**

        - Independent encoding of users and items
        - Pre-compute item embeddings (fast inference)
        - Approximate nearest neighbor search for candidate generation
        - Scalable to billions of users and items

        ---

        ## Implementation (PyTorch)

        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class TwoTowerModel(nn.Module):
            """
            Two-tower neural network for recommendation

            Learns separate embeddings for users and items,
            scores via dot product
            """

            def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128):
                """
                Args:
                    user_feature_dim: Dimension of user features
                    item_feature_dim: Dimension of item features
                    embedding_dim: Dimension of final embeddings
                """
                super(TwoTowerModel, self).__init__()

                # User tower
                self.user_tower = nn.Sequential(
                    nn.Linear(user_feature_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),

                    nn.Linear(256, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                )

                # Item tower
                self.item_tower = nn.Sequential(
                    nn.Linear(item_feature_dim, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),

                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),

                    nn.Linear(256, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                )

            def forward(self, user_features, item_features):
                """
                Forward pass

                Args:
                    user_features: Tensor of shape (batch_size, user_feature_dim)
                    item_features: Tensor of shape (batch_size, item_feature_dim)

                Returns:
                    scores: Tensor of shape (batch_size,)
                """
                # Generate embeddings
                user_embeddings = self.user_tower(user_features)
                item_embeddings = self.item_tower(item_features)

                # L2 normalize embeddings (for cosine similarity)
                user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
                item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

                # Dot product score
                scores = (user_embeddings * item_embeddings).sum(dim=1)

                return scores, user_embeddings, item_embeddings

            def get_user_embedding(self, user_features):
                """Get user embedding for inference"""
                with torch.no_grad():
                    embedding = self.user_tower(user_features)
                    embedding = F.normalize(embedding, p=2, dim=1)
                return embedding

            def get_item_embedding(self, item_features):
                """Get item embedding for inference"""
                with torch.no_grad():
                    embedding = self.item_tower(item_features)
                    embedding = F.normalize(embedding, p=2, dim=1)
                return embedding


        class TwoTowerTrainer:
            """
            Training pipeline for two-tower model
            """

            def __init__(self, model, device='cuda'):
                self.model = model.to(device)
                self.device = device
                self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            def train_epoch(self, train_loader):
                """
                Train for one epoch

                Args:
                    train_loader: DataLoader with (user_features, item_features, labels)

                Returns:
                    Average loss
                """
                self.model.train()
                total_loss = 0
                num_batches = 0

                for batch in train_loader:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    labels = batch['label'].to(self.device)  # 1 for positive, 0 for negative

                    # Forward pass
                    scores, _, _ = self.model(user_features, item_features)

                    # Binary cross-entropy loss
                    loss = F.binary_cross_entropy_with_logits(scores, labels.float())

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                return total_loss / num_batches

            def train_with_negative_sampling(self, train_loader, num_negatives=4):
                """
                Train with in-batch negative sampling

                For each positive (user, item) pair, sample num_negatives negative items

                Args:
                    train_loader: DataLoader with positive examples
                    num_negatives: Number of negative samples per positive

                Returns:
                    Average loss
                """
                self.model.train()
                total_loss = 0
                num_batches = 0

                for batch in train_loader:
                    user_features = batch['user_features'].to(self.device)
                    positive_item_features = batch['item_features'].to(self.device)

                    batch_size = user_features.size(0)

                    # Generate user and item embeddings
                    user_embeddings = self.model.user_tower(user_features)
                    positive_item_embeddings = self.model.item_tower(positive_item_features)

                    # Normalize
                    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
                    positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=1)

                    # Positive scores
                    positive_scores = (user_embeddings * positive_item_embeddings).sum(dim=1)

                    # Negative sampling (random items from batch)
                    negative_item_embeddings = []
                    for _ in range(num_negatives):
                        # Shuffle item embeddings for negatives
                        perm = torch.randperm(batch_size)
                        negative_item_embeddings.append(positive_item_embeddings[perm])

                    # Calculate negative scores
                    negative_scores = []
                    for neg_embeddings in negative_item_embeddings:
                        scores = (user_embeddings * neg_embeddings).sum(dim=1)
                        negative_scores.append(scores)

                    # Concatenate positive and negative scores
                    all_scores = torch.cat([positive_scores.unsqueeze(1)] +
                                          [s.unsqueeze(1) for s in negative_scores], dim=1)

                    # Labels: [1, 0, 0, 0, 0] (first is positive)
                    labels = torch.zeros_like(all_scores)
                    labels[:, 0] = 1

                    # Softmax cross-entropy loss
                    loss = F.cross_entropy(all_scores, labels.argmax(dim=1))

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                return total_loss / num_batches

            def evaluate(self, val_loader, k=10):
                """
                Evaluate model with Recall@K and NDCG@K

                Args:
                    val_loader: Validation DataLoader
                    k: Top-K for metrics

                Returns:
                    Dictionary with metrics
                """
                self.model.eval()

                recall_at_k = []
                ndcg_at_k = []

                with torch.no_grad():
                    for batch in val_loader:
                        user_features = batch['user_features'].to(self.device)
                        positive_item_id = batch['item_id'].to(self.device)

                        # Get all item candidates (from batch)
                        all_item_features = batch['all_items'].to(self.device)

                        # Score all items
                        user_embedding = self.model.get_user_embedding(user_features)
                        item_embeddings = self.model.get_item_embedding(all_item_features)

                        # Compute scores
                        scores = torch.matmul(user_embedding, item_embeddings.T)

                        # Get top-K predictions
                        _, top_k_indices = torch.topk(scores, k, dim=1)

                        # Calculate metrics
                        for i in range(user_features.size(0)):
                            relevant = positive_item_id[i].item()
                            predicted = top_k_indices[i].cpu().tolist()

                            # Recall@K
                            recall_at_k.append(1.0 if relevant in predicted else 0.0)

                            # NDCG@K
                            if relevant in predicted:
                                rank = predicted.index(relevant) + 1
                                ndcg_at_k.append(1.0 / np.log2(rank + 1))
                            else:
                                ndcg_at_k.append(0.0)

                return {
                    f'recall@{k}': np.mean(recall_at_k),
                    f'ndcg@{k}': np.mean(ndcg_at_k)
                }
        ```

        ---

        ## Feature Engineering

        **User features:**

        ```python
        def extract_user_features(user_id, feature_store, interaction_db):
            """
            Extract comprehensive user features

            Returns:
                Feature vector (dense)
            """
            features = {}

            # Demographic features
            user_profile = feature_store.get_user_profile(user_id)
            features['age_bucket'] = user_profile['age'] // 10  # 20s, 30s, etc.
            features['gender'] = encode_categorical(user_profile['gender'])
            features['country'] = encode_categorical(user_profile['country'])

            # Historical features
            history = interaction_db.get_user_history(user_id, days=90)
            features['total_views'] = len(history)
            features['total_clicks'] = sum(1 for h in history if h['type'] == 'click')
            features['total_ratings'] = sum(1 for h in history if h['type'] == 'rating')
            features['avg_rating'] = np.mean([h['rating'] for h in history if 'rating' in h])

            # Category preferences (TF-IDF)
            categories = [h['category'] for h in history]
            category_counts = Counter(categories)
            top_categories = category_counts.most_common(5)
            for cat, count in top_categories:
                features[f'category_{cat}'] = count / len(categories)

            # Temporal features
            features['active_days'] = len(set(h['timestamp'].date() for h in history))
            features['avg_session_duration'] = calculate_avg_session_duration(history)

            # Recent behavior (last 24 hours)
            recent_history = [h for h in history if h['timestamp'] > datetime.now() - timedelta(days=1)]
            features['views_24h'] = len(recent_history)
            features['sessions_24h'] = count_sessions(recent_history)

            # Engagement metrics
            features['engagement_score'] = calculate_engagement_score(history)
            features['diversity_score'] = calculate_diversity_score(history)

            return vectorize_features(features)
        ```

        **Item features:**

        ```python
        def extract_item_features(item_id, item_db, stats_db):
            """
            Extract comprehensive item features

            Returns:
                Feature vector (dense)
            """
            features = {}

            # Content features
            item = item_db.get_item(item_id)
            features['category'] = encode_categorical(item['category'])
            features['subcategory'] = encode_categorical(item['subcategory'])
            features['tags'] = encode_multi_hot(item['tags'], vocab_size=1000)

            # Quality features
            features['avg_rating'] = item['avg_rating']
            features['rating_count'] = np.log1p(item['rating_count'])
            features['quality_score'] = calculate_quality_score(item)

            # Popularity features
            stats = stats_db.get_item_stats(item_id)
            features['view_count_24h'] = np.log1p(stats['view_count_24h'])
            features['view_count_7d'] = np.log1p(stats['view_count_7d'])
            features['ctr_24h'] = stats['click_count_24h'] / max(stats['view_count_24h'], 1)

            # Temporal features
            features['days_since_created'] = (datetime.now() - item['created_at']).days
            features['recency_score'] = calculate_recency_score(item['created_at'])

            # Velocity features (trending indicator)
            features['view_velocity_1h'] = stats['view_count_1h'] / stats['view_count_24h']
            features['trending_score'] = calculate_trending_score(stats)

            # Text embeddings (from item description)
            features['text_embedding'] = get_text_embedding(item['description'])

            return vectorize_features(features)
        ```

        ---

        ## Transformer-based Sequential Model

        **For session-based recommendations (next item prediction):**

        ```python
        class TransformerRecommender(nn.Module):
            """
            Transformer model for sequential recommendations

            Predicts next item based on user's interaction sequence
            """

            def __init__(self, num_items, embedding_dim=128, num_heads=8, num_layers=4):
                super(TransformerRecommender, self).__init__()

                # Item embedding
                self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

                # Positional encoding
                self.positional_encoding = PositionalEncoding(embedding_dim)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # Output projection
                self.output_layer = nn.Linear(embedding_dim, num_items)

            def forward(self, item_sequence, sequence_mask=None):
                """
                Args:
                    item_sequence: Tensor of shape (batch_size, seq_len)
                    sequence_mask: Tensor of shape (batch_size, seq_len) for padding

                Returns:
                    logits: Tensor of shape (batch_size, num_items)
                """
                # Embed items
                embedded = self.item_embedding(item_sequence)  # (batch, seq_len, embed_dim)

                # Add positional encoding
                embedded = self.positional_encoding(embedded)

                # Transformer encoding
                encoded = self.transformer(embedded, src_key_padding_mask=sequence_mask)

                # Take last item representation (for next item prediction)
                last_hidden = encoded[:, -1, :]  # (batch, embed_dim)

                # Project to item logits
                logits = self.output_layer(last_hidden)  # (batch, num_items)

                return logits


        class PositionalEncoding(nn.Module):
            """Positional encoding for transformer"""

            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                    (-np.log(10000.0) / d_model))

                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)

                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                """
                Args:
                    x: Tensor of shape (batch, seq_len, d_model)
                """
                return x + self.pe[:, :x.size(1), :]
        ```

    === "🎯 Feature Engineering"

        ## The Challenge

        **Problem:** What features drive recommendation relevance? How to engineer features at scale?

        **Feature categories:**

        1. **User features:** Demographics, history, preferences
        2. **Item features:** Content, popularity, quality
        3. **Context features:** Time, device, location, session
        4. **Cross features:** User-item interactions, similarity

        ---

        ## Feature Store Architecture

        **Centralized feature management for online/offline consistency:**

        ```python
        from feast import FeatureStore, Entity, FeatureView, Field, FileSource
        from feast.types import Float32, Int64, String
        from datetime import timedelta

        class RecommendationFeatureStore:
            """
            Feature store for recommendation system

            Manages online and offline features with consistency
            """

            def __init__(self, repo_path="feature_repo/"):
                self.store = FeatureStore(repo_path=repo_path)

            def define_entities(self):
                """Define entities (user, item)"""
                user_entity = Entity(
                    name="user",
                    join_keys=["user_id"],
                    description="User entity"
                )

                item_entity = Entity(
                    name="item",
                    join_keys=["item_id"],
                    description="Item entity"
                )

                return user_entity, item_entity

            def define_user_features(self):
                """
                Define user feature views

                Includes both batch and real-time features
                """
                # Batch features (computed daily)
                user_batch_features = FeatureView(
                    name="user_batch_features",
                    entities=["user"],
                    ttl=timedelta(days=1),
                    schema=[
                        Field(name="age_bucket", dtype=Int64),
                        Field(name="gender", dtype=String),
                        Field(name="country", dtype=String),
                        Field(name="total_views_90d", dtype=Int64),
                        Field(name="total_clicks_90d", dtype=Int64),
                        Field(name="avg_rating", dtype=Float32),
                        Field(name="favorite_category", dtype=String),
                        Field(name="engagement_score", dtype=Float32),
                        Field(name="diversity_score", dtype=Float32),
                    ],
                    source=FileSource(
                        path="s3://feature-store/user_batch_features.parquet",
                        timestamp_field="event_timestamp"
                    )
                )

                # Real-time features (streaming)
                user_realtime_features = FeatureView(
                    name="user_realtime_features",
                    entities=["user"],
                    ttl=timedelta(hours=1),
                    schema=[
                        Field(name="views_1h", dtype=Int64),
                        Field(name="clicks_1h", dtype=Int64),
                        Field(name="sessions_24h", dtype=Int64),
                        Field(name="current_session_duration", dtype=Int64),
                        Field(name="items_viewed_session", dtype=Int64),
                        Field(name="last_category_viewed", dtype=String),
                    ],
                    source=FileSource(
                        path="redis://feature-cache/user_realtime",
                        timestamp_field="event_timestamp"
                    )
                )

                return user_batch_features, user_realtime_features

            def define_item_features(self):
                """Define item feature views"""
                # Batch features (computed daily)
                item_batch_features = FeatureView(
                    name="item_batch_features",
                    entities=["item"],
                    ttl=timedelta(days=1),
                    schema=[
                        Field(name="category", dtype=String),
                        Field(name="subcategory", dtype=String),
                        Field(name="avg_rating", dtype=Float32),
                        Field(name="rating_count", dtype=Int64),
                        Field(name="quality_score", dtype=Float32),
                        Field(name="days_since_created", dtype=Int64),
                        Field(name="view_count_7d", dtype=Int64),
                        Field(name="ctr_7d", dtype=Float32),
                    ],
                    source=FileSource(
                        path="s3://feature-store/item_batch_features.parquet",
                        timestamp_field="event_timestamp"
                    )
                )

                # Real-time features (streaming)
                item_realtime_features = FeatureView(
                    name="item_realtime_features",
                    entities=["item"],
                    ttl=timedelta(hours=1),
                    schema=[
                        Field(name="view_count_1h", dtype=Int64),
                        Field(name="click_count_1h", dtype=Int64),
                        Field(name="ctr_1h", dtype=Float32),
                        Field(name="trending_score", dtype=Float32),
                        Field(name="view_velocity", dtype=Float32),
                    ],
                    source=FileSource(
                        path="redis://feature-cache/item_realtime",
                        timestamp_field="event_timestamp"
                    )
                )

                return item_batch_features, item_realtime_features

            def get_online_features(self, user_ids, item_ids):
                """
                Fetch features for online serving (low latency)

                Args:
                    user_ids: List of user IDs
                    item_ids: List of item IDs

                Returns:
                    Feature vectors for users and items
                """
                # Prepare entity rows
                entity_rows = [
                    {
                        "user_id": uid,
                        "item_id": iid
                    }
                    for uid, iid in zip(user_ids, item_ids)
                ]

                # Fetch features
                feature_vector = self.store.get_online_features(
                    features=[
                        "user_batch_features:age_bucket",
                        "user_batch_features:total_views_90d",
                        "user_batch_features:engagement_score",
                        "user_realtime_features:views_1h",
                        "user_realtime_features:sessions_24h",
                        "item_batch_features:category",
                        "item_batch_features:avg_rating",
                        "item_batch_features:ctr_7d",
                        "item_realtime_features:trending_score",
                        "item_realtime_features:view_velocity",
                    ],
                    entity_rows=entity_rows
                ).to_dict()

                return feature_vector
        ```

        ---

        ## Real-time Feature Computation

        **Streaming features using Kafka Streams:**

        ```python
        from kafka import KafkaConsumer
        import redis
        from collections import defaultdict
        import json

        class RealtimeFeatureComputer:
            """
            Compute real-time features from interaction stream
            """

            def __init__(self, kafka_brokers, redis_host):
                self.consumer = KafkaConsumer(
                    'user_interactions',
                    bootstrap_servers=kafka_brokers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )
                self.redis = redis.Redis(host=redis_host, decode_responses=True)

                # In-memory state (or use Redis for distributed)
                self.user_session_state = defaultdict(dict)
                self.item_velocity_state = defaultdict(list)

            def process_stream(self):
                """Process interaction events and update features"""
                for message in self.consumer:
                    event = message.value
                    self.update_user_features(event)
                    self.update_item_features(event)

            def update_user_features(self, event):
                """
                Update user real-time features

                Args:
                    event: Interaction event
                """
                user_id = event['user_id']
                event_type = event['event_type']
                timestamp = event['timestamp']

                # Get current features
                feature_key = f"user_features:{user_id}"
                features = self.redis.hgetall(feature_key)
                if not features:
                    features = {
                        'views_1h': 0,
                        'clicks_1h': 0,
                        'sessions_24h': 1,
                        'current_session_duration': 0,
                        'items_viewed_session': 0,
                        'last_category_viewed': '',
                    }

                # Update counters
                if event_type == 'view':
                    features['views_1h'] = int(features.get('views_1h', 0)) + 1
                    features['items_viewed_session'] = int(features.get('items_viewed_session', 0)) + 1
                elif event_type == 'click':
                    features['clicks_1h'] = int(features.get('clicks_1h', 0)) + 1

                # Update session duration
                if 'session_start' in self.user_session_state[user_id]:
                    session_start = self.user_session_state[user_id]['session_start']
                    duration = (timestamp - session_start).total_seconds()
                    features['current_session_duration'] = int(duration)
                else:
                    self.user_session_state[user_id]['session_start'] = timestamp

                # Update last category
                features['last_category_viewed'] = event.get('category', '')

                # Store back to Redis with TTL
                pipeline = self.redis.pipeline()
                for key, value in features.items():
                    pipeline.hset(feature_key, key, value)
                pipeline.expire(feature_key, 3600)  # 1 hour TTL
                pipeline.execute()

            def update_item_features(self, event):
                """
                Update item real-time features

                Args:
                    event: Interaction event
                """
                item_id = event['item_id']
                event_type = event['event_type']
                timestamp = event['timestamp']

                # Get current features
                feature_key = f"item_features:{item_id}"
                features = self.redis.hgetall(feature_key)
                if not features:
                    features = {
                        'view_count_1h': 0,
                        'click_count_1h': 0,
                        'ctr_1h': 0.0,
                        'trending_score': 0.0,
                        'view_velocity': 0.0,
                    }

                # Update counters
                if event_type == 'view':
                    features['view_count_1h'] = int(features.get('view_count_1h', 0)) + 1

                    # Update velocity (views per minute)
                    self.item_velocity_state[item_id].append(timestamp)
                    # Keep only last hour
                    cutoff = timestamp - timedelta(hours=1)
                    self.item_velocity_state[item_id] = [
                        ts for ts in self.item_velocity_state[item_id] if ts > cutoff
                    ]
                    features['view_velocity'] = len(self.item_velocity_state[item_id]) / 60.0

                elif event_type == 'click':
                    features['click_count_1h'] = int(features.get('click_count_1h', 0)) + 1

                # Calculate CTR
                view_count = int(features['view_count_1h'])
                click_count = int(features['click_count_1h'])
                features['ctr_1h'] = click_count / max(view_count, 1)

                # Calculate trending score (velocity × CTR)
                features['trending_score'] = float(features['view_velocity']) * features['ctr_1h']

                # Store back to Redis
                pipeline = self.redis.pipeline()
                for key, value in features.items():
                    pipeline.hset(feature_key, key, value)
                pipeline.expire(feature_key, 3600)
                pipeline.execute()
        ```

        ---

        ## Cross Features

        **Interaction features between user and item:**

        ```python
        def generate_cross_features(user_features, item_features, interaction_history):
            """
            Generate cross features from user-item interactions

            Args:
                user_features: User feature dict
                item_features: Item feature dict
                interaction_history: Past user-item interactions

            Returns:
                Cross feature dict
            """
            cross_features = {}

            # Category affinity
            user_favorite_category = user_features['favorite_category']
            item_category = item_features['category']
            cross_features['category_match'] = 1 if user_favorite_category == item_category else 0

            # Historical interaction
            cross_features['user_has_viewed_item'] = int(
                interaction_history.has_interaction(user_id, item_id, 'view')
            )
            cross_features['user_has_clicked_item'] = int(
                interaction_history.has_interaction(user_id, item_id, 'click')
            )

            # Timing match
            user_active_hours = user_features['active_hours']  # [19, 20, 21, 22]
            current_hour = datetime.now().hour
            cross_features['timing_match'] = 1 if current_hour in user_active_hours else 0

            # Popularity vs user preference
            user_engagement = user_features['engagement_score']
            item_popularity = item_features['popularity_score']
            cross_features['popularity_user_match'] = abs(user_engagement - item_popularity)

            # Recency match
            user_prefers_new = user_features.get('prefers_new_content', 0)
            item_recency = item_features['recency_score']
            cross_features['recency_match'] = user_prefers_new * item_recency

            # Quality match
            user_avg_rating = user_features['avg_rating']
            item_avg_rating = item_features['avg_rating']
            cross_features['quality_match'] = abs(user_avg_rating - item_avg_rating)

            return cross_features
        ```

    === "🧪 A/B Testing Framework"

        ## The Challenge

        **Problem:** How to measure recommendation quality? How to safely deploy model changes?

        **Requirements:**

        - Controlled experiments with statistical significance
        - Multiple metrics (CTR, engagement, revenue)
        - Fast iteration (deploy new models weekly)
        - Guard rails (prevent major regressions)

        ---

        ## Experimentation Platform

        ```python
        import hashlib
        from enum import Enum
        from dataclasses import dataclass
        from typing import Dict, List

        class ExperimentStatus(Enum):
            DRAFT = "draft"
            RUNNING = "running"
            PAUSED = "paused"
            COMPLETED = "completed"

        @dataclass
        class ExperimentConfig:
            """Configuration for A/B test"""
            experiment_id: str
            name: str
            description: str
            status: ExperimentStatus
            variants: List[Dict]  # [{"name": "control", "weight": 0.5}, ...]
            start_date: str
            end_date: str
            target_sample_size: int
            metrics: List[str]  # ["ctr", "engagement_time", "conversion_rate"]

        class ABTestService:
            """
            A/B testing service for recommendation experiments
            """

            def __init__(self, db, cache):
                self.db = db
                self.cache = cache

            def assign_variant(self, user_id: str, experiment_id: str) -> str:
                """
                Assign user to experiment variant

                Uses consistent hashing for stable assignment

                Args:
                    user_id: User ID
                    experiment_id: Experiment ID

                Returns:
                    Variant name (e.g., "control", "treatment_1")
                """
                # Check cache first
                cache_key = f"experiment:{experiment_id}:user:{user_id}"
                cached_variant = self.cache.get(cache_key)
                if cached_variant:
                    return cached_variant

                # Get experiment config
                experiment = self.get_experiment(experiment_id)
                if not experiment or experiment.status != ExperimentStatus.RUNNING:
                    return "control"  # Default to control if experiment not active

                # Consistent hashing for variant assignment
                hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
                hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
                bucket = hash_value % 100  # 0-99

                # Assign based on variant weights
                cumulative_weight = 0
                for variant in experiment.variants:
                    cumulative_weight += variant['weight'] * 100
                    if bucket < cumulative_weight:
                        assigned_variant = variant['name']
                        break
                else:
                    assigned_variant = "control"

                # Cache assignment (TTL = experiment duration)
                self.cache.setex(cache_key, 86400 * 30, assigned_variant)

                return assigned_variant

            def get_variant_config(self, experiment_id: str, variant: str) -> Dict:
                """
                Get model configuration for variant

                Args:
                    experiment_id: Experiment ID
                    variant: Variant name

                Returns:
                    Configuration dict (model version, hyperparameters, etc.)
                """
                config_key = f"experiment:{experiment_id}:variant:{variant}:config"
                config = self.cache.get(config_key)

                if not config:
                    config = self.db.get_variant_config(experiment_id, variant)
                    self.cache.setex(config_key, 3600, json.dumps(config))
                else:
                    config = json.loads(config)

                return config

            def log_experiment_event(self, user_id: str, experiment_id: str,
                                    variant: str, event_type: str, metadata: Dict):
                """
                Log event for experiment analysis

                Args:
                    user_id: User ID
                    experiment_id: Experiment ID
                    variant: Assigned variant
                    event_type: Event type (impression, click, conversion)
                    metadata: Event metadata
                """
                event = {
                    'user_id': user_id,
                    'experiment_id': experiment_id,
                    'variant': variant,
                    'event_type': event_type,
                    'timestamp': datetime.utcnow().isoformat(),
                    'metadata': metadata
                }

                # Publish to Kafka for analytics pipeline
                self.kafka_producer.send('experiment_events', value=json.dumps(event))

            def calculate_metrics(self, experiment_id: str) -> Dict:
                """
                Calculate experiment metrics for all variants

                Args:
                    experiment_id: Experiment ID

                Returns:
                    Metrics dict by variant
                """
                experiment = self.get_experiment(experiment_id)

                results = {}
                for variant in experiment.variants:
                    variant_name = variant['name']

                    # Query events from data warehouse
                    events = self.db.query_experiment_events(
                        experiment_id=experiment_id,
                        variant=variant_name
                    )

                    # Calculate metrics
                    impressions = len([e for e in events if e['event_type'] == 'impression'])
                    clicks = len([e for e in events if e['event_type'] == 'click'])
                    conversions = len([e for e in events if e['event_type'] == 'conversion'])

                    results[variant_name] = {
                        'impressions': impressions,
                        'clicks': clicks,
                        'conversions': conversions,
                        'ctr': clicks / max(impressions, 1),
                        'conversion_rate': conversions / max(impressions, 1),
                        'engagement_time': self._calculate_engagement_time(events),
                    }

                return results

            def run_significance_test(self, experiment_id: str,
                                     metric: str = 'ctr') -> Dict:
                """
                Run statistical significance test

                Args:
                    experiment_id: Experiment ID
                    metric: Metric to test (ctr, conversion_rate, etc.)

                Returns:
                    Test results with p-value and confidence interval
                """
                from scipy import stats

                results = self.calculate_metrics(experiment_id)

                # Compare each treatment to control
                control_data = results['control']

                test_results = {}
                for variant_name, variant_data in results.items():
                    if variant_name == 'control':
                        continue

                    # Proportion z-test for CTR/conversion rate
                    control_successes = control_data['clicks']
                    control_trials = control_data['impressions']
                    treatment_successes = variant_data['clicks']
                    treatment_trials = variant_data['impressions']

                    # Calculate z-score and p-value
                    p_control = control_successes / control_trials
                    p_treatment = treatment_successes / treatment_trials

                    pooled_p = (control_successes + treatment_successes) / \
                              (control_trials + treatment_trials)

                    se = np.sqrt(pooled_p * (1 - pooled_p) *
                                (1/control_trials + 1/treatment_trials))

                    z_score = (p_treatment - p_control) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                    # Calculate confidence interval (95%)
                    ci_margin = 1.96 * se
                    ci_lower = (p_treatment - p_control) - ci_margin
                    ci_upper = (p_treatment - p_control) + ci_margin

                    test_results[variant_name] = {
                        'control_ctr': p_control,
                        'treatment_ctr': p_treatment,
                        'lift': (p_treatment - p_control) / p_control,
                        'z_score': z_score,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'ci_95': (ci_lower, ci_upper)
                    }

                return test_results
        ```

        ---

        ## Recommendation API with A/B Testing

        ```python
        class RecommendationAPIWithExperiments:
            """
            Recommendation API integrated with A/B testing
            """

            def __init__(self, ab_service, model_registry):
                self.ab_service = ab_service
                self.model_registry = model_registry

            async def get_recommendations(self, request):
                """
                Get recommendations with A/B testing

                Args:
                    request: Recommendation request

                Returns:
                    Recommendations with experiment metadata
                """
                user_id = request.user_id
                context = request.context

                # Get active experiments
                active_experiments = self.ab_service.get_active_experiments(user_id)

                # Assign variants
                experiment_assignments = {}
                for experiment in active_experiments:
                    variant = self.ab_service.assign_variant(user_id, experiment.experiment_id)
                    experiment_assignments[experiment.experiment_id] = variant

                # Get model config based on experiment assignments
                model_config = self._get_model_config(experiment_assignments)

                # Load appropriate model
                model = self.model_registry.get_model(
                    model_version=model_config['model_version']
                )

                # Generate recommendations
                recommendations = await model.recommend(
                    user_id=user_id,
                    context=context,
                    config=model_config
                )

                # Log impression event
                for experiment_id, variant in experiment_assignments.items():
                    self.ab_service.log_experiment_event(
                        user_id=user_id,
                        experiment_id=experiment_id,
                        variant=variant,
                        event_type='impression',
                        metadata={
                            'recommendations': [r['item_id'] for r in recommendations],
                            'request_id': request.request_id
                        }
                    )

                # Return with experiment metadata
                return {
                    'recommendations': recommendations,
                    'experiments': experiment_assignments,
                    'model_version': model_config['model_version']
                }
        ```

        ---

        ## Key Metrics

        **Primary metrics:**

        | Metric | Formula | Target | Interpretation |
        |--------|---------|--------|----------------|
        | **CTR** | Clicks / Impressions | > 5% | User interest |
        | **Engagement Rate** | (Clicks + Views) / Impressions | > 30% | Overall interaction |
        | **Conversion Rate** | Conversions / Impressions | > 1% | Business value |
        | **Session Time** | Avg time spent per session | > 10 min | User satisfaction |

        **Secondary metrics:**

        | Metric | Formula | Target | Interpretation |
        |--------|---------|--------|----------------|
        | **Diversity** | Unique categories / Total items | > 0.3 | Content variety |
        | **Coverage** | Items recommended / Total catalog | > 20% | Long-tail exposure |
        | **Novelty** | New items / Total recommendations | > 0.1 | Discovery |
        | **Serendipity** | Surprising hits / Total clicks | > 0.1 | Delight factor |

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling recommendation system from 1M to 200M DAU.

    **Scaling challenges at 200M DAU:**

    - **Recommendation QPS:** 104K peak QPS
    - **Interaction ingestion:** 115K events/sec
    - **Model training:** 36T interactions, daily retraining
    - **Feature serving:** < 10ms latency for feature lookup

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Candidate generation** | ✅ Yes | Vector database (ANN), pre-computed candidates, caching |
    | **Feature serving** | ✅ Yes | Feature store with Redis cache, batch feature lookup |
    | **Model serving** | ✅ Yes | Model caching, GPU inference, batch scoring |
    | **Real-time features** | ✅ Yes | Kafka Streams, distributed state, incremental computation |
    | **Training pipeline** | 🟡 Approaching | Spark cluster (100+ nodes), incremental learning |

    ---

    ## Optimization Strategies

    ### 1. Multi-Stage Ranking

    **Funnel approach to reduce latency:**

    ```
    10M items (catalog)
    ↓
    [Stage 1: Fast Candidate Generation]
    → 1000 candidates (< 10ms)
    ↓
    [Stage 2: Coarse Ranking]
    → 100 candidates (< 50ms)
    ↓
    [Stage 3: Fine Ranking (ML model)]
    → 50 candidates (< 100ms)
    ↓
    [Stage 4: Reranking (diversity, business rules)]
    → 20 final recommendations (< 150ms)
    ```

    ---

    ### 2. Caching Strategy

    **Multi-level caching:**

    ```python
    class MultiLevelCache:
        """
        Multi-level caching for recommendations

        L1: In-memory (local server)
        L2: Redis (distributed)
        L3: Pre-computed (database)
        """

        def __init__(self):
            self.l1_cache = {}  # In-memory
            self.l2_cache = redis.Redis(host='redis-cluster')
            self.db = database.connect()

        def get_recommendations(self, user_id, context):
            """Get recommendations with caching"""
            cache_key = self._generate_cache_key(user_id, context)

            # L1: In-memory cache (< 1ms)
            if cache_key in self.l1_cache:
                return self.l1_cache[cache_key]

            # L2: Redis cache (< 5ms)
            cached = self.l2_cache.get(cache_key)
            if cached:
                recs = json.loads(cached)
                self.l1_cache[cache_key] = recs  # Promote to L1
                return recs

            # L3: Pre-computed recommendations (< 20ms)
            recs = self.db.get_precomputed_recs(user_id)
            if recs:
                # Personalize with context
                recs = self._personalize(recs, context)

                # Cache at L2 and L1
                self.l2_cache.setex(cache_key, 300, json.dumps(recs))
                self.l1_cache[cache_key] = recs

                return recs

            # Cache miss: Generate fresh recommendations
            recs = self._generate_fresh_recs(user_id, context)

            # Cache at all levels
            self.l2_cache.setex(cache_key, 300, json.dumps(recs))
            self.l1_cache[cache_key] = recs

            return recs
    ```

    ---

    ### 3. Approximate Nearest Neighbors (ANN)

    **Fast similarity search using vector databases:**

    ```python
    from pinecone import Pinecone

    class VectorSearchService:
        """
        Vector similarity search using Pinecone
        """

        def __init__(self, api_key, index_name):
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)

        def search_similar_users(self, user_embedding, top_k=100):
            """
            Find similar users using ANN

            Args:
                user_embedding: User embedding vector (256-dim)
                top_k: Number of similar users

            Returns:
                List of (user_id, similarity_score)
            """
            results = self.index.query(
                vector=user_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                namespace="user_embeddings"
            )

            return [(r.id, r.score) for r in results.matches]

        def search_similar_items(self, item_embedding, top_k=1000,
                                 filters=None):
            """
            Find similar items using ANN with optional filters

            Args:
                item_embedding: Item embedding vector
                top_k: Number of similar items
                filters: Optional metadata filters (category, rating, etc.)

            Returns:
                List of (item_id, similarity_score)
            """
            results = self.index.query(
                vector=item_embedding.tolist(),
                top_k=top_k,
                filter=filters,  # e.g., {"category": "action", "rating": {"$gte": 4.0}}
                include_metadata=True,
                namespace="item_embeddings"
            )

            return [(r.id, r.score) for r in results.matches]
    ```

    ---

    ### 4. Model Optimization

    **Reduce inference latency:**

    1. **Model quantization:** INT8 quantization (4x smaller, 2-4x faster)
    2. **Model distillation:** Train smaller student model from large teacher
    3. **GPU inference:** Batch scoring on GPU (10x faster)
    4. **ONNX optimization:** Convert to ONNX for optimized inference

    ```python
    import onnxruntime as ort

    class OptimizedModelServing:
        """
        Optimized model serving with ONNX
        """

        def __init__(self, model_path):
            # Load ONNX model
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

        def batch_score(self, user_features, item_features):
            """
            Batch score multiple user-item pairs

            Args:
                user_features: Numpy array (batch_size, user_feature_dim)
                item_features: Numpy array (batch_size, item_feature_dim)

            Returns:
                Scores: Numpy array (batch_size,)
            """
            # Prepare inputs
            inputs = {
                'user_features': user_features.astype(np.float32),
                'item_features': item_features.astype(np.float32)
            }

            # Run inference
            outputs = self.session.run(None, inputs)
            scores = outputs[0]

            return scores
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 200M DAU:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $43,200 (300 × m5.2xlarge) |
    | **GPU inference servers** | $86,400 (40 × p3.2xlarge) |
    | **Cassandra cluster** | $108,000 (250 nodes) |
    | **Redis cache** | $21,600 (100 nodes) |
    | **Vector database** | $32,400 (Pinecone) |
    | **Kafka cluster** | $10,800 (25 brokers) |
    | **Spark cluster (training)** | $14,400 (100 nodes, 8 hrs/day) |
    | **S3 storage** | $4,600 (2 PB) |
    | **Data transfer** | $8,500 (100 TB egress) |
    | **Total** | **$329,900/month** |

    **Optimization opportunities:**

    - Use spot instances for training (50% savings)
    - Archive old interactions to cheaper storage (Glacier)
    - Cache optimization (reduce cache size by 30%)
    - Model compression (reduce GPU requirements)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Recommendation Latency (P95)** | < 200ms | > 500ms |
    | **Cache Hit Rate** | > 70% | < 50% |
    | **Model Inference Latency** | < 50ms | > 200ms |
    | **Feature Serving Latency** | < 10ms | > 50ms |
    | **CTR** | > 5% | < 3% |
    | **Engagement Rate** | > 30% | < 20% |
    | **Kafka Lag** | < 1000 msgs | > 10000 msgs |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Hybrid approach:** Combine collaborative filtering, content-based, and deep learning
    2. **Multi-stage ranking:** Fast candidate generation → ranking → reranking
    3. **Feature engineering:** Rich features from user, item, context, and cross-interactions
    4. **Two-tower model:** Scalable deep learning with independent user/item encoding
    5. **Vector database:** Fast ANN search for candidate generation
    6. **A/B testing:** Continuous experimentation for model improvement
    7. **Real-time features:** Kafka Streams for low-latency feature updates
    8. **Multi-level caching:** In-memory + Redis + pre-computed recommendations

    ---

    ## Interview Tips

    ✅ **Start with problem clarification** - Understand the domain (movies vs products vs videos)

    ✅ **Discuss cold start early** - Show awareness of new user/item challenges

    ✅ **Explain trade-offs** - Collaborative vs content-based vs hybrid

    ✅ **Multi-stage ranking** - Critical for latency at scale

    ✅ **Feature engineering matters** - Real-time features drive relevance

    ✅ **A/B testing is essential** - How to measure and improve

    ✅ **Diversity vs relevance** - Balance exploration and exploitation

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle cold start?"** | New users: popular items + demographics; New items: content-based + exploration |
    | **"How to ensure diversity?"** | Reranking with MMR algorithm, category spread, novelty bonus |
    | **"How to make recommendations explainable?"** | Store attribution ("Because you watched X"), content-based reasons |
    | **"How to handle seasonality?"** | Time-decay features, seasonal multipliers, trending signals |
    | **"How to measure recommendation quality?"** | CTR, engagement rate, conversion rate, A/B testing, long-term retention |
    | **"How to prevent filter bubbles?"** | Exploration (ε-greedy), diversity reranking, serendipity boost |
    | **"How to handle real-time personalization?"** | Real-time feature store, incremental learning, context-aware ranking |
    | **"How to scale to billions of users?"** | Vector database for ANN, distributed feature store, multi-level caching |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Netflix, Amazon, YouTube, Spotify, TikTok, LinkedIn, Pinterest

---

*Master this problem and you'll be ready for: YouTube (video recommendations), Spotify (music recommendations), TikTok (short-form video), Amazon (product recommendations), Uber Eats (restaurant recommendations)*
