# Design Yelp / Nearby Places

A location-based search platform where users can discover nearby businesses, restaurants, and services, read reviews, view photos, and get recommendations based on their current location.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 50-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 200M users, 100M businesses, 500M searches/day, 10M reviews/day |
| **Key Challenges** | Geospatial indexing, sub-200ms search, relevance ranking, real-time updates |
| **Core Concepts** | Quadtree, Geohash, Redis Geo, spatial indexing, ranking algorithms |
| **Companies** | Yelp, Google Maps, Foursquare, TripAdvisor, Zomato, Uber Eats |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Search Nearby** | Find businesses within radius based on location | P0 (Must have) |
    | **Filter Results** | Filter by distance, rating, price, category, open now | P0 (Must have) |
    | **View Details** | Business info: hours, photos, menu, contact | P0 (Must have) |
    | **Reviews & Ratings** | Read/write reviews, rate businesses (1-5 stars) | P0 (Must have) |
    | **Upload Photos** | Users can upload business photos | P1 (Should have) |
    | **Check-ins** | Users can check-in at locations | P1 (Should have) |
    | **Recommendations** | Personalized suggestions based on history | P2 (Nice to have) |
    | **Navigation** | Get directions to business | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Reservation system (OpenTable integration)
    - Food delivery (UberEats/DoorDash)
    - Payment processing
    - Business analytics dashboard
    - Social features (friends, activity feed)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users depend on search for immediate needs |
    | **Latency (Search)** | < 200ms p95 | Fast results critical for mobile UX |
    | **Latency (Reviews)** | < 500ms | Slightly more tolerance for writes |
    | **Consistency** | Eventual consistency | Brief delays acceptable for reviews/ratings |
    | **Scalability** | 500M searches/day | Handle peak times (lunch, dinner hours) |
    | **Accuracy** | < 50m location error | Precise geospatial calculations |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Users:
    - Total users: 200M
    - Daily Active Users (DAU): 50M (25% of total)
    - Monthly Active Users (MAU): 120M (60% of total)

    Search traffic:
    - Searches per DAU: 10 searches/day
    - Daily searches: 50M × 10 = 500M searches/day
    - Search QPS: 500M / 86,400 = ~5,800 searches/sec
    - Peak QPS (lunch/dinner): 3x average = ~17,400 searches/sec

    Business views:
    - 40% of searches → business view
    - Daily views: 500M × 0.4 = 200M views/day
    - View QPS: 200M / 86,400 = ~2,300 views/sec

    Reviews:
    - 5% of views → review read
    - Daily review reads: 200M × 0.05 = 10M reads/day
    - 1% of views → review write
    - Daily review writes: 200M × 0.01 = 2M writes/day
    - Review write QPS: 2M / 86,400 = ~23 writes/sec

    Photo uploads:
    - 10% of reviews include photos
    - Daily photo uploads: 2M × 0.1 = 200K photos/day
    - Photo QPS: 200K / 86,400 = ~2.3 photos/sec

    Total Read QPS: ~8,100 (search + views + reviews)
    Total Write QPS: ~25 (reviews + photos + check-ins)
    Read/Write ratio: 324:1 (heavily read-dominant)
    ```

    ### Storage Estimates

    ```
    Business data:
    - Total businesses: 100M
    - Per business: 2 KB (name, address, category, hours, coordinates)
    - Total: 100M × 2 KB = 200 GB

    Reviews:
    - Daily reviews: 2M/day
    - 10 years: 2M × 365 × 10 = 7.3 billion reviews
    - Per review: 1 KB (text, rating, timestamp, user_id)
    - Total: 7.3B × 1 KB = 7.3 TB

    Photos:
    - Daily photos: 200K/day
    - 10 years: 200K × 365 × 10 = 730M photos
    - Per photo: 2 MB (compressed)
    - Total: 730M × 2 MB = 1.46 PB

    Geospatial index:
    - Quadtree nodes: ~10M nodes (for global coverage)
    - Per node: 100 bytes (boundaries, business IDs)
    - Total: 10M × 100 bytes = 1 GB

    User data:
    - 200M users × 5 KB = 1 TB

    Total: 200 GB (business) + 7.3 TB (reviews) + 1.46 PB (photos) + 1 GB (index) + 1 TB (users) ≈ 1.47 PB
    ```

    ### Bandwidth Estimates

    ```
    Search ingress (query + location):
    - 5,800 searches/sec × 500 bytes = 2.9 MB/sec ≈ 23 Mbps

    Photo uploads:
    - 2.3 photos/sec × 2 MB = 4.6 MB/sec ≈ 37 Mbps

    Search egress (results):
    - 5,800 searches/sec × 20 results × 2 KB = 232 MB/sec ≈ 1.86 Gbps

    Photo downloads (thumbnails + full):
    - 50% of views see photos
    - 100M photo views/day = 1,157 photos/sec × 200 KB (avg) = 231 MB/sec ≈ 1.85 Gbps

    Total ingress: ~60 Mbps
    Total egress: ~3.71 Gbps (CDN critical for photos)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot businesses (frequently searched):
    - 20% of 100M businesses = 20M
    - 20M × 2 KB = 40 GB

    Search results cache:
    - Common queries: 10M unique (lat, lng, radius) combinations
    - 10M × 20 results × 2 KB = 400 GB

    Geospatial index (in-memory):
    - Quadtree structure: 1 GB
    - Redis Geo index: 5 GB

    Review aggregates (avg rating, count):
    - 100M businesses × 100 bytes = 10 GB

    User sessions:
    - 5M concurrent users × 10 KB = 50 GB

    Total cache: 40 GB + 400 GB + 6 GB + 10 GB + 50 GB ≈ 506 GB
    ```

    ---

    ## Key Assumptions

    1. Read-heavy system (324:1 read/write ratio)
    2. Search radius typically 1-10 km (most queries are local)
    3. Location accuracy within 50 meters acceptable
    4. 80% of searches are for top 20% of businesses (power law)
    5. Business data changes infrequently (hours, photos, etc.)
    6. Reviews can have eventual consistency (1-5 minutes acceptable)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Geospatial-first:** Quadtree/Geohash for efficient proximity search
    2. **Read optimization:** Aggressive caching for hot businesses and locations
    3. **Relevance ranking:** Combine distance, rating, popularity
    4. **Eventual consistency:** Reviews can be slightly delayed
    5. **CDN-heavy:** Photos and static content served from edge locations

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App]
            Web[Web Browser]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>Photos & static content]
            LB[Load Balancer<br/>Geographic routing]
        end

        subgraph "API Layer"
            Search_API[Search Service<br/>Nearby places]
            Business_API[Business Service<br/>Details, hours]
            Review_API[Review Service<br/>Read/write reviews]
            Photo_API[Photo Service<br/>Upload/retrieve]
            Rec_API[Recommendation Service<br/>Personalized suggestions]
        end

        subgraph "Data Processing"
            Geo_Indexer[Geo Indexer<br/>Build quadtree/geohash]
            Rating_Aggregator[Rating Aggregator<br/>Update avg ratings]
            ML_Ranker[ML Ranking<br/>Personalized results]
            Search_Indexer[Search Indexer<br/>Elasticsearch]
        end

        subgraph "Caching Layer"
            Redis_Geo[Redis Geo<br/>Geospatial queries]
            Redis_Business[Redis<br/>Business cache]
            Redis_Search[Redis<br/>Search results]
            Redis_Session[Redis<br/>User sessions]
        end

        subgraph "Storage Layer"
            Business_DB[(Business DB<br/>PostgreSQL<br/>Sharded)]
            Review_DB[(Review DB<br/>Cassandra<br/>Time-series)]
            User_DB[(User DB<br/>PostgreSQL)]
            Geo_DB[(Geospatial DB<br/>PostGIS)]
            Search_DB[(Elasticsearch<br/>Full-text search)]
            S3[Object Storage<br/>S3<br/>Photos/media]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        Mobile --> CDN
        Web --> CDN
        Mobile --> LB
        Web --> LB

        CDN --> S3

        LB --> Search_API
        LB --> Business_API
        LB --> Review_API
        LB --> Photo_API
        LB --> Rec_API

        Search_API --> Redis_Geo
        Search_API --> Redis_Search
        Search_API --> Geo_DB
        Search_API --> Search_DB

        Business_API --> Redis_Business
        Business_API --> Business_DB

        Review_API --> Review_DB
        Review_API --> Kafka

        Photo_API --> S3
        Photo_API --> Kafka

        Kafka --> Rating_Aggregator
        Kafka --> Search_Indexer
        Kafka --> Geo_Indexer
        Kafka --> ML_Ranker

        Rating_Aggregator --> Redis_Business
        Rating_Aggregator --> Business_DB

        Geo_Indexer --> Geo_DB
        Geo_Indexer --> Redis_Geo

        Search_Indexer --> Search_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Geo fill:#fff4e1
        style Redis_Business fill:#fff4e1
        style Redis_Search fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Business_DB fill:#ffe1e1
        style Review_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Geo_DB fill:#e1f5e1
        style Search_DB fill:#e8eaf6
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Redis Geo** | O(log N) geospatial queries, GEORADIUS command optimized | QuadTree in app (no distribution), Database queries (too slow) |
    | **PostGIS** | Advanced spatial operations, polygon queries, accurate calculations | MySQL spatial (limited features), NoSQL (no spatial indexes) |
    | **Cassandra (Reviews)** | Time-series optimized, handles 25 write QPS with spikes | MySQL (write bottleneck), MongoDB (consistency issues) |
    | **PostgreSQL (Business)** | ACID for business data, complex joins, sharding support | NoSQL (need transactions), Single DB (doesn't scale) |
    | **Elasticsearch** | Full-text search for business names, fuzzy matching | Database LIKE (too slow), Custom search (complex) |
    | **CDN** | 1.85 Gbps photo egress, low latency globally | Origin servers (expensive), Regional caching only (higher latency) |

    **Key Trade-off:** We chose **Redis Geo + PostGIS hybrid**. Redis for hot queries (< 10ms), PostGIS for complex spatial operations.

    ---

    ## API Design

    ### 1. Search Nearby Places

    **Request:**
    ```http
    GET /api/v1/search/nearby?lat=37.7749&lng=-122.4194&radius=5000&category=restaurants&filters=rating:4.0,price:2,open_now:true&limit=20
    Authorization: Bearer <token>
    ```

    **Query Parameters:**
    - `lat`, `lng`: User location (required)
    - `radius`: Search radius in meters (default: 5000m)
    - `category`: Business category (restaurants, hotels, etc.)
    - `filters`: rating, price (1-4), open_now, accepts_credit_cards
    - `sort`: distance, rating, review_count (default: relevance)
    - `limit`: Results per page (default: 20, max: 50)

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "results": [
        {
          "business_id": "biz_123",
          "name": "Tartine Bakery",
          "category": "Bakery",
          "rating": 4.5,
          "review_count": 3842,
          "price_level": 2,
          "location": {
            "address": "600 Guerrero St, San Francisco, CA 94110",
            "coordinates": {
              "lat": 37.7614,
              "lng": -122.4238
            }
          },
          "distance": 1247,  // meters
          "is_open": true,
          "photo_url": "https://cdn.yelp.com/biz/biz_123_thumb.jpg"
        },
        // ... 19 more results
      ],
      "metadata": {
        "total_results": 156,
        "search_radius": 5000,
        "center": {
          "lat": 37.7749,
          "lng": -122.4194
        },
        "query_time_ms": 87
      },
      "next_cursor": "eyJvZmZzZXQiOjIwfQ=="
    }
    ```

    **Design Notes:**

    - Return results sorted by relevance (distance + rating + popularity)
    - Include distance in meters from search center
    - Cursor-based pagination for consistency
    - Cache results for 5 minutes (location + filters as cache key)

    ---

    ### 2. Get Business Details

    **Request:**
    ```http
    GET /api/v1/businesses/biz_123
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "business_id": "biz_123",
      "name": "Tartine Bakery",
      "category": "Bakery",
      "categories": ["Bakery", "Cafe", "French"],
      "rating": 4.5,
      "review_count": 3842,
      "price_level": 2,
      "location": {
        "address": "600 Guerrero St, San Francisco, CA 94110",
        "coordinates": {
          "lat": 37.7614,
          "lng": -122.4238
        },
        "neighborhood": "Mission District"
      },
      "contact": {
        "phone": "+1-415-487-2600",
        "website": "https://www.tartinebakery.com"
      },
      "hours": {
        "monday": {"open": "08:00", "close": "19:00"},
        "tuesday": {"open": "08:00", "close": "19:00"},
        // ... other days
        "sunday": {"open": "09:00", "close": "18:00"}
      },
      "attributes": {
        "accepts_credit_cards": true,
        "outdoor_seating": true,
        "wheelchair_accessible": true,
        "wifi": true
      },
      "photos": [
        "https://cdn.yelp.com/photos/photo_1.jpg",
        "https://cdn.yelp.com/photos/photo_2.jpg"
      ],
      "is_open_now": true
    }
    ```

    **Design Notes:**

    - Cache business details for 1 hour (business data changes infrequently)
    - Calculate `is_open_now` based on current time and business hours
    - Serve photos from CDN

    ---

    ### 3. Submit Review

    **Request:**
    ```http
    POST /api/v1/reviews
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "business_id": "biz_123",
      "rating": 5,
      "text": "Best croissants in SF! The morning bun is a must-try.",
      "photo_ids": ["photo_456", "photo_789"]  // Optional
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "review_id": "rev_987",
      "business_id": "biz_123",
      "user": {
        "user_id": "user_123",
        "name": "John D.",
        "profile_pic": "https://cdn.yelp.com/users/user_123.jpg",
        "review_count": 24
      },
      "rating": 5,
      "text": "Best croissants in SF! The morning bun is a must-try.",
      "photos": [
        "https://cdn.yelp.com/photos/photo_456.jpg",
        "https://cdn.yelp.com/photos/photo_789.jpg"
      ],
      "created_at": "2026-02-02T10:30:00Z",
      "useful_count": 0,
      "funny_count": 0,
      "cool_count": 0
    }
    ```

    **Design Notes:**

    - Rate limit: 5 reviews per day per user
    - Trigger async rating aggregation (update business avg rating)
    - Spam detection for suspicious reviews
    - One review per user per business (can update existing)

    ---

    ## Database Schema

    ### Businesses (PostgreSQL - Sharded by business_id)

    ```sql
    -- Business table
    CREATE TABLE businesses (
        business_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        category VARCHAR(100) NOT NULL,
        categories TEXT[],  -- Array of categories
        rating DECIMAL(2,1) DEFAULT 0.0,
        review_count INT DEFAULT 0,
        price_level INT CHECK (price_level BETWEEN 1 AND 4),
        address TEXT NOT NULL,
        city VARCHAR(100),
        state VARCHAR(50),
        country VARCHAR(50),
        postal_code VARCHAR(20),
        latitude DECIMAL(10, 7) NOT NULL,
        longitude DECIMAL(10, 7) NOT NULL,
        phone VARCHAR(20),
        website TEXT,
        is_open BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_category (category),
        INDEX idx_rating (rating),
        INDEX idx_location (latitude, longitude)
    ) PARTITION BY HASH (business_id);

    -- Business hours
    CREATE TABLE business_hours (
        business_id VARCHAR(50) REFERENCES businesses(business_id),
        day_of_week INT CHECK (day_of_week BETWEEN 0 AND 6),  -- 0=Monday
        open_time TIME,
        close_time TIME,
        PRIMARY KEY (business_id, day_of_week)
    );

    -- Business attributes
    CREATE TABLE business_attributes (
        business_id VARCHAR(50) REFERENCES businesses(business_id),
        attribute_key VARCHAR(100),
        attribute_value BOOLEAN,
        PRIMARY KEY (business_id, attribute_key)
    );
    ```

    **Why PostgreSQL:**

    - **ACID transactions:** Business data requires consistency
    - **Rich queries:** Complex filters (rating, price, attributes)
    - **Proven at scale:** Sharding handles 100M businesses
    - **PostGIS extension:** Advanced spatial queries

    ---

    ### Reviews (Cassandra - Time-series)

    ```sql
    -- Reviews table (partition by business_id, cluster by created_at)
    CREATE TABLE reviews (
        review_id UUID PRIMARY KEY,
        business_id VARCHAR(50),
        user_id VARCHAR(50),
        rating INT,
        text TEXT,
        photo_urls LIST<TEXT>,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        useful_count COUNTER,
        funny_count COUNTER,
        cool_count COUNTER
    );

    -- Reviews by business (for business detail page)
    CREATE TABLE reviews_by_business (
        business_id VARCHAR(50),
        created_at TIMESTAMP,
        review_id UUID,
        user_id VARCHAR(50),
        rating INT,
        PRIMARY KEY (business_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Reviews by user (for user profile)
    CREATE TABLE reviews_by_user (
        user_id VARCHAR(50),
        created_at TIMESTAMP,
        review_id UUID,
        business_id VARCHAR(50),
        rating INT,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Rating aggregates (for fast queries)
    CREATE TABLE business_ratings (
        business_id VARCHAR(50) PRIMARY KEY,
        total_reviews INT,
        total_rating_sum INT,
        avg_rating DECIMAL(2,1),
        rating_5_count INT,
        rating_4_count INT,
        rating_3_count INT,
        rating_2_count INT,
        rating_1_count INT,
        updated_at TIMESTAMP
    );
    ```

    **Why Cassandra:**

    - **Write-optimized:** Handles review spikes
    - **Time-series:** Reviews naturally ordered by time
    - **Denormalized:** Multiple views (by business, by user)
    - **Scalable:** Linear scaling for 7B+ reviews

    ---

    ### Geospatial (PostGIS)

    ```sql
    -- Geospatial index (PostGIS extension)
    CREATE TABLE business_locations (
        business_id VARCHAR(50) PRIMARY KEY,
        location GEOGRAPHY(POINT, 4326),  -- WGS 84 coordinate system
        geohash VARCHAR(12),  -- Geohash for fast filtering
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Spatial index for fast radius queries
    CREATE INDEX idx_business_location ON business_locations USING GIST(location);
    CREATE INDEX idx_geohash ON business_locations(geohash);

    -- Query nearby businesses (within 5km radius)
    -- SELECT business_id
    -- FROM business_locations
    -- WHERE ST_DWithin(
    --     location,
    --     ST_MakePoint(-122.4194, 37.7749)::geography,
    --     5000  -- 5000 meters
    -- );
    ```

    ---

    ## Data Flow Diagrams

    ### Nearby Search Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Search_API
        participant Redis_Geo
        participant PostGIS
        participant Business_DB
        participant Redis_Cache

        Client->>Search_API: GET /search/nearby?lat=37.7749&lng=-122.4194&radius=5000
        Search_API->>Search_API: Parse params, validate location

        Search_API->>Redis_Cache: GET search:{lat,lng,radius,filters}
        alt Cache HIT (60% of requests)
            Redis_Cache-->>Search_API: Cached results
        else Cache MISS (40% of requests)
            Redis_Cache-->>Search_API: null

            Search_API->>Redis_Geo: GEORADIUS 37.7749 -122.4194 5 km
            Redis_Geo-->>Search_API: List of business_ids (hot businesses)

            alt Hot location (80% of queries)
                Search_API->>Business_DB: Batch get business details
                Business_DB-->>Search_API: Business objects
            else Cold location (20% of queries)
                Search_API->>PostGIS: ST_DWithin spatial query
                PostGIS-->>Search_API: business_ids within radius
                Search_API->>Business_DB: Batch get business details
                Business_DB-->>Search_API: Business objects
            end

            Search_API->>Search_API: Apply filters (rating, price, open_now)
            Search_API->>Search_API: Calculate relevance scores
            Search_API->>Search_API: Sort by score, limit to 20

            Search_API->>Redis_Cache: SET search:{key} (TTL: 5min)
        end

        Search_API-->>Client: 200 OK (20 results, 150ms)
    ```

    **Flow Explanation:**

    1. **Check cache** - Search results cached for 5 minutes (60% hit rate)
    2. **Geo query** - Redis Geo for hot locations, PostGIS for cold
    3. **Fetch details** - Batch query business data
    4. **Filter & rank** - Apply filters, calculate relevance scores
    5. **Cache result** - Store for subsequent requests

    ---

    ### Review Submission Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Review_API
        participant Review_DB
        participant Kafka
        participant Rating_Aggregator
        participant Business_DB
        participant Redis_Cache

        Client->>Review_API: POST /reviews (rating, text, business_id)
        Review_API->>Review_API: Validate, spam detection

        Review_API->>Review_DB: INSERT review
        Review_DB-->>Review_API: review_id

        Review_API->>Kafka: Publish review_created event
        Review_API-->>Client: 201 Created (review_id)

        Kafka->>Rating_Aggregator: Process review_created
        Rating_Aggregator->>Business_DB: UPDATE avg_rating, review_count

        Rating_Aggregator->>Redis_Cache: Invalidate business:{business_id}
        Rating_Aggregator->>Redis_Cache: Invalidate search results containing business

        Note over Rating_Aggregator: Rating updates eventually consistent (1-5 min)
    ```

    **Flow Explanation:**

    1. **Validate review** - Spam detection, rate limiting
    2. **Store review** - Cassandra write (< 50ms)
    3. **Async processing** - Kafka event for rating update
    4. **Update aggregates** - Recalculate avg rating
    5. **Cache invalidation** - Clear stale business data

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical geospatial search subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Geospatial Indexing** | How to find nearby businesses in <100ms? | Quadtree + Redis Geo + Geohash |
    | **Ranking Algorithm** | How to rank results by relevance? | Distance + Rating + Popularity + ML |
    | **Real-time Updates** | How to keep business data fresh? | Event-driven updates + cache invalidation |
    | **Scalability** | How to handle 500M searches/day? | Sharding + caching + CDN |

    ---

    === "🗺️ Geospatial Indexing"

        ## The Challenge

        **Problem:** Find all businesses within 5km radius from user location. Naive approach: calculate distance to all 100M businesses. **Too slow!**

        **Requirements:**

        - **Fast:** < 100ms for radius queries
        - **Accurate:** Within 50m precision
        - **Scalable:** Handle 100M businesses
        - **Dynamic:** Support adding/removing businesses

        ---

        ## Solution 1: Quadtree

        **Concept:** Recursively divide 2D space into 4 quadrants until each leaf contains ≤ N businesses.

        **Structure:**

        ```
        Root (World)
        ├── NW Quadrant
        │   ├── NW-NW (Empty)
        │   ├── NW-NE (50 businesses)
        │   ├── NW-SW (120 businesses - subdivide further)
        │   └── NW-SE (30 businesses)
        └── NE, SW, SE Quadrants...
        ```

        **Time Complexity:**

        - **Build:** O(N log N) - one-time cost
        - **Search:** O(log N + K) where K = results in radius
        - **Insert:** O(log N)
        - **Delete:** O(log N)

        ---

        ## Quadtree Implementation

        ```python
        from typing import List, Tuple
        import math

        class Point:
            """Represents a business location"""
            def __init__(self, lat: float, lng: float, business_id: str):
                self.lat = lat
                self.lng = lng
                self.business_id = business_id

        class Boundary:
            """Represents a rectangular region"""
            def __init__(self, center_lat: float, center_lng: float, half_width: float, half_height: float):
                self.center_lat = center_lat
                self.center_lng = center_lng
                self.half_width = half_width
                self.half_height = half_height

            def contains(self, point: Point) -> bool:
                """Check if point is within boundary"""
                return (abs(point.lat - self.center_lat) <= self.half_width and
                        abs(point.lng - self.center_lng) <= self.half_height)

            def intersects_circle(self, center_lat: float, center_lng: float, radius_km: float) -> bool:
                """Check if boundary intersects with search circle"""
                # Approximate: 1 degree ≈ 111 km
                radius_deg = radius_km / 111.0

                # Find closest point in rectangle to circle center
                closest_lat = max(self.center_lat - self.half_width,
                                min(center_lat, self.center_lat + self.half_width))
                closest_lng = max(self.center_lng - self.half_height,
                                min(center_lng, self.center_lng + self.half_height))

                # Calculate distance
                dist = self._haversine_distance(center_lat, center_lng, closest_lat, closest_lng)
                return dist <= radius_km

            def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """Calculate distance between two points in km"""
                R = 6371  # Earth radius in km

                lat1_rad = math.radians(lat1)
                lat2_rad = math.radians(lat2)
                delta_lat = math.radians(lat2 - lat1)
                delta_lng = math.radians(lng2 - lng1)

                a = (math.sin(delta_lat / 2) ** 2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) *
                     math.sin(delta_lng / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                return R * c

        class QuadTree:
            """
            Quadtree for efficient geospatial search

            Args:
                boundary: Region covered by this node
                capacity: Max points per node before subdividing
            """
            def __init__(self, boundary: Boundary, capacity: int = 50):
                self.boundary = boundary
                self.capacity = capacity
                self.points: List[Point] = []
                self.divided = False

                # Children (created on subdivision)
                self.northwest = None
                self.northeast = None
                self.southwest = None
                self.southeast = None

            def insert(self, point: Point) -> bool:
                """
                Insert business location into quadtree

                Returns:
                    True if inserted successfully, False otherwise
                """
                # Point outside boundary
                if not self.boundary.contains(point):
                    return False

                # Room available, add point
                if len(self.points) < self.capacity:
                    self.points.append(point)
                    return True

                # Need to subdivide
                if not self.divided:
                    self.subdivide()

                # Try inserting into children
                if self.northwest.insert(point):
                    return True
                if self.northeast.insert(point):
                    return True
                if self.southwest.insert(point):
                    return True
                if self.southeast.insert(point):
                    return True

                return False

            def subdivide(self):
                """Split node into 4 quadrants"""
                x = self.boundary.center_lat
                y = self.boundary.center_lng
                w = self.boundary.half_width / 2
                h = self.boundary.half_height / 2

                # Create 4 children
                nw_boundary = Boundary(x - w, y + h, w, h)
                ne_boundary = Boundary(x + w, y + h, w, h)
                sw_boundary = Boundary(x - w, y - h, w, h)
                se_boundary = Boundary(x + w, y - h, w, h)

                self.northwest = QuadTree(nw_boundary, self.capacity)
                self.northeast = QuadTree(ne_boundary, self.capacity)
                self.southwest = QuadTree(sw_boundary, self.capacity)
                self.southeast = QuadTree(se_boundary, self.capacity)

                self.divided = True

                # Move existing points to children
                for point in self.points:
                    self.northwest.insert(point) or \
                    self.northeast.insert(point) or \
                    self.southwest.insert(point) or \
                    self.southeast.insert(point)

                # Clear points from parent
                self.points = []

            def query_radius(self, center_lat: float, center_lng: float, radius_km: float) -> List[Point]:
                """
                Find all businesses within radius of center point

                Args:
                    center_lat: Search center latitude
                    center_lng: Search center longitude
                    radius_km: Search radius in kilometers

                Returns:
                    List of Point objects within radius
                """
                found = []

                # Boundary doesn't intersect search circle
                if not self.boundary.intersects_circle(center_lat, center_lng, radius_km):
                    return found

                # Check points in this node
                for point in self.points:
                    dist = self.boundary._haversine_distance(center_lat, center_lng, point.lat, point.lng)
                    if dist <= radius_km:
                        found.append(point)

                # Recursively search children
                if self.divided:
                    found.extend(self.northwest.query_radius(center_lat, center_lng, radius_km))
                    found.extend(self.northeast.query_radius(center_lat, center_lng, radius_km))
                    found.extend(self.southwest.query_radius(center_lat, center_lng, radius_km))
                    found.extend(self.southeast.query_radius(center_lat, center_lng, radius_km))

                return found

        # Usage example
        def build_quadtree(businesses: List[dict]) -> QuadTree:
            """
            Build quadtree from business data

            Args:
                businesses: List of {business_id, lat, lng}

            Returns:
                Populated QuadTree
            """
            # World boundary (entire globe)
            world_boundary = Boundary(
                center_lat=0.0,
                center_lng=0.0,
                half_width=90.0,  # -90 to 90 degrees
                half_height=180.0  # -180 to 180 degrees
            )

            quadtree = QuadTree(world_boundary, capacity=50)

            # Insert all businesses
            for biz in businesses:
                point = Point(biz['lat'], biz['lng'], biz['business_id'])
                quadtree.insert(point)

            return quadtree

        # Search example
        def search_nearby(quadtree: QuadTree, user_lat: float, user_lng: float, radius_km: float = 5.0):
            """
            Search for businesses near user

            Returns:
                List of business_ids within radius
            """
            points = quadtree.query_radius(user_lat, user_lng, radius_km)
            return [p.business_id for p in points]
        ```

        **Quadtree Trade-offs:**

        | Pros | Cons |
        |------|------|
        | Fast search: O(log N + K) | Rebuild required for bulk updates |
        | Memory efficient (only stores leaf data) | Unbalanced for uneven distributions (NYC vs rural) |
        | Supports dynamic updates | Complex implementation |
        | Accurate distance calculations | Not suitable for distributed systems |

        ---

        ## Solution 2: Geohash

        **Concept:** Encode lat/lng into a base-32 string. Nearby locations share common prefixes.

        **Example:**

        ```
        Location: (37.7749, -122.4194) [San Francisco]
        Geohash: "9q8yy" (5 characters, ~4.9km × 4.9km precision)

        Nearby locations:
        - 9q8yy (same grid cell)
        - 9q8yv, 9q8yw, 9q8yx, 9q8yz (adjacent cells)
        ```

        **Precision levels:**

        | Length | Lat precision | Lng precision | Cell size |
        |--------|--------------|--------------|-----------|
        | 1 | ±23° | ±23° | 5,000 km × 5,000 km |
        | 3 | ±0.17° | ±0.17° | 20 km × 20 km |
        | 5 | ±0.021° | ±0.021° | 2.4 km × 2.4 km |
        | 7 | ±0.0026° | ±0.0026° | 153 m × 153 m |
        | 9 | ±0.00033° | ±0.00033° | 19 m × 19 m |

        ---

        ## Geohash Implementation

        ```python
        import base64

        class Geohash:
            """Geohash encoding/decoding for geospatial indexing"""

            BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

            @staticmethod
            def encode(lat: float, lng: float, precision: int = 9) -> str:
                """
                Encode latitude/longitude to geohash

                Args:
                    lat: Latitude (-90 to 90)
                    lng: Longitude (-180 to 180)
                    precision: Geohash length (1-12)

                Returns:
                    Geohash string
                """
                lat_range = [-90.0, 90.0]
                lng_range = [-180.0, 180.0]
                geohash = []
                bits = 0
                bit_count = 0

                while len(geohash) < precision:
                    # Alternate between longitude and latitude bits
                    if bit_count % 2 == 0:  # Even bits for longitude
                        mid = (lng_range[0] + lng_range[1]) / 2
                        if lng > mid:
                            bits |= (1 << (4 - bit_count // 2))
                            lng_range[0] = mid
                        else:
                            lng_range[1] = mid
                    else:  # Odd bits for latitude
                        mid = (lat_range[0] + lat_range[1]) / 2
                        if lat > mid:
                            bits |= (1 << (4 - bit_count // 2))
                            lat_range[0] = mid
                        else:
                            lat_range[1] = mid

                    bit_count += 1

                    # Every 5 bits, encode to base32 character
                    if bit_count == 5:
                        geohash.append(Geohash.BASE32[bits])
                        bits = 0
                        bit_count = 0

                return ''.join(geohash)

            @staticmethod
            def decode(geohash: str) -> Tuple[float, float]:
                """
                Decode geohash to latitude/longitude

                Args:
                    geohash: Geohash string

                Returns:
                    Tuple of (latitude, longitude)
                """
                lat_range = [-90.0, 90.0]
                lng_range = [-180.0, 180.0]

                is_lng = True
                for char in geohash:
                    idx = Geohash.BASE32.index(char)

                    # Process 5 bits
                    for i in range(4, -1, -1):
                        bit = (idx >> i) & 1

                        if is_lng:
                            mid = (lng_range[0] + lng_range[1]) / 2
                            if bit == 1:
                                lng_range[0] = mid
                            else:
                                lng_range[1] = mid
                        else:
                            mid = (lat_range[0] + lat_range[1]) / 2
                            if bit == 1:
                                lat_range[0] = mid
                            else:
                                lat_range[1] = mid

                        is_lng = not is_lng

                lat = (lat_range[0] + lat_range[1]) / 2
                lng = (lng_range[0] + lng_range[1]) / 2

                return (lat, lng)

            @staticmethod
            def neighbors(geohash: str) -> List[str]:
                """
                Get 8 neighboring geohash cells

                Returns:
                    List of 8 neighbor geohashes (N, NE, E, SE, S, SW, W, NW)
                """
                # Simplified: decode, calculate neighbors, encode
                # Production: Use lookup tables for efficiency
                lat, lng = Geohash.decode(geohash)

                # Approximate cell size at this precision
                precision = len(geohash)
                cell_sizes = {
                    5: 0.024,  # ~2.4 km
                    7: 0.003,  # ~300 m
                    9: 0.0004  # ~40 m
                }
                offset = cell_sizes.get(precision, 0.024)

                neighbors = []
                for dlat in [-offset, 0, offset]:
                    for dlng in [-offset, 0, offset]:
                        if dlat == 0 and dlng == 0:
                            continue  # Skip center
                        neighbor_hash = Geohash.encode(lat + dlat, lng + dlng, precision)
                        neighbors.append(neighbor_hash)

                return neighbors

        # Usage with database
        class GeohashSearchService:
            """Search businesses using geohash indexing"""

            def __init__(self, db):
                self.db = db

            def search_nearby(self, lat: float, lng: float, radius_km: float = 5.0) -> List[str]:
                """
                Search businesses near location using geohash

                Args:
                    lat: Search center latitude
                    lng: Search center longitude
                    radius_km: Search radius

                Returns:
                    List of business_ids
                """
                # Determine geohash precision based on radius
                if radius_km <= 0.5:
                    precision = 9  # ~20m cells
                elif radius_km <= 2:
                    precision = 7  # ~150m cells
                else:
                    precision = 5  # ~2.4km cells

                # Get geohash for center point
                center_geohash = Geohash.encode(lat, lng, precision)

                # Get neighboring cells (to handle edge cases)
                geohashes_to_search = [center_geohash] + Geohash.neighbors(center_geohash)

                # Query database for businesses in these geohash cells
                query = """
                    SELECT business_id, latitude, longitude
                    FROM business_locations
                    WHERE geohash LIKE ANY(%s)
                """
                patterns = [f"{gh}%" for gh in geohashes_to_search]
                results = self.db.execute(query, (patterns,))

                # Filter by exact distance (geohash is approximate)
                nearby_businesses = []
                for biz in results:
                    dist = self._calculate_distance(lat, lng, biz['latitude'], biz['longitude'])
                    if dist <= radius_km:
                        nearby_businesses.append({
                            'business_id': biz['business_id'],
                            'distance': dist
                        })

                # Sort by distance
                nearby_businesses.sort(key=lambda x: x['distance'])

                return [b['business_id'] for b in nearby_businesses]

            def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """Haversine distance in kilometers"""
                R = 6371  # Earth radius in km
                lat1_rad = math.radians(lat1)
                lat2_rad = math.radians(lat2)
                delta_lat = math.radians(lat2 - lat1)
                delta_lng = math.radians(lng2 - lng1)

                a = (math.sin(delta_lat / 2) ** 2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) *
                     math.sin(delta_lng / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                return R * c
        ```

        **Geohash Trade-offs:**

        | Pros | Cons |
        |------|------|
        | Simple implementation | Edge case issues (cell boundaries) |
        | Works with standard databases (LIKE query) | Less accurate than quadtree |
        | Easy to shard (prefix-based) | Requires neighbor cell searches |
        | Human-readable | Distance still needs calculation |

        ---

        ## Solution 3: Redis Geo (Production)

        **Best approach:** Use Redis GEORADIUS for hot locations, PostGIS for cold/complex queries.

        ```python
        import redis
        from typing import List, Dict

        class RedisGeoService:
            """
            Production geospatial search using Redis Geo

            Redis Geo uses geohash internally but provides optimized commands
            """

            def __init__(self, redis_client):
                self.redis = redis_client
                self.geo_key = "businesses:locations"

            def add_business(self, business_id: str, lat: float, lng: float):
                """
                Add business location to Redis Geo index

                Args:
                    business_id: Unique business identifier
                    lat: Latitude
                    lng: Longitude
                """
                self.redis.geoadd(self.geo_key, lng, lat, business_id)

            def bulk_add_businesses(self, businesses: List[Dict]):
                """
                Bulk add businesses for efficiency

                Args:
                    businesses: List of {business_id, lat, lng}
                """
                # GEOADD supports multiple entries
                items = []
                for biz in businesses:
                    items.extend([biz['lng'], biz['lat'], biz['business_id']])

                self.redis.geoadd(self.geo_key, *items)

            def search_radius(self, lat: float, lng: float, radius_km: float = 5.0, limit: int = 100) -> List[Dict]:
                """
                Search businesses within radius (GEORADIUS)

                Args:
                    lat: Search center latitude
                    lng: Search center longitude
                    radius_km: Search radius in kilometers
                    limit: Max results

                Returns:
                    List of {business_id, distance, lat, lng}
                """
                # GEORADIUS: O(N+log(M)) where N=results, M=total items
                results = self.redis.georadius(
                    self.geo_key,
                    lng,
                    lat,
                    radius_km,
                    unit='km',
                    withdist=True,  # Include distance
                    withcoord=True,  # Include coordinates
                    sort='ASC',  # Nearest first
                    count=limit
                )

                # Parse results
                businesses = []
                for business_id, distance, coords in results:
                    businesses.append({
                        'business_id': business_id.decode('utf-8'),
                        'distance': float(distance),
                        'lng': coords[0],
                        'lat': coords[1]
                    })

                return businesses

            def search_member_radius(self, business_id: str, radius_km: float = 10.0) -> List[str]:
                """
                Find businesses near another business (GEORADIUSBYMEMBER)

                Args:
                    business_id: Reference business
                    radius_km: Search radius

                Returns:
                    List of nearby business_ids
                """
                results = self.redis.georadiusbymember(
                    self.geo_key,
                    business_id,
                    radius_km,
                    unit='km',
                    withdist=True,
                    sort='ASC'
                )

                return [biz_id.decode('utf-8') for biz_id, _ in results if biz_id.decode('utf-8') != business_id]

            def get_distance(self, business_id1: str, business_id2: str) -> float:
                """
                Calculate distance between two businesses

                Returns:
                    Distance in kilometers
                """
                distance = self.redis.geodist(self.geo_key, business_id1, business_id2, unit='km')
                return float(distance) if distance else None

            def remove_business(self, business_id: str):
                """Remove business from geo index"""
                self.redis.zrem(self.geo_key, business_id)

        # Hybrid approach: Redis + PostGIS
        class HybridGeoService:
            """
            Combine Redis (fast, hot data) with PostGIS (accurate, cold data)
            """

            def __init__(self, redis_geo: RedisGeoService, postgis_db):
                self.redis_geo = redis_geo
                self.postgis = postgis_db

            def search_nearby(self, lat: float, lng: float, radius_km: float = 5.0) -> List[str]:
                """
                Hybrid search: try Redis first, fallback to PostGIS

                Redis: Hot locations (80% of queries)
                PostGIS: Cold locations, complex queries (20%)
                """
                # Try Redis Geo first (< 10ms)
                redis_results = self.redis_geo.search_radius(lat, lng, radius_km, limit=100)

                if len(redis_results) >= 20:
                    # Sufficient results from Redis
                    return [b['business_id'] for b in redis_results]

                # Fallback to PostGIS (50-100ms)
                query = """
                    SELECT business_id,
                           ST_Distance(location, ST_MakePoint(%s, %s)::geography) as distance
                    FROM business_locations
                    WHERE ST_DWithin(
                        location,
                        ST_MakePoint(%s, %s)::geography,
                        %s
                    )
                    ORDER BY distance
                    LIMIT 100
                """
                results = self.postgis.execute(query, (lng, lat, lng, lat, radius_km * 1000))

                return [row['business_id'] for row in results]
        ```

        **Redis Geo Advantages:**

        1. **Performance:** < 10ms for 100k+ locations
        2. **Built-in:** No external dependencies
        3. **Sorted results:** Automatic distance sorting
        4. **Simple API:** One command for radius queries
        5. **Production-proven:** Used by Uber, Foursquare

    === "🎯 Ranking Algorithm"

        ## The Challenge

        **Problem:** 200 businesses within 5km radius. How to rank them?

        **User expectations:**

        - Nearby businesses ranked higher
        - High-rated businesses preferred
        - Popular places (more reviews) favored
        - Open businesses prioritized
        - Personalized based on history

        ---

        ## Ranking Factors

        | Factor | Weight | Formula | Rationale |
        |--------|--------|---------|-----------|
        | **Distance** | 35% | `1 / (1 + distance_km)` | Closer is better |
        | **Rating** | 30% | `rating / 5.0` | Quality matters |
        | **Popularity** | 20% | `log(1 + review_count) / log(10000)` | More reviews = more reliable |
        | **Recency** | 10% | `1 - (days_inactive / 365)` | Recently reviewed businesses |
        | **Personalization** | 5% | ML model score | User preferences |

        ---

        ## Relevance Score Implementation

        ```python
        import math
        from typing import List, Dict
        from datetime import datetime, timedelta

        class RankingService:
            """
            Business ranking based on multiple factors

            Combines distance, rating, popularity, recency
            """

            # Weights for different factors
            WEIGHT_DISTANCE = 0.35
            WEIGHT_RATING = 0.30
            WEIGHT_POPULARITY = 0.20
            WEIGHT_RECENCY = 0.10
            WEIGHT_PERSONALIZATION = 0.05

            # Normalization constants
            MAX_REVIEW_COUNT = 10000  # Reviews for max popularity score
            MAX_DISTANCE_KM = 10  # Distance for min distance score

            def __init__(self, ml_ranker=None):
                self.ml_ranker = ml_ranker

            def rank_businesses(
                self,
                businesses: List[Dict],
                user_lat: float,
                user_lng: float,
                user_id: str = None,
                filters: Dict = None
            ) -> List[Dict]:
                """
                Rank businesses by relevance score

                Args:
                    businesses: List of business objects with lat, lng, rating, review_count
                    user_lat: User latitude
                    user_lng: User longitude
                    user_id: User ID for personalization (optional)
                    filters: Additional filters (open_now, price, etc.)

                Returns:
                    Sorted list of businesses with relevance scores
                """
                scored_businesses = []

                for biz in businesses:
                    # Apply filters first
                    if filters and not self._passes_filters(biz, filters):
                        continue

                    # Calculate relevance score
                    score = self._calculate_relevance_score(biz, user_lat, user_lng, user_id)

                    biz['relevance_score'] = score
                    scored_businesses.append(biz)

                # Sort by score (descending)
                scored_businesses.sort(key=lambda x: x['relevance_score'], reverse=True)

                return scored_businesses

            def _calculate_relevance_score(
                self,
                business: Dict,
                user_lat: float,
                user_lng: float,
                user_id: str = None
            ) -> float:
                """
                Calculate overall relevance score

                Returns:
                    Score between 0 and 1 (higher is better)
                """
                # 1. Distance score (closer is better)
                distance_score = self._distance_score(
                    user_lat, user_lng,
                    business['latitude'], business['longitude']
                )

                # 2. Rating score
                rating_score = self._rating_score(business['rating'])

                # 3. Popularity score (based on review count)
                popularity_score = self._popularity_score(business['review_count'])

                # 4. Recency score (recently reviewed businesses)
                recency_score = self._recency_score(business.get('last_review_date'))

                # 5. Personalization score (ML model)
                personalization_score = self._personalization_score(business, user_id)

                # Weighted sum
                total_score = (
                    self.WEIGHT_DISTANCE * distance_score +
                    self.WEIGHT_RATING * rating_score +
                    self.WEIGHT_POPULARITY * popularity_score +
                    self.WEIGHT_RECENCY * recency_score +
                    self.WEIGHT_PERSONALIZATION * personalization_score
                )

                return total_score

            def _distance_score(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """
                Distance score: closer is better

                Returns:
                    Score between 0 and 1
                """
                distance_km = self._haversine_distance(lat1, lng1, lat2, lng2)

                # Inverse relationship: closer = higher score
                # Uses sigmoid-like curve
                score = 1.0 / (1.0 + (distance_km / 2.0))

                return min(score, 1.0)

            def _rating_score(self, rating: float) -> float:
                """
                Rating score: higher rating is better

                Returns:
                    Score between 0 and 1
                """
                if rating is None or rating == 0:
                    return 0.5  # Neutral for unrated businesses

                # Normalize to 0-1 range
                return rating / 5.0

            def _popularity_score(self, review_count: int) -> float:
                """
                Popularity score: more reviews = more trustworthy

                Uses logarithmic scale to avoid over-weighting viral businesses

                Returns:
                    Score between 0 and 1
                """
                if review_count <= 0:
                    return 0.1  # Minimum score for new businesses

                # Log scale: 1 review = 0.1, 100 reviews = 0.5, 10000 reviews = 1.0
                score = math.log(1 + review_count) / math.log(1 + self.MAX_REVIEW_COUNT)

                return min(score, 1.0)

            def _recency_score(self, last_review_date: datetime = None) -> float:
                """
                Recency score: recently reviewed businesses preferred

                Returns:
                    Score between 0 and 1
                """
                if last_review_date is None:
                    return 0.5  # Neutral for businesses without reviews

                days_since_review = (datetime.utcnow() - last_review_date).days

                # Decay over 1 year
                # Recent (0-30 days) = 1.0
                # 6 months = 0.5
                # 1 year+ = 0.1
                score = 1.0 - min(days_since_review / 365.0, 0.9)

                return max(score, 0.1)

            def _personalization_score(self, business: Dict, user_id: str = None) -> float:
                """
                Personalization score based on user preferences

                Uses ML model if available, otherwise simple heuristics

                Returns:
                    Score between 0 and 1
                """
                if user_id is None or self.ml_ranker is None:
                    return 0.5  # Neutral if no personalization

                # ML model predicts user's likelihood to engage with business
                score = self.ml_ranker.predict_score(user_id, business['business_id'])

                return score

            def _passes_filters(self, business: Dict, filters: Dict) -> bool:
                """
                Check if business passes all filters

                Filters:
                - rating: Minimum rating (e.g., 4.0+)
                - price: Price level (1-4)
                - open_now: Currently open
                - category: Business category
                """
                # Rating filter
                if 'rating' in filters:
                    if business['rating'] < filters['rating']:
                        return False

                # Price filter
                if 'price' in filters:
                    if business['price_level'] != filters['price']:
                        return False

                # Open now filter
                if filters.get('open_now', False):
                    if not self._is_open_now(business):
                        return False

                # Category filter
                if 'category' in filters:
                    if filters['category'] not in business['categories']:
                        return False

                return True

            def _is_open_now(self, business: Dict) -> bool:
                """
                Check if business is currently open

                Requires business hours data
                """
                if 'hours' not in business:
                    return True  # Assume open if hours unknown

                now = datetime.utcnow()
                day_of_week = now.weekday()  # 0=Monday, 6=Sunday
                current_time = now.time()

                # Get hours for today
                hours_today = business['hours'].get(str(day_of_week))
                if not hours_today:
                    return False  # Closed today

                open_time = hours_today['open']
                close_time = hours_today['close']

                # Handle overnight hours (e.g., 22:00 - 02:00)
                if close_time < open_time:
                    return current_time >= open_time or current_time <= close_time
                else:
                    return open_time <= current_time <= close_time

            def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
                """Calculate distance between two points in km"""
                R = 6371  # Earth radius in km

                lat1_rad = math.radians(lat1)
                lat2_rad = math.radians(lat2)
                delta_lat = math.radians(lat2 - lat1)
                delta_lng = math.radians(lng2 - lng1)

                a = (math.sin(delta_lat / 2) ** 2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) *
                     math.sin(delta_lng / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                return R * c
        ```

        ---

        ## ML-Based Ranking (Advanced)

        **Features for personalization model:**

        ```python
        class MLRankingModel:
            """
            Machine learning model for personalized business ranking

            Uses gradient boosting (XGBoost) to predict user engagement
            """

            def __init__(self):
                self.model = None  # Trained XGBoost model

            def extract_features(self, user: Dict, business: Dict, context: Dict) -> Dict:
                """
                Extract features for ML model

                Args:
                    user: User profile and history
                    business: Business details
                    context: Search context (time, location, etc.)

                Returns:
                    Feature dictionary
                """
                features = {}

                # User features
                features['user_avg_rating'] = user.get('avg_rating_given', 3.5)
                features['user_review_count'] = user.get('review_count', 0)
                features['user_days_active'] = (datetime.utcnow() - user['created_at']).days

                # Business features
                features['business_rating'] = business['rating']
                features['business_review_count'] = business['review_count']
                features['business_price_level'] = business['price_level']
                features['business_age_days'] = (datetime.utcnow() - business['created_at']).days

                # User-business interaction
                features['user_previous_visits'] = self._count_user_visits(user['user_id'], business['business_id'])
                features['user_reviewed_this_business'] = self._has_reviewed(user['user_id'], business['business_id'])
                features['category_match'] = self._category_affinity(user['favorite_categories'], business['categories'])

                # Context features
                features['search_distance_km'] = context['distance']
                features['search_hour'] = context['hour']  # Time of day
                features['search_day_of_week'] = context['day_of_week']
                features['is_weekend'] = context['day_of_week'] >= 5

                return features

            def predict_score(self, user_id: str, business_id: str, context: Dict = None) -> float:
                """
                Predict user engagement score for business

                Returns:
                    Score between 0 and 1 (probability of engagement)
                """
                if self.model is None:
                    return 0.5  # Neutral if model not loaded

                # Fetch user and business data
                user = self._get_user(user_id)
                business = self._get_business(business_id)

                # Extract features
                features = self.extract_features(user, business, context or {})

                # Predict
                score = self.model.predict_proba([features])[0][1]  # Probability of positive class

                return score

            def train_model(self, training_data: List[Dict]):
                """
                Train ML model on historical engagement data

                Training data format:
                - features: User, business, context features
                - label: 1 (engaged: clicked, reviewed) or 0 (not engaged)
                """
                import xgboost as xgb

                X = [sample['features'] for sample in training_data]
                y = [sample['label'] for sample in training_data]

                self.model = xgb.XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    objective='binary:logistic'
                )

                self.model.fit(X, y)
        ```

        **Training data generation:**

        - **Positive examples:** User clicked/reviewed business
        - **Negative examples:** User saw but didn't engage
        - **Features:** 50+ features (user, business, context)
        - **Update frequency:** Retrain daily with new data

    === "⚡ Real-time Updates"

        ## The Challenge

        **Problem:** Business hours change, reviews added, ratings updated. How to keep data fresh?

        **Requirements:**

        - **Low latency:** Updates visible within 1-5 minutes
        - **Cache consistency:** Invalidate stale cache
        - **No downtime:** Updates without service interruption
        - **Idempotent:** Handle duplicate events

        ---

        ## Event-Driven Architecture

        ```python
        from kafka import KafkaConsumer, KafkaProducer
        import json
        from typing import Dict

        class BusinessUpdateService:
            """
            Handle real-time business updates via Kafka events

            Events:
            - business_updated: Name, hours, location changed
            - review_added: New review, update ratings
            - photo_uploaded: New photo added
            """

            def __init__(self, kafka_brokers, redis_client, db):
                self.consumer = KafkaConsumer(
                    'business_events',
                    bootstrap_servers=kafka_brokers,
                    auto_offset_reset='latest',
                    group_id='business_update_service'
                )
                self.producer = KafkaProducer(bootstrap_servers=kafka_brokers)
                self.redis = redis_client
                self.db = db

            def start(self):
                """Start consuming events"""
                for message in self.consumer:
                    event = json.loads(message.value)
                    self._process_event(event)

            def _process_event(self, event: Dict):
                """
                Process business event

                Event types:
                - business_updated
                - review_added
                - photo_uploaded
                - hours_changed
                """
                event_type = event['type']
                business_id = event['business_id']

                if event_type == 'business_updated':
                    self._handle_business_update(event)
                elif event_type == 'review_added':
                    self._handle_review_added(event)
                elif event_type == 'photo_uploaded':
                    self._handle_photo_upload(event)
                elif event_type == 'hours_changed':
                    self._handle_hours_changed(event)

            def _handle_business_update(self, event: Dict):
                """
                Handle business info update (name, address, phone)

                Actions:
                1. Update database
                2. Invalidate cache
                3. Update search index
                4. Update geo index (if location changed)
                """
                business_id = event['business_id']
                updates = event['updates']

                # Update database
                self.db.update_business(business_id, updates)

                # Invalidate cache
                self.redis.delete(f"business:{business_id}")
                self._invalidate_search_cache(business_id)

                # Update search index
                if 'name' in updates or 'categories' in updates:
                    self._update_search_index(business_id, updates)

                # Update geo index if location changed
                if 'latitude' in updates and 'longitude' in updates:
                    self._update_geo_index(business_id, updates['latitude'], updates['longitude'])

                logger.info(f"Business {business_id} updated: {updates.keys()}")

            def _handle_review_added(self, event: Dict):
                """
                Handle new review

                Actions:
                1. Store review in Cassandra
                2. Update rating aggregates
                3. Invalidate business cache
                4. Update search index (review count)
                """
                business_id = event['business_id']
                review = event['review']

                # Store review
                self.db.insert_review(review)

                # Update rating aggregates (async)
                self._update_rating_aggregates(business_id, review['rating'])

                # Invalidate cache
                self.redis.delete(f"business:{business_id}")
                self._invalidate_search_cache(business_id)

                logger.info(f"Review added for business {business_id}, rating: {review['rating']}")

            def _handle_photo_upload(self, event: Dict):
                """
                Handle photo upload

                Actions:
                1. Update business photos count
                2. Invalidate cache
                3. Trigger CDN cache warming
                """
                business_id = event['business_id']
                photo_url = event['photo_url']

                # Update database
                self.db.add_business_photo(business_id, photo_url)

                # Invalidate cache
                self.redis.delete(f"business:{business_id}")

                # Warm CDN cache (pre-fetch to edge locations)
                self._warm_cdn_cache(photo_url)

                logger.info(f"Photo uploaded for business {business_id}")

            def _handle_hours_changed(self, event: Dict):
                """
                Handle business hours update

                Actions:
                1. Update hours in database
                2. Invalidate cache
                3. Recalculate is_open_now for affected searches
                """
                business_id = event['business_id']
                new_hours = event['hours']

                # Update database
                self.db.update_business_hours(business_id, new_hours)

                # Invalidate cache
                self.redis.delete(f"business:{business_id}")
                self._invalidate_search_cache(business_id)

                logger.info(f"Hours updated for business {business_id}")

            def _update_rating_aggregates(self, business_id: str, new_rating: int):
                """
                Update average rating and rating distribution

                Uses Redis for real-time aggregation
                """
                # Increment counters
                self.redis.hincrby(f"rating:{business_id}", "total_reviews", 1)
                self.redis.hincrby(f"rating:{business_id}", "total_rating_sum", new_rating)
                self.redis.hincrby(f"rating:{business_id}", f"rating_{new_rating}_count", 1)

                # Calculate new average
                aggregates = self.redis.hgetall(f"rating:{business_id}")
                total_reviews = int(aggregates[b"total_reviews"])
                total_rating_sum = int(aggregates[b"total_rating_sum"])
                avg_rating = total_rating_sum / total_reviews

                # Update database (async, eventual consistency)
                self.db.update_business_rating(business_id, avg_rating, total_reviews)

            def _invalidate_search_cache(self, business_id: str):
                """
                Invalidate all search results containing this business

                Approach: Use cache tags or pattern matching
                """
                # Get all search cache keys containing this business
                # In production: use cache tags or bloom filter
                pattern = f"search:*:{business_id}:*"
                keys = self.redis.keys(pattern)

                if keys:
                    self.redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} search cache entries")

            def _update_search_index(self, business_id: str, updates: Dict):
                """
                Update Elasticsearch index

                Partial update for changed fields only
                """
                from elasticsearch import Elasticsearch
                es = Elasticsearch()

                es.update(
                    index='businesses',
                    id=business_id,
                    body={'doc': updates}
                )

            def _update_geo_index(self, business_id: str, lat: float, lng: float):
                """
                Update Redis Geo and PostGIS indexes

                Location change is rare but critical
                """
                # Update Redis Geo
                self.redis.geoadd('businesses:locations', lng, lat, business_id)

                # Update PostGIS
                self.db.execute(
                    """UPDATE business_locations
                       SET location = ST_MakePoint(%s, %s)::geography,
                           geohash = %s
                       WHERE business_id = %s""",
                    (lng, lat, self._encode_geohash(lat, lng), business_id)
                )

            def _warm_cdn_cache(self, photo_url: str):
                """
                Pre-fetch photo to CDN edge locations

                Improves first-load experience
                """
                # Trigger CDN cache warming
                # In production: use CDN API (CloudFront, Cloudflare)
                pass
        ```

        ---

        ## Cache Invalidation Strategy

        **Multi-layer caching:**

        | Layer | TTL | Invalidation |
        |-------|-----|--------------|
        | **Business details** | 1 hour | Event-driven (on update) |
        | **Search results** | 5 minutes | Tag-based (on business update) |
        | **Rating aggregates** | 10 minutes | Write-through (on new review) |
        | **Photos (CDN)** | 7 days | Version-based URLs |

        **Cache consistency pattern:**

        ```python
        class CacheService:
            """
            Multi-layer cache with consistency guarantees
            """

            def __init__(self, redis_client):
                self.redis = redis_client

            def get_business(self, business_id: str) -> Dict:
                """
                Get business with cache-aside pattern

                1. Check cache
                2. If miss, query database
                3. Store in cache
                """
                cache_key = f"business:{business_id}"

                # Try cache
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)

                # Cache miss - query database
                business = self.db.get_business(business_id)
                if business:
                    # Store in cache (1 hour TTL)
                    self.redis.setex(cache_key, 3600, json.dumps(business))

                return business

            def update_business(self, business_id: str, updates: Dict):
                """
                Update business with write-through caching

                1. Update database
                2. Update cache immediately
                3. Publish event for downstream services
                """
                # Update database
                self.db.update_business(business_id, updates)

                # Update cache (write-through)
                cache_key = f"business:{business_id}"
                business = self.db.get_business(business_id)
                self.redis.setex(cache_key, 3600, json.dumps(business))

                # Publish event
                self.publish_event('business_updated', {
                    'business_id': business_id,
                    'updates': updates
                })

            def invalidate_search_cache_by_tag(self, business_id: str):
                """
                Invalidate search cache using tags

                Redis Bloom filter to track which searches contain business
                """
                # Get affected search keys
                affected_keys = self._get_tagged_keys(business_id)

                # Delete all affected keys
                if affected_keys:
                    self.redis.delete(*affected_keys)
        ```

    === "📈 Scalability"

        ## The Challenge

        **Problem:** Scale from 1M to 500M searches/day. Handle peak traffic (lunch, dinner hours).

        **Bottlenecks:**

        - **Database reads:** 8,100 read QPS
        - **Geospatial queries:** Complex spatial operations
        - **Cache memory:** 500 GB in-memory cache
        - **Search index:** 100M businesses in Elasticsearch

        ---

        ## Sharding Strategy

        ### Database Sharding (Horizontal)

        **PostgreSQL (Business DB):**

        ```python
        class DatabaseSharding:
            """
            Shard businesses by geographic region

            Sharding key: geohash prefix (first 2 characters)
            - 32^2 = 1024 possible shards
            - ~100K businesses per shard
            """

            SHARD_COUNT = 64  # Start with 64 shards

            def __init__(self, db_connections: List):
                self.shards = db_connections

            def get_shard(self, business_id: str = None, lat: float = None, lng: float = None) -> int:
                """
                Determine shard for business or location

                Args:
                    business_id: Business ID (deterministic routing)
                    lat, lng: Location (geo-based routing)

                Returns:
                    Shard ID (0 to SHARD_COUNT-1)
                """
                if business_id:
                    # Hash-based sharding for direct lookup
                    return hash(business_id) % self.SHARD_COUNT
                elif lat is not None and lng is not None:
                    # Geo-based sharding for proximity queries
                    geohash = self._encode_geohash(lat, lng, precision=2)
                    return hash(geohash) % self.SHARD_COUNT
                else:
                    raise ValueError("Must provide business_id or location")

            def query_shard(self, shard_id: int, query: str, params: tuple):
                """Execute query on specific shard"""
                return self.shards[shard_id].execute(query, params)

            def query_all_shards(self, query: str, params: tuple):
                """
                Scatter-gather: query all shards and merge results

                Used for global queries (search by name, etc.)
                """
                results = []
                with ThreadPoolExecutor(max_workers=self.SHARD_COUNT) as executor:
                    futures = [
                        executor.submit(self.query_shard, shard_id, query, params)
                        for shard_id in range(self.SHARD_COUNT)
                    ]

                    for future in as_completed(futures):
                        results.extend(future.result())

                return results

            def search_nearby(self, lat: float, lng: float, radius_km: float):
                """
                Search nearby businesses across shards

                Optimization: Only query shards that overlap with search radius
                """
                # Determine affected shards (geohash prefixes in radius)
                affected_shards = self._get_affected_shards(lat, lng, radius_km)

                # Query only affected shards
                results = []
                with ThreadPoolExecutor(max_workers=len(affected_shards)) as executor:
                    futures = [
                        executor.submit(
                            self.query_shard,
                            shard_id,
                            "SELECT * FROM businesses WHERE ST_DWithin(...)",
                            (lng, lat, radius_km * 1000)
                        )
                        for shard_id in affected_shards
                    ]

                    for future in as_completed(futures):
                        results.extend(future.result())

                return results

            def _get_affected_shards(self, lat: float, lng: float, radius_km: float) -> List[int]:
                """
                Determine which shards overlap with search radius

                Uses geohash approximation
                """
                # Get center geohash
                center_geohash = self._encode_geohash(lat, lng, precision=2)

                # Get neighboring geohashes (8 neighbors + center)
                affected_geohashes = [center_geohash] + self._get_neighbors(center_geohash)

                # Map to shard IDs
                affected_shards = list(set(hash(gh) % self.SHARD_COUNT for gh in affected_geohashes))

                return affected_shards
        ```

        ---

        ### Redis Sharding (Cluster Mode)

        **Redis Cluster:** 16,384 hash slots distributed across nodes

        ```python
        from redis.cluster import RedisCluster

        class RedisGeoCluster:
            """
            Redis Cluster for distributed geospatial indexing

            Configuration:
            - 20 nodes (masters)
            - 3x replication (60 nodes total)
            - ~5M businesses per node
            """

            def __init__(self, cluster_nodes):
                self.redis = RedisCluster(
                    startup_nodes=cluster_nodes,
                    decode_responses=True,
                    skip_full_coverage_check=True
                )

            def search_radius(self, lat: float, lng: float, radius_km: float):
                """
                Search across Redis Cluster

                Redis handles sharding automatically via hash slots
                """
                # GEORADIUS works transparently across cluster
                results = self.redis.georadius(
                    'businesses:locations',
                    lng,
                    lat,
                    radius_km,
                    unit='km',
                    withdist=True,
                    sort='ASC',
                    count=100
                )

                return results
        ```

        ---

        ## Load Balancing

        **Geographic load balancing:**

        ```
        User Request (San Francisco)
        └── Global Load Balancer (Route 53)
            ├── US-West Region (primary)
            │   ├── Search API (10 instances)
            │   ├── Redis Cluster (20 nodes)
            │   └── PostgreSQL (8 shards)
            └── US-East Region (replica, read-only)
                ├── Search API (5 instances)
                ├── Redis Cluster (read replica)
                └── PostgreSQL (read replicas)
        ```

        **Benefits:**

        - **Low latency:** Users routed to nearest region
        - **High availability:** Failover to secondary region
        - **Read scaling:** Read replicas handle 80% of traffic

        ---

        ## Performance Optimization

        ### Query Optimization

        ```sql
        -- Before: Full table scan (8 seconds)
        SELECT * FROM businesses
        WHERE ST_DWithin(location, ST_MakePoint(-122.4194, 37.7749)::geography, 5000)
        ORDER BY rating DESC
        LIMIT 20;

        -- After: Spatial index + filter pushdown (87ms)
        SELECT * FROM businesses
        WHERE ST_DWithin(
            location,
            ST_MakePoint(-122.4194, 37.7749)::geography,
            5000
        )
        AND rating >= 3.0  -- Pre-filter before sorting
        ORDER BY ST_Distance(location, ST_MakePoint(-122.4194, 37.7749)::geography)
        LIMIT 100;  -- Fetch more, rank in application

        -- Indexes
        CREATE INDEX idx_business_location_gist ON businesses USING GIST(location);
        CREATE INDEX idx_business_rating ON businesses(rating) WHERE rating >= 3.0;
        ```

        ---

        ### Caching Layers

        | Layer | Technology | Hit Rate | Latency |
        |-------|------------|----------|---------|
        | **L1: CDN Edge** | CloudFront | 90% | 10-30ms |
        | **L2: Redis** | ElastiCache | 80% | 1-5ms |
        | **L3: Database** | PostgreSQL | - | 50-200ms |

        **Cascading cache:**

        ```python
        async def get_business(business_id: str):
            """Multi-layer cache lookup"""
            # L1: CDN (for static data)
            cdn_url = f"https://cdn.yelp.com/business/{business_id}.json"
            # (Handled by client)

            # L2: Redis
            cached = await redis.get(f"business:{business_id}")
            if cached:
                return json.loads(cached)

            # L3: Database
            business = await db.get_business(business_id)
            if business:
                await redis.setex(f"business:{business_id}", 3600, json.dumps(business))

            return business
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Yelp from 10M to 500M searches/day.

    **Scaling challenges at 500M searches/day:**

    - **Search QPS:** 5,800 average, 17,400 peak
    - **Database load:** 8,100 read QPS across shards
    - **Cache size:** 500 GB in-memory
    - **CDN egress:** 3.7 Gbps for photos

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **PostgreSQL reads** | ✅ Yes | Read replicas (5x), query optimization, sharding (64 shards) |
    | **Redis memory** | ✅ Yes | Cluster mode (20 nodes), LRU eviction, cache warming |
    | **Geospatial queries** | ✅ Yes | Redis Geo for hot locations, PostGIS for complex queries |
    | **Search index** | 🟡 Approaching | Elasticsearch sharding (32 shards), hot/warm/cold architecture |
    | **API servers** | 🟢 OK | Horizontal scaling (100 instances), auto-scaling |

    ---

    ## Cost Optimization

    **Monthly cost at 500M searches/day:**

    | Component | Configuration | Monthly Cost |
    |-----------|--------------|--------------|
    | **API Servers** | 100 × c5.2xlarge (8 vCPU, 16 GB) | $24,000 |
    | **PostgreSQL (Business)** | 64 shards × db.r5.xlarge | $76,800 |
    | **PostgreSQL (Read replicas)** | 64 × db.r5.large | $38,400 |
    | **Cassandra (Reviews)** | 30 nodes × i3.2xlarge | $54,000 |
    | **Redis Cluster** | 20 nodes × cache.r5.2xlarge | $25,600 |
    | **Elasticsearch** | 32 nodes × r5.2xlarge | $51,200 |
    | **S3 Storage** | 1.5 PB photos | $34,500 |
    | **CloudFront CDN** | 3.7 Gbps × 30 days | $47,000 |
    | **Kafka** | 10 brokers × m5.xlarge | $3,600 |
    | **NAT Gateway** | 3 × $0.045/GB egress | $5,400 |
    | **Total** | | **$360,500/month** |

    **Optimization strategies:**

    1. **S3 Intelligent Tiering:** Save 40% on infrequently accessed photos
    2. **Reserved Instances:** 30% savings on EC2/RDS (1-year commitment)
    3. **Spot Instances:** 70% savings for batch processing (ML ranking)
    4. **Compression:** Gzip API responses (reduce bandwidth 60%)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold | Action |
    |--------|--------|-----------------|--------|
    | **Search Latency (P95)** | < 200ms | > 500ms | Scale API servers, check database |
    | **Cache Hit Rate** | > 80% | < 70% | Increase cache size, adjust TTL |
    | **Database CPU** | < 70% | > 85% | Add read replicas, optimize queries |
    | **Geo Query Latency** | < 50ms | > 100ms | Check Redis Geo, rebalance shards |
    | **CDN Hit Rate** | > 90% | < 85% | Check cache headers, warm cache |
    | **Error Rate** | < 0.1% | > 1% | Check logs, rollback deployment |

    **Observability stack:**

    - **Metrics:** Prometheus + Grafana
    - **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana)
    - **Tracing:** Jaeger (distributed tracing)
    - **Alerts:** PagerDuty (on-call rotation)

    ---

    ## Disaster Recovery

    **RTO (Recovery Time Objective):** 15 minutes
    **RPO (Recovery Point Objective):** 5 minutes

    **Strategy:**

    1. **Multi-region deployment:** Active-active in US-West and US-East
    2. **Database replication:** Continuous replication with 5-minute lag
    3. **Automated failover:** Route 53 health checks trigger failover
    4. **Backup:** Daily snapshots to S3 (7-day retention)

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Geospatial indexing:** Redis Geo + PostGIS hybrid for fast proximity search
    2. **Quadtree/Geohash:** Efficient spatial partitioning for 100M businesses
    3. **Relevance ranking:** Distance + Rating + Popularity + ML personalization
    4. **Event-driven updates:** Kafka for real-time rating/review updates
    5. **Multi-layer caching:** CDN (photos) + Redis (queries) + Database
    6. **Sharding:** Geographic sharding for locality, 64 shards for 100M businesses

    ---

    ## Interview Tips

    ✅ **Start with geospatial fundamentals** - Explain quadtree or geohash clearly

    ✅ **Discuss indexing trade-offs** - Redis Geo vs QuadTree vs Geohash

    ✅ **Ranking is critical** - Don't just return nearest, rank by relevance

    ✅ **Caching strategy** - 80% cache hit rate essential for performance

    ✅ **Real-time updates** - Event-driven architecture for reviews/ratings

    ✅ **Scale numbers** - Calculate QPS, storage, bandwidth confidently

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle high-density areas (NYC)?"** | Quadtree auto-subdivides, Redis Geo handles millions of points, shard by geohash prefix |
    | **"How to rank results by relevance?"** | Multi-factor: distance (35%), rating (30%), popularity (20%), recency (10%), personalization (5%) |
    | **"How to keep ratings updated in real-time?"** | Event-driven: Kafka → Rating Aggregator → Redis + Database, 1-5 minute eventual consistency |
    | **"How to search by business name?"** | Elasticsearch full-text search, fuzzy matching, autocomplete with prefix queries |
    | **"How to handle 'open now' filter?"** | Store hours in database, calculate in application, cache is_open_now for 5 minutes |
    | **"How to prevent fake reviews?"** | Spam detection: ML model, rate limiting (5 reviews/day), one review per user per business |

    ---

    ## Comparison with Similar Systems

    | System | Key Difference | Yelp's Approach |
    |--------|---------------|-----------------|
    | **Google Maps** | Larger scale (200M+ businesses) | Hierarchical quadtree, global CDN, ML-heavy ranking |
    | **Foursquare** | Focus on check-ins | Event-driven check-in processing, gamification |
    | **TripAdvisor** | Travel-focused | Category-specific ranking, seasonal trends |
    | **Uber Eats** | Real-time delivery | Live driver location tracking, ETA prediction |

---

**Difficulty:** 🟡 Medium | **Interview Time:** 50-60 minutes | **Companies:** Yelp, Google Maps, Foursquare, TripAdvisor, Zomato, Uber Eats

---

*Master this problem and you'll be ready for: Google Maps, Uber (rider/driver matching), Airbnb (property search), DoorDash (restaurant discovery)*
