# Design Music Streaming (Spotify)

A global music streaming platform where users can play music, create playlists, discover new songs, download for offline listening, and share music with friends.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 500M users, 100M songs, 1B streams/day, 70M concurrent streams |
| **Key Challenges** | Audio delivery, CDN optimization, offline playback, personalized recommendations, low latency startup |
| **Core Concepts** | Audio encoding (MP3/AAC/Ogg), adaptive bitrate streaming, collaborative filtering, CDN strategy, offline sync |
| **Companies** | Spotify, Apple Music, YouTube Music, Amazon Music, Tidal, Pandora |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Play Music** | Stream songs with adaptive bitrate | P0 (Must have) |
    | **Search** | Search songs, artists, albums, playlists | P0 (Must have) |
    | **Playlists** | Create, edit, share playlists | P0 (Must have) |
    | **Offline Mode** | Download songs for offline playback | P0 (Must have) |
    | **Recommendations** | Personalized song/playlist recommendations | P0 (Must have) |
    | **Queue Management** | Play queue, shuffle, repeat | P1 (Should have) |
    | **Social Features** | Follow users, share songs, collaborative playlists | P1 (Should have) |
    | **Lyrics** | Display synchronized lyrics | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Podcasts and audiobooks
    - Live radio streaming
    - Music uploads by users
    - Video content
    - Payment processing
    - Social media integration (external)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users expect always-on music access |
    | **Latency (Playback Start)** | < 200ms p95 | Fast playback critical for UX |
    | **Latency (Search)** | < 100ms p95 | Instant search results expected |
    | **Consistency** | Eventual consistency | Brief sync delays acceptable (playlist updates) |
    | **Scalability** | Billions of streams per day | Handle viral songs, new releases |
    | **Bandwidth Efficiency** | Adaptive bitrate | Minimize data usage, optimize quality |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total Users: 500M monthly active users (MAU)
    Daily Active Users (DAU): 200M (40% of MAU)
    Premium Users: 200M (40% of MAU)
    Free Users: 300M (60% of MAU)

    Music streaming:
    - Average listening time: 3 hours/day per DAU
    - Average song duration: 3 minutes
    - Songs per DAU: 3 hours × 60 / 3 = 60 songs/day
    - Daily streams: 200M × 60 = 12B streams/day
    - Stream QPS: 12B / 86,400 = ~139,000 streams/sec
    - Peak QPS: 3x average = ~417,000 streams/sec (evening hours)

    Concurrent streams:
    - Average concurrent users: 200M × 15% = 30M
    - Peak concurrent: 70M (evening hours)

    Search requests:
    - Searches per DAU: ~10 searches/day
    - Daily searches: 200M × 10 = 2B searches/day
    - Search QPS: 2B / 86,400 = ~23,000 searches/sec

    Playlist operations:
    - Playlist updates per DAU: ~5 updates/day
    - Daily updates: 200M × 5 = 1B updates/day
    - Update QPS: 1B / 86,400 = ~11,600 updates/sec

    Offline downloads:
    - Downloads per premium DAU: ~5 songs/day
    - Daily downloads: 80M × 5 = 400M downloads/day
    - Download QPS: 400M / 86,400 = ~4,600 downloads/sec

    Total Read QPS: ~162K (streams + searches)
    Total Write QPS: ~16K (playlist updates + plays tracking + downloads)
    Read/Write ratio: 10:1 (read-heavy system)
    ```

    ### Storage Estimates

    ```
    Music catalog storage:
    - Total songs: 100M songs
    - Average song size (320kbps MP3): 10 MB
    - Multiple formats (MP3, AAC, Ogg) × 3 quality levels: 30 MB/song
    - Total music: 100M × 30 MB = 3 PB

    Metadata storage:
    - Song metadata: 100M × 5 KB = 500 GB
    - Artist metadata: 10M artists × 10 KB = 100 GB
    - Album metadata: 20M albums × 8 KB = 160 GB
    - User data: 500M × 10 KB = 5 TB

    Playlist storage:
    - Average playlists per user: 10
    - Total playlists: 500M × 10 = 5B playlists
    - Playlist metadata: 5B × 2 KB = 10 TB

    Listening history:
    - Streams per day: 12B
    - Record per stream: 200 bytes (user_id, song_id, timestamp, duration, etc.)
    - Daily: 12B × 200 bytes = 2.4 TB/day
    - 1 year: 2.4 TB × 365 = 876 TB
    - 5 years: 876 TB × 5 = 4.38 PB

    Recommendation models:
    - User embeddings: 500M × 256 floats × 4 bytes = 512 GB
    - Song embeddings: 100M × 256 floats × 4 bytes = 102 GB
    - Model parameters: 50 GB

    Total: 3 PB (music) + 4.38 PB (history) + 15 TB (metadata) ≈ 7.4 PB
    ```

    ### Bandwidth Estimates

    ```
    Audio streaming (ingress - CDN pull from origin):
    - Average bitrate: 160 kbps (adaptive: 96-320 kbps)
    - Concurrent streams: 70M (peak)
    - Bandwidth: 70M × 160 kbps = 11.2 Tbps
    - Actual CDN origin: ~5% (cache miss) = 560 Gbps

    Audio streaming (egress - to users):
    - Total egress: 70M × 160 kbps = 11.2 Tbps (handled by CDN)

    Downloads (offline):
    - 4,600 downloads/sec × 10 MB = 46 GB/sec = 368 Gbps

    Metadata/API traffic:
    - Search, playlists, recommendations: ~50 Gbps

    Total ingress (to origin): ~560 Gbps
    Total egress (CDN): ~11.6 Tbps (CDN critical)
    ```

    ### Memory Estimates (Caching)

    ```
    Hot songs cache (top 1% most played):
    - Songs: 1M × 30 MB = 30 TB
    - Cache in CDN edge locations

    Metadata cache:
    - Hot songs metadata: 10M × 5 KB = 50 GB
    - User sessions: 30M × 20 KB = 600 GB
    - Playlist cache: 50M × 2 KB = 100 GB
    - Search results: 10 GB

    Recommendation cache:
    - Personalized playlists: 100M × 5 KB = 500 GB
    - Similar songs: 50M × 2 KB = 100 GB

    Total cache: 50 GB + 600 GB + 100 GB + 10 GB + 500 GB + 100 GB ≈ 1.36 TB
    (excluding CDN audio cache)
    ```

    ---

    ## Key Assumptions

    1. Average song: 3 minutes, 10 MB at 320kbps
    2. 40% DAU engagement from MAU
    3. Read-heavy system (10:1 ratio)
    4. 95% CDN cache hit rate for popular songs
    5. Adaptive bitrate: 96-320 kbps based on network
    6. Offline downloads limited to premium users
    7. Real-time recommendations not critical (can be pre-computed)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **CDN-first architecture:** Audio delivery via global CDN (low latency)
    2. **Adaptive bitrate streaming:** Dynamic quality based on network
    3. **Offline-first mobile apps:** Sync for seamless offline playback
    4. **ML-powered recommendations:** Collaborative filtering + deep learning
    5. **Eventual consistency:** Prioritize availability over strict consistency
    6. **Multi-tier caching:** CDN → Redis → Database

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App<br/>iOS/Android]
            Web[Web Player]
            Desktop[Desktop App]
        end

        subgraph "Edge Layer"
            CDN[Global CDN<br/>Audio delivery<br/>1000+ PoPs]
            LB[Load Balancer<br/>API Gateway]
        end

        subgraph "API Layer"
            Stream_API[Streaming Service<br/>Get audio URL]
            Search_API[Search Service<br/>Songs/artists/playlists]
            Playlist_API[Playlist Service<br/>CRUD operations]
            User_API[User Service<br/>Profile/preferences]
            Offline_API[Offline Service<br/>Download management]
        end

        subgraph "Core Services"
            Rec_Service[Recommendation Engine<br/>Personalized playlists]
            Analytics[Analytics Service<br/>Play tracking]
            Auth[Auth Service<br/>JWT tokens]
            Social[Social Service<br/>Follow/share]
        end

        subgraph "Data Processing"
            Stream_Processor[Stream Processor<br/>Kafka Streams<br/>Play events]
            Rec_Pipeline[ML Pipeline<br/>Model training<br/>Batch processing]
            Search_Indexer[Search Indexer<br/>Elasticsearch]
            ETL[ETL Pipeline<br/>Data warehouse]
        end

        subgraph "Caching"
            Redis_Meta[Redis<br/>Metadata cache]
            Redis_User[Redis<br/>User sessions]
            Redis_Rec[Redis<br/>Recommendations]
        end

        subgraph "Storage"
            Music_Store[Object Storage<br/>S3/GCS<br/>Audio files<br/>3 PB]
            Catalog_DB[(Catalog DB<br/>PostgreSQL<br/>Songs/artists/albums)]
            User_DB[(User DB<br/>Cassandra<br/>Profiles/playlists)]
            Play_DB[(Analytics DB<br/>ClickHouse<br/>Play history)]
            Search_DB[(Elasticsearch<br/>Full-text search)]
            Graph_DB[(Neo4j<br/>Social graph)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming<br/>Play events, metrics]
        end

        Mobile --> CDN
        Web --> CDN
        Desktop --> CDN
        Mobile --> LB
        Web --> LB
        Desktop --> LB

        CDN --> Music_Store

        LB --> Stream_API
        LB --> Search_API
        LB --> Playlist_API
        LB --> User_API
        LB --> Offline_API

        Stream_API --> Redis_Meta
        Stream_API --> Catalog_DB
        Stream_API --> CDN
        Stream_API --> Kafka

        Search_API --> Redis_Meta
        Search_API --> Search_DB

        Playlist_API --> Redis_Meta
        Playlist_API --> User_DB

        User_API --> Redis_User
        User_API --> User_DB

        Offline_API --> Music_Store
        Offline_API --> User_DB

        Rec_Service --> Redis_Rec
        Rec_Service --> User_DB
        Rec_Service --> Play_DB

        Analytics --> Kafka
        Kafka --> Stream_Processor
        Kafka --> ETL

        Stream_Processor --> Play_DB
        Stream_Processor --> Rec_Pipeline

        Rec_Pipeline --> Redis_Rec
        Rec_Pipeline --> User_DB

        Search_Indexer --> Search_DB
        ETL --> Play_DB

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Meta fill:#fff4e1
        style Redis_User fill:#fff4e1
        style Redis_Rec fill:#fff4e1
        style Music_Store fill:#f3e5f5
        style Catalog_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Play_DB fill:#e8eaf6
        style Search_DB fill:#e8eaf6
        style Graph_DB fill:#e1f5e1
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Global CDN** | Low latency audio delivery (<200ms), 95% cache hit rate, reduces origin load | Direct streaming from origin (high latency, expensive bandwidth) |
    | **Cassandra (User DB)** | Write-heavy (playlist updates), horizontal scaling, high availability | PostgreSQL (can't handle write throughput), MongoDB (consistency issues) |
    | **ClickHouse (Analytics)** | Fast analytical queries on billions of play events, columnar storage | PostgreSQL (too slow for TB-scale analytics), BigQuery (higher cost) |
    | **Elasticsearch** | Fast full-text search (<100ms), fuzzy matching, typo tolerance | Database LIKE queries (too slow), custom search (complex) |
    | **Kafka** | Reliable event streaming (play events, metrics), replay capability | Direct database writes (can't handle 139K QPS), RabbitMQ (throughput limits) |
    | **Redis** | Sub-10ms metadata reads, session storage, recommendation cache | No cache (databases can't handle 162K read QPS), Memcached (limited features) |

    **Key Trade-off:** We chose **availability over consistency**. Playlist updates may take seconds to sync across devices, but music playback never stops.

    ---

    ## API Design

    ### 1. Get Stream URL

    **Request:**
    ```http
    GET /api/v1/stream/{song_id}?quality=high
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "stream_url": "https://cdn.spotify.com/audio/12345.mp3?token=xyz&expires=1643712000",
      "song_id": "12345",
      "format": "mp3",
      "bitrate": 320,
      "duration_ms": 180000,
      "cdn_endpoints": [
        "https://cdn-us-east.spotify.com/audio/12345.mp3?token=xyz",
        "https://cdn-us-west.spotify.com/audio/12345.mp3?token=xyz"
      ],
      "expires_at": "2026-02-02T11:00:00Z"
    }
    ```

    **Design Notes:**

    - Return CDN URL with signed token (prevents hotlinking)
    - Multiple CDN endpoints for failover
    - Token expires in 1 hour (refresh required)
    - Quality parameter: low (96kbps), medium (160kbps), high (320kbps)
    - Track stream initiation in Kafka

    ---

    ### 2. Search

    **Request:**
    ```http
    GET /api/v1/search?q=bohemian+rhapsody&type=track,artist&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tracks": [
        {
          "song_id": "67890",
          "title": "Bohemian Rhapsody",
          "artist": {
            "artist_id": "queen",
            "name": "Queen"
          },
          "album": {
            "album_id": "night-opera",
            "name": "A Night at the Opera",
            "cover_art": "https://cdn.spotify.com/covers/night-opera.jpg"
          },
          "duration_ms": 354000,
          "preview_url": "https://cdn.spotify.com/preview/67890.mp3"
        }
      ],
      "artists": [
        {
          "artist_id": "queen",
          "name": "Queen",
          "image": "https://cdn.spotify.com/artists/queen.jpg",
          "followers": 42000000
        }
      ],
      "total": 1542,
      "next_offset": 20
    }
    ```

    **Design Notes:**

    - Elasticsearch for full-text search
    - Support filters: artist, album, genre, year
    - Fuzzy matching for typos
    - Return preview URLs (30-second clips)
    - Autocomplete endpoint separate

    ---

    ### 3. Create/Update Playlist

    **Request:**
    ```http
    POST /api/v1/playlists
    Authorization: Bearer <token>
    Content-Type: application/json

    {
      "name": "My Favorites",
      "description": "Best songs of all time",
      "public": true,
      "collaborative": false,
      "song_ids": ["12345", "67890", "11111"]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "playlist_id": "playlist_abc123",
      "name": "My Favorites",
      "description": "Best songs of all time",
      "owner": {
        "user_id": "user123",
        "username": "john_doe"
      },
      "public": true,
      "collaborative": false,
      "follower_count": 0,
      "song_count": 3,
      "created_at": "2026-02-02T10:30:00Z",
      "updated_at": "2026-02-02T10:30:00Z",
      "cover_image": "https://cdn.spotify.com/playlist-covers/default.jpg"
    }
    ```

    **Design Notes:**

    - Async replication across regions (eventual consistency)
    - Generate mosaic cover from first 4 songs
    - Rate limit: 100 playlist updates per hour
    - Support batch add/remove songs

    ---

    ### 4. Get Recommendations

    **Request:**
    ```http
    GET /api/v1/recommendations?seed_songs=12345,67890&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tracks": [
        {
          "song_id": "99999",
          "title": "Stairway to Heaven",
          "artist": { "artist_id": "led-zeppelin", "name": "Led Zeppelin" },
          "album": { "album_id": "led-iv", "name": "Led Zeppelin IV" },
          "duration_ms": 482000,
          "reason": "Based on your listening history"
        }
      ],
      "seed_songs": ["12345", "67890"],
      "recommendation_id": "rec_xyz789"
    }
    ```

    **Design Notes:**

    - Pre-computed recommendations (batch processing)
    - Real-time fallback using song embeddings
    - Cache results for 1 hour
    - Track which recommendations user plays

    ---

    ## Database Schema

    ### Songs Catalog (PostgreSQL)

    ```sql
    -- Songs table (read-heavy, rarely updated)
    CREATE TABLE songs (
        song_id VARCHAR(36) PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        artist_id VARCHAR(36) NOT NULL,
        album_id VARCHAR(36),
        duration_ms INT NOT NULL,
        release_date DATE,
        genre VARCHAR(100),
        explicit BOOLEAN DEFAULT FALSE,
        isrc VARCHAR(20), -- International Standard Recording Code

        -- Audio files (different formats/qualities)
        audio_files JSONB, -- {"mp3_320": "s3://...", "aac_256": "s3://...", "ogg_160": "s3://..."}

        -- Metadata
        lyrics_url TEXT,
        preview_url TEXT, -- 30-second preview

        -- Aggregated stats (updated periodically)
        play_count BIGINT DEFAULT 0,
        like_count INT DEFAULT 0,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_artist (artist_id),
        INDEX idx_album (album_id),
        INDEX idx_genre (genre),
        INDEX idx_release_date (release_date)
    );

    -- Artists table
    CREATE TABLE artists (
        artist_id VARCHAR(36) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        bio TEXT,
        image_url TEXT,
        verified BOOLEAN DEFAULT FALSE,
        follower_count INT DEFAULT 0,
        monthly_listeners BIGINT DEFAULT 0,
        genres VARCHAR(100)[], -- Array of genres
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_name (name),
        INDEX idx_monthly_listeners (monthly_listeners DESC)
    );

    -- Albums table
    CREATE TABLE albums (
        album_id VARCHAR(36) PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        artist_id VARCHAR(36) NOT NULL,
        release_date DATE,
        cover_art_url TEXT,
        album_type VARCHAR(20), -- album, single, compilation
        total_tracks INT,
        label VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_artist (artist_id),
        INDEX idx_release_date (release_date)
    );
    ```

    **Why PostgreSQL:**

    - **ACID compliance:** Catalog data requires strong consistency
    - **Complex queries:** Joins across songs, artists, albums
    - **Read-heavy:** 99% reads (catalog rarely changes)
    - **Mature ecosystem:** Well-understood, reliable

    ---

    ### User Data (Cassandra)

    ```sql
    -- Users table
    CREATE TABLE users (
        user_id UUID PRIMARY KEY,
        username VARCHAR,
        email VARCHAR,
        password_hash VARCHAR,
        display_name VARCHAR,
        profile_pic_url TEXT,
        country VARCHAR(2), -- ISO country code
        subscription_type VARCHAR, -- free, premium, family
        created_at TIMESTAMP,
        last_login TIMESTAMP
    );

    -- User playlists (high write volume)
    CREATE TABLE user_playlists (
        user_id UUID,
        playlist_id UUID,
        name VARCHAR,
        description TEXT,
        public BOOLEAN,
        collaborative BOOLEAN,
        song_count INT,
        follower_count INT,
        cover_image_url TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        PRIMARY KEY (user_id, created_at)
    ) WITH CLUSTERING ORDER BY (created_at DESC);

    -- Playlist songs (for quick playlist access)
    CREATE TABLE playlist_songs (
        playlist_id UUID,
        position INT,
        song_id VARCHAR,
        added_at TIMESTAMP,
        added_by UUID, -- user who added song
        PRIMARY KEY (playlist_id, position)
    ) WITH CLUSTERING ORDER BY (position ASC);

    -- User library (liked songs)
    CREATE TABLE user_library (
        user_id UUID,
        song_id VARCHAR,
        liked_at TIMESTAMP,
        PRIMARY KEY (user_id, liked_at)
    ) WITH CLUSTERING ORDER BY (liked_at DESC);

    -- Offline downloads (per device)
    CREATE TABLE offline_downloads (
        user_id UUID,
        device_id VARCHAR,
        song_id VARCHAR,
        downloaded_at TIMESTAMP,
        quality VARCHAR, -- low, medium, high
        file_size_mb INT,
        PRIMARY KEY ((user_id, device_id), song_id)
    );
    ```

    **Why Cassandra:**

    - **High write throughput:** Playlist updates, song likes
    - **Linear scalability:** Add nodes for more capacity
    - **No single point of failure:** Multi-master replication
    - **Time-series data:** Clustering by timestamp for recency queries

    ---

    ### Play History (ClickHouse)

    ```sql
    -- Play events (append-only, billions of rows)
    CREATE TABLE play_events (
        event_id UUID,
        user_id UUID,
        song_id VARCHAR,

        -- Playback details
        started_at DateTime,
        duration_played_ms UInt32, -- How much was played
        total_duration_ms UInt32,   -- Song duration
        completion_rate Float32,    -- duration_played / total_duration

        -- Context
        source VARCHAR, -- playlist, album, radio, search, recommendation
        source_id VARCHAR, -- ID of playlist/album/etc
        device_type VARCHAR, -- mobile, desktop, web
        platform VARCHAR, -- ios, android, windows, etc

        -- Quality
        bitrate UInt16, -- 96, 160, 256, 320 kbps

        -- Location
        country VARCHAR(2),
        city VARCHAR,

        -- Derived columns (for fast aggregations)
        date Date DEFAULT toDate(started_at),
        hour UInt8 DEFAULT toHour(started_at)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(started_at)
    ORDER BY (user_id, started_at);

    -- Materialized view for daily aggregations
    CREATE MATERIALIZED VIEW daily_song_plays
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (song_id, date)
    AS SELECT
        song_id,
        date,
        count() as play_count,
        uniq(user_id) as unique_listeners,
        avg(completion_rate) as avg_completion
    FROM play_events
    GROUP BY song_id, date;
    ```

    **Why ClickHouse:**

    - **Columnar storage:** 10-100x compression for analytics
    - **Fast aggregations:** Billions of rows queried in seconds
    - **Time-series optimized:** Partitioning by month for fast queries
    - **Materialized views:** Pre-aggregated metrics for dashboards

    ---

    ### Search Index (Elasticsearch)

    ```json
    // Song search index
    {
      "mappings": {
        "properties": {
          "song_id": { "type": "keyword" },
          "title": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
              "keyword": { "type": "keyword" },
              "autocomplete": {
                "type": "text",
                "analyzer": "autocomplete"
              }
            }
          },
          "artist_name": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
              "keyword": { "type": "keyword" }
            }
          },
          "album_name": { "type": "text" },
          "genre": { "type": "keyword" },
          "release_year": { "type": "integer" },
          "duration_ms": { "type": "integer" },
          "popularity_score": { "type": "float" },
          "play_count": { "type": "long" },
          "explicit": { "type": "boolean" }
        }
      },
      "settings": {
        "analysis": {
          "analyzer": {
            "autocomplete": {
              "type": "custom",
              "tokenizer": "standard",
              "filter": ["lowercase", "autocomplete_filter"]
            }
          },
          "filter": {
            "autocomplete_filter": {
              "type": "edge_ngram",
              "min_gram": 2,
              "max_gram": 20
            }
          }
        }
      }
    }
    ```

    ---

    ## Data Flow Diagrams

    ### Music Playback Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API_Gateway
        participant Stream_API
        participant Redis
        participant Catalog_DB
        participant CDN
        participant Music_Store
        participant Kafka

        Client->>API_Gateway: Play song (song_id)
        API_Gateway->>Stream_API: GET /stream/{song_id}

        Stream_API->>Redis: GET song:12345
        alt Cache HIT
            Redis-->>Stream_API: Song metadata + CDN URL
        else Cache MISS
            Redis-->>Stream_API: null
            Stream_API->>Catalog_DB: SELECT * FROM songs WHERE song_id='12345'
            Catalog_DB-->>Stream_API: Song metadata
            Stream_API->>Redis: SET song:12345 (TTL: 1 hour)
        end

        Stream_API->>Stream_API: Generate signed CDN URL (expires 1h)
        Stream_API->>Kafka: Publish play_started event
        Stream_API-->>Client: 200 OK {stream_url, metadata}

        Client->>CDN: GET https://cdn.spotify.com/audio/12345.mp3
        alt CDN Cache HIT (95% of requests)
            CDN-->>Client: Audio stream (206 Partial Content)
        else CDN Cache MISS (5% of requests)
            CDN->>Music_Store: GET s3://music/12345.mp3
            Music_Store-->>CDN: Audio file
            CDN->>CDN: Cache audio file
            CDN-->>Client: Audio stream (206 Partial Content)
        end

        Client->>Client: Play audio, track progress

        Note over Client: Every 30 seconds
        Client->>Kafka: Send play_progress event

        Note over Client: Song ends or user skips
        Client->>Kafka: Send play_completed event
    ```

    **Flow Explanation:**

    1. **Get stream URL** - API returns CDN URL with signed token
    2. **Cache check** - Redis cache for song metadata (sub-10ms)
    3. **CDN delivery** - 95% cache hit rate, HTTP range requests for streaming
    4. **Event tracking** - Kafka events for analytics, recommendations
    5. **Adaptive bitrate** - Client selects quality based on bandwidth

    ---

    ### Offline Download Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Offline_API
        participant User_DB
        participant Music_Store
        participant CDN

        Client->>Offline_API: POST /offline/download {song_ids, quality}
        Offline_API->>Offline_API: Check subscription (premium only)

        alt Not premium
            Offline_API-->>Client: 403 Forbidden (premium required)
        else Premium user
            Offline_API->>User_DB: Check download limit (10,000 songs max)
            User_DB-->>Offline_API: Current count: 5,234

            Offline_API->>Music_Store: Get audio URLs
            Music_Store-->>Offline_API: Signed download URLs

            Offline_API->>User_DB: INSERT INTO offline_downloads
            Offline_API-->>Client: 200 OK {download_urls}

            loop For each song
                Client->>CDN: Download audio file
                CDN->>Music_Store: GET s3://music/12345.mp3
                Music_Store-->>CDN: Audio file
                CDN-->>Client: Audio file
                Client->>Client: Encrypt and store locally
            end

            Client->>Offline_API: Mark downloads complete
        end
    ```

    **Flow Explanation:**

    1. **Premium check** - Only premium users can download
    2. **Limit enforcement** - Max 10,000 downloaded songs per account
    3. **Encrypted storage** - Files encrypted on device (DRM)
    4. **License refresh** - Downloads expire after 30 days offline
    5. **Sync on connect** - Verify licenses when online

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Spotify subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Audio Delivery & CDN** | How to stream audio to 70M concurrent users? | Global CDN + adaptive bitrate + HTTP range requests |
    | **Recommendation System** | How to personalize music for 500M users? | Collaborative filtering + deep learning embeddings |
    | **Offline Sync** | How to sync 10,000 songs per user? | Differential sync + priority queue + encryption |
    | **Search & Discovery** | How to search 100M songs in <100ms? | Elasticsearch + popularity boosting + autocomplete |

    ---

    === "🎵 Audio Delivery & CDN"

        ## The Challenge

        **Problem:** Deliver audio to 70M concurrent users with <200ms startup latency and handle 11.6 Tbps bandwidth.

        **Requirements:**

        - **Low latency:** < 200ms playback start
        - **High availability:** 99.9% uptime
        - **Bandwidth efficiency:** Adaptive bitrate (96-320 kbps)
        - **Cost optimization:** CDN cache hit > 95%

        ---

        ## CDN Architecture

        **Multi-tier CDN strategy:**

        ```
        Origin (S3/GCS)
        └── Regional Edge Servers (20 regions)
            └── Edge PoPs (1000+ locations)
                └── Users
        ```

        **Benefits:**

        - **95% cache hit rate** at edge PoPs (popular songs)
        - **5% cache miss** falls back to regional servers
        - **<1% origin requests** (only for long-tail content)

        ---

        ## Adaptive Bitrate Streaming

        ```python
        class AdaptiveBitrateSelector:
            """
            Select audio quality based on network conditions

            Quality levels:
            - Low: 96 kbps (Ogg Vorbis)
            - Medium: 160 kbps (AAC)
            - High: 320 kbps (MP3)
            """

            QUALITY_LEVELS = {
                'low': {'bitrate': 96, 'format': 'ogg', 'min_bandwidth_kbps': 128},
                'medium': {'bitrate': 160, 'format': 'aac', 'min_bandwidth_kbps': 256},
                'high': {'bitrate': 320, 'format': 'mp3', 'min_bandwidth_kbps': 512}
            }

            def __init__(self):
                self.bandwidth_history = []
                self.current_quality = 'medium'

            def select_quality(self, measured_bandwidth_kbps: int,
                             buffer_health_percent: float,
                             user_preference: str = 'auto') -> str:
                """
                Select optimal audio quality

                Args:
                    measured_bandwidth_kbps: Current network bandwidth
                    buffer_health_percent: Audio buffer fill (0-100)
                    user_preference: 'auto', 'low', 'medium', 'high'

                Returns:
                    Quality level: 'low', 'medium', 'high'
                """
                # User manually selected quality
                if user_preference != 'auto':
                    return user_preference

                # Track bandwidth history (last 10 measurements)
                self.bandwidth_history.append(measured_bandwidth_kbps)
                if len(self.bandwidth_history) > 10:
                    self.bandwidth_history.pop(0)

                # Use 25th percentile (conservative estimate)
                bandwidth_p25 = sorted(self.bandwidth_history)[len(self.bandwidth_history) // 4]

                # Select quality with 2x safety margin
                if bandwidth_p25 >= 512 * 2 and buffer_health_percent > 50:
                    return 'high'
                elif bandwidth_p25 >= 256 * 2 and buffer_health_percent > 30:
                    return 'medium'
                else:
                    return 'low'

            def should_switch_quality(self, new_quality: str) -> bool:
                """
                Decide if quality switch is beneficial

                Avoid thrashing between qualities
                """
                if new_quality == self.current_quality:
                    return False

                # Hysteresis: require 3 consecutive measurements before switching
                # (prevents rapid quality changes)
                if len(self.bandwidth_history) < 3:
                    return False

                self.current_quality = new_quality
                return True


        class AudioStreamClient:
            """Client-side audio streaming with adaptive bitrate"""

            def __init__(self, song_id: str):
                self.song_id = song_id
                self.quality_selector = AdaptiveBitrateSelector()
                self.buffer = AudioBuffer(target_seconds=30)
                self.current_cdn_url = None

            async def start_playback(self):
                """Initialize playback"""
                # Get initial stream URL
                metadata = await self._get_stream_metadata('medium')
                self.current_cdn_url = metadata['stream_url']

                # Start buffering
                asyncio.create_task(self._buffer_audio())

                # Start bandwidth monitoring
                asyncio.create_task(self._monitor_bandwidth())

            async def _buffer_audio(self):
                """
                Download audio chunks and fill buffer

                Uses HTTP range requests for efficiency
                """
                byte_offset = 0
                chunk_size = 256 * 1024  # 256 KB chunks

                while True:
                    # Check buffer health
                    buffer_seconds = self.buffer.get_fill_seconds()

                    if buffer_seconds >= 30:
                        # Buffer full, wait
                        await asyncio.sleep(1)
                        continue

                    # Download next chunk (HTTP range request)
                    chunk = await self._download_chunk(
                        self.current_cdn_url,
                        byte_offset,
                        byte_offset + chunk_size - 1
                    )

                    self.buffer.append(chunk)
                    byte_offset += len(chunk)

                    # Check if quality switch needed
                    await self._check_quality_switch()

            async def _download_chunk(self, url: str, start_byte: int,
                                    end_byte: int) -> bytes:
                """
                Download audio chunk using HTTP range request

                Request format: Range: bytes=0-1024
                Response: 206 Partial Content
                """
                headers = {
                    'Range': f'bytes={start_byte}-{end_byte}'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 206:  # Partial Content
                            return await response.read()
                        else:
                            raise Exception(f"Expected 206, got {response.status}")

            async def _monitor_bandwidth(self):
                """
                Monitor network bandwidth and adjust quality
                """
                while True:
                    # Measure bandwidth (bytes downloaded / time)
                    bandwidth_kbps = await self._measure_bandwidth()

                    # Select optimal quality
                    buffer_health = self.buffer.get_fill_percent()
                    new_quality = self.quality_selector.select_quality(
                        bandwidth_kbps,
                        buffer_health
                    )

                    # Switch quality if needed
                    if self.quality_selector.should_switch_quality(new_quality):
                        await self._switch_quality(new_quality)

                    await asyncio.sleep(5)  # Check every 5 seconds

            async def _switch_quality(self, new_quality: str):
                """
                Switch to different quality level

                Seamlessly transition without interruption
                """
                logger.info(f"Switching quality to {new_quality}")

                # Get new stream URL
                metadata = await self._get_stream_metadata(new_quality)

                # Calculate current position in song
                current_position_ms = self.buffer.get_playback_position_ms()

                # Get new URL with position offset
                self.current_cdn_url = metadata['stream_url']

                # Continue buffering from current position
                # (buffer will seamlessly transition to new quality)
        ```

        ---

        ## CDN Optimization Strategies

        **1. Pre-warming Cache:**

        ```python
        class CDNPreWarmer:
            """
            Pre-warm CDN cache with predicted popular songs

            Strategy:
            - New releases (album drops)
            - Viral songs (trending on TikTok)
            - Regional hits (morning playlists)
            """

            def pre_warm_new_releases(self, album_id: str, cdn_regions: List[str]):
                """
                Pre-load new album to CDN before release

                Prevents origin overload on release day
                """
                songs = self.catalog_db.get_album_songs(album_id)

                for song in songs:
                    audio_url = song['audio_files']['mp3_320']

                    for region in cdn_regions:
                        # Send request to CDN edge to cache file
                        self._cdn_pre_warm_request(
                            cdn_region=region,
                            origin_url=audio_url
                        )

                logger.info(f"Pre-warmed {len(songs)} songs in {len(cdn_regions)} regions")

            def pre_warm_morning_playlists(self):
                """
                Pre-warm popular morning playlists by timezone

                6 AM Tokyo -> 6 AM Sydney -> 6 AM Mumbai -> ...
                """
                current_hour_utc = datetime.utcnow().hour

                # Find regions where it's 6-9 AM
                regions = self._get_regions_by_hour(target_hours=[6, 7, 8, 9])

                for region in regions:
                    # Get top playlists for this region
                    playlists = self.analytics.get_top_morning_playlists(
                        region=region,
                        limit=20
                    )

                    # Pre-warm songs in these playlists
                    for playlist in playlists:
                        songs = playlist['songs'][:50]  # First 50 songs
                        self._pre_warm_songs(songs, region)
        ```

        **2. Intelligent Caching:**

        | Song Tier | Cache Duration | Storage Tier | Example |
        |-----------|---------------|--------------|---------|
        | **Hot** (top 1%) | 30 days | SSD | Current hits, new releases |
        | **Warm** (top 10%) | 7 days | SSD | Recent popular songs |
        | **Cold** (remaining) | 24 hours | HDD | Long-tail catalog |

        **Benefits:**

        - **95% cache hit** for hot songs (majority of streams)
        - **Cost optimization** (SSD for hot, HDD for cold)
        - **Origin protection** (only 5% of requests reach origin)

        ---

        ## Audio Format Selection

        **Multi-format strategy:**

        | Format | Bitrate | Use Case | File Size (3 min song) |
        |--------|---------|----------|----------------------|
        | **Ogg Vorbis** | 96 kbps | Low bandwidth (mobile data) | 2.1 MB |
        | **AAC** | 160 kbps | Balanced (default) | 3.6 MB |
        | **MP3** | 320 kbps | High quality (premium/WiFi) | 7.2 MB |

        **Why multiple formats:**

        - **Ogg Vorbis:** Better compression than MP3 at low bitrates
        - **AAC:** Standard for mobile platforms (iOS native)
        - **MP3:** Universal compatibility, high quality

    === "🤖 Recommendation System"

        ## The Challenge

        **Problem:** Generate personalized recommendations for 500M users across 100M songs with diverse tastes.

        **Requirements:**

        - **Personalized:** Different recommendations per user
        - **Diverse:** Mix of familiar and discovery
        - **Real-time-ish:** Update daily (not instant)
        - **Scalable:** Handle 500M users, 100M songs

        ---

        ## Recommendation Architecture

        **Multi-stage pipeline:**

        ```
        Stage 1: Candidate Generation (100M songs → 1,000 candidates)
        └── Collaborative filtering
        └── Content-based filtering
        └── Trending/popular songs

        Stage 2: Ranking (1,000 → 50 recommendations)
        └── Deep learning model
        └── Diversity re-ranking

        Stage 3: Serving (50 recommendations)
        └── Redis cache (pre-computed)
        └── Real-time fallback
        ```

        ---

        ## Collaborative Filtering

        ```python
        import numpy as np
        from scipy.sparse import csr_matrix
        from implicit.als import AlternatingLeastSquares

        class CollaborativeFilteringRecommender:
            """
            Collaborative filtering using matrix factorization (ALS)

            Approach:
            - User-item matrix (500M users × 100M songs)
            - Factorize into user embeddings and song embeddings
            - Recommend songs with high user-song similarity
            """

            def __init__(self, embedding_dim: int = 256):
                self.embedding_dim = embedding_dim
                self.model = AlternatingLeastSquares(
                    factors=embedding_dim,
                    regularization=0.01,
                    iterations=15,
                    use_gpu=True
                )
                self.user_embeddings = None
                self.song_embeddings = None

            def train(self, play_history: List[dict]):
                """
                Train collaborative filtering model

                Args:
                    play_history: List of {user_id, song_id, play_count}

                Creates:
                    User-item sparse matrix (500M × 100M)
                    Too large for memory -> use implicit library (optimized for sparse)
                """
                logger.info("Building user-item matrix...")

                # Map IDs to indices
                user_ids = sorted(set(p['user_id'] for p in play_history))
                song_ids = sorted(set(p['song_id'] for p in play_history))

                user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
                song_id_to_idx = {sid: idx for idx, sid in enumerate(song_ids)}

                # Build sparse matrix
                row_indices = []
                col_indices = []
                data = []

                for play in play_history:
                    user_idx = user_id_to_idx[play['user_id']]
                    song_idx = song_id_to_idx[play['song_id']]
                    # Weight by play count (log scale to reduce bias)
                    weight = np.log1p(play['play_count'])

                    row_indices.append(user_idx)
                    col_indices.append(song_idx)
                    data.append(weight)

                user_item_matrix = csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(len(user_ids), len(song_ids))
                )

                logger.info(f"Matrix shape: {user_item_matrix.shape}")
                logger.info(f"Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.6f}")

                # Train ALS model
                logger.info("Training ALS model...")
                self.model.fit(user_item_matrix.T)  # Transpose for implicit library

                # Extract embeddings
                self.user_embeddings = self.model.user_factors
                self.song_embeddings = self.model.item_factors

                logger.info("Training complete!")

            def get_recommendations(self, user_id: str, n: int = 1000) -> List[str]:
                """
                Get top N song recommendations for user

                Args:
                    user_id: User ID
                    n: Number of recommendations

                Returns:
                    List of song IDs, ranked by predicted score
                """
                user_idx = self.user_id_to_idx.get(user_id)
                if user_idx is None:
                    # New user, return popular songs
                    return self._get_popular_songs(n)

                # Get user embedding
                user_emb = self.user_embeddings[user_idx]

                # Compute scores for all songs
                # scores = user_emb · song_emb^T
                scores = self.song_embeddings.dot(user_emb)

                # Get top N songs (excluding already played)
                top_song_indices = np.argsort(scores)[::-1][:n*2]  # Get 2x for filtering

                # Filter out already played songs
                played_songs = set(self._get_user_played_songs(user_id))
                recommendations = []

                for idx in top_song_indices:
                    song_id = self.idx_to_song_id[idx]
                    if song_id not in played_songs:
                        recommendations.append(song_id)
                        if len(recommendations) >= n:
                            break

                return recommendations

            def get_similar_songs(self, song_id: str, n: int = 50) -> List[str]:
                """
                Get similar songs (for "More like this")

                Args:
                    song_id: Seed song ID
                    n: Number of similar songs

                Returns:
                    List of similar song IDs
                """
                song_idx = self.song_id_to_idx.get(song_id)
                if song_idx is None:
                    return []

                # Get song embedding
                song_emb = self.song_embeddings[song_idx]

                # Compute cosine similarity with all songs
                # similarity = (song_emb · other_emb) / (||song_emb|| × ||other_emb||)
                similarities = self.song_embeddings.dot(song_emb) / (
                    np.linalg.norm(self.song_embeddings, axis=1) * np.linalg.norm(song_emb)
                )

                # Get top N similar songs (excluding self)
                top_indices = np.argsort(similarities)[::-1][1:n+1]

                return [self.idx_to_song_id[idx] for idx in top_indices]
        ```

        ---

        ## Deep Learning Ranking Model

        ```python
        import torch
        import torch.nn as nn

        class SongRankingModel(nn.Module):
            """
            Deep learning model for ranking song candidates

            Input features:
            - User features: age, country, listening history, preferences
            - Song features: genre, artist, tempo, acousticness, energy
            - Context features: time of day, day of week, device
            - Interaction features: user-song similarity, popularity

            Output:
            - Predicted play probability (0-1)
            """

            def __init__(self, user_emb_dim=256, song_emb_dim=256, hidden_dim=512):
                super().__init__()

                # User embedding tower
                self.user_tower = nn.Sequential(
                    nn.Linear(user_emb_dim + 50, hidden_dim),  # +50 for user features
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, 128)
                )

                # Song embedding tower
                self.song_tower = nn.Sequential(
                    nn.Linear(song_emb_dim + 30, hidden_dim),  # +30 for song features
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, 128)
                )

                # Interaction layer
                self.interaction = nn.Sequential(
                    nn.Linear(128 + 128 + 20, 256),  # User + Song + Context
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()  # Output probability
                )

            def forward(self, user_features, song_features, context_features):
                """
                Forward pass

                Args:
                    user_features: [batch_size, user_emb_dim + user_feat_dim]
                    song_features: [batch_size, song_emb_dim + song_feat_dim]
                    context_features: [batch_size, context_dim]

                Returns:
                    play_probability: [batch_size, 1]
                """
                # Process user and song through their towers
                user_repr = self.user_tower(user_features)
                song_repr = self.song_tower(song_features)

                # Concatenate representations with context
                combined = torch.cat([user_repr, song_repr, context_features], dim=1)

                # Predict play probability
                play_prob = self.interaction(combined)

                return play_prob


        class RecommendationRanker:
            """Re-rank candidates using deep learning model"""

            def __init__(self, model_path: str):
                self.model = SongRankingModel()
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()

            def rank_candidates(self, user_id: str, candidate_songs: List[str],
                              context: dict) -> List[str]:
                """
                Rank candidate songs for user

                Args:
                    user_id: User ID
                    candidate_songs: List of candidate song IDs (e.g., 1000 songs)
                    context: Context dict {time_of_day, device, etc.}

                Returns:
                    Ranked list of song IDs
                """
                # Get user features
                user_features = self._get_user_features(user_id)

                # Get song features for all candidates
                songs_features = [self._get_song_features(sid) for sid in candidate_songs]

                # Get context features
                context_features = self._get_context_features(context)

                # Batch inference
                with torch.no_grad():
                    user_tensor = torch.tensor([user_features] * len(candidate_songs))
                    songs_tensor = torch.tensor(songs_features)
                    context_tensor = torch.tensor([context_features] * len(candidate_songs))

                    scores = self.model(user_tensor, songs_tensor, context_tensor)
                    scores = scores.squeeze().numpy()

                # Sort by score (descending)
                ranked_indices = np.argsort(scores)[::-1]
                ranked_songs = [candidate_songs[i] for i in ranked_indices]

                return ranked_songs
        ```

        ---

        ## Diversity Re-ranking

        **Problem:** Top recommendations often too similar (same genre/artist).

        **Solution:** Maximal Marginal Relevance (MMR)

        ```python
        def diversify_recommendations(ranked_songs: List[str],
                                     lambda_param: float = 0.7,
                                     final_count: int = 50) -> List[str]:
            """
            Re-rank songs to balance relevance and diversity

            MMR formula:
            MMR = λ × Relevance(song) - (1-λ) × max(Similarity(song, selected))

            Args:
                ranked_songs: Songs ranked by relevance
                lambda_param: Balance between relevance (1.0) and diversity (0.0)
                final_count: Number of final recommendations

            Returns:
                Diversified recommendations
            """
            selected = []
            candidates = ranked_songs.copy()

            # First song: highest relevance
            selected.append(candidates.pop(0))

            while len(selected) < final_count and candidates:
                mmr_scores = []

                for candidate in candidates:
                    # Relevance: position in ranked list (higher = better)
                    relevance = 1.0 / (candidates.index(candidate) + 1)

                    # Diversity: max similarity to already selected songs
                    max_similarity = max(
                        get_song_similarity(candidate, s) for s in selected
                    )

                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    mmr_scores.append((candidate, mmr))

                # Select song with highest MMR
                best_candidate = max(mmr_scores, key=lambda x: x[1])[0]
                selected.append(best_candidate)
                candidates.remove(best_candidate)

            return selected
        ```

        ---

        ## Batch Processing Pipeline

        **Daily recommendation generation:**

        ```python
        class RecommendationPipeline:
            """
            Batch generate recommendations for all users

            Pipeline:
            1. Load latest play history (yesterday's data)
            2. Update user/song embeddings (incremental training)
            3. Generate candidates (collaborative filtering)
            4. Rank candidates (deep learning model)
            5. Diversify (MMR)
            6. Cache in Redis (TTL: 24 hours)
            """

            def run_daily_pipeline(self):
                """Run full recommendation pipeline"""
                logger.info("Starting daily recommendation pipeline")

                # 1. Load data from ClickHouse
                play_history = self.analytics.get_play_history(
                    start_date=date.today() - timedelta(days=90),
                    end_date=date.today()
                )
                logger.info(f"Loaded {len(play_history)} play events")

                # 2. Train/update collaborative filtering model
                logger.info("Training collaborative filtering model")
                self.cf_model.train(play_history)

                # 3. Generate recommendations for all active users (parallel)
                logger.info("Generating recommendations for 200M active users")

                active_users = self.user_db.get_active_users(days=30)

                # Process in batches of 10K users
                batch_size = 10000
                for i in range(0, len(active_users), batch_size):
                    batch = active_users[i:i+batch_size]

                    # Parallel processing
                    with ThreadPoolExecutor(max_workers=50) as executor:
                        futures = [
                            executor.submit(self._generate_user_recommendations, user_id)
                            for user_id in batch
                        ]

                        for future in as_completed(futures):
                            try:
                                user_id, recommendations = future.result()
                                # Cache in Redis
                                self.cache.setex(
                                    f"rec:{user_id}",
                                    86400,  # 24 hours
                                    json.dumps(recommendations)
                                )
                            except Exception as e:
                                logger.error(f"Failed to generate recommendations: {e}")

                logger.info("Pipeline complete!")

            def _generate_user_recommendations(self, user_id: str) -> Tuple[str, List[str]]:
                """Generate recommendations for single user"""
                # Stage 1: Candidate generation (1000 songs)
                candidates = self.cf_model.get_recommendations(user_id, n=1000)

                # Stage 2: Ranking (deep learning)
                ranked = self.ranker.rank_candidates(
                    user_id,
                    candidates,
                    context={'time_of_day': 'evening'}  # Default context
                )

                # Stage 3: Diversify (50 final recommendations)
                diversified = diversify_recommendations(ranked[:200], final_count=50)

                return user_id, diversified
        ```

    === "📥 Offline Sync"

        ## The Challenge

        **Problem:** Sync up to 10,000 downloaded songs per user across multiple devices with encrypted storage.

        **Requirements:**

        - **Premium only:** Offline downloads for paying users
        - **Limit enforcement:** Max 10,000 songs per account
        - **License management:** DRM, expires after 30 days offline
        - **Efficient sync:** Differential sync (only changed songs)
        - **Cross-device:** Sync across mobile, desktop, tablet

        ---

        ## Offline Architecture

        ```
        Device 1 (iPhone)
        ├── Local SQLite DB (metadata)
        ├── Encrypted audio files
        └── Sync Service

        Cloud Sync State (Cassandra)
        ├── User's download manifest
        ├── Device states
        └── License expiry

        Device 2 (Android)
        ├── Local SQLite DB (metadata)
        ├── Encrypted audio files
        └── Sync Service
        ```

        ---

        ## Differential Sync Protocol

        ```python
        class OfflineSyncService:
            """
            Manage offline downloads and sync across devices

            Sync protocol:
            1. Device sends local manifest (song_ids + versions)
            2. Server compares with cloud manifest
            3. Server responds with delta (add/remove/update)
            4. Device applies delta
            """

            def __init__(self, user_db, music_store):
                self.user_db = user_db
                self.music_store = music_store

            def sync_device(self, user_id: str, device_id: str,
                          local_manifest: dict) -> dict:
                """
                Sync device with cloud state

                Args:
                    user_id: User ID
                    device_id: Device ID
                    local_manifest: {song_id: version, ...}

                Returns:
                    Sync delta: {
                        'to_download': [song_ids],
                        'to_remove': [song_ids],
                        'to_update': [song_ids]
                    }
                """
                # Check subscription
                if not self._is_premium_user(user_id):
                    return {'error': 'Premium subscription required'}

                # Get cloud manifest
                cloud_manifest = self.user_db.get_offline_manifest(user_id)

                # Compare manifests
                delta = self._compute_delta(local_manifest, cloud_manifest)

                # Enforce download limit
                total_songs = len(cloud_manifest) + len(delta['to_download']) - len(delta['to_remove'])
                if total_songs > 10000:
                    delta['to_download'] = delta['to_download'][:10000 - len(cloud_manifest)]
                    delta['error'] = 'Download limit reached (10,000 songs max)'

                # Update device state
                self.user_db.update_device_state(
                    user_id,
                    device_id,
                    {
                        'last_sync': datetime.utcnow(),
                        'song_count': len(local_manifest)
                    }
                )

                return delta

            def _compute_delta(self, local: dict, cloud: dict) -> dict:
                """
                Compute sync delta

                Args:
                    local: Local manifest {song_id: version}
                    cloud: Cloud manifest {song_id: version}

                Returns:
                    Delta dict
                """
                local_songs = set(local.keys())
                cloud_songs = set(cloud.keys())

                # Songs to download (in cloud, not local)
                to_download = list(cloud_songs - local_songs)

                # Songs to remove (in local, not cloud)
                to_remove = list(local_songs - cloud_songs)

                # Songs to update (version mismatch)
                to_update = [
                    song_id for song_id in (local_songs & cloud_songs)
                    if local[song_id] != cloud[song_id]
                ]

                return {
                    'to_download': to_download,
                    'to_remove': to_remove,
                    'to_update': to_update
                }

            def add_to_offline(self, user_id: str, song_ids: List[str]) -> dict:
                """
                Add songs to offline library

                Args:
                    user_id: User ID
                    song_ids: Songs to add

                Returns:
                    Download URLs with licenses
                """
                # Check premium
                if not self._is_premium_user(user_id):
                    raise PermissionError("Premium required")

                # Get current manifest
                manifest = self.user_db.get_offline_manifest(user_id)

                # Check limit
                if len(manifest) + len(song_ids) > 10000:
                    raise ValueError("Download limit exceeded (10,000 max)")

                # Generate download URLs with DRM licenses
                download_info = []
                for song_id in song_ids:
                    # Get audio file URL
                    audio_url = self.music_store.get_download_url(
                        song_id,
                        quality='high',
                        expires_in=3600  # 1 hour
                    )

                    # Generate license
                    license = self._generate_license(user_id, song_id)

                    download_info.append({
                        'song_id': song_id,
                        'download_url': audio_url,
                        'license': license,
                        'version': 1
                    })

                    # Update manifest
                    manifest[song_id] = 1

                # Save updated manifest
                self.user_db.update_offline_manifest(user_id, manifest)

                return {'downloads': download_info}

            def _generate_license(self, user_id: str, song_id: str) -> dict:
                """
                Generate DRM license for offline playback

                License allows:
                - Offline playback up to 30 days
                - Decrypt audio file
                - Verify user ownership
                """
                license = {
                    'user_id': user_id,
                    'song_id': song_id,
                    'issued_at': datetime.utcnow().isoformat(),
                    'expires_at': (datetime.utcnow() + timedelta(days=30)).isoformat(),
                    'encryption_key': self._get_encryption_key(song_id),
                    'signature': self._sign_license(user_id, song_id)
                }

                return license

            def verify_offline_playback(self, user_id: str, song_id: str,
                                      license: dict) -> bool:
                """
                Verify license for offline playback

                Returns:
                    True if playback allowed, False otherwise
                """
                # Verify signature
                if not self._verify_signature(license):
                    return False

                # Check expiry
                expires_at = datetime.fromisoformat(license['expires_at'])
                if datetime.utcnow() > expires_at:
                    return False

                # Verify user owns song
                manifest = self.user_db.get_offline_manifest(user_id)
                if song_id not in manifest:
                    return False

                return True
        ```

        ---

        ## Priority Queue Downloads

        **Challenge:** User adds 1,000 songs to offline. Download order matters!

        ```python
        class DownloadPriorityQueue:
            """
            Prioritize downloads for best UX

            Priority factors:
            1. Recently played songs (high priority)
            2. Songs in active playlists (high priority)
            3. Older songs in library (low priority)
            4. File size (smaller files first for quick wins)
            """

            def prioritize_downloads(self, user_id: str,
                                   songs_to_download: List[str]) -> List[str]:
                """
                Order downloads by priority

                Args:
                    user_id: User ID
                    songs_to_download: Unordered list of song IDs

                Returns:
                    Ordered list (highest priority first)
                """
                scores = []

                # Get user context
                recent_plays = self._get_recent_plays(user_id, days=7)
                active_playlists = self._get_active_playlists(user_id)

                for song_id in songs_to_download:
                    score = 0

                    # Recently played: +100 per play
                    play_count = recent_plays.get(song_id, 0)
                    score += play_count * 100

                    # In active playlist: +50
                    if self._is_in_active_playlist(song_id, active_playlists):
                        score += 50

                    # File size penalty (smaller files first)
                    file_size_mb = self._get_file_size(song_id)
                    score -= file_size_mb  # Smaller = higher score

                    # Popularity boost (top 100 songs: +20)
                    if self._is_popular(song_id):
                        score += 20

                    scores.append((song_id, score))

                # Sort by score (descending)
                scores.sort(key=lambda x: x[1], reverse=True)

                return [song_id for song_id, _ in scores]
        ```

        ---

        ## Storage Encryption

        **DRM protection:**

        ```python
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

        class AudioEncryption:
            """Encrypt audio files for offline storage"""

            def encrypt_audio_file(self, audio_data: bytes, song_id: str,
                                 user_secret: str) -> bytes:
                """
                Encrypt audio file for local storage

                Args:
                    audio_data: Raw audio bytes
                    song_id: Song ID
                    user_secret: User-specific secret (from license)

                Returns:
                    Encrypted audio bytes
                """
                # Derive encryption key from user secret + song ID
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=song_id.encode(),
                    iterations=100000
                )
                key = kdf.derive(user_secret.encode())

                # Encrypt
                f = Fernet(key)
                encrypted = f.encrypt(audio_data)

                return encrypted

            def decrypt_audio_file(self, encrypted_data: bytes, song_id: str,
                                 user_secret: str) -> bytes:
                """
                Decrypt audio file for playback

                Requires valid license (user_secret)
                """
                # Derive decryption key
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=song_id.encode(),
                    iterations=100000
                )
                key = kdf.derive(user_secret.encode())

                # Decrypt
                f = Fernet(key)
                decrypted = f.decrypt(encrypted_data)

                return decrypted
        ```

    === "🔍 Search & Discovery"

        ## The Challenge

        **Problem:** Search 100M songs in <100ms with typo tolerance, autocomplete, and smart ranking.

        **Requirements:**

        - **Fast:** < 100ms p95 latency
        - **Fuzzy matching:** Handle typos ("bohemain rapsodie")
        - **Autocomplete:** Real-time suggestions
        - **Smart ranking:** Popularity + relevance

        ---

        ## Elasticsearch Index Design

        ```python
        class MusicSearchIndex:
            """Elasticsearch index for music search"""

            SONG_INDEX_MAPPING = {
                "settings": {
                    "number_of_shards": 20,  # Horizontal scaling
                    "number_of_replicas": 2,  # High availability
                    "analysis": {
                        "analyzer": {
                            "song_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "asciifolding",  # Remove accents
                                    "song_synonym"
                                ]
                            },
                            "autocomplete_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "autocomplete_filter"
                                ]
                            }
                        },
                        "filter": {
                            "autocomplete_filter": {
                                "type": "edge_ngram",
                                "min_gram": 2,
                                "max_gram": 20
                            },
                            "song_synonym": {
                                "type": "synonym",
                                "synonyms": [
                                    "ft, feat, featuring",
                                    "&, and"
                                ]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "song_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "song_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "autocomplete": {
                                    "type": "text",
                                    "analyzer": "autocomplete_analyzer"
                                }
                            }
                        },
                        "artist_name": {
                            "type": "text",
                            "analyzer": "song_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "autocomplete": {
                                    "type": "text",
                                    "analyzer": "autocomplete_analyzer"
                                }
                            }
                        },
                        "album_name": {
                            "type": "text",
                            "analyzer": "song_analyzer"
                        },
                        "genre": {"type": "keyword"},
                        "release_year": {"type": "integer"},
                        "duration_ms": {"type": "integer"},
                        "explicit": {"type": "boolean"},

                        # Ranking signals
                        "popularity_score": {"type": "float"},  # 0-100
                        "play_count_30d": {"type": "long"},    # Last 30 days
                        "like_count": {"type": "integer"},

                        # Full-text content
                        "lyrics": {"type": "text", "analyzer": "song_analyzer"}
                    }
                }
            }

            def create_index(self):
                """Create Elasticsearch index"""
                self.es.indices.create(
                    index='songs',
                    body=self.SONG_INDEX_MAPPING
                )
        ```

        ---

        ## Search Query Implementation

        ```python
        class MusicSearchService:
            """Search service with ranking and autocomplete"""

            def search(self, query: str, filters: dict = None,
                      limit: int = 20) -> List[dict]:
                """
                Search songs with smart ranking

                Args:
                    query: Search query ("bohemian rhapsody")
                    filters: Optional filters {genre, year, explicit}
                    limit: Number of results

                Returns:
                    Ranked search results
                """
                # Build Elasticsearch query
                must_clauses = []
                filter_clauses = []

                # Full-text search with fuzzy matching
                if query:
                    must_clauses.append({
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",        # Boost title 3x
                                "artist_name^2",  # Boost artist 2x
                                "album_name",
                                "lyrics"
                            ],
                            "type": "best_fields",
                            "fuzziness": "AUTO",  # Handle typos
                            "operator": "or"
                        }
                    })

                # Apply filters
                if filters:
                    if 'genre' in filters:
                        filter_clauses.append({"term": {"genre": filters['genre']}})

                    if 'year_min' in filters:
                        filter_clauses.append({"range": {"release_year": {"gte": filters['year_min']}}})

                    if 'explicit' in filters:
                        filter_clauses.append({"term": {"explicit": filters['explicit']}})

                # Construct query with function score
                search_body = {
                    "query": {
                        "function_score": {
                            "query": {
                                "bool": {
                                    "must": must_clauses,
                                    "filter": filter_clauses
                                }
                            },
                            "functions": [
                                # Boost by popularity (logarithmic)
                                {
                                    "field_value_factor": {
                                        "field": "popularity_score",
                                        "modifier": "log1p",
                                        "factor": 1.2
                                    },
                                    "weight": 2
                                },
                                # Boost recent plays
                                {
                                    "field_value_factor": {
                                        "field": "play_count_30d",
                                        "modifier": "log1p",
                                        "factor": 0.5
                                    },
                                    "weight": 1
                                },
                                # Recency boost (newer songs slightly preferred)
                                {
                                    "gauss": {
                                        "release_year": {
                                            "origin": 2026,
                                            "scale": "10y",
                                            "decay": 0.5
                                        }
                                    },
                                    "weight": 0.5
                                }
                            ],
                            "score_mode": "sum",
                            "boost_mode": "multiply"
                        }
                    },
                    "size": limit
                }

                # Execute search
                response = self.es.search(index='songs', body=search_body)

                # Extract results
                results = []
                for hit in response['hits']['hits']:
                    song = hit['_source']
                    song['relevance_score'] = hit['_score']
                    results.append(song)

                return results

            def autocomplete(self, prefix: str, limit: int = 10) -> List[dict]:
                """
                Autocomplete suggestions

                Args:
                    prefix: Partial query ("boh")
                    limit: Max suggestions

                Returns:
                    Autocomplete suggestions
                """
                search_body = {
                    "query": {
                        "multi_match": {
                            "query": prefix,
                            "fields": [
                                "title.autocomplete^3",
                                "artist_name.autocomplete^2"
                            ],
                            "type": "phrase_prefix"
                        }
                    },
                    "size": limit,
                    "_source": ["song_id", "title", "artist_name"]
                }

                response = self.es.search(index='songs', body=search_body)

                suggestions = []
                for hit in response['hits']['hits']:
                    suggestions.append({
                        'song_id': hit['_source']['song_id'],
                        'display': f"{hit['_source']['title']} - {hit['_source']['artist_name']}"
                    })

                return suggestions
        ```

        ---

        ## Real-time Index Updates

        **Challenge:** Keep search index fresh (new songs indexed within 5 minutes).

        ```python
        class SearchIndexer:
            """Index songs in Elasticsearch in real-time"""

            def __init__(self, es_client, kafka_consumer):
                self.es = es_client
                self.consumer = kafka_consumer

            def run_indexing_pipeline(self):
                """
                Consume song events from Kafka and index

                Events:
                - song_created: New song added to catalog
                - song_updated: Metadata changed
                - song_stats_updated: Play counts, popularity changed
                """
                while True:
                    messages = self.consumer.poll(timeout_ms=1000)

                    batch = []
                    for topic_partition, records in messages.items():
                        for record in records:
                            event = json.loads(record.value)

                            if event['type'] == 'song_created':
                                batch.append(self._create_index_action(event['song']))
                            elif event['type'] == 'song_updated':
                                batch.append(self._update_index_action(event['song']))
                            elif event['type'] == 'song_stats_updated':
                                batch.append(self._update_stats_action(event))

                    # Bulk index (efficient)
                    if batch:
                        helpers.bulk(self.es, batch)
                        logger.info(f"Indexed {len(batch)} songs")

            def _create_index_action(self, song: dict) -> dict:
                """Create index action for new song"""
                return {
                    '_op_type': 'index',
                    '_index': 'songs',
                    '_id': song['song_id'],
                    '_source': {
                        'song_id': song['song_id'],
                        'title': song['title'],
                        'artist_name': song['artist_name'],
                        'album_name': song['album_name'],
                        'genre': song['genre'],
                        'release_year': song['release_year'],
                        'duration_ms': song['duration_ms'],
                        'explicit': song['explicit'],
                        'popularity_score': 0,
                        'play_count_30d': 0,
                        'like_count': 0
                    }
                }
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling Spotify from 1M to 500M users.

    **Scaling challenges at 500M users:**

    - **Concurrent streams:** 70M concurrent users
    - **Bandwidth:** 11.6 Tbps CDN egress
    - **Storage:** 7.4 PB total (3 PB music, 4.38 PB history)
    - **Cache:** 1.36 TB Redis + 30 TB CDN audio cache

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **CDN bandwidth** | ✅ Yes | 1000+ PoPs, 95% cache hit rate, pre-warming |
    | **Cassandra writes** | ✅ Yes | Shard across 200 nodes, SSD storage, batch writes |
    | **Elasticsearch queries** | ✅ Yes | 20 shards, 2 replicas, query cache, result pagination |
    | **ClickHouse analytics** | 🟡 Approaching | Partition by month, materialized views, columnar compression |
    | **Redis cache** | 🟡 Approaching | Cluster mode (100 nodes), read replicas, cache warming |

    ---

    ## Caching Strategy

    **Multi-tier caching:**

    | Tier | Data | TTL | Size | Hit Rate |
    |------|------|-----|------|----------|
    | **CDN (Edge)** | Audio files (hot songs) | 30 days | 30 TB | 95% |
    | **Redis** | Metadata, recommendations | 1 hour - 24 hours | 1.36 TB | 80% |
    | **Database** | Full catalog, user data | Permanent | 7.4 PB | - |

    **Cache warming strategies:**

    1. **New releases:** Pre-warm CDN before album drop
    2. **Regional hits:** Pre-warm based on timezone (morning playlists)
    3. **Viral songs:** Detect trending and pre-warm
    4. **Recommendations:** Batch generate daily, cache 24h

    ---

    ## Database Sharding

    **Cassandra sharding (User DB):**

    ```
    500M users → 200 Cassandra nodes
    └── Shard by user_id hash
        ├── Node 1: users 0 - 2.5M
        ├── Node 2: users 2.5M - 5M
        └── ...

    Replication factor: 3 (for 99.9% availability)
    ```

    **PostgreSQL sharding (Catalog DB):**

    ```
    100M songs → 20 PostgreSQL shards
    └── Shard by song_id hash
        ├── Shard 1: songs 0 - 5M
        ├── Shard 2: songs 5M - 10M
        └── ...

    Read replicas: 3 per shard (read-heavy workload)
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 500M users:**

    | Component | Cost |
    |-----------|------|
    | **CDN (11.6 Tbps egress)** | $464,000 (1,500 TB/day @ $0.01/GB) |
    | **EC2 (API servers)** | $108,000 (750 × m5.2xlarge) |
    | **Cassandra cluster** | $86,400 (200 nodes × r5.2xlarge) |
    | **PostgreSQL cluster** | $43,200 (60 nodes × r5.xlarge) |
    | **ClickHouse cluster** | $64,800 (100 nodes × r5.2xlarge) |
    | **Elasticsearch cluster** | $86,400 (150 nodes × r5.xlarge) |
    | **Redis cache** | $43,200 (100 nodes × r5.large) |
    | **S3 storage (3 PB music)** | $69,120 (3 PB @ $0.023/GB) |
    | **S3 storage (4.38 PB history)** | $23,040 (Glacier @ $0.004/GB) |
    | **Kafka cluster** | $21,600 (40 brokers × r5.xlarge) |
    | **Total** | **$1,009,760/month** |

    **Cost optimization strategies:**

    1. **CDN:** Negotiate volume discounts, use cheaper regional PoPs
    2. **Storage:** Archive old play history to Glacier (10x cheaper)
    3. **Compute:** Use spot instances for batch jobs (70% savings)
    4. **Compression:** Ogg Vorbis reduces bandwidth 20% vs MP3

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Playback Start Latency (P95)** | < 200ms | > 500ms |
    | **Search Latency (P95)** | < 100ms | > 300ms |
    | **CDN Cache Hit Rate** | > 95% | < 90% |
    | **Audio Quality (Bitrate)** | 160 kbps avg | < 128 kbps |
    | **Concurrent Streams** | 70M peak | > 80M (capacity limit) |
    | **Cassandra Write Latency** | < 10ms | > 50ms |
    | **Elasticsearch Query Latency** | < 50ms | > 200ms |
    | **Recommendation Cache Hit** | > 80% | < 70% |

    **Alerting strategy:**

    - **P0 (critical):** Playback failures, CDN outage, auth service down
    - **P1 (high):** High latency (>500ms), cache hit rate drop, database errors
    - **P2 (medium):** Recommendation pipeline failures, search indexing lag
    - **P3 (low):** Slow analytics queries, non-critical feature degradation

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **CDN-first architecture:** 95% cache hit rate for audio delivery
    2. **Adaptive bitrate:** Dynamic quality (96-320 kbps) based on network
    3. **Cassandra for user data:** High write throughput for playlists
    4. **ClickHouse for analytics:** Fast queries on billions of play events
    5. **ML recommendations:** Collaborative filtering + deep learning ranking
    6. **Offline sync:** Differential sync, encrypted storage, license management
    7. **Elasticsearch search:** Sub-100ms with fuzzy matching and autocomplete

    ---

    ## Interview Tips

    ✅ **Emphasize CDN strategy** - Audio delivery is the hardest challenge

    ✅ **Discuss adaptive bitrate** - Critical for user experience

    ✅ **Explain recommendation pipeline** - Batch vs real-time trade-offs

    ✅ **Cover offline complexity** - DRM, sync, storage encryption

    ✅ **Search ranking matters** - Popularity + relevance + recency

    ✅ **Cost optimization** - CDN is 45% of total cost, optimize aggressively

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle new album releases?"** | Pre-warm CDN, handle traffic spike (10x normal), ensure high availability |
    | **"How to prevent music piracy?"** | DRM encryption, signed URLs, license expiry, watermarking |
    | **"How to personalize recommendations?"** | Collaborative filtering (users like you), content-based (song features), hybrid |
    | **"How to sync offline across devices?"** | Cloud manifest, differential sync, device state tracking, conflict resolution |
    | **"How to reduce bandwidth costs?"** | Higher CDN cache hit rate, Ogg Vorbis compression, regional CDN optimization |
    | **"How to handle regional licensing?"** | Geo-blocking, license database, content availability by country |

    ---

    ## Scaling Evolution

    **Stage 1: 1M users**

    - Single region
    - Monolithic architecture
    - PostgreSQL for everything
    - Direct S3 streaming (no CDN)
    - Simple popularity recommendations

    **Stage 2: 10M users**

    - Multi-region (US, EU)
    - Microservices
    - CDN for audio delivery
    - Redis caching
    - Basic collaborative filtering

    **Stage 3: 100M users**

    - Global CDN (1000+ PoPs)
    - Cassandra for user data
    - ClickHouse for analytics
    - Deep learning recommendations
    - Offline sync

    **Stage 4: 500M users (current)**

    - All of above +
    - Advanced CDN optimization (pre-warming)
    - Real-time ML inference
    - Multi-device sync
    - Aggressive cost optimization

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Spotify, Apple Music, YouTube Music, Amazon Music, Tidal, Pandora

---

*Master this problem and you'll be ready for: YouTube, Netflix (for video), SoundCloud, Audible*
