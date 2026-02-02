# Design Live Streaming (Twitch)

A live video streaming platform where users can broadcast live content, viewers can watch streams in real-time with low latency, and users can interact through real-time chat.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M concurrent viewers, 1M concurrent streamers, 5 Tbps bandwidth, 100M chat messages/minute |
| **Key Challenges** | <3s latency, transcoding at scale, CDN optimization, chat scalability, adaptive bitrate |
| **Core Concepts** | RTMP/WebRTC ingestion, HLS/DASH delivery, transcoding pipeline, CDN strategy, pub/sub chat |
| **Companies** | Twitch, YouTube Live, Facebook Live, Instagram Live, TikTok Live, Discord |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Stream Ingestion** | Streamers broadcast video via RTMP/WebRTC | P0 (Must have) |
    | **Live Playback** | Viewers watch live streams with adaptive bitrate | P0 (Must have) |
    | **Real-time Chat** | Viewers send/receive chat messages in real-time | P0 (Must have) |
    | **Stream Discovery** | Browse/search live streams by category | P0 (Must have) |
    | **Follow/Notifications** | Users follow streamers, get notified when live | P1 (Should have) |
    | **Viewer Count** | Display concurrent viewer count | P1 (Should have) |
    | **Stream Recording (VOD)** | Save streams for later viewing | P1 (Should have) |
    | **Donations/Bits** | Viewers support streamers financially | P2 (Nice to have) |
    | **Emotes/Badges** | Custom emotes and subscriber badges | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Payment processing (assume external service)
    - Content moderation AI/ML
    - Video editing features
    - Multi-stream viewing
    - Clips generation (highlights)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Standard)** | < 3 seconds glass-to-glass | Acceptable for most viewers, enables CDN caching |
    | **Latency (Low-latency)** | < 1 second | Interactive streams (gaming, auctions), higher cost |
    | **Latency (Ultra-low)** | < 500ms (WebRTC) | Real-time interactions, video calls, limited scale |
    | **Availability** | 99.9% uptime | Streamers rely on platform for income |
    | **Scalability** | 10M concurrent viewers | Must handle viral events (esports, concerts) |
    | **Video Quality** | 1080p60fps (streamers), adaptive (viewers) | Smooth playback without buffering |
    | **Chat Throughput** | 100M messages/minute | Large streams (50K+ viewers) generate massive chat |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Users:
    - Total registered users: 500M
    - Monthly Active Users (MAU): 150M
    - Daily Active Users (DAU): 40M

    Streamers:
    - Peak concurrent streamers: 1M
    - Average stream duration: 3 hours
    - Average viewers per stream: 10 (median), 1000 (popular), 100K (top)

    Viewers:
    - Peak concurrent viewers: 10M
    - Average watch time: 2 hours/day
    - Top streams: 100K-500K concurrent viewers

    Video metrics:
    - Stream bitrate: 6 Mbps (1080p60fps source)
    - Transcoded outputs: 160p (300 Kbps), 360p (700 Kbps), 480p (1.5 Mbps),
                          720p (3 Mbps), 1080p (6 Mbps), 1080p60 (8 Mbps)
    - Segment duration: 2 seconds (HLS), 4 seconds (standard)

    Chat:
    - Average messages per viewer: 2 messages/minute (active chatters ~20%)
    - Large streams: 10-20 messages/second
    - Peak chat rate: 100M messages/minute globally

    Bandwidth:
    - Ingress (streamers): 1M streams × 6 Mbps = 6 Tbps
    - Egress (viewers): 10M viewers × avg 3 Mbps = 30 Tbps
    - Peak egress: 50 Tbps (during major events)
    - Total: 5-50 Tbps (CDN critical)
    ```

    ### Storage Estimates

    ```
    Live stream buffer (DVR):
    - Buffer last 2 hours per stream for rewind
    - 1M streams × 6 Mbps × 2 hours = 5.4 PB live buffer

    VOD storage (recordings):
    - 30% of streams saved (300K streams/day)
    - Average 3 hours per VOD
    - 300K × 3 hours × 6 Mbps = 2.43 PB/day raw
    - With compression: ~1 PB/day
    - 10 years: 1 PB × 365 × 10 = 3.65 exabytes

    Thumbnails & metadata:
    - 1M live streams × 1 MB thumbnail = 1 TB
    - Metadata: 10 TB

    Chat history:
    - 100M messages/minute × 1 KB = 100 GB/minute
    - Daily: 100 GB × 60 × 24 = 144 TB/day
    - 30-day retention: 4.32 PB
    - 10 years: 525 PB

    Total: 3.65 EB (VOD) + 10 PB (live buffer) + 525 PB (chat) ≈ 4.2 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Ingress (from streamers):
    - 1M concurrent streams × 6 Mbps = 6 Tbps
    - Peak: 2M streams = 12 Tbps

    Egress (to viewers):
    - 10M viewers × avg 3 Mbps = 30 Tbps
    - Peak (major event): 20M × 4 Mbps = 80 Tbps
    - CDN offload: 95% (CDN serves 28.5 Tbps, origin serves 1.5 Tbps)

    Transcoding data transfer:
    - 1M streams × 6 Mbps input + 20 Mbps output (all qualities) = 26 Tbps

    Total ingress: 6-12 Tbps (from streamers)
    Total egress: 30-80 Tbps (to viewers, mostly CDN)
    Internal: 26 Tbps (transcoding)
    ```

    ### Memory Estimates (Caching)

    ```
    Live stream manifests (HLS/DASH):
    - 1M streams × 100 KB manifest = 100 GB

    Thumbnails & metadata:
    - 1M live streams × 1 MB = 1 TB

    Chat messages (recent):
    - Last 1000 messages per stream
    - 1M streams × 1000 messages × 1 KB = 1 TB

    Viewer sessions:
    - 10M concurrent viewers × 10 KB session = 100 GB

    CDN edge cache:
    - Recent segments (last 30 seconds × all qualities)
    - 1M streams × 15 segments × 6 qualities × 1 MB = 90 TB per edge location
    - 200 edge locations: 18 PB total edge cache

    Total cache: 100 GB (manifests) + 1 TB (metadata) + 1 TB (chat) + 100 GB (sessions)
                + 18 PB (CDN) ≈ 18 PB
    ```

    ### Transcoding Estimates

    ```
    Transcoding requirements:
    - 1M concurrent streams need transcoding
    - Each stream: 6 quality outputs (160p, 360p, 480p, 720p, 1080p, 1080p60)
    - Real-time transcoding: 1x speed (3-second chunk in 3 seconds)

    GPU requirements:
    - 1 NVIDIA T4 GPU: ~20 concurrent 1080p60 transcodes
    - Total needed: 1M / 20 = 50,000 GPUs
    - With redundancy (1.5x): 75,000 GPUs
    - Cost: 75K × $500/month = $37.5M/month (transcoding alone!)

    CPU fallback:
    - 1 vCPU: ~2 concurrent 1080p30 transcodes
    - Alternative: 500K vCPUs (~62,500 c5.4xlarge instances)

    Cloud transcoding:
    - AWS MediaLive: $1.50/hour per channel
    - 1M streams × $1.50 = $1.5M/hour = $1.08B/month
    - Custom GPU fleet: $37.5M/month (34x cheaper!)
    ```

    ---

    ## Key Assumptions

    1. Peak 10M concurrent viewers, 1M concurrent streamers
    2. Average stream bitrate: 6 Mbps (1080p60fps)
    3. Viewers require adaptive bitrate (6 qualities)
    4. 95% of traffic served by CDN (critical for cost)
    5. <3s latency acceptable for most use cases (enables segment caching)
    6. Chat must be real-time (< 100ms delivery)
    7. Top streams can have 100K+ concurrent viewers

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Edge-first architecture:** Push processing close to users (CDN, edge transcoding)
    2. **Adaptive bitrate:** Multiple quality levels for smooth playback
    3. **Latency tiers:** 3-tier latency model (standard/low/ultra-low) for cost optimization
    4. **Horizontal scaling:** Every component can scale independently
    5. **CDN-centric:** 95% of egress bandwidth served by CDN

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Streamer Layer"
            OBS[OBS Studio<br/>Streaming Software]
            Mobile_Stream[Mobile App<br/>IRL streaming]
        end

        subgraph "Ingestion Layer"
            Ingest_LB[Ingest Load Balancer<br/>Route to nearest ingest]
            RTMP_Server[RTMP Ingest<br/>Stream reception]
            WebRTC_Server[WebRTC Ingest<br/>Ultra-low latency]
        end

        subgraph "Transcoding Layer"
            Transcoder[Transcoding Service<br/>GPU-accelerated<br/>6 quality outputs]
            Packager[Stream Packager<br/>HLS/DASH segments]
        end

        subgraph "Storage & Origin"
            Segment_Store[Segment Storage<br/>S3/Object Store<br/>Last 2 hours]
            VOD_Store[VOD Storage<br/>S3 Glacier<br/>Recordings]
            Origin[Origin Server<br/>Segment serving]
        end

        subgraph "CDN Layer"
            CDN_Edge[CDN Edge Nodes<br/>200+ global POPs<br/>95% of traffic]
        end

        subgraph "Viewer Layer"
            Web[Web Browser<br/>HLS/DASH player]
            Mobile_View[Mobile App<br/>Native player]
        end

        subgraph "Chat Layer"
            Chat_LB[Chat Load Balancer]
            Chat_Server[Chat WebSocket<br/>Pub/sub messaging]
            Chat_DB[(Chat History<br/>Cassandra<br/>30-day retention)]
        end

        subgraph "API Layer"
            Stream_API[Stream Service<br/>Start/stop stream]
            Discovery_API[Discovery Service<br/>Browse/search]
            User_API[User Service<br/>Follow/notify]
        end

        subgraph "Metadata & Control"
            Stream_DB[(Stream Metadata<br/>PostgreSQL<br/>Live streams)]
            User_DB[(User DB<br/>PostgreSQL<br/>Profiles)]
            Cache[Redis Cache<br/>Sessions, metadata]
        end

        subgraph "Analytics & Monitoring"
            Analytics[Analytics Pipeline<br/>Kafka + Flink<br/>Viewer metrics]
            Monitoring[Monitoring<br/>Prometheus/Grafana<br/>Stream health]
        end

        OBS --> Ingest_LB
        Mobile_Stream --> Ingest_LB
        Ingest_LB --> RTMP_Server
        Ingest_LB --> WebRTC_Server

        RTMP_Server --> Transcoder
        WebRTC_Server --> Transcoder

        Transcoder --> Packager
        Packager --> Segment_Store
        Packager --> VOD_Store

        Segment_Store --> Origin
        Origin --> CDN_Edge

        CDN_Edge --> Web
        CDN_Edge --> Mobile_View

        Web --> Chat_LB
        Mobile_View --> Chat_LB
        Chat_LB --> Chat_Server
        Chat_Server --> Chat_DB

        Web --> Stream_API
        Mobile_View --> Stream_API
        Stream_API --> Stream_DB
        Stream_API --> Cache

        Discovery_API --> Stream_DB
        User_API --> User_DB

        Transcoder --> Analytics
        Chat_Server --> Analytics
        CDN_Edge --> Analytics

        style CDN_Edge fill:#e8f5e9
        style Transcoder fill:#fff4e1
        style Cache fill:#fff4e1
        style Segment_Store fill:#f3e5f5
        style VOD_Store fill:#f3e5f5
        style Stream_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style Chat_DB fill:#ffe1e1
        style Chat_Server fill:#e1f5ff
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **GPU Transcoding** | Real-time conversion to 6 qualities (20 streams/GPU), 34x cheaper than cloud | CPU transcoding (too slow, 2x slower than real-time), Cloud services (AWS MediaLive too expensive) |
    | **CDN (Multi-tier)** | Serve 95% of 30-80 Tbps egress, reduce origin load, global low latency | Direct origin serving (can't handle 80 Tbps), Single CDN (vendor lock-in, no failover) |
    | **HLS/DASH (not RTMP)** | Works in browsers without plugins, adaptive bitrate, CDN-cacheable segments | RTMP delivery (Flash required, no adaptive), WebRTC (can't cache, expensive) |
    | **WebSocket Chat** | Real-time bidirectional, handles 100M messages/minute | HTTP polling (too slow, wasteful), Message queue only (no real-time delivery) |
    | **Cassandra (Chat)** | High write throughput (1.67M writes/sec), time-series optimized | PostgreSQL (write bottleneck), MongoDB (consistency issues at scale) |
    | **Redis Cache** | Fast metadata lookups (<10ms), stream manifest caching | No cache (can't handle 10M QPS), Direct DB (too slow) |

    **Key Trade-off:** We chose **standard latency (3s) as default** over ultra-low latency. The 3-second latency enables segment caching at CDN, reducing cost by 95%. Ultra-low latency (WebRTC) available but requires direct origin serving (expensive).

    ---

    ## API Design

    ### 1. Start Stream

    **Request:**
    ```http
    POST /api/v1/streams/start
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "title": "Playing Dark Souls - First Playthrough!",
      "category": "Gaming",
      "tags": ["darksouls", "firstplaythrough", "soulslike"],
      "latency_mode": "standard",  // standard (3s), low (1s), ultra-low (500ms)
      "enable_recording": true,
      "enable_chat": true
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "stream_id": "stream_abc123",
      "stream_key": "live_xyz789secret",  // Secret key for RTMP
      "ingest_urls": {
        "rtmp": "rtmp://ingest-us-west.example.com/live",
        "rtmps": "rtmps://ingest-us-west.example.com:443/live",
        "webrtc": "https://webrtc-ingest-us-west.example.com/whip"  // For ultra-low latency
      },
      "playback_url": "https://example.com/streams/abc123",
      "status": "ready",  // ready -> live -> ended
      "created_at": "2026-02-02T10:00:00Z"
    }
    ```

    **Design Notes:**

    - Return ingest URLs immediately (stream not yet live)
    - Stream goes "live" when first video packet received
    - Stream key is secret, used for RTMP authentication
    - Latency mode determines infrastructure routing

    ---

    ### 2. Get Stream Playback

    **Request:**
    ```http
    GET /api/v1/streams/abc123/playback
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "stream_id": "stream_abc123",
      "status": "live",
      "title": "Playing Dark Souls - First Playthrough!",
      "streamer": {
        "user_id": "user_123",
        "username": "darksouls_pro",
        "avatar": "https://cdn.example.com/avatars/user123.jpg"
      },
      "viewer_count": 1523,
      "started_at": "2026-02-02T10:05:23Z",
      "playback_urls": {
        "hls": "https://cdn.example.com/hls/abc123/master.m3u8",
        "dash": "https://cdn.example.com/dash/abc123/manifest.mpd"
      },
      "qualities": [
        {"name": "1080p60", "bitrate": 8000000, "resolution": "1920x1080", "fps": 60},
        {"name": "1080p", "bitrate": 6000000, "resolution": "1920x1080", "fps": 30},
        {"name": "720p", "bitrate": 3000000, "resolution": "1280x720", "fps": 30},
        {"name": "480p", "bitrate": 1500000, "resolution": "854x480", "fps": 30},
        {"name": "360p", "bitrate": 700000, "resolution": "640x360", "fps": 30},
        {"name": "160p", "bitrate": 300000, "resolution": "284x160", "fps": 30}
      ],
      "chat_url": "wss://chat.example.com/streams/abc123"
    }
    ```

    **Design Notes:**

    - Returns HLS master playlist URL (contains all qualities)
    - CDN serves all video segments (95% of bandwidth)
    - Viewer count updated every 10 seconds
    - Chat WebSocket URL separate from video

    ---

    ### 3. Send Chat Message

    **WebSocket Connection:**
    ```javascript
    // Client connects to WebSocket
    const ws = new WebSocket('wss://chat.example.com/streams/abc123?token=<jwt>');

    // Send message
    ws.send(JSON.stringify({
      "type": "message",
      "content": "Great gameplay! 🔥",
      "color": "#FF5733",  // Optional: custom color for subscribers
      "badges": ["subscriber", "moderator"]  // Optional
    }));

    // Receive messages
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      // {
      //   "type": "message",
      //   "message_id": "msg_123",
      //   "user": {
      //     "user_id": "user_456",
      //     "username": "viewer123",
      //     "badges": ["subscriber"]
      //   },
      //   "content": "Great gameplay! 🔥",
      //   "timestamp": "2026-02-02T10:15:30Z"
      // }
    };
    ```

    **Design Notes:**

    - WebSocket for bidirectional real-time chat
    - Messages broadcast to all connected viewers
    - Server validates rate limits (20 messages/30 seconds)
    - Large streams use sharded chat servers (10K users per shard)

    ---

    ### 4. Search Live Streams

    **Request:**
    ```http
    GET /api/v1/streams/search?q=dark+souls&category=Gaming&min_viewers=100&limit=20
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "streams": [
        {
          "stream_id": "stream_abc123",
          "title": "Playing Dark Souls - First Playthrough!",
          "streamer": {
            "user_id": "user_123",
            "username": "darksouls_pro",
            "avatar": "https://cdn.example.com/avatars/user123.jpg"
          },
          "thumbnail": "https://cdn.example.com/thumbnails/abc123.jpg",
          "viewer_count": 1523,
          "category": "Gaming",
          "tags": ["darksouls", "firstplaythrough"],
          "started_at": "2026-02-02T10:05:23Z"
        }
        // ... more streams
      ],
      "total_results": 47,
      "next_cursor": "cursor_xyz"
    }
    ```

    ---

    ## Database Schema

    ### Streams (PostgreSQL)

    ```sql
    -- Live streams table
    CREATE TABLE streams (
        stream_id VARCHAR(64) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        stream_key VARCHAR(128) UNIQUE NOT NULL,
        title VARCHAR(200),
        category VARCHAR(100),
        tags TEXT[],
        status VARCHAR(20),  -- ready, live, ended
        latency_mode VARCHAR(20),  -- standard, low, ultra-low
        viewer_count INT DEFAULT 0,
        peak_viewer_count INT DEFAULT 0,
        enable_recording BOOLEAN DEFAULT true,
        enable_chat BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP,
        ended_at TIMESTAMP,
        duration_seconds INT,
        ingest_url TEXT,
        playback_url TEXT,
        INDEX idx_user_id (user_id),
        INDEX idx_status (status),
        INDEX idx_category (category),
        INDEX idx_viewer_count (viewer_count DESC)
    );

    -- VOD recordings
    CREATE TABLE vod_recordings (
        vod_id VARCHAR(64) PRIMARY KEY,
        stream_id VARCHAR(64) REFERENCES streams(stream_id),
        user_id BIGINT NOT NULL,
        title VARCHAR(200),
        duration_seconds INT,
        file_size_bytes BIGINT,
        s3_bucket VARCHAR(100),
        s3_key TEXT,
        thumbnail_url TEXT,
        view_count BIGINT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_stream_id (stream_id),
        INDEX idx_user_id (user_id)
    );

    -- User follows
    CREATE TABLE follows (
        follower_id BIGINT NOT NULL,
        following_id BIGINT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (follower_id, following_id),
        INDEX idx_following (following_id)
    );
    ```

    ---

    ### Chat Messages (Cassandra)

    ```sql
    -- Chat messages (time-series, 30-day retention)
    CREATE TABLE chat_messages (
        stream_id TEXT,
        timestamp TIMESTAMP,
        message_id TEXT,
        user_id BIGINT,
        username TEXT,
        content TEXT,
        badges LIST<TEXT>,
        PRIMARY KEY (stream_id, timestamp, message_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC)
      AND default_time_to_live = 2592000;  -- 30 days

    -- Chat statistics
    CREATE TABLE chat_stats (
        stream_id TEXT,
        window_start TIMESTAMP,
        message_count COUNTER,
        active_users COUNTER,
        PRIMARY KEY (stream_id, window_start)
    );
    ```

    **Why Cassandra for Chat:**

    - **High write throughput:** 1.67M messages/second (100M/minute)
    - **Time-series optimized:** Natural fit for chat history
    - **TTL support:** Auto-delete old messages after 30 days
    - **Linear scalability:** Add nodes for more write capacity

    ---

    ### Users (PostgreSQL)

    ```sql
    -- Users table (sharded by user_id)
    CREATE TABLE users (
        user_id BIGINT PRIMARY KEY,
        username VARCHAR(30) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        display_name VARCHAR(50),
        avatar_url TEXT,
        bio VARCHAR(500),
        follower_count INT DEFAULT 0,
        following_count INT DEFAULT 0,
        total_views BIGINT DEFAULT 0,
        is_verified BOOLEAN DEFAULT FALSE,
        is_partner BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_username (username)
    ) PARTITION BY HASH (user_id);
    ```

    ---

    ## Data Flow Diagrams

    ### Stream Ingestion Flow

    ```mermaid
    sequenceDiagram
        participant Streamer
        participant RTMP_Server
        participant Transcoder
        participant Packager
        participant S3
        participant CDN
        participant Stream_API
        participant DB

        Streamer->>Stream_API: POST /streams/start
        Stream_API->>DB: INSERT stream (status=ready)
        Stream_API-->>Streamer: stream_key, ingest_url

        Streamer->>RTMP_Server: Connect RTMP + stream_key
        RTMP_Server->>RTMP_Server: Authenticate stream_key
        RTMP_Server->>Stream_API: Notify stream live
        Stream_API->>DB: UPDATE status=live

        loop Every 2-second segment
            RTMP_Server->>Transcoder: Raw video/audio (1080p60, 6 Mbps)

            par Parallel Transcoding
                Transcoder->>Transcoder: Encode 160p (300 Kbps)
                Transcoder->>Transcoder: Encode 360p (700 Kbps)
                Transcoder->>Transcoder: Encode 480p (1.5 Mbps)
                Transcoder->>Transcoder: Encode 720p (3 Mbps)
                Transcoder->>Transcoder: Encode 1080p (6 Mbps)
                Transcoder->>Transcoder: Encode 1080p60 (8 Mbps)
            end

            Transcoder->>Packager: 6 quality outputs
            Packager->>Packager: Create HLS segments (.ts files)
            Packager->>Packager: Update master playlist (master.m3u8)
            Packager->>Packager: Update quality playlists (720p.m3u8, etc.)

            Packager->>S3: Upload segments + playlists
            S3-->>CDN: CDN pulls new segments (cache)
        end

        Note over CDN: Segments cached at edge<br/>95% of requests served by CDN
    ```

    **Flow Explanation:**

    1. **Stream start** - API creates stream record, returns ingest URL
    2. **RTMP connection** - Streamer connects, server authenticates stream key
    3. **Transcoding** - GPU transcodes source to 6 qualities in parallel (real-time)
    4. **Packaging** - Segments packaged as HLS (.ts files), playlists updated
    5. **Storage** - Segments uploaded to S3, CDN pulls and caches at edge
    6. **Latency** - 3 seconds glass-to-glass (1s transcode + 1s packaging + 1s delivery)

    ---

    ### Viewer Playback Flow

    ```mermaid
    sequenceDiagram
        participant Viewer
        participant API
        participant CDN
        participant Origin
        participant Chat_WS

        Viewer->>API: GET /streams/abc123/playback
        API->>API: Check stream status
        API-->>Viewer: playback_url (HLS master.m3u8)

        Viewer->>CDN: GET /hls/abc123/master.m3u8
        alt CDN Cache HIT (99% of requests)
            CDN-->>Viewer: Master playlist (list of qualities)
        else CDN Cache MISS
            CDN->>Origin: GET master.m3u8
            Origin-->>CDN: Master playlist
            CDN-->>Viewer: Master playlist
        end

        Viewer->>Viewer: Select quality (auto or manual)

        loop Every 2 seconds (adaptive)
            Viewer->>CDN: GET /hls/abc123/720p/segment_001.ts
            CDN-->>Viewer: Video segment (cached)

            Viewer->>Viewer: Monitor bandwidth, adjust quality

            alt Bandwidth decreased
                Viewer->>CDN: GET /hls/abc123/480p/segment_002.ts
            else Bandwidth increased
                Viewer->>CDN: GET /hls/abc123/1080p/segment_003.ts
            end
        end

        par Chat Connection
            Viewer->>Chat_WS: WebSocket connect
            Chat_WS-->>Viewer: Connected

            loop Real-time messages
                Chat_WS->>Viewer: Broadcast chat messages
                Viewer->>Chat_WS: Send chat message
            end
        end
    ```

    **Flow Explanation:**

    1. **Get playback URL** - API returns HLS master playlist URL
    2. **Fetch master playlist** - Lists available qualities (CDN cached)
    3. **Adaptive bitrate** - Player selects quality based on bandwidth
    4. **Segment fetching** - Request segments every 2 seconds (CDN cached)
    5. **Chat WebSocket** - Separate connection for real-time chat
    6. **Latency** - 3 seconds behind live (2-3 segments buffered)

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical live streaming subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Transcoding Pipeline** | How to transcode 1M streams in real-time? | GPU-accelerated transcoding, 20 streams/GPU, auto-scaling |
    | **Low-Latency Streaming** | How to achieve <1s latency? | LL-HLS, WebRTC, chunked transfer, edge processing |
    | **CDN Strategy** | How to serve 30-80 Tbps efficiently? | Multi-tier CDN, origin shield, segment caching |
    | **Chat at Scale** | How to handle 100M messages/minute? | Pub/sub architecture, sharded chat servers, Cassandra |

    ---

    === "🎬 Transcoding Pipeline"

        ## The Challenge

        **Problem:** Transcode 1M concurrent streams in real-time (6 quality outputs each) while keeping latency <3s.

        **Constraints:**

        - Real-time requirement: 1x speed (transcode 1 second of video in ≤1 second)
        - 6 quality outputs per stream (160p to 1080p60)
        - GPU resources limited and expensive (75K GPUs needed)
        - Must handle traffic spikes (2x streams during events)

        ---

        ## GPU vs CPU Transcoding

        | Metric | GPU (NVIDIA T4) | CPU (Intel Xeon) | Winner |
        |--------|-----------------|------------------|--------|
        | **Concurrent 1080p60 streams** | 20 streams/GPU | 2 streams/instance | GPU (10x) |
        | **Latency** | 0.5s per segment | 1.5s per segment | GPU (3x faster) |
        | **Cost** | $500/month per GPU | $300/month per c5.4xlarge | GPU (6x more cost-effective) |
        | **Power efficiency** | 70W per GPU | 400W per instance | GPU (5x better) |
        | **Scalability** | Horizontal scaling | Horizontal scaling | Tie |

        **Decision: GPU transcoding (NVIDIA T4)** - 10x density, 3x faster, cost-effective at scale.

        ---

        ## Transcoding Architecture

        ```python
        import ffmpeg
        import boto3
        from concurrent.futures import ThreadPoolExecutor

        class GPUTranscoder:
            """
            GPU-accelerated real-time transcoding service

            Handles RTMP input, outputs 6 HLS qualities
            """

            # Quality ladder (bitrate, resolution, fps)
            QUALITIES = [
                {'name': '160p', 'bitrate': '300k', 'resolution': '284x160', 'fps': 30},
                {'name': '360p', 'bitrate': '700k', 'resolution': '640x360', 'fps': 30},
                {'name': '480p', 'bitrate': '1500k', 'resolution': '854x480', 'fps': 30},
                {'name': '720p', 'bitrate': '3000k', 'resolution': '1280x720', 'fps': 30},
                {'name': '1080p', 'bitrate': '6000k', 'resolution': '1920x1080', 'fps': 30},
                {'name': '1080p60', 'bitrate': '8000k', 'resolution': '1920x1080', 'fps': 60}
            ]

            def __init__(self, gpu_id=0):
                self.gpu_id = gpu_id
                self.s3 = boto3.client('s3')
                self.segment_duration = 2  # seconds (trade-off: latency vs cacheability)

            def transcode_stream(self, stream_id: str, rtmp_input: str):
                """
                Transcode single RTMP stream to multiple qualities

                Args:
                    stream_id: Unique stream identifier
                    rtmp_input: RTMP input URL (rtmp://localhost/live/stream_key)
                """
                logger.info(f"Starting transcoding for stream {stream_id} on GPU {self.gpu_id}")

                # Build FFmpeg command with NVIDIA GPU acceleration
                outputs = []
                for quality in self.QUALITIES:
                    output = self._build_output_config(stream_id, quality)
                    outputs.append(output)

                # FFmpeg command (pseudo-code, actual command is complex)
                cmd = [
                    'ffmpeg',
                    '-hwaccel', 'cuda',  # Use NVIDIA GPU
                    '-hwaccel_device', str(self.gpu_id),  # Select GPU
                    '-i', rtmp_input,  # Input stream
                    '-f', 'hls',  # HLS output format
                    '-hls_time', str(self.segment_duration),  # 2-second segments
                    '-hls_list_size', '5',  # Keep last 5 segments in playlist
                    '-hls_flags', 'delete_segments+append_list',  # Clean up old segments
                ]

                # Add output configs for each quality
                for output in outputs:
                    cmd.extend([
                        '-c:v', 'h264_nvenc',  # NVIDIA H.264 encoder
                        '-preset', 'p4',  # Quality preset (p1-p7, p4 = balanced)
                        '-b:v', output['bitrate'],
                        '-maxrate', output['bitrate'],
                        '-bufsize', f"{int(output['bitrate'][:-1]) * 2}k",  # 2x bitrate
                        '-s', output['resolution'],
                        '-r', str(output['fps']),
                        '-c:a', 'aac',  # Audio codec
                        '-b:a', '128k',  # Audio bitrate
                        '-ar', '48000',  # Audio sample rate
                        '-hls_segment_filename', f"s3://bucket/streams/{stream_id}/{output['name']}/segment_%03d.ts",
                        f"s3://bucket/streams/{stream_id}/{output['name']}/playlist.m3u8"
                    ])

                # Execute FFmpeg (blocks until stream ends)
                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Monitor transcoding health
                    while process.poll() is None:
                        self._check_health(stream_id)
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Transcoding failed for {stream_id}: {e}")
                    self._notify_failure(stream_id)

            def _build_output_config(self, stream_id: str, quality: dict) -> dict:
                """Build output configuration for a quality level"""
                return {
                    'name': quality['name'],
                    'bitrate': quality['bitrate'],
                    'resolution': quality['resolution'],
                    'fps': quality['fps'],
                    'output_path': f"s3://bucket/streams/{stream_id}/{quality['name']}/"
                }

            def _check_health(self, stream_id: str):
                """Monitor transcoding health (dropped frames, latency)"""
                # Check GPU utilization
                gpu_util = self._get_gpu_utilization()
                if gpu_util > 95:
                    logger.warning(f"GPU {self.gpu_id} overloaded: {gpu_util}%")

                # Check segment upload latency
                upload_latency = self._get_upload_latency(stream_id)
                if upload_latency > 1.0:
                    logger.warning(f"High upload latency for {stream_id}: {upload_latency}s")
        ```

        ---

        ## Transcoding Cluster Management

        ```python
        class TranscodingCluster:
            """
            Manages fleet of GPU transcoding servers

            Handles auto-scaling, load balancing, failover
            """

            def __init__(self):
                self.servers = {}  # server_id -> TranscodingServer
                self.stream_assignments = {}  # stream_id -> server_id
                self.target_gpu_utilization = 75  # Trigger scale-up at 75%

            def assign_stream(self, stream_id: str, rtmp_input: str) -> str:
                """
                Assign stream to least-loaded transcoding server

                Returns:
                    server_id where stream was assigned
                """
                # Find server with available capacity
                server = self._find_available_server()

                if not server:
                    # No capacity available, scale up
                    logger.info("No available servers, scaling up cluster")
                    server = self._scale_up()

                # Assign stream to server
                server.start_transcode(stream_id, rtmp_input)
                self.stream_assignments[stream_id] = server.server_id

                logger.info(f"Assigned stream {stream_id} to server {server.server_id}")
                return server.server_id

            def _find_available_server(self) -> 'TranscodingServer':
                """Find server with capacity (< 20 streams per GPU)"""
                for server in self.servers.values():
                    if server.available_slots() > 0:
                        return server
                return None

            def _scale_up(self) -> 'TranscodingServer':
                """
                Launch new transcoding server (auto-scaling)

                Returns:
                    Newly launched server
                """
                # Launch EC2 instance (g4dn.xlarge with NVIDIA T4)
                instance = self._launch_ec2_instance(
                    instance_type='g4dn.xlarge',
                    ami='ami-transcoder-v1.2.3',
                    tags={'Role': 'Transcoder', 'AutoScaling': 'true'}
                )

                # Wait for instance to be ready
                instance.wait_until_running()

                # Register server
                server = TranscodingServer(
                    server_id=instance.id,
                    ip_address=instance.private_ip_address,
                    gpu_count=1,
                    max_streams_per_gpu=20
                )

                self.servers[server.server_id] = server
                logger.info(f"Scaled up: launched server {server.server_id}")

                return server

            def _scale_down(self):
                """
                Terminate idle transcoding servers (cost optimization)

                Called periodically to remove unused capacity
                """
                for server_id, server in list(self.servers.items()):
                    if server.active_streams == 0 and server.idle_time > 300:  # 5 min idle
                        # Gracefully terminate
                        server.shutdown()
                        del self.servers[server_id]
                        logger.info(f"Scaled down: terminated server {server_id}")

            def monitor_cluster(self):
                """
                Continuous monitoring and auto-scaling

                Runs in background thread
                """
                while True:
                    # Calculate cluster utilization
                    total_capacity = sum(s.max_streams for s in self.servers.values())
                    active_streams = sum(s.active_streams for s in self.servers.values())
                    utilization = (active_streams / total_capacity * 100) if total_capacity > 0 else 0

                    logger.info(f"Cluster utilization: {utilization:.1f}% ({active_streams}/{total_capacity})")

                    # Auto-scaling decisions
                    if utilization > self.target_gpu_utilization:
                        # Scale up: add more servers
                        servers_to_add = math.ceil((utilization - self.target_gpu_utilization) / 100 * len(self.servers))
                        for _ in range(servers_to_add):
                            self._scale_up()

                    elif utilization < 30:
                        # Scale down: remove idle servers
                        self._scale_down()

                    time.sleep(30)  # Check every 30 seconds
        ```

        ---

        ## Segment Duration Trade-off

        **Key decision: How long should each HLS segment be?**

        | Segment Duration | Latency Impact | CDN Cacheability | Bandwidth Overhead | Twitch Choice |
        |------------------|----------------|------------------|--------------------|---------------|
        | **10 seconds** | +10s latency | Excellent (99% hit rate) | Low (1% overhead) | ❌ Too slow |
        | **4 seconds** | +4s latency | Good (95% hit rate) | Low (2% overhead) | Standard mode |
        | **2 seconds** | +2s latency | Fair (85% hit rate) | Medium (5% overhead) | ✅ Low-latency |
        | **1 second** | +1s latency | Poor (70% hit rate) | High (10% overhead) | Ultra-low latency |
        | **0.2 seconds (LL-HLS)** | +0.2s latency | Very poor (40% hit rate) | Very high (20% overhead) | WebRTC alternative |

        **Decision:** 2-second segments for standard latency (good balance), 1-second for low-latency mode.

    === "⚡ Low-Latency Streaming"

        ## The Challenge

        **Problem:** Standard HLS has 6-10s latency. How to achieve <1s latency for interactive streams (gaming, auctions)?

        **Latency breakdown (standard HLS, 4s segments):**

        ```
        Encoding:         1.0s (GPU transcoding)
        Packaging:        0.5s (HLS segmentation)
        Upload to CDN:    0.5s (network transfer)
        CDN propagation:  1.0s (edge distribution)
        Player buffer:    8.0s (2x segment duration for smooth playback)
        ─────────────────────
        Total latency:   11.0s
        ```

        **Target:** Reduce to <1s (11x improvement!)

        ---

        ## Low-Latency Protocols Comparison

        | Protocol | Latency | Browser Support | CDN Cacheable | Scalability | Cost |
        |----------|---------|----------------|---------------|-------------|------|
        | **Standard HLS** | 6-10s | ✅ Universal | ✅ Yes (99% hit rate) | ✅ Millions | $ |
        | **Low-Latency HLS (LL-HLS)** | 2-3s | ✅ iOS, partial others | 🟡 Partial (70% hit rate) | ✅ Millions | $$ |
        | **DASH Low-Latency** | 2-4s | ✅ Most browsers | 🟡 Partial | ✅ Millions | $$ |
        | **WebRTC** | 0.2-0.5s | ✅ Universal (modern) | ❌ No (live only) | 🟡 Thousands | $$$$ |
        | **SRT** | 0.5-1s | ❌ Requires plugin | ❌ No | 🟡 Thousands | $$$ |

        **Decision:** Three-tier latency model

        1. **Standard (3s):** HLS with 2s segments - 95% of viewers (cost-effective)
        2. **Low-latency (1s):** LL-HLS - streamers and engaged viewers (moderate cost)
        3. **Ultra-low (<500ms):** WebRTC - competitive gaming, auctions (expensive)

        ---

        ## Low-Latency HLS (LL-HLS) Implementation

        **Key innovations:**

        1. **Partial segments:** Deliver video in 200ms chunks instead of 2s segments
        2. **HTTP/2 push:** Server pushes next chunk before client requests
        3. **Chunked transfer encoding:** Stream segment as it's being encoded
        4. **Rendition reports:** Playlist includes upcoming chunk timings

        ```python
        class LowLatencyHLSPackager:
            """
            Low-Latency HLS packager

            Implements LL-HLS spec (Apple HLS Extensions RFC 8216)
            """

            PART_DURATION = 0.2  # 200ms partial segments
            SEGMENT_DURATION = 2.0  # 2-second full segments (10 parts)

            def __init__(self, stream_id: str):
                self.stream_id = stream_id
                self.segment_number = 0
                self.part_number = 0

            def package_stream(self, transcoded_input: str):
                """
                Package transcoded stream as LL-HLS

                Args:
                    transcoded_input: Path to transcoded video stream
                """
                while True:
                    # Read 200ms of video
                    part_data = self._read_part(transcoded_input)

                    if not part_data:
                        break  # Stream ended

                    # Upload partial segment immediately (don't wait for full segment)
                    part_url = self._upload_part(part_data)

                    # Update playlist with new part
                    self._update_playlist_with_part(part_url)

                    # Every 10 parts (2 seconds), finalize full segment
                    if self.part_number % 10 == 0:
                        self._finalize_segment()

                    self.part_number += 1

            def _update_playlist_with_part(self, part_url: str):
                """
                Update HLS playlist with partial segment

                Playlist example:
                #EXTM3U
                #EXT-X-VERSION:6
                #EXT-X-TARGETDURATION:2
                #EXT-X-PART-INF:PART-TARGET=0.2
                #EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=0.6

                #EXTINF:2.0
                #EXT-X-PART:DURATION=0.2,URI="part_001.m4s"
                #EXT-X-PART:DURATION=0.2,URI="part_002.m4s"
                #EXT-X-PART:DURATION=0.2,URI="part_003.m4s"
                ...
                segment_001.m4s
                """
                playlist = self._load_playlist()

                # Add partial segment
                playlist.add_part({
                    'duration': self.PART_DURATION,
                    'uri': part_url,
                    'independent': self.part_number % 10 == 0  # First part of segment
                })

                # Save playlist
                self._save_playlist(playlist)

                # Notify CDN of update (HTTP/2 push)
                self._push_to_cdn(playlist)

            def _push_to_cdn(self, playlist):
                """
                Push playlist update to CDN edge nodes

                Uses HTTP/2 Server Push to proactively send next part
                """
                # Send playlist update
                cdn_client.push_update(
                    f"s3://bucket/streams/{self.stream_id}/playlist.m3u8",
                    playlist.to_string()
                )

                # Preload upcoming parts (reduce latency)
                cdn_client.push_hint(
                    f"s3://bucket/streams/{self.stream_id}/part_{self.part_number + 1}.m4s"
                )
        ```

        ---

        ## WebRTC for Ultra-Low Latency

        **When to use:** Competitive gaming streams, live auctions, real-time collaboration.

        **Architecture differences:**

        | Aspect | HLS/DASH | WebRTC |
        |--------|----------|---------|
        | **Protocol** | HTTP (TCP) | RTP/SRTP (UDP) |
        | **Latency** | 2-10s | 0.2-0.5s |
        | **CDN caching** | Yes (segments cached) | No (live UDP streams) |
        | **Scalability** | Millions (CDN) | Thousands (direct from origin) |
        | **Cost at 10K viewers** | $100/hour | $5,000/hour |

        **Implementation:**

        ```python
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        import asyncio

        class WebRTCStreamer:
            """
            WebRTC streaming for ultra-low latency

            Handles WHIP (WebRTC HTTP Ingestion Protocol) for publishing
            and WHEP (WebRTC HTTP Egress Protocol) for playback
            """

            def __init__(self):
                self.peer_connections = {}  # viewer_id -> RTCPeerConnection
                self.source_track = None  # Video track from streamer

            async def handle_publish(self, stream_id: str, offer_sdp: str) -> str:
                """
                Handle streamer publishing via WHIP

                Args:
                    stream_id: Stream identifier
                    offer_sdp: WebRTC offer SDP from streamer

                Returns:
                    answer_sdp: WebRTC answer SDP to send back to streamer
                """
                pc = RTCPeerConnection()

                # Handle incoming video track
                @pc.on("track")
                async def on_track(track):
                    if track.kind == "video":
                        logger.info(f"Received video track from streamer for {stream_id}")
                        self.source_track = track

                        # Forward to all viewers (fan-out)
                        await self._fanout_to_viewers(track)

                # Set remote description (offer from streamer)
                await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                return pc.localDescription.sdp

            async def handle_playback(self, stream_id: str, viewer_id: str, offer_sdp: str) -> str:
                """
                Handle viewer playback via WHEP

                Args:
                    stream_id: Stream identifier
                    viewer_id: Viewer identifier
                    offer_sdp: WebRTC offer SDP from viewer

                Returns:
                    answer_sdp: WebRTC answer SDP to send back to viewer
                """
                pc = RTCPeerConnection()

                # Add source track to peer connection
                if self.source_track:
                    pc.addTrack(self.source_track)
                else:
                    raise Exception(f"Stream {stream_id} not live")

                # Set remote description (offer from viewer)
                await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # Store peer connection
                self.peer_connections[viewer_id] = pc

                # Monitor connection health
                @pc.on("connectionstatechange")
                async def on_connection_state_change():
                    if pc.connectionState == "failed":
                        logger.warning(f"Connection failed for viewer {viewer_id}")
                        await self._cleanup_viewer(viewer_id)

                return pc.localDescription.sdp

            async def _fanout_to_viewers(self, track):
                """
                Fan out video track to all viewers

                Problem: With 10K viewers, this becomes expensive!
                Solution: Use SFU (Selective Forwarding Unit) for scalability
                """
                for viewer_id, pc in self.peer_connections.items():
                    try:
                        pc.addTrack(track)
                    except Exception as e:
                        logger.error(f"Failed to add track for viewer {viewer_id}: {e}")
        ```

        ---

        ## SFU (Selective Forwarding Unit) for WebRTC Scaling

        **Problem:** Direct peer-to-peer WebRTC doesn't scale beyond ~100 viewers.

        **Solution:** SFU architecture - server forwards stream to multiple viewers without transcoding.

        ```
        Streamer (1 upload)
            ↓
        SFU Server (receives 1 stream)
            ├─→ Viewer 1
            ├─→ Viewer 2
            ├─→ Viewer 3
            └─→ ... 10K viewers
        ```

        **Benefits:**

        - Streamer uploads once (saves bandwidth)
        - SFU forwards without transcoding (low CPU)
        - Scales to 10K+ viewers per SFU server

        **Cost:** Still expensive vs HLS (no CDN caching, origin serves all traffic).

    === "🌍 CDN Strategy"

        ## The Challenge

        **Problem:** Serve 30-80 Tbps of video to 10M concurrent viewers globally with <3s latency.

        **Constraints:**

        - Origin bandwidth: 1.5 Tbps (bottleneck if serving directly)
        - CDN cost: $0.085/GB egress = $3.2M/hour at 80 Tbps
        - Latency requirement: <100ms CDN response time
        - 200+ global edge locations needed for worldwide coverage

        ---

        ## Multi-Tier CDN Architecture

        ```
        Viewer Requests (30-80 Tbps)
            ↓
        [Tier 1: Edge CDN (Cloudflare, Fastly)]  ← 90% cache hit
            ↓ (10% miss)
        [Tier 2: Regional CDN (Akamai)]  ← 80% cache hit
            ↓ (2% miss)
        [Tier 3: Origin Shield (AWS CloudFront)]  ← 95% cache hit
            ↓ (0.1% miss)
        [Origin Servers (S3)]  ← 1.5 Tbps
        ```

        **Why multi-tier:**

        1. **Cost optimization:** Each tier reduces origin load by 10-20x
        2. **Redundancy:** Tier 1 failure → automatic failover to Tier 2
        3. **Geographic coverage:** Tier 1 (global), Tier 2 (continents), Shield (regions)

        **Traffic flow:**

        ```
        100% requests → Edge CDN
        ├─ 90% served from edge cache (0.3 Tbps origin)
        └─ 10% miss → Regional CDN
            ├─ 80% served from regional cache (0.06 Tbps origin)
            └─ 20% miss → Origin Shield
                ├─ 95% served from shield cache (0.003 Tbps origin)
                └─ 5% miss → Origin (0.0015 Tbps = 1.5 Tbps total)
        ```

        **Result:** 1.5 Tbps origin bandwidth serves 30 Tbps viewer demand (20x amplification).

        ---

        ## CDN Cache Strategy

        ```python
        class CDNCacheStrategy:
            """
            CDN caching strategy for live streaming

            Optimizes cache hit rate, TTL, and purging
            """

            # Cache TTL by content type
            CACHE_TTL = {
                'master_playlist': 2,      # 2 seconds (updated frequently)
                'quality_playlist': 4,     # 4 seconds (matches segment duration)
                'video_segment': 3600,     # 1 hour (immutable once created)
                'thumbnail': 300,          # 5 minutes
                'vod': 86400              # 24 hours (rarely changes)
            }

            def get_cache_headers(self, content_type: str, stream_id: str) -> dict:
                """
                Return HTTP cache headers for CDN

                Args:
                    content_type: Type of content (master_playlist, segment, etc.)
                    stream_id: Stream identifier

                Returns:
                    dict of HTTP headers
                """
                ttl = self.CACHE_TTL.get(content_type, 60)

                headers = {
                    'Cache-Control': f'public, max-age={ttl}, s-maxage={ttl}',
                    'CDN-Cache-Control': f'max-age={ttl}',
                    'Surrogate-Control': f'max-age={ttl}',
                    'ETag': self._generate_etag(stream_id, content_type),
                }

                # Video segments are immutable (aggressive caching)
                if content_type == 'video_segment':
                    headers['Cache-Control'] = 'public, max-age=31536000, immutable'

                # Playlists must be revalidated (low TTL)
                elif content_type in ['master_playlist', 'quality_playlist']:
                    headers['Cache-Control'] = f'public, max-age={ttl}, must-revalidate'

                return headers

            def purge_cache(self, stream_id: str, content_type: str = None):
                """
                Purge CDN cache (e.g., stream ended, VOD deleted)

                Args:
                    stream_id: Stream identifier
                    content_type: Optional specific content type to purge
                """
                if content_type:
                    # Purge specific content type
                    patterns = [f"/streams/{stream_id}/*.{content_type}"]
                else:
                    # Purge all content for stream
                    patterns = [f"/streams/{stream_id}/*"]

                # Purge from all CDN tiers
                for cdn in self.cdn_clients:
                    try:
                        cdn.purge(patterns)
                        logger.info(f"Purged {stream_id} from {cdn.name}")
                    except Exception as e:
                        logger.error(f"Failed to purge from {cdn.name}: {e}")

            def prefetch_to_edge(self, stream_id: str, segments: List[str]):
                """
                Prefetch segments to CDN edge (before viewers request)

                Used for large events to pre-warm CDN cache
                """
                for segment_url in segments:
                    for edge_location in self.high_traffic_edges:
                        # Send prefetch request
                        edge_location.prefetch(segment_url)

                logger.info(f"Prefetched {len(segments)} segments to {len(self.high_traffic_edges)} edges")
        ```

        ---

        ## Origin Shield Implementation

        **Problem:** Without shield, 1000 edge locations might all request same segment from origin (thundering herd).

        **Solution:** Origin Shield - single regional cache that collapses requests.

        ```
        Before (no shield):
        1000 edge locations → 1000 requests to origin (1000x load)

        After (with shield):
        1000 edge locations → Origin Shield (caches) → 1 request to origin (1x load)
        ```

        **Implementation:**

        ```python
        class OriginShield:
            """
            Origin Shield to protect origin from thundering herd

            Collapses concurrent requests for same content
            """

            def __init__(self, redis_client):
                self.cache = redis_client
                self.inflight_requests = {}  # URL -> Future (request coalescing)

            async def fetch_segment(self, url: str) -> bytes:
                """
                Fetch segment with request coalescing

                If multiple edge nodes request same segment concurrently,
                only one request is sent to origin.
                """
                # Check cache
                cached = await self.cache.get(url)
                if cached:
                    logger.debug(f"Cache HIT for {url}")
                    return cached

                # Check if request is already in-flight
                if url in self.inflight_requests:
                    logger.debug(f"Request coalescing for {url}")
                    return await self.inflight_requests[url]

                # Fetch from origin
                logger.debug(f"Cache MISS for {url}, fetching from origin")
                future = asyncio.create_task(self._fetch_from_origin(url))
                self.inflight_requests[url] = future

                try:
                    data = await future

                    # Cache for 1 hour (segments are immutable)
                    await self.cache.setex(url, 3600, data)

                    return data
                finally:
                    # Remove from inflight
                    del self.inflight_requests[url]

            async def _fetch_from_origin(self, url: str) -> bytes:
                """Fetch segment from origin (S3)"""
                s3_key = self._url_to_s3_key(url)

                s3 = boto3.client('s3')
                response = s3.get_object(Bucket='live-segments', Key=s3_key)

                return response['Body'].read()
        ```

        ---

        ## CDN Cost Optimization

        **Monthly CDN cost at 30 Tbps average:**

        ```
        Without multi-tier:
        - 30 Tbps × 3600 sec/hour × 730 hours/month = 78,840,000 TB/month
        - 78.84M TB × $0.085/GB ÷ 1000 = $6,701,400/month

        With multi-tier (95% cache hit):
        - Edge serves: 95% × 78.84M TB = 74.9M TB
        - Origin serves: 5% × 78.84M TB = 3.94M TB
        - Cost: (74.9M × $0.02) + (3.94M × $0.085) = $1,498,000 + $334,900 = $1,832,900/month

        Savings: $6.7M - $1.8M = $4.9M/month (73% reduction!)
        ```

    === "💬 Chat at Scale"

        ## The Challenge

        **Problem:** Handle 100M chat messages/minute (1.67M messages/second) with <100ms delivery latency.

        **Constraints:**

        - Large streams have 50K-100K concurrent viewers
        - Top streams generate 10-20 messages/second
        - Chat must be real-time (< 100ms from send to all viewers)
        - Must store 30 days of chat history

        ---

        ## Chat Architecture

        ```python
        import asyncio
        import websockets
        import redis.asyncio as redis
        from cassandra.cluster import Cluster

        class ChatServer:
            """
            Scalable WebSocket chat server with pub/sub

            Each server handles 10K concurrent connections
            Messages broadcast via Redis pub/sub
            """

            MAX_CONNECTIONS_PER_SERVER = 10_000
            RATE_LIMIT_MESSAGES = 20  # messages per 30 seconds

            def __init__(self, server_id: str):
                self.server_id = server_id
                self.connections = {}  # user_id -> WebSocket
                self.redis = redis.Redis(host='redis-host')
                self.cassandra = Cluster(['cassandra-host']).connect('chat')

            async def handle_connection(self, websocket, stream_id: str, user_id: str):
                """
                Handle new WebSocket connection

                Args:
                    websocket: WebSocket connection
                    stream_id: Stream the user is watching
                    user_id: User identifier
                """
                # Check server capacity
                if len(self.connections) >= self.MAX_CONNECTIONS_PER_SERVER:
                    await websocket.close(code=1008, reason="Server full")
                    return

                # Register connection
                self.connections[user_id] = {
                    'websocket': websocket,
                    'stream_id': stream_id,
                    'joined_at': time.time(),
                    'message_count': 0
                }

                logger.info(f"User {user_id} joined stream {stream_id}. Total connections: {len(self.connections)}")

                # Subscribe to stream's pub/sub channel
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(f"chat:{stream_id}")

                # Handle incoming messages from user
                send_task = asyncio.create_task(self._handle_send(websocket, stream_id, user_id))

                # Handle outgoing messages to user (from pub/sub)
                receive_task = asyncio.create_task(self._handle_receive(websocket, pubsub, user_id))

                try:
                    # Wait for either task to complete (connection closed or error)
                    await asyncio.gather(send_task, receive_task)
                except Exception as e:
                    logger.error(f"Error handling connection for {user_id}: {e}")
                finally:
                    # Cleanup
                    await pubsub.unsubscribe(f"chat:{stream_id}")
                    await pubsub.close()
                    del self.connections[user_id]
                    logger.info(f"User {user_id} disconnected")

            async def _handle_send(self, websocket, stream_id: str, user_id: str):
                """
                Handle messages sent by user

                Validates, rate limits, then broadcasts to all viewers
                """
                async for message_data in websocket:
                    try:
                        message = json.loads(message_data)

                        # Validate message
                        if not self._validate_message(message):
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Invalid message format'
                            }))
                            continue

                        # Rate limiting
                        if not self._check_rate_limit(user_id):
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'message': 'Rate limit exceeded (20 messages per 30 seconds)'
                            }))
                            continue

                        # Create message object
                        chat_message = {
                            'type': 'message',
                            'message_id': self._generate_message_id(),
                            'stream_id': stream_id,
                            'user': {
                                'user_id': user_id,
                                'username': self._get_username(user_id),
                                'badges': self._get_user_badges(user_id)
                            },
                            'content': message['content'],
                            'timestamp': time.time()
                        }

                        # Persist to Cassandra (async, don't wait)
                        asyncio.create_task(self._persist_message(chat_message))

                        # Broadcast to all viewers via Redis pub/sub
                        await self.redis.publish(
                            f"chat:{stream_id}",
                            json.dumps(chat_message)
                        )

                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        }))

            async def _handle_receive(self, websocket, pubsub, user_id: str):
                """
                Handle messages from Redis pub/sub

                Forwards broadcasted messages to user's WebSocket
                """
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            # Forward to user's WebSocket
                            await websocket.send(message['data'])
                        except websockets.exceptions.ConnectionClosed:
                            logger.info(f"Connection closed for {user_id}")
                            break

            def _check_rate_limit(self, user_id: str) -> bool:
                """
                Check if user exceeded rate limit (20 messages per 30 seconds)

                Uses sliding window algorithm
                """
                now = time.time()
                window_start = now - 30

                # Get message timestamps from last 30 seconds
                key = f"ratelimit:{user_id}"
                timestamps = self.redis.zrangebyscore(key, window_start, now)

                if len(timestamps) >= self.RATE_LIMIT_MESSAGES:
                    return False  # Rate limit exceeded

                # Add current timestamp
                self.redis.zadd(key, {now: now})
                self.redis.expire(key, 30)  # Expire after 30 seconds

                return True

            async def _persist_message(self, message: dict):
                """
                Persist message to Cassandra (30-day retention)

                Async operation, doesn't block broadcast
                """
                try:
                    self.cassandra.execute(
                        """
                        INSERT INTO chat_messages (stream_id, timestamp, message_id, user_id, username, content, badges)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            message['stream_id'],
                            message['timestamp'],
                            message['message_id'],
                            message['user']['user_id'],
                            message['user']['username'],
                            message['content'],
                            message['user']['badges']
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to persist message {message['message_id']}: {e}")
        ```

        ---

        ## Chat Sharding for Large Streams

        **Problem:** Stream with 100K viewers → single chat server can't handle (10K limit).

        **Solution:** Shard chat servers, route viewers to specific shard.

        ```
        100K viewers watching same stream
            ↓
        Chat Load Balancer (routes by user_id hash)
            ├─→ Chat Server 1 (10K viewers, shard 0)
            ├─→ Chat Server 2 (10K viewers, shard 1)
            ├─→ Chat Server 3 (10K viewers, shard 2)
            └─→ ... 10 servers

        All servers subscribe to same Redis channel: chat:stream_123
        When user sends message → broadcast to all shards via Redis pub/sub
        ```

        **Benefits:**

        - Horizontal scaling (add more servers for larger streams)
        - Each server only handles 10K WebSocket connections
        - Redis pub/sub ensures all shards receive messages

        **Code:**

        ```python
        class ChatLoadBalancer:
            """
            Route users to chat server shards

            Uses consistent hashing for even distribution
            """

            def __init__(self, num_shards=10):
                self.num_shards = num_shards
                self.servers = [
                    f"chat-{i}.example.com"
                    for i in range(num_shards)
                ]

            def get_chat_server(self, stream_id: str, user_id: str) -> str:
                """
                Determine which chat server to route user to

                Args:
                    stream_id: Stream identifier
                    user_id: User identifier

                Returns:
                    Chat server URL
                """
                # Hash user_id to determine shard
                shard_id = hash(user_id) % self.num_shards

                return self.servers[shard_id]
        ```

        ---

        ## Chat Storage Optimization

        **Problem:** 100M messages/minute × 1 KB = 144 TB/day. Expensive to store!

        **Optimization strategies:**

        1. **TTL:** Auto-delete after 30 days (Cassandra TTL)
        2. **Compression:** gzip messages (3x compression ratio)
        3. **Cold storage:** Move old messages to S3 Glacier (10x cheaper)
        4. **Sampling:** Store only 10% of messages from large streams (analytics)

        ```python
        # Cassandra table with TTL
        CREATE TABLE chat_messages (
            stream_id TEXT,
            timestamp TIMESTAMP,
            message_id TEXT,
            user_id BIGINT,
            username TEXT,
            content TEXT,
            badges LIST<TEXT>,
            PRIMARY KEY (stream_id, timestamp, message_id)
        ) WITH CLUSTERING ORDER BY (timestamp DESC)
          AND default_time_to_live = 2592000;  -- 30 days auto-deletion
        ```

        **Cost comparison:**

        | Strategy | Storage (30 days) | Cost/month | Savings |
        |----------|-------------------|------------|---------|
        | **No optimization** | 4.32 PB | $86,400 | 0% |
        | **With TTL** | 4.32 PB | $86,400 | 0% (but prevents growth) |
        | **TTL + Compression (3x)** | 1.44 PB | $28,800 | 67% |
        | **TTL + Compression + Sampling (10%)** | 144 TB | $2,880 | 97% |

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling live streaming from 100K to 10M concurrent viewers.

    **Scaling challenges at 10M viewers:**

    - **Bandwidth:** 30-80 Tbps egress (CDN critical)
    - **Transcoding:** 1M concurrent streams, 50K GPUs
    - **Chat:** 100M messages/minute (1.67M/sec writes)
    - **Storage:** 3.65 exabytes (10 years of VODs)

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Transcoding** | ✅ Yes | GPU fleet (75K GPUs), auto-scaling, edge transcoding |
    | **CDN bandwidth** | ✅ Yes | Multi-tier CDN, 95% cache hit rate, origin shield |
    | **Chat WebSockets** | ✅ Yes | Shard to 10K connections/server, Redis pub/sub |
    | **Cassandra writes (chat)** | ✅ Yes | 500-node cluster, SSD, TTL, compression |
    | **S3 storage** | 🟢 No | Infinite scalability, use S3 Glacier for old VODs |
    | **RTMP ingestion** | 🟡 Approaching | Shard by stream_id, 100 ingest servers |

    ---

    ## Performance Optimizations

    ### 1. Transcoding Optimization

    **Problem:** Transcoding costs $37.5M/month (75K GPUs).

    **Optimizations:**

    | Optimization | Savings | Trade-off |
    |--------------|---------|-----------|
    | **Adaptive transcoding** | 30% | Skip transcoding if 0 viewers |
    | **Dynamic quality ladder** | 15% | Fewer qualities for small streams |
    | **Hardware encoding (NVENC)** | 50% | vs software (x264), slight quality loss |
    | **Edge transcoding** | 20% | Transcode at edge, reduce data transfer |

    **Adaptive transcoding:**

    ```python
    def should_transcode_quality(stream_id: str, quality: str) -> bool:
        """
        Decide whether to transcode a specific quality

        Skip transcoding qualities with 0 viewers (saves GPU)
        """
        viewer_count = get_viewer_count_by_quality(stream_id, quality)

        # Always transcode at least 2 qualities (360p, 720p)
        if quality in ['360p', '720p']:
            return True

        # Transcode 1080p only if viewers request it
        if quality == '1080p' and viewer_count > 0:
            return True

        # Skip other qualities if no viewers
        return False
    ```

    **Result:** 30% reduction in transcoding cost ($11M/month savings).

    ---

    ### 2. CDN Optimization

    **Problem:** CDN costs $1.8M/month even with multi-tier architecture.

    **Optimizations:**

    | Optimization | Savings | Implementation |
    |--------------|---------|----------------|
    | **Increase cache TTL** | 10% | Longer TTL = fewer origin requests |
    | **Preload popular streams** | 5% | Pre-warm CDN before large events |
    | **Regional CDN selection** | 15% | Cheaper CDN in low-cost regions (Asia) |
    | **P2P delivery (WebTorrent)** | 20% | Viewers share with each other |

    **P2P delivery:**

    ```javascript
    // WebTorrent for P2P video delivery
    // Viewers share video segments with each other (reduces CDN load)

    const client = new WebTorrent();

    // Download stream via P2P + CDN hybrid
    client.add(streamMagnetLink, {
      // Download from peers (P2P)
      maxConns: 50,  // Max peer connections

      // Fallback to CDN if P2P slow
      urlList: [
        'https://cdn.example.com/streams/abc123/720p/'
      ]
    }, (torrent) => {
      // Play stream
      torrent.files[0].appendTo('#video-player');
    });
    ```

    **Result:** 20% reduction in CDN cost ($360K/month savings) for large streams.

    ---

    ### 3. Chat Optimization

    **Problem:** Cassandra cluster costs $216K/month (500 nodes for 1.67M writes/sec).

    **Optimizations:**

    | Optimization | Savings | Trade-off |
    |--------------|---------|-----------|
    | **Message sampling** | 70% | Store only 10% of messages from large streams |
    | **Compression** | 50% | gzip messages (3x compression) |
    | **Batch writes** | 30% | Batch 100 messages per write |
    | **SSD tier optimization** | 20% | Hot data on SSD, cold on HDD |

    **Batch writes:**

    ```python
    class ChatBatchWriter:
        """
        Batch chat messages for efficient Cassandra writes

        Reduces write amplification from 1.67M writes/sec to 17K writes/sec
        """

        BATCH_SIZE = 100
        FLUSH_INTERVAL = 1.0  # seconds

        def __init__(self):
            self.buffer = defaultdict(list)  # stream_id -> messages
            self.last_flush = time.time()

        def add_message(self, stream_id: str, message: dict):
            """Add message to batch buffer"""
            self.buffer[stream_id].append(message)

            # Flush if buffer full
            if len(self.buffer[stream_id]) >= self.BATCH_SIZE:
                self.flush_stream(stream_id)

            # Flush if interval expired
            if time.time() - self.last_flush > self.FLUSH_INTERVAL:
                self.flush_all()

        def flush_stream(self, stream_id: str):
            """Flush messages for a single stream"""
            messages = self.buffer[stream_id]

            if not messages:
                return

            # Batch INSERT
            cassandra.batch_execute(
                "INSERT INTO chat_messages ...",
                messages
            )

            logger.debug(f"Flushed {len(messages)} messages for stream {stream_id}")
            self.buffer[stream_id] = []
    ```

    **Result:** 100x reduction in Cassandra write ops (1.67M → 17K writes/sec), 50% cost savings ($108K/month).

    ---

    ## Cost Optimization

    **Monthly cost at 10M viewers, 1M streamers:**

    | Component | Cost | Optimization | Optimized Cost |
    |-----------|------|--------------|----------------|
    | **Transcoding (75K GPUs)** | $37,500,000 | Adaptive transcoding | $26,250,000 |
    | **CDN (multi-tier)** | $1,832,900 | P2P delivery | $1,466,320 |
    | **Cassandra (chat)** | $216,000 | Batch writes, compression | $108,000 |
    | **S3 storage (live buffer)** | $107,500 | Reduce buffer to 1 hour | $53,750 |
    | **EC2 (API servers)** | $86,400 | Spot instances | $25,920 |
    | **PostgreSQL (metadata)** | $43,200 | Read replicas, caching | $21,600 |
    | **Redis cache** | $43,200 | Compress cache data | $21,600 |
    | **RTMP ingest** | $21,600 | Auto-scaling | $15,120 |
    | **WebSocket chat servers** | $86,400 | Connection pooling | $60,480 |
    | **Monitoring & logging** | $10,800 | Sample 10% of logs | $3,240 |
    | **Total** | **$39,948,000/month** | | **$28,026,030/month** |

    **Savings:** $11.9M/month (30% reduction)

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold | Action |
    |--------|--------|-----------------|--------|
    | **Glass-to-glass latency** | < 3s | > 5s | Check transcoding queue, CDN health |
    | **Transcoding latency** | < 1s | > 2s | Scale up GPU fleet |
    | **CDN cache hit rate** | > 95% | < 90% | Increase TTL, check purge frequency |
    | **Chat delivery latency** | < 100ms | > 500ms | Scale chat servers, check Redis |
    | **Ingest failure rate** | < 0.1% | > 1% | Check RTMP server health |
    | **Viewer buffering ratio** | < 5% | > 10% | Check CDN, reduce bitrate |
    | **Stream availability** | > 99.9% | < 99% | Check transcoding, CDN failover |

    ---

    ## Disaster Recovery

    **Failure scenarios:**

    | Failure | Impact | Recovery |
    |---------|--------|----------|
    | **CDN outage** | 30% of viewers can't watch | Automatic failover to Tier 2 CDN (30s) |
    | **Transcoding cluster failure** | New streams can't start | Auto-scale new GPU servers (5 min) |
    | **Redis cluster failure** | Chat stops working | Failover to replica cluster (10s) |
    | **S3 outage** | VODs unavailable, live buffer lost | Multi-region replication (instant) |
    | **Cassandra cluster failure** | Chat history lost | Restore from backups (1 hour) |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **GPU transcoding:** 10x more efficient than CPU, 34x cheaper than cloud
    2. **Multi-tier CDN:** 95% cache hit rate, $4.9M/month savings
    3. **HLS with 2s segments:** Balance latency (<3s) and CDN cacheability
    4. **Three-tier latency model:** Standard (3s), low-latency (1s), ultra-low (<500ms WebRTC)
    5. **Sharded chat:** 10K connections per server, Redis pub/sub for broadcast
    6. **Cassandra for chat:** 1.67M writes/sec, TTL for auto-deletion
    7. **Adaptive bitrate:** 6 quality outputs (160p-1080p60) for smooth playback

    ---

    ## Interview Tips

    ✅ **Emphasize bandwidth scale** - 30-80 Tbps egress is massive, CDN critical

    ✅ **Discuss latency trade-offs** - Standard (3s) vs low-latency (<1s) vs WebRTC (<500ms)

    ✅ **Transcoding complexity** - GPU vs CPU, real-time requirement, cost optimization

    ✅ **CDN strategy** - Multi-tier architecture, origin shield, cache optimization

    ✅ **Chat scalability** - 100M messages/minute, sharding, pub/sub, rate limiting

    ✅ **Cost optimization** - Adaptive transcoding, P2P delivery, compression

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to achieve <1s latency?"** | LL-HLS (partial segments), WebRTC (direct), edge transcoding, reduce buffer |
    | **"How to handle 1M concurrent streams?"** | GPU transcoding (75K GPUs), auto-scaling, adaptive transcoding (skip unused qualities) |
    | **"How to serve 80 Tbps bandwidth?"** | Multi-tier CDN, 95% cache hit rate, origin shield, P2P delivery |
    | **"How to handle chat for 100K viewer stream?"** | Shard chat servers (10K/server), Redis pub/sub, rate limiting, batch writes |
    | **"How to reduce costs?"** | Adaptive transcoding (30%), P2P delivery (20%), chat sampling (70%), spot instances |
    | **"What if CDN goes down?"** | Multi-tier redundancy, automatic failover to Tier 2, P2P fallback |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Twitch, YouTube Live, Facebook Live, Instagram Live, TikTok Live, Discord

---

*Master this problem and you'll be ready for: Video streaming platforms, live events, gaming platforms, video conferencing*
