# Design YouTube (Video Streaming Platform)

A video streaming platform where users can upload, watch, like, comment, and share videos. Supports billions of users watching billions of videos daily.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 2B users, 500 hours of video uploaded/min, 1B hours watched/day |
| **Key Challenges** | Video encoding, CDN delivery, recommendation system, storage optimization |
| **Core Concepts** | Adaptive bitrate streaming, video transcoding, CDN, recommendation ML |
| **Companies** | Google, Netflix, Amazon Prime, Hulu, Disney+, TikTok |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Upload Video** | Users can upload videos (up to 12 hours, max 256 GB) | P0 (Must have) |
    | **Watch Video** | Stream videos with adaptive quality | P0 (Must have) |
    | **Search Videos** | Search by title, description, tags | P0 (Must have) |
    | **Recommendations** | Personalized video suggestions | P0 (Must have) |
    | **Like/Comment** | Users can like and comment on videos | P1 (Should have) |
    | **Subscribe** | Users can subscribe to channels | P1 (Should have) |
    | **Analytics** | View counts, watch time for creators | P2 (Nice to have) |
    | **Live Streaming** | Real-time video broadcast | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Content moderation (AI/ML models)
    - Copyright detection (Content ID)
    - Monetization (ads system)
    - YouTube Premium features
    - Community posts/Stories

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users expect 24/7 access to videos |
    | **Latency** | < 2s to start playback | Fast video start critical for retention |
    | **Reliability** | No video loss | Uploaded videos are permanent |
    | **Scalability** | Billions of daily views | Must handle viral videos (100M views in hours) |
    | **Cost-efficiency** | Optimize storage/bandwidth | Video storage/delivery is extremely expensive |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total Users: 2B monthly active users
    Daily Active Users (DAU): 1B (50% engagement)

    Video uploads:
    - Upload rate: 500 hours of video per minute
    - Daily uploads: 500 × 60 × 24 = 720,000 hours/day
    - Average video length: 10 minutes
    - Daily video count: 720,000 × 6 = 4.32M videos/day

    Video views:
    - Watch time per DAU: 60 minutes/day (average)
    - Daily watch time: 1B × 60 min = 60B minutes = 1B hours/day
    - Average video length: 10 minutes
    - Daily views: 6B views/day
    - View QPS: 6B / 86,400 = ~69,400 views/sec

    Peak traffic: 3x average = ~208,200 concurrent streams
    ```

    ### Storage Estimates

    ```
    Video storage per upload:
    - Original: 1 GB per 10 minutes (1080p raw)
    - Multiple resolutions: 4K, 1080p, 720p, 480p, 360p, 240p
    - Per resolution (10 min video):
      - 4K (2160p): 3 GB
      - 1080p: 500 MB
      - 720p: 250 MB
      - 480p: 125 MB
      - 360p: 60 MB
      - 240p: 30 MB
    - Total per 10-min video: ~4 GB (all resolutions)
    - Compression: H.265 reduces by 50% → 2 GB per video

    Daily storage:
    - Videos: 4.32M videos × 2 GB = 8.64 TB/day
    - Yearly: 8.64 TB × 365 = 3.15 PB/year
    - 10 years: 31.5 PB

    Thumbnails:
    - 3 thumbnails per video × 50 KB = 150 KB per video
    - Daily: 4.32M × 150 KB = 648 GB/day
    - 10 years: 2.36 PB

    Total: 31.5 PB (videos) + 2.36 PB (thumbnails) ≈ 34 PB
    ```

    ### Bandwidth Estimates

    ```
    Upload bandwidth:
    - 4.32M videos/day × 2 GB = 8.64 TB/day
    - Ingress: 8.64 TB / 86,400 sec = 100 MB/sec ≈ 800 Mbps

    Streaming bandwidth:
    - Average bitrate: 2 Mbps (720p)
    - Concurrent streams: 69,400 streams
    - Egress: 69,400 × 2 Mbps = 138,800 Mbps ≈ 139 Gbps
    - Peak: 3x = 417 Gbps

    CDN critical at this scale!
    ```

    ### Memory Estimates (Caching)

    ```
    Video metadata cache:
    - Hot videos (last 24h): 4.32M videos
    - Metadata per video: 5 KB
    - Cache: 4.32M × 5 KB = 21.6 GB

    Popular video chunks (CDN edge):
    - Top 1% videos: 43,200 videos
    - Average video: 2 GB across resolutions
    - Cache: 43,200 × 2 GB = 86.4 TB (distributed across CDN nodes)

    User sessions:
    - Concurrent users: 100M
    - Session data: 10 KB per user
    - Total: 100M × 10 KB = 1 TB

    Total cache: 22 GB (metadata) + 86 TB (video chunks) + 1 TB (sessions) ≈ 87 TB
    ```

    ---

    ## Key Assumptions

    1. Average video length: 10 minutes
    2. Multiple quality levels: 240p to 4K
    3. 50% of users watch daily (1B DAU from 2B MAU)
    4. Average watch time: 60 minutes per DAU per day
    5. Read-heavy: 6B views vs 4.32M uploads (~1,400:1 ratio)
    6. CDN serves 90% of video traffic

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Upload-encode-distribute pipeline** - Async video processing
    2. **CDN-first delivery** - Serve videos from edge locations
    3. **Adaptive bitrate streaming** - Match quality to user bandwidth
    4. **Metadata in database, video in object storage** - Separate concerns
    5. **ML-based recommendations** - Personalized video discovery

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Player<br/>HTML5 Video]
            Mobile[Mobile Apps<br/>iOS/Android]
            TV[Smart TV Apps]
        end

        subgraph "CDN Layer"
            CDN[CDN<br/>CloudFront/Akamai<br/>Video chunks]
        end

        subgraph "API Gateway"
            LB[Load Balancer]
            API_GW[API Gateway]
        end

        subgraph "Application Services"
            Upload_Service[Upload Service<br/>Chunked upload]
            Stream_Service[Streaming Service<br/>Video delivery]
            Search_Service[Search Service<br/>Elasticsearch]
            Rec_Service[Recommendation Service<br/>ML models]
            Analytics_Service[Analytics Service<br/>View tracking]
        end

        subgraph "Processing Pipeline"
            Transcode[Transcoding Service<br/>FFmpeg cluster]
            Thumbnail[Thumbnail Generator]
            Quality[Quality Check]
        end

        subgraph "Caching"
            Redis_Meta[Redis<br/>Metadata cache]
            Redis_Session[Redis<br/>User sessions]
        end

        subgraph "Storage"
            Video_DB[(Video Metadata DB<br/>Cassandra)]
            User_DB[(User DB<br/>PostgreSQL)]
            S3_Raw[S3<br/>Raw uploads]
            S3_Encoded[S3<br/>Encoded videos]
            S3_Thumb[S3<br/>Thumbnails]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        subgraph "ML Pipeline"
            Feature_Store[Feature Store]
            Model_Serving[Model Serving<br/>TensorFlow Serving]
            Training[Offline Training<br/>Spark/Airflow]
        end

        Web --> CDN
        Mobile --> CDN
        TV --> CDN

        Web --> LB
        Mobile --> LB
        TV --> LB

        LB --> API_GW

        API_GW --> Upload_Service
        API_GW --> Stream_Service
        API_GW --> Search_Service
        API_GW --> Rec_Service
        API_GW --> Analytics_Service

        Upload_Service --> S3_Raw
        Upload_Service --> Kafka
        Upload_Service --> Video_DB

        Kafka --> Transcode
        Kafka --> Thumbnail
        Kafka --> Analytics_Service

        Transcode --> S3_Encoded
        Thumbnail --> S3_Thumb
        Transcode --> Video_DB

        Stream_Service --> CDN
        CDN --> S3_Encoded

        Stream_Service --> Redis_Meta
        Stream_Service --> Video_DB

        Search_Service --> Elasticsearch[(Elasticsearch)]
        Rec_Service --> Model_Serving
        Model_Serving --> Feature_Store

        Analytics_Service --> Kafka
        Kafka --> Training
        Training --> Feature_Store

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Redis_Meta fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style Video_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style S3_Raw fill:#f3e5f5
        style S3_Encoded fill:#f3e5f5
        style S3_Thumb fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **CDN** | Serve videos globally with < 50ms latency, reduce origin load by 90% | Direct streaming (high latency, expensive bandwidth) |
    | **S3 (Video Storage)** | Unlimited storage, 99.999999999% durability, cost-effective | Database BLOBs (expensive, doesn't scale) |
    | **Cassandra (Metadata)** | Fast video metadata queries, horizontal scaling | PostgreSQL (can't handle write volume for analytics) |
    | **FFmpeg (Transcoding)** | Industry-standard video encoder, supports all codecs | Cloud transcoding only (expensive, vendor lock-in) |
    | **Kafka** | High-throughput event streaming for video processing pipeline | Direct processing (no retry, no replay) |
    | **Elasticsearch** | Fast full-text search, complex queries | Database LIKE queries (too slow for billions of videos) |

    **Key Trade-off:** We chose **cost optimization over quality**. Not all videos encoded to 4K (only popular ones), older videos moved to cheaper storage.

    ---

    ## API Design

    ### 1. Upload Video

    **Request (Initiate Upload):**
    ```http
    POST /api/v1/videos/upload/init
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "title": "My Awesome Video",
      "description": "Learn system design",
      "tags": ["education", "tech", "system design"],
      "category": "Education",
      "visibility": "public",  // public, unlisted, private
      "file_size": 1073741824,  // 1 GB
      "file_name": "video.mp4"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "video_id": "abc123xyz",
      "upload_url": "https://upload.youtube.com/upload/abc123xyz",
      "chunk_size": 10485760,  // 10 MB chunks
      "expires_at": "2026-01-29T11:30:00Z"
    }
    ```

    **Upload Chunks:**
    ```http
    PUT https://upload.youtube.com/upload/abc123xyz?chunk=1
    Content-Type: application/octet-stream
    Content-Range: bytes 0-10485759/1073741824

    <binary data>
    ```

    **Design Notes:**

    - Chunked upload for reliability (resume failed uploads)
    - Signed upload URL with expiration
    - Direct upload to S3 (bypasses API servers)
    - Async processing (return immediately, transcode in background)

    ---

    ### 2. Watch Video (Get Stream URL)

    **Request:**
    ```http
    GET /api/v1/videos/{video_id}/stream
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "video_id": "abc123xyz",
      "title": "My Awesome Video",
      "manifest_url": "https://cdn.youtube.com/videos/abc123xyz/manifest.m3u8",
      "qualities": [
        {
          "resolution": "1080p",
          "bitrate": 5000000,
          "url": "https://cdn.youtube.com/videos/abc123xyz/1080p/manifest.m3u8"
        },
        {
          "resolution": "720p",
          "bitrate": 2500000,
          "url": "https://cdn.youtube.com/videos/abc123xyz/720p/manifest.m3u8"
        }
      ],
      "thumbnail_url": "https://cdn.youtube.com/thumbnails/abc123xyz.jpg",
      "duration": 600,  // seconds
      "views": 1523456
    }
    ```

    **Design Notes:**

    - Adaptive bitrate streaming (HLS/DASH)
    - CDN URLs for all video qualities
    - Manifest file contains chunk URLs

    ---

    ## Database Schema

    ### Video Metadata (Cassandra)

    ```sql
    -- Videos table
    CREATE TABLE videos (
        video_id UUID PRIMARY KEY,
        uploader_id UUID,
        title TEXT,
        description TEXT,
        duration INT,  -- seconds
        upload_date TIMESTAMP,
        view_count COUNTER,
        like_count COUNTER,
        comment_count COUNTER,
        tags LIST<TEXT>,
        category TEXT,
        thumbnail_url TEXT,
        manifest_url TEXT,  -- HLS manifest
        status TEXT,  -- processing, ready, failed
        INDEX idx_uploader (uploader_id),
        INDEX idx_category (category),
        INDEX idx_upload_date (upload_date)
    );

    -- Channel videos (for creator page)
    CREATE TABLE channel_videos (
        uploader_id UUID,
        upload_date TIMESTAMP,
        video_id UUID,
        PRIMARY KEY (uploader_id, upload_date)
    ) WITH CLUSTERING ORDER BY (upload_date DESC);

    -- Video views (analytics)
    CREATE TABLE video_views (
        video_id UUID,
        user_id UUID,
        view_date TIMESTAMP,
        watch_duration INT,  -- seconds watched
        PRIMARY KEY (video_id, user_id, view_date)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Video Upload Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant Upload_API
        participant S3_Raw
        participant Video_DB
        participant Kafka
        participant Transcode
        participant S3_Encoded
        participant CDN

        Client->>Upload_API: POST /api/v1/videos/upload/init
        Upload_API->>S3_Raw: Generate presigned URL
        S3_Raw-->>Upload_API: Signed upload URL
        Upload_API->>Video_DB: INSERT video (status: uploading)
        Upload_API-->>Client: upload_url, video_id

        loop Upload chunks
            Client->>S3_Raw: PUT chunk (10 MB)
            S3_Raw-->>Client: Chunk uploaded
        end

        Client->>Upload_API: POST /api/v1/videos/{video_id}/complete
        Upload_API->>Video_DB: UPDATE status = processing
        Upload_API->>Kafka: Publish video_uploaded event
        Upload_API-->>Client: Processing started

        Kafka->>Transcode: video_uploaded event
        Transcode->>S3_Raw: Download raw video
        Transcode->>Transcode: Encode to multiple resolutions<br/>(4K, 1080p, 720p, 480p, 360p, 240p)
        Transcode->>S3_Encoded: Upload encoded videos (HLS chunks)
        Transcode->>S3_Encoded: Generate manifest.m3u8
        Transcode->>Video_DB: UPDATE status = ready, manifest_url
        Transcode->>CDN: Invalidate cache (if re-upload)

        CDN->>S3_Encoded: Pull video chunks (lazy loading)
    ```

    **Flow Explanation:**

    1. **Init upload** - Get presigned S3 URL (client uploads directly to S3)
    2. **Chunked upload** - 10 MB chunks, resume on failure
    3. **Complete upload** - Trigger processing pipeline
    4. **Async transcoding** - Encode to 6 resolutions (4K to 240p)
    5. **Generate HLS** - Split into 10-second chunks, create manifest
    6. **Update status** - Mark video as ready
    7. **CDN population** - Lazy loading, populate on first request

    **Latency:**
    - Upload: Depends on file size and bandwidth
    - Transcoding: 10-30 minutes for 1-hour video (parallel encoding)
    - Available for streaming: Within minutes of upload completion

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical YouTube subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Video Encoding** | How to optimize video for streaming? | Adaptive bitrate + H.265 codec + HLS/DASH |
    | **CDN Strategy** | How to stream to billions globally? | Multi-tier CDN + origin shield + edge caching |
    | **Recommendation System** | How to suggest relevant videos? | Collaborative filtering + deep learning + A/B testing |
    | **Storage Optimization** | How to reduce 34 PB storage cost? | Tiered storage + deduplication + cold archival |

    ---

    === "🎬 Video Encoding & Streaming"

        ## The Challenge

        **Problem:** Stream videos globally with minimal buffering, adapt to varying bandwidth.

        **Naive approach:** Store one high-quality video, stream directly. **Fails:** Users with slow connections can't watch.

        **Solution:** Adaptive Bitrate Streaming (HLS/DASH)

        ---

        ## Adaptive Bitrate Streaming

        **Concept:** Encode video at multiple bitrates, player switches quality based on bandwidth.

        **How it works:**

        1. **Encode multiple qualities:** 240p, 360p, 480p, 720p, 1080p, 4K
        2. **Split into chunks:** 10-second segments
        3. **Player adapts:** Monitor bandwidth, switch quality seamlessly

        **HLS (HTTP Live Streaming):**

        ```
        video_abc123/
        ├── manifest.m3u8 (master playlist)
        ├── 240p/
        │   ├── segment_0.ts (10 seconds)
        │   ├── segment_1.ts
        │   └── playlist.m3u8
        ├── 720p/
        │   ├── segment_0.ts
        │   ├── segment_1.ts
        │   └── playlist.m3u8
        └── 1080p/
            ├── segment_0.ts
            ├── segment_1.ts
            └── playlist.m3u8
        ```

        **Master manifest (manifest.m3u8):**
        ```m3u8
        #EXTM3U
        #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
        360p/playlist.m3u8
        #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
        720p/playlist.m3u8
        #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
        1080p/playlist.m3u8
        ```

        ---

        ## Video Transcoding Pipeline

        **Using FFmpeg for parallel encoding:**

        ```python
        import subprocess
        import boto3
        from concurrent.futures import ThreadPoolExecutor

        class VideoTranscoder:
            """Transcode videos to multiple resolutions"""

            RESOLUTIONS = {
                '4k': {'width': 3840, 'height': 2160, 'bitrate': '15000k'},
                '1080p': {'width': 1920, 'height': 1080, 'bitrate': '5000k'},
                '720p': {'width': 1280, 'height': 720, 'bitrate': '2500k'},
                '480p': {'width': 854, 'height': 480, 'bitrate': '1000k'},
                '360p': {'width': 640, 'height': 360, 'bitrate': '800k'},
                '240p': {'width': 426, 'height': 240, 'bitrate': '400k'}
            }

            def __init__(self, s3_client):
                self.s3 = s3_client

            def transcode_video(self, video_id: str, input_path: str):
                """
                Transcode video to multiple resolutions in parallel

                Args:
                    video_id: Unique video identifier
                    input_path: Path to raw uploaded video

                Returns:
                    dict: Manifest URLs for each resolution
                """
                manifest_urls = {}

                # Transcode all resolutions in parallel
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = []
                    for resolution, config in self.RESOLUTIONS.items():
                        future = executor.submit(
                            self._transcode_resolution,
                            video_id,
                            input_path,
                            resolution,
                            config
                        )
                        futures.append((resolution, future))

                    # Collect results
                    for resolution, future in futures:
                        manifest_url = future.result()
                        manifest_urls[resolution] = manifest_url

                # Generate master manifest
                master_manifest = self._create_master_manifest(manifest_urls)
                master_url = self._upload_manifest(video_id, master_manifest)

                return {'master_url': master_url, 'resolutions': manifest_urls}

            def _transcode_resolution(
                self,
                video_id: str,
                input_path: str,
                resolution: str,
                config: dict
            ) -> str:
                """
                Transcode to specific resolution using FFmpeg

                Returns:
                    S3 URL of HLS manifest
                """
                output_dir = f"/tmp/{video_id}/{resolution}"
                os.makedirs(output_dir, exist_ok=True)

                # FFmpeg command for HLS encoding
                command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-vf', f"scale={config['width']}:{config['height']}",
                    '-c:v', 'libx265',  # H.265 codec (better compression)
                    '-b:v', config['bitrate'],
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-f', 'hls',
                    '-hls_time', '10',  # 10-second segments
                    '-hls_list_size', '0',
                    '-hls_segment_filename', f"{output_dir}/segment_%03d.ts",
                    f"{output_dir}/playlist.m3u8"
                ]

                # Run FFmpeg
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"FFmpeg failed: {result.stderr}")

                # Upload segments to S3
                manifest_url = self._upload_segments(video_id, resolution, output_dir)

                return manifest_url

            def _upload_segments(self, video_id: str, resolution: str, output_dir: str) -> str:
                """Upload HLS segments to S3"""
                s3_prefix = f"videos/{video_id}/{resolution}/"

                # Upload all segments
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    s3_key = f"{s3_prefix}{file}"

                    self.s3.upload_file(
                        file_path,
                        'youtube-videos',
                        s3_key,
                        ExtraArgs={
                            'ContentType': 'application/vnd.apple.mpegurl' if file.endswith('.m3u8') else 'video/MP2T'
                        }
                    )

                manifest_url = f"https://cdn.youtube.com/{s3_prefix}playlist.m3u8"
                return manifest_url
        ```

        ---

        ## Codec Comparison

        | Codec | Compression | Quality | Encoding Speed | Browser Support |
        |-------|-------------|---------|----------------|-----------------|
        | **H.264 (AVC)** | Good | Good | Fast | 100% |
        | **H.265 (HEVC)** | Excellent (50% smaller) | Better | Slower (2x) | 80% (Safari, Edge) |
        | **VP9** | Excellent | Better | Slow | 90% (Chrome, Firefox) |
        | **AV1** | Best (30% smaller than H.265) | Best | Very slow (10x) | 70% (modern browsers) |

        **YouTube's strategy:** Encode in H.264 (baseline), H.265 (popular videos), AV1 (trending videos)

        ---

        ## Streaming Protocols

        **HLS vs DASH:**

        | Aspect | HLS (Apple) | DASH (MPEG) |
        |--------|-------------|-------------|
        | **Browser support** | All browsers | Chrome, Firefox (not Safari) |
        | **Segment format** | MPEG-TS (.ts) | MP4 (.m4s) |
        | **Manifest** | .m3u8 | .mpd (XML) |
        | **Encryption** | AES-128 | CENC |
        | **Use case** | **Production (YouTube uses HLS)** | Alternative |

        **Recommendation:** Use HLS for widest compatibility. YouTube primarily uses HLS.

    === "🌐 CDN Strategy"

        ## The Challenge

        **Problem:** Stream 139 Gbps to users worldwide with < 50ms latency.

        **Cost:** Direct S3 streaming would cost $12.5M/month (139 Gbps × $0.09/GB × 30 days).

        **Solution:** Multi-tier CDN architecture

        ---

        ## CDN Architecture

        ```
        User Request
        ↓
        Edge CDN (Tier 1) - 2,000+ locations worldwide
        ├─ Hit: Serve immediately (< 20ms)
        └─ Miss ↓
              Regional CDN (Tier 2) - 100 locations
              ├─ Hit: Serve (< 50ms)
              └─ Miss ↓
                    Origin Shield (Tier 3) - 10 locations
                    ├─ Hit: Serve (< 100ms)
                    └─ Miss ↓
                          S3 Origin - Single source of truth
                          └─ Fetch and cache up the chain
        ```

        **Benefits:**

        - **Edge hit (90%):** Serve from nearest location (< 20ms)
        - **Regional hit (8%):** Still close to user (< 50ms)
        - **Origin shield hit (1.5%):** Protect origin from thundering herd
        - **Origin miss (0.5%):** Only for unpopular/new videos

        ---

        ## CDN Configuration

        ```javascript
        // CloudFront distribution config
        {
          "Origins": [
            {
              "Id": "s3-origin",
              "DomainName": "youtube-videos.s3.amazonaws.com",
              "CustomHeaders": [
                {
                  "HeaderName": "X-Origin-Shield",
                  "HeaderValue": "enabled"
                }
              ]
            }
          ],
          "CacheBehaviors": [
            {
              "PathPattern": "/videos/*",
              "TargetOriginId": "s3-origin",
              "ViewerProtocolPolicy": "https-only",
              "Compress": false,  // Video already compressed
              "MinTTL": 86400,     // 1 day
              "DefaultTTL": 2592000,  // 30 days
              "MaxTTL": 31536000,     // 1 year (immutable videos)
              "AllowedMethods": ["GET", "HEAD"],
              "ForwardedValues": {
                "QueryString": false,
                "Headers": ["Range"],  // Support byte-range requests
                "Cookies": {
                  "Forward": "none"
                }
              }
            }
          ],
          "CustomErrorResponses": [
            {
              "ErrorCode": 404,
              "ErrorCachingMinTTL": 300  // Cache 404s for 5 min
            }
          ]
        }
        ```

        ---

        ## Cost Optimization

        **Without CDN:**
        ```
        Data transfer: 139 Gbps × 86,400 sec/day = 12,009,600 GB/day
        Daily cost: 12M GB × $0.09/GB = $1,080,864/day
        Monthly: $32.4M
        ```

        **With CDN (90% edge hit):**
        ```
        Origin requests: 10% of 139 Gbps = 13.9 Gbps
        S3 transfer: 1.2M GB/day × $0.09/GB = $108,086/day
        CDN cost: 12M GB/day × $0.020/GB = $240,000/day
        Total: $348,086/day ≈ $10.4M/month
        ```

        **Savings: $22M/month (68% reduction)**

        ---

        ## Cache Warmingfor Popular Videos

        **Problem:** New viral video gets 10M views in first hour → cold CDN cache.

        **Solution:** Proactive cache warming

        ```python
        class CDNWarmer:
            """Pre-populate CDN with trending videos"""

            def warm_video(self, video_id: str, resolutions: List[str]):
                """
                Trigger CDN cache for video across edge locations

                Args:
                    video_id: Video to warm
                    resolutions: ['1080p', '720p', '480p']
                """
                edge_locations = self.get_top_edge_locations(100)  # Top 100 locations

                for location in edge_locations:
                    for resolution in resolutions:
                        manifest_url = f"https://cdn.youtube.com/videos/{video_id}/{resolution}/playlist.m3u8"

                        # Request manifest from specific edge location
                        self._request_from_edge(manifest_url, location)

                        # Pre-fetch first 3 segments (first 30 seconds)
                        for i in range(3):
                            segment_url = f"https://cdn.youtube.com/videos/{video_id}/{resolution}/segment_{i:03d}.ts"
                            self._request_from_edge(segment_url, location)

                logger.info(f"Warmed {video_id} in {len(edge_locations)} edge locations")
        ```

        **When to warm:**
        - Newly uploaded videos from popular channels
        - Videos trending on social media
        - Videos in "Trending" section

    === "🤖 Recommendation System"

        ## The Challenge

        **Problem:** With 4.32M new videos daily, how to show users relevant content?

        **Goal:** Maximize watch time and engagement.

        **Approach:** Two-stage recommendation

        1. **Candidate Generation:** Narrow 4.32M videos to top 1,000 candidates
        2. **Ranking:** Rank top 1,000 to show best 20

        ---

        ## Candidate Generation

        **Methods:**

        1. **Collaborative Filtering:** Users who watched A also watched B
        2. **Content-Based:** Similar to videos you watched (tags, category)
        3. **Trending:** Popular videos right now
        4. **Subscriptions:** New videos from subscribed channels

        **Implementation:**

        ```python
        class CandidateGenerator:
            """Generate candidate videos for recommendation"""

            def generate_candidates(self, user_id: str, count: int = 1000) -> List[str]:
                """
                Generate candidate videos from multiple sources

                Returns:
                    List of video_ids (top 1000)
                """
                candidates = set()

                # 1. Collaborative filtering (40% of candidates)
                collab_candidates = self._collaborative_filtering(user_id, 400)
                candidates.update(collab_candidates)

                # 2. Content-based (30%)
                content_candidates = self._content_based(user_id, 300)
                candidates.update(content_candidates)

                # 3. Trending videos (20%)
                trending_candidates = self._trending_videos(200)
                candidates.update(trending_candidates)

                # 4. Subscribed channels (10%)
                subscription_candidates = self._subscription_videos(user_id, 100)
                candidates.update(subscription_candidates)

                return list(candidates)[:count]

            def _collaborative_filtering(self, user_id: str, count: int) -> List[str]:
                """
                Find users with similar watch history, recommend their videos

                Algorithm: User-User Collaborative Filtering
                """
                # Get user's watch history
                user_watches = db.query(
                    "SELECT video_id FROM video_views WHERE user_id = %s ORDER BY view_date DESC LIMIT 100",
                    (user_id,)
                )
                watched_ids = [v['video_id'] for v in user_watches]

                # Find similar users (cosine similarity on watch history)
                similar_users = self._find_similar_users(user_id, watched_ids, top_k=50)

                # Get videos watched by similar users (but not by current user)
                recommended_videos = []
                for similar_user_id, similarity_score in similar_users:
                    their_watches = db.query(
                        "SELECT video_id FROM video_views WHERE user_id = %s ORDER BY view_date DESC LIMIT 20",
                        (similar_user_id,)
                    )

                    for video in their_watches:
                        if video['video_id'] not in watched_ids:
                            recommended_videos.append(video['video_id'])

                # Deduplicate and return top N
                return list(set(recommended_videos))[:count]

            def _content_based(self, user_id: str, count: int) -> List[str]:
                """
                Recommend videos similar to user's watch history

                Algorithm: Content-based filtering using tags/category
                """
                # Get user's favorite categories/tags
                user_preferences = db.query("""
                    SELECT v.category, v.tags, COUNT(*) as watch_count
                    FROM video_views vw
                    JOIN videos v ON vw.video_id = v.video_id
                    WHERE vw.user_id = %s
                    GROUP BY v.category, v.tags
                    ORDER BY watch_count DESC
                    LIMIT 10
                """, (user_id,))

                # Find videos matching preferences
                recommended_videos = []
                for pref in user_preferences:
                    similar_videos = db.query("""
                        SELECT video_id FROM videos
                        WHERE category = %s OR %s = ANY(tags)
                        ORDER BY view_count DESC
                        LIMIT 50
                    """, (pref['category'], pref['tags'][0] if pref['tags'] else ''))

                    recommended_videos.extend([v['video_id'] for v in similar_videos])

                return list(set(recommended_videos))[:count]
        ```

        ---

        ## Ranking Model

        **Deep Neural Network for CTR prediction:**

        **Features:**
        - **User features:** Age, location, watch history, subscriptions
        - **Video features:** Title, tags, upload date, duration, engagement
        - **Context features:** Time of day, device, last watched

        **Model architecture:**

        ```python
        import tensorflow as tf

        def build_ranking_model():
            """
            Deep neural network for video ranking

            Predicts: P(user will watch video)
            """
            # Input layers
            user_id = tf.keras.Input(shape=(1,), name='user_id')
            video_id = tf.keras.Input(shape=(1,), name='video_id')
            user_features = tf.keras.Input(shape=(50,), name='user_features')
            video_features = tf.keras.Input(shape=(30,), name='video_features')
            context_features = tf.keras.Input(shape=(10,), name='context_features')

            # Embedding layers
            user_embedding = tf.keras.layers.Embedding(10_000_000, 64)(user_id)
            video_embedding = tf.keras.layers.Embedding(100_000_000, 64)(video_id)

            # Concatenate all features
            concat = tf.keras.layers.Concatenate()([
                tf.keras.layers.Flatten()(user_embedding),
                tf.keras.layers.Flatten()(video_embedding),
                user_features,
                video_features,
                context_features
            ])

            # Deep layers
            x = tf.keras.layers.Dense(512, activation='relu')(concat)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)

            # Output: probability of click
            output = tf.keras.layers.Dense(1, activation='sigmoid', name='ctr')(x)

            model = tf.keras.Model(
                inputs=[user_id, video_id, user_features, video_features, context_features],
                outputs=output
            )

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['auc']
            )

            return model
        ```

        **Training:**
        - **Positive examples:** Videos user watched
        - **Negative examples:** Videos user saw but didn't watch
        - **Labels:** Binary (watched=1, not watched=0)
        - **Re-train:** Daily with fresh data

    === "💾 Storage Optimization"

        ## The Challenge

        **Problem:** 34 PB of storage over 10 years. At $0.023/GB/month = $800K/month!

        **Goal:** Reduce storage cost by 50-70%.

        ---

        ## Tiered Storage Strategy

        **YouTube's approach:** Move old videos to cheaper storage.

        | Age | Storage Tier | Cost/GB/month | Access Pattern |
        |-----|-------------|---------------|----------------|
        | **0-30 days** | S3 Standard | $0.023 | Hot (90% of views) |
        | **30-90 days** | S3 IA | $0.0125 | Warm (8% of views) |
        | **90-365 days** | S3 Glacier | $0.004 | Cold (1.5% of views) |
        | **1+ years** | S3 Glacier Deep Archive | $0.00099 | Rare (0.5% of views) |

        **Lifecycle policy:**

        ```json
        {
          "Rules": [
            {
              "Id": "MoveOldVideos",
              "Status": "Enabled",
              "Prefix": "videos/",
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
        Without lifecycle (all Standard):
        - 34 PB × $0.023/GB/month = $800,720/month

        With lifecycle (50% in cheaper tiers):
        - 17 PB Standard: $400,360/month
        - 10 PB IA: $131,250/month
        - 5 PB Glacier: $21,000/month
        - 2 PB Deep Archive: $2,048/month
        - Total: $554,658/month
        - Savings: $246,062/month (31%)
        ```

        ---

        ## Video Deduplication

        **Problem:** Same video uploaded multiple times (re-uploads, copies).

        **Solution:** Perceptual hash matching

        ```python
        import imagehash
        from PIL import Image

        def deduplicate_video(video_path: str) -> tuple:
            """
            Check if video is duplicate using perceptual hash

            Returns:
                (is_duplicate, existing_video_id)
            """
            # Extract keyframes (every 10 seconds)
            keyframes = extract_keyframes(video_path, interval=10)

            # Generate perceptual hash for each keyframe
            frame_hashes = []
            for frame in keyframes:
                img = Image.fromarray(frame)
                phash = str(imagehash.phash(img))
                frame_hashes.append(phash)

            # Combine frame hashes into video signature
            video_signature = '-'.join(frame_hashes[:10])  # First 10 keyframes

            # Check database for similar signature
            similar = db.query("""
                SELECT video_id, signature
                FROM video_signatures
                WHERE signature LIKE %s
                LIMIT 10
            """, (f"{video_signature[:20]}%",))

            # Calculate similarity score
            for video in similar:
                similarity = calculate_similarity(video_signature, video['signature'])
                if similarity > 0.95:  # 95% similar
                    return (True, video['video_id'])

            # Not a duplicate
            return (False, None)
        ```

        **Savings:** 5-10% storage reduction (common re-uploads, viral videos)

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling YouTube from 1M to 2B users.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Video transcoding** | ✅ Yes | 1,000 FFmpeg workers, parallel encoding, GPU acceleration |
    | **CDN delivery** | ✅ Yes | Multi-tier CDN (edge + regional + origin shield) |
    | **Search** | ✅ Yes | Elasticsearch cluster (200 nodes), sharded by upload_date |
    | **Recommendations** | ✅ Yes | ML model serving with TensorFlow, batch predictions |
    | **Storage** | ✅ Yes | Tiered storage (S3 → Glacier), deduplication |

    ---

    ## Cost Optimization

    **Monthly cost at 2B users:**

    | Component | Cost |
    |-----------|------|
    | **S3 storage** | $554,658 (with lifecycle) |
    | **CDN** | $10,400,000 (with 90% edge hit) |
    | **EC2 (API/transcode)** | $432,000 (2,000 servers) |
    | **Cassandra cluster** | $324,000 (750 nodes) |
    | **Elasticsearch** | $108,000 (200 nodes) |
    | **Redis cache** | $64,800 (300 nodes) |
    | **Total** | **$11.9M/month** |

    **Revenue:** YouTube generates ~$30B/year = $2.5B/month (ads). Infrastructure is 0.5% of revenue.

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Video Start Time** | < 2s | > 5s |
    | **Buffering Ratio** | < 1% | > 5% |
    | **CDN Hit Rate** | > 90% | < 80% |
    | **Transcode Queue** | < 5 min | > 30 min |
    | **Search Latency (P95)** | < 100ms | > 500ms |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Adaptive bitrate streaming** - HLS with multiple resolutions
    2. **Multi-tier CDN** - 90% edge hit, 68% cost savings
    3. **Parallel transcoding** - FFmpeg cluster, 6 resolutions simultaneously
    4. **ML recommendations** - Two-stage (candidate + ranking)
    5. **Tiered storage** - Move old videos to Glacier (31% cost savings)
    6. **Cassandra for metadata** - Handle 69K video views/sec

    ---

    ## Interview Tips

    ✅ **Emphasize CDN importance** - Video delivery is 90% of cost

    ✅ **Discuss adaptive streaming** - HLS/DASH, multiple bitrates

    ✅ **Transcoding pipeline** - FFmpeg, parallel encoding, H.265

    ✅ **Recommendation system** - Collaborative filtering + deep learning

    ✅ **Cost optimization** - Tiered storage, CDN strategy

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle 4K videos?"** | Transcode only popular videos to 4K (top 1%), others max 1080p |
    | **"How to reduce buffering?"** | Adaptive bitrate, pre-fetch next segments, CDN edge caching |
    | **"How to recommend videos?"** | Two-stage: candidate generation (1000 videos) + ML ranking (top 20) |
    | **"How to handle copyright?"** | Content ID system: fingerprint videos, match uploads, auto-block/monetize |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Google, Netflix, Amazon Prime, Hulu, Disney+

---

*Master this problem and you'll be ready for: Netflix, TikTok, Twitch, Vimeo*
