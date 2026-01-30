# Design Netflix

Design a scalable video streaming platform like Netflix that supports on-demand video streaming, personalized recommendations, content management, multi-device playback with resume capability, and efficient content delivery to millions of concurrent users globally.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 200M subscribers, 15M concurrent streams, 100K hours of content |
| **Key Challenges** | Adaptive bitrate streaming, CDN optimization, personalized recommendations, global content delivery |
| **Core Concepts** | HLS streaming, per-title encoding, Open Connect CDN, recommendation ML, multi-device sync |
| **Companies** | Netflix, Amazon Prime, Disney+, Hulu, HBO Max |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Video Streaming** | On-demand playback with adaptive bitrate | P0 (Must have) |
    | **Content Browse** | Browse catalog by genre, trending, new releases | P0 (Must have) |
    | **Search** | Search by title, actor, director, genre | P0 (Must have) |
    | **Recommendations** | Personalized content suggestions | P0 (Must have) |
    | **Multi-Device** | Resume playback across devices | P0 (Must have) |
    | **Profiles** | Multiple user profiles per account | P1 (Should have) |
    | **Watchlist** | Save content to watch later | P1 (Should have) |
    | **Subtitles** | Multiple language subtitles and audio | P1 (Should have) |
    | **Download** | Offline viewing | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Live TV streaming
    - User-generated content
    - Social features (watch parties)
    - Parental controls
    - Payment processing

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Streaming must always work |
    | **Video Start Time** | < 2 seconds | Fast playback startup |
    | **Buffering** | < 1% of playback time | Smooth viewing experience |
    | **Scalability** | 15M concurrent streams | Handle peak viewing hours |
    | **Global CDN** | < 50ms latency to edge | Low-latency content delivery |
    | **Quality** | Adaptive bitrate (360p-4K) | Optimize for bandwidth |

    ---

    ## Capacity Estimation

    ### Users & Traffic

    ```
    Total subscribers: 200 million
    Daily active users (DAU): 100 million (50%)
    Average watch time: 3 hours/day per user
    Peak concurrent viewers: 15 million

    Video starts per day: 100M users × 5 videos/day = 500M starts
    Total viewing hours: 100M × 3 hours = 300M hours/day
    ```

    ### Storage Requirements

    ```
    Content library: 100,000 hours of video

    Per-quality storage (1 hour):
    - 4K (25 Mbps): 11.25 GB
    - 1080p (8 Mbps): 3.6 GB
    - 720p (5 Mbps): 2.25 GB
    - 480p (2.5 Mbps): 1.125 GB
    - 360p (1 Mbps): 450 MB

    Total per hour (all qualities): 18.7 GB

    Total storage:
    - Videos: 100,000 × 18.7 GB = 1.87 PB
    - With audio tracks + subtitles: 2 PB
    - With CDN copies (3x): 6 PB
    ```

    ### Bandwidth Requirements

    ```
    Peak concurrent streams: 15 million
    Average bitrate: 5 Mbps (mix of resolutions)
    Peak bandwidth: 15M × 5 Mbps = 75 Tbps

    With 98% CDN cache hit rate:
    - CDN serves: 75 Tbps × 0.98 = 73.5 Tbps
    - Origin serves: 75 Tbps × 0.02 = 1.5 Tbps
    ```

    ### QPS Estimates

    ```
    Video playback start: 500M/day = 5,800 QPS
    Playback events (progress): 15M concurrent × 1/min = 250K QPS
    Recommendation requests: 500M/day = 5,800 QPS
    Search queries: 50M/day = 580 QPS
    Content metadata: 1B/day = 11,600 QPS
    ```

    ---

    ## Key Assumptions

    1. Average video duration: 45 minutes (movies) to 30 minutes (episodes)
    2. 60% watch from mobile, 25% smart TV, 15% web
    3. Peak hours: 7pm-11pm local time in each region
    4. 80/20 rule: 20% of content generates 80% of views
    5. Content licensed for specific regions (geo-restrictions)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Content-optimized delivery** - Use custom Open Connect CDN embedded in ISP networks
    2. **Adaptive streaming** - Adjust quality based on bandwidth (HLS protocol)
    3. **Predictive caching** - Pre-populate CDN with popular content
    4. **Personalization** - ML-driven recommendations for engagement
    5. **Multi-device sync** - Seamless resume across devices

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Devices"
            TV[Smart TV]
            Mobile[Mobile Apps]
            Web[Web Browser]
        end

        subgraph "CDN - Open Connect"
            CDN[Open Connect CDN<br/>ISP Embedded]
            EdgeCache[Edge Servers<br/>100-200 TB each]
        end

        subgraph "API Gateway & Services"
            Gateway[API Gateway<br/>Zuul]
            VideoService[Video Metadata<br/>Service]
            StreamService[Streaming<br/>Service]
            UserService[User Profile<br/>Service]
            RecoService[Recommendation<br/>Service]
            SearchService[Search<br/>Service]
        end

        subgraph "Content Pipeline"
            Upload[Content Upload]
            Encoding[Encoding Pipeline<br/>FFmpeg Cluster]
            QC[Quality Control]
            Publish[CDN Publisher]
        end

        subgraph "ML & Personalization"
            RecoML[Recommendation<br/>Models]
            ABTest[A/B Testing]
            Analytics[Viewing<br/>Analytics]
        end

        subgraph "Data Storage"
            VideoMeta[(Video Metadata<br/>Cassandra)]
            UserDB[(User Data<br/>PostgreSQL)]
            PlaybackState[(Playback State<br/>Redis)]
            ContentStore[(Content Storage<br/>S3)]
            SearchIndex[(Search Index<br/>Elasticsearch)]
        end

        TV --> CDN
        Mobile --> CDN
        Web --> Gateway

        Gateway --> VideoService
        Gateway --> StreamService
        Gateway --> UserService
        Gateway --> RecoService
        Gateway --> SearchService

        VideoService --> VideoMeta
        UserService --> UserDB
        StreamService --> PlaybackState
        SearchService --> SearchIndex
        RecoService --> RecoML

        Upload --> Encoding
        Encoding --> QC
        QC --> ContentStore
        QC --> Publish
        Publish --> CDN

        RecoML --> Analytics
        StreamService --> Analytics

        CDN --> ContentStore

        style CDN fill:#e1f5ff
        style Encoding fill:#fff4e1
        style RecoService fill:#ffe1e1
        style PlaybackState fill:#ffe1f5
    ```

    ---

    ## Component Rationale

    | Component | Technology Choice | Why This? | Alternative Considered |
    |-----------|-------------------|-----------|----------------------|
    | **CDN** | Open Connect (custom) | Embedded in ISP networks, 98% cache hit, lowest latency | CloudFront, Akamai (more expensive, higher latency) |
    | **Streaming** | HLS (HTTP Live Streaming) | Adaptive bitrate, works over HTTP, wide device support | DASH (more complex), RTMP (requires special server) |
    | **Encoding** | FFmpeg + Per-Title Optimization | 20-50% bitrate savings, high quality | Fixed bitrate ladder (wasteful bandwidth) |
    | **Video Metadata** | Cassandra | High write throughput, handles billions of events | MongoDB (limited scalability), MySQL (not optimized for time-series) |
    | **User Data** | PostgreSQL | ACID transactions for subscriptions, joins for user data | DynamoDB (harder to query), Cassandra (overkill) |
    | **Playback State** | Redis | Fast read/write, TTL support, cross-device sync | DynamoDB (higher latency), Cassandra (overkill) |
    | **Search** | Elasticsearch | Full-text search, faceted search, personalized ranking | Algolia (expensive), Solr (harder to maintain) |
    | **Recommendations** | TensorFlow + Matrix Factorization | Proven ML algorithms, personalization at scale | Simple collaborative filtering (less accurate) |

    ---

    ## Core Data Flow

    **Video Streaming Flow:**

    ```mermaid
    sequenceDiagram
        participant Client
        participant CDN as Open Connect CDN
        participant API as Streaming Service
        participant Redis as Playback State
        participant S3 as Content Storage

        Client->>API: Request video manifest
        API->>Redis: Get resume position
        Redis-->>API: Position: 1234s
        API->>API: Generate HLS manifest<br/>(multiple qualities)
        API-->>Client: Return manifest + resume position

        Client->>CDN: Request video segment<br/>(720p, segment 205)

        alt CDN Cache Hit (98%)
            CDN-->>Client: Serve from edge cache
        else CDN Cache Miss (2%)
            CDN->>S3: Fetch segment
            S3-->>CDN: Return segment
            CDN->>CDN: Cache segment
            CDN-->>Client: Serve segment
        end

        loop Every 10 seconds
            Client->>API: Update playback position
            API->>Redis: Store position
        end
    ```

    **Content Upload & Encoding Flow:**

    ```mermaid
    sequenceDiagram
        participant Studio as Content Studio
        participant Upload as Upload Service
        participant S3
        participant Queue as Encoding Queue
        participant Worker as Encoding Workers
        participant QC as Quality Control
        participant CDN

        Studio->>Upload: Upload original (4K)
        Upload->>S3: Store original
        Upload->>Queue: Enqueue encoding job

        Queue->>Worker: Dispatch job
        Worker->>Worker: Analyze content<br/>(per-title optimization)
        Worker->>Worker: Encode to 5 qualities<br/>(360p, 480p, 720p, 1080p, 4K)
        Worker->>S3: Upload encoded files
        Worker->>QC: Submit for QC

        QC->>QC: Automated checks<br/>(audio sync, artifacts)
        QC->>CDN: Publish to edge servers
        CDN->>CDN: Pre-populate popular regions
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Adaptive Bitrate Streaming (ABR)

    === "HLS Protocol"

        **HTTP Live Streaming (HLS):**

        HLS is Apple's adaptive bitrate streaming protocol, now industry standard for VOD.

        **How It Works:**

        1. **Video Encoding**: Encode video into multiple bitrates (360p to 4K)
        2. **Segmentation**: Split each bitrate into 6-second chunks
        3. **Manifest Files**: Create playlist files (m3u8) listing all segments
        4. **Client Selection**: Client picks appropriate quality based on bandwidth

        **Master Playlist (manifest.m3u8):**

        ```m3u8
        #EXTM3U
        #EXT-X-VERSION:6

        #EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=640x360,CODECS="avc1.64001e,mp4a.40.2"
        360p/playlist.m3u8

        #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=854x480,CODECS="avc1.64001f,mp4a.40.2"
        480p/playlist.m3u8

        #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1280x720,CODECS="avc1.64001f,mp4a.40.2"
        720p/playlist.m3u8

        #EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080,CODECS="avc1.640028,mp4a.40.2"
        1080p/playlist.m3u8

        #EXT-X-STREAM-INF:BANDWIDTH=25000000,RESOLUTION=3840x2160,CODECS="hvc1.2.4.L150.B0,mp4a.40.2"
        4K/playlist.m3u8
        ```

        **Variant Playlist (720p/playlist.m3u8):**

        ```m3u8
        #EXTM3U
        #EXT-X-VERSION:6
        #EXT-X-TARGETDURATION:6
        #EXT-X-MEDIA-SEQUENCE:0

        #EXTINF:6.000,
        segment000.ts
        #EXTINF:6.000,
        segment001.ts
        #EXTINF:6.000,
        segment002.ts
        ...
        #EXT-X-ENDLIST
        ```

        **Client-Side ABR Algorithm:**

        ```python
        class AdaptiveBitrateClient:
            def __init__(self):
                self.buffer_target = 30  # seconds
                self.buffer_min = 10     # seconds
                self.current_quality = '720p'
                self.bandwidth_estimate = 5_000_000  # 5 Mbps

            def select_next_segment_quality(self, buffer_level: float) -> str:
                """
                Select quality for next segment

                Rules:
                1. Buffer low (<10s) → switch to lower quality (prevent rebuffering)
                2. Buffer high (>30s) + bandwidth good → switch to higher quality
                3. Otherwise maintain current quality
                """
                if buffer_level < self.buffer_min:
                    # Emergency: buffer critical, go to safe quality
                    return self._select_safe_quality()

                if buffer_level > self.buffer_target:
                    # Buffer healthy, can try higher quality
                    return self._try_higher_quality()

                # Buffer okay, maintain current
                return self.current_quality

            def _select_safe_quality(self) -> str:
                """Select quality that won't cause rebuffering"""
                qualities = {
                    '360p': 1_000_000,
                    '480p': 2_500_000,
                    '720p': 5_000_000,
                    '1080p': 8_000_000,
                    '4K': 25_000_000
                }

                # Select highest quality below 80% of bandwidth
                safe_bandwidth = self.bandwidth_estimate * 0.8

                for quality, bitrate in sorted(qualities.items(), key=lambda x: x[1]):
                    if bitrate <= safe_bandwidth:
                        selected = quality

                return selected

            def update_bandwidth_estimate(self, segment_size: int, download_time: float):
                """
                Update bandwidth using exponential moving average

                Gives more weight to recent measurements
                """
                measured_bw = (segment_size * 8) / download_time

                # EMA with alpha = 0.2
                self.bandwidth_estimate = (
                    0.8 * self.bandwidth_estimate +
                    0.2 * measured_bw
                )
        ```

        **Benefits:**

        - ✅ Fast startup (begin with low quality)
        - ✅ No buffering (adapt to bandwidth changes)
        - ✅ Efficient bandwidth usage
        - ✅ Works over standard HTTP (no special server)

    === "Per-Title Encoding"

        **Challenge:** One-size-fits-all encoding wastes bandwidth

        **Solution:** Analyze each title and optimize encoding settings

        **Content Analysis:**

        ```python
        class PerTitleEncoder:
            def analyze_content(self, video_path: str) -> dict:
                """
                Analyze video complexity to optimize encoding

                Factors:
                - Spatial complexity (detail level)
                - Temporal complexity (motion)
                - Noise/grain level
                """
                frames = self._sample_frames(video_path, fps=1)

                spatial = []
                temporal = []
                noise = []

                for i, frame in enumerate(frames):
                    # Spatial complexity (edge detection)
                    edges = cv2.Canny(frame, 100, 200)
                    spatial.append(np.sum(edges) / frame.size)

                    # Temporal complexity (frame difference)
                    if i > 0:
                        diff = np.abs(frame - frames[i-1])
                        temporal.append(np.mean(diff))

                    # Noise estimation
                    noise.append(self._estimate_noise(frame))

                return {
                    'spatial_complexity': np.mean(spatial),
                    'temporal_complexity': np.mean(temporal),
                    'noise_level': np.mean(noise)
                }

            def calculate_optimal_bitrates(self, analysis: dict) -> dict:
                """
                Calculate optimal bitrates based on content

                Simple content (animation) → Lower bitrates (save 30-50%)
                Complex content (action movies) → Higher bitrates
                """
                base_bitrates = {
                    '4K': 25_000_000,
                    '1080p': 8_000_000,
                    '720p': 5_000_000,
                    '480p': 2_500_000,
                    '360p': 1_000_000
                }

                # Calculate adjustment factor (0.5 to 1.5)
                complexity_score = (
                    analysis['spatial_complexity'] * 0.4 +
                    analysis['temporal_complexity'] * 0.4 +
                    analysis['noise_level'] * 0.2
                )

                adjustment = 0.5 + complexity_score  # Range: 0.5 to 1.5

                # Apply adjustment
                optimal = {}
                for resolution, bitrate in base_bitrates.items():
                    optimal[resolution] = int(bitrate * adjustment)

                return optimal

            def encode_video(self, input_path: str, resolution: str, bitrate: int):
                """Encode video with optimized settings using FFmpeg"""

                output_path = f"output/{resolution}.mp4"

                # FFmpeg command
                cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-c:v', 'libx265',  # H.265 codec (better compression)
                    '-b:v', str(bitrate),
                    '-vf', f'scale=-2:{self._get_height(resolution)}',
                    '-preset', 'medium',  # Balance speed/quality
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-movflags', '+faststart',  # Enable streaming
                    '-f', 'hls',  # HLS output
                    '-hls_time', '6',  # 6-second segments
                    '-hls_playlist_type', 'vod',
                    output_path
                ]

                subprocess.run(cmd, check=True)
        ```

        **Bitrate Savings:**

        | Content Type | Standard Bitrate | Optimized Bitrate | Savings |
        |--------------|------------------|-------------------|---------|
        | Animation (simple) | 5 Mbps | 2.5 Mbps | 50% |
        | Talk show (medium) | 5 Mbps | 4 Mbps | 20% |
        | Action movie (complex) | 5 Mbps | 6 Mbps | -20% (need more) |

        **Impact:** 20-30% average bandwidth savings across catalog

    === "Video Preloading"

        **Challenge:** Instant playback and smooth transitions

        **Solution:** Preload next content while user watches

        **Preloading Strategy:**

        ```python
        class VideoPreloader:
            def preload_next_episode(self, current_video: dict, progress: float):
                """
                Preload next episode when user is near the end

                Trigger: When user reaches 80% of current episode
                """
                if progress < 0.8:
                    return

                # Get next episode
                next_episode = self._get_next_episode(current_video)

                if not next_episode:
                    return

                # Preload first segment (6 seconds)
                manifest_url = self._get_manifest_url(next_episode['video_id'])
                first_segment_url = self._get_first_segment(manifest_url)

                # Fetch in background
                self._prefetch_url(first_segment_url)

                logger.info(f"Preloaded next episode: {next_episode['title']}")

            def preload_recommendations(self, user_id: int, current_video: dict):
                """
                Preload recommendations shown after video ends

                Preload: First segment of top 3 recommendations
                """
                recommendations = self._get_recommendations(user_id, current_video)

                for rec in recommendations[:3]:
                    manifest_url = self._get_manifest_url(rec['video_id'])
                    first_segment_url = self._get_first_segment(manifest_url)
                    self._prefetch_url(first_segment_url)
        ```

        **Preloading Priorities:**

        1. **Immediate next** (80% progress): First 2 segments (12 seconds)
        2. **Recommendations** (90% progress): First segment (6 seconds) of top 3
        3. **Continue watching** (on homepage): First segment of top 5

    === "Subscription Tier QoS"

        **Quality Limits by Subscription:**

        ```python
        class QualityOfService:
            SUBSCRIPTION_LIMITS = {
                'Basic': {
                    'max_resolution': '480p',  # SD only
                    'max_bitrate': 2_500_000,
                    'concurrent_streams': 1
                },
                'Standard': {
                    'max_resolution': '1080p',  # HD
                    'max_bitrate': 8_000_000,
                    'concurrent_streams': 2
                },
                'Premium': {
                    'max_resolution': '4K',  # Ultra HD
                    'max_bitrate': 25_000_000,
                    'concurrent_streams': 4
                }
            }

            def get_manifest(self, video_id: str, user: dict) -> str:
                """
                Generate HLS manifest based on subscription tier

                Filter qualities based on user's subscription
                """
                subscription = user['subscription_tier']
                limits = self.SUBSCRIPTION_LIMITS[subscription]

                # Get all available qualities
                all_qualities = self._get_video_qualities(video_id)

                # Filter based on subscription
                allowed_qualities = [
                    q for q in all_qualities
                    if self._is_quality_allowed(q, limits)
                ]

                # Generate manifest
                manifest = self._build_hls_manifest(allowed_qualities)

                return manifest

            def _is_quality_allowed(self, quality: dict, limits: dict) -> bool:
                """Check if quality is allowed for subscription"""
                resolution_order = ['360p', '480p', '720p', '1080p', '4K']

                max_res_idx = resolution_order.index(limits['max_resolution'])
                quality_res_idx = resolution_order.index(quality['resolution'])

                return quality_res_idx <= max_res_idx
        ```

    ---

    ## 3.2 Open Connect CDN

    === "CDN Architecture"

        **Netflix Open Connect:** Custom CDN embedded in ISP networks

        **Why Custom CDN?**

        1. **Lower latency**: Content closer to users (inside ISP)
        2. **Lower cost**: Reduce internet transit costs
        3. **Higher quality**: Direct path, no internet congestion
        4. **ISP benefits**: Reduce bandwidth on expensive backbone links

        **Architecture:**

        ```
        Open Connect Deployment:

        Tier 1: Edge Servers (ISP POPs)
        ├── Location: Inside ISP data centers
        ├── Capacity: 100-200 TB per server (SSD)
        ├── Coverage: 200+ ISP locations globally
        └── Purpose: Serve 95%+ of requests

        Tier 2: Regional Fill Servers
        ├── Location: Major internet exchanges
        ├── Capacity: 1-2 PB per region
        ├── Coverage: 10-15 regions
        └── Purpose: Fill edge cache misses

        Tier 3: Origin Servers (AWS S3)
        ├── Location: AWS regions
        ├── Capacity: Unlimited
        ├── Coverage: Global
        └── Purpose: Source of truth, cold content
        ```

        **Cache Management:**

        ```python
        class OpenConnectCacheManager:
            def predict_content_demand(self, region: str) -> list:
                """
                Predict what content will be watched in region

                Factors:
                - Historical viewing patterns (80/20 rule)
                - New releases (upcoming Friday releases)
                - Trending content (viral growth)
                - Regional preferences (local language content)
                - Time of day (evening peak hours)
                """
                regional_stats = self._get_viewing_stats(region, days=30)

                # Score each content item
                content_scores = []

                for content in self._get_all_content():
                    score = self._calculate_cache_score(
                        content,
                        regional_stats,
                        region
                    )

                    content_scores.append({
                        'content_id': content['id'],
                        'score': score,
                        'size_gb': content['size_gb']
                    })

                # Sort by score
                content_scores.sort(key=lambda x: x['score'], reverse=True)

                return content_scores

            def _calculate_cache_score(self, content: dict,
                                      stats: dict, region: str) -> float:
                """
                Score content for caching priority

                Scoring factors:
                - View count (60%): How many times watched
                - Recency (20%): Newer content scored higher
                - Trending (10%): Accelerating views
                - Upcoming (10%): New releases this week
                """
                # View count score (normalized to 1.0)
                view_score = min(stats.get(content['id'], {}).get('views', 0) / 1_000_000, 1.0)

                # Recency score (0 to 1, decays over 1 year)
                days_old = (time.time() - content['release_date']) / 86400
                recency_score = max(0, 1 - days_old / 365)

                # Trending score (view acceleration)
                growth = stats.get(content['id'], {}).get('weekly_growth', 0)
                trending_score = min(growth, 1.0)

                # Upcoming score (new season, high-profile release)
                upcoming_score = 1.0 if self._is_upcoming_release(content) else 0.0

                # Weighted sum
                total = (
                    view_score * 0.6 +
                    recency_score * 0.2 +
                    trending_score * 0.1 +
                    upcoming_score * 0.1
                )

                return total

            def prefill_cache(self, appliance_id: str, content_list: list):
                """
                Pre-fill cache during off-peak hours (2am-6am local time)

                Transfer from origin to edge servers
                """
                for content in content_list:
                    # Schedule transfer
                    self._schedule_transfer(
                        appliance_id,
                        content['content_id'],
                        priority='low',
                        time_window='off_peak'  # 2am-6am local
                    )

                    logger.info(f"Scheduled prefill: {content['title']}")
        ```

        **Cache Hit Rate:** 98% (only 2% requests go to origin)

    === "Content Popularity"

        **80/20 Rule in Practice:**

        | Content Tier | % of Catalog | % of Views | Cache Location |
        |--------------|--------------|------------|----------------|
        | **Top Hits** | 1% (1,000 titles) | 50% | All edge servers |
        | **Popular** | 10% (10,000 titles) | 35% | Most edge servers |
        | **Long Tail** | 89% (89,000 titles) | 15% | Regional servers only |

        **Cache Allocation:**

        ```python
        class CacheAllocationStrategy:
            def allocate_cache_space(self, appliance_capacity_tb: int,
                                    region: str) -> dict:
                """
                Allocate cache space across content tiers

                Strategy:
                - 60% for top hits (guaranteed availability)
                - 30% for popular content (regional preferences)
                - 10% for LRU long-tail (on-demand caching)
                """
                capacity_gb = appliance_capacity_tb * 1000

                allocation = {
                    'top_hits': {
                        'size_gb': capacity_gb * 0.6,
                        'policy': 'static',  # Always cached
                        'content': self._get_global_top_hits(1000)
                    },
                    'popular': {
                        'size_gb': capacity_gb * 0.3,
                        'policy': 'predictive',  # Pre-populated
                        'content': self._get_regional_popular(region, 10000)
                    },
                    'long_tail': {
                        'size_gb': capacity_gb * 0.1,
                        'policy': 'lru',  # On-demand, evict least recently used
                        'content': []  # Dynamically filled
                    }
                }

                return allocation
        ```

    === "Geo-Restrictions"

        **Content Licensing:** Different content available per region

        **Implementation:**

        ```python
        class GeoRestrictionService:
            def check_access(self, user_id: int, content_id: str) -> bool:
                """
                Check if user can access content based on location

                Factors:
                - Content licensing region
                - User's current location (IP geolocation)
                - VPN detection
                """
                # Get user location
                user_ip = self._get_user_ip(user_id)
                user_country = self._geolocate_ip(user_ip)

                # Check for VPN/proxy
                if self._is_vpn(user_ip):
                    logger.warning(f"VPN detected for user {user_id}")
                    return False

                # Get content licensing info
                content = self._get_content(content_id)
                licensed_regions = content['licensed_regions']

                # Check if user's country is licensed
                if user_country in licensed_regions:
                    return True

                logger.info(f"Content {content_id} not available in {user_country}")
                return False

            def _geolocate_ip(self, ip: str) -> str:
                """Get country code from IP (MaxMind GeoIP)"""
                response = self.geoip_client.country(ip)
                return response.country.iso_code  # 'US', 'GB', etc.

            def _is_vpn(self, ip: str) -> bool:
                """Detect VPN/proxy usage"""
                # Check against known VPN IP ranges
                # Use commercial VPN detection services
                return self.vpn_detector.is_vpn(ip)
        ```

    ---

    ## 3.3 Personalization & Recommendations

    === "Recommendation Engine"

        **Multi-Algorithm Approach:**

        Netflix uses dozens of recommendation algorithms, each optimized for different contexts:

        1. **Personalized Video Ranker (PVR)** - Main homepage rows
        2. **Top-N Video Ranker** - Trending/popular
        3. **Continue Watching** - Resume incomplete
        4. **Because You Watched X** - Content-based similarity
        5. **Trending Now** - Real-time popularity
        6. **New Releases** - Recent additions

        **Homepage Generation:**

        ```python
        class HomepageGenerator:
            def generate_homepage(self, user_id: int, profile_id: int) -> list:
                """
                Generate personalized homepage with multiple rows

                Each row uses different algorithm
                """
                rows = []

                # Row 1: Continue Watching (priority)
                continue_watching = self._get_continue_watching(profile_id)
                if continue_watching:
                    rows.append({
                        'title': 'Continue Watching',
                        'algorithm': 'resume',
                        'items': continue_watching
                    })

                # Row 2: Trending Now (regional)
                region = self._get_user_region(user_id)
                trending = self._get_trending(region, limit=20)
                rows.append({
                    'title': 'Trending Now',
                    'algorithm': 'trending',
                    'items': trending
                })

                # Row 3-4: Personalized rows (ML-based)
                personalized = self._generate_personalized_rows(profile_id, count=2)
                rows.extend(personalized)

                # Row 5: Because You Watched X
                recent = self._get_recently_watched(profile_id, limit=1)[0]
                similar = self._get_similar_content(recent['content_id'])
                rows.append({
                    'title': f"Because You Watched {recent['title']}",
                    'algorithm': 'content_similarity',
                    'items': similar
                })

                # Row 6-7: Genre-based (top 2 genres)
                favorite_genres = self._get_favorite_genres(profile_id)
                for genre in favorite_genres[:2]:
                    genre_picks = self._get_top_in_genre(profile_id, genre)
                    rows.append({
                        'title': f"Top Picks in {genre}",
                        'algorithm': 'genre_personalized',
                        'items': genre_picks
                    })

                # Row 8: Popular in Your Country
                popular_local = self._get_popular_in_country(region)
                rows.append({
                    'title': 'Popular in Your Country',
                    'algorithm': 'regional_popular',
                    'items': popular_local
                })

                return rows
        ```

    === "ML Model Architecture"

        **Two-Tower Neural Network:**

        ```
        User Tower                     Video Tower
        +-----------------+           +-----------------+
        | User Features   |           | Video Features  |
        | - Watch history |           | - Genre         |
        | - Preferences   |           | - Cast/Crew     |
        | - Demographics  |           | - Tags          |
        +-----------------+           | - Popularity    |
               |                      +-----------------+
               v                             v
        +-----------------+           +-----------------+
        | Dense Layers    |           | Dense Layers    |
        | 512 -> 256      |           | 512 -> 256      |
        +-----------------+           +-----------------+
               |                             |
               v                             v
        +-----------------+           +-----------------+
        | User Embedding  |           | Video Embedding |
        | (128 dims)      |           | (128 dims)      |
        +-----------------+           +-----------------+
               |                             |
               +-------- Dot Product --------+
                             |
                             v
                    Similarity Score (0-1)
        ```

        **Training:**

        ```python
        class RecommendationModelTraining:
            def train_model(self):
                """
                Train two-tower model on viewing history

                Positive examples: Watched >50% of duration
                Negative examples: Watched <5% or random unseen
                """
                # Load training data (30 days)
                training_data = self._load_training_data(days=30)

                # Extract features
                user_features = self._extract_user_features(training_data)
                video_features = self._extract_video_features(training_data)

                # Create positive/negative pairs
                positive_pairs = self._create_positive_pairs(training_data)
                negative_pairs = self._create_negative_pairs(training_data)

                # Build model
                model = self._build_two_tower_model()

                # Train with triplet loss
                model.fit(
                    [user_features, video_features],
                    labels,
                    epochs=10,
                    batch_size=2048
                )

                # Evaluate
                metrics = self._evaluate_model(model)
                logger.info(f"Model metrics: {metrics}")

                # Deploy
                self._deploy_model(model)
        ```

    === "A/B Testing"

        **Continuous Experimentation:**

        Netflix runs hundreds of A/B tests simultaneously to optimize:

        - Recommendation algorithms
        - Homepage layouts
        - Thumbnail artwork
        - Video quality settings
        - UI/UX changes

        **A/B Test Framework:**

        ```python
        class ABTestingFramework:
            def assign_user_to_variant(self, user_id: int, experiment_id: str) -> str:
                """
                Assign user to A/B test variant

                Deterministic assignment (same user always gets same variant)
                """
                # Hash user ID + experiment ID
                hash_input = f"{user_id}:{experiment_id}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

                # Get experiment config
                experiment = self._get_experiment(experiment_id)

                # Assign variant based on hash
                total_allocation = sum(experiment['variants'].values())
                threshold = hash_value % total_allocation

                cumulative = 0
                for variant, allocation in experiment['variants'].items():
                    cumulative += allocation
                    if threshold < cumulative:
                        return variant

                return 'control'

            def track_metric(self, user_id: int, experiment_id: str,
                           metric: str, value: float):
                """Track metric for A/B test analysis"""

                variant = self.assign_user_to_variant(user_id, experiment_id)

                self.analytics.record_event({
                    'experiment_id': experiment_id,
                    'variant': variant,
                    'user_id': user_id,
                    'metric': metric,
                    'value': value,
                    'timestamp': time.time()
                })

            def analyze_experiment(self, experiment_id: str) -> dict:
                """
                Analyze experiment results

                Statistical significance testing (t-test)
                """
                results = self.analytics.query_experiment_results(experiment_id)

                # Compare variants
                control_metrics = results['control']
                treatment_metrics = results['treatment']

                # Calculate lift
                lift = (treatment_metrics['mean'] - control_metrics['mean']) / control_metrics['mean']

                # Statistical significance (t-test)
                p_value = self._t_test(control_metrics, treatment_metrics)

                return {
                    'lift': lift,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        ```

        **Key Metrics:**

        - Watch time per user
        - Completion rate
        - Content discovery (% of catalog consumed)
        - Session frequency
        - Retention rate

    === "Real-Time Personalization"

        **Update preferences in real-time based on viewing:**

        ```python
        class RealtimePersonalization:
            def on_video_watched(self, profile_id: int, video_id: str,
                                watch_duration: int, video_duration: int):
                """
                Update user profile when video watched

                Impact immediate recommendations
                """
                # Calculate completion rate
                completion = watch_duration / video_duration

                if completion < 0.05:
                    # User didn't like it - downweight similar content
                    self._downweight_preferences(profile_id, video_id)
                    return

                # Get video features
                video = self._get_video(video_id)

                # Update profile
                profile = self._get_profile(profile_id)

                # Update genre preferences (EMA)
                for genre in video['genres']:
                    current = profile['genre_scores'].get(genre, 0.5)
                    new_score = 0.9 * current + 0.1 * completion
                    profile['genre_scores'][genre] = new_score

                # Update actor preferences
                for actor in video['cast']:
                    current = profile['actor_scores'].get(actor, 0.5)
                    new_score = 0.9 * current + 0.1 * completion
                    profile['actor_scores'][actor] = new_score

                # Save profile
                self._save_profile(profile_id, profile)

                # Invalidate recommendation cache
                self.cache.delete(f"recommendations:{profile_id}")
        ```

    ---

    ## 3.4 Multi-Device Playback

    === "Resume Position Sync"

        **Seamless resume across devices:**

        ```python
        class PlaybackStateManager:
            def track_playback(self, session_id: str, profile_id: int,
                             video_id: str, position: int):
                """
                Track playback position (called every 10 seconds)

                Store in Redis for fast cross-device access
                """
                key = f"playback:{profile_id}:{video_id}"

                # Update position
                self.redis.hset(key, mapping={
                    'position': position,
                    'last_updated': time.time(),
                    'session_id': session_id
                })

                # Set TTL (30 days)
                self.redis.expire(key, 86400 * 30)

                # Async persist to database
                self.kafka.send('playback_events', {
                    'profile_id': profile_id,
                    'video_id': video_id,
                    'position': position,
                    'session_id': session_id,
                    'timestamp': time.time()
                })

            def get_resume_position(self, profile_id: int, video_id: str) -> int:
                """Get resume position for video (cross-device)"""

                key = f"playback:{profile_id}:{video_id}"

                # Check Redis cache
                cached = self.redis.hget(key, 'position')

                if cached:
                    return int(cached)

                # Cache miss - query database
                result = self.db.query("""
                    SELECT position FROM playback_state
                    WHERE profile_id = %s AND video_id = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, [profile_id, video_id])

                if result:
                    position = result[0]['position']

                    # Populate cache
                    self.redis.hset(key, 'position', position)
                    self.redis.expire(key, 86400 * 30)

                    return position

                return 0  # Start from beginning

            def handle_device_switch(self, profile_id: int, video_id: str,
                                   old_session: str, new_session: str):
                """
                Handle user switching devices mid-playback

                Pause old session, sync position to new session
                """
                # Get current position
                position = self.get_resume_position(profile_id, video_id)

                # Send pause command to old session (WebSocket)
                self._send_pause_command(old_session)

                # Return position for new session
                return {
                    'resume_position': position,
                    'switched_from_device': self._get_device_type(old_session)
                }
        ```

    === "Continue Watching"

        **Show partially watched content for easy resume:**

        ```python
        class ContinueWatchingManager:
            def update_continue_watching(self, profile_id: int,
                                        video_id: str, completion: float):
                """
                Update continue watching list

                Show if 5-95% watched (not just started, not finished)
                """
                if completion < 0.05 or completion > 0.95:
                    # Too little or too much - remove from continue watching
                    self.redis.zrem(f"continue_watching:{profile_id}", video_id)
                    return

                # Add to sorted set (score = timestamp for recency)
                self.redis.zadd(
                    f"continue_watching:{profile_id}",
                    {video_id: time.time()}
                )

                # Keep only last 50 items
                self.redis.zremrangebyrank(
                    f"continue_watching:{profile_id}",
                    0, -51
                )

            def get_continue_watching(self, profile_id: int) -> list:
                """Get continue watching list (sorted by recency)"""

                # Get video IDs from sorted set
                video_ids = self.redis.zrevrange(
                    f"continue_watching:{profile_id}",
                    0, 49  # Top 50
                )

                # Hydrate with video metadata
                videos = []
                for vid in video_ids:
                    video = self._get_video_metadata(vid.decode())

                    # Get resume position
                    position = self.get_resume_position(profile_id, vid.decode())

                    video['resume_position'] = position
                    video['completion'] = position / video['duration']

                    videos.append(video)

                return videos
        ```

=== "🚀 Step 4: Scale & Optimize"

    ## Performance Optimization

    === "Throughput"

        **Target:** 15 million concurrent streams

        **Scaling Strategy:**

        ```
        Architecture for 15M concurrent:

        1. API Servers: 5,000 instances
           - Each handles 3,000 connections
           - Total: 15M concurrent connections
           - Cost: $600K/month (c5.2xlarge)

        2. CDN Edge Servers: 10,000 appliances
           - 200 TB SSD storage each
           - Serve 98% of requests
           - Cost: Depreciation of hardware

        3. Encoding Workers: 200 instances
           - Process 100M hours/month
           - Per-title optimization
           - Cost: $80K/month (c5.9xlarge spot)

        4. Recommendation Inference: 100 GPU instances
           - Real-time personalization
           - Cost: $300K/month (p3.2xlarge)
        ```

        **Load Distribution:**

        - 98% traffic served by CDN edge
        - 2% hits regional or origin servers
        - Peak hours: 8pm-11pm local time (3x average)

    === "Latency Targets"

        **Performance SLAs:**

        | Metric | Target | p99 | Strategy |
        |--------|--------|-----|----------|
        | Video start time | < 2s | < 3s | Preload first segment, CDN proximity |
        | Segment fetch | < 100ms | < 200ms | Edge server <50ms away |
        | API response | < 100ms | < 500ms | Redis caching, read replicas |
        | Search | < 300ms | < 500ms | Elasticsearch, query optimization |
        | Homepage | < 500ms | < 1s | Cached recommendations (5min TTL) |

        **Optimizations:**

        ```
        1. Video Playback:
           - Preload first segment (instant start)
           - Start with lowest quality (fast decode)
           - Upgrade quality as buffer fills

        2. API Latency:
           - Cache everything (Redis)
           - Database read replicas (10x)
           - Connection pooling

        3. Recommendation:
           - Pre-compute homepage rows (cache 5 min)
           - Batch inference (process 100 users at once)
           - Model optimization (quantization, pruning)

        4. CDN:
           - Edge servers inside ISP (1-2 hops)
           - Predictive pre-population
           - Persistent connections (HTTP/2)
        ```

    === "Availability"

        **Target:** 99.99% uptime (52 minutes downtime/year)

        **High Availability Architecture:**

        ```
        Multi-Region Deployment:

        Region 1 (US-East):
        ├── API Servers: 2,000
        ├── Databases: Primary (PostgreSQL, Cassandra)
        ├── CDN: 4,000 edge appliances
        └── Encoding: 100 workers

        Region 2 (US-West):
        ├── API Servers: 1,500
        ├── Databases: Read replicas + standby
        ├── CDN: 3,000 edge appliances
        └── Encoding: 50 workers

        Region 3 (EU-West):
        ├── API Servers: 1,000
        ├── Databases: Read replicas
        ├── CDN: 2,000 edge appliances
        └── Encoding: 30 workers

        Failover Strategy:
        - CDN edge: Fail to regional fill servers
        - API: Route to nearest healthy region
        - Database: Promote read replica to master
        - Encoding: Drain queue, route to other region

        RTO: 5 minutes
        RPO: 0 (no data loss)
        ```

    === "Cost Optimization"

        **Monthly Cost Breakdown (200M subscribers):**

        ```
        1. CDN & Bandwidth: $80M/month
           - Open Connect deployment: $60M
           - Internet transit: $20M
           - Optimization: ISP partnerships reduce cost by 70%

        2. Cloud Infrastructure (AWS): $15M/month
           - Compute (EC2): $8M
           - Storage (S3): $5M
           - Databases (RDS, Cassandra): $2M

        3. Encoding: $5M/month
           - Spot instances: $3M
           - Per-title optimization saves 20% bandwidth
           - Storage after encoding: $2M

        4. Machine Learning: $3M/month
           - Training (Spark): $1M
           - Inference (GPU): $1.5M
           - Data pipelines: $0.5M

        5. Third-Party Services: $2M/month
           - MaxMind GeoIP: $0.5M
           - Monitoring (Datadog): $0.5M
           - Other: $1M

        Total: ~$105M/month = $1.26B/year

        Revenue:
        - 200M subscribers × $15/month = $3B/month
        - Annual: $36B

        Margin: $36B - $15.1B (total costs) = $20.9B (58% margin)
        ```

        **Cost Optimization Strategies:**

        1. **Open Connect CDN**
           - Reduces bandwidth costs by 70%
           - Embedded in ISP networks
           - Savings: $100M+/year

        2. **Per-Title Encoding**
           - 20-30% bandwidth savings
           - Lower storage costs
           - Savings: $50M+/year

        3. **Spot Instances**
           - Use for encoding (70% discount)
           - Stateless workloads
           - Savings: $30M+/year

        4. **Intelligent Caching**
           - 98% cache hit rate
           - Reduce origin traffic
           - Savings: $40M+/year

    ---

    ## Trade-offs & Design Decisions

    | Aspect | Option A | Option B | Netflix Choice | Reasoning |
    |--------|----------|----------|----------------|-----------|
    | **CDN** | Cloud CDN (CloudFront) | Custom Open Connect | **Open Connect** | Lower latency (inside ISP), lower cost (no transit), better control |
    | **Streaming Protocol** | HLS (Apple) | DASH (MPEG) | **HLS** | Simpler, better device support, proven at scale |
    | **Encoding** | Fixed bitrate ladder | Per-title optimization | **Per-title** | 20-50% bandwidth savings, better quality/cost ratio |
    | **Recommendations** | Simple collaborative filtering | Deep learning ensemble | **Deep learning** | Higher accuracy, better personalization, more engagement |
    | **Database** | MySQL only | PostgreSQL + Cassandra | **PostgreSQL + Cassandra** | ACID for user data, scale for time-series playback data |
    | **Video Storage** | Origin-only | Multi-tier (origin + edge) | **Multi-tier** | 98% cache hit rate, lower latency, better UX |

    ---

    ## Interview Tips & Follow-up Questions

    **Common Follow-ups:**

    1. **"How does adaptive bitrate streaming work?"**
       - Video encoded at multiple bitrates (360p to 4K)
       - Split into 6-second segments (HLS protocol)
       - Client measures bandwidth every segment
       - Selects next segment quality based on buffer level + bandwidth
       - Smooth transitions (quality changes between segments, not mid-segment)

    2. **"How do you prevent buffering?"**
       - Adaptive bitrate (auto-adjust quality)
       - Maintain 30-second buffer target
       - Preload first segment (instant start)
       - CDN edge servers close to users (<50ms)
       - Downgrade quality before buffering

    3. **"How does Netflix CDN work differently?"**
       - **Open Connect:** Custom CDN embedded in ISP networks
       - Edge servers with 100-200 TB storage inside ISP data centers
       - Predictive pre-population during off-peak hours
       - Serves 98% of requests (vs 70-80% for cloud CDN)
       - Lower cost (no internet transit), lower latency (1-2 hops)

    4. **"How do recommendations work?"**
       - Multi-algorithm approach (collaborative filtering + content-based + trending)
       - Two-tower neural network (user embedding + video embedding)
       - Train on viewing history (>50% watched = positive)
       - Real-time updates (preferences updated immediately after watch)
       - A/B testing everything (hundreds of experiments running)

    5. **"How do you handle 15 million concurrent streams?"**
       - 98% served by CDN (no backend hit)
       - Stateless API servers (horizontal scaling)
       - Database read replicas (10x) for remaining 2%
       - Redis caching for all metadata
       - Multi-region deployment for global coverage

    6. **"How do you optimize encoding costs?"**
       - Per-title optimization (analyze content complexity)
       - Simple content (animation) → lower bitrates (-30%)
       - Complex content (action) → higher bitrates
       - H.265 codec (better compression than H.264)
       - Spot instances for encoding (70% cost savings)

    7. **"How do you handle resume across devices?"**
       - Track position every 10 seconds during playback
       - Store in Redis (fast access) + Cassandra (durability)
       - Get resume position when video starts on any device
       - Handle concurrent sessions (pause old, resume new)
       - Continue watching list shows partially watched (5-95%)

    **Key Points to Mention:**

    - Open Connect CDN is Netflix's key differentiator (embedded in ISPs)
    - Adaptive bitrate streaming prevents buffering (HLS protocol)
    - Per-title encoding optimizes bandwidth (20-50% savings)
    - Personalization drives engagement (two-tower neural network)
    - Multi-device sync seamless (Redis + Cassandra)
    - 98% cache hit rate (predictive pre-population)
    - Multi-region for global availability
    - Continuous A/B testing for optimization

---

## Summary

**System Characteristics:**

- **Scale:** 200M subscribers, 15M concurrent streams, 100K hours content
- **Performance:** <2s video start, 99% no buffering, <500ms API
- **Availability:** 99.99% uptime, multi-region deployment
- **Cost:** $1.26B/year infrastructure, $36B revenue (58% margin)

**Core Components:**

1. **Open Connect CDN:** Custom CDN embedded in ISP networks (98% cache hit)
2. **Adaptive Streaming:** HLS protocol, client-side quality selection
3. **Encoding Pipeline:** Per-title optimization, FFmpeg cluster, 20-50% savings
4. **Recommendations:** Two-tower neural network, multi-algorithm approach
5. **Multi-Device Sync:** Redis + Cassandra, seamless resume

**Key Design Decisions:**

- Open Connect CDN (lower cost, lower latency, better control)
- HLS adaptive bitrate streaming (prevents buffering)
- Per-title encoding (optimizes bandwidth)
- PostgreSQL + Cassandra (ACID + scale)
- Two-tower neural network (personalization)
- Predictive cache pre-population (98% hit rate)
- Multi-region deployment (global availability)

This design delivers high-quality, personalized video streaming to hundreds of millions of users globally with minimal buffering and optimal bandwidth usage.
