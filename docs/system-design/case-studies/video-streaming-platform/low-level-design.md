# Low-Level Design: Component Deep Dive

This section provides a more detailed look into the design of key components and the database schema.

## 1. Video Upload & Processing Pipeline

This pipeline is responsible for ingesting, transcoding, and preparing videos for streaming.

-   **Upload Service:** Acts as the entry point. It receives the raw video file and its metadata from the user. To handle large files reliably, it uses **multipart and resumable uploads**.
-   **Object Storage:** Raw video files are stored in a durable and cost-effective object store like AWS S3 or Google Cloud Storage. This is the source of truth for original video content.
-   **Processing Queue (e.g., Kafka, SQS):** After a video is uploaded, a message is placed in a queue. This decouples the upload process from the compute-intensive transcoding process, improving system resilience.
-   **Transcoding Service:** A fleet of workers (e.g., Kubernetes pods, EC2 instances) consumes jobs from the queue.
    -   **Adaptive Bitrate Streaming:** Each video is transcoded into multiple resolutions (e.g., 2160p, 1080p, 720p) and bitrates. The video is also segmented into small chunks (e.g., 2-4 seconds).
    -   **Manifest File:** A manifest file (HLS `.m3u8` or DASH `.mpd`) is generated, which lists the available streams and segment locations. Video players use this manifest to switch between quality levels dynamically based on network conditions.
-   **Metadata Database:** After processing, the database is updated with the video's status, CDN URLs for the manifest file, thumbnails, and other metadata.

## 2. Content Delivery Network (CDN)

The CDN is critical for low-latency global video delivery.

-   **Architecture:** A multi-tiered CDN approach is used.
    -   **Edge Servers:** Thousands of servers located globally, close to users. They cache video segments.
    -   **Origin/Shield Cache:** A regional layer of servers that sits between the edge and the object storage. This reduces the load on the primary storage.
-   **Cache Strategy:**
    -   When a user requests a video, the request hits the nearest edge server.
    -   If the video segment is cached (**cache hit**), it's served directly.
    -   If not (**cache miss**), the request goes to the origin cache, and then to the object storage. The segment is then cached at both the edge and origin for future requests.
    -   **Cache Warming:** Popular content can be proactively pushed to edge caches based on analytics (e.g., a viral video in a specific region).

## 3. Recommendation Engine

The goal is to provide personalized content suggestions.

-   **Candidate Generation:** Multiple sources are used to generate a list of potential videos to recommend.
    -   **Collaborative Filtering:** "Users who watched X also watched Y."
    -   **Content-Based Filtering:** "Because you watched videos about 'System Design', here are more."
    -   **Trending/Popular Videos:** What's currently popular in the user's region.
-   **Ranking:** A machine learning model ranks the candidate videos based on the probability of user engagement (clicks, watch time). It considers hundreds of features, such as user watch history, video popularity, time of day, and device type.
-   **System Architecture:** A hybrid approach is common, with batch processing (for collaborative filtering) and real-time updates (for session-based recommendations).

## 💾 Database Design

Choosing the right database for each service is crucial.

### Video Metadata Schema

A NoSQL database like MongoDB or a relational database like PostgreSQL can be used. NoSQL offers flexibility, while SQL provides strong consistency.

**`videos` collection/table:**
```json
{
  "video_id": "uuid",
  "user_id": "uuid",
  "title": "string",
  "description": "text",
  "status": "string (uploading, processing, processed, failed)",
  "tags": ["array", "of", "strings"],
  "duration_seconds": "integer",
  "upload_timestamp": "datetime",
  "view_count": "integer",
  "like_count": "integer",
  "manifest_url": "string",
  "thumbnail_urls": {
    "small": "url",
    "large": "url"
  }
}
```

### User Data Schema

A relational database (e.g., PostgreSQL) is a good choice for user data due to the need for transactional integrity (e.g., for subscriptions).

**`users` table:**
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**`watch_history` table:**
```sql
CREATE TABLE watch_history (
    user_id UUID REFERENCES users(user_id),
    video_id UUID,
    watched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    watch_duration_seconds INTEGER,
    PRIMARY KEY (user_id, video_id)
);
```

### Sharding Strategy

To handle massive scale, the databases must be partitioned (sharded).

-   **User Data:** Shard by `user_id`. This keeps all data for a single user on the same shard.
-   **Video Metadata:** Shard by `video_id`.
-   **Analytics Data:** Shard by time (e.g., monthly) and/or by a geographic region to make time-series queries efficient.
