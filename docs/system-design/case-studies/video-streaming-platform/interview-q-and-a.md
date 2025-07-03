# Interview Questions and Answers

This section covers common follow-up questions, trade-offs, and deeper discussions that occur in a system design interview.

### Q1: How would you handle the "thundering herd" problem for a live stream event, like a major sports final?

**Answer:** The "thundering herd" problem occurs when a massive number of users try to access the same resource simultaneously. For a live stream, this means millions of viewers requesting the same video segments at the same time.

**Strategies:**

1.  **CDN Layering:** Use multiple layers of caching. Requests first hit local edge caches. A group of edge caches would then fetch from a regional shield/origin cache. This prevents all requests from hitting the central origin servers at once.
2.  **Pre-warming the Cache:** Before the event starts, proactively push the initial stream segments and manifest files to the edge caches in anticipated high-traffic regions.
3.  **Jitter & Jitter Buffer:** Introduce a small, random delay (jitter) in when clients request the next segment. Even a few hundred milliseconds of variance, managed by the video player's buffer, can spread out the requests over time and smooth out traffic spikes.
4.  **Multicast:** For network-level optimization, especially on managed networks (like an ISP), IP multicast can be used to send a single packet to many recipients simultaneously, though this is less common over the public internet.

### Q2: How do you ensure the accuracy of view counts? Why does it sometimes seem delayed?

**Answer:** View count is a classic example of balancing consistency, availability, and performance. Perfect real-time accuracy is extremely difficult and expensive at scale, and it's not a critical requirement for the user experience.

**Design:**

1.  **Client-Side Beaconing:** The video player sends periodic "heartbeat" events to an analytics service (e.g., every 30 seconds of watch time).
2.  **Ingestion & Aggregation:** These events are ingested into a high-throughput message queue like Kafka. A stream processing system (like Apache Flink or Spark Streaming) consumes these events, aggregates them in near real-time (e.g., in 1-minute windows), and updates a cache (like Redis).
3.  **Batch Reconciliation:** The raw event data is also dumped into a data lake. A daily or hourly batch job runs to perform a more accurate count, reconciling any discrepancies and updating the main database. This ensures long-term accuracy.
4.  **Serving the Count:** The video playback service reads the view count from the Redis cache for low-latency access. This is why the count appears to update periodically rather than instantly—it reflects the latest aggregation window.

### Q3: What are the trade-offs in choosing a video codec?

**Answer:** The choice of a video codec is a trade-off between **compression efficiency**, **computational cost (for encoding/decoding)**, and **licensing fees**.

-   **H.264 (AVC):** The most widely supported codec, compatible with almost all devices. It offers good compression but is less efficient than newer codecs. It's a safe, universal choice.
-   **H.265 (HEVC):** Offers about 50% better compression than H.264, meaning higher quality at the same bitrate or the same quality at a lower bitrate. This saves on storage and bandwidth costs. However, it can have licensing fees and requires more processing power to encode.
-   **AV1:** A newer, open-source, royalty-free codec developed by the Alliance for Open Media (AOMedia), which includes Google, Netflix, and Amazon. It offers even better compression than HEVC (around 30% more efficient). The main drawback is the high computational cost of encoding, making it challenging for real-time/live streaming.

**Strategy:** Use a mix of codecs. Serve AV1 to modern browsers and devices that support it to save bandwidth, while keeping H.264 streams available for older devices to ensure maximum compatibility.

### Q4: How would you design the storage system to be cost-effective?

**Answer:** Video storage is a major cost driver. A tiered storage strategy is essential.

1.  **Hot Tier (e.g., S3 Standard):** For newly uploaded, popular, and frequently accessed videos. This tier offers the lowest latency but has the highest storage cost.
2.  **Warm Tier (e.g., S3 Infrequent Access):** For videos that are accessed less frequently (e.g., content older than 90 days with declining viewership). This tier has lower storage costs but higher retrieval fees.
3.  **Cold/Archive Tier (e.g., S3 Glacier Deep Archive):** For archival purposes, such as original high-resolution masters of old content that is rarely accessed. This tier is extremely cheap for storage but has long retrieval times (hours) and high retrieval costs.

**Lifecycle Policies:** Automate the movement of data between tiers based on access patterns and business rules. For example, a video that hasn't been watched in 180 days could be automatically moved from the warm to the cold tier.
