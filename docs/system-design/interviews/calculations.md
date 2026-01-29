# Calculations Guide

Back-of-envelope calculations are essential in system design interviews. They help you estimate scale, storage, bandwidth, and costs quickly and accurately.

---

## Key Numbers to Remember

### Powers of Two

Memorize these for quick calculations:

| Power | Exact Value | Approximation | Name |
|-------|-------------|---------------|------|
| 2^10 | 1,024 | ~1 thousand | 1 KB |
| 2^20 | 1,048,576 | ~1 million | 1 MB |
| 2^30 | 1,073,741,824 | ~1 billion | 1 GB |
| 2^40 | 1,099,511,627,776 | ~1 trillion | 1 TB |
| 2^50 | ~1,125,899,906,842,624 | ~1 quadrillion | 1 PB |

**Pro tip:** For interviews, rounding is OK:

- 1 KB = 1,000 bytes
- 1 MB = 1,000 KB = 1 million bytes
- 1 GB = 1,000 MB = 1 billion bytes
- 1 TB = 1,000 GB = 1 trillion bytes

### Time Conversions

Critical for QPS and throughput calculations:

| Unit | Seconds | Useful For |
|------|---------|------------|
| 1 minute | 60 sec | Short duration calculations |
| 1 hour | 3,600 sec | Hourly metrics |
| 1 day | 86,400 sec | Daily active users → QPS |
| 1 month | 2,592,000 sec | Monthly scale (30 days) |
| 1 year | 31,536,000 sec | Long-term storage |

**Approximations for mental math:**

- 1 day ≈ 100,000 seconds (actually 86,400)
- 1 month ≈ 2.5 million seconds
- 1 year ≈ 30 million seconds

### Typical Data Sizes

Use these as baselines:

| Data Type | Size | Notes |
|-----------|------|-------|
| Character (ASCII) | 1 byte | |
| Character (Unicode) | 2-4 bytes | Average 2 bytes |
| Integer | 4 bytes | 32-bit int |
| Long | 8 bytes | 64-bit int |
| Timestamp | 8 bytes | Unix timestamp |
| UUID | 16 bytes | 128-bit |
| MD5 Hash | 16 bytes | |
| SHA-256 Hash | 32 bytes | |
| Tweet (text only) | 200 bytes | 280 chars + metadata |
| Small JSON | 1 KB | API response |
| Web page (HTML) | 100 KB | Without images |
| Photo (compressed) | 200 KB | JPEG, medium quality |
| Photo (high-res) | 2 MB | Original quality |
| Video (1 min, HD) | 50 MB | 1080p, compressed |
| Video (1 min, 4K) | 200 MB | 4K, compressed |

### Network & Latency

Important for understanding system performance:

| Operation | Latency | Notes |
|-----------|---------|-------|
| L1 cache reference | 0.5 ns | |
| L2 cache reference | 7 ns | |
| RAM reference | 100 ns | |
| Send 1KB over 1 Gbps | 10 μs | |
| SSD random read | 150 μs | |
| HDD seek | 10 ms | |
| Round trip in datacenter | 0.5 ms | |
| Round trip CA to Netherlands | 150 ms | |

**Bandwidth Conversions:**

- 1 Gbps = 125 MB/s
- 1 Gbps = 10 billion bits per second
- Typical server NIC: 1-10 Gbps
- Typical home internet: 100-1000 Mbps

---

## QPS Calculations

### Formula

```
QPS (Queries Per Second) = Total Operations / Time Period (seconds)
```

### Example 1: Twitter Reads

```
Given:
- 200M daily active users (DAU)
- Each user views 50 tweets per day

Calculate daily reads:
  200M users × 50 tweets = 10 billion reads/day

Calculate QPS:
  10B reads / 86,400 seconds = ~116,000 reads/sec

Calculate peak QPS (2x average):
  Peak QPS = 232,000 reads/sec
```

### Example 2: URL Shortener Writes

```
Given:
- 100M short URLs created per month

Calculate daily writes:
  100M / 30 days = ~3.3M writes/day

Calculate QPS:
  3.3M / 86,400 = ~38 writes/sec

Calculate peak QPS (2x):
  Peak QPS = 76 writes/sec
```

### Example 3: Mixed Read/Write

```
Given:
- 1M posts created per day
- 100:1 read-to-write ratio

Calculate write QPS:
  1M posts / 86,400 sec = ~12 writes/sec

Calculate read QPS:
  12 writes/sec × 100 = 1,200 reads/sec

Total QPS:
  12 + 1,200 = 1,212 QPS
```

### Quick Reference Table

| Scale | Daily Ops | QPS | Peak QPS (2x) |
|-------|-----------|-----|---------------|
| Small | 1M | ~12 | ~24 |
| Medium | 10M | ~120 | ~240 |
| Large | 100M | ~1,200 | ~2,400 |
| Very Large | 1B | ~12,000 | ~24,000 |
| Massive | 10B | ~120,000 | ~240,000 |

---

## Storage Calculations

### Formula

```
Storage = Number of Objects × Size per Object × Time Period
```

### Example 1: Tweet Storage

```
Given:
- 500M tweets per day
- Each tweet: 200 bytes (text + metadata)
- Retention: 5 years

Calculate daily storage:
  500M × 200 bytes = 100 GB/day

Calculate 5-year storage:
  100 GB/day × 365 days × 5 years = 182.5 TB

Round up for overhead:
  ~200 TB for 5 years
```

### Example 2: Photo Storage (Instagram)

```
Given:
- 100M photos uploaded per day
- Each photo: 2 MB (original)
- Store 3 versions: original, medium (500 KB), thumbnail (50 KB)
- Retention: Forever

Calculate storage per photo:
  Original: 2 MB
  Medium: 500 KB = 0.5 MB
  Thumbnail: 50 KB = 0.05 MB
  Total: 2.55 MB per photo

Calculate daily storage:
  100M photos × 2.55 MB = 255 TB/day

Calculate yearly storage:
  255 TB/day × 365 = 93 PB/year

Note: This is massive! Needs compression, deduplication, cold storage.
```

### Example 3: Video Storage (YouTube)

```
Given:
- 500 hours of video uploaded per minute
- Average video quality: 1080p
- Bitrate: 5 Mbps = 0.625 MB/s
- Store multiple qualities: 4K, 1080p, 720p, 480p

Calculate storage for 1 hour of video (1080p):
  1 hour = 3,600 seconds
  3,600 sec × 0.625 MB/s = 2,250 MB = ~2.25 GB

Calculate daily uploads (1080p only):
  500 hours/min × 60 min/hour × 24 hours = 720,000 hours/day
  720,000 hours × 2.25 GB = 1.62 PB/day

With multiple qualities (3x storage):
  1.62 PB × 3 = ~5 PB/day

This requires massive infrastructure!
```

### Storage Estimation Template

```
1. Identify object types (tweets, photos, videos, etc.)
2. Estimate count per day
3. Estimate size per object
4. Calculate daily storage: count × size
5. Calculate total storage: daily × retention period
6. Add 20-30% overhead (metadata, indexes, replication)
```

---

## Bandwidth Calculations

### Formula

```
Bandwidth = Data Size × QPS
```

### Example 1: URL Shortener

```
Given:
- 4,000 redirects per second
- Each redirect: 500 bytes (HTTP response)

Calculate bandwidth:
  4,000 req/sec × 500 bytes = 2 MB/sec

Convert to Mbps:
  2 MB/sec × 8 = 16 Mbps

Conclusion: Single server with 1 Gbps NIC can easily handle this.
```

### Example 2: Video Streaming (Netflix)

```
Given:
- 10M concurrent streams
- Average bitrate: 5 Mbps per stream

Calculate total bandwidth:
  10M streams × 5 Mbps = 50,000,000 Mbps = 50 Tbps

This is why Netflix uses CDN!
- Distribute load across edge locations globally
- Each edge location handles a fraction of traffic
```

### Example 3: Image CDN

```
Given:
- 100,000 image requests per second
- Average image size: 200 KB

Calculate bandwidth:
  100,000 req/sec × 200 KB = 20 GB/sec

Convert to Gbps:
  20 GB/sec × 8 = 160 Gbps

Conclusion: Need multiple servers or CDN
  - 10 servers with 10 Gbps each = 100 Gbps capacity
  - Add CDN for geographic distribution
```

---

## Server Calculations

### Formula

```
Number of Servers = Total Load / Capacity per Server
```

### Example 1: API Servers

```
Given:
- 10,000 QPS total
- Each server handles 1,000 QPS

Calculate servers needed:
  10,000 / 1,000 = 10 servers

Add redundancy (3x for high availability):
  10 × 1.3 = 13 servers

Note: 30% extra for peaks and failover
```

### Example 2: Database Servers

```
Given:
- 5,000 read QPS
- 500 write QPS
- Single DB handles: 10,000 reads, 1,000 writes

Option 1: Single master-slave setup
  Master: Handles 500 writes (well below 1,000 limit)
  Slaves: 5,000 reads / 10,000 per slave = 1 slave (add 2 more for redundancy)
  Total: 1 master + 3 slaves = 4 DB servers

Option 2: Sharded setup (if needed)
  Not needed yet, current scale fits single DB
```

### Example 3: WebSocket Servers

```
Given:
- 1M concurrent connections
- Each server handles 10,000 connections

Calculate servers:
  1M / 10,000 = 100 servers

Add redundancy:
  100 × 1.2 = 120 servers

Note: WebSocket servers are mostly idle, just maintaining connections
```

### Server Capacity Reference

| Server Type | Typical Capacity | Notes |
|-------------|------------------|-------|
| API Server | 1,000-5,000 QPS | CPU-bound |
| Static File Server | 10,000-50,000 QPS | I/O-bound |
| Database (SQL) | 10,000 reads, 1,000 writes | Depends on query complexity |
| Cache (Redis) | 100,000+ ops/sec | Very fast |
| WebSocket | 10,000 connections | Memory-bound |
| Message Queue | 100,000+ msg/sec | Kafka, RabbitMQ |

---

## Cost Estimation

### Cloud Pricing Approximations (2026)

**Compute (per month):**

- Small instance (2 vCPU, 4 GB RAM): $50
- Medium instance (4 vCPU, 8 GB RAM): $100
- Large instance (8 vCPU, 16 GB RAM): $200

**Storage (per month):**

- SSD (per TB): $100
- HDD (per TB): $25
- S3/Object storage (per TB): $23

**Bandwidth (per TB):**

- Outbound: $90
- Inbound: Free
- Within datacenter: Free

**Database (managed):**

- Small (2 vCPU, 4 GB): $100
- Medium (4 vCPU, 8 GB): $200
- Large (8 vCPU, 16 GB): $400

### Example: URL Shortener Cost

```
Components:
1. API Servers: 5 medium instances
   5 × $100 = $500/month

2. Load Balancer: 1 managed LB
   $100/month

3. Database: 1 large (master) + 2 large (replicas)
   3 × $400 = $1,200/month

4. Cache: 1 medium Redis
   $200/month

5. Storage: 3 TB
   3 × $100 = $300/month

6. Bandwidth: 10 TB/month
   10 × $90 = $900/month

Total: $3,200/month

Annual: $38,400/year

Per user cost (if 10M users):
  $38,400 / 10M = $0.00384 per user per year
```

### Optimization Tips

```
1. Reserved instances: 30-50% savings
2. Spot instances for batch jobs: 70% savings
3. CDN for static content: Reduce bandwidth costs
4. Aggressive caching: Reduce database load
5. Compression: Reduce storage and bandwidth
6. Cold storage for old data: 80% cheaper
```

---

## Calculation Template

Use this template for any system:

### Step 1: Define Scale

```
Users:
- Total users: [X]
- Daily active users (DAU): [Y]
- Monthly active users (MAU): [Z]

Activity:
- Actions per user per day: [A]
- Read vs write ratio: [R:W]
```

### Step 2: Calculate QPS

```
Write QPS:
  [DAU] × [Actions] / 86,400 = [W] writes/sec

Read QPS:
  [W] × [Read ratio] = [R] reads/sec

Total QPS: [W] + [R] = [Total]

Peak QPS (2x): [Total] × 2 = [Peak]
```

### Step 3: Calculate Storage

```
Per object:
- Object type: [Type]
- Size: [Size] bytes
- Count per day: [Count]

Daily storage:
  [Count] × [Size] = [Daily] GB/day

Retention:
  [Daily] × 365 × [Years] = [Total] TB

Add overhead (30%):
  [Total] × 1.3 = [Final] TB
```

### Step 4: Calculate Bandwidth

```
Ingress (uploads):
  [Write QPS] × [Object size] = [X] MB/sec

Egress (downloads):
  [Read QPS] × [Object size] = [Y] MB/sec

Total bandwidth:
  [X + Y] MB/sec = [Z] Gbps
```

### Step 5: Calculate Servers

```
API servers:
  [Total QPS] / [QPS per server] = [N] servers
  Add redundancy: [N] × 1.3 = [M] servers

Database servers:
  [Write QPS] → [Master count]
  [Read QPS] / [Reads per slave] = [Slave count]

Cache servers:
  [Cache size needed] / [Memory per server] = [Cache servers]
```

### Step 6: Calculate Cost

```
Compute: [Instances] × [Price] = $[X]
Storage: [TB] × [Price/TB] = $[Y]
Bandwidth: [TB] × [Price/TB] = $[Z]
Database: [Instances] × [Price] = $[W]

Total: $[X + Y + Z + W] per month
```

---

## Example Walkthrough: Twitter

Let's estimate Twitter's infrastructure needs.

### Given Requirements

```
Users:
- 300M monthly active users
- 50M daily active users (DAU)
- 100M tweets per day

Activity:
- Average user reads 200 tweets per day
- 100:1 read-to-write ratio
- Tweet size: 200 bytes (text + metadata)
- Store tweets forever
```

### Step 1: Calculate QPS

**Write QPS:**

```
100M tweets/day / 86,400 sec = ~1,160 writes/sec
Peak: 1,160 × 2 = 2,320 writes/sec
```

**Read QPS:**

```
50M DAU × 200 tweets/day = 10B reads/day
10B / 86,400 = ~116,000 reads/sec
Peak: 116,000 × 2 = 232,000 reads/sec
```

### Step 2: Calculate Storage

**Tweet storage:**

```
Daily: 100M tweets × 200 bytes = 20 GB/day
Yearly: 20 GB × 365 = 7.3 TB/year
5 years: 7.3 TB × 5 = 36.5 TB

Add media (photos, videos) - assume 10% of tweets:
  10M tweets × 200 KB avg = 2 TB/day
  2 TB × 365 × 5 = 3.65 PB

Total 5-year storage: ~3.7 PB
```

### Step 3: Calculate Bandwidth

**Ingress (tweet creation):**

```
1,160 tweets/sec × 200 bytes = 232 KB/sec = ~2 Mbps
Negligible for modern servers
```

**Egress (reading tweets):**

```
116,000 reads/sec × 200 bytes = 23 MB/sec = 184 Mbps
Peak: 368 Mbps

With media:
  Assume 30% of reads include media (avg 100 KB)
  34,800 media reads/sec × 100 KB = 3.48 GB/sec = 27.8 Gbps
```

**Conclusion:** Need CDN for media delivery.

### Step 4: Calculate Servers

**API servers:**

```
Capacity: 1,000 QPS per server (assume complex operations)

Total QPS: 1,160 writes + 116,000 reads = 117,160 QPS

Servers: 117,160 / 1,000 = 118 servers
With redundancy: 118 × 1.3 = ~154 servers
```

**Database servers:**

```
Master (writes):
  1,160 writes/sec - single master can handle this
  Add 1 hot standby for failover

Slaves (reads):
  116,000 reads/sec
  Assume 10,000 reads/sec per slave
  116,000 / 10,000 = 12 read replicas
  With redundancy: 12 × 1.3 = ~16 read replicas

Total DB: 1 master + 1 standby + 16 replicas = 18 DB servers
```

**Cache servers:**

```
Hot tweets in cache: Assume 20% of tweets accessed frequently
  36.5 TB × 0.2 = 7.3 TB cache size

Redis memory: 64 GB per instance
  7.3 TB / 64 GB = 114 instances

With replication (master-slave): 114 × 2 = 228 cache instances

Note: This is expensive - optimize by:
  - Only cache last 24 hours (7.3 TB → 20 GB, manageable!)
  - Use LRU eviction
```

### Step 5: Calculate Cost

```
API Servers: 154 medium instances
  154 × $100 = $15,400/month

Load Balancers: 10 LBs
  10 × $100 = $1,000/month

Database: 18 large instances
  18 × $400 = $7,200/month

Cache: 10 Redis instances (24h cache only)
  10 × $200 = $2,000/month

Storage: 3.7 PB
  3,700 TB × $23 (S3) = $85,100/month

Bandwidth: ~100 TB/month (with CDN)
  100 × $90 = $9,000/month

CDN: 500 TB/month
  Negotiated rate: ~$20/TB = $10,000/month

Total: ~$130,000/month = $1.56M/year

Per MAU: $1.56M / 300M = $0.0052/user/year
```

### Conclusion

```
Infrastructure Summary:
- 154 API servers
- 18 database servers
- 10 cache servers
- 10 load balancers
- CDN for global delivery
- ~3.7 PB storage

Cost: ~$130K/month

Bottlenecks:
- Storage cost is highest (65% of total)
- Read QPS requires significant caching
- CDN essential for media delivery

Optimizations:
- Archive old tweets to cold storage
- Aggressive caching (24h window)
- CDN for static content
- Compress media aggressively
```

---

## Practice Problems

### Problem 1: Instagram Photo Storage

```
Given:
- 100M photos uploaded per day
- Each photo: 2 MB average
- Store 3 versions (original, medium, thumb)
- Retention: Forever

Calculate:
1. Daily storage needed
2. 5-year storage
3. Monthly cost (at $23/TB)
```

<details>
<summary>Solution</summary>

```
Per photo storage:
  Original: 2 MB
  Medium: 500 KB = 0.5 MB
  Thumb: 50 KB = 0.05 MB
  Total: 2.55 MB

Daily storage:
  100M × 2.55 MB = 255 TB/day

5-year storage:
  255 TB/day × 365 × 5 = 465,375 TB = ~465 PB

Monthly cost:
  255 TB/day × 30 = 7,650 TB/month
  7,650 × $23 = $175,950/month

Optimization needed:
  - Deduplication
  - Compression
  - Cold storage for old photos
```
</details>

---

### Problem 2: WhatsApp Message Volume

```
Given:
- 1B monthly active users
- 100M daily active users
- Each user sends 50 messages per day
- Each message: 1 KB average

Calculate:
1. Write QPS
2. Read QPS (assume 5:1 read-to-write)
3. Daily storage
4. Yearly storage
```

<details>
<summary>Solution</summary>

```
Write QPS:
  100M DAU × 50 messages = 5B messages/day
  5B / 86,400 = ~57,870 writes/sec

Read QPS:
  57,870 × 5 = 289,350 reads/sec

Daily storage:
  5B messages × 1 KB = 5 TB/day

Yearly storage:
  5 TB × 365 = 1,825 TB = ~1.8 PB/year
```
</details>

---

### Problem 3: YouTube Video Bandwidth

```
Given:
- 1B hours watched per day
- Average bitrate: 5 Mbps

Calculate:
1. Total bandwidth needed
2. Number of CDN edge locations (if each handles 100 Gbps)
```

<details>
<summary>Solution</summary>

```
Bandwidth calculation:
  1B hours/day = 1B × 3,600 sec = 3.6 trillion seconds/day
  3.6T seconds/day / 86,400 = ~41.6M concurrent streams

Total bandwidth:
  41.6M streams × 5 Mbps = 208 million Mbps = 208 Tbps

CDN edge locations:
  208 Tbps / 100 Gbps = 2,080 edge locations

In reality: Netflix has ~thousands of edge locations globally
```
</details>

---

### Problem 4: Netflix CDN Requirements

```
Given:
- 200M subscribers
- 10M concurrent streams during peak
- HD stream: 5 Mbps
- 4K stream: 25 Mbps
- 80% HD, 20% 4K

Calculate bandwidth during peak
```

<details>
<summary>Solution</summary>

```
HD streams: 10M × 0.8 = 8M streams
4K streams: 10M × 0.2 = 2M streams

HD bandwidth:
  8M × 5 Mbps = 40 million Mbps = 40 Tbps

4K bandwidth:
  2M × 25 Mbps = 50 million Mbps = 50 Tbps

Total: 90 Tbps during peak

This is why CDN is essential!
```
</details>

---

## Tips for Interviews

### 1. Show Your Work

```
Don't just state the answer. Show calculations:

Bad: "We need 100 servers."

Good: "Let me calculate:
       Total QPS: 10,000
       QPS per server: 1,000
       10,000 / 1,000 = 10 base servers
       Add 30% for redundancy: 10 × 1.3 = 13 servers"
```

### 2. Round Sensibly

```
86,400 seconds/day ≈ 100,000 (makes mental math easier)

Example:
  100M requests/day / 100,000 = 1,000 QPS
  vs
  100M / 86,400 = 1,157 QPS

The difference doesn't matter for high-level design!
```

### 3. State Assumptions

```
"I'm assuming a tweet is about 200 bytes - 280 characters plus some
metadata. If that's different, I can recalculate."

"I'm using 2x for peak traffic. If your system has more extreme peaks,
we'd need to adjust."
```

### 4. Sanity Check

```
After calculations, ask yourself:
- Does this number make sense?
- Is it too small or too large?
- Can I compare to known systems?

Example: "10 PB/day seems huge. Let me double-check...
          Actually, YouTube uploads 500 hours/minute, so that's plausible."
```

---

## Related Resources

- [Framework](framework.md) - When to do calculations in the interview
- [Practice Problems](practice-problems.md) - Apply these calculations to real problems
- [Communication Guide](communication.md) - How to explain your calculations
- [Common Mistakes](common-mistakes.md) - Calculation mistakes to avoid
- [Back to Interviews](index.md)
