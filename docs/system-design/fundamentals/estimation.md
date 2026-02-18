# Back-of-Envelope Estimation

Every system design discussion eventually reaches the question: "Will this actually work at
scale?" Back-of-envelope estimation is the practice of making rough calculations using
simplified assumptions to determine whether a design is feasible before writing any code.
The goal is not precision -- it is directional correctness. You need to know whether your
system needs 10 servers or 10,000, whether your data fits in memory or requires distributed
storage, and whether your network can handle the traffic. Getting within an order of
magnitude is good enough to make architectural decisions. Getting it wrong by three orders
of magnitude leads to systems that collapse under real load or waste millions on
over-provisioned infrastructure.

```
               Back-of-Envelope Estimation Flow
               ================================

  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  Understand   │    │  Identify Key     │    │  Make Reasonable  │
  │  the Problem  │───>│  Parameters       │───>│  Assumptions      │
  │              │    │                  │    │                  │
  │  "Design a   │    │  - Users          │    │  - Avg tweet size │
  │   Twitter"   │    │  - Actions/day    │    │  - Read/write     │
  │              │    │  - Data per action │    │    ratio          │
  └──────────────┘    └──────────────────┘    └────────┬─────────┘
                                                       │
                                                       v
  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  Validate &   │    │  Convert to       │    │  Calculate Core   │
  │  Sanity Check │<───│  Infrastructure   │<───│  Metrics          │
  │              │    │                  │    │                  │
  │  "Does this  │    │  - Servers needed │    │  - QPS            │
  │   make       │    │  - Storage (TB)   │    │  - Storage/day    │
  │   sense?"    │    │  - Bandwidth      │    │  - Bandwidth      │
  └──────────────┘    └──────────────────┘    └──────────────────┘
```

---

=== "Numbers to Know"

    ## Numbers Every Engineer Should Know

    Jeff Dean's famous latency numbers give you the intuition for how fast operations
    actually are. These numbers shift with hardware generations, but the relative
    magnitudes stay remarkably stable. Knowing these lets you instantly reason about
    whether a design will be fast enough.

    ### Latency Numbers

    ```
    Operation                              Time          Relative Scale
    ─────────────────────────────────────────────────────────────────────
    L1 cache reference                     0.5 ns        ■
    Branch mispredict                      5 ns          ■■
    L2 cache reference                     7 ns          ■■
    Mutex lock/unlock                      25 ns         ■■■
    Main memory reference                  100 ns        ■■■■■
    Compress 1KB with Snappy               3,000 ns      ■■■■■■■■■■
    Send 1KB over 1 Gbps network           10,000 ns     ■■■■■■■■■■■■
    Read 4KB randomly from SSD             150,000 ns    ■■■■■■■■■■■■■■■
    Read 1MB sequentially from memory      250,000 ns    ■■■■■■■■■■■■■■■■
    Round trip within same datacenter      500,000 ns    ■■■■■■■■■■■■■■■■■
    Read 1MB sequentially from SSD         1,000,000 ns  ■■■■■■■■■■■■■■■■■■
    Disk seek                              10,000,000 ns ■■■■■■■■■■■■■■■■■■■■■
    Read 1MB sequentially from disk        20,000,000 ns ■■■■■■■■■■■■■■■■■■■■■■
    Send packet CA -> Netherlands -> CA    150,000,000 ns■■■■■■■■■■■■■■■■■■■■■■■■■
    ```

    !!! tip "Key Intuitions from These Numbers"
        - **Memory is 1000x faster than SSD, SSD is 100x faster than disk.** This is
          why caching strategies matter so much. Moving data from disk to memory gives
          you three orders of magnitude improvement.
        - **Network within a datacenter (~500us) is 300x faster than cross-continent
          (~150ms).** This is why multi-region architectures need careful design around
          which data must cross regions.
        - **Sequential reads are dramatically faster than random reads.** Append-only
          logs (Kafka, LSM trees) exploit this by converting random writes into
          sequential ones.
        - **Compression is fast.** At 3us for 1KB, compressing data before sending it
          over the network (10us for 1KB) almost always saves time for data larger than
          a few hundred bytes.

    ### Latency Comparison Visual

    ```
    1 ns   ├─ L1 cache
    10 ns  ├─ L2 cache, branch mispredict
    100 ns ├─ Main memory                          ← "fast"
           │
    1 μs   │
    10 μs  ├─ Network send 1KB
    100 μs ├─ SSD random read                      ← "okay"
           │
    1 ms   ├─ SSD read 1MB
    10 ms  ├─ Disk seek                            ← "slow"
    100 ms ├─ Cross-continent round trip            ← "user-visible"
    1 s    ├─ User patience threshold
    ```

    ### Powers of 2 Reference

    Computers deal in powers of 2. You will use these constantly for storage and memory
    calculations.

    | Power | Exact Value | Approx. Size | Common Name |
    |-------|-------------|-------------|-------------|
    | 2^10  | 1,024       | ~1 Thousand | 1 KB (Kilobyte) |
    | 2^20  | 1,048,576   | ~1 Million  | 1 MB (Megabyte) |
    | 2^30  | 1,073,741,824 | ~1 Billion | 1 GB (Gigabyte) |
    | 2^40  | 1,099,511,627,776 | ~1 Trillion | 1 TB (Terabyte) |
    | 2^50  | 1,125,899,906,842,624 | ~1 Quadrillion | 1 PB (Petabyte) |

    !!! note "Practical Shortcuts"
        - 1 Million bytes is roughly 1 MB
        - 1 Billion bytes is roughly 1 GB
        - For quick math, treat KB/MB/GB/TB as exact powers of 1000
        - A single server typically has 64-256 GB RAM and 1-10 TB disk

    ### Common Data Size References

    | Data Type | Typical Size |
    |-----------|-------------|
    | A single character (ASCII) | 1 byte |
    | A single character (UTF-8, English) | 1 byte |
    | A single character (UTF-8, Chinese/emoji) | 3-4 bytes |
    | A UUID | 16 bytes |
    | A tweet (280 chars + metadata) | ~1 KB |
    | A typical web page | 2-3 MB |
    | A smartphone photo | 3-5 MB |
    | A minute of HD video (compressed) | 5-10 MB |
    | A minute of 4K video (compressed) | 30-50 MB |

    ### Availability Numbers

    System uptime is commonly expressed in "nines." Each additional nine reduces
    allowed downtime by 10x.

    | Availability | Downtime/Year | Downtime/Month | Downtime/Week |
    |-------------|---------------|----------------|---------------|
    | 99% (two nines) | 3.65 days | 7.31 hours | 1.68 hours |
    | 99.9% (three nines) | 8.77 hours | 43.8 minutes | 10.1 minutes |
    | 99.99% (four nines) | 52.6 minutes | 4.38 minutes | 1.01 minutes |
    | 99.999% (five nines) | 5.26 minutes | 26.3 seconds | 6.05 seconds |

    Most web services target three to four nines. Five nines is exceptionally hard and
    typically requires active-active multi-region deployment with automatic failover.

=== "Estimation Method"

    ## Step-by-Step Estimation Methodology

    A structured approach prevents you from getting lost in numbers. Follow these five
    steps for any estimation problem.

    ### Step 1: Clarify the Scope

    Before calculating anything, identify exactly what you are estimating. A vague
    question like "how much storage does Twitter need?" has many valid answers depending
    on whether you mean tweets only, media, user profiles, indexes, or all of the above.

    !!! tip "Always Ask"
        - What time period? (per day, per year, for 5-year growth)
        - What data? (just the primary content, or metadata/indexes too)
        - Peak or average? (peak can be 3-10x average)
        - Read or write path? (they often have very different requirements)

    ### Step 2: Estimate the Scale

    Start with users, then work down to actions per user.

    ```
    Total Users  ──>  Daily Active Users (DAU)  ──>  Actions per User
         │                    │                            │
         │                    │                            v
         │                    │                     Total Actions/Day
         │                    v                            │
         │              DAU = Total x                      v
         │              Active Ratio             Actions/Second (QPS)
         │              (typically 20-40%)        = Actions/Day ÷ 86,400
         v
    Registered users
    (from problem statement
    or public data)
    ```

    !!! note "The 86,400 Rule"
        There are 86,400 seconds in a day (24 x 60 x 60). For quick mental math,
        round to ~100,000 (10^5). This makes division easy:

        - 1 million actions/day = ~10 QPS
        - 100 million actions/day = ~1,000 QPS
        - 1 billion actions/day = ~10,000 QPS

    ### Step 3: Estimate Per-Unit Resource Usage

    For each action, estimate how much of each resource it consumes.

    | Category | What to Estimate | How to Think About It |
    |----------|-----------------|----------------------|
    | **Storage** | Bytes per write action | Sum up all fields: text, metadata, IDs, timestamps |
    | **Bandwidth** | Bytes in + bytes out per request | Request size + response size, multiply by QPS |
    | **Memory** | Bytes of hot data that must be cached | Typically 20% of daily data, or top-N items |
    | **Compute** | CPU time per request | Usually estimated as QPS per server (varies by operation) |

    ### Step 4: Multiply and Convert

    Apply dimensional analysis to get the final infrastructure numbers.

    ```
    Storage per day  = (actions/day) x (bytes/action)
    Storage per year = storage/day x 365
    Bandwidth        = QPS x (bytes/request)
    Servers needed   = Peak QPS ÷ QPS per server
    Cache size       = hot data set x replication factor
    ```

    !!! warning "Don't Forget These Multipliers"
        - **Replication:** Most production systems replicate data 3x
        - **Peak vs average:** Peak traffic is typically 2-5x average
        - **Overhead:** Indexes, metadata, and filesystem overhead add 20-50%
        - **Growth:** Design for 3-5 years of growth (multiply current by 3-10x)

    ### Step 5: Sanity Check

    Compare your results against known real-world numbers and common sense.

    | If Your Estimate Says... | Reality Check |
    |-------------------------|--------------|
    | Single server handles it | Probably correct for < 1K QPS with simple operations |
    | Need 10-100 servers | Reasonable for a large-scale service |
    | Need 10,000+ servers | You're either Google-scale or made an error |
    | Need petabytes of storage/year | Only applies to video/image-heavy services |
    | Bandwidth exceeds 1 Tbps | Recheck -- this is datacenter-level traffic |

=== "Worked Examples"

    ## Example 1: Twitter Storage Estimation

    **Problem:** Estimate Twitter's daily storage requirement for tweets.

    **Known facts:**

    - ~500 million tweets per day
    - Each tweet: up to 280 characters
    - Approximately 10% of tweets contain media (images/video handled separately)

    **Step-by-step calculation:**

    ```
    Tweet text storage:
    ───────────────────
    Average tweet length:         ~140 characters (half of 280 limit)
    Character encoding (UTF-8):   ~2 bytes/char average (multilingual)
    Text per tweet:               140 x 2 = 280 bytes

    Metadata per tweet:
    ───────────────────
    tweet_id (snowflake):         8 bytes
    user_id:                      8 bytes
    timestamp:                    8 bytes
    reply/retweet references:     16 bytes
    counters (likes, RTs, etc.):  32 bytes
    geo/source metadata:          ~30 bytes
    ──────────────────────────────────────
    Metadata subtotal:            ~100 bytes

    Total per tweet:              280 + 100 = ~400 bytes
                                  Round up to ~500 bytes (overhead, indexes)

    Daily storage:
    ──────────────
    500M tweets/day x 500 bytes = 250 GB/day (text + metadata only)

    With 3x replication:          250 x 3 = 750 GB/day
    Annual storage:               750 GB x 365 = ~274 TB/year
    5-year projection:            ~1.4 PB (text + metadata only)
    ```

    **QPS calculation:**

    ```
    Write QPS:
    ──────────
    500M tweets / 86,400 seconds = ~5,800 tweets/second average
    Peak (assume 3x):              ~17,400 tweets/second

    Read QPS (assume 100:1 read-to-write ratio for timelines):
    ─────────────────────────────────────────────────────────
    Average: 5,800 x 100 = 580,000 reads/second
    Peak:    ~1.7 million reads/second
    ```

    !!! note "What This Tells the Architect"
        - Text storage alone is manageable (~250 GB/day). The real storage challenge
          is media (images and video dwarf text by 1000x).
        - Read QPS is enormous (580K/s). This demands aggressive caching -- you cannot
          serve timelines directly from a database at this scale.
        - Write QPS (~6K/s) is moderate for a distributed system. The fan-out problem
          (delivering each tweet to millions of followers) is the harder challenge.

    ---

    ## Example 2: YouTube Bandwidth Estimation

    **Problem:** Estimate YouTube's bandwidth requirements for video streaming.

    **Known facts:**

    - ~500 hours of video uploaded per minute
    - ~1 billion hours of video watched per day
    - Average video resolution mix: 30% at 480p, 40% at 720p, 25% at 1080p, 5% at 4K

    **Upload bandwidth:**

    ```
    Video upload rate:
    ──────────────────
    500 hours/minute = 30,000 hours/hour = 720,000 hours/day

    Average upload bitrate (pre-transcoding): ~5 Mbps

    Upload bandwidth:
    ─────────────────
    500 hours/min x 60 min = 30,000 hours of video arriving per hour
    30,000 hours x 5 Mbps = 150,000 Mbps = 150 Gbps sustained upload

    Storage per day:
    ────────────────
    720,000 hours x 3,600 seconds/hour x 5 Mbps = ~1.6 PB/day (raw)
    After transcoding to multiple resolutions: ~3-5 PB/day
    ```

    **Streaming (download) bandwidth:**

    ```
    Viewing rate:
    ─────────────
    1 billion hours/day = ~41.7 million hours/hour

    Average streaming bitrate (weighted by resolution):
    ──────────────────────────────────────────────────
    480p @ 1.5 Mbps x 30% = 0.45 Mbps
    720p @ 3.0 Mbps x 40% = 1.20 Mbps
    1080p @ 6.0 Mbps x 25% = 1.50 Mbps
    4K   @ 20 Mbps  x 5%  = 1.00 Mbps
    ────────────────────────────────────
    Weighted average:         ~4.15 Mbps

    Streaming bandwidth:
    ────────────────────
    Concurrent viewers at any moment:
    1B hours/day ÷ 24 hours = ~41.7M concurrent hours
    41.7M viewers x 4.15 Mbps = ~173 Tbps

    Peak (assume 2x average): ~346 Tbps
    ```

    !!! note "What This Tells the Architect"
        - YouTube's bandwidth is staggering (~170 Tbps average). This is why they
          operate their own CDN (Google Global Cache) with edge servers in ISP networks.
        - Upload storage (~3-5 PB/day) means YouTube adds an exabyte roughly every
          six months. Cold storage tiers and intelligent archival are essential.
        - No single datacenter handles this. Geographic distribution with CDN caching
          of popular content is the only viable architecture.

    ---

    ## Example 3: Quick Estimation -- Chat Application

    **Problem:** Estimate storage for a WhatsApp-like chat system.

    ```
    Scale:
    ──────
    2 billion users, 500 million DAU
    Average messages per user per day: 40
    Average message size: 100 bytes (text) + 100 bytes (metadata) = 200 bytes

    Daily messages:   500M x 40 = 20 billion messages/day
    Daily storage:    20B x 200 bytes = 4 TB/day
    With replication: 4 TB x 3 = 12 TB/day
    Annual storage:   12 TB x 365 = ~4.4 PB/year

    QPS:
    ────
    20B messages / 86,400 = ~230,000 messages/second average
    Peak (3x):              ~700,000 messages/second
    ```

=== "Common Estimation Categories"

    ## The Four Core Categories

    Every system design estimation ultimately answers questions in four areas.
    Understanding what drives each one helps you focus your calculations.

    ```
    ┌─────────────────────────────────────────────────────────┐
    │                System Resource Model                    │
    │                                                         │
    │   ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌──────┐ │
    │   │  QPS    │  │  Storage  │  │ Bandwidth │  │Memory│ │
    │   │         │  │           │  │           │  │      │ │
    │   │ How many│  │ How much  │  │ How much  │  │ What │ │
    │   │requests │  │ data over │  │ data/sec  │  │ fits │ │
    │   │per sec? │  │ time?     │  │ in/out?   │  │in RAM│ │
    │   └────┬────┘  └─────┬─────┘  └─────┬─────┘  └──┬───┘ │
    │        │             │              │            │     │
    │        v             v              v            v     │
    │   # Servers     # Disks/       Network        Cache   │
    │   # Cores       Storage        capacity       size    │
    │                 nodes          planning                │
    └─────────────────────────────────────────────────────────┘
    ```

    ### QPS (Queries Per Second)

    QPS determines how many servers and how much compute you need.

    | Metric | How to Calculate | Typical Values |
    |--------|-----------------|----------------|
    | Average QPS | DAU x actions/user / 86,400 | Varies widely |
    | Peak QPS | Average QPS x peak multiplier | 2-5x average |
    | Read QPS | Total QPS x read ratio | Often 90-99% of traffic |
    | Write QPS | Total QPS x write ratio | Often 1-10% of traffic |

    **Server capacity rules of thumb:**

    - A single web server handles ~1,000-10,000 simple QPS (static/cached)
    - A single web server handles ~100-1,000 QPS for dynamic requests with DB calls
    - A single database server handles ~1,000-10,000 read QPS (indexed queries)
    - A single database server handles ~100-1,000 write QPS (depends on durability)

    ### Storage

    Storage estimation determines disk capacity and database sizing.

    ```
    Formula:  Daily storage = Write QPS x 86,400 x bytes per write

    Timeline: ──────────────────────────────────────────────>
              Day 1       Month 1      Year 1       Year 5
              raw size    x 30         x 365        x 1,825
                          + indexes    + replicas   + growth
                          + replicas   + backups    + backups
    ```

    !!! tip "Storage Estimation Checklist"
        1. Calculate raw data size per write
        2. Multiply by writes per day
        3. Add index overhead (typically 10-30% of data size)
        4. Apply replication factor (usually 3x)
        5. Add backup storage (often 1-2 additional copies)
        6. Project for 3-5 years

    ### Bandwidth

    Bandwidth determines network capacity requirements.

    ```
    Ingress (incoming):   Write QPS x average request size
    Egress (outgoing):    Read QPS x average response size

    Note: Egress is almost always much larger than ingress.
    A single image view sends 200KB out for a 200-byte request in.
    ```

    ### Memory (Cache Sizing)

    Most systems cache frequently accessed data to reduce database load.

    **The 80/20 Rule for Caching:**

    20% of the data serves 80% of the requests. Caching the top 20% of daily data
    usually provides a dramatic hit rate improvement.

    ```
    Cache size = Daily data volume x 0.20

    Example:
    ────────
    If your system writes 100 GB/day of data
    Cache the top 20%: 100 GB x 0.20 = 20 GB
    With replication across cache nodes: 20 GB x 3 = 60 GB
    This fits in a single large-memory server (64-256 GB RAM)
    ```

=== "Pitfalls & Tips"

    ## Common Mistakes in Estimation

    Even experienced engineers make systematic errors in estimation. Knowing the common
    traps helps you avoid them.

    ### Mistake 1: Forgetting the Read/Write Ratio

    Most systems are read-heavy. A 100:1 read-to-write ratio is common for social
    media. Designing for write QPS alone will underestimate infrastructure by 100x.

    ```
    WRONG:  "Twitter has 5,800 tweets/second, so we need servers for 5,800 QPS"
    RIGHT:  "Twitter has 5,800 writes/second + 580,000 reads/second"
    ```

    ### Mistake 2: Ignoring Peak vs Average

    Systems must handle peak load, not average load. Provisioning for average means
    your system crashes during peak hours.

    | Service | Peak/Average Ratio | Peak Cause |
    |---------|-------------------|------------|
    | E-commerce | 10-50x | Black Friday, flash sales |
    | Social media | 2-5x | Major events, trending topics |
    | Streaming | 2-3x | Evening prime time, premieres |
    | Banking | 3-5x | Payroll days, market open |

    ### Mistake 3: Confusing Bits and Bytes

    Network speeds are measured in **bits** per second (Mbps, Gbps). Storage is
    measured in **bytes** (MB, GB). 1 byte = 8 bits.

    ```
    A 1 Gbps network link transfers: 1,000,000,000 bits/sec
                                    = 125,000,000 bytes/sec
                                    = 125 MB/sec
                                    ≠ 1 GB/sec (common mistake!)
    ```

    ### Mistake 4: Linear Scaling Assumption

    Doubling servers does not double capacity. Coordination overhead, network
    contention, and shared resources mean you often get 60-80% of theoretical
    linear scaling.

    ```
    Expected:  10 servers at 1,000 QPS each = 10,000 QPS total
    Reality:   10 servers at 1,000 QPS each ≈ 7,000-8,000 QPS total
    ```

    ### Mistake 5: Forgetting Replication and Redundancy

    Production systems replicate data (typically 3x) and need capacity headroom
    for failover. Your raw storage estimate must be multiplied accordingly.

    !!! warning "The Replication Tax"
        If you estimate 100 TB of raw storage:

        - 3x replication: 300 TB
        - Backups (1 copy): +100 TB
        - Filesystem overhead (~10%): +30 TB
        - Index overhead (~20%): +60 TB
        - **Actual provisioning: ~490 TB (nearly 5x raw estimate)**

    ### Mistake 6: Precision Theater

    Spending time computing exact numbers to three decimal places adds zero value.
    The assumptions underlying your estimate have 2-5x error bars. Round aggressively.

    ```
    WRONG:  "We need 237.4 GB of cache"
    RIGHT:  "We need roughly 250 GB of cache, so a 256 GB instance works"

    WRONG:  "QPS will be 11,574.07"
    RIGHT:  "QPS will be roughly 12K, call it ~15K with headroom"
    ```

    ---

    ## Tips for Better Estimates

    !!! tip "Round to Powers of 10"
        Use 10^3 = 1K, 10^6 = 1M, 10^9 = 1B. This makes mental multiplication
        trivial and keeps you focused on order of magnitude rather than false precision.

    !!! tip "State Your Assumptions Explicitly"
        In an interview or design review, say "I'm assuming 40% DAU ratio" out loud.
        This shows structured thinking and lets others correct bad assumptions early.

    !!! tip "Work Top-Down, Then Bottom-Up"
        Start from total users and work down to QPS (top-down). Then estimate per-request
        resource usage and work up to total infrastructure (bottom-up). If the two
        approaches converge, your estimate is likely reasonable.

    !!! tip "Use Known Systems as Anchors"
        Compare your estimates to known real-world systems. If your chat app estimate
        says it needs more bandwidth than YouTube, something is wrong.

=== "Real-World Scale"

    ## Real Company Numbers

    These are approximate public numbers useful as reference points and sanity checks.
    They change over time but the order of magnitude remains instructive.

    ### User Scale

    | Company | Total Users | DAU | Peak Events |
    |---------|------------|-----|-------------|
    | Facebook | ~3B | ~2B | Super Bowl, elections |
    | YouTube | ~2.5B monthly | ~122M (US) | Music premieres, live events |
    | Twitter/X | ~500M | ~250M | World Cup, breaking news |
    | WhatsApp | ~2B | ~500M | New Year's Eve |
    | Instagram | ~2B | ~500M | Celebrity posts, events |
    | Netflix | ~250M subscribers | ~100M concurrent (peak) | New season drops |

    ### Throughput Scale

    | Company | Metric | Scale |
    |---------|--------|-------|
    | Twitter/X | Tweets per day | ~500 million |
    | Twitter/X | Timeline views per day | ~200 billion |
    | Google Search | Queries per day | ~8.5 billion (~100K QPS) |
    | YouTube | Hours uploaded per minute | ~500 hours |
    | YouTube | Hours watched per day | ~1 billion |
    | WhatsApp | Messages per day | ~100 billion |
    | Instagram | Photos uploaded per day | ~100 million |
    | Netflix | Hours streamed per day | ~400 million |
    | Uber | Trips per day | ~25 million |
    | Stripe | API requests per day | ~500 million |

    ### Infrastructure Scale

    | Company | Metric | Scale |
    |---------|--------|-------|
    | Google | Servers worldwide | ~4 million+ |
    | Google | Data stored | ~15 exabytes |
    | Facebook | Cache pool (Memcached) | ~100 TB+ |
    | Facebook | Photos stored | ~4 billion photos (at peak era) |
    | Netflix | CDN traffic | ~15% of global internet bandwidth |
    | Amazon | Peak orders/second (Prime Day) | ~100,000+ |
    | Cloudflare | Requests per second (global) | ~50 million |

    ### Scale Tiers

    When designing systems, it helps to categorize where your system falls.

    ```
    Tier 1: Small Scale          Tier 2: Medium Scale
    ──────────────────          ────────────────────
    Users:    < 100K            Users:    100K - 10M
    QPS:      < 100             QPS:      100 - 10K
    Storage:  < 1 TB            Storage:  1 TB - 100 TB
    Servers:  1-5               Servers:  10-100
    Strategy: Single DB,        Strategy: Read replicas,
              vertical scale              caching, CDN

    Tier 3: Large Scale          Tier 4: Hyperscale
    ───────────────────          ──────────────────
    Users:    10M - 1B           Users:    > 1B
    QPS:      10K - 1M           QPS:      > 1M
    Storage:  100 TB - 10 PB     Storage:  > 10 PB
    Servers:  100 - 10,000       Servers:  > 10,000
    Strategy: Sharding,          Strategy: Custom infra,
              microservices,               global distribution,
              multi-region                 own CDN/hardware
    ```

=== "Quick Reference"

    ## Estimation Cheat Sheet

    Cut out and keep -- the essential numbers for fast estimation.

    ### Time Conversions

    | Period | Seconds | Rounded |
    |--------|---------|---------|
    | 1 minute | 60 | ~10^2 |
    | 1 hour | 3,600 | ~4 x 10^3 |
    | 1 day | 86,400 | ~10^5 |
    | 1 month | 2,592,000 | ~2.5 x 10^6 |
    | 1 year | 31,536,000 | ~3 x 10^7 |

    ### Size Conversions

    | From | To | Factor |
    |------|-----|--------|
    | 1 KB | bytes | 10^3 |
    | 1 MB | KB | 10^3 |
    | 1 GB | MB | 10^3 |
    | 1 TB | GB | 10^3 |
    | 1 PB | TB | 10^3 |
    | 1 byte | bits | 8 |
    | 1 Gbps | MB/s | 125 |

    ### Quick QPS Math

    ```
    1 million requests/day    =    ~12 QPS
    10 million requests/day   =    ~120 QPS
    100 million requests/day  =    ~1,200 QPS
    1 billion requests/day    =    ~12,000 QPS
    10 billion requests/day   =    ~120,000 QPS
    ```

    ### Quick Storage Math

    ```
    1 million rows x 1 KB each    =    1 GB
    1 million rows x 1 MB each    =    1 TB
    1 billion rows x 1 KB each    =    1 TB
    1 billion rows x 1 MB each    =    1 PB
    ```

    ### Common Ratios to Remember

    | Ratio | Typical Value | Example |
    |-------|--------------|---------|
    | DAU / Total users | 20-40% | 1B users -> 200-400M DAU |
    | Peak / Average QPS | 2-5x | 10K avg -> 20-50K peak |
    | Read / Write ratio | 10:1 to 1000:1 | Social media: 100:1 |
    | Cache hit rate | 80-99% | Well-tuned: 95%+ |
    | Replication factor | 3x | 100 TB raw -> 300 TB replicated |
    | Index overhead | 10-30% | 100 TB data -> 10-30 TB indexes |
    | Compression ratio | 2-10x | Text compresses ~5x |

    ### The Estimation Template

    For any system, fill in this template:

    ```
    SYSTEM: _______________

    SCALE
    ─────
    Total users:           ___
    DAU:                   ___ (Total x __%)
    Actions per user/day:  ___
    Total actions/day:     ___

    QPS
    ───
    Average write QPS:     ___ (actions/day ÷ 86,400)
    Peak write QPS:        ___ (avg x ___x)
    Average read QPS:      ___ (write QPS x read:write ratio)
    Peak read QPS:         ___ (avg x ___x)

    STORAGE
    ───────
    Per-action data size:  ___ bytes
    Daily new data:        ___ (write QPS x 86,400 x size)
    Monthly:               ___ (daily x 30)
    Yearly:                ___ (daily x 365)
    With replication (3x): ___
    5-year projection:     ___

    BANDWIDTH
    ─────────
    Ingress:               ___ (write QPS x request size)
    Egress:                ___ (read QPS x response size)

    MEMORY (CACHE)
    ──────────────
    Daily data:            ___
    Cache (20% rule):      ___ (daily x 0.20)
    With replication:      ___

    INFRASTRUCTURE
    ──────────────
    Web servers:           ___ (peak QPS ÷ QPS per server)
    DB servers:            ___
    Cache servers:         ___ (cache size ÷ memory per node)
    Storage nodes:         ___ (total storage ÷ capacity per node)
    ```

---

## Related Topics

- **[Core Principles](principles.md)** -- scalability, reliability, availability, performance, and maintainability trade-offs
- **[Networking Fundamentals](networking-fundamentals.md)** -- latency numbers in depth, network behavior and failure modes
- **[Data Consistency](data-consistency.md)** -- consistency models that affect replication and storage estimates
- **[Scalability Patterns](../scalability/patterns.md)** -- how to implement the scaling decisions your estimates suggest
- **[Caching Strategies](../data/caching/strategies.md)** -- designing cache layers once you know your memory requirements
