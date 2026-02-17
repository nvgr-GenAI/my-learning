# Performance Fundamentals

Performance engineering is the discipline of understanding how systems behave under load and systematically improving their responsiveness. Unlike scalability, which asks "can we handle more?", performance asks "can we handle it faster?" The distinction matters because throwing hardware at a poorly written query will never make it fast -- but rewriting the query might make additional hardware unnecessary.

This guide covers how to measure performance accurately, where to look for bottlenecks, and how to think about optimization in the right order.

---

=== "Measuring Performance"

    ## Why Measurement Comes First

    The single most important principle in performance work is this: measure before you optimize. Engineers routinely spend days optimizing code paths that account for 2% of total latency while ignoring the database query responsible for 80% of it. Without measurement, you are guessing, and guesses in performance work are almost always wrong.

    ## Latency and Percentiles

    Latency is the time elapsed between sending a request and receiving a response. It is the metric users feel most directly -- when someone says an application "feels slow," they are describing latency.

    The critical mistake most teams make is tracking average latency. Averages are misleading because they hide the experience of your worst-served users. Consider an API serving 1,000 requests where 950 complete in 40ms and 50 take 2,000ms. The average is 138ms, which sounds acceptable, but 5% of your users are waiting two full seconds.

    Percentile metrics solve this problem by describing the distribution of response times:

    ```
    Percentile Latency Distribution
    ────────────────────────────────────────────────────
    P50 (median)    What the typical user experiences
    P95             The slowest 5% -- often your heaviest users
    P99             The slowest 1% -- reveals systemic issues
    P99.9           Extreme tail -- timeout and retry territory

    Example: E-commerce checkout API (10,000 requests)
    ────────────────────────────────────────────────────
    P50:    45ms    Most checkouts feel instant
    P95:   210ms    Some users notice a brief pause
    P99:   890ms    A few users see real delay
    P99.9: 3200ms   Occasional timeouts trigger retries
    ```

    Amazon found that every 100ms of additional latency cost them approximately 1% in sales. Google observed that a 500ms increase in search latency reduced traffic by 20%. These are not hypothetical numbers -- they drove both companies to optimize aggressively for P99 latency, not averages.

    The reason P95 and P99 matter disproportionately is that your heaviest users -- the ones with the most data, the most orders, the most complex queries -- are the ones most likely to hit tail latency. These are also your most valuable customers.

    ## Throughput

    Throughput measures how much work a system completes per unit of time. Depending on the system, this might be expressed as requests per second (RPS) for web servers, queries per second (QPS) for databases, or transactions per second (TPS) for payment systems.

    Latency and throughput have a well-known inverse relationship. At low load, latency stays flat because resources are plentiful. As throughput increases toward capacity, latency begins to climb as requests queue for resources. Past a saturation point, both metrics degrade -- throughput actually decreases as the system spends more time managing contention than doing useful work.

    ```
    Throughput vs Latency Relationship
    ────────────────────────────────────────────────────

    Latency
    (ms)
     500 |                                        ****
         |                                    ****
     300 |                                 ***
         |                              **
     100 |          ********************
         |  ********
      20 |**
         +──────────────────────────────────────────
          0    2k    4k    6k    8k    10k   12k
                    Throughput (RPS)
                         ^              ^
                     Sweet spot     Saturation
                    (70-80% util)   (degrading)
    ```

    The sweet spot for most systems is 70-80% resource utilization. This leaves enough headroom to absorb traffic spikes, run garbage collection, and recover from individual server failures without cascading.

    ## Utilization and Saturation

    Resource utilization measures what percentage of a resource's capacity is currently in use. The four resources to monitor are CPU, memory, disk I/O, and network bandwidth. When any one of these approaches saturation (roughly 90%+), queueing theory predicts that wait times will increase nonlinearly.

    At 70% utilization, a resource handles requests with minimal queueing delay. At 90%, queue times roughly double. At 95%, they quadruple. At 99%, the queue effectively explodes and the system becomes unresponsive. This is why Netflix targets 60-70% CPU utilization across their fleet -- it provides enough headroom to absorb a full availability zone failure without customer impact.

    ## SLIs, SLOs, and How to Set Them

    A Service Level Indicator (SLI) is a quantitative measure of some aspect of service quality -- for example, "the proportion of requests completing in under 200ms." A Service Level Objective (SLO) sets a target for that indicator -- for example, "99.9% of requests will complete in under 200ms, measured over a rolling 30-day window."

    Google's SRE practice recommends choosing SLIs that reflect user experience directly. For a web application, the most meaningful SLIs are typically request latency at P95 and P99, error rate as a percentage of total requests, and availability measured as successful requests divided by total requests.

    SLOs should be ambitious but achievable. Setting an SLO of 99.99% availability (52 minutes of downtime per year) when your infrastructure realistically supports 99.9% (8.7 hours per year) creates an error budget of zero, which means every deploy is a crisis.

    ## Observability Tools

    Prometheus and Grafana form the most widely adopted open-source monitoring stack. Prometheus scrapes metrics from your services at regular intervals and stores them as time series. Grafana provides dashboards for visualization and alerting.

    ```
    Typical Monitoring Pipeline
    ────────────────────────────────────────────────────

    [Service A] --metrics--> [Prometheus] --query--> [Grafana]
    [Service B] --metrics-->   (scrape      (dashboards,
    [Service C] --metrics-->    + store)      alerts)
         |
         v
    [Distributed Tracing]     [Log Aggregation]
     Jaeger / Zipkin           ELK / Loki

    What each layer answers:
      Metrics:  "Is something wrong?"
      Logs:     "What went wrong?"
      Traces:   "Where in the call chain did it go wrong?"
    ```

    Stripe monitors over 1,000 SLIs across their payment infrastructure. Each API endpoint has latency percentiles (P50, P95, P99), error rates, and throughput tracked in real time. When any metric breaches its SLO, on-call engineers are paged with a Grafana dashboard link showing exactly which service and endpoint degraded.

=== "Optimization Hierarchy"

    ## The Right Order to Optimize

    Not all optimizations are created equal. There is a natural hierarchy where improvements at higher levels yield dramatically larger gains than those at lower levels. Working in the wrong order is the most common performance engineering mistake.

    ```
    Optimization Impact Hierarchy
    ────────────────────────────────────────────────────

    Level 1: Algorithm + Data Structure     [100-10000x]
             Fix O(n^2) to O(n log n)
                      |
    Level 2: Architecture + Design           [10-100x]
             Caching, async, read replicas
                      |
    Level 3: Implementation                   [2-10x]
             Connection pooling, batching
                      |
    Level 4: Infrastructure                   [1.5-3x]
             Faster hardware, more memory
                      |
    Level 5: Tuning                           [1.1-1.5x]
             GC flags, kernel params
    ```

    Working from the bottom up is the classic premature optimization trap. Donald Knuth's famous quote -- "premature optimization is the root of all evil" -- is often misunderstood as "never optimize." The full quote is more nuanced: "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%." The point is not to avoid optimization but to profile first so you know which 3% matters.

    ## Level 1: Algorithm and Data Structure

    Choosing the right algorithm is the highest-leverage optimization available. No amount of caching or horizontal scaling will fix a fundamentally inefficient algorithm.

    When Slack rebuilt their channel membership system, they discovered that checking whether a user belonged to a channel was implemented as a linear scan through a list. For channels with 10,000 members, this single operation took hundreds of milliseconds. Replacing the list with a hash set reduced it to microseconds -- a 100,000x improvement that no infrastructure change could have achieved.

    ```python
    # Linear scan: O(n) -- 500ms for 1M users
    def find_user(users, target_id):
        for user in users:
            if user.id == target_id:
                return user

    # Hash lookup: O(1) -- 0.001ms for 1M users
    users_by_id = {u.id: u for u in users}
    user = users_by_id[target_id]
    ```

    ## Level 2: Architecture and Design

    Once algorithms are correct, architectural decisions drive the next largest gains. This level includes adding caching layers, moving work to asynchronous processing, introducing read replicas, and restructuring service boundaries.

    Twitter's shift from a pull-based to a push-based timeline (fan-out on write) is a canonical example. Instead of assembling a user's timeline from scratch on every request by querying all followed accounts, Twitter pre-computes timelines when tweets are posted. This moved the expensive work from the read path (millions of reads per second) to the write path (thousands of writes per second), reducing timeline latency from seconds to milliseconds.

    ## Level 3: Implementation

    Implementation-level optimizations include connection pooling, batch processing, response compression, and efficient serialization. These typically yield 2-10x improvements for specific operations.

    LinkedIn reduced their API latency by 60% by switching from JSON to Protocol Buffers for internal service communication. The serialization was faster, the payloads were smaller, and the schema enforcement caught bugs earlier. But this optimization only made sense after they had already fixed their algorithmic and architectural bottlenecks.

    ## Level 4: Infrastructure

    Faster CPUs, more memory, SSDs instead of spinning disks, and upgraded network links fall into this category. These are legitimate optimizations but offer diminishing returns -- you cannot buy your way out of an O(n^2) algorithm.

    Shopify found that upgrading their database instances from general-purpose to memory-optimized nodes reduced P99 query latency by 40%. This was worthwhile, but only after they had already added appropriate indexes and eliminated N+1 queries. Without those prior fixes, the hardware upgrade would have been barely noticeable.

    ## Profile Before You Optimize

    Every optimization effort should start with profiling. The workflow is straightforward: instrument your system, generate a representative workload, identify the hottest code paths, and focus exclusively on those.

    ```
    Profiling Workflow
    ────────────────────────────────────────────────────

    1. INSTRUMENT
       Add metrics to your services
       (request duration, DB query time, cache hits)
              |
    2. REPRODUCE
       Generate realistic load
       (load testing with production-like data)
              |
    3. IDENTIFY
       Find the bottleneck
       "85% of request time is in DB query X"
              |
    4. FIX
       Optimize the bottleneck only
       (add index, fix query, add cache)
              |
    5. VALIDATE
       Measure again to confirm improvement
       "P95 dropped from 800ms to 120ms"
              |
       (repeat from step 3)
    ```

    Datadog's internal engineering team has a rule: no performance PR is merged without before-and-after profiling data. This prevents well-intentioned optimizations that either do not help or accidentally make things worse due to unexpected interactions.

=== "Common Bottlenecks"

    ## Database Bottlenecks

    The database is the most common performance bottleneck in web applications. Three patterns account for the majority of database performance issues.

    **N+1 Query Problem.** This occurs when code fetches a list of records, then issues a separate query for each record's related data. ORMs with lazy loading make this pattern deceptively easy to introduce. A page displaying 100 orders with their associated customers generates 101 queries -- one for the orders list and one per order for customer details. Replacing this with a single JOIN or eager-loaded query reduces it to one round trip.

    Shopify identified N+1 queries as their single largest source of performance issues. They built automated detection into their CI pipeline: any test that generates more than a configurable number of queries for a single request triggers a warning.

    **Missing Indexes.** Without an index, a database must scan every row in a table to find matches (a "full table scan"). On a users table with 10 million rows, looking up a user by email without an index takes seconds. Adding a B-tree index on the email column reduces it to milliseconds, because the database can navigate directly to the matching rows.

    ```sql
    -- Without index: full table scan, ~5 seconds on 10M rows
    SELECT * FROM users WHERE email = 'jane@example.com';

    -- After adding index: B-tree lookup, ~5ms
    CREATE INDEX idx_users_email ON users(email);
    ```

    Pinterest's database team found that 60% of their slow query alerts were caused by missing or incorrect indexes. They now require index analysis as part of every schema migration review.

    **Fetching Too Much Data.** Using SELECT * when you only need three columns transfers 10x more data than necessary across the network and through the application's memory. This is especially costly for tables with large TEXT or BLOB columns.

    ## Network Bottlenecks

    **Chatty Services.** Microservices architectures can introduce excessive network round trips. If rendering a single page requires sequential calls to five different services, each adding 20ms of network latency, the minimum response time is 100ms before any actual processing. The fix is to batch calls, parallelize independent requests, or introduce an aggregation layer (the Backend-for-Frontend pattern).

    Airbnb discovered that their search results page was making 15 sequential service calls. By parallelizing independent calls and batching related ones, they reduced the network overhead from 300ms to 60ms.

    **No Connection Pooling.** Establishing a new TCP connection (and especially a TLS handshake) for every request adds 50-100ms of overhead. Connection pools maintain a set of pre-established connections that requests can borrow and return, amortizing the setup cost across thousands of requests.

    ## Memory Bottlenecks

    **Garbage Collection Pressure.** In languages with automatic memory management (Java, Go, C#), allocating and discarding objects rapidly forces the garbage collector to run more frequently. GC pauses manifest as latency spikes -- the application literally stops processing requests while it collects garbage. Discord moved their Read States service from Go to Rust partly because Go's garbage collector was causing latency spikes every two minutes as it scanned millions of objects in memory.

    **Memory Leaks.** A slow memory leak may not crash the application for days, but as available memory shrinks, the OS begins swapping to disk, and performance degrades dramatically. Monitoring memory usage over time (not just instantaneously) catches leaks before they cause outages.

    ## CPU Bottlenecks

    **Inefficient Serialization.** JSON parsing and generation is surprisingly CPU-intensive at scale. Uber found that JSON serialization accounted for 30% of CPU usage in some of their highest-throughput services. Switching hot paths to Protocol Buffers or FlatBuffers reduced CPU consumption significantly.

    **Unoptimized Regular Expressions.** A regex with catastrophic backtracking can pin a CPU core at 100% for a single request. Cloudflare experienced a global outage in 2019 caused by a single regular expression that consumed excessive CPU across their edge network.

    ## Profiling Workflow for Bottleneck Identification

    ```
    Systematic Bottleneck Investigation
    ────────────────────────────────────────────────────

    Start: "API response time is 2 seconds"
        |
        v
    [Check metrics dashboard]
        |
        +-- CPU > 85%? -----> Profile CPU (flame graph)
        |                      Look for hot functions,
        |                      tight loops, serialization
        |
        +-- Memory > 90%? --> Check GC metrics, heap dumps
        |                      Look for leaks, large objects
        |
        +-- DB query > 1s? -> Run EXPLAIN on slow queries
        |                      Check for full table scans,
        |                      missing indexes, N+1 patterns
        |
        +-- Network calls --> Trace request through services
           taking > 100ms?    Check for sequential calls,
                              missing connection pools,
                              large payloads
        |
        v
    Fix the single largest bottleneck first.
    Measure again. Repeat.
    ```

    The key discipline is fixing one bottleneck at a time and re-measuring after each fix. Performance work is iterative because fixing one bottleneck often reveals the next one. What was previously 5% of request time might become 40% once the larger issue is resolved.

=== "Scaling for Performance"

    ## When Optimization Is Not Enough

    There comes a point where a single optimized server cannot handle the required throughput. At that point, performance engineering intersects with scaling strategy. The goal shifts from "make each request faster" to "serve more requests concurrently without making any single request slower."

    ## Horizontal Scaling

    Adding more servers behind a load balancer is the most straightforward scaling approach. Each server handles a fraction of the total traffic, and total throughput scales roughly linearly with server count -- provided the application is stateless and there are no shared bottlenecks.

    Netflix runs over 10,000 EC2 instances across multiple AWS regions. Their stateless microservices architecture means they can scale any service independently based on its specific traffic patterns. During peak evening hours in the US, they automatically scale up streaming services while keeping metadata services at baseline levels.

    The prerequisite for effective horizontal scaling is statelessness. If a server stores session data locally, requests from the same user must always route to the same server (sticky sessions), which limits load distribution and complicates failure recovery. Moving session state to Redis or a similar external store eliminates this constraint.

    ## Read Replicas

    Most web applications are read-heavy -- typically 90-95% reads versus 5-10% writes. Read replicas allow you to distribute read queries across multiple database copies while directing all writes to a single primary. This multiplies read throughput without the complexity of full sharding.

    GitHub uses MySQL read replicas extensively. Their primary database handles writes, while reads are distributed across multiple replicas. During high-traffic events like Hacktoberfest, they scale up read replicas to handle the surge in repository browsing without affecting write performance.

    The trade-off is replication lag. A write to the primary may take 10-100ms to propagate to replicas, meaning a user who updates their profile might see stale data if their next read hits a replica. The standard mitigation is "read your own writes" -- routing a user's reads to the primary for a short window after they perform a write.

    ## Caching Layers

    Caching is the single most effective technique for improving read performance. A well-designed caching layer can reduce database load by 90% or more, because the same data is requested far more often than it changes.

    ```
    Multi-Layer Cache Architecture
    ────────────────────────────────────────────────────

    [Client]
       |
    [Browser Cache]         0ms    (static assets, TTL-based)
       |  miss
    [CDN Edge Cache]       10ms    (HTML, images, API responses)
       |  miss
    [Application Cache]     2ms    (Redis/Memcached, hot data)
       |  miss
    [Database]            50ms     (source of truth)

    With 95% cache hit rate at the Redis layer:
      Effective latency = 0.95 * 2ms + 0.05 * 50ms = 4.4ms
      vs 50ms without cache = 11x improvement
    ```

    Facebook (Meta) operates the largest Memcached deployment in the world, caching trillions of items across thousands of servers. Their cache hit rate exceeds 99% for many workloads, which means their database fleet handles only 1% of total read traffic. Without caching, they would need roughly 100x more database capacity.

    The 80/20 rule (Pareto principle) governs cache sizing: 20% of your data typically accounts for 80% of requests. Caching just the hot 20% is usually sufficient to achieve hit rates above 85%, keeping memory costs manageable.

    ## Asynchronous Processing

    Moving non-urgent work off the request path is one of the most effective ways to reduce user-facing latency. When a user uploads an image, they do not need to wait for thumbnail generation, virus scanning, and CDN propagation before seeing a confirmation.

    ```python
    # Synchronous: user waits for everything (5 seconds)
    def upload_image(image):
        save_to_storage(image)       # 500ms
        generate_thumbnails(image)   # 2000ms
        scan_for_viruses(image)      # 1500ms
        update_database(image)       # 1000ms
        return "Done"                # total: 5000ms

    # Async: user waits only for the save (500ms)
    def upload_image(image):
        save_to_storage(image)       # 500ms
        queue.enqueue(process_image, image)
        return "Processing..."       # total: 500ms
    ```

    Instagram processes over 100 million photo uploads daily. The upload endpoint returns a success response within milliseconds after saving the raw image to storage. Everything else -- filter application, multiple resolution generation, story creation, notification delivery -- happens asynchronously through a task queue. This keeps the upload experience feeling instant even as the backend processing grows more complex.

    Message queues like Kafka, RabbitMQ, and SQS decouple producers from consumers, allowing each to scale independently. If the image processing pipeline falls behind, images simply queue up rather than making the upload endpoint slow down.

    ## CDN for Static Content

    A Content Delivery Network places copies of your static content (images, CSS, JavaScript, videos) on servers distributed around the world. When a user in Tokyo requests an image, it is served from a nearby edge server rather than traveling across the Pacific to your origin server in Virginia.

    ```
    Without CDN                    With CDN
    ──────────────────            ──────────────────
    Tokyo --> Virginia            Tokyo --> Tokyo Edge
    10,000 km                     50 km
    200ms round trip              10ms round trip
    Origin serves all traffic     Edge serves 95% of traffic
    ```

    Netflix serves over 100 petabytes of video content daily through their Open Connect CDN. By placing custom cache appliances directly inside ISP networks, they deliver video with minimal latency and virtually no impact on internet backbone traffic. For their scale, this architecture reduced their bandwidth costs by orders of magnitude compared to serving everything from centralized data centers.

    ## Instagram's Performance Journey

    Instagram's evolution illustrates how these techniques compose. At launch in 2010, Instagram ran on a single Django server with a PostgreSQL database. As they grew to millions of users, they systematically applied each scaling technique.

    First, they added Redis caching for the feed, reducing database reads by 90%. Next, they introduced PostgreSQL read replicas to distribute the remaining read load. They moved photo processing to asynchronous Celery workers so uploads stayed fast. They adopted a CDN (initially Amazon CloudFront) for serving photos globally. Finally, they horizontally scaled their application servers behind a load balancer.

    By the time Facebook acquired Instagram in 2012, these optimizations allowed a team of 13 engineers to serve 30 million users. The performance principles were sound -- each optimization addressed the actual bottleneck at that stage of growth rather than prematurely scaling everything at once.

---

## Key Takeaways

**Measure, then optimize.** Profiling reveals where time is actually spent. Without data, you are guessing, and performance guesses are almost always wrong.

**Optimize in the right order.** Algorithm improvements (100-10,000x) dwarf infrastructure upgrades (1.5-3x). Fix the query before you upgrade the server.

**Track percentiles, not averages.** P95 and P99 latency reveal what your worst-served users experience. Your most valuable customers are often in that tail.

**Target 70-80% utilization.** Operating at 90%+ leaves no headroom for traffic spikes, garbage collection, or failure recovery, and queueing theory guarantees latency will spike.

**Cache the hot 20%.** The Pareto principle means caching a small fraction of your data serves the vast majority of requests. A 95% cache hit rate can reduce effective latency by 10x or more.

**Move heavy work off the request path.** Users should not wait for thumbnail generation, email delivery, or analytics processing. Queue it and return immediately.

---

## Related Topics

| Topic | Connection |
| ----- | ---------- |
| [Caching Strategies](../data/caching/strategies.md) | Deep dive into cache patterns, eviction, and invalidation |
| [Database Indexing](../data/databases/indexing.md) | How B-tree and hash indexes accelerate queries |
| [Load Balancing](../networking/load-balancers.md) | Distributing traffic across horizontally scaled servers |
| [Messaging Patterns](../communication/messaging/patterns.md) | Async processing infrastructure for decoupling services |
