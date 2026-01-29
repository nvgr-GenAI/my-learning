# The 4-Step Interview Framework

A proven approach to system design interviews. Use this framework for any design problem to structure your thinking and communicate effectively.

---

## Framework Overview

| Step | Time | Goal |
|------|------|------|
| **1. Clarify Requirements** | 5-10 min | Define what you're building and constraints |
| **2. High-Level Design** | 10-15 min | Draw main components and data flow |
| **3. Deep Dive** | 20-30 min | Detailed design decisions and trade-offs |
| **4. Wrap Up** | 5-10 min | Discuss scalability, monitoring, and issues |

**Total:** 45-60 minutes (typical interview length)

---

## The 4 Steps

=== "Step 1: Clarify Requirements"

    ## 📋 Step 1: Clarify Requirements (5-10 min)

    **Never skip this step!** Jumping straight to design is the #1 interview mistake.

    ---

    === "Overview & Goal"

        ### 🎯 Goal

        Define exactly what you're building and under what constraints. Get alignment with interviewer before writing any code or drawing diagrams.

        ### Why This Matters

        - **Prevents building the wrong system** - Assumptions are dangerous
        - **Shows business understanding** - You think about user needs
        - **Provides constraints to work with** - Helps make technical decisions
        - **Demonstrates communication** - Critical for senior roles

        ### What You'll Learn

        By the end of this step, you should know:
        - ✅ Core features (functional requirements)
        - ✅ Scale expectations (users, QPS, data size)
        - ✅ Performance targets (latency, throughput)
        - ✅ Reliability needs (availability, consistency)
        - ✅ What's in/out of scope

    === "Functional Requirements"

        ### Ask: "What features does the system need?"

        **Template questions:**

        | Question | Why Ask This |
        |----------|--------------|
        | What are the core features? | Prioritize what's essential |
        | Who are the users? | Understand the audience |
        | What actions can they perform? | Define capabilities |
        | What's in scope vs out of scope? | Avoid scope creep |
        | Any special requirements? | Edge cases, compliance |

        **Example dialogue for URL Shortener:**

        ```
        You: What should this system do?
        Interviewer: Allow users to shorten URLs and redirect.

        You: Should users be able to customize their short URLs?
        Interviewer: Optional feature, nice to have.

        You: Do we need analytics like click tracking?
        Interviewer: Yes, basic click counts.

        You: How long should URLs remain active?
        Interviewer: 5 years minimum.

        You: What about authentication? Can anyone create URLs?
        Interviewer: Yes, public access is fine.
        ```

        **Document your understanding:**

        ```
        ✓ Core Features (Must-Have):
          - Generate short URL from long URL
          - Redirect short URL to original URL
          - Track click counts

        ? Optional Features (Nice-to-Have):
          - Custom URL aliases
          - Expiration dates
          - User accounts

        ✗ Out of Scope:
          - QR code generation
          - Link preview generation
          - Spam detection (assume handled elsewhere)
        ```

    === "Non-Functional Requirements"

        ### Ask: "What scale and performance do we need?"

        **Critical questions:**

        | Aspect | Questions to Ask | Why It Matters |
        |--------|------------------|----------------|
        | **Scale** | How many users? DAU/MAU? QPS? | Determines architecture complexity |
        | **Latency** | Response time expectations? | Affects caching strategy |
        | **Availability** | Uptime requirements? (99.9% vs 99.99%) | Impacts redundancy needs |
        | **Consistency** | Strong or eventual? | Database choice |
        | **Durability** | Can we lose data? | Backup strategy |
        | **Growth** | Expected growth rate? | Plan for scaling |

        **Example for URL Shortener:**

        ```
        Scale:
        - 100M new URLs per month
        - 10B redirects per month
        - 100:1 read/write ratio

        Performance:
        - Redirect latency: < 100ms (p99)
        - Generation: < 500ms (less critical)

        Reliability:
        - 99.9% availability (acceptable downtime: 43 min/month)
        - No data loss tolerance
        - Eventual consistency acceptable for click counts

        Storage:
        - 5 year retention minimum
        - Estimated 6B total URLs
        ```

    === "Calculations"

        ### Quick Math to Validate Scale

        **Always calculate these:**

        ```
        QPS (Queries Per Second)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Write QPS = 100M URLs/month ÷ (30 days × 24h × 3600s)
                  = 100,000,000 ÷ 2,592,000
                  = ~40 writes/sec

        Read QPS = 10B redirects/month ÷ (30 × 24 × 3600)
                 = 10,000,000,000 ÷ 2,592,000
                 = ~4,000 reads/sec

        Peak QPS = 2-3x average
                 = 80-120 writes/sec, 8K-12K reads/sec
        ```

        ```
        Storage Requirements
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total URLs = 100M/month × 12 months × 5 years
                   = 6 billion URLs

        Size per URL object:
        - short_url: 7 bytes
        - long_url: 200 bytes (average)
        - metadata: 100 bytes (user_id, timestamps, clicks)
        - overhead: ~200 bytes
        Total: ~500 bytes per URL

        Total Storage = 6B × 500 bytes
                      = 3TB (text data)
        ```

        ```
        Bandwidth Requirements
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Read bandwidth = 4K QPS × 1KB response
                       = 4 MB/sec = 32 Mbps

        Write bandwidth = 40 QPS × 2KB request
                        = 80 KB/sec = 640 Kbps

        Peak bandwidth = 3x average = ~100 Mbps
        ```

        **Pro tip:** Round numbers and show your work. Interviewers care about the process, not exact precision.

    === "Requirements Checklist"

        ### ✅ Before Moving to Step 2

        Use this checklist to ensure you've covered everything:

        **Functional:**
        
        - [ ] Core features identified and prioritized
        - [ ] Users and use cases understood
        - [ ] In-scope vs out-of-scope defined
        - [ ] Edge cases discussed

        **Scale:**
        - [ ] Number of users (DAU/MAU)
        - [ ] Expected traffic (QPS)
        - [ ] Data volume (storage needs)
        - [ ] Growth projections

        **Performance:**
        - [ ] Latency requirements (p50, p95, p99)
        - [ ] Throughput expectations
        - [ ] Peak load scenarios

        **Reliability:**
        - [ ] Availability requirements (SLA %)
        - [ ] Consistency needs (strong vs eventual)
        - [ ] Data durability requirements
        - [ ] Disaster recovery expectations

        **When ready:** Write requirements on the whiteboard and confirm with interviewer before proceeding.

=== "Step 2: High-Level Design"

    ## 🏗️ Step 2: High-Level Design (10-15 min)

    **Goal:** Draw the main components and explain data flow. Keep it simple - details come in Step 3.

    ---

    === "Overview & Goal"

        ### 🎯 Goal

        Create a high-level architecture diagram showing major components and how they interact. This gives the interviewer a "30,000 foot view" of your solution.

        ### What to Include

        **Core Components:**
        - Client (web/mobile)
        - Load Balancer
        - Application Servers (API layer)
        - Databases (primary/replicas)
        - Cache layer (Redis/Memcached)
        - CDN (if serving static content)
        - Message queues (if async processing)

        ### What NOT to Include

        - Implementation details (code, algorithms)
        - Database schemas (save for Step 3)
        - Specific configurations
        - Deployment details

        **Keep it at the box-and-arrow level!**

    === "Architecture Diagram"

        ### URL Shortener High-Level Design

        ```mermaid
        graph TD
            Client[Clients<br/>Web/Mobile] --> LB[Load Balancer]
            LB --> API1[API Server 1]
            LB --> API2[API Server 2]
            LB --> API3[API Server 3]
            API1 --> Cache[(Redis<br/>Cache)]
            API2 --> Cache
            API3 --> Cache
            API1 --> DB[(Database<br/>Primary)]
            API2 --> DB
            API3 --> DB
            DB --> Replica[(Database<br/>Replicas)]

            style Client fill:#e1f5ff
            style LB fill:#fff4e1
            style API1 fill:#e8f5e9
            style API2 fill:#e8f5e9
            style API3 fill:#e8f5e9
            style Cache fill:#ffe1e1
            style DB fill:#f3e5f5
            style Replica fill:#f3e5f5
        ```

        **Component count for 4K read QPS:**
        - Load Balancer: 1 (HAProxy/nginx)
        - API Servers: 3-5 (1K QPS each)
        - Redis Cache: 1 instance (handles 100K+ QPS)
        - Database: 1 primary + 3 replicas

    === "Data Flow"

        ### Write Flow (Generate Short URL)

        ```mermaid
        sequenceDiagram
            participant Client
            participant LB as Load Balancer
            participant API as API Server
            participant DB as Database
            participant Cache as Redis Cache

            Client->>LB: POST /shorten {long_url}
            LB->>API: Route request
            Note over API: Generate short URL<br/>(base62 encoding)
            API->>DB: INSERT url_mapping
            DB-->>API: Success
            API->>Cache: SET short_code → long_url
            Cache-->>API: Cached
            API-->>Client: 200 OK {short_url}
        ```

        **Latency breakdown:**
        - Network: 10-20ms
        - API processing: 5-10ms
        - Database write: 10-20ms
        - Cache write: 1-2ms
        - **Total: ~50ms (well under 500ms target)**

        ---

        ### Read Flow (Redirect)

        ```mermaid
        sequenceDiagram
            participant Client
            participant LB as Load Balancer
            participant API as API Server
            participant Cache as Redis Cache
            participant DB as Database Replica
            participant Queue as Kafka Queue

            Client->>LB: GET /{short_code}
            LB->>API: Route request
            API->>Cache: GET short_code

            alt Cache HIT (90% of requests)
                Cache-->>API: long_url (1-2ms)
            else Cache MISS (10% of requests)
                Cache-->>API: null
                API->>DB: SELECT long_url
                DB-->>API: long_url (20-30ms)
                API->>Cache: SET short_code → long_url
            end

            API-->>Client: 302 Redirect to long_url
            Note over API,Queue: Async (non-blocking)
            API->>Queue: Increment click counter
        ```

        **Latency for cache hit:** 1-2ms (ideal)
        **Latency for cache miss:** 20-30ms (still good)

    === "Component Decisions"

        ### Why These Technologies?

        | Component | Choice | Reasoning |
        |-----------|--------|-----------|
        | **Cache** | Redis | In-memory = microsecond latency. 100K+ QPS per instance. Handles 90%+ of reads. Built-in TTL support |
        | **Database** | PostgreSQL | ACID guarantees for URL mappings. Strong consistency required. Good at 40 writes/sec. Mature replication |
        | **Load Balancer** | HAProxy/nginx | Distributes 8K QPS. Health checking. SSL termination. Battle-tested |
        | **API Servers** | Node.js/Python | Stateless (easy to scale). Fast I/O for cache/DB. Simple logic |

        ### Key Decisions & Trade-offs

        **Decision 1: Cache-aside pattern**
        - ✅ Simple to implement
        - ✅ Cache only popular URLs (80/20 rule)
        - ❌ Cache miss requires DB query
        - **Why:** Optimizes for read-heavy workload

        **Decision 2: Read replicas**
        - ✅ Horizontal read scaling
        - ✅ Isolate reads from writes
        - ❌ Slight replication lag
        - **Why:** 100:1 read/write ratio benefits from this

        **Decision 3: Async click counter**
        - ✅ Doesn't block redirect response
        - ✅ Can batch updates
        - ❌ Eventual consistency for counts
        - **Why:** Latency is more important than exact counts

    === "Pro Tips"

        ### ✅ Do This

        - **Draw the diagram yourself** - Use whiteboard, don't just describe
        - **Label everything** - Component names, data flows, counts
        - **Start simple** - Can always add complexity later
        - **Explain data flow** - Walk through read and write paths
        - **Mention why** - Justify every component choice
        - **Ask for feedback** - "Does this approach make sense?"

        ### ❌ Avoid This

        - **Jumping to details** - No code, schemas, or algorithms yet
        - **Over-complicating** - Don't add unnecessary components
        - **Forgetting scale** - Mention expected QPS for each component
        - **Silent drawing** - Talk while you draw
        - **Defensive** - Be open to suggestions

        ### Time Management

        - Diagram: 5 minutes
        - Explain flows: 3 minutes
        - Discuss choices: 3 minutes
        - Q&A: 2 minutes
        - **Total: 10-15 minutes**

=== "Step 3: Deep Dive"

    ## 🔬 Step 3: Deep Dive (20-30 min)

    **The interviewer will pick 2-3 areas to explore in depth.** Be ready to discuss any component.

    ---

    === "Overview & Goal"

        ### 🎯 Goal

        Provide detailed design for specific components. Show depth of knowledge and ability to handle complexity.

        ### Common Deep Dive Topics

        1. **Database Design** - Schema, indexes, partitioning
        2. **API Design** - Endpoints, request/response formats, errors
        3. **Algorithms** - URL generation, ranking, matching
        4. **Scaling** - How to handle 10x, 100x traffic
        5. **Failures** - What if X goes down? How to recover?

        ### How to Handle

        - **Listen for hints** - Interviewer will guide you
        - **Go deep, not wide** - Focus on what they ask
        - **Show multiple options** - Discuss trade-offs
        - **Think out loud** - Explain your reasoning
        - **Draw when helpful** - Diagrams for complex flows

    === "Database Design"

        ### Schema Design for URL Shortener

        ```sql
        Table: url_mappings
        ┌────────────┬────────────┬─────────────────────────┐
        │ Column     │ Type       │ Notes                   │
        ├────────────┼────────────┼─────────────────────────┤
        │ id         │ BIGINT     │ Primary key, auto-inc   │
        │ short_code │ VARCHAR(7) │ Unique, indexed         │
        │ long_url   │ TEXT       │ Original URL (2KB max)  │
        │ user_id    │ BIGINT     │ Nullable, indexed       │
        │ created_at │ TIMESTAMP  │ For analytics/cleanup   │
        │ expires_at │ TIMESTAMP  │ Nullable, indexed       │
        │ clicks     │ INTEGER    │ Default 0, updated async│
        └────────────┴────────────┴─────────────────────────┘

        Indexes:
        - PRIMARY KEY (id)
        - UNIQUE INDEX idx_short_code (short_code)  -- O(log n) lookups
        - INDEX idx_user_id (user_id)               -- List user's URLs
        - INDEX idx_expires_at (expires_at)         -- Cleanup job
        ```

        **Why these choices?**

        | Decision | Reasoning |
        |----------|-----------|
        | `short_code` VARCHAR(7) | 62^7 = 3.5T combinations, enough for scale |
        | `long_url` TEXT | Some URLs are very long (2KB+) |
        | `user_id` nullable | Support anonymous URL creation |
        | `clicks` in same table | Simple, denormalized for speed |
        | Separate `expires_at` index | Fast cleanup queries |

        ### Partitioning Strategy (if needed at scale)

        ```
        Shard by: hash(short_code) % num_shards

        10 shards:
        - Shard 0: short_codes with hash % 10 = 0
        - Shard 1: short_codes with hash % 10 = 1
        - ...
        - Shard 9: short_codes with hash % 10 = 9

        Each shard handles:
        - 600M URLs (6B total ÷ 10)
        - 400 reads/sec (4K total ÷ 10)
        - 4 writes/sec (40 total ÷ 10)
        ```

    === "API Design"

        ### RESTful API Endpoints

        ```
        POST /api/v1/shorten
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Create a short URL

        Request:
        {
          "long_url": "https://example.com/very/long/url",
          "custom_alias": "mylink",      // optional
          "expires_in": 86400            // optional, seconds
        }

        Response 200 OK:
        {
          "short_url": "https://short.ly/abc123",
          "long_url": "https://example.com/very/long/url",
          "short_code": "abc123",
          "created_at": "2024-01-29T10:00:00Z",
          "expires_at": "2024-01-30T10:00:00Z"
        }

        Errors:
        400 Bad Request - Invalid URL format
        409 Conflict - Custom alias already taken
        429 Too Many Requests - Rate limit exceeded
        500 Internal Server Error - Server error
        ```

        ```
        GET /{short_code}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Redirect to original URL

        Response:
        302 Found
        Location: https://example.com/very/long/url

        Errors:
        404 Not Found - Short code doesn't exist
        410 Gone - URL has expired
        ```

        ```
        GET /api/v1/analytics/{short_code}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Get click statistics

        Response 200 OK:
        {
          "short_code": "abc123",
          "total_clicks": 1523,
          "created_at": "2024-01-29T10:00:00Z",
          "last_accessed": "2024-01-29T15:30:00Z"
        }
        ```

        ### Rate Limiting

        ```
        Per user (authenticated):
        - 100 URLs per hour
        - 1,000 URLs per day

        Per IP (anonymous):
        - 10 URLs per hour
        - 50 URLs per day

        Global:
        - 10K URL creations per minute
        ```

    === "URL Generation"

        ### Three Algorithm Options

        === "Base62 Encoding"

            **Algorithm:**
            ```python
            def generate_short_code(id):
                """Convert auto-increment ID to base62"""
                chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                result = []

                while id > 0:
                    result.append(chars[id % 62])
                    id //= 62

                return ''.join(reversed(result)).rjust(7, '0')

            # Example:
            # ID 1        → "0000001"
            # ID 62       → "0000010"
            # ID 12345    → "000003D7"
            # ID 3521614  → "000aBcD"
            ```

            **Pros:**
            - ✅ Simple and deterministic
            - ✅ No collisions (ID is unique)
            - ✅ Reversible (decode to get ID)
            - ✅ Fast (O(log n) time)

            **Cons:**
            - ❌ Sequential (predictable pattern)
            - ❌ Exposes total count
            - ❌ Not truly random

            **Best for:** Simple systems, okay with sequential IDs

        === "MD5 Hash"

            **Algorithm:**
            ```python
            import hashlib

            def generate_short_code(long_url):
                """Hash URL and take first 7 characters"""
                hash_value = hashlib.md5(long_url.encode()).hexdigest()
                short_code = hash_value[:7]

                # Handle collisions
                suffix = 0
                while exists_in_db(short_code):
                    suffix += 1
                    short_code = hash_value[:6] + str(suffix)

                return short_code

            # Example:
            # "https://example.com" → "8b1a9953"[:7] → "8b1a995"
            ```

            **Pros:**
            - ✅ Same long URL → same short URL (idempotent)
            - ✅ Non-sequential
            - ✅ Distributed generation (no coordination)

            **Cons:**
            - ❌ Need collision detection
            - ❌ Extra DB query to check existence
            - ❌ Low probability collision handling

            **Best for:** Distributed systems, idempotency important

        === "Random + Check"

            **Algorithm:**
            ```python
            import random, string

            def generate_short_code():
                """Generate random string"""
                chars = string.ascii_letters + string.digits

                while True:
                    short_code = ''.join(random.choices(chars, k=7))

                    if not exists_in_db(short_code):
                        return short_code

            # Example: "aB3xK9m" (completely random)
            ```

            **Pros:**
            - ✅ Truly random (unpredictable)
            - ✅ Good distribution

            **Cons:**
            - ❌ Need DB lookup every time
            - ❌ Collision probability increases with volume
            - ❌ Performance degrades as table fills

            **Best for:** Security-sensitive, small scale

        **Which to choose?**
        - For interviews: **Base62** (simplest to explain)
        - For production: Depends on requirements

    === "Scaling Strategies"

        ### How to Scale from 4K → 40K → 400K QPS

        **Stage 1: Current (4K reads/sec)**
        ```
        1 Load Balancer
        3 API Servers (handle 1-2K QPS each)
        1 Redis (handles 100K+ QPS easily)
        1 Primary DB + 3 Replicas
        ```

        **Stage 2: 10x Growth (40K reads/sec)**
        ```
        Caching Optimization:
        - Increase cache hit rate to 99%
        - TTL: 1 hour → 24 hours for popular URLs
        - Result: Only 400 reads/sec hit DB

        Add capacity:
        - 10 API Servers (4K QPS each)
        - Redis Cluster (5 nodes, 500K QPS total)
        - 1 Primary + 10 Replicas (each handles 40 QPS)
        ```

        **Stage 3: 100x Growth (400K reads/sec)**
        ```
        CDN Layer:
        - Add CloudFlare/Fastly in front
        - Cache redirects at edge (1-2ms latency globally)
        - 80% of traffic never hits our servers
        - Result: Only 80K reads/sec to our infrastructure

        Multi-Region:
        - 3 regions (US-East, US-West, EU)
        - Database sharding (10 shards per region)
        - Cross-region replication for writes

        Final architecture handles:
        - 400K reads/sec globally
        - 400 writes/sec
        - < 50ms p99 latency worldwide
        ```

    === "Failure Scenarios"

        ### "What If...?" Questions

        **Q: What if the primary database goes down?**

        ```
        Failure detection:
        - Health check every 1 second
        - Failover after 3 consecutive failures

        Recovery process:
        1. Stop accepting writes immediately
        2. Promote healthiest replica to primary (10-30 sec)
        3. Update API servers with new primary endpoint
        4. Resume write operations

        Data loss: None (replicas are caught up)
        Downtime: 30-60 seconds for writes only
        Reads: Continue working (hit replicas)
        ```

        **Q: What if Redis cache goes down?**

        ```
        Impact:
        - All reads go directly to database
        - Latency increases: 2ms → 20ms
        - Database load spikes: 200 QPS → 4K QPS

        Mitigation:
        1. Database can handle it (designed for 4K QPS)
        2. Auto-restart Redis (30-60 seconds)
        3. Cache warms up gradually
        4. No data loss (Redis is cache, not primary store)

        Result: Slight latency increase, no downtime
        ```

        **Q: What if one API server crashes?**

        ```
        Detection:
        - Load balancer health check (every 2 seconds)
        - Remove from pool after 2 failures

        Impact:
        - Traffic redistributes to healthy servers
        - Each server picks up extra 500 QPS
        - Still within capacity (1-2K QPS each)

        Recovery:
        - Auto-scaling launches new instance (2-3 min)
        - New instance passes health check
        - Added back to load balancer pool

        Result: No user impact, seamless failover
        ```

        **Q: What if the entire data center goes down?**

        ```
        Multi-region setup:
        - Primary region: US-East
        - Backup region: US-West (hot standby)

        Failover:
        1. DNS TTL: 60 seconds
        2. Update DNS to point to US-West
        3. US-West becomes primary
        4. Writes go to US-West database

        RTO (Recovery Time Objective): 2-3 minutes
        RPO (Recovery Point Objective): 0 (continuous replication)
        ```

=== "Step 4: Wrap Up"

    ## ✅ Step 4: Wrap Up (5-10 min)

    **Show you can think about production systems holistically.**

    ---

    === "Overview & Goal"

        ### 🎯 Goal

        Demonstrate you understand production concerns beyond just making it work. Show systems thinking.

        ### Topics to Cover

        1. **Bottlenecks** - What will fail first at scale?
        2. **Monitoring** - How do we know the system is healthy?
        3. **Cost** - How much does this cost to run?
        4. **Security** - What are the vulnerabilities?

        ### Time Management

        - Bottlenecks: 2 minutes
        - Monitoring: 2 minutes
        - Cost/Security: 2 minutes
        - Final Q&A: 2 minutes

    === "Bottlenecks"

        ### Identify Weak Points

        | Component | Bottleneck | Solution |
        |-----------|-----------|----------|
        | **Database** | Write capacity (40 writes/sec limit) | Vertical scaling (bigger machine). Sharding if > 100 writes/sec. Batch writes |
        | **Cache** | Single Redis instance (100K QPS max) | Redis Cluster (distributed). Multiple read replicas. Regional caches |
        | **API Servers** | CPU during peak (80%+ usage) | Horizontal scaling. Auto-scaling rules. Connection pooling |
        | **Network** | Bandwidth limits (100 Mbps) | CDN for static content. Response compression. Multiple data centers |

        ### When Will We Hit Limits?

        ```
        Current capacity: 4K reads/sec, 40 writes/sec

        Database bottleneck: ~100 writes/sec
        - Timeline: If growth is 10% monthly, ~6 months
        - Solution: Plan sharding now, implement at 70 writes/sec

        Cache bottleneck: ~80K reads/sec (80% hit rate)
        - Timeline: 20x growth = 2-3 years at 10% monthly growth
        - Solution: Redis Cluster when hit 50K reads/sec

        Network bottleneck: ~50 Mbps sustained
        - Timeline: Regional traffic spikes during events
        - Solution: CDN implementation (immediate)
        ```

    === "Monitoring"

        ### Key Metrics to Track

        **Application Metrics:**
        ```
        QPS (Queries Per Second):
        - Write QPS (target: 40, alert: > 70)
        - Read QPS (target: 4K, alert: > 6K)
        - Cache hit rate (target: 90%, alert: < 80%)

        Latency (milliseconds):
        - p50 latency (target: < 10ms)
        - p95 latency (target: < 50ms)
        - p99 latency (target: < 100ms, alert: > 200ms)

        Error Rate:
        - 4xx errors (target: < 1%, alert: > 2%)
        - 5xx errors (target: < 0.1%, alert: > 0.5%)
        - Timeout rate (target: < 0.1%, alert: > 1%)
        ```

        **Infrastructure Metrics:**
        ```
        Compute:
        - CPU usage (alert: > 80%)
        - Memory usage (alert: > 85%)
        - Network I/O (alert: > 70% capacity)

        Database:
        - Connection pool usage (alert: > 80%)
        - Query latency (alert: > 50ms p99)
        - Replication lag (alert: > 5 seconds)
        - Disk space (alert: > 80% full)

        Cache:
        - Hit rate (alert: < 80%)
        - Memory usage (alert: > 90%)
        - Eviction rate (alert: > 1000/sec)
        ```

        **Business Metrics:**
        ```
        - URLs created per day
        - Active URLs (accessed in last 30 days)
        - Average clicks per URL
        - User retention rate
        ```

        ### Alerting Strategy

        | Severity | Response Time | Notification |
        |----------|--------------|--------------|
        | **Critical** | Immediate | Page on-call engineer |
        | **High** | 15 minutes | Slack + email |
        | **Medium** | 1 hour | Email team |
        | **Low** | Daily digest | Email report |

    === "Cost & Security"

        ### Cost Analysis

        **Monthly costs for 4K reads/sec, 40 writes/sec:**

        ```
        Compute (API Servers):
        - 3 × m5.large instances @ $73/month
        - Total: $219/month

        Database (RDS PostgreSQL):
        - 1 × db.m5.large (primary) @ $140/month
        - 3 × db.m5.large (replicas) @ $420/month
        - Storage: 500GB @ $115/month
        - Total: $675/month

        Cache (Redis):
        - 1 × cache.m5.large @ $146/month
        - Total: $146/month

        Load Balancer:
        - ALB @ $20/month + $0.008/LCU
        - Total: ~$50/month

        Data Transfer:
        - 10TB outbound @ $0.09/GB
        - Total: $900/month

        Monitoring (CloudWatch/Datadog):
        - Total: ~$100/month

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        TOTAL: ~$2,100/month
        ```

        **Cost Optimization:**
        - Reserved instances: Save 30-40% on compute
        - Spot instances: Save 70% for non-critical work
        - Data transfer: CDN reduces costs by 80%
        - Storage tiers: Archive old URLs to S3 Glacier

        **Optimized cost: ~$1,200/month**

        ---

        ### Security Considerations

        | Threat | Impact | Mitigation |
        |--------|--------|-----------|
        | **Rate Limit Abuse** | Service degradation | 100 URLs/hour per user. 1000/hour per IP. Exponential backoff |
        | **Malicious URLs** | Brand reputation | URL scanning service. Blocklist maintenance. User reporting |
        | **SSRF Attacks** | Internal network access | Validate URL format. Block private IP ranges. Whitelist protocols (http/https only) |
        | **SQL Injection** | Data breach | Parameterized queries. Input validation. Least privilege DB accounts |
        | **DDoS** | Service unavailability | CloudFlare DDoS protection. Rate limiting. Auto-scaling |
        | **URL Enumeration** | Privacy concern | Non-sequential generation. 7+ character codes. Rate limit guessing attempts |

        **Security checklist:**
        - [ ] HTTPS everywhere (TLS 1.3)
        - [ ] Input validation on all endpoints
        - [ ] Rate limiting per user/IP
        - [ ] SQL injection prevention (parameterized queries)
        - [ ] CORS policies configured
        - [ ] Security headers (HSTS, CSP, X-Frame-Options)
        - [ ] Regular security audits
        - [ ] Incident response plan

    === "Final Architecture"

        ### Complete System Design

        ```mermaid
        graph TD
            CDN[CloudFlare CDN<br/>DDoS Protection] --> LB[Load Balancer<br/>HAProxy/nginx]
            LB --> API1[API Server 1]
            LB --> API2[API Server 2]
            LB --> API3[API Server 3]

            API1 --> Redis[(Redis Cluster)]
            API2 --> Redis
            API3 --> Redis

            API1 --> DB[(Database<br/>Primary)]
            API2 --> DB
            API3 --> DB

            API1 --> Kafka[Kafka<br/>Async Queue]
            API2 --> Kafka
            API3 --> Kafka

            DB --> Replica[(Database<br/>Replicas<br/>3 nodes)]

            style CDN fill:#e1f5ff
            style LB fill:#fff4e1
            style API1 fill:#e8f5e9
            style API2 fill:#e8f5e9
            style API3 fill:#e8f5e9
            style Redis fill:#ffe1e1
            style DB fill:#f3e5f5
            style Replica fill:#f3e5f5
            style Kafka fill:#fff3e0
        ```

        ### System Capabilities

        **Performance:**
        - Handles 4,000 reads/sec (99% from cache)
        - Handles 40 writes/sec
        - < 100ms p99 latency for redirects
        - < 500ms p99 latency for URL generation

        **Reliability:**
        - 99.9% availability (43 min downtime/month)
        - Zero data loss (replicated database)
        - Automatic failover in < 60 seconds
        - Disaster recovery with multi-region

        **Scalability:**
        - Can scale to 40K reads/sec with cache tuning
        - Can scale to 400K reads/sec with CDN
        - Database sharding path planned for 100+ writes/sec
        - Horizontal scaling for all components

        **Cost:**
        - $1,200/month optimized
        - $0.30 per 1000 URL creations
        - $0.003 per 1000 redirects

---

## Using This Framework

**For every interview:**

1. ✅ Always start with Step 1 (clarify requirements)
2. ✅ Draw diagrams in Step 2 (don't skip visuals)
3. ✅ Be ready for any deep dive in Step 3
4. ✅ End strong with Step 4 (show production thinking)

**Time management:**

- Set a timer during practice
- Aim to finish each step on time
- If running long, interviewer will redirect you

**Communication tips:**

- Think out loud constantly
- Ask clarifying questions
- Discuss trade-offs for every decision
- Be open to feedback and suggestions

---

[← Back to Interview Prep](index.md) | [Practice Problems](practice-problems.md) | [Communication Tips](communication.md)
