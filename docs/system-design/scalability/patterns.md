# Scalability Patterns

**Master scaling from hundreds to millions of users** | 📈 Patterns | ⚖️ Trade-offs | 💼 Interview Ready

## Quick Reference

**Scalability** - The ability to handle increased load by adding resources while maintaining performance:

| Pattern | Purpose | Complexity | Cost | When to Use |
|---------|---------|------------|------|-------------|
| **Horizontal Scaling** | Add more servers | High | Linear | Unlimited growth, fault tolerance needed |
| **Vertical Scaling** | Bigger servers | Low | Exponential | Quick wins, simple systems |
| **Load Balancing** | Distribute traffic | Medium | Low | Multiple servers, even distribution |
| **Caching** | Store frequently accessed data | Low | Low | Read-heavy workloads, repetitive queries |
| **Database Sharding** | Partition data across DBs | Very High | High | Massive datasets, write-heavy loads |
| **Read Replicas** | Separate read/write DBs | Medium | Medium | Read-heavy workloads, 90%+ reads |
| **CDN** | Cache static content globally | Low | Medium | Global users, static assets |
| **Microservices** | Decompose monolith | Very High | High | Large teams, independent scaling |

**Key Insight:** Scale incrementally. Don't over-engineer early. Start simple, measure, then optimize bottlenecks.

---

=== "🎯 Understanding Scalability"

    ## What is Scalability?

    **Scalability** is your system's ability to handle growth - more users, more data, more transactions - without degrading performance.

    ### The Restaurant Analogy

    === "Vertical Scaling"

        ```
        🍽️ Vertical Scaling (Bigger Kitchen)
        ─────────────────────────────────────────

        Week 1: Small kitchen, 1 chef
                Serves 50 customers/day

        Week 2: Bigger kitchen, professional chef, better equipment
                Serves 150 customers/day

        Week 3: Massive kitchen, expert chef, industrial equipment
                Serves 300 customers/day

        Week 4: Can't expand kitchen anymore! ❌
                Hit physical limits

        ✅ Simple (same management style)
        ❌ Expensive (industrial equipment costs 10x)
        ❌ Limited (can't expand beyond building)
        ```

    === "Horizontal Scaling"

        ```
        🍽️ Horizontal Scaling (More Locations)
        ─────────────────────────────────────────

        Week 1: 1 restaurant location
                Serves 50 customers/day

        Week 2: 3 restaurant locations
                Serves 150 customers/day (50 × 3)

        Week 3: 10 restaurant locations
                Serves 500 customers/day (50 × 10)

        Week 4: 100 restaurant locations
                Serves 5,000 customers/day (50 × 100)

        ✅ Unlimited scaling (open more locations)
        ✅ Fault tolerant (one closes, others still work)
        ❌ Complex (coordination, consistency, supply chain)
        ```

    ---

    ## Scaling Dimensions

    ```mermaid
    graph TB
        subgraph "3 Dimensions of Scale"
        A[Size<br/>More users, more data]
        B[Complexity<br/>More features, more integrations]
        C[Geography<br/>Global users, low latency]
        end

        A -->|Need| D[Horizontal Scaling<br/>Sharding]
        B -->|Need| E[Microservices<br/>Service Decomposition]
        C -->|Need| F[CDN<br/>Edge Computing]

        style A fill:#ff6b6b
        style B fill:#fab005
        style C fill:#51cf66
    ```

    ### Scalability vs Performance

    | Aspect | Scalability | Performance |
    |--------|------------|-------------|
    | **Definition** | Handles growth by adding resources | Speed of operations |
    | **Metric** | Throughput (requests/sec) | Latency (response time) |
    | **Solution** | Add more machines | Optimize algorithms, code |
    | **Example** | 1 server → 10 servers = 10x capacity | Query optimization: 500ms → 50ms |
    | **Cost** | Predictable (linear) | Variable (effort-based) |

=== "📊 Scaling Strategies"

    ## The Two Approaches

    === "Horizontal Scaling"

        ### Scale Out: Add More Machines

        **How It Works:**
        ```
        Single Server (before):
        ───────────────────────
        All traffic → [Server] → Database
                      (100% load)

        Horizontal Scale (after):
        ─────────────────────────
        Traffic → [Load Balancer]
                       ├─> [Server 1] (33% load)
                       ├─> [Server 2] (33% load)
                       └─> [Server 3] (33% load)
                              ↓
                       [Shared Database]
        ```

        ---

        ### Real-World Example: Web Servers

        ```
        Scaling WhatsApp Web Servers:
        ─────────────────────────────────────

        Stage 1: 1 server, 10,000 users
        Stage 2: 10 servers, 100,000 users
        Stage 3: 100 servers, 1,000,000 users
        Stage 4: 1,000 servers, 10,000,000 users

        Cost per user: ~$0.10/month (predictable)
        ```

        ---

        ### Pros & Cons

        **Pros:**
        - ✅ No ceiling on growth (add infinite servers)
        - ✅ Fault tolerant (one fails, others continue)
        - ✅ Cost-effective at scale ($10K → $20K for 2x capacity)
        - ✅ Incremental scaling (add 1 server at a time)

        **Cons:**
        - ❌ Complex architecture (coordination needed)
        - ❌ Data consistency challenges
        - ❌ Network latency between servers
        - ❌ Requires redesign (can't just "add more")

        ---

        ### When to Use

        | Scenario | Why Horizontal |
        |----------|---------------|
        | **Stateless web apps** | Each request independent, easy to distribute |
        | **Read-heavy workloads** | Replicate data, distribute reads |
        | **Unpredictable growth** | Scale up/down based on demand |
        | **24/7 availability** | Rolling updates, no downtime |
        | **Global users** | Deploy in multiple regions |

    === "Vertical Scaling"

        ### Scale Up: Bigger Machines

        **How It Works:**
        ```
        Before:
        ───────────────────────────────
        Server: 4 CPU cores, 16GB RAM
                Handles 1,000 req/sec

        After Vertical Scale:
        ───────────────────────────────
        Server: 32 CPU cores, 128GB RAM
                Handles 8,000 req/sec

        Same architecture, just bigger hardware
        ```

        ---

        ### Real-World Example: Database

        ```
        Scaling PostgreSQL Database:
        ───────────────────────────────────

        Week 1: 4 cores, 16GB RAM
                1,000 queries/sec
                Cost: $100/month

        Week 4: 8 cores, 32GB RAM
                2,000 queries/sec
                Cost: $250/month

        Week 12: 16 cores, 64GB RAM
                 4,000 queries/sec
                 Cost: $800/month

        Week 24: 32 cores, 128GB RAM
                 8,000 queries/sec
                 Cost: $2,500/month ⚠️ Expensive!

        Week 36: Can't upgrade anymore! ❌
        ```

        ---

        ### Pros & Cons

        **Pros:**
        - ✅ Simple (no code changes needed)
        - ✅ Quick implementation (order bigger server)
        - ✅ Strong consistency (single machine)
        - ✅ No distributed system complexity

        **Cons:**
        - ❌ Physical limits (max 128 cores, 1TB RAM)
        - ❌ Expensive (2x capacity = 5x cost)
        - ❌ Single point of failure
        - ❌ Downtime during upgrades

        ---

        ### When to Use

        | Scenario | Why Vertical |
        |----------|-------------|
        | **Legacy monoliths** | Can't redesign architecture |
        | **Databases** | Strong consistency required |
        | **Quick wins** | Need temporary scale, buy time |
        | **Small-medium systems** | <10,000 concurrent users |
        | **Stateful applications** | Session state on single machine |

    === "Comparison"

        ### Side-by-Side Comparison

        | Factor | Horizontal (Scale Out) | Vertical (Scale Up) |
        |--------|----------------------|---------------------|
        | **Max Capacity** | Unlimited (add infinite servers) | Limited (hardware max) |
        | **Cost Curve** | Linear (double cost = double capacity) | Exponential (double cost = 1.5x capacity) |
        | **Complexity** | High (distributed systems) | Low (single machine) |
        | **Fault Tolerance** | High (redundancy built-in) | Low (SPOF) |
        | **Availability** | 99.99% (redundancy) | 99.9% (single machine) |
        | **Consistency** | Challenging (eventual) | Easy (ACID) |
        | **Time to Scale** | Minutes (provision new server) | Hours (migration, downtime) |
        | **Implementation** | Redesign required | Drop-in upgrade |
        | **Example Systems** | Google, Facebook, Netflix | Startups, small apps |

        ---

        ### Hybrid Approach (Best Practice)

        ```
        Real-World: Combine Both
        ─────────────────────────────────────────

        Stage 1: Vertical scaling (0-10K users)
        - Single beefy server
        - Fast development
        - Low complexity

        Stage 2: Horizontal web tier (10K-100K users)
        - Vertical scale database
        - Horizontal scale web servers
        - Add load balancer

        Stage 3: Horizontal everything (100K+ users)
        - Horizontal web servers
        - Database sharding
        - Distributed cache
        - CDN for static assets

        Best of both worlds! ✓
        ```

=== "🏗️ Core Patterns"

    ## Essential Scalability Patterns

    === "1. Load Balancing"

        ### Distribute Traffic Across Servers

        **Architecture:**
        ```
        ┌─────────────────────────────────────────┐
        │         Internet Traffic                │
        └───────────────┬─────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Load Balancer │ ← Single entry point
                │  (Layer 4/7)  │
                └───────┬───────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        ┌────────┐  ┌────────┐  ┌────────┐
        │ Web 1  │  │ Web 2  │  │ Web 3  │
        │ 33%    │  │ 33%    │  │ 33%    │
        └────────┘  └────────┘  └────────┘
        ```

        ---

        ### Load Balancing Algorithms

        | Algorithm | How It Works | Use Case | Pros/Cons |
        |-----------|-------------|----------|-----------|
        | **Round Robin** | Request 1→A, 2→B, 3→C, 4→A... | Equal servers | Simple, but ignores load |
        | **Weighted RR** | A gets 3x traffic vs B | Unequal servers | Respects capacity |
        | **Least Connections** | Route to server with fewest active connections | Long-lived connections | Fair, but overhead |
        | **IP Hash** | hash(client_ip) → server | Session affinity | Sticky sessions |
        | **Least Response Time** | Route to fastest server | Performance critical | Requires monitoring |

        ---

        ### Example: Round Robin

        ```
        Incoming Requests:
        ─────────────────────────────────────
        Request 1 (user A) → Server 1
        Request 2 (user B) → Server 2
        Request 3 (user C) → Server 3
        Request 4 (user D) → Server 1 (loop back)
        Request 5 (user E) → Server 2
        Request 6 (user F) → Server 3

        Result: Evenly distributed (33% each)
        ```

        ---

        ### Load Balancer Layers

        **Layer 4 (Transport Layer):**
        ```
        Fast but basic:
        - Routes based on IP/Port
        - No content inspection
        - 100,000+ req/sec
        - Use: TCP/UDP routing
        ```

        **Layer 7 (Application Layer):**
        ```
        Smart but slower:
        - Routes based on URL, headers, cookies
        - Content-aware decisions
        - 10,000-50,000 req/sec
        - Use: HTTP routing, A/B testing
        ```

        ---

        ### High Availability

        ```
        Problem: Load balancer is SPOF
        ─────────────────────────────────────
        Solution: Multiple load balancers

        ┌────────────────────┐
        │   DNS Round Robin  │
        └─────────┬──────────┘
                  │
          ┌───────┼───────┐
          │               │
          ▼               ▼
    ┌───────────┐   ┌───────────┐
    │    LB 1   │   │    LB 2   │
    │  (Active) │   │ (Standby) │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  ▼
            Web Servers
        ```

    === "2. Caching"

        ### Store Frequently Accessed Data

        **Multi-Layer Caching:**
        ```
        Request Flow (with caching):
        ─────────────────────────────────────────────

        1. Browser Cache (0ms)
           ↓ Miss
        2. CDN Edge (50ms)
           ↓ Miss
        3. Application Cache (100ms) ← Redis/Memcached
           ↓ Miss
        4. Database Query Cache (150ms)
           ↓ Miss
        5. Database (500ms) ← Expensive!

        Cache Hit at Layer 3: 100ms (5x faster!)
        ```

        ---

        ### Caching Strategies

        | Strategy | How It Works | Consistency | Use Case |
        |----------|-------------|-------------|----------|
        | **Cache-Aside** | App checks cache, fetches if miss | Eventual | General purpose, read-heavy |
        | **Write-Through** | Write to cache + DB simultaneously | Strong | Write + read frequently |
        | **Write-Behind** | Write to cache, async to DB | Eventual | High write volume |
        | **Read-Through** | Cache handles DB fetch | Strong | Read-heavy, complex queries |
        | **Refresh-Ahead** | Proactively refresh before expiry | Eventual | Predictable access patterns |

        ---

        ### Cache-Aside Pattern (Most Common)

        ```
        Read Flow:
        ──────────────────────────────────────
        1. App checks cache
           ├─ Hit: Return cached data (fast!) ✓
           └─ Miss:
              2. Query database
              3. Store in cache (TTL = 5 min)
              4. Return data

        Write Flow:
        ──────────────────────────────────────
        1. Write to database
        2. Invalidate cache (or update)
        3. Next read will fetch fresh data
        ```

        ---

        ### Cache Eviction Policies

        ```
        When cache is full, what to remove?
        ─────────────────────────────────────────────

        LRU (Least Recently Used):
        - Remove oldest accessed item
        - Best for: Temporal locality
        - Example: Recent news articles

        LFU (Least Frequently Used):
        - Remove least accessed item
        - Best for: Popular content
        - Example: Trending videos

        TTL (Time To Live):
        - Remove after X seconds
        - Best for: Time-sensitive data
        - Example: Stock prices

        FIFO (First In First Out):
        - Remove oldest added item
        - Best for: Simple, predictable
        - Example: Logs, events
        ```

        ---

        ### Cache Sizing

        ```
        Rule of Thumb:
        ─────────────────────────────────────────────
        Cache Hit Rate Target: 80-90%

        Memory Sizing:
        - Identify hot data (Pareto principle: 20% of data = 80% of requests)
        - Cache size = Hot data × 1.5 (buffer)

        Example:
        - Total data: 100GB
        - Hot data: 20GB (20% of total)
        - Cache size: 30GB (20GB × 1.5)
        - Expected hit rate: 85%
        ```

    === "3. Database Scaling"

        ### Read Replicas

        **Architecture:**
        ```
        Write-Heavy Application:
        ─────────────────────────────────────────────
        Problem: All traffic hits single database

        Solution: Master-Replica Setup
        ─────────────────────────────────────────────

                    Application
                         │
                    ┌────┴────┐
                    │         │
              Writes│         │Reads (90%)
                    │         │
                    ▼         ▼
            ┌──────────┐   ┌──────────┐
            │  Master  │──>│ Replica 1│
            │   (DB)   │   │ (Read)   │
            └──────────┘   └──────────┘
                    │      ┌──────────┐
                    └─────>│ Replica 2│
                           │ (Read)   │
                           └──────────┘

        Replication lag: 100ms - 1s (eventual consistency)
        ```

        ---

        ### Database Sharding

        **Horizontal Partitioning:**
        ```
        Single Database (before):
        ─────────────────────────────────────────────
        Users Table: 10M users
        Size: 500GB
        Queries: 50,000/sec → Bottleneck!

        Sharded Database (after):
        ─────────────────────────────────────────────
        Shard 1: Users 0-2.5M     (125GB)
        Shard 2: Users 2.5M-5M    (125GB)
        Shard 3: Users 5M-7.5M    (125GB)
        Shard 4: Users 7.5M-10M   (125GB)

        Each shard: 12,500 queries/sec ✓
        Total capacity: 50,000 queries/sec ✓
        ```

        ---

        ### Sharding Strategies

        | Strategy | How It Works | Pros | Cons |
        |----------|-------------|------|------|
        | **Range-Based** | Users 1-1000 → Shard 1 | Simple, range queries easy | Hotspots (uneven distribution) |
        | **Hash-Based** | hash(user_id) % num_shards | Even distribution | Range queries hard |
        | **Geographic** | US users → US shard | Low latency, compliance | Uneven growth |
        | **Directory-Based** | Lookup table: user_id → shard | Flexible | Extra lookup overhead |

        ---

        ### Sharding Example: Hash-Based

        ```
        Shard Selection:
        ─────────────────────────────────────────────
        user_id = 12345
        shard = hash(12345) % 4 = 1

        → Route to Shard 1

        Consistency:
        - Same user_id always goes to same shard ✓
        - Different users evenly distributed ✓

        Challenge:
        - Join across shards expensive ❌
        - Rebalancing complex ❌
        ```

        ---

        ### Comparison: Replicas vs Sharding

        | Aspect | Read Replicas | Sharding |
        |--------|--------------|----------|
        | **Purpose** | Scale reads | Scale writes |
        | **Data** | Full copy on each | Partitioned across shards |
        | **Complexity** | Low | Very High |
        | **Queries** | Any query on any replica | Must know shard key |
        | **Consistency** | Eventual (replication lag) | Strong (within shard) |
        | **Use Case** | 90%+ reads | Write-heavy, huge datasets |

    === "4. CDN"

        ### Content Delivery Network

        **Global Distribution:**
        ```
        Without CDN:
        ─────────────────────────────────────────────
        User in Tokyo → US Origin Server
        - Distance: 10,000 km
        - Latency: 200ms
        - Bandwidth cost: High

        With CDN:
        ─────────────────────────────────────────────
        User in Tokyo → Tokyo Edge Server
        - Distance: 50 km
        - Latency: 10ms (20x faster!) ✓
        - Bandwidth cost: Low (95% from edge) ✓

        First request: 200ms (cache miss)
        Subsequent requests: 10ms (cache hit)
        ```

        ---

        ### CDN Architecture

        ```
                    [Origin Server]
                      (US West)
                          │
                    Replication
                          │
            ┌─────────────┼─────────────┐
            │             │             │
        ┌───────┐    ┌───────┐    ┌───────┐
        │  US   │    │  EU   │    │ Asia  │
        │ Edge  │    │ Edge  │    │ Edge  │
        └───┬───┘    └───┬───┘    └───┬───┘
            │            │            │
        US Users     EU Users    Asia Users
        ```

        ---

        ### What to Cache on CDN

        | Content Type | Cache TTL | Example |
        |-------------|-----------|---------|
        | **Static Assets** | 1 year | JS, CSS, images (versioned) |
        | **User-Generated** | 1 week | Profile photos, uploads |
        | **Dynamic Pages** | 5 minutes | Homepage, product listings |
        | **API Responses** | 1 minute | Leaderboards, trending |
        | **Video Segments** | 1 day | Streaming video chunks |

        ---

        ### CDN Benefits

        **Performance:**
        ```
        Global Latency Reduction:
        ─────────────────────────────────────
        Tokyo user: 200ms → 10ms (20x faster)
        London user: 150ms → 15ms (10x faster)
        Sydney user: 250ms → 20ms (12x faster)

        Bandwidth Savings:
        ─────────────────────────────────────
        Origin bandwidth: 1TB/day
        CDN offloads: 95%
        Origin bandwidth after CDN: 50GB/day
        Cost savings: ~$800/month
        ```

        **Availability:**
        - DDoS protection (distributed infrastructure)
        - Origin failures transparent (serve from cache)
        - Geographic redundancy

    === "5. Microservices"

        ### Service Decomposition

        **Monolith to Microservices:**
        ```
        Monolith (before):
        ─────────────────────────────────────────────
        ┌─────────────────────────────────────────┐
        │         Single Application              │
        │  ┌──────────────────────────────────┐  │
        │  │  Users │ Orders │ Products │ ... │  │
        │  └──────────────────────────────────┘  │
        │                                         │
        │  Problem:                              │
        │  - Scale entire app (wasteful)         │
        │  - One bug brings down everything      │
        │  - Deploy all or nothing               │
        └─────────────────────────────────────────┘

        Microservices (after):
        ─────────────────────────────────────────────
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  User    │  │  Order   │  │ Product  │
        │ Service  │  │ Service  │  │ Service  │
        │ (3 pods) │  │ (5 pods) │  │ (2 pods) │
        └──────────┘  └──────────┘  └──────────┘
              │             │              │
        Independent scaling, deployment, tech stack
        ```

        ---

        ### When to Use Microservices

        | Factor | Monolith ✓ | Microservices ✓ |
        |--------|-----------|----------------|
        | **Team Size** | <10 engineers | 10+ engineers (multiple teams) |
        | **Scale Requirements** | Uniform | Different services need different scale |
        | **Release Frequency** | Monthly | Multiple times/day |
        | **Tech Stack** | Uniform (e.g., all Python) | Polyglot (Python, Go, Java) |
        | **System Complexity** | Simple (<10 features) | Complex (10+ bounded contexts) |
        | **Operational Maturity** | Basic | Advanced (monitoring, tracing, logging) |

        ---

        ### Microservices Challenges

        ```
        Added Complexity:
        ─────────────────────────────────────────────

        1. Network Calls (latency)
           Monolith: Function call (0.001ms)
           Microservices: HTTP call (10ms) - 10,000x slower!

        2. Data Consistency
           Monolith: ACID transactions
           Microservices: Eventual consistency, sagas

        3. Monitoring
           Monolith: Single log file
           Microservices: Distributed tracing across 50+ services

        4. Deployment
           Monolith: Deploy 1 app
           Microservices: Deploy 20+ services (orchestration needed)

        Trade-off: Complexity for independent scalability
        ```

=== "💡 Interview Tips"

    ## Common Interview Questions

    **Q1: "How would you scale a system from 100 to 1 million users?"**

    **Good Answer:**
    ```
    Phase-by-Phase Approach:

    Phase 1 (100-1K users):
    ─────────────────────────
    - Single server (web + DB on same machine)
    - Vertical scaling as needed
    - Cost: ~$50/month

    Phase 2 (1K-10K users):
    ─────────────────────────
    - Separate web and database servers
    - Add caching layer (Redis)
    - Vertical scale both
    - Cost: ~$500/month

    Phase 3 (10K-100K users):
    ─────────────────────────
    - Horizontal scaling: Multiple web servers
    - Load balancer (NGINX/HAProxy)
    - Database read replicas (1 master, 2 replicas)
    - CDN for static assets
    - Cost: ~$5,000/month

    Phase 4 (100K-1M users):
    ─────────────────────────
    - Auto-scaling web tier (10-50 servers)
    - Database sharding (by user_id hash)
    - Distributed cache cluster
    - Message queues for async processing
    - Multi-region deployment
    - Cost: ~$50,000/month

    Key: Scale incrementally, measure at each phase
    ```

    ---

    **Q2: "What's the difference between horizontal and vertical scaling?"**

    **Good Answer:**
    ```
    Vertical Scaling (Scale Up):
    ─────────────────────────────
    - Add more power to existing machine
    - Example: 4 CPU → 16 CPU cores
    - Pros: Simple, no code changes
    - Cons: Hardware limits, expensive, SPOF
    - Use: Quick wins, databases, legacy systems

    Horizontal Scaling (Scale Out):
    ─────────────────────────────
    - Add more machines
    - Example: 1 server → 10 servers
    - Pros: Unlimited growth, fault tolerant
    - Cons: Complex (distributed systems)
    - Use: Modern web apps, stateless services

    Real World: Use both!
    - Vertical scale database
    - Horizontal scale web tier
    - Best of both worlds
    ```

    ---

    **Q3: "Explain database sharding and when to use it"**

    **Good Answer:**
    ```
    Sharding = Horizontal database partitioning

    How It Works:
    ─────────────────────────────
    Split data across multiple databases
    Example: Users table
    - Shard 1: Users with ID 0-999999
    - Shard 2: Users with ID 1000000-1999999
    - Shard 3: Users with ID 2000000-2999999

    Shard Key Selection:
    ─────────────────────────────
    Good: user_id (even distribution, no hotspots)
    Bad: signup_date (recent dates = hotspot)

    When to Use:
    ─────────────────────────────
    - Database >1TB (single machine limit)
    - Write-heavy workload (replicas don't help)
    - Predictable shard key (user_id, tenant_id)

    Challenges:
    ─────────────────────────────
    - Cross-shard queries expensive
    - Rebalancing complex
    - Application-level routing needed

    Alternative: Try read replicas first (simpler!)
    ```

    ---

    **Q4: "How do you handle session state in a scaled system?"**

    **Good Answer:**
    ```
    Problem: User sessions on single server
    ─────────────────────────────────────────────
    Server 1: User A logged in
    Server 2: User A session unknown → logged out!

    Solutions:

    1. Sticky Sessions (Simple):
       - Load balancer routes user to same server
       - Pros: Simple
       - Cons: Uneven load, SPOF

    2. Session Replication (Medium):
       - Replicate sessions across all servers
       - Pros: Fault tolerant
       - Cons: Network overhead, slow

    3. External Session Store (Best):
       - Store sessions in Redis/Memcached
       - All servers read from shared store
       - Pros: Scalable, fault tolerant
       - Cons: Additional infrastructure

    4. Stateless (Modern):
       - Use JWT tokens (client-side state)
       - No server-side sessions needed
       - Pros: Infinite scaling
       - Cons: Token size, revocation challenges

    Recommendation: External session store (Redis)
    ```

    ---

    **Q5: "What's the difference between CDN and caching?"**

    **Good Answer:**
    ```
    Both cache content, but different locations:

    CDN (Content Delivery Network):
    ─────────────────────────────────────────────
    - Geographic caching (edge servers globally)
    - Purpose: Reduce latency for global users
    - Content: Static assets (images, JS, CSS)
    - TTL: Hours to days (or longer)
    - Example: CloudFront, Cloudflare, Akamai
    - Benefit: Tokyo user gets content from Tokyo (20ms vs 200ms)

    Application Cache:
    ─────────────────────────────────────────────
    - In-memory caching (near application)
    - Purpose: Reduce database load
    - Content: Query results, computed data
    - TTL: Seconds to minutes
    - Example: Redis, Memcached
    - Benefit: Database query avoided (5ms vs 100ms)

    Use Both:
    ─────────────────────────────────────────────
    Browser → CDN (static) → App Cache (dynamic) → Database
    Fast at every layer!
    ```

    ---

    ## Interview Cheat Sheet

    **Quick Comparisons:**

    | Pattern | Complexity | Cost | Scale Limit | Use Case |
    |---------|------------|------|-------------|----------|
    | **Vertical Scaling** | Low | High (exponential) | Hardware limit | Quick wins, <10K users |
    | **Horizontal Scaling** | High | Medium (linear) | Unlimited | Modern apps, >10K users |
    | **Caching** | Low | Low | Memory limit | Read-heavy (90%+ reads) |
    | **Read Replicas** | Medium | Medium | ~10 replicas | Read-heavy, 80%+ reads |
    | **Sharding** | Very High | High | Unlimited | Write-heavy, >1TB data |
    | **CDN** | Low | Medium | Unlimited | Global users, static content |

    **Scaling Milestones:**

    ```
    0-1K users: Single server
    1K-10K users: Separate web + DB
    10K-100K users: Load balancer + replicas
    100K-1M users: Sharding + CDN + caching
    1M-10M users: Microservices + multi-region
    10M+ users: Custom solutions (Google-scale)
    ```

    **Common Metrics:**
    - **Throughput**: Requests per second (RPS)
    - **Latency**: Response time (p50, p95, p99)
    - **Availability**: Uptime percentage (99.9% = 8.7 hours downtime/year)
    - **Scalability**: How throughput changes with resources

=== "⚠️ Common Mistakes"

    ## Scalability Pitfalls

    | Mistake | Problem | Solution |
    |---------|---------|----------|
    | **Premature optimization** | Microservices for 100 users | Start simple, scale when needed |
    | **Ignoring bottlenecks** | Scale everything equally | Profile, find bottleneck, scale that |
    | **No caching** | Every request hits database | Add cache layer (easy 10x gain) |
    | **Vertical scaling only** | Hit hardware limits at 10K users | Plan horizontal scaling early |
    | **Wrong shard key** | Uneven distribution, hotspots | Choose evenly distributed key |
    | **No monitoring** | Don't know what's slow | Monitor everything (APM, metrics) |
    | **Stateful services** | Can't scale horizontally | Make services stateless |

    ---

    ## Design Pitfalls

    | Pitfall | Impact | Prevention |
    |---------|--------|-----------|
    | **Single database** | 10K users = bottleneck | Read replicas early |
    | **Synchronous calls** | Latency adds up (10 calls = 100ms) | Async where possible |
    | **N+1 queries** | 100 users = 101 queries | Batch queries, JOIN |
    | **No connection pooling** | Create new connection per request | Connection pool (10-50 connections) |
    | **Large payloads** | 1MB response = slow | Paginate, compress, CDN |
    | **Blocking operations** | Long-running task blocks server | Queue + background workers |

    ---

    ## Interview Red Flags

    **Avoid Saying:**
    - ❌ "Just add more servers" (no plan)
    - ❌ "We'll use microservices from day 1" (over-engineering)
    - ❌ "NoSQL is always faster" (wrong - depends on use case)
    - ❌ "Caching solves everything" (cache invalidation is hard)
    - ❌ "We need to handle 1M users immediately" (unrealistic)

    **Say Instead:**
    - ✅ "Start with monolith, profile bottlenecks, scale incrementally"
    - ✅ "Measure first: if DB is bottleneck, add replicas"
    - ✅ "Use caching for read-heavy workloads (80/20 rule)"
    - ✅ "Horizontal scaling for stateless web tier, vertical for database"
    - ✅ "Target 10K users first, then optimize for 100K"

---

## 🎯 Key Takeaways

**The 10 Rules of Scalability:**

1. **Start simple** - Don't over-engineer early. Optimize when you have real data.

2. **Measure everything** - Can't optimize what you don't measure. Add monitoring early.

3. **Scale incrementally** - 1 server → 3 servers → 10 servers. Not 1 → 100 immediately.

4. **Horizontal for web, vertical for DB** - Proven pattern. Web tier is stateless, DB needs strong consistency.

5. **Cache aggressively** - 80% of requests hit 20% of data. Cache that 20% (easy 10x gain).

6. **CDN for global users** - Tokyo users shouldn't wait 200ms. Edge servers = 10ms.

7. **Read replicas before sharding** - Sharding is complex. Try replicas first (handles 90% of cases).

8. **Stateless services** - Can't scale stateful services horizontally. Session in Redis, not memory.

9. **Async where possible** - Email, notifications, reports = queue + background workers.

10. **Different parts, different strategies** - Web tier horizontal, DB vertical, static content CDN.

---

## 📚 Further Reading

**Master these related concepts:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **Load Balancing** | Distribute traffic efficiently | [Load Balancing Deep Dive →](../networking/load-balancing.md) |
| **Caching Strategies** | 10x performance boost | [Caching Patterns →](../performance/caching.md) |
| **Database Scaling** | Handle millions of records | [Database Patterns →](../databases/scaling.md) |
| **CDN** | Global low-latency delivery | [CDN Guide →](../networking/cdn.md) |

**Practice with real systems:**
- [Design Instagram](../problems/instagram.md) - Image storage, CDN, caching
- [Design URL Shortener](../problems/url-shortener.md) - High throughput, database scaling
- [Design Twitter](../problems/twitter.md) - Fan-out, caching, read-heavy optimization

---

**Master scalability patterns and build systems that handle millions! 🚀**
