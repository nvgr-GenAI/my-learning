# System Design Principles

**Master the fundamentals that drive every architecture decision** | 🎯 Core Concepts | ⚖️ Trade-offs | 💼 Interview Ready

## Quick Reference

**The 5 pillars of system design** - memorize these for interviews:

| Principle | Key Question | Common Trade-off | Interview Focus |
|-----------|-------------|------------------|-----------------|
| **Scalability** | Can it handle 10x load? | Cost vs Performance | Horizontal vs Vertical scaling |
| **Reliability** | What if components fail? | Complexity vs Uptime | MTBF, MTTR, Circuit Breakers |
| **Consistency** | Do all nodes see same data? | Speed vs Accuracy | CAP theorem, Eventual consistency |
| **Availability** | Is it always accessible? | Consistency vs Uptime | 99.9% vs 99.99% vs 99.999% |
| **Performance** | How fast does it respond? | Latency vs Throughput | Caching, CDN, Load balancing |

---

=== "📈 Scalability"

    ## What is Scalability?

    **The ability to handle increased load by adding resources.**

    ### Core Concepts

    | Approach | Description | When to Use | Limits |
    |----------|-------------|-------------|--------|
    | **Horizontal Scaling** | Add more machines | Web servers, stateless services, microservices | Network overhead, data consistency |
    | **Vertical Scaling** | Add more power (CPU, RAM) | Databases, legacy apps, monoliths | Hardware limits (~96 cores, 1TB RAM) |

    ---

    ### Horizontal vs Vertical Comparison

    ```mermaid
    graph LR
        subgraph Vertical Scaling
        A[Single Server<br/>4 cores, 8GB RAM] -->|Upgrade| B[Single Server<br/>16 cores, 64GB RAM]
        end

        subgraph Horizontal Scaling
        C[Server 1<br/>4 cores, 8GB RAM]
        D[Server 2<br/>4 cores, 8GB RAM]
        E[Server 3<br/>4 cores, 8GB RAM]
        F[Load Balancer] --> C
        F --> D
        F --> E
        end
    ```

    **Decision Matrix:**

    | Factor | Horizontal | Vertical |
    |--------|-----------|----------|
    | **Cost** | Linear growth | Exponential at high end |
    | **Complexity** | Higher (distributed systems) | Lower (single machine) |
    | **Scalability** | Near unlimited | Hardware ceiling |
    | **Downtime** | Zero (rolling updates) | Required for upgrades |
    | **Single Point of Failure** | No (redundancy built-in) | Yes (unless clustered) |

    ---

    ### Real-World Examples

    | Company | Strategy | Scale Achieved |
    |---------|----------|----------------|
    | **Netflix** | Horizontal (AWS EC2, microservices) | 200M+ subscribers, thousands of instances |
    | **Stack Overflow** | Vertical (powerful servers) | 13M users with <10 servers |
    | **WhatsApp** | Hybrid (vertical for efficiency, horizontal for users) | 2B users, 50 engineers |

    ---

    ### Capacity Estimation Example

    **Problem:** Design a URL shortener for 100M users

    ```
    Assumptions:
    - 100M users
    - 10% active daily = 10M DAU
    - Each user creates 1 short URL per week = 1.4M URLs/day
    - Read:Write ratio = 100:1 (100M reads/day)

    QPS Calculation:
    - Write QPS = 1.4M / 86400 = ~16 writes/sec
    - Read QPS = 100M / 86400 = ~1150 reads/sec
    - Peak QPS (3x average) = ~3500 reads/sec

    Scaling Strategy:
    - Start: 1 web server + 1 database (handles ~5K QPS)
    - Phase 1 (1K QPS): Add read replicas for database
    - Phase 2 (5K+ QPS): Horizontal scaling + load balancer
    - Phase 3 (10K+ QPS): Add caching layer (Redis)
    - Phase 4 (50K+ QPS): Database sharding
    ```

    ---

    ### Interview Tips

    **Common Questions:**
    - "How would you scale to 10M users?"
    - "What happens when your database becomes the bottleneck?"
    - "Explain horizontal vs vertical scaling"

    **How to Answer:**
    1. Start with current scale and bottleneck
    2. Identify what component can't handle load
    3. Propose scaling strategy (vertical first, then horizontal)
    4. Mention monitoring and gradual rollout
    5. Discuss trade-offs (cost, complexity, consistency)

    **Red Flags to Avoid:**
    - ❌ Immediately jumping to microservices for small scale
    - ❌ Not mentioning database scaling (it's often the bottleneck)
    - ❌ Ignoring stateful vs stateless services
    - ❌ Not discussing how to handle database writes

    **Good Answer Template:**
    ```
    "Initially, we'd use a monolith with vertical scaling since it's simpler.
    As we approach hardware limits or need independent scaling, we'd:
    1. Split read/write traffic (read replicas)
    2. Add horizontal scaling for stateless services (web/app tier)
    3. Implement caching to reduce database load
    4. Consider database sharding for write scalability
    5. Move to microservices only if team/deployment velocity requires it"
    ```

    ---

    ### Key Takeaways

    - Start simple: Vertical scaling is easier to implement
    - Database is usually the first bottleneck (not application servers)
    - Horizontal scaling requires stateless design
    - Load balancers are essential for horizontal scaling
    - Monitor before you scale (measure, then optimize)

    **Further Reading:** [Scalability Patterns →](scalability-patterns.md)

=== "🛡️ Reliability"

    ## What is Reliability?

    **The probability a system performs correctly over time, even when things fail.**

    ### Key Metrics

    | Metric | Formula | Target | Meaning |
    |--------|---------|--------|---------|
    | **MTBF** | Average time between failures | Days to months | How often failures occur |
    | **MTTR** | Average time to recover | Minutes to hours | How fast you recover |
    | **Availability** | MTBF / (MTBF + MTTR) × 100% | 99.9%+ | % of time system works |

    **Availability Table:**

    | Target | Downtime/Year | Downtime/Month | Downtime/Week | Use Case |
    |--------|---------------|----------------|---------------|----------|
    | **90%** | 36.5 days | 3 days | 16.8 hours | Development, internal tools |
    | **99%** ("two nines") | 3.65 days | 7.2 hours | 1.68 hours | Basic web services |
    | **99.9%** ("three nines") | 8.76 hours | 43.2 minutes | 10.1 minutes | Most web applications |
    | **99.99%** ("four nines") | 52.6 minutes | 4.32 minutes | 1.01 minutes | Business-critical services |
    | **99.999%** ("five nines") | 5.26 minutes | 25.9 seconds | 6.05 seconds | Financial systems, healthcare |

    ---

    ### Reliability Patterns

    === "Circuit Breaker"

        **Prevents cascading failures by failing fast**

        ```python
        from enum import Enum
        import time

        class CircuitState(Enum):
            CLOSED = "closed"    # Normal operation
            OPEN = "open"        # Failing, reject requests
            HALF_OPEN = "half_open"  # Testing if recovered

        class CircuitBreaker:
            def __init__(self, failure_threshold=5, timeout=60):
                self.failure_threshold = failure_threshold
                self.timeout = timeout  # Seconds before trying again
                self.failure_count = 0
                self.last_failure_time = None
                self.state = CircuitState.CLOSED

            def call(self, func, *args):
                if self.state == CircuitState.OPEN:
                    if time.time() - self.last_failure_time > self.timeout:
                        self.state = CircuitState.HALF_OPEN
                    else:
                        raise Exception("Circuit breaker OPEN")

                try:
                    result = func(*args)
                    self.on_success()
                    return result
                except Exception as e:
                    self.on_failure()
                    raise e

            def on_success(self):
                self.failure_count = 0
                self.state = CircuitState.CLOSED

            def on_failure(self):
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN

        # Usage
        breaker = CircuitBreaker(failure_threshold=3, timeout=30)

        def call_external_api():
            breaker.call(api_request, data)
        ```

        **When to Use:**
        - Calling external services that might fail
        - Preventing cascading failures in microservices
        - Protecting your system from slow/unresponsive dependencies

    === "Retry with Backoff"

        **Handle transient failures gracefully**

        ```python
        import time
        import random

        def exponential_backoff_retry(func, max_retries=3, base_delay=1):
            """
            Retry with exponential backoff and jitter

            Delay pattern: 1s, 2s, 4s, 8s (with random jitter)
            """
            for attempt in range(max_retries):
                try:
                    return func()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise  # Last attempt, give up

                    # Exponential backoff: 2^attempt * base_delay
                    delay = (2 ** attempt) * base_delay

                    # Add jitter (±25%) to prevent thundering herd
                    jitter = delay * 0.25 * (random.random() * 2 - 1)
                    sleep_time = delay + jitter

                    print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)

        # Usage
        result = exponential_backoff_retry(
            lambda: api_call(),
            max_retries=4,
            base_delay=0.5
        )
        ```

        **Best Practices:**
        - Use exponential backoff to reduce load on failing service
        - Add jitter to prevent thundering herd problem
        - Set max retry limit to avoid infinite loops
        - Only retry idempotent operations

    === "Graceful Degradation"

        **Maintain core functionality when components fail**

        ```python
        def get_user_profile(user_id):
            """
            Show user profile even if some features fail
            """
            # Core data (must succeed)
            try:
                user = database.get_user(user_id)
            except Exception:
                return {"error": "User not found"}  # Fail completely

            # Optional features (degrade gracefully)
            profile = {
                "id": user.id,
                "name": user.name,
                "email": user.email
            }

            # Try to get recommendations (optional)
            try:
                profile["recommendations"] = recommendation_service.get(user_id)
            except Exception:
                profile["recommendations"] = []  # Empty, but don't fail

            # Try to get activity feed (optional)
            try:
                profile["activity"] = activity_service.get_recent(user_id)
            except Exception:
                profile["activity"] = []  # Empty, but don't fail

            return profile
        ```

        **Strategy:**
        - Identify critical vs optional features
        - Return partial data rather than failing completely
        - Show cached/stale data when fresh data unavailable
        - Display error messages for non-critical failures

    ---

    ### Interview Tips

    **Common Questions:**
    - "How do you achieve 99.99% availability?"
    - "What's the difference between MTBF and MTTR?"
    - "Design a system that handles server failures"

    **How to Answer:**

    **For "99.99% availability":**
    ```
    "To achieve 99.99% (52 min downtime/year), we need:

    1. Redundancy:
       - Multi-region deployment
       - Load balancing across multiple instances
       - Database replication (master-replica)

    2. Fault Tolerance:
       - Circuit breakers for external dependencies
       - Retry logic with exponential backoff
       - Graceful degradation for non-critical features

    3. Monitoring & Alerting:
       - Health checks every 30 seconds
       - Alert on-call engineer within 1 minute
       - Automated failover to standby systems

    4. Deployment Strategy:
       - Blue-green deployments (zero downtime)
       - Canary releases (catch issues early)
       - Automated rollback on error threshold

    This allows us to handle failures within our 4-minute monthly budget."
    ```

    **Red Flags to Avoid:**
    - ❌ Claiming 100% availability (physically impossible)
    - ❌ Not mentioning monitoring/alerting
    - ❌ Forgetting about database failures
    - ❌ Ignoring deployment-related downtime

    ---

    ### Chaos Engineering

    **Netflix's approach: Break things intentionally to build resilience**

    | Tool | What It Does | Lesson |
    |------|-------------|--------|
    | **Chaos Monkey** | Randomly kills production instances | Systems must handle instance failures |
    | **Chaos Kong** | Simulates entire AWS region failure | Need multi-region architecture |
    | **Latency Monkey** | Adds artificial delays | Must timeout and degrade gracefully |

    **Further Reading:** [Reliability & Fault Tolerance →](reliability-fault-tolerance.md)

=== "⚖️ Consistency & CAP"

    ## What is Consistency?

    **All nodes in a distributed system see the same data at the same time.**

    ### Consistency Models

    | Model | Guarantee | Latency | Use Case |
    |-------|-----------|---------|----------|
    | **Strong Consistency** | All nodes see same data instantly | High | Banking, inventory, booking systems |
    | **Eventual Consistency** | All nodes converge to same state | Low | Social media, content delivery, DNS |
    | **Weak Consistency** | No guarantees | Very Low | Live streaming, gaming, VoIP |
    | **Causal Consistency** | Related operations ordered | Medium | Collaborative editing, comments |

    ---

    ## CAP Theorem

    **In a distributed system with network partitions, you must choose between Consistency and Availability.**

    ### The Three Properties

    ```mermaid
    graph TB
        subgraph "CAP Theorem Triangle"
        C[Consistency<br/>All nodes see same data]
        A[Availability<br/>System always responds]
        P[Partition Tolerance<br/>Works despite network failures]
        C -.-> A
        A -.-> P
        P -.-> C
        end

        subgraph "Reality: Pick 2"
        CP[CP Systems<br/>MongoDB, HBase<br/>Banking]
        AP[AP Systems<br/>Cassandra, DynamoDB<br/>Social Media]
        CA[CA Systems<br/>Single RDBMS<br/>Legacy Apps]
        end
    ```

    **Key Insight:** Network partitions will happen in any distributed system, so you're really choosing between **C** and **A** during partition events.

    ---

    ### CAP Trade-offs in Practice

    | System Type | Choice | Example Systems | Behavior During Partition |
    |-------------|--------|-----------------|---------------------------|
    | **CP** | Consistency + Partition Tolerance | MongoDB, Redis Cluster, HBase, ZooKeeper | Reject writes to minority partition, return errors |
    | **AP** | Availability + Partition Tolerance | Cassandra, DynamoDB, Riak, Voldemort | Accept all writes, resolve conflicts later |
    | **CA** | Consistency + Availability | Traditional RDBMS (single node), VoltDB | Not truly distributed, single point of failure |

    ---

    ### Real-World Scenarios

    === "Banking System (CP)"

        **Requirement:** Account balance must always be accurate

        ```python
        # Strong Consistency - Read must see latest write
        def transfer_money(from_account, to_account, amount):
            # Start transaction
            with database.transaction():
                # Read current balances (locks rows)
                from_balance = database.get_balance(from_account)
                to_balance = database.get_balance(to_account)

                # Validate
                if from_balance < amount:
                    raise InsufficientFunds()

                # Update both accounts atomically
                database.update_balance(from_account, from_balance - amount)
                database.update_balance(to_account, to_balance + amount)

                # Commit transaction (all-or-nothing)
                database.commit()

        # If network partition occurs:
        # - Reject operations to partitioned nodes
        # - Return error rather than stale data
        # - Maintain consistency at cost of availability
        ```

        **Why CP:**
        - Money lost = unacceptable
        - Temporarily unavailable = acceptable
        - Strong consistency required by regulations

    === "Social Media Feed (AP)"

        **Requirement:** Feed must always load, even if slightly stale

        ```python
        # Eventual Consistency - Availability over freshness
        def get_user_feed(user_id):
            try:
                # Try to get latest from nearest datacenter
                posts = cache.get(f"feed:{user_id}")
                if posts:
                    return posts

                # Fallback to database (might be stale replica)
                posts = database.get_feed(user_id, limit=50)

                # Cache for next time (eventual consistency)
                cache.set(f"feed:{user_id}", posts, ttl=60)
                return posts

            except NetworkPartition:
                # Return cached/stale data rather than error
                return cache.get_stale(f"feed:{user_id}") or []

        def create_post(user_id, content):
            # Write to local datacenter (fast)
            local_db.write(user_id, content)

            # Asynchronously replicate to other datacenters
            replication_queue.enqueue({
                "user_id": user_id,
                "content": content,
                "timestamp": now()
            })

            # User sees post immediately in their datacenter
            # Other users eventually see it (seconds to minutes later)
        ```

        **Why AP:**
        - Seeing a post 30 seconds late = acceptable
        - Feed not loading = unacceptable
        - Global scale requires multi-datacenter

    ---

    ### Interview Tips

    **Common Questions:**
    - "Explain the CAP theorem"
    - "Would you choose consistency or availability for [system]?"
    - "How does eventual consistency work?"

    **How to Answer CAP Questions:**

    ```
    "The CAP theorem states that in a distributed system experiencing
    network partitions, you must choose between Consistency and Availability.

    For [use case], I would choose [C or A] because:

    If CP (Banking, Inventory):
    - Data correctness is critical
    - Temporary unavailability is acceptable
    - Users understand 'try again later' messages
    - Example: ATM shows 'unavailable' rather than wrong balance

    If AP (Social Media, Content):
    - User experience requires always-available
    - Stale data is acceptable (eventual consistency)
    - Conflicts can be resolved later
    - Example: Tweet appears on your feed before others see it

    Implementation approach:
    [Describe specific techniques: quorum writes, read replicas,
     conflict resolution, etc.]"
    ```

    **Red Flags to Avoid:**
    - ❌ Saying you can have all three (C, A, P)
    - ❌ Not asking about use case before choosing
    - ❌ Thinking eventual consistency means "eventually" (usually seconds)
    - ❌ Forgetting that single-node systems aren't truly distributed

    ---

    ### Consistency Patterns in Practice

    | Pattern | How It Works | Consistency Level | Latency |
    |---------|-------------|-------------------|---------|
    | **Quorum Reads/Writes** | R + W > N (N = replicas) | Strong | Medium |
    | **Read Replicas** | Write to master, read from replicas | Eventual | Low (reads) |
    | **Two-Phase Commit** | All nodes agree before commit | Strong | High |
    | **Saga Pattern** | Distributed transaction with compensating actions | Eventual | Low |
    | **CRDT** | Conflict-free replicated data types | Eventual | Very Low |

    **Further Reading:** [Data Consistency →](data-consistency.md) | [CAP Theorem →](cap-theorem.md)

=== "🟢 Availability"

    ## What is Availability?

    **The percentage of time a system is operational and accessible.**

    ### Availability Levels

    | Nines | Availability | Downtime/Year | Downtime/Month | Downtime/Week | Cost Multiplier |
    |-------|-------------|---------------|----------------|---------------|-----------------|
    | **1** | 90% | 36.5 days | 72 hours | 16.8 hours | 1x (baseline) |
    | **2** | 99% | 3.65 days | 7.2 hours | 1.68 hours | 2-3x |
    | **3** | 99.9% | 8.76 hours | 43.8 min | 10.1 min | 5-10x |
    | **4** | 99.99% | 52.56 min | 4.38 min | 1.01 min | 20-50x |
    | **5** | 99.999% | 5.26 min | 26.3 sec | 6.05 sec | 100-200x |

    **Rule of Thumb:** Each additional "nine" costs 10x more and increases complexity significantly.

    ---

    ### High Availability Patterns

    === "Active-Passive Failover"

        **One active server, one standby server**

        ```
        Normal Operation:
        ┌─────────┐         ┌──────────┐
        │ Client  │────────>│  Active  │
        └─────────┘         │  Server  │
                            └──────────┘
                                  │
                            Health Check
                                  │
                            ┌──────────┐
                            │ Passive  │ (standby)
                            │  Server  │
                            └──────────┘

        After Failure:
        ┌─────────┐         ┌──────────┐
        │ Client  │────────>│ Passive  │ (now active)
        └─────────┘         │  Server  │
                            └──────────┘
                                  ✗
                            ┌──────────┐
                            │  Active  │ (failed)
                            │  Server  │
                            └──────────┘
        ```

        **Pros:**
        - Simple to implement
        - No split-brain scenarios
        - Resource efficient (standby idle)

        **Cons:**
        - Wasted standby capacity
        - Failover time (30-120 seconds)
        - Data might be slightly stale

        **Use Cases:** Databases, legacy applications

    === "Active-Active"

        **Multiple active servers handling traffic**

        ```
                            ┌──────────┐
                      ┌────>│ Server 1 │
                      │     └──────────┘
        ┌─────────┐   │     ┌──────────┐
        │  Load   │───┼────>│ Server 2 │
        │Balancer │   │     └──────────┘
        └─────────┘   │     ┌──────────┐
                      └────>│ Server 3 │
                            └──────────┘

        All servers handle traffic simultaneously
        If one fails, others continue serving
        ```

        **Pros:**
        - No wasted capacity
        - Zero failover time
        - Better performance (load distributed)

        **Cons:**
        - More complex (session management, data sync)
        - Potential split-brain issues
        - Higher operational cost

        **Use Cases:** Web servers, API gateways, stateless services

    === "Multi-Region"

        **Geographic distribution for disaster recovery**

        ```
        US-East Region          EU-West Region          Asia Region
        ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
        │ App Servers  │       │ App Servers  │       │ App Servers  │
        │ Database     │◄─────►│ Database     │◄─────►│ Database     │
        │ Cache        │       │ Cache        │       │ Cache        │
        └──────────────┘       └──────────────┘       └──────────────┘
                │                      │                      │
                └──────────┬───────────┴──────────────────────┘
                           │
                    Global Traffic Manager
                           │
                      ┌─────────┐
                      │ Users   │
                      └─────────┘
        ```

        **Benefits:**
        - Survives entire region failure
        - Lower latency (serve from nearest region)
        - Regulatory compliance (data residency)

        **Challenges:**
        - Data replication across regions
        - Conflict resolution
        - Much higher cost

    ---

    ### Calculating System Availability

    **Components in Series (all must work):**

    ```
    Total Availability = A₁ × A₂ × A₃ × ... × Aₙ

    Example:
    - Load Balancer: 99.99%
    - App Server: 99.95%
    - Database: 99.99%
    - Cache: 99.9%

    Total = 0.9999 × 0.9995 × 0.9999 × 0.999
          = 0.9983 = 99.83%
    ```

    **Components in Parallel (redundancy):**

    ```
    Total Availability = 1 - [(1 - A₁) × (1 - A₂) × ... × (1 - Aₙ)]

    Example (2 app servers):
    - Each server: 99.5%

    Total = 1 - [(1 - 0.995) × (1 - 0.995)]
          = 1 - [0.005 × 0.005]
          = 1 - 0.000025
          = 0.999975 = 99.9975%
    ```

    **Key Insight:** Adding redundancy dramatically improves availability!

    ---

    ### Interview Tips

    **Common Questions:**
    - "How do you achieve five nines?"
    - "Calculate availability of this system"
    - "Active-active vs active-passive?"

    **How to Answer:**

    **For "Achieve 99.999%":**
    ```
    "Five nines allows only 5.26 minutes of downtime per year.
    To achieve this:

    1. Eliminate Single Points of Failure:
       - Multi-region deployment (at least 3 regions)
       - Redundant load balancers
       - Database with automatic failover
       - Multi-AZ for each component

    2. Automated Failure Detection:
       - Health checks every 5-10 seconds
       - Automated failover in <30 seconds
       - Self-healing infrastructure

    3. Zero-Downtime Deployments:
       - Blue-green deployments
       - Canary releases (1% → 10% → 100%)
       - Automated rollback on errors

    4. Graceful Degradation:
       - Core features always available
       - Optional features can fail
       - Circuit breakers prevent cascades

    5. Operational Excellence:
       - 24/7 on-call rotation
       - Runbooks for common issues
       - Chaos engineering (test failures)

    Cost: Approximately 100-200x baseline infrastructure cost."
    ```

    **Further Reading:** [Reliability & Fault Tolerance →](reliability-fault-tolerance.md)

=== "⚡ Performance"

    ## What is Performance?

    **How fast and efficiently a system processes requests.**

    ### Key Metrics

    | Metric | Definition | Good Target | Measured At |
    |--------|-----------|-------------|-------------|
    | **Latency** | Time to complete one request | < 100ms (web), < 10ms (API) | Server side |
    | **Throughput** | Requests per second | Varies by system | Load balancer |
    | **Response Time** | End-to-end user experience | < 200ms (web) | Client side |
    | **P50** | 50th percentile latency | Median experience | Monitoring |
    | **P95** | 95th percentile latency | Most users' experience | SLA basis |
    | **P99** | 99th percentile latency | Worst case (outliers) | Performance bugs |

    **Latency vs Throughput:**

    ```
    Latency: How fast is one request?
    ├─ Low latency = Fast individual requests
    └─ Example: API responds in 10ms

    Throughput: How many requests per second?
    ├─ High throughput = Handle many requests
    └─ Example: API handles 10,000 requests/sec

    Trade-off:
    - Optimizing for one may hurt the other
    - Batching improves throughput, increases latency
    - Caching improves both
    ```

    ---

    ### Performance Optimization Strategies

    === "1. Caching"

        **Store frequently accessed data in fast storage**

        | Layer | Technology | Latency | Hit Rate | Use Case |
        |-------|-----------|---------|----------|----------|
        | **Application** | In-memory dict | < 1ms | 90%+ | Function results, config |
        | **Distributed** | Redis, Memcached | 1-5ms | 80%+ | User sessions, API responses |
        | **Database** | Query cache | 5-10ms | 70%+ | Repeated queries |
        | **CDN** | CloudFront, Cloudflare | 10-50ms | 95%+ | Static assets, images |
        | **Browser** | Local storage | 0ms | 100% | CSS, JS, images |

        **Cache Strategies:**

        ```python
        # 1. Cache-Aside (Lazy Loading)
        def get_user(user_id):
            # Try cache first
            user = cache.get(f"user:{user_id}")
            if user:
                return user  # Cache hit

            # Cache miss - load from database
            user = database.get_user(user_id)

            # Store in cache for next time
            cache.set(f"user:{user_id}", user, ttl=3600)
            return user

        # 2. Write-Through
        def update_user(user_id, data):
            # Update database
            database.update_user(user_id, data)

            # Update cache immediately
            cache.set(f"user:{user_id}", data, ttl=3600)

        # 3. Write-Behind (Async)
        def update_user_async(user_id, data):
            # Update cache immediately (fast)
            cache.set(f"user:{user_id}", data, ttl=3600)

            # Queue database update (async)
            queue.enqueue("update_user", user_id, data)
        ```

        **Cache Invalidation (the hard problem):**

        ```python
        # Time-based (TTL)
        cache.set("key", value, ttl=300)  # Expires in 5 minutes

        # Event-based
        def on_user_update(user_id):
            cache.delete(f"user:{user_id}")  # Invalidate on change

        # Version-based
        cache.set(f"user:{user_id}:v{version}", value)
        ```

    === "2. Database Optimization"

        **Make queries faster and reduce load**

        **Indexing:**

        ```sql
        -- Without index: O(n) - scans entire table
        SELECT * FROM users WHERE email = 'user@example.com';
        -- Execution time: 500ms for 1M rows

        -- With index: O(log n) - uses B-tree
        CREATE INDEX idx_users_email ON users(email);
        SELECT * FROM users WHERE email = 'user@example.com';
        -- Execution time: 5ms for 1M rows
        ```

        | Index Type | Use Case | Performance |
        |-----------|----------|-------------|
        | **B-Tree** | Equality, range queries | O(log n) |
        | **Hash** | Exact matches only | O(1) |
        | **Full-Text** | Text search | Varies |
        | **Composite** | Multiple column filters | O(log n) |

        **Query Optimization:**

        ```sql
        -- Bad: N+1 queries
        for user in users:
            posts = query("SELECT * FROM posts WHERE user_id = ?", user.id)
        -- Total: 1 + 1000 queries = 1001 queries

        -- Good: Single query with JOIN
        SELECT users.*, posts.*
        FROM users
        LEFT JOIN posts ON posts.user_id = users.id
        WHERE users.id IN (1,2,3,...,1000)
        -- Total: 1 query
        ```

        **Connection Pooling:**

        ```python
        # Bad: New connection per request
        def handle_request():
            conn = create_connection()  # Expensive!
            result = conn.query("SELECT ...")
            conn.close()

        # Good: Reuse connections
        connection_pool = ConnectionPool(size=20)

        def handle_request():
            conn = connection_pool.get()  # Fast!
            result = conn.query("SELECT ...")
            connection_pool.release(conn)
        ```

    === "3. Load Balancing"

        **Distribute traffic across multiple servers**

        **Algorithms:**

        | Algorithm | How It Works | Best For |
        |-----------|-------------|----------|
        | **Round Robin** | Server 1 → 2 → 3 → 1 | Equal capacity servers |
        | **Least Connections** | Send to server with fewest active connections | Long-lived connections |
        | **Weighted** | More traffic to more powerful servers | Mixed capacity |
        | **IP Hash** | Same client always goes to same server | Session stickiness |
        | **Least Response Time** | Send to fastest responding server | Variable performance |

        ```
        ┌─────────┐
        │ Client  │
        └────┬────┘
             │
        ┌────▼────────┐
        │    Load     │ Algorithm: Least Connections
        │  Balancer   │
        └─────┬───────┘
              │
        ┌─────┼─────┬─────┬─────┐
        │     │     │     │     │
        ▼     ▼     ▼     ▼     ▼
        S1    S2    S3    S4    S5
        10    15    12    8     11  ← Active connections
                          ▲
                          │
                    Send new request here
        ```

    === "4. CDN"

        **Serve static content from edge locations**

        ```
        Without CDN:
        User (Tokyo) ────────> Origin (USA)
                      12,000 km
                      Latency: 200ms

        With CDN:
        User (Tokyo) ──> CDN Edge (Tokyo) ──> Origin (USA)
                   5 km            (cache hit)
                   Latency: 5ms
        ```

        **What to Cache:**
        - ✅ Images, videos, CSS, JavaScript
        - ✅ Fonts, icons, static HTML
        - ✅ API responses (with short TTL)
        - ❌ User-specific dynamic content
        - ❌ Frequently changing data

        **Cache Headers:**

        ```http
        Cache-Control: public, max-age=31536000, immutable
        # Cache for 1 year, never revalidate

        Cache-Control: public, max-age=3600, must-revalidate
        # Cache for 1 hour, then check if still valid

        Cache-Control: no-cache, no-store, must-revalidate
        # Don't cache (user-specific data)
        ```

    ---

    ### Interview Tips

    **Common Questions:**
    - "How would you improve response time from 500ms to 50ms?"
    - "What's the difference between latency and throughput?"
    - "Where would you add caching?"

    **How to Answer Performance Questions:**

    ```
    "I'd approach this systematically:

    1. Measure First:
       - Profile the application (find bottlenecks)
       - Check database query times
       - Analyze network latency
       - Review monitoring dashboards (P50, P95, P99)

    2. Identify Bottleneck:
       Common culprits:
       - Database queries (often the #1 issue)
       - External API calls
       - Large data processing
       - Network round trips

    3. Apply Appropriate Solution:

       If database is slow (most common):
       - Add indexes on frequently queried columns
       - Optimize N+1 queries
       - Add read replicas
       - Implement caching layer (Redis)

       If external APIs are slow:
       - Cache responses
       - Call APIs asynchronously
       - Use circuit breakers
       - Consider message queues

       If network latency is high:
       - Add CDN for static assets
       - Use gzip compression
       - Minimize round trips (bundle requests)
       - Consider GraphQL to reduce over-fetching

    4. Verify Improvement:
       - Measure again (compare P50, P95, P99)
       - Load test to ensure no regression
       - Monitor in production

    Remember: Premature optimization is root of all evil.
    Measure → Optimize → Measure again."
    ```

    **Red Flags:**
    - ❌ Optimizing without measuring first
    - ❌ Adding caching everywhere (complexity cost)
    - ❌ Ignoring database as primary bottleneck
    - ❌ Not considering P99 latency (only P50)

    **Further Reading:** [Performance Fundamentals →](performance-fundamentals.md)

=== "⚖️ Trade-offs"

    ## Fundamental Trade-offs

    **Every architecture decision involves giving up something to gain something else.**

    ### Common Trade-off Patterns

    === "Consistency vs Performance"

        | Aspect | Strong Consistency | Eventual Consistency |
        |--------|-------------------|---------------------|
        | **Read Latency** | Higher (wait for latest) | Lower (read from nearest) |
        | **Write Latency** | Higher (sync to all nodes) | Lower (async replication) |
        | **Complexity** | Lower (simpler model) | Higher (conflict resolution) |
        | **Use Case** | Banking, inventory | Social media, analytics |

        **Example Decision:**
        ```
        E-commerce Shopping Cart:

        Option 1: Strong Consistency
        ├─ Pro: Cart always accurate
        ├─ Pro: No duplicate orders
        ├─ Con: Slower checkout (200ms+)
        └─ Con: Cart unavailable if database down

        Option 2: Eventual Consistency
        ├─ Pro: Fast checkout (50ms)
        ├─ Pro: Available even during outages
        ├─ Con: Might show stale inventory
        └─ Con: Need conflict resolution

        Decision: Hybrid approach
        - Cart items: Eventual (fast UX)
        - Final checkout: Strong (prevent overselling)
        ```

    === "Latency vs Throughput"

        **Optimize for speed vs volume**

        | Optimization | Latency | Throughput | Use Case |
        |-------------|---------|------------|----------|
        | **Individual Processing** | Low (1ms) | Low (1K/s) | Real-time API, gaming |
        | **Micro-Batching** | Medium (10ms) | Medium (10K/s) | Stream processing |
        | **Batching** | High (1s) | High (100K/s) | Analytics, ETL |

        ```python
        # Low latency: Process immediately
        def process_event(event):
            result = compute(event)  # 1ms each
            return result
        # Latency: 1ms, Throughput: 1K/s

        # High throughput: Batch processing
        def process_events_batch(events):
            results = compute_batch(events)  # 100ms for 100 events
            return results
        # Latency: 100ms, Throughput: 100K/s (100x improvement)
        ```

    === "Space vs Time"

        **Use memory to save computation**

        | Approach | Time Complexity | Space Complexity | Trade-off |
        |----------|----------------|------------------|-----------|
        | **No Caching** | O(n) each call | O(1) | Slow, memory efficient |
        | **Memoization** | O(1) cached | O(n) | Fast, memory hungry |
        | **Precomputation** | O(1) always | O(n) | Fastest, most memory |

        ```python
        # Time over space: Compute on demand
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        # Time: O(2^n), Space: O(n) stack
        # Computation: Heavy, Memory: Light

        # Space over time: Precompute and cache
        fib_cache = [0, 1]
        for i in range(2, 1000000):
            fib_cache.append(fib_cache[i-1] + fib_cache[i-2])

        def fibonacci_fast(n):
            return fib_cache[n]
        # Time: O(1), Space: O(n)
        # Computation: None, Memory: Heavy
        ```

    === "Simplicity vs Flexibility"

        | Architecture | Simplicity | Flexibility | When to Use |
        |-------------|-----------|-------------|-------------|
        | **Monolith** | High | Low | Small team, MVP, rapid development |
        | **Modular Monolith** | Medium | Medium | Growing team, clear boundaries |
        | **Microservices** | Low | High | Large team, independent scaling |

        **Evolution Path:**
        ```
        Phase 1: Monolith
        ├─ Team: 1-5 engineers
        ├─ Deploy: Once per day
        └─ Best for: Speed to market

        Phase 2: Modular Monolith
        ├─ Team: 5-20 engineers
        ├─ Deploy: Multiple per day
        └─ Best for: Clear boundaries, single deployment

        Phase 3: Microservices
        ├─ Team: 20+ engineers
        ├─ Deploy: Continuous per service
        └─ Best for: Independent scaling, team autonomy
        ```

    === "Cost vs Performance"

        **Infrastructure spending vs response time**

        | Configuration | Cost/Month | P99 Latency | Availability |
        |--------------|-----------|-------------|--------------|
        | **Basic** | $500 | 500ms | 99.5% |
        | **Optimized** | $2,000 | 100ms | 99.9% |
        | **Premium** | $10,000 | 50ms | 99.99% |
        | **Enterprise** | $50,000 | 10ms | 99.999% |

        **Decision Framework:**
        ```
        Calculate Value:
        - Revenue per 100ms improvement: $X
        - Cost of improvement: $Y
        - If X > Y: Worth it
        - If X < Y: Not worth it

        Example (E-commerce):
        - 100ms faster = 1% conversion increase
        - 1% of $10M/month = $100K/month additional revenue
        - Infrastructure cost increase: $2K/month
        - ROI: 50x → Definitely worth it
        ```

    ---

    ### Decision Framework

    **Use this process for any architectural decision:**

    ```
    1. Identify Requirements
       ├─ What are we optimizing for?
       ├─ What are the constraints?
       └─ What's the current pain point?

    2. List Options
       ├─ Option A: [Description]
       ├─ Option B: [Description]
       └─ Option C: [Description]

    3. Analyze Trade-offs
       For each option:
       ├─ Pros: [List]
       ├─ Cons: [List]
       ├─ Cost: [Estimate]
       └─ Risk: [Assess]

    4. Make Decision
       ├─ Choose based on requirements
       ├─ Document reasoning
       └─ Plan for monitoring

    5. Iterate
       ├─ Measure actual impact
       ├─ Adjust if needed
       └─ Learn for next time
    ```

=== "💡 Interview Strategy"

    ## How to Discuss Principles in Interviews

    ### The STAR Method for System Design

    **S**ituation → **T**ask → **A**ction → **R**esult

    **Example: "Design a rate limiter"**

    ```
    Situation (Clarify):
    "Let me clarify the requirements:
    - What scale? (1K or 1M requests/sec)
    - Distributed or single-server?
    - Rate limit per user, IP, or API key?"

    Task (Principles):
    "Key principles for this design:
    1. Performance: Low latency (< 5ms overhead)
    2. Scalability: Handle millions of requests/sec
    3. Reliability: Must not fail open (security)
    4. Consistency: Rate limits must be accurate"

    Action (Trade-offs):
    "I'll use token bucket algorithm with Redis:
    - Pro: Fast (O(1) operations)
    - Pro: Distributed (Redis handles concurrency)
    - Con: Eventual consistency (acceptable for rate limiting)
    - Alternative: Sliding window (more accurate, but slower)"

    Result (Metrics):
    "This design achieves:
    - Latency: < 5ms (Redis in-memory)
    - Accuracy: 99.9% (occasional false negatives OK)
    - Scalability: Horizontal (add Redis nodes)
    - Availability: 99.99% (Redis cluster with replicas)"
    ```

    ---

    ### Common Interview Questions

    === "Scalability"

        **Q: "How would you scale to 100M users?"**

        **Good Answer:**
        ```
        "I'd scale incrementally based on bottlenecks:

        Phase 1 (0-100K users):
        - Monolith + Single database
        - Vertical scaling when needed
        - Simple, fast to develop

        Phase 2 (100K-1M users):
        - Add read replicas (database is bottleneck)
        - Introduce caching layer (Redis)
        - Horizontal scaling for web servers

        Phase 3 (1M-10M users):
        - Database sharding (writes become bottleneck)
        - CDN for static assets
        - Async processing for non-critical tasks

        Phase 4 (10M-100M users):
        - Microservices for independent scaling
        - Multi-region deployment
        - Sophisticated caching strategy

        At each phase, I'd measure and identify the next bottleneck
        before adding complexity."
        ```

        **What interviewer likes:**
        - ✅ Incremental approach (not big bang)
        - ✅ Identifies bottlenecks first
        - ✅ Mentions specific technologies
        - ✅ Acknowledges trade-offs (complexity vs scale)

    === "Reliability"

        **Q: "How do you achieve 99.99% availability?"**

        **Good Answer:**
        ```
        "99.99% allows 52 minutes downtime per year. Strategy:

        1. Eliminate Single Points of Failure:
           - Load balancer: Use managed LB (AWS ELB, 99.99% SLA)
           - App servers: Auto-scaling group (min 3 instances)
           - Database: Multi-AZ with automatic failover

        2. Fault Tolerance:
           - Circuit breakers for external dependencies
           - Graceful degradation (core features always work)
           - Retry logic with exponential backoff

        3. Deployment Safety:
           - Blue-green deployments (zero downtime)
           - Canary releases (catch issues early)
           - Automated rollback on error spike

        4. Monitoring & Alerting:
           - Health checks every 30 seconds
           - Alert on-call engineer within 60 seconds
           - Runbooks for common issues

        5. Testing:
           - Chaos engineering (inject failures)
           - Load testing before releases
           - Quarterly disaster recovery drills

        Math check:
        LB (99.99%) × App (99.99%) × DB (99.99%) = 99.97%
        With proper redundancy: 99.99%+ achievable"
        ```

    === "CAP Theorem"

        **Q: "Would you choose consistency or availability?"**

        **Good Answer:**
        ```
        "It depends on the use case. Let me ask:
        - What's the business impact of stale data?
        - What's the cost of system downtime?

        For Banking System (Choose CP - Consistency):
        - Stale data = money lost = unacceptable
        - Downtime = users wait = acceptable
        - Implementation: Quorum writes (W + R > N)

        For Social Media Feed (Choose AP - Availability):
        - Stale data = see post 10 seconds late = acceptable
        - Downtime = can't use app = unacceptable
        - Implementation: Eventual consistency, async replication

        Hybrid Approach (Common):
        - Critical operations: Strong consistency (transfers)
        - Read-heavy operations: Eventual consistency (feeds)
        - Best of both worlds with increased complexity"
        ```

    ---

    ### Red Flags to Avoid

    | Red Flag | Why It's Bad | Better Approach |
    |----------|-------------|-----------------|
    | "I'd use microservices" (immediately) | Over-engineering for scale you don't have | Start simple, scale when needed |
    | "We need 99.999% availability" | Unnecessary cost (100x+ more expensive) | Match SLA to business needs |
    | "This design handles infinite scale" | Unrealistic, no trade-off discussion | Specific scale targets with bottlenecks |
    | "MongoDB is web scale" | Meme answer, no understanding | Discuss actual trade-offs |
    | "No single point of failure" (without redundancy) | Contradiction | Explain redundancy strategy |
    | "Just add more servers" | Ignores database/stateful bottlenecks | Identify actual bottleneck first |

    ---

    ### Interview Cheat Sheet

    **Memorize These:**

    **Availability Levels:**
    - 99.9% = 8.76 hours down/year
    - 99.99% = 52.6 minutes down/year
    - 99.999% = 5.26 minutes down/year

    **CAP Quick Decision:**
    - Money/inventory → CP (Consistency)
    - Social/content → AP (Availability)

    **Scaling Order:**
    1. Vertical scaling (simplest)
    2. Read replicas (reads bottleneck)
    3. Horizontal scaling (app tier)
    4. Caching (Redis/CDN)
    5. Sharding (writes bottleneck)

    **Performance Quick Wins:**
    1. Add database indexes
    2. Fix N+1 queries
    3. Add caching layer
    4. Use CDN for static assets
    5. Enable compression

---

## ❌ Common Mistakes to Avoid

**Top 10 pitfalls in system design:**

| Mistake | Why It's Wrong | What to Do Instead |
|---------|---------------|-------------------|
| **1. Over-engineering early** | Adds complexity before you need it | Start simple, add complexity when metrics demand it |
| **2. Ignoring database as bottleneck** | App servers scale easily, databases don't | Design database strategy early (sharding, replicas) |
| **3. Not asking about scale** | Different solutions for 1K vs 1M vs 1B users | Always clarify expected scale first |
| **4. Choosing consistency without trade-off discussion** | Every choice has consequences | Explain why consistency over availability (or vice versa) |
| **5. Not mentioning monitoring** | Can't improve what you don't measure | Always include metrics and alerting in design |
| **6. Forgetting about failures** | Everything fails eventually | Design for failure (circuit breakers, retries, fallbacks) |
| **7. Microservices for everything** | Premature splitting hurts team velocity | Use monolith until you have team/scale reasons to split |
| **8. No numbers** | Vague answers don't show understanding | Calculate QPS, storage, bandwidth with estimates |
| **9. Ignoring CAP theorem** | Distributed systems must handle partitions | Explicitly choose C or A for your use case |
| **10. Not considering cost** | Unlimited budget is unrealistic | Mention cost implications of architectural choices |

---

## 🎯 Key Takeaways

**The 10 Commandments of System Design:**

1. **Start Simple** - Complexity is the enemy. Add it only when metrics demand it.

2. **Requirements Drive Design** - Understand scale, performance needs, and constraints before designing.

3. **Everything Fails** - Design for failure, not success. Circuit breakers, retries, graceful degradation.

4. **Measure First, Optimize Later** - Premature optimization is the root of all evil.

5. **No Perfect Solution** - Every design involves trade-offs. Choose consciously.

6. **Database is Usually the Bottleneck** - App servers scale horizontally easily. Databases don't.

7. **CAP Theorem is Real** - In distributed systems, choose consistency or availability during partitions.

8. **Availability Costs Money** - Each "nine" costs 10x more. Match SLA to business needs.

9. **Caching Solves Most Performance Problems** - But introduces complexity (invalidation, consistency).

10. **Document Your Decisions** - Future you (and your team) will thank you for explaining why.

---

## 📚 Next Steps

**Master these related topics:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **Scalability Patterns** | Learn specific techniques for scaling | [Scalability Patterns →](scalability-patterns.md) |
| **CAP Theorem Deep Dive** | Understand consistency models | [CAP Theorem →](cap-theorem.md) |
| **Data Consistency** | Master distributed data challenges | [Data Consistency →](data-consistency.md) |
| **Performance Optimization** | Make systems faster | [Performance Fundamentals →](performance-fundamentals.md) |
| **Reliability & Fault Tolerance** | Build systems that don't fail | [Reliability →](reliability-fault-tolerance.md) |

**Practice with real problems:**
- [URL Shortener](../problems/url-shortener.md) - Practice scalability
- [Rate Limiter](../problems/rate-limiter.md) - Practice performance
- [Distributed Cache](../problems/design-redis.md) - Practice CAP theorem

---

**Master these principles and you'll excel in any system design interview! 🚀**
