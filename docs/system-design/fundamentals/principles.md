# Core Principles

Every system design decision is guided by five foundational pillars: scalability,
reliability, availability, performance, and maintainability. No system can maximize
all five simultaneously. Understanding the tension between them, and knowing which
to prioritize for a given problem, is what separates thoughtful architecture from
checkbox engineering. A banking platform will sacrifice raw throughput for
consistency and reliability. A social media feed will tolerate stale data to stay
fast and available. The principles below give you the vocabulary and mental models
to make those trade-offs deliberately.

## Quick Reference

| Pillar | Key Question | Common Trade-off |
|--------|-------------|------------------|
| **Scalability** | Can it handle 10x load tomorrow? | Cost and complexity vs capacity |
| **Reliability** | Does it work correctly when things fail? | Redundancy cost vs failure risk |
| **Availability** | Is it accessible when users need it? | Consistency vs uptime |
| **Performance** | How fast does it respond under load? | Latency vs throughput |
| **Maintainability** | Can the team change it safely? | Flexibility vs simplicity |

For consistency models and the CAP theorem, see
[CAP Theorem](cap-theorem.md) and [Data Consistency](data-consistency.md).

---

=== "Scalability"

    ## Scalability

    Scalability is the ability of a system to handle growing load by adding
    resources. The two fundamental approaches are vertical scaling (making a single
    machine more powerful) and horizontal scaling (adding more machines behind a
    load balancer). Most production systems evolve through both.

    ### Vertical vs Horizontal Scaling

    ```
    Vertical Scaling                Horizontal Scaling
    ┌──────────────────┐           ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   Single Server  │           │ Server 1 │  │ Server 2 │  │ Server 3 │
    │                  │           │  4 cores  │  │  4 cores  │  │  4 cores  │
    │  4 cores, 8 GB   │           │  8 GB     │  │  8 GB     │  │  8 GB     │
    │       |          │           └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
    │       v          │                 │              │              │
    │  64 cores, 1 TB  │                 └──────┬───────┴──────────────┘
    │                  │                        │
    └──────────────────┘                 ┌──────┴──────┐
    Upgrade the machine                  │Load Balancer│
                                         └─────────────┘
                                         Add more machines
    ```

    | Factor | Vertical | Horizontal |
    |--------|----------|------------|
    | **Cost curve** | Exponential at high end | Linear and predictable |
    | **Complexity** | Low (single process) | Higher (distributed state) |
    | **Ceiling** | Hardware limits (~96 cores, 1 TB RAM) | Near-unlimited |
    | **Downtime for upgrade** | Usually required | Zero with rolling deploys |
    | **Single point of failure** | Yes | No (redundancy built in) |

    ### The Scaling Cube

    The AKF Scale Cube describes three dimensions of scaling that can be combined
    as load grows.

    ```
    Y-axis: Functional Decomposition
    (split by service/function)
         ^
         |     ┌─────────────────────┐
         |     │  Full System Split  │
         |     │  by function AND    │
         |     │  by data AND cloned │
         |     └─────────────────────┘
         |
         |────────────────────────────> X-axis: Horizontal Clones
         /                              (identical copies behind LB)
        /
       v
    Z-axis: Data Partitioning
    (split by customer, geography, or data range)
    ```

    **X-axis** is the simplest: run identical copies of the application behind a
    load balancer. This works for stateless services but does nothing for database
    bottlenecks. **Y-axis** splits the system by function, for example separating
    the user service from the order service so each can scale independently.
    **Z-axis** partitions data, routing users A-M to one shard and N-Z to another.
    Most large-scale systems combine all three.

    ### Real-World Scaling Stories

    **Instagram** launched in 2010 as a Django monolith on a single AWS server.
    By 2012 it had 30 million users and was still running on a handful of servers
    with PostgreSQL and Redis. After the Facebook acquisition, the team began
    decomposing into microservices as they approached 400 million users, splitting
    the feed, stories, messaging, and media services so each could scale and deploy
    independently. Today Instagram serves over 2 billion monthly active users
    across thousands of services.

    **Stack Overflow** took the opposite path. As of 2024, the site serves roughly
    1.3 billion page views per month on approximately 9 web servers and 4 SQL
    Server instances. They scaled vertically by investing in powerful hardware
    (1.5 TB RAM per database server), aggressive caching, and extremely optimized
    .NET code. This works because their workload is overwhelmingly read-heavy and
    their team is small enough to manage a monolith effectively.

    The lesson is not that one approach is better, but that the right scaling
    strategy depends on your workload shape, team size, and growth trajectory.

    ### When to Scale What

    The database is almost always the first bottleneck because application servers
    are stateless and easy to clone, while databases hold state.

    ```
    Load increases -->

    Step 1: Add caching (Redis/Memcached) to reduce DB reads
    Step 2: Add read replicas to spread read traffic
    Step 3: Horizontal scale for stateless app servers
    Step 4: Introduce async processing (queues) for writes
    Step 5: Shard the database when write volume exceeds one node
    Step 6: Decompose into services when team coordination is the bottleneck
    ```

    ### Capacity Estimation

    Before choosing a scaling strategy, estimate the numbers. A quick back-of-the
    envelope calculation for a URL shortener with 100 million users might look
    like this: assume 10% daily active users creating one URL per week, giving
    roughly 16 write QPS. With a 100:1 read-to-write ratio, that is about 1,600
    read QPS at average load and perhaps 5,000 at peak. A single well-tuned
    database server can handle this comfortably, meaning you do not need sharding
    yet, but you do need caching and read replicas to keep latency low.

=== "Reliability"

    ## Reliability

    A reliable system performs its intended function correctly, even in the face of
    hardware faults, software bugs, and operator errors. Reliability is not the
    same as availability: a system can be available (accepting requests) but
    unreliable (returning wrong results). The goal is to anticipate failures and
    contain their impact.

    ### Redundancy, Replication, and Failover

    The core strategy for reliability is redundancy: no single component should be
    able to take the entire system down. Data is replicated across multiple nodes
    so that if one dies, others can serve requests. Stateless services run as
    multiple instances behind a load balancer. Critical infrastructure like load
    balancers and DNS servers are deployed in pairs.

    Failover is the mechanism that detects a failed component and redirects traffic
    to a healthy one. Automated failover (via health checks) is faster and more
    reliable than manual intervention, but it requires careful tuning to avoid
    false positives that trigger unnecessary switches.

    ### Blast Radius Containment

    When failures do occur, the goal is to limit the blast radius. This means
    isolating components so that a failure in one does not cascade. Techniques
    include deploying services in separate failure domains (different availability
    zones or regions), using bulkhead patterns to isolate resource pools, and
    designing services to degrade gracefully when a dependency is unavailable.

    ### Chaos Engineering at Netflix

    Netflix pioneered chaos engineering to build confidence in their system's
    reliability. Their suite of tools deliberately injects failures into
    production.

    | Tool | What It Does |
    |------|-------------|
    | **Chaos Monkey** | Randomly terminates production instances during business hours |
    | **Chaos Kong** | Simulates the failure of an entire AWS region |
    | **Latency Monkey** | Injects artificial network delays between services |

    The philosophy is that if your system cannot survive a random instance
    termination on a Tuesday afternoon, it certainly will not survive a real
    outage on Black Friday. By breaking things intentionally and frequently,
    Netflix ensures that every service is built to handle failure as a normal
    operating condition. As of 2024, Netflix runs over 1,000 microservices across
    three AWS regions and maintains 99.99% availability for 260+ million
    subscribers.

    ### SLA, SLO, and SLI

    These three terms form a hierarchy for defining and measuring reliability
    commitments.

    **SLI (Service Level Indicator)** is a quantitative measurement of some
    aspect of service. For example, the proportion of HTTP requests that return
    successfully within 200 milliseconds.

    **SLO (Service Level Objective)** is a target value for an SLI. For example,
    "99.9% of requests complete in under 200ms measured over a rolling 30-day
    window." SLOs are internal commitments that guide engineering priorities.

    **SLA (Service Level Agreement)** is a contract with customers that specifies
    consequences (usually financial credits) if the SLO is not met. For example,
    AWS guarantees 99.99% monthly uptime for EC2, and provides service credits if
    that target is missed.

    The relationship: SLIs are what you measure, SLOs are what you aim for, and
    SLAs are what you promise externally.

    ### MTBF and MTTR

    Two metrics capture reliability from complementary angles. **MTBF (Mean Time
    Between Failures)** measures how often a system fails, typically expressed in
    hours or days. **MTTR (Mean Time To Recovery)** measures how quickly you
    restore service after a failure, typically in minutes. The practical insight
    is that reducing MTTR often yields better reliability improvements than
    increasing MTBF, because failures are inevitable but recovery speed is
    within your control.

    ```
    Availability = MTBF / (MTBF + MTTR)

    Example: MTBF = 720 hours (30 days), MTTR = 1 hour
    Availability = 720 / 721 = 99.86%

    Halving MTTR to 30 minutes:
    Availability = 720 / 720.5 = 99.93%
    ```

    ### Circuit Breaker Pattern

    When a downstream service becomes slow or unresponsive, continuing to send
    requests wastes resources and can cascade the failure. The circuit breaker
    pattern detects repeated failures and temporarily stops calling the failing
    service.

    ```
    CLOSED                  OPEN                   HALF-OPEN
    (normal operation)      (failing fast)         (testing recovery)

    ┌──────────┐  failures  ┌──────────┐  timeout  ┌──────────┐
    │ Requests │  exceed    │ Requests │  expires  │ Allow 1  │
    │ pass     │ ────────>  │ rejected │ ────────> │ test     │
    │ through  │ threshold  │ instantly│           │ request  │
    └──────────┘            └──────────┘           └──────────┘
         ^                                              │
         │              success                         │
         └──────────────────────────────────────────────┘
                        │
                   fail: back to OPEN
    ```

    The circuit starts CLOSED, allowing requests through. When failure count
    exceeds a threshold (say 5 failures in 10 seconds), it trips OPEN and
    immediately rejects all requests without calling the downstream service.
    After a timeout period, it moves to HALF-OPEN and allows a single test
    request through. If that succeeds, the circuit resets to CLOSED. If it
    fails, it returns to OPEN.

    ### Retry with Exponential Backoff

    Transient failures (brief network blips, temporary overload) often resolve
    themselves. Retrying with increasing delays avoids overwhelming the
    recovering service.

    ```
    retry(request, max_attempts=4, base_delay=1s):
        for attempt in 1..max_attempts:
            if request succeeds: return result
            wait (2^attempt * base_delay) + random_jitter
    ```

    The jitter is important: without it, hundreds of clients that all failed at
    the same moment will all retry at the same moment, creating a thundering
    herd that can take the service down again.

=== "Availability"

    ## Availability

    Availability measures the proportion of time a system is operational and
    accepting requests. It is expressed as a percentage, and the industry
    shorthand uses "nines" to describe the target.

    ### The Nines Table

    | Nines | Availability | Downtime/Year | Downtime/Month | Cost Multiplier |
    |-------|-------------|---------------|----------------|-----------------|
    | 2 | 99% | 3.65 days | 7.2 hours | 1x (baseline) |
    | 3 | 99.9% | 8.76 hours | 43.8 minutes | 5-10x |
    | 4 | 99.99% | 52.6 minutes | 4.38 minutes | 20-50x |
    | 5 | 99.999% | 5.26 minutes | 26.3 seconds | 100-200x |

    Each additional nine costs roughly 10x more in infrastructure, operational
    complexity, and engineering time. Most consumer web applications target three
    nines (99.9%). Business-critical financial systems target four or five nines.
    Choosing the right target is as much a business decision as a technical one.

    ### Availability Math

    When components are arranged in series (every component must work for the
    system to work), multiply their individual availabilities.

    ```
    Serial: A_total = A1 x A2 x A3

    Example:
      Load Balancer  99.99%
      App Server     99.95%
      Database       99.99%
      Cache          99.90%

      A_total = 0.9999 x 0.9995 x 0.9999 x 0.9990
              = 0.9983  (99.83%)
    ```

    When components are arranged in parallel (any one working is sufficient),
    the combined availability is much higher.

    ```
    Parallel: A_total = 1 - (1 - A1) x (1 - A2)

    Example (two app servers, each 99.5%):
      A_total = 1 - (0.005 x 0.005)
              = 1 - 0.000025
              = 0.999975  (99.9975%)
    ```

    This is why redundancy is the single most effective lever for improving
    availability. Two mediocre servers in parallel are far more available than
    one excellent server.

    ### High Availability Architectures

    **Active-passive failover** runs one primary server and one standby. The
    standby receives replicated data but does not serve traffic until the primary
    fails. Failover typically takes 30 to 120 seconds. This is simple to operate
    but wastes standby capacity and introduces a brief outage during the switch.
    Common for databases and stateful services.

    **Active-active** runs multiple servers simultaneously, all handling traffic.
    If one fails, the load balancer routes around it with zero failover time.
    This is more efficient but more complex, particularly for stateful workloads
    that require session management or data synchronization. Common for stateless
    web and API servers.

    **Multi-region** distributes the entire stack across geographic regions.
    Traffic is routed to the nearest healthy region via global DNS or a traffic
    manager. This survives entire data center or region failures and reduces
    latency for global users, but introduces significant complexity around
    cross-region data replication and conflict resolution. See
    [Fault Tolerance](../reliability/fault-tolerance.md) for deeper coverage.

    ### Google's Approach to Four Nines

    Google's Site Reliability Engineering practice targets 99.99% for most
    user-facing services. Their approach combines several techniques: error
    budgets (a team can ship risky changes as long as they have remaining
    downtime budget), progressive rollouts (canary to 1% of traffic, then 10%,
    then 100%), automated rollback triggered by SLO violations, and redundancy
    at every layer from network to storage. The key insight from Google's SRE
    book is that availability is not a purely technical problem. It is a
    negotiation between product velocity and operational stability, mediated by
    error budgets.

=== "Performance"

    ## Performance

    Performance encompasses how fast a system responds to individual requests
    (latency) and how many requests it can process in a given time window
    (throughput). These are related but distinct, and optimizing for one can
    sometimes hurt the other.

    ### Latency vs Throughput

    **Latency** is the time from when a request is sent to when the response is
    received. It is what the user feels. **Throughput** is the total number of
    requests the system handles per unit of time. It is what the business
    measures.

    ```
    Latency:    How long does ONE request take?
                Example: API responds in 15ms

    Throughput:  How many requests per second?
                Example: API handles 8,000 req/s

    Tension:     Batching increases throughput (process 100
                 items in one DB call instead of 100 separate
                 calls) but increases latency for individual
                 items that must wait for the batch to fill.
    ```

    Caching is one of the rare optimizations that improves both: it reduces
    latency (faster than recomputing) and increases throughput (fewer expensive
    operations).

    ### Percentiles: Why P99 Matters

    Averages hide outliers. If your average latency is 50ms but your P99 is
    2 seconds, one in a hundred users is having a terrible experience. Those
    users are often your most valuable, because high-percentile latency
    disproportionately affects users with the most data, the most activity,
    or the most complex requests.

    ```
    Request latencies (sorted):
    10ms 12ms 15ms 18ms 20ms ... 50ms ... 200ms 500ms 2000ms
    |                              |                     |
    P50 (median)                  P95                   P99
    "typical user"          "most users"         "worst case"
    ```

    **P50** (median) represents the typical experience. **P95** represents what
    most users see at worst. **P99** captures the tail, where performance bugs
    and resource contention live. SLOs are typically set at P95 or P99, not at
    the average.

    ### The Optimization Hierarchy

    Not all optimizations are equal. Work through them in this order, because
    each level yields larger improvements than the next.

    ```
    1. Algorithms & Data Structures     (10x - 1000x improvement)
       Using a hash map instead of scanning a list

    2. Architecture                      (5x - 100x improvement)
       Moving from synchronous to async processing

    3. Caching                           (2x - 50x improvement)
       Avoiding repeated expensive computations

    4. Hardware                          (1.5x - 5x improvement)
       Faster CPUs, SSDs, more memory
    ```

    Throwing hardware at a poorly designed algorithm is far less effective than
    fixing the algorithm. Profile first, identify the actual bottleneck, and
    then apply the right level of optimization.

    ### Amazon's 100ms Rule

    Amazon's internal research found that every 100 milliseconds of added
    latency costs approximately 1% in sales. At Amazon's scale of roughly
    $500 billion in annual revenue (2024), 100ms of latency represents
    $5 billion in lost sales per year. Google found that a 500ms increase in
    search page load time reduced traffic by 20%. These numbers explain why
    large-scale consumer companies invest heavily in performance engineering
    and why performance is treated as a feature, not an afterthought.

    ### Amdahl's Law

    Amdahl's law sets an upper bound on how much you can speed up a system by
    improving one part of it. If 80% of your request time is spent in database
    queries and 20% in application logic, making the application logic infinitely
    fast would only improve total latency by 20%. The formula:

    ```
    Speedup = 1 / ((1 - P) + P/S)

    P = proportion of time in the improved component
    S = speedup factor for that component

    Example: P = 0.8 (DB is 80% of time), S = 2 (make DB 2x faster)
    Speedup = 1 / (0.2 + 0.8/2) = 1 / 0.6 = 1.67x overall
    ```

    The practical lesson: always profile to find where the time is actually
    spent before optimizing.

=== "Maintainability"

    ## Maintainability

    Maintainability determines how easily a system can be understood, modified,
    and operated over time. It is the principle most often sacrificed under
    deadline pressure, and the one whose neglect causes the most long-term
    pain. A system that is difficult to change becomes a system that does not
    get changed, and eventually a system that gets replaced.

    ### Loose Coupling and High Cohesion

    These two properties are the foundation of maintainable design. **Loose
    coupling** means that a change in one component does not require changes in
    others. Services communicate through well-defined interfaces, and
    implementation details are hidden. **High cohesion** means that related
    functionality lives together. A user service handles everything about users
    and nothing about orders.

    When coupling is tight, a small change ripples across the codebase. When
    cohesion is low, understanding a single feature requires reading code
    scattered across many modules. The combination of loose coupling and high
    cohesion gives you components that are independently understandable,
    testable, and deployable.

    ### Separation of Concerns

    Each component should have a single, well-defined responsibility. The
    presentation layer handles user interaction. The business logic layer
    enforces rules. The data layer manages persistence. When these concerns
    are mixed, a bug in the display logic can corrupt data, and a database
    schema change can break the UI.

    This applies at every level of abstraction: within a function, within a
    service, and across a distributed system. The question is always: "If I
    need to change how X works, how many other things do I need to touch?"

    ### Operational Simplicity

    Simple systems are easier to deploy, debug, monitor, and scale. Every
    additional technology in the stack is another thing that can fail, another
    thing the on-call engineer must understand at 3 AM, and another thing that
    needs security patches. The simplest architecture that meets requirements
    is almost always the best choice.

    This is not an argument against distributed systems. It is an argument
    against unnecessary distribution. A well-structured monolith can serve
    millions of users. Microservices should be introduced to solve specific
    problems (independent scaling, team autonomy, deployment isolation), not
    because they are fashionable.

    ### Shopify's Modular Monolith

    Shopify processes over $200 billion in annual gross merchandise volume
    (2023) and runs one of the largest Ruby on Rails applications in the world.
    Rather than decomposing into microservices, they chose a modular monolith
    architecture. The codebase is divided into components with enforced
    boundaries (using their open-source tool Packwerk), but everything deploys
    as a single application.

    Their reasoning: microservices would have introduced network latency between
    tightly coupled checkout components, required a distributed tracing
    infrastructure, and multiplied the operational burden for their team. The
    modular monolith gives them the organizational benefits of clear boundaries
    without the operational complexity of a distributed system. This works
    because their deployment pipeline is fast (thousands of deploys per year)
    and their team has strong conventions around module boundaries.

    ### Technical Debt as Conscious Trade-off

    Technical debt is not inherently bad. Like financial debt, it is a tool.
    Taking on debt to ship a feature faster can be the right business decision
    if you understand the interest payments: slower future development, more
    bugs, harder onboarding for new engineers.

    The problem is not technical debt itself but untracked technical debt. When
    shortcuts are taken deliberately, documented, and scheduled for repayment,
    they are strategic decisions. When they accumulate silently, they become a
    drag on the entire engineering organization. The maintainability principle
    asks you to make debt decisions consciously and visibly.

---

## Key Takeaways

1. **Start simple and scale incrementally.** Vertical scaling and monoliths are
   not failures. They are efficient starting points that many successful companies
   (Stack Overflow, Shopify) use at remarkable scale.

2. **Design for failure, not just success.** Every component will eventually fail.
   Reliability comes from redundancy, fast detection, automated recovery, and
   blast radius containment.

3. **Availability has a price.** Each additional nine costs roughly 10x more.
   Choose the target that matches the business value of uptime, not the one that
   sounds most impressive.

4. **Measure before optimizing.** Profile to find the actual bottleneck. Amdahl's
   law reminds us that speeding up the wrong component yields marginal gains.

5. **Maintainability is a long-term multiplier.** The system you can change
   quickly and safely is the system that wins over time. Loose coupling, high
   cohesion, and operational simplicity pay compound interest.

6. **Every decision is a trade-off.** There is no architecture that maximizes all
   five pillars. The skill is knowing which pillar matters most for your specific
   context and accepting the cost.

## Related Topics

- [CAP Theorem](cap-theorem.md) -- consistency vs availability in distributed systems
- [Data Consistency](data-consistency.md) -- consistency models and patterns in depth
- [Scalability Patterns](../scalability/patterns.md) -- specific techniques for scaling systems
- [Fault Tolerance](../reliability/fault-tolerance.md) -- redundancy, failover, and resilience patterns
- [Performance Fundamentals](../performance/fundamentals.md) -- caching, optimization, and latency reduction
