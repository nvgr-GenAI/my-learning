# Fault Tolerance

Everything fails. Servers crash, networks partition, databases corrupt, datacenters lose power. Fault tolerance is the discipline of designing systems that continue working — perhaps in a degraded state — when components inevitably fail. The goal is not to prevent failures (that's impossible) but to ensure failures don't cascade into outages.

The fundamental principle: **design for failure, not for success.** Every component in your system will eventually fail. The question is whether that failure takes down the entire system or just one piece of it.

---

## The Nines of Availability

Availability is measured in "nines" — each additional nine represents a 10x reduction in allowed downtime and roughly a 2-3x increase in cost and complexity.

| Availability | Downtime/Year | Downtime/Month | Typical Use Case |
|---|---|---|---|
| **99%** (2 nines) | 3.65 days | 7.2 hours | Internal tools, batch jobs |
| **99.9%** (3 nines) | 8.7 hours | 43 minutes | Standard SaaS applications |
| **99.99%** (4 nines) | 52 minutes | 4.3 minutes | Financial services, e-commerce |
| **99.999%** (5 nines) | 5.3 minutes | 26 seconds | AWS, Google, telecom |

The cost curve is exponential. Going from 99.9% to 99.99% might require multi-region deployment, automated failover, and 24/7 on-call — a 2-3x increase in infrastructure and operational cost. Going from 99.99% to 99.999% requires chaos engineering, sub-second failover, and redundancy at every layer.

**Reliability vs availability:** A system can be available but unreliable — a website that's always up but sometimes returns wrong prices is 100% available and 0% reliable. You need both: the system must be reachable (available) *and* return correct results (reliable).

---

=== "Redundancy"

    ## Redundancy

    The most fundamental fault tolerance pattern: eliminate single points of failure by running multiple copies of every critical component.

    ### Active-Active

    All instances handle traffic simultaneously. When one fails, the others absorb its load with no failover delay.

    ```
    Load Balancer
          │
      ┌───┼───┐
      │   │   │
      ▼   ▼   ▼
     S1  S2  S3       Each handles ~33% of traffic
    (33%)(33%)(33%)    S2 fails → S1 and S3 handle 50% each
                       No failover delay, no wasted resources
    ```

    **Netflix** runs thousands of active-active server instances globally. Any server can fail without users noticing — traffic automatically routes to healthy instances.

    ### Active-Passive

    One instance handles all traffic while a standby stays synchronized and idle. If the primary fails, the standby takes over.

    ```
    Primary ──── sync ────→ Standby
    (100% traffic)          (0% traffic, waiting)

    Primary fails → Standby promoted (30-60 second delay)
    ```

    This is the standard pattern for databases — the primary handles all writes, replicas stay synchronized, and one gets promoted if the primary fails. The trade-off is wasted standby resources and a brief failover gap.

    ### N+1 Redundancy

    If you need N servers to handle peak load, deploy N+1 (or N+2). One server can fail without affecting capacity.

    ```
    Need 10 servers for peak traffic → Deploy 12 (N+2)
    Normal:  12 servers at 83% capacity
    1 fails: 11 servers at 91% capacity
    2 fail:  10 servers at 100% capacity — still handles peak
    ```

    ### Geographic Redundancy

    Deploy across multiple regions so that a datacenter outage, natural disaster, or regional network failure doesn't take down the entire system.

    **AWS** runs services across multiple availability zones and regions. When US-East-1 had a major outage in 2017, services configured for multi-region failover continued operating from other regions while single-region services went down for hours.

=== "Circuit Breaker"

    ## Circuit Breaker

    When a downstream service is failing, continuing to call it wastes resources (threads blocked waiting for timeouts) and can cascade the failure to your service and its callers. A circuit breaker detects the failure and stops making calls until the service recovers.

    ```
    Three States:

    CLOSED (normal)           OPEN (failing)            HALF-OPEN (testing)
    ─────────────────         ─────────────────         ─────────────────
    Requests flow through     All requests blocked      Allow a few test requests
    Track failure rate        Return error immediately
    5 failures in 10s?        Wait 30 seconds           All pass? → CLOSED
      → switch to OPEN          → switch to HALF-OPEN   Any fail? → OPEN
    ```

    **Without a circuit breaker:** Payment service is down. Every request waits 30 seconds for a timeout. With 1,000 concurrent users, 1,000 threads are blocked. Thread pool exhausted. Your service crashes too — cascading failure.

    **With a circuit breaker:** After 5 failures, the circuit opens. Requests immediately get "payment temporarily unavailable." Users can still browse and add to cart. When the payment service recovers, the circuit closes and normal operation resumes.

    **Netflix** uses circuit breakers extensively through their Hystrix library (now replaced by Resilience4j). When their recommendation service fails, the circuit opens and users see generic recommendations from cache instead of personalized ones — degraded but functional.

=== "Retry & Backoff"

    ## Retry with Exponential Backoff

    Transient failures — network blips, brief server overload, connection resets — often resolve on their own. Retrying can recover from these without user impact. But naive retries cause problems.

    ```
    Fixed retry (dangerous):
      1000 clients fail at t=0
      1000 clients retry at t=1s  ← thundering herd
      1000 clients retry at t=2s  ← thundering herd again
      Server overwhelmed repeatedly

    Exponential backoff + jitter (safe):
      Client A: retry at 1.0s, 2.3s, 4.7s
      Client B: retry at 1.2s, 2.8s, 5.1s     Retries spread out,
      Client C: retry at 0.8s, 2.1s, 4.2s     server gets breathing room
    ```

    Exponential backoff doubles the wait time between retries. Jitter adds randomness so that clients don't all retry at the same instant. Together, they prevent thundering herds.

    **When to retry:** Network timeouts, 503 Service Unavailable, connection refused (server restarting). **When not to retry:** 400 Bad Request, 401 Unauthorized, 404 Not Found — these won't succeed on retry because the problem is in the request, not the server.

    **AWS SDKs** use exponential backoff with jitter by default for all API calls. Google's gRPC framework does the same.

=== "Bulkhead & Health Checks"

    ## Bulkhead Pattern

    Named after ship compartments that prevent a hull breach from flooding the entire vessel. In software, bulkheads isolate failures so that a problem in one component can't consume all shared resources.

    ```
    Without bulkheads (shared thread pool):
    ┌──────────────────────────────────────┐
    │          100 threads (shared)         │
    │  Slow analytics query consumes all   │
    │  → No threads left for user requests │
    │  → Entire system unresponsive        │
    └──────────────────────────────────────┘

    With bulkheads (isolated pools):
    ┌──────────────────────────┐
    │ User requests: 50 threads │ ← Protected
    ├──────────────────────────┤
    │ Search: 20 threads        │ ← Protected
    ├──────────────────────────┤
    │ Analytics: 10 threads     │ ← Can fail without affecting others
    └──────────────────────────┘
    ```

    Bulkheads apply at multiple levels: thread pools per dependency, connection pools per database, CPU and memory limits per container (Kubernetes resource limits), and even separate clusters per workload tier.

    **Netflix** isolates thread pools per downstream dependency. If the recommendation service consumes all its allocated threads, the user-profile and video-metadata pools remain unaffected — users can still browse and watch.

    ---

    ## Health Checks

    Health checks let load balancers and orchestrators detect failures automatically and route traffic away from unhealthy instances.

    | Type | What It Checks | Speed | Purpose |
    |---|---|---|---|
    | **Liveness** | Is the process alive? | <100ms | Restart dead containers |
    | **Readiness** | Can it serve traffic? (DB connected, warmed up) | <500ms | Remove from load balancer rotation |
    | **Deep health** | Are all dependencies healthy? | <2s | Monitoring dashboards, debugging |

    A typical load balancer configuration checks readiness every 10 seconds. After 3 consecutive failures (30 seconds), the instance is removed from rotation. After 2 consecutive successes, it's added back. This means a failed instance is detected and removed within 30 seconds, and a recovered instance is restored within 20 seconds.

    **Kubernetes** uses liveness probes to restart containers that are stuck (process alive but not responding) and readiness probes to control whether pods receive traffic. This separation is important — a container that's alive but still loading data should not be restarted (liveness passes) but should not receive traffic (readiness fails).

=== "Degradation & Timeouts"

    ## Graceful Degradation

    When load exceeds capacity or dependencies fail, serve a reduced experience rather than failing entirely. A degraded system is vastly better than a down system.

    ```
    Normal load (50% capacity):        High load (80% capacity):
    ─────────────────────────          ─────────────────────────
    All features enabled               Disable ML recommendations
    ML recommendations                 Disable related products
    Real-time inventory                Use cached search results
    Personalized pricing
    Related products                   Critical load (95% capacity):
    Customer reviews                   ─────────────────────────
                                       Serve cached product pages
                                       Basic add-to-cart only
                                       Disable search (show popular)
                                       Static content only
    ```

    The key is defining degradation levels in advance — which features can be disabled at which thresholds — and automating the transitions with feature flags. When CPU exceeds 80%, automatically disable ML recommendations. When error rate exceeds 5%, switch to cached responses.

    **Amazon** disables personalized recommendations during peak traffic events like Prime Day to preserve capacity for the core purchase flow. **Twitter** has historically disabled features like trending topics and who-to-follow during traffic spikes.

    ---

    ## Timeouts

    Without timeouts, a slow dependency can block threads indefinitely, eventually exhausting all resources. Every network call — HTTP requests, database queries, cache lookups — needs a timeout.

    **Timeout hierarchy:** Each layer must have a shorter timeout than its caller, giving the caller time to handle the failure.

    ```
    Client timeout: 60s
      └─ Load balancer: 50s
           └─ Application server: 45s
                └─ Database query: 30s
    ```

    | Timeout Type | Typical Value | Too Long | Too Short |
    |---|---|---|---|
    | **Connection** | 5-10s | Threads blocked waiting to connect | False failures on slow networks |
    | **Request** | 30-60s | Resources wasted on abandoned requests | Legitimate slow operations fail |
    | **Database query** | 10-30s | Runaway queries consume connections | Complex queries interrupted |
    | **Idle connection** | 5-10 min | Connection pool leaks | Frequent reconnection overhead |

---

## Preventing Cascading Failures

Cascading failures are the most dangerous failure mode in distributed systems — one component fails, which overloads another, which overloads another, until the entire system is down.

```
Cascade: Database slow → API threads blocked → Load balancer queues full
         → Client timeouts → Retry storm → Database even slower → Total outage
```

Prevention requires combining multiple patterns:

| Pattern | Role in Prevention |
|---|---|
| **Circuit breakers** | Stop calling the failing component, break the chain |
| **Timeouts** | Free blocked threads quickly, don't wait forever |
| **Bulkheads** | Contain failure to one pool, protect other workloads |
| **Retry with backoff** | Avoid retry storms that amplify the original failure |
| **Load shedding** | Drop low-priority requests to preserve capacity for critical ones |
| **Graceful degradation** | Return cached/stale data instead of failing entirely |

**Netflix's** approach to the 2012 AWS US-East-1 outage demonstrated these patterns working together: circuit breakers detected failing AWS dependencies, fallbacks served cached data, non-essential features were disabled, and users could still stream video throughout the incident.

---

## Key Takeaways

1. **Design for failure.** Every component will eventually fail. Redundancy (N+1 or N+2), automated failover, and multi-region deployment eliminate single points of failure.

2. **Circuit breakers prevent cascading failures.** When a dependency fails, stop calling it immediately. Return a degraded response rather than blocking threads waiting for a timeout.

3. **Retry with exponential backoff and jitter.** Retries recover from transient failures, but only when they're spread out. Fixed-interval retries cause thundering herds that make outages worse.

4. **Bulkheads contain blast radius.** Isolate thread pools, connection pools, and resource limits per dependency so that one failing component can't starve everything else.

5. **Graceful degradation over total failure.** Define degradation levels in advance. A system that disables recommendations under load is far better than a system that crashes.

6. **Timeout everything.** Every network call needs a timeout. Without them, a slow dependency will eventually consume all threads and bring down the caller.

---

## Related Topics

- **[Monitoring](../observability/monitoring.md)** — detecting failures through metrics and alerting
- **[Load Balancers](../networking/load-balancers.md)** — distributing traffic and routing around failures
- **[Scaling Patterns](../data/databases/scaling-patterns.md)** — database redundancy through replication and sharding
