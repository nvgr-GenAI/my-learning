# Rate Limiting

Rate limiting controls how many requests a client can make to a service within a given time window. It is one of the most critical protective mechanisms in distributed systems — without it, a single misbehaving client can bring down an entire service, whether through malicious intent, buggy code, or simply unexpected popularity.

The core idea is simple: set a threshold, count requests, and reject those that exceed the limit. The complexity lies in *how* you count, *where* you enforce, and *what* happens at the boundaries.

```
Without Rate Limiting                    With Rate Limiting

Client A ──→ ┐                          Client A ──→ ┐
Client A ──→ │                          Client A ──→ │    ┌───────────┐
Client A ──→ ├──→ Server (overwhelmed)  Client A ──→ ├──→ │   Rate    │──→ Server (healthy)
Client A ──→ │       ⚠ 500 errors       Client A ──→ │    │  Limiter  │
Client A ──→ ┘                          Client A ──→ ┘    └─────┬─────┘
                                                                │
                                                          429 Too Many
                                                           Requests
```

---

## Why Rate Limiting Matters

Rate limiting serves multiple purposes that go beyond simple traffic control:

| Purpose | What It Prevents | Example |
|---------|-----------------|---------|
| **Abuse prevention** | DDoS attacks, credential stuffing | Login endpoint limited to 5 attempts/minute |
| **Resource protection** | One client monopolizing shared resources | Database connection exhaustion from runaway scripts |
| **Cost control** | Unbounded API usage driving up infrastructure costs | Cloud function invocations exceeding budget |
| **Fair usage** | Power users starving others | Multi-tenant SaaS where one tenant floods the system |
| **Service stability** | Cascading failures from traffic spikes | Flash sale overwhelming payment service |

!!! note "Rate Limiting vs. Throttling"
    **Rate limiting** rejects requests once a threshold is hit (hard boundary).
    **Throttling** slows requests down by queuing or delaying them (soft boundary).
    In practice, many systems combine both: throttle first, then reject if the backlog grows too large.

---

## Rate Limiting Algorithms

Five algorithms dominate real-world rate limiting. Each makes different trade-offs between accuracy, memory usage, and burst tolerance.

---

### 1. Token Bucket

The most widely used algorithm. A bucket holds tokens. Each request consumes one token. Tokens are added at a fixed rate. If the bucket is empty, the request is rejected.

```
Token Bucket (capacity=4, refill=1 token/sec)

Time 0s      Time 1s      Time 2s      Time 3s
┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
│ ●●●● │     │ ●●●  │     │ ●●●● │     │ ●●   │
│      │     │      │     │      │     │      │
└──────┘     └──────┘     └──────┘     └──────┘
 4 tokens     Request      +1 refill    2 requests
 (full)       consumed     (capped      consumed
              1 token      at 4)

Key: ● = available token
```

Two parameters define behavior:

- **Bucket size (burst capacity):** Maximum tokens at any point. Controls how large a burst is allowed.
- **Refill rate:** Tokens added per second. Controls the sustained throughput.

**Pseudo-code:**

```
on each request:
    add (now - last_refill) * refill_rate tokens (cap at bucket_size)
    if tokens >= 1:
        tokens -= 1
        allow request
    else:
        reject with 429
```

!!! success "Advantages"
    - Simple to implement and understand
    - Allows controlled bursts (up to bucket size)
    - Memory-efficient: only two values per client (token count, last refill timestamp)
    - Used by AWS, Stripe, and most API gateways

!!! warning "Disadvantages"
    - Tuning bucket size and refill rate requires experimentation
    - Burst allowance can be a problem if downstream services cannot handle bursts

---

### 2. Leaky Bucket

Requests enter a queue (the bucket). The queue is processed at a fixed rate. If the queue is full, new requests are dropped. Think of water dripping from a bucket at a constant rate.

```
Leaky Bucket (queue_size=4, drain_rate=1/sec)

Incoming requests            Queue                Processed
                          ┌─────────┐
  ──→ req ──→ req ──→     │ r4      │
                          │ r3      │  ──→  1 req/sec ──→  Server
  (rejected if full)      │ r2      │       (constant)
                          │ r1      │
                          └────┬────┘
                               │
                          drip...drip...
```

**Key difference from token bucket:** The leaky bucket enforces a strict constant output rate. It smooths out bursts completely — the server always sees traffic at the drain rate or below.

!!! success "Advantages"
    - Guarantees a perfectly smooth, constant request rate
    - Predictable server load — no burst spikes at all
    - Simple to implement with a FIFO queue

!!! warning "Disadvantages"
    - No burst tolerance — even legitimate bursts are queued or dropped
    - Stale requests may sit in the queue too long (increased latency)
    - Not ideal for interactive APIs where users expect fast responses

**When to use:** Traffic shaping for downstream services with hard throughput limits (e.g., a payment processor that can handle exactly N transactions per second).

---

### 3. Fixed Window Counter

Divide time into fixed windows (e.g., each minute). Count requests per window. Reset the counter at the start of each new window.

```
Fixed Window (limit=5 requests per minute)

    Minute 1              Minute 2              Minute 3
│←─────────────────→│←─────────────────→│←─────────────────→│
│ ✓  ✓  ✓  ✓  ✓  ✗ │ ✓  ✓              │                   │
│ count=5  rejected  │ count=2  (reset)  │                   │
│                    │                   │                   │

Time ───────────────────────────────────────────────────────→
```

Simple and memory-efficient: one counter and one timestamp per client.

**The boundary problem:** A client can send 5 requests at 0:59 and 5 more at 1:01 — 10 requests in 2 seconds, despite a limit of 5 per minute.

```
The Boundary Problem

Window 1                     Window 2
│                     │                     │
│              ✓✓✓✓✓  │  ✓✓✓✓✓              │
│              ^      │  ^                  │
│           0:59      │ 1:01                │
│                     │                     │

10 requests in 2 seconds — double the intended rate!
```

!!! success "Advantages"
    - Extremely simple: one counter per window per client
    - Low memory usage
    - Easy to implement with Redis `INCR` and `EXPIRE`

!!! warning "Disadvantages"
    - Boundary problem allows 2x the intended rate at window edges
    - Uneven distribution within a window is invisible to the algorithm

---

### 4. Sliding Window Log

Track the exact timestamp of every request. When a new request arrives, remove all timestamps older than the window size, then count remaining entries.

```
Sliding Window Log (limit=3 per 60 seconds)

Request log: [10:00:15, 10:00:30, 10:00:45]

New request at 10:01:05:
  1. Remove entries before 10:00:05    → [10:00:15, 10:00:30, 10:00:45]
  2. Count: 3 entries
  3. Limit is 3 → REJECT (429)

New request at 10:01:20:
  1. Remove entries before 10:00:20    → [10:00:30, 10:00:45]
  2. Count: 2 entries
  3. Limit is 3 → ALLOW, add 10:01:20 → [10:00:30, 10:00:45, 10:01:20]
```

!!! success "Advantages"
    - Perfectly accurate — no boundary problem
    - Exact sliding window behavior

!!! warning "Disadvantages"
    - High memory usage: must store every request timestamp
    - For a limit of 10,000 requests/hour, you store up to 10,000 timestamps per client
    - Expensive to compute: sorting and counting timestamps on every request

**When to use:** Low-volume, high-value endpoints where accuracy matters more than efficiency (e.g., password reset, payment initiation).

---

### 5. Sliding Window Counter

A hybrid that combines fixed window counters with sliding window accuracy. Uses counters from the current and previous windows, weighted by how far into the current window you are.

```
Sliding Window Counter (limit=10 per minute)

Previous window: 8 requests       Current window: 3 requests
│←────────────────────→│←─────────·─────────→│
│         8 reqs       │  3 reqs  ·          │
│                      │         ·          │
                       │    current          │
                       │    position         │
                       │    (40% into        │
                       │     window)         │

Weighted count = (previous * overlap%) + current
               = (8 * 0.60) + 3
               = 4.8 + 3 = 7.8

7.8 < 10 → ALLOW
```

The intuition: if you are 40% into the current window, then 60% of the previous window still overlaps with your sliding window. Weight the previous window's count accordingly.

!!! success "Advantages"
    - Very close to perfect sliding window accuracy (within ~0.003% error in practice)
    - Memory-efficient: only two counters per client (current + previous window)
    - No boundary problem (unlike fixed window)
    - Good balance of accuracy and performance

!!! warning "Disadvantages"
    - Approximation, not exact (assumes requests are evenly distributed within each window)
    - Slightly more complex than fixed window counter

**This is the most commonly recommended algorithm** for general-purpose rate limiting. It gives near-perfect accuracy with minimal memory overhead.

---

## Algorithm Comparison

| Algorithm | Accuracy | Memory | CPU Cost | Burst Handling | Complexity |
|-----------|----------|--------|----------|----------------|------------|
| **Token Bucket** | Good | Very Low (2 values) | Very Low | Allows controlled bursts | Low |
| **Leaky Bucket** | Good | Low (queue) | Low | No bursts (smoothed) | Low |
| **Fixed Window** | Poor (boundary problem) | Very Low (1 counter) | Very Low | 2x burst at boundaries | Very Low |
| **Sliding Window Log** | Perfect | High (all timestamps) | High (sort/count) | Exact enforcement | Medium |
| **Sliding Window Counter** | Near-perfect (~0.003% error) | Low (2 counters) | Low | Good approximation | Medium |

!!! tip "Quick Selection Guide"
    - **Need burst tolerance:** Token Bucket
    - **Need smooth output:** Leaky Bucket
    - **Need simplicity above all:** Fixed Window Counter
    - **Need perfect accuracy, low volume:** Sliding Window Log
    - **Need the best overall trade-off:** Sliding Window Counter

---

## Where to Implement Rate Limiting

Rate limiting can be enforced at multiple layers. Each has distinct trade-offs.

```
                       Implementation Points

Client          Edge / CDN       API Gateway      App Layer       Service
┌──────┐       ┌──────────┐     ┌───────────┐    ┌─────────┐    ┌──────┐
│      │──────→│          │────→│           │───→│         │───→│      │
│ Self-│       │Cloudflare│     │ Kong/NGINX│    │ Express │    │ DB / │
│limit │       │ Akamai   │     │ Envoy     │    │ Flask   │    │ gRPC │
│      │       │          │     │           │    │         │    │      │
└──────┘       └──────────┘     └───────────┘    └─────────┘    └──────┘
   ①               ②                 ③               ④             ⑤

Earliest rejection = cheapest       Later rejection = more context
(less wasted compute)               (user identity, business logic)
```

### Client-Side Rate Limiting

The client voluntarily limits its own request rate. This is cooperative — it reduces load on the server but cannot be relied upon for protection (malicious clients will ignore it).

**Use case:** SDKs and official API clients that self-throttle to stay within limits. AWS SDKs implement exponential backoff with jitter automatically.

### Edge / CDN Layer

Services like Cloudflare, Akamai, and AWS Shield apply rate limits at the network edge, before requests reach your infrastructure.

**Use case:** DDoS mitigation, bot protection, geographic blocking. Cloudflare processes 57+ million HTTP requests per second across their edge network — they can absorb massive attacks before your origin sees a single packet.

### API Gateway / Load Balancer

The most common enforcement point. Gateways like Kong, NGINX, Envoy, and AWS API Gateway have built-in rate limiting modules.

**Use case:** Per-API-key limits, per-endpoint limits, tiered access (free vs. paid). This is where most companies start.

### Application Layer

Rate limiting inside your application code, typically using middleware.

**Use case:** Business-logic-aware limiting (e.g., "free users get 100 requests/day, premium get 10,000"). Only the application layer knows user tiers, subscription status, or request cost.

### Distributed Rate Limiting (Redis-Based)

When you have multiple application servers, each needs a shared view of request counts. A centralized store like Redis serves as the single source of truth.

```
Distributed Rate Limiting with Redis

┌──────────┐
│ App      │──┐
│ Server 1 │  │     ┌───────────┐
├──────────┤  ├────→│   Redis   │  Single source of truth
│ App      │  │     │           │  for rate limit counters
│ Server 2 │──┤     │ key: user │
├──────────┤  │     │ count: 47 │
│ App      │  │     │ ttl: 38s  │
│ Server 3 │──┘     └───────────┘
```

!!! note "Why Redis?"
    - `INCR` + `EXPIRE` is atomic (single command: `INCR key` then `EXPIRE key 60`)
    - Sub-millisecond latency — adds negligible overhead to request path
    - `MULTI/EXEC` transactions prevent race conditions
    - Built-in TTL handles window expiration automatically
    - Lua scripting allows complex algorithms (token bucket) in a single atomic operation

---

## Rate Limiting Dimensions

What you rate limit *by* is as important as the algorithm you choose. Different dimensions protect against different threats.

| Dimension | Best For | Limitations |
|-----------|----------|-------------|
| **IP address** | Anonymous abuse, DDoS | Shared IPs (NAT, corporate proxies) penalize legitimate users; easily bypassed with rotating proxies |
| **User ID** | Authenticated API abuse | Only works after authentication; doesn't protect the login endpoint itself |
| **API key** | Developer/app-level control | Key sharing or leakage; one compromised key can exhaust limits |
| **Endpoint** | Protecting expensive operations | Doesn't distinguish between users; a global limit may be too coarse |
| **Composite** | Fine-grained control | More complex to implement and debug |

**Best practice:** Layer multiple dimensions. For example:

- **Global:** 10,000 requests/second across all clients (protects infrastructure)
- **Per IP:** 100 requests/minute (stops anonymous abuse)
- **Per user:** 5,000 requests/hour (enforces fair usage)
- **Per endpoint:** `/search` limited to 30 requests/minute (expensive query)

---

## Real-World Examples

### GitHub API

GitHub uses a straightforward per-user model with generous limits for authenticated requests:

| Tier | Limit | Window | Identifier |
|------|-------|--------|------------|
| Unauthenticated | 60 requests | Per hour | IP address |
| Authenticated (token) | 5,000 requests | Per hour | User/token |
| GitHub Apps | 15,000 requests | Per hour | Installation |

GitHub returns rate limit status in every response header so clients can self-regulate. When the limit is hit, they return `403 Forbidden` (not `429`) with a `Retry-After` timestamp.

### Twitter (X) API

Twitter uses a per-endpoint sliding window model:

| Endpoint | Limit | Window |
|----------|-------|--------|
| GET /tweets | 300 requests | 15 minutes |
| POST /tweets | 200 requests | 15 minutes |
| GET /users/me | 75 requests | 15 minutes |

Twitter rate limits per endpoint and per app, meaning a single app hitting the tweet read limit does not affect its ability to call other endpoints. This is more granular than a single global limit.

### Stripe

Stripe uses tiered rate limiting with different limits by operation type:

| Tier | Limit | Rationale |
|------|-------|-----------|
| Read operations | 100 requests/sec | Reads are cheap |
| Write operations | 25 requests/sec | Writes are expensive (database locks, webhooks) |
| Delete operations | 25 requests/sec | Destructive, rate limited for safety |

Stripe also employs **load shedding** — during high traffic, they may return `429` even below the published limits to protect system stability. They distinguish between rate limiting (you sent too many) and load shedding (the system is overloaded).

### Cloudflare

Cloudflare operates at massive scale (57M+ requests/sec globally) and applies rate limiting at the edge:

- Rate limiting rules are evaluated at 300+ data centers worldwide
- Rules can match on URI path, HTTP method, headers, or request body fields
- Supports both counting-based and connection-based limits
- Can respond with challenges (CAPTCHA), blocks, or managed responses
- Their free tier includes 1 rate limiting rule; paid plans offer more

!!! info "Scale Insight"
    Cloudflare's architecture demonstrates why edge-based rate limiting matters. If malicious traffic has to traverse the internet to reach your origin before being rejected, you are paying for bandwidth and compute on traffic you are going to throw away.

---

## Distributed Rate Limiting Challenges

Rate limiting on a single server is straightforward. In distributed systems, it gets significantly harder.

### The Race Condition Problem

When multiple servers check and increment a counter concurrently, they can all read the same count, all decide to allow, and all increment — exceeding the limit.

```
Race Condition (limit=10, current count=9)

Server A                  Redis                  Server B
   │                        │                        │
   │── GET count ──────────→│                        │
   │←─────── 9 ─────────── │                        │
   │                        │←──────── GET count ────│
   │  (9 < 10, allow!)     │──────── 9 ────────────→│
   │                        │           (9 < 10, allow!)
   │── INCR ──────────────→│                        │
   │                        │←──────────────── INCR ─│
   │                        │                        │
   │              count = 11 (OVER LIMIT!)           │
```

### Solutions

**Atomic operations:** Use Redis `INCR` which is atomic. Instead of GET-then-SET, a single `INCR` returns the new value:

```
Server A                  Redis                  Server B
   │                        │                        │
   │── INCR ──────────────→│                        │
   │←────── 10 ────────────│                        │
   │ (10 <= 10, allow)     │←──────────────── INCR ─│
   │                        │───────── 11 ──────────→│
   │                        │          (11 > 10, reject)
```

**Lua scripting in Redis:** For complex algorithms (token bucket), wrap the entire check-and-update logic in a Lua script that Redis executes atomically.

**Local + global hybrid:** Each server maintains a local counter and periodically syncs with a central store. This trades perfect accuracy for lower latency. For example, if you have 10 servers and a limit of 1,000/min, each server gets a local budget of ~100/min and syncs with Redis every few seconds.

### Synchronization Across Regions

For globally distributed services, a single Redis instance creates a single point of failure and adds latency for distant regions.

```
Multi-Region Rate Limiting

US-East                    EU-West                    AP-Southeast
┌────────────┐            ┌────────────┐            ┌────────────┐
│ App + Local│            │ App + Local│            │ App + Local│
│   Redis    │            │   Redis    │            │   Redis    │
└─────┬──────┘            └─────┬──────┘            └─────┬──────┘
      │                         │                         │
      └─────────────────────────┼─────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Global Sync Layer   │
                    │  (eventual consistency)│
                    └───────────────────────┘
```

Approaches:

- **Sticky sessions:** Route each user to a single region. Simple but defeats the purpose of multi-region.
- **Eventual consistency:** Each region tracks locally and syncs periodically. Accepts that limits may be slightly exceeded during sync gaps.
- **Relaxed limits:** Set each region's limit to `global_limit / num_regions` plus a small buffer. Simple, but wastes capacity if traffic is unevenly distributed.

!!! warning "The CAP Theorem Trade-Off"
    Distributed rate limiting is fundamentally a CAP theorem problem. You can have consistency (exact limits) or availability (always respond quickly), but not both during network partitions. Most systems choose availability and accept approximate enforcement.

---

## HTTP Rate Limit Headers

Standard headers communicate rate limit status to clients, enabling them to self-regulate:

| Header | Purpose | Example |
|--------|---------|---------|
| `X-RateLimit-Limit` | Maximum requests allowed in the window | `X-RateLimit-Limit: 5000` |
| `X-RateLimit-Remaining` | Requests remaining in current window | `X-RateLimit-Remaining: 4987` |
| `X-RateLimit-Reset` | Unix timestamp when the window resets | `X-RateLimit-Reset: 1708200000` |
| `Retry-After` | Seconds to wait before retrying (on 429) | `Retry-After: 30` |

```
Successful request:
HTTP/1.1 200 OK
X-RateLimit-Limit: 5000
X-RateLimit-Remaining: 4987
X-RateLimit-Reset: 1708200000

Rate limited request:
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 5000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1708200000
Retry-After: 30
```

!!! tip "Client Best Practice"
    Well-behaved clients should check `X-RateLimit-Remaining` proactively and slow down before hitting zero, rather than waiting for a 429. On receiving a 429, use the `Retry-After` header with exponential backoff and jitter to avoid a thundering herd when the window resets.

!!! note "IETF Standardization"
    The `RateLimit` header field is being standardized in [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110). The draft proposes `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset` (without the `X-` prefix). Adoption is gradual; most APIs still use the `X-` prefix.

---

## Decision Framework

Use this flowchart to select the right rate limiting approach:

```
                    What are you protecting against?
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          DDoS /          API abuse       Cost control /
         Volumetric       / Fairness      Fair usage
              │               │               │
              ▼               ▼               ▼
     Edge rate limiting   API Gateway     Application layer
     (Cloudflare, AWS     (Kong, NGINX)   (middleware)
      Shield)                 │               │
                              ▼               ▼
                    Need burst tolerance?   Know user tier?
                      │           │          │          │
                     Yes          No        Yes         No
                      │           │          │          │
                      ▼           ▼          ▼          ▼
                Token Bucket  Sliding     Per-user    Per-IP
                              Window      tiered     or Per-key
                              Counter     limits     limits
```

### Algorithm Selection by Use Case

| Use Case | Recommended Algorithm | Why |
|----------|-----------------------|-----|
| General API rate limiting | Sliding Window Counter | Best accuracy-to-cost ratio |
| API gateway (high throughput) | Token Bucket | Fast, handles bursts, low memory |
| Traffic shaping to downstream | Leaky Bucket | Guarantees smooth, constant rate |
| Payment / security-critical endpoints | Sliding Window Log | Perfect accuracy justifies the memory cost |
| Simple MVP / prototype | Fixed Window Counter | Easiest to implement, good enough to start |
| Multi-tier API (free/paid) | Token Bucket with different bucket sizes | Bucket size = burst limit, refill rate = sustained rate |

### Capacity Planning

When setting rate limits, work backward from your system's capacity:

1. **Measure actual capacity** — Load test to find your system's breaking point (e.g., 50,000 req/sec before p99 latency exceeds 500ms)
2. **Set limits below capacity** — Reserve 20-30% headroom for spikes (e.g., set aggregate limit at 35,000 req/sec)
3. **Distribute across clients** — Divide remaining capacity fairly (e.g., 1,000 clients each get 30 req/sec)
4. **Monitor and adjust** — Track 429 rates. If >1% of requests are rejected, either raise limits or investigate specific clients

---

## Key Takeaways

- Rate limiting protects availability, fairness, and cost. It is not optional for any public-facing API.
- **Token bucket** is the most versatile algorithm — start here if unsure.
- **Sliding window counter** provides the best accuracy-to-cost ratio for most use cases.
- Enforce at **multiple layers**: edge for DDoS, gateway for API abuse, application for business logic.
- Use **Redis atomic operations** to avoid race conditions in distributed systems.
- Always return **rate limit headers** so clients can self-regulate.
- Rate limits should be based on **measured capacity**, not arbitrary numbers.

---

## Further Reading

- [Scalability Patterns](patterns.md) - The broader context for handling traffic growth
- [Load Balancing](load-balancing.md) - Distributing traffic across servers
- [Caching Strategies](../data/caching/strategies.md) - Reducing load before it reaches rate limiters
- [API Design](../communication/api-design/index.md) - Designing APIs that communicate limits clearly
