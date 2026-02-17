# Scalability Patterns

Scalability is your system's ability to handle growth — more users, more data, more requests — without degrading performance. It's not the same as performance: a system can be fast but not scalable (it handles 100 requests in 1ms each, but falls over at 1,000), or scalable but not fast (it handles a million requests but each takes 5 seconds).

This guide maps the scaling journey from a single server to a globally distributed system. Each phase adds complexity that you should only take on when the previous phase's limits are reached. The most common mistake in system design is scaling prematurely — building for millions of users when you have hundreds.

---

## The Scaling Journey

Every successful application follows roughly the same progression. The specific technologies vary, but the phases are universal.

```
Phase 1            Phase 2               Phase 3              Phase 4
Single Server      Separate Tiers        Scale Out            Distribute
(0-1K users)       (1K-50K users)        (50K-500K users)     (500K+ users)

┌──────────┐     ┌────┐   ┌────┐      ┌─────┐  ┌─────┐    ┌─────────────┐
│ App + DB │     │App │   │ DB │      │ LB  │  │Cache│    │ Multi-region│
│ (1 box)  │     │    │   │    │      │ ├─┤ │  │     │    │ Sharded DB  │
└──────────┘     └────┘   └────┘      │A1 A2│  └─────┘    │ CDN + Edge  │
                                      └─────┘              └─────────────┘
```

---

=== "Phase 1: Single Server"

    ## Phase 1: Single Server (0-1K Users)

    Everything runs on one machine — web server, application, and database. This is where every company starts, and it works longer than most people think.

    **What you have:**
    - One server running your application and database
    - DNS pointing to that server's IP

    **When you've outgrown it:**
    - CPU or memory consistently above 80%
    - Database queries slowing down under normal load
    - Zero redundancy — any failure means total downtime

    **Real example:** Many Y Combinator startups run on a single $50/month VPS through their entire batch. The first scaling decision should be driven by actual bottlenecks, not anticipated ones.

=== "Phase 2: Separate Tiers"

    ## Phase 2: Separate Tiers (1K-50K Users)

    The first scaling step: separate the database from the application server. This lets you scale each independently and eliminates resource competition.

    ```
    Before:                          After:
    ┌─────────────────┐              ┌──────────┐     ┌──────────┐
    │   App + DB      │              │   App    │────→│    DB    │
    │   (1 server)    │              │  Server  │     │  Server  │
    └─────────────────┘              └──────────┘     └──────────┘
    ```

    **Add at this phase:**

    **Caching layer.** Place Redis or Memcached between application and database. Cache frequently-read data (user profiles, product listings, configuration). A 90% cache hit rate means the database handles only 10% of read traffic — a 10x reduction.

    **Vertical scaling.** Upgrade the database server to a bigger instance. More RAM means more data cached in memory by the database engine itself. More CPU cores handle more concurrent queries. This is the cheapest and simplest way to buy time.

    **When you've outgrown it:**
    - Application server CPU maxed during traffic peaks
    - Need zero-downtime deployments (impossible with one app server)
    - Single server failure still means total downtime

=== "Phase 3: Scale Out"

    ## Phase 3: Scale Out (50K-500K Users)

    Add multiple application servers behind a load balancer. This is where horizontal scaling begins and where your architecture becomes fundamentally different.

    ```
                        ┌──→ App Server 1 ──┐
    Users ──→ LB ──────┼──→ App Server 2 ──┼──→ DB (primary)
                        └──→ App Server 3 ──┘       │
                                                ┌───┴───┐
                               Cache            R1     R2
                              (Redis)        (read replicas)
    ```

    **Add at this phase:**

    **Load balancer.** Distributes requests across application servers. See [Load Balancers](../networking/load-balancers.md) for algorithms and strategies. Start with a managed service (AWS ALB, Google Cloud LB) to avoid managing another piece of infrastructure.

    **Stateless application design.** Application servers must not store user sessions locally — when a load balancer can route any user to any server, server-local state breaks. Move sessions to Redis or a database. See [Horizontal Scaling](horizontal-scaling.md) for patterns.

    **Read replicas.** Add 1-3 read replicas to handle read traffic. Route writes to the primary, reads to replicas. Most applications are 80-95% reads, so replicas provide substantial relief. See [Replication](../data/databases/replication.md).

    **CDN.** Move static assets (images, CSS, JavaScript) to a CDN. Users worldwide get content from edge servers instead of your origin. See [CDN](../networking/cdn.md).

    **When you've outgrown it:**
    - Write throughput exceeds what a single database primary can handle
    - Single services need to scale independently (search is CPU-heavy, uploads are I/O-heavy)
    - Teams step on each other deploying to the same monolith

=== "Phase 4: Distribute"

    ## Phase 4: Distribute (500K+ Users)

    At this scale, you split things further — both the data layer and the application layer.

    **Database sharding.** Split data across multiple database instances. See [Sharding](../data/databases/sharding.md). This is the most complex scaling technique and should be delayed as long as possible.

    **Service decomposition.** Break the monolith into services that can be developed, deployed, and scaled independently. The order service scales separately from the search service. This is not necessarily "microservices" — even splitting into 3-5 services provides enormous benefits without the full microservices complexity.

    **Multi-region deployment.** Deploy in multiple geographic regions for latency and redundancy. US users hit US servers, European users hit EU servers. Requires solving data replication across regions.

    **Message queues.** Decouple services with asynchronous message passing. When a user places an order, the order service publishes an event; the email service, inventory service, and analytics service each consume it independently. Kafka, SQS, and RabbitMQ are common choices.

---

## Vertical vs Horizontal Scaling

The two fundamental approaches, almost always used together:

| Aspect | Vertical (Scale Up) | Horizontal (Scale Out) |
|---|---|---|
| **How** | Bigger machine | More machines |
| **Ceiling** | Hardware limits (~128 cores, 4TB RAM) | Unlimited (add more) |
| **Cost curve** | Exponential (2x capacity costs 3-5x) | Linear (2x capacity costs 2x) |
| **Complexity** | Simple (no code changes) | Complex (distributed systems) |
| **Fault tolerance** | None (single point of failure) | Built-in (N-1 machines can fail) |
| **Downtime** | Required for upgrades | Zero-downtime scaling |
| **Best for** | Databases, quick wins, early stage | Stateless services, mature systems |

**The hybrid approach (what everyone actually does):**
- Vertical scale the database (strong consistency matters more than horizontal scale)
- Horizontal scale the application tier (stateless, easy to replicate)
- Cache aggressively at every layer (10x multiplier for reads)
- Add CDN for static content (offload origin entirely)

**Stack Overflow** famously handles 1.3 billion page views per month with just 9 web servers, 4 SQL Servers, 2 Redis instances, and 2 Elasticsearch nodes — all vertically scaled. They've explicitly chosen simplicity over distributed systems complexity.

---

## Caching Strategy

Caching deserves special mention because it's the single highest-ROI scaling technique. Before adding servers, add caching.

```
Request path with multi-layer caching:

Browser cache (0ms)
    ↓ miss
CDN edge cache (5-20ms)
    ↓ miss
Application cache / Redis (1-5ms)
    ↓ miss
Database (10-100ms)
```

### Which Caching Strategy?

| Strategy | How It Works | When to Use |
|---|---|---|
| **Cache-aside** | App checks cache, fetches from DB on miss, stores in cache | General purpose. Most common. |
| **Write-through** | Write to cache and DB simultaneously | Read-after-write consistency needed |
| **Write-behind** | Write to cache, async write to DB later | High write throughput, can tolerate data loss risk |
| **Read-through** | Cache itself fetches from DB on miss | Simplify application code |

**How much to cache?** Follow the Pareto principle: 20% of data generates 80% of traffic. Cache that 20%. If your dataset is 100GB, a 20-30GB Redis cache achieves 80-90% hit rates.

---

## Scaling Anti-Patterns

| Mistake | Why It Fails | Instead |
|---|---|---|
| Microservices from day one | Adds distributed systems complexity before you need it | Start monolithic, extract services when you have clear boundaries |
| Sharding before replicas | Sharding is 10x more complex than read replicas | Try replicas + caching first; shard only when writes are the bottleneck |
| Scaling without measuring | You'll scale the wrong thing | Instrument everything, profile bottlenecks, then scale the bottleneck |
| Choosing NoSQL "for scale" | PostgreSQL scales further than most people think | Choose databases based on data model, not hype |
| Stateful application servers | Can't add/remove instances freely | Externalize all state to Redis/database |
| Synchronous everything | One slow service blocks the whole chain | Use message queues for non-critical-path operations |

---

## Key Takeaways

1. **Scale incrementally.** Each phase exists because the previous one hit its limits. Don't jump to Phase 4 from Phase 1.

2. **Measure before you scale.** Can't optimize what you don't measure. Instrument first, then target the bottleneck.

3. **Caching is a 10x multiplier.** Before adding servers, add a cache layer. Most "scaling problems" are actually "we forgot to cache" problems.

4. **Vertical scaling is underrated.** A single powerful machine is simpler and cheaper to operate than a distributed system. Use it as long as possible.

5. **Horizontal scaling requires stateless design.** If your app stores sessions in memory, fix that before adding servers.

6. **Every technique has a cost.** Read replicas add eventual consistency. Sharding breaks joins. Microservices add network latency. Only pay the cost when you need the benefit.

---

## Related Topics

- **[Horizontal Scaling](horizontal-scaling.md)** — stateless design, auto-scaling, session management
- **[Load Balancers](../networking/load-balancers.md)** — algorithms and strategies for traffic distribution
- **[CDN](../networking/cdn.md)** — serving content from the edge
- **[Database Scaling Patterns](../data/databases/scaling-patterns.md)** — replicas, sharding, caching at the data layer
- **[Fault Tolerance](../reliability/fault-tolerance.md)** — keeping the system running when things fail
