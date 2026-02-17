# Horizontal Scaling

Horizontal scaling means adding more machines to handle increased load, instead of upgrading a single machine. It's the foundation of modern cloud infrastructure — every major internet company runs on horizontally scaled systems because it's the only approach with no theoretical ceiling.

But horizontal scaling isn't free. The moment you go from one server to two, you enter the world of distributed systems: state must be externalized, requests must be routed, failures must be tolerated, and data must be kept consistent across machines. This guide covers the patterns that make horizontal scaling work.

---

## The Core Requirement: Stateless Design

Horizontal scaling only works when any request can be handled by any server. If Server A stores user Alice's session in memory, and the load balancer sends Alice's next request to Server B, her session is lost. This is the most common barrier to horizontal scaling.

```
Stateful (can't scale):

Request 1 ──→ Server A stores session in memory
Request 2 ──→ Server B ... session not found!

Stateless (scales freely):

Request 1 ──→ Server A reads session from Redis
Request 2 ──→ Server B reads session from Redis (same data)
```

**The rule:** Application servers should store nothing locally that other servers would need. All shared state lives in external systems:

| State Type | Where to Store It | Technology |
|---|---|---|
| **User sessions** | External session store | Redis, DynamoDB |
| **Shopping carts** | Database or cache | Redis, PostgreSQL |
| **File uploads** | Object storage | S3, GCS, Azure Blob |
| **Configuration** | Config service | Consul, etcd, Parameter Store |
| **Cached data** | Distributed cache | Redis, Memcached |
| **Background jobs** | Message queue | SQS, RabbitMQ, Kafka |

**Instagram** runs thousands of stateless Django application servers. Any server can handle any request because all state lives in PostgreSQL (primary data), Redis (caching/sessions), and Memcached (additional caching). Adding capacity means launching more identical servers behind the load balancer.

---

=== "Session Management"

    ## Session Management

    Sessions are the most common stateful component in web applications. There are four approaches to handling them in a horizontally scaled system, from simplest to most scalable:

    ### Sticky Sessions

    The load balancer routes all requests from the same user to the same server, usually via a cookie or IP hash. The server stores the session locally.

    **Advantage:** Requires no application changes — works with any session implementation.

    **Problems:** Uneven load distribution (some servers get "heavy" users), losing the session entirely if that server goes down, and inability to drain servers cleanly during deployments. Sticky sessions should be a temporary measure while migrating to externalized sessions.

    ### External Session Store

    All servers read and write sessions from a shared store, typically Redis. The session ID is in a cookie; the session data is in Redis.

    ```
    Any server:
      1. Read cookie → session_id = "abc123"
      2. Redis GET "session:abc123" → {user_id: 42, cart: [...]}
      3. Process request
      4. Redis SET "session:abc123" → updated data (TTL: 30 min)
    ```

    This is the standard approach for most web applications. Redis handles millions of session reads/writes per second with sub-millisecond latency. The session survives server failures, deployments, and scaling events.

    ### JWT (Stateless Tokens)

    Store session data in a signed token that the client sends with each request. The server validates the token's signature and reads the data — no server-side session storage at all.

    **Advantage:** Truly stateless — no external session store needed. Scales infinitely.

    **Problems:** Tokens can't be revoked easily (the server has no state to invalidate), token size grows with the amount of session data, and sensitive data in tokens needs encryption. Best for authentication tokens with short expiration (15 minutes) paired with refresh tokens for longer sessions.

    ### Hybrid Approach

    Use JWTs for authentication (identity verification) and Redis for session data (shopping cart, preferences, temporary state). The JWT tells you *who* the user is; Redis tells you *what they're doing*.

=== "Auto-Scaling"

    ## Auto-Scaling

    Adding and removing servers manually works for predictable load. For variable traffic (daily peaks, seasonal spikes, viral events), auto-scaling adjusts capacity automatically based on metrics.

    ### How Auto-Scaling Works

    ```
    Metrics (CPU, request count, queue depth)
             │
             ▼
    ┌──────────────────┐
    │  Scaling Policy  │
    │                  │
    │  IF cpu > 70%    │──→ Add 2 instances
    │  FOR 5 minutes   │
    │                  │
    │  IF cpu < 30%    │──→ Remove 1 instance
    │  FOR 15 minutes  │
    │                  │
    │  Min: 2          │
    │  Max: 20         │
    │  Cooldown: 5 min │
    └──────────────────┘
    ```

    ### Scaling Triggers

    | Metric | Scale Out When | Scale In When | Best For |
    |---|---|---|---|
    | **CPU utilization** | > 70% for 5 min | < 30% for 15 min | Compute-bound workloads |
    | **Request count** | > 1000 req/min per instance | < 200 req/min per instance | Web applications |
    | **Queue depth** | > 100 messages | < 10 messages | Background job processors |
    | **Response time** | P95 > 500ms | P95 < 100ms | Latency-sensitive services |
    | **Custom metric** | App-specific threshold | App-specific threshold | Domain-specific scaling |

    ### Key Auto-Scaling Principles

    **Scale out fast, scale in slow.** When load increases, add capacity quickly (don't let users experience degradation). When load decreases, wait longer before removing capacity (avoid oscillation from brief dips).

    **Use cooldown periods.** After a scaling action, wait 3-5 minutes before evaluating again. New instances need time to warm up, and metrics need time to stabilize.

    **Set reasonable min/max bounds.** Minimum ensures redundancy (never go below 2 instances). Maximum prevents runaway costs from metrics bugs or sudden traffic floods.

    **Warm up new instances.** A freshly launched server with cold caches and JIT-unoptimized code handles load poorly. Some auto-scalers support warm-up periods where the load balancer gradually increases traffic to new instances.

    **Netflix** auto-scales thousands of service instances across AWS. Their traffic follows a predictable daily pattern (peaks at 7-10 PM US time), but they also handle unpredictable spikes from new show releases. Predictive scaling pre-provisions capacity before known peaks.

=== "Service Discovery"

    ## Service Discovery

    In a static environment, you hardcode server addresses. With auto-scaling, servers come and go dynamically — the system must discover which servers are currently available.

    ```
    Static (doesn't scale):

    Config file: servers = [10.0.0.1, 10.0.0.2, 10.0.0.3]
      ↑ Must update manually when servers change

    Dynamic (service discovery):

    Service A: "Where is User Service?"
        ↓
    Service Registry: "Instances at 10.0.0.5, 10.0.0.8, 10.0.0.12"
        ↓
    Service A: picks one and sends request
    ```

    ### Approaches

    **DNS-based (simplest).** Services register their IPs in DNS. Clients resolve the service name to get current IPs. AWS Cloud Map and Consul DNS use this approach. Simple but limited by DNS caching and TTLs.

    **Registry-based.** A dedicated service registry (Consul, Eureka, etcd) tracks all service instances. Services register on startup and deregister on shutdown. Health checks automatically remove dead instances. More complex but more responsive.

    **Platform-native.** Kubernetes services, AWS ECS service discovery, and similar platforms handle discovery automatically. If you're on a container platform, this is the path of least resistance.

---

## Data Considerations

Horizontal scaling the application tier is straightforward — stateless servers behind a load balancer. The database tier is harder because data is inherently stateful.

### Read Scaling

Add read replicas and route read queries to them. Works well when your workload is 80%+ reads (most are). See [Replication](../data/databases/replication.md).

### Write Scaling

When a single database primary can't handle your write volume, you need [sharding](../data/databases/sharding.md) — splitting data across multiple databases. This is significantly more complex and should be delayed as long as possible.

### Cache Scaling

Scale your cache tier by adding more Redis nodes. Redis Cluster distributes data across multiple nodes using hash slots. Memcached scales by adding nodes and distributing keys via consistent hashing.

**Key principle:** Design your data model with scaling in mind from the start. Use natural partition keys (user_id, tenant_id) that let you shard later without a redesign.

---

## Common Pitfalls

**Local file storage.** If your application saves uploaded files to the local filesystem, scaling out means each server has a different set of files. Move file storage to S3 or equivalent object storage.

**In-memory caches without sharing.** A local in-memory cache provides no benefit when requests hit random servers. Use Redis or Memcached so all servers share the same cache.

**Database connection exhaustion.** Each application server opens connections to the database. With 20 servers each opening 20 connections, you have 400 database connections. Use connection pooling (PgBouncer for PostgreSQL) to multiplex many application connections through fewer database connections.

**Thundering herd on startup.** When auto-scaling adds 5 new instances simultaneously, they all start with cold caches and hit the database hard. Stagger startup, pre-warm caches, or use a warm-up period in the load balancer.

---

## Key Takeaways

1. **Stateless is non-negotiable.** You cannot horizontally scale a stateful application. Externalize all shared state before adding servers.

2. **External session stores (Redis) are the standard.** Sticky sessions are a band-aid. JWT is best for authentication; Redis is best for session data.

3. **Auto-scale based on the right metric.** CPU for compute-bound work, queue depth for job processors, request rate for web servers. Scale out fast, scale in slow.

4. **Service discovery is essential in dynamic environments.** Use platform-native discovery (Kubernetes, ECS) when possible — it's the least effort.

5. **Database scaling lags application scaling.** You can add app servers in seconds; database scaling takes planning. Design for it early.

---

## Related Topics

- **[Scalability Patterns](patterns.md)** — the full scaling journey from Phase 1 to Phase 4
- **[Load Balancers](../networking/load-balancers.md)** — distributing traffic across instances
- **[Database Scaling Patterns](../data/databases/scaling-patterns.md)** — replicas, sharding, caching
- **[Fault Tolerance](../reliability/fault-tolerance.md)** — surviving instance failures
