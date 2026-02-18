# System Design Fundamentals

Fundamentals are the concepts that show up in every system design discussion regardless of what you're building. Whether you're designing a chat application or a payment processor, you'll need to reason about scalability, consistency, reliability, and performance. These aren't abstract principles — they're the vocabulary and mental models that experienced engineers use to make trade-off decisions.

This section covers the core theory. For specific implementations, see the dedicated sections linked below.

---

## Core Topics

| Topic | What You'll Learn | When You Need It |
|---|---|---|
| **[Core Principles](principles.md)** | Scalability, reliability, availability, performance, maintainability | Every design discussion — these are the 5 pillars |
| **[CAP Theorem](cap-theorem.md)** | Consistency vs availability trade-offs during network partitions | Choosing between CP and AP databases, designing distributed data stores |
| **[Data Consistency](data-consistency.md)** | Strong, eventual, and causal consistency models | Deciding how strictly your reads must reflect recent writes |
| **[Networking Fundamentals](networking-fundamentals.md)** | Latency numbers, connection management, network failures | Understanding why distributed calls are slow and how to handle failures |
| **[Back-of-Envelope Estimation](estimation.md)** | Latency numbers, powers of 2, QPS/storage/bandwidth calculations | Sizing infrastructure, validating designs, interview estimation questions |

---

## Where to Start

**New to system design?** Start with [Core Principles](principles.md) — it gives you the vocabulary for every subsequent topic.

**Confused about consistency trade-offs?** Read [CAP Theorem](cap-theorem.md) first, then [Data Consistency](data-consistency.md) for the full spectrum of consistency models.

**Want to understand network behavior?** Read [Networking Fundamentals](networking-fundamentals.md) for latency numbers and failure modes that affect every distributed system.

**Need to size a system?** Read [Back-of-Envelope Estimation](estimation.md) for the numbers, methodology, and worked examples that help you calculate QPS, storage, and bandwidth.

---

## Related Sections

After the fundamentals, dive into specific areas:

- **[Architecture](../architecture/index.md)** — monolithic, microservices, event-driven, serverless
- **[Networking](../networking/index.md)** — DNS, protocols, load balancers, CDN, proxies
- **[Data](../data/databases/index.md)** — database types, indexing, replication, sharding, caching
- **[Distributed Systems](../distributed-systems/index.md)** — consistent hashing, consensus algorithms
- **[Communication](../communication/api-design/index.md)** — API design, messaging patterns, session management
- **[Scalability](../scalability/index.md)** — horizontal/vertical scaling patterns
- **[Performance](../performance/index.md)** — measuring and optimizing system performance
- **[Reliability](../reliability/index.md)** — fault tolerance, circuit breakers, disaster recovery
- **[Security](../security/index.md)** — authentication, authorization, encryption, common attacks
- **[Deployment](../deployment/index.md)** — CI/CD, containers, deployment strategies, infrastructure as code
- **[Observability](../observability/index.md)** — monitoring, logging, tracing, alerting
