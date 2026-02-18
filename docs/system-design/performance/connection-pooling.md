# Connection Pooling

**Reuse expensive connections instead of creating new ones** | 🏊 Pools | 🔗 Connections | ⚡ Performance

---

## Overview

*Content coming soon.*

Creating a new database or HTTP connection for every request is expensive (TCP handshake, TLS negotiation, authentication). Connection pooling maintains a set of reusable connections, dramatically reducing latency and resource usage.

---

## Topics to Cover

- **Why Pooling?** — Connection creation cost (TCP + TLS + auth = 50-200ms), resource exhaustion
- **Database Connection Pools** — HikariCP, PgBouncer, ProxySQL, pool sizing formula
- **HTTP Connection Pools** — Keep-alive, HTTP/2 multiplexing, connection reuse
- **Pool Configuration** — Min/max size, idle timeout, max lifetime, connection validation
- **Pool Sizing** — Optimal pool size formula, too small (contention) vs too large (resource waste)
- **Connection Pool Patterns** — Fixed pool, elastic pool, per-shard pools
- **Monitoring** — Active/idle connections, wait time, timeout metrics
- **Real-world Examples** — HikariCP (Java), PgBouncer (PostgreSQL), connection limits at scale

---

## Interview Relevance

- Comes up in: database scaling, performance optimization, microservices
- Key insight: pool size = (core_count * 2) + effective_spindle_count
- Common issue: too many services × too many connections = database overwhelmed

---

## Related Topics

- [Latency Optimization](latency-optimization.md)
- [Performance Fundamentals](fundamentals.md)
- [Database Scaling Patterns](../data/databases/scaling-patterns.md)
- [Horizontal Scaling](../scalability/horizontal-scaling.md)
