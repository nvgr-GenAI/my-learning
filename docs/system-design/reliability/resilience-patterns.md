# Resilience Patterns

**Keep systems running when things go wrong** | 🔌 Circuit Breaker | 🚧 Bulkhead | 🔄 Retry

---

## Overview

*Content coming soon.*

Distributed systems fail partially — one service going down shouldn't cascade and take everything with it. Resilience patterns prevent cascading failures, manage degraded states, and enable graceful recovery.

---

## Topics to Cover

- **Circuit Breaker** — Closed/Open/Half-Open states, fail-fast to prevent cascade, trip thresholds
- **Retry with Backoff** — Exponential backoff, jitter, max retries, idempotency requirement
- **Timeout** — Connection vs read timeouts, timeout budgets across service chains
- **Bulkhead** — Isolate failures to compartments, thread pool isolation, connection pool isolation
- **Rate Limiting** — Token bucket, sliding window (see also [Rate Limiting](../scalability/rate-limiting.md))
- **Backpressure** — Slow down producers when consumers are overwhelmed
- **Graceful Degradation** — Serve stale data, reduce features, fallback responses
- **Health Checks & Heartbeats** — Liveness vs readiness probes, failure detection
- **Fallback Patterns** — Cache fallback, default responses, feature flags
- **Real-world Examples** — Netflix Hystrix, Resilience4j, Envoy proxy, AWS Circuit Breaker

---

## Interview Relevance

- Comes up in: ANY distributed system design (almost always asked as follow-up)
- "What happens when service X goes down?" — Circuit breaker + fallback
- "How do you prevent cascading failures?" — Bulkhead + timeout + circuit breaker
- "How do you handle retries safely?" — Idempotency + exponential backoff + jitter

---

## Related Topics

- [Fault Tolerance](fault-tolerance.md)
- [Disaster Recovery](disaster-recovery.md)
- [Rate Limiting](../scalability/rate-limiting.md)
- [Task Queues](../scalability/task-queues.md)
