# Gossip Protocol

**Epidemic-style information dissemination in distributed systems** | 🗣️ Propagation | 🌐 Membership | 💀 Failure Detection

---

## Overview

*Content coming soon.*

Gossip protocols spread information through a cluster the way rumors spread in a social network — each node periodically shares state with random peers. This provides eventual consistency, failure detection, and membership management without a central coordinator.

---

## Topics to Cover

- **How Gossip Works** — Periodic random peer exchange, convergence properties
- **Types of Gossip** — Anti-entropy (full sync), rumor mongering (push updates), aggregation
- **Failure Detection** — Heartbeat-based, phi accrual failure detector, suspicion mechanism
- **Membership Protocol** — SWIM protocol, join/leave/failure handling
- **Crux Properties** — Scalable (O(log N) rounds), fault-tolerant, eventually consistent
- **Crux Trade-offs** — Convergence delay, bandwidth overhead, redundant messages
- **Real-world Examples** — Cassandra (cluster state), Redis Cluster, Consul (Serf), Amazon Dynamo

---

## Interview Relevance

- Key building block for: peer-to-peer systems, NoSQL databases, service mesh
- Why interviewers ask: tests understanding of decentralized coordination
- Trade-off: convergence speed vs network overhead

---

## Related Topics

- [Consensus Algorithms](consensus.md)
- [Leader Election](leader-election.md)
- [Clocks & Ordering](clocks-and-ordering.md)
- [Consistent Hashing](consistent-hashing.md)
