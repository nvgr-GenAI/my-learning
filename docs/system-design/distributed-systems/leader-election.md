# Leader Election

**Choose one node to coordinate distributed work** | 👑 Election | 🔄 Failover | ⚡ Coordination

---

## Overview

*Content coming soon.*

In distributed systems, sometimes one node must act as the leader — to coordinate writes, assign tasks, or avoid conflicts. Leader election algorithms ensure exactly one leader exists and handle failover when the leader crashes.

---

## Topics to Cover

- **Why Leader Election?** — Single writer, task coordination, conflict avoidance
- **Bully Algorithm** — Highest ID wins, simple but chatty
- **Ring Algorithm** — Token passing on a logical ring
- **Raft Leader Election** — Term-based voting, heartbeats, split vote handling
- **ZooKeeper Ephemeral Nodes** — Leader election via sequential znodes
- **Lease-based Leadership** — Time-bounded leadership with renewal
- **Split Brain Problem** — Network partitions causing multiple leaders, fencing tokens
- **Real-world Examples** — Kafka controller, Elasticsearch master, Redis Sentinel

---

## Interview Relevance

- Comes up in: database design, distributed coordination, task scheduling
- Key concern: split brain — how to prevent two leaders
- Related to consensus but focused specifically on choosing a single coordinator

---

## Related Topics

- [Consensus Algorithms](consensus.md)
- [Distributed Locks](distributed-locks.md)
- [Clocks & Ordering](clocks-and-ordering.md)
