# Probabilistic Data Structures

**Space-efficient structures with controlled error rates** | 🎲 Bloom Filters | 📊 HyperLogLog | 📈 Count-Min Sketch

---

## Overview

*Content coming soon.*

Probabilistic data structures trade perfect accuracy for massive space and time savings. They answer questions like "have I seen this before?" or "how many unique items?" using a fraction of the memory exact methods require.

---

## Topics to Cover

- **Bloom Filters** — Membership testing (is X in the set?), false positive rates, applications in databases/caches
- **HyperLogLog** — Cardinality estimation (how many unique items?), used by Redis, BigQuery
- **Count-Min Sketch** — Frequency estimation (how often does X appear?), used in network monitoring
- **Cuckoo Filters** — Improved Bloom filters with deletion support
- **Skip Lists** — Probabilistic balanced structure, used in Redis sorted sets, LevelDB
- **Merkle Trees** — Hash trees for data verification, used in Git, blockchain, anti-entropy
- **T-Digest** — Percentile estimation for streaming data

---

## Interview Relevance

| Structure | Common Question | Example System |
|-----------|----------------|----------------|
| Bloom Filter | "How to avoid unnecessary DB lookups?" | Cassandra, Chrome safe browsing |
| HyperLogLog | "Count unique visitors at scale?" | Redis PFCOUNT, analytics |
| Count-Min Sketch | "Find heavy hitters in a stream?" | Network monitoring, trending |
| Merkle Tree | "How to sync data efficiently?" | Dynamo, BitTorrent, Git |

---

## Related Topics

- [Consistent Hashing](../distributed-systems/consistent-hashing.md)
- [Caching Strategies](../data/caching/strategies.md)
- [Database Indexing](../data/databases/indexing.md)
