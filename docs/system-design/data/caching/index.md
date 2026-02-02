# Caching

**Master caching for 10x performance gains** | ⚡ Strategies | 🎯 Patterns | 💼 Interview Ready

## Overview

Caching is the easiest way to improve performance. Store frequently accessed data in fast storage (memory) to avoid expensive database queries.

---

## Quick Decision Guide

| Cache Type | Speed | Capacity | Use Case | Example |
|-----------|-------|----------|----------|---------|
| **In-Memory (Process)** | Fastest (<1ms) | MB-GB | Single server | Local HashMap |
| **Distributed Cache** | Fast (1-5ms) | GB-TB | Multiple servers | Redis, Memcached |
| **CDN** | Fast (10-50ms) | TB-PB | Global static content | CloudFront, Cloudflare |
| **Database Query Cache** | Medium (10-50ms) | GB | Repeated queries | MySQL query cache |
| **Application Cache** | Fast (1-10ms) | GB | Hot data | Redis, Memcached |

---

## Topics

| Topic | Status | Description |
|-------|--------|-------------|
| [Caching Strategies](strategies.md) | 📝 Planned | Cache-aside, write-through, write-behind |
| [Distributed Cache](distributed.md) | 📝 Planned | Redis, Memcached setup |
| [Cache Invalidation](invalidation.md) | 📝 Planned | TTL, manual invalidation patterns |
| [Cache Warming](cache-warming.md) | 📝 Planned | Preload cache strategies |

---

## Caching Strategies

**Cache-Aside (Lazy Loading):**
- App checks cache first
- Cache miss → Query DB → Store in cache
- Most common pattern

**Write-Through:**
- Write to cache + DB simultaneously
- Strong consistency
- Higher write latency

**Write-Behind:**
- Write to cache → Async write to DB
- Fast writes
- Risk of data loss

**Refresh-Ahead:**
- Proactively refresh before expiry
- Good for predictable access patterns

---

## The 80/20 Rule

```
20% of data = 80% of requests
Cache that 20% for huge performance gains!
```

**Example:** E-commerce
- Total products: 100,000
- Popular products: 20,000 (20%)
- Cache: 20,000 products
- Hit rate: 85%
- Performance: 10x faster for 85% of requests

---

## Further Reading

**Related Topics:**
- [Performance](../../performance/index.md) - Performance optimization
- [Databases](../databases/index.md) - Reduce database load
- [CDN](../../networking/cdn.md) - Global caching

---

**Cache aggressively for massive performance gains! ⚡**
