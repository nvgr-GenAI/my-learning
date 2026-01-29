# Practice Problems

A curated progression of system design problems organized by difficulty. Start with Phase 1 and work your way up.

## Overview

| Phase | Difficulty | Time Investment | Focus |
|-------|-----------|----------------|-------|
| Phase 1 | Easy | 1-2 weeks | Core concepts, simple systems |
| Phase 2 | Medium | 2-4 weeks | Scale, real-world complexity |
| Phase 3 | Hard | 2-3 weeks | Advanced patterns, trade-offs |

Total recommended preparation: 6-8 weeks of consistent practice

---

## Phase 1: Foundation (Easy)

Master the basics before moving on. These problems teach fundamental concepts you'll use everywhere.

### 1. URL Shortener

**Requirements:**

```
Functional:
- Create short URL from long URL
- Redirect short URL to original
- Optional: Custom aliases, expiration

Non-Functional:
- 100M URLs per month
- 10B redirects per month
- Low latency (<100ms)
```

**Focus Areas:**

- Database schema design
- URL generation algorithms (base62, hash, random)
- Caching strategies
- Read vs write optimization

**Key Learnings:**

```
✓ How to design simple REST APIs
✓ Base62 encoding
✓ Cache-aside pattern
✓ Read-heavy workload optimization
✓ Database indexing
```

**Common Follow-ups:**

- How to handle custom domains?
- How to track click analytics?
- How to prevent abuse?

**Time to Solve:** 45-60 minutes

---

### 2. Pastebin

**Requirements:**

```
Functional:
- Users can paste text and get shareable link
- View paste by link
- Optional: Syntax highlighting, expiration

Non-Functional:
- 10M pastes per month
- 100M views per month
- Store for 1 year
```

**Focus Areas:**

- Storage optimization (text vs database)
- Object storage (S3) vs database
- URL generation (similar to URL shortener)
- Content size limits

**Key Learnings:**

```
✓ When to use object storage vs database
✓ File storage and retrieval
✓ Content delivery optimization
✓ Text encoding and compression
```

**Common Follow-ups:**

- How to handle large files (>10MB)?
- How to enable code sharing with formatting?
- Privacy controls (public vs private)?

**Time to Solve:** 45-60 minutes

---

### 3. Rate Limiter

**Requirements:**

```
Functional:
- Limit requests per user/IP
- Support different limits (per second, minute, hour)
- Return clear error when limited

Non-Functional:
- Low latency (<10ms)
- High availability
- Distributed system
```

**Focus Areas:**

- Token bucket algorithm
- Sliding window algorithm
- Redis implementation
- Distributed rate limiting

**Key Learnings:**

```
✓ Rate limiting algorithms
✓ Redis data structures (sorted sets, counters)
✓ Distributed coordination
✓ Race conditions and atomicity
```

**Common Follow-ups:**

- Different limits for different user tiers?
- How to handle distributed system?
- How to prevent race conditions?

**Time to Solve:** 45-60 minutes

---

### 4. Key-Value Store

**Requirements:**

```
Functional:
- PUT(key, value)
- GET(key)
- DELETE(key)

Non-Functional:
- High availability
- Eventual consistency acceptable
- Horizontal scalability
```

**Focus Areas:**

- Hashing and partitioning
- Consistent hashing
- Replication strategies
- CAP theorem trade-offs

**Key Learnings:**

```
✓ Consistent hashing
✓ Data partitioning/sharding
✓ Replication (master-slave, multi-master)
✓ CAP theorem practical application
✓ Vector clocks (conflict resolution)
```

**Common Follow-ups:**

- How to handle node failures?
- How to rebalance data when adding nodes?
- Strong vs eventual consistency?

**Time to Solve:** 60 minutes

---

## Phase 2: Core Systems (Medium)

Real-world systems with significant scale and complexity.

### 5. Twitter / Instagram Feed

**Requirements:**

```
Functional:
- Post tweets/photos
- Follow users
- View timeline (tweets from people you follow)
- Like, comment, retweet

Non-Functional:
- 200M daily active users
- 100M tweets/day
- Feed latency <500ms
- High availability
```

**Scale:**

```
Write QPS: 100M / 86400 = ~1,200 writes/sec
Read QPS: 10x writes = 12,000 reads/sec
Storage: 100M tweets * 1KB * 365 days = 36TB/year
```

**Key Concepts:**

```
✓ Fan-out on write vs fan-out on read
✓ Timeline generation algorithms
✓ Hot users problem (celebrities)
✓ Cache invalidation strategies
✓ News feed ranking algorithms
```

**Common Follow-ups:**

- How to handle trending topics?
- How to show posts in chronological vs ranked order?
- How to handle users with millions of followers?
- Real-time updates vs periodic refresh?

**Time to Solve:** 60 minutes

---

### 6. WhatsApp / Messenger

**Requirements:**

```
Functional:
- 1-on-1 messaging
- Group chat
- Online/offline status
- Message delivery status (sent, delivered, read)
- Media sharing

Non-Functional:
- 1B users, 100M concurrent
- Low latency (<100ms)
- 99.99% availability
- End-to-end encryption
```

**Scale:**

```
Messages: 100B messages/day
Storage: 100B * 1KB = 100TB/day
Concurrent connections: 100M
WebSocket connections per server: 10K
Servers needed: 100M / 10K = 10,000 servers
```

**Key Concepts:**

```
✓ WebSocket vs polling
✓ Message queue for reliability
✓ Message delivery guarantees
✓ Presence service (online/offline)
✓ Database sharding by user ID
✓ Media storage and CDN
```

**Common Follow-ups:**

- How to ensure message ordering?
- How to handle offline users?
- How to sync across multiple devices?
- How to implement end-to-end encryption?

**Time to Solve:** 60 minutes

---

### 7. Uber / Lyft

**Requirements:**

```
Functional:
- Riders request rides
- Drivers accept rides
- Real-time location tracking
- ETA calculation
- Payment processing
- Rating system

Non-Functional:
- 10M riders, 1M drivers
- Real-time updates (<1s)
- High accuracy for location
- 99.9% availability
```

**Scale:**

```
Active rides: 100K concurrent
Location updates: 1M drivers * 4/min = 67K updates/sec
Database writes: High write throughput
Geospatial queries: Need efficient indexing
```

**Key Concepts:**

```
✓ Geospatial indexing (geohash, quadtree, S2)
✓ WebSocket for real-time updates
✓ Matching algorithm (rider to driver)
✓ ETA calculation and route optimization
✓ Surge pricing
✓ Idempotency (prevent double charges)
```

**Common Follow-ups:**

- How to find nearby drivers efficiently?
- How to handle high-demand areas?
- How to calculate dynamic pricing?
- How to ensure payment reliability?

**Time to Solve:** 60 minutes

---

### 8. YouTube

**Requirements:**

```
Functional:
- Upload videos
- Watch videos
- Search videos
- Comments, likes, subscriptions
- Recommendations

Non-Functional:
- 2B users
- 500 hours uploaded per minute
- 1B hours watched per day
- Low latency streaming
```

**Scale:**

```
Storage: 500 hours/min * 60 min/hour * 1GB/hour = 30TB/min
Upload bandwidth: 30TB/min = 4Gbps
Watch bandwidth: Much higher (10x-100x)
CDN: Critical for global delivery
```

**Key Concepts:**

```
✓ Video transcoding and formats
✓ CDN and edge caching
✓ Adaptive bitrate streaming (HLS, DASH)
✓ Thumbnail generation
✓ Search and indexing (Elasticsearch)
✓ Recommendation engine
```

**Common Follow-ups:**

- How to handle different video qualities?
- How to generate recommendations?
- How to handle copyright detection?
- How to optimize for mobile vs desktop?

**Time to Solve:** 60 minutes

---

### 9. Notification System

**Requirements:**

```
Functional:
- Send notifications (push, email, SMS)
- Support multiple platforms (iOS, Android, web)
- User preferences (opt-in/out)
- Delivery confirmation

Non-Functional:
- 100M notifications per day
- Low latency (<5s)
- High delivery rate (>95%)
- Handle spikes (breaking news)
```

**Scale:**

```
QPS: 100M / 86400 = ~1,200 notifications/sec
Peak QPS: 10x = 12,000/sec
Multiple channels: 3x overhead = 36,000/sec
```

**Key Concepts:**

```
✓ Message queue (Kafka, RabbitMQ)
✓ Fan-out pattern
✓ Third-party integrations (FCM, APNs, SendGrid)
✓ Retry logic and dead letter queue
✓ Rate limiting per channel
✓ Template management
```

**Common Follow-ups:**

- How to handle delivery failures?
- How to prevent notification spam?
- How to prioritize notifications?
- How to track delivery and open rates?

**Time to Solve:** 60 minutes

---

### 10. Web Crawler

**Requirements:**

```
Functional:
- Crawl web pages
- Extract links
- Store content
- Respect robots.txt
- Avoid duplicates

Non-Functional:
- Crawl 1B pages per month
- Politeness (don't overload servers)
- Distributed crawling
- Handle failures gracefully
```

**Scale:**

```
Pages per second: 1B / (30 * 86400) = ~400 pages/sec
Storage: 1B pages * 100KB = 100TB
Bandwidth: 400 pages/sec * 100KB = 40MB/s
```

**Key Concepts:**

```
✓ BFS vs DFS crawling
✓ URL frontier (priority queue)
✓ URL deduplication (bloom filter)
✓ Distributed task queue
✓ DNS resolution optimization
✓ Politeness policy (rate limiting per domain)
```

**Common Follow-ups:**

- How to prioritize which pages to crawl?
- How to detect duplicate content?
- How to handle dynamic JavaScript sites?
- How to distribute crawling across machines?

**Time to Solve:** 60 minutes

---

## Phase 3: Advanced (Hard)

Complex systems requiring advanced trade-offs and deep technical knowledge.

### 11. Netflix

**Requirements:**

```
Functional:
- Video streaming
- Personalized recommendations
- Multiple profiles
- Resume watching
- Download for offline
- Multiple device support

Non-Functional:
- 200M subscribers
- 1B hours watched per week
- 4K video support
- 99.99% availability
- Global distribution
```

**Scale:**

```
Concurrent streams: 10M
Bandwidth: 10M * 5Mbps = 50 Tbps
Storage: 100K titles * 10 versions * 5GB = 5 PB
CDN: Global edge locations
```

**Key Concepts:**

```
✓ Microservices architecture
✓ Content delivery network (CDN)
✓ Adaptive bitrate streaming
✓ Machine learning recommendations
✓ Open Connect (Netflix CDN)
✓ Chaos engineering (resilience)
✓ A/B testing infrastructure
```

**Complex Requirements:**

- How to optimize encoding for quality/size?
- How to handle network congestion?
- How to make recommendations in real-time?
- How to ensure content security (DRM)?
- How to test at Netflix scale?

**Advanced Patterns:**

```
✓ Circuit breaker pattern
✓ Bulkhead pattern
✓ Event-driven architecture
✓ CQRS (Command Query Responsibility Segregation)
✓ Eventual consistency with compensation
```

**Time to Solve:** 60 minutes

---

### 12. Payment System

**Requirements:**

```
Functional:
- Process payments
- Multiple payment methods
- Refunds and chargebacks
- Transaction history
- Fraud detection

Non-Functional:
- 1M transactions per day
- 99.999% availability (5 9's)
- Strong consistency
- ACID transactions
- PCI DSS compliance
```

**Scale:**

```
TPS: 1M / 86400 = ~12 transactions/sec
Peak TPS: 10x = 120/sec
Storage: Financial records (keep forever)
Audit logs: Complete transaction trail
```

**Key Concepts:**

```
✓ ACID transactions
✓ Idempotency (crucial for payments)
✓ Two-phase commit
✓ Saga pattern for distributed transactions
✓ Double-entry bookkeeping
✓ Reconciliation processes
✓ Fraud detection ML models
```

**Complex Requirements:**

- How to ensure exactly-once processing?
- How to handle partial failures?
- How to implement refunds atomically?
- How to detect fraudulent transactions?
- How to handle currency conversion?
- Compliance and auditing?

**Advanced Patterns:**

```
✓ Event sourcing
✓ CQRS
✓ Compensating transactions
✓ Ledger-based architecture
✓ Real-time fraud scoring
```

**Time to Solve:** 60 minutes

---

### 13. Google Search

**Requirements:**

```
Functional:
- Web search
- Ranking results
- Auto-complete
- Spell correction
- Image/video search
- Location-based results

Non-Functional:
- Index 60B pages
- Handle 5B searches per day
- Latency <100ms
- 99.9% availability
```

**Scale:**

```
QPS: 5B / 86400 = ~58K queries/sec
Index size: 60B pages * 100KB = 6 EB
Crawl rate: Need to recrawl regularly
Computation: Massive ranking calculations
```

**Key Concepts:**

```
✓ Inverted index
✓ PageRank algorithm
✓ Distributed indexing
✓ Query processing pipeline
✓ Trie for autocomplete
✓ Spell correction algorithms
✓ MapReduce for batch processing
```

**Complex Requirements:**

- How to rank search results?
- How to update index in real-time?
- How to handle typos and synonyms?
- How to personalize results?
- How to prevent spam/manipulation?
- How to handle multi-language search?

**Advanced Patterns:**

```
✓ Distributed computing (MapReduce, Spark)
✓ Machine learning ranking
✓ Real-time indexing pipeline
✓ Caching at multiple layers
✓ Geographic distribution
```

**Time to Solve:** 60 minutes

---

### 14. Ad System

**Requirements:**

```
Functional:
- Display targeted ads
- Click tracking
- Impression tracking
- Billing advertisers
- Ad auction/bidding
- Fraud prevention

Non-Functional:
- 10B ad impressions per day
- 100M clicks per day
- Latency <50ms
- High availability
- Accurate billing
```

**Scale:**

```
QPS: 10B / 86400 = ~116K impressions/sec
Click QPS: 100M / 86400 = ~1,200 clicks/sec
Real-time bidding: <10ms decision time
Data: Massive user behavior data
```

**Key Concepts:**

```
✓ Real-time bidding (RTB)
✓ Ad auction algorithms (Vickrey, GSP)
✓ Click-through rate (CTR) prediction
✓ User targeting and segmentation
✓ Fraud detection
✓ Attribution modeling
✓ Stream processing (Kafka, Flink)
```

**Complex Requirements:**

- How to select best ad in <50ms?
- How to prevent click fraud?
- How to optimize ad placement?
- How to handle billing accurately?
- How to target users without PII?
- How to measure campaign effectiveness?

**Advanced Patterns:**

```
✓ Machine learning pipelines
✓ Real-time feature engineering
✓ A/B testing framework
✓ Event streaming architecture
✓ Lambda architecture (batch + stream)
```

**Time to Solve:** 60 minutes

---

## Problem Selection Guide

### How to Choose Problems

**Week 1-2: Foundation**

```
Do all 4 Phase 1 problems in order:
1. URL Shortener (core concepts)
2. Pastebin (storage patterns)
3. Rate Limiter (distributed systems intro)
4. Key-Value Store (sharding and replication)
```

**Week 3-6: Core Systems**

```
Choose 4-6 Phase 2 problems based on interest:

Must Do:
- Twitter/Instagram (feed systems)
- WhatsApp/Messenger (real-time systems)

Pick 2-4 from:
- Uber/Lyft (geospatial)
- YouTube (video streaming)
- Notification System (message queues)
- Web Crawler (distributed crawling)
```

**Week 7-9: Advanced**

```
Choose 2-3 Phase 3 problems:

Highly Recommended:
- Netflix (if interested in streaming)
- Payment System (financial systems)

Optional:
- Google Search (search engines)
- Ad System (ad tech)
```

### Progression Strategy

**Stage 1: Learn**

```
First time solving a problem:
1. Read requirements carefully (10 min)
2. Try to design on your own (30 min)
3. Get stuck? That's OK - note where
4. Read solution/watch video (30 min)
5. Understand the approach (20 min)

Total: ~90 minutes per problem
```

**Stage 2: Practice**

```
Second time (after 2-3 days):
1. Solve from scratch (45 min)
2. Compare with solution
3. Note improvements
4. Focus on weak areas

Total: 60 minutes per problem
```

**Stage 3: Master**

```
Third time (after 1 week):
1. Solve within time limit (45 min)
2. Explain out loud
3. Handle follow-up questions
4. Discuss trade-offs confidently

Total: 45 minutes (interview speed)
```

### When You're Ready for Next Level

**Phase 1 → Phase 2:**

```
✓ Can design Phase 1 systems in 45 minutes
✓ Understand caching, sharding, replication
✓ Can explain trade-offs clearly
✓ Handle 2-3 follow-up questions
```

**Phase 2 → Phase 3:**

```
✓ Can design Phase 2 systems in 60 minutes
✓ Understand distributed systems patterns
✓ Can discuss consistency models
✓ Handle 3-5 follow-up questions
✓ Can critique your own design
```

**Phase 3 → Interview Ready:**

```
✓ Can design any system in 45-60 minutes
✓ Fluent in trade-off discussions
✓ Can adapt design based on feedback
✓ Can estimate scale accurately
✓ Can handle adversarial questions
```

---

## Practice Tips

### Deliberate Practice

```
Don't just read solutions:
1. Set timer (45-60 minutes)
2. Design on paper/whiteboard
3. Explain out loud (record yourself)
4. Review and critique
```

### Mock Interviews

```
After 4 weeks of practice:
1. Find a practice partner
2. Take turns being interviewer
3. Give constructive feedback
4. Focus on communication
```

### Track Your Progress

```
Keep a spreadsheet:
Problem | Date | Time Taken | Confidence (1-10) | Notes
```

### Focus Areas by Role

**Backend Engineer:**

```
Priority: All Phase 1, WhatsApp, YouTube, Uber
Learn: Databases, caching, APIs, message queues
```

**Frontend Engineer:**

```
Priority: URL Shortener, Twitter feed, Netflix
Learn: CDN, caching, real-time updates, API design
```

**Full Stack:**

```
Priority: Balanced across all phases
Learn: End-to-end system design
```

**SRE/DevOps:**

```
Priority: Monitoring, scaling, high availability
Learn: Rate limiter, web crawler, distributed systems
```

---

## Related Resources

- [Framework](framework.md) - The 4-step approach to solving these problems
- [Communication Guide](communication.md) - How to explain your solutions
- [Calculations](calculations.md) - Estimate scale for any problem
- [Common Mistakes](common-mistakes.md) - What to avoid while practicing
- [Back to Interviews](index.md)
