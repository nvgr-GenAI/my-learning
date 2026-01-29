# System Design Learning Path

Structured programs to take you from beginner to expert. Choose the path that fits your timeline and learning style.

---

## Choose Your Path

| Path | Duration | Intensity | Best For | Outcome |
|------|----------|-----------|----------|---------|
| **[Beginner](#beginner-path)** | 8-10 weeks | 8-10h/week | New to system design | Understand core concepts, design simple systems |
| **[Intermediate](#intermediate-path)** | 6-8 weeks | 10-12h/week | Know basics, want depth | Design production systems, interview ready |
| **[Advanced](#advanced-path)** | 4-6 weeks | 12-15h/week | Experienced, need polish | Master complex patterns, senior-level interviews |

**Not sure where you fit?** Take the [skill assessment](#skill-assessment) below.

---

## Beginner Path

**For:** Developers new to system design or distributed systems
**Goal:** Understand fundamental concepts and design simple web applications
**Duration:** 8-10 weeks (8-10 hours/week)

### Week 1-2: Core Foundations

**Topics:**
- What is system design and why it matters
- Scalability basics (vertical vs horizontal)
- Client-server architecture
- HTTP, DNS, and networking basics

**Study:**
- Read: [Fundamentals](fundamentals/index.md)
- Read: [Scalability Introduction](scalability/index.md)
- Read: [Networking Basics](networking/index.md)

**Practice:**
- Draw a simple 3-tier web app architecture
- Explain how a URL request travels through the internet

**Checkpoint:** Can you explain the difference between scalability and performance? Can you draw a basic web architecture?

---

### Week 3-4: Data Storage

**Topics:**
- SQL vs NoSQL databases
- When to use which database
- ACID properties
- Database indexing basics
- Data modeling fundamentals

**Study:**
- Read: [Databases](databases/index.md)
- Learn: Entity-relationship diagrams
- Compare: PostgreSQL, MongoDB, Redis

**Practice:**
- Design database schema for a blog platform
- Design database schema for e-commerce store
- Identify which database type for different use cases

**Checkpoint:** Can you choose the right database for a given problem? Can you design a normalized database schema?

---

### Week 5-6: Making Systems Fast

**Topics:**
- Caching strategies (browser, CDN, application, database)
- Cache invalidation patterns
- CDN and content delivery
- Load balancing algorithms

**Study:**
- Read: [Caching](caching/index.md)
- Read: [Load Balancing](load-balancing/index.md)
- Learn: LRU cache, write-through vs write-back

**Practice:**
- Design: URL shortener with caching
- Design: Image hosting service with CDN
- Implement: LRU cache in code

**Checkpoint:** Can you explain when and how to use caching? Can you design a system with multiple cache layers?

---

### Week 7-8: Scaling Basics

**Topics:**
- Scaling strategies (horizontal, vertical, database sharding)
- Replication and redundancy
- Single points of failure
- Basic distributed system concepts

**Study:**
- Read: [Scalability Patterns](scalability/index.md)
- Learn: Master-slave replication
- Learn: Database sharding strategies

**Practice:**
- Design: Scale a blog from 100 to 100K users
- Design: Instagram-like photo sharing (basic version)
- Identify bottlenecks in existing designs

**Checkpoint:** Can you scale a simple web app to handle 100K+ users? Can you identify and eliminate single points of failure?

---

### Week 9-10: Real Design Practice

**Practice Problems:**

1. **URL Shortener (like bit.ly)**
   - Requirements: Shorten URLs, redirect, track clicks
   - Focus: Database design, caching, scaling reads

2. **Pastebin (text sharing)**
   - Requirements: Store text, generate shareable links
   - Focus: Storage estimation, expiration policies

3. **Simple Social Feed**
   - Requirements: Post updates, view friend posts
   - Focus: Database design, caching strategy

**Milestone:** ✅ Can design and explain simple web applications with proper caching and database choices

---

## Intermediate Path

**For:** Developers who understand basics and want to design production systems
**Goal:** Design complex, scalable systems and prepare for technical interviews
**Duration:** 6-8 weeks (10-12 hours/week)

### Week 1-2: Advanced Data Strategies

**Topics:**
- Consistent hashing
- Database partitioning and sharding
- Replication strategies (master-slave, multi-master)
- CAP theorem deep dive
- Eventual consistency

**Study:**
- Read: [Distributed Systems](distributed-systems/index.md)
- Read: [Consistent Hashing](consistent-hashing/index.md)
- Learn: Cassandra, DynamoDB architecture

**Practice:**
- Design: Distributed cache system
- Design: Global key-value store
- Solve: How to rebalance data when adding servers

**Checkpoint:** Can you explain CAP theorem with real examples? Can you design a sharded database?

---

### Week 3-4: Asynchronous Systems

**Topics:**
- Message queues (RabbitMQ, SQS)
- Event streaming (Kafka, Kinesis)
- Pub/Sub patterns
- Event-driven architecture
- Async processing patterns

**Study:**
- Read: [Messaging Systems](messaging/index.md)
- Learn: At-least-once vs exactly-once delivery
- Learn: Dead letter queues

**Practice:**
- Design: Notification system (email, push, SMS)
- Design: Order processing pipeline
- Design: Real-time analytics system

**Checkpoint:** Can you explain when to use message queues vs event streams? Can you design an async processing pipeline?

---

### Week 5-6: Distributed Systems Patterns

**Topics:**
- Consensus algorithms (Raft, Paxos basics)
- Service discovery
- API gateway patterns
- Circuit breakers and retries
- Rate limiting strategies

**Study:**
- Read: [Reliability & Security](reliability-security/index.md)
- Learn: How distributed locks work
- Learn: Timeouts and backoff strategies

**Practice:**
- Design: Rate limiter
- Design: Distributed lock service
- Design: API gateway

**Checkpoint:** Can you design resilient systems that handle failures gracefully?

---

### Week 7-8: Complex Application Designs

**Practice Problems:**

1. **Chat Application (like WhatsApp)**
   - Requirements: 1-on-1 chat, group chat, online presence
   - Focus: Real-time messaging, WebSockets, message storage

2. **Instagram/Twitter Feed**
   - Requirements: Post content, follow users, personalized feed
   - Focus: Fan-out strategies, timeline generation, caching

3. **Ride Sharing (like Uber basics)**
   - Requirements: Match riders with drivers, location tracking
   - Focus: Geospatial indexing, real-time matching, availability

4. **Video Streaming (like YouTube basics)**
   - Requirements: Upload, transcode, stream videos
   - Focus: CDN, storage tiers, adaptive streaming

**Milestone:** ✅ Can handle most interview questions and design production-ready systems

---

## Advanced Path

**For:** Experienced engineers preparing for senior+ interviews or complex projects
**Goal:** Master advanced patterns and handle any system design question
**Duration:** 4-6 weeks (12-15 hours/week)

### Week 1-2: Advanced Distributed Systems

**Topics:**
- Strong vs eventual consistency trade-offs
- Distributed transactions (2PC, Saga pattern)
- CRDT (Conflict-free Replicated Data Types)
- Vector clocks and conflict resolution
- Quorum reads and writes

**Study:**
- Deep dive: Consensus algorithms
- Case study: Dynamo paper (Amazon)
- Case study: Spanner paper (Google)

**Practice:**
- Design: Global database with strong consistency
- Design: Multi-region active-active setup
- Solve: Conflict resolution in distributed systems

---

### Week 3-4: Performance at Scale

**Topics:**
- Performance optimization techniques
- Monitoring and observability
- A/B testing infrastructure
- Global traffic management
- Cost optimization

**Study:**
- Read: [Performance Optimization](performance/index.md)
- Learn: How Netflix does chaos engineering
- Learn: Distributed tracing

**Practice:**
- Optimize: Given system to handle 10x load
- Design: Monitoring and alerting system
- Design: Feature flag system

---

### Week 5-6: Complex System Designs

**Practice Problems:**

1. **Payment System (like Stripe)**
   - Requirements: Process payments, handle failures, idempotency
   - Focus: Distributed transactions, exactly-once processing

2. **Search Engine (like Google basics)**
   - Requirements: Crawl web, index, search
   - Focus: Distributed crawling, inverted index, ranking

3. **Ad System (like Google Ads)**
   - Requirements: Serve ads, track clicks, real-time bidding
   - Focus: Low latency, high throughput, revenue optimization

4. **Netflix Architecture**
   - Requirements: Stream video globally, recommendations
   - Focus: CDN strategy, microservices, personalization

**Milestone:** ✅ Can design any system and discuss trade-offs at depth. Ready for staff+ interviews.

---

## Skill Assessment

Answer these to find your level:

**Beginner if you answer no to most:**
- [ ] Can you explain the difference between SQL and NoSQL?
- [ ] Do you know what a load balancer does?
- [ ] Can you explain what caching is and when to use it?
- [ ] Have you designed a basic web application architecture?

**Intermediate if you answer yes to most:**
- [ ] Can you explain CAP theorem?
- [ ] Do you know when to use message queues?
- [ ] Can you design a system to handle 1M users?
- [ ] Have you worked with distributed systems?

**Advanced if you answer yes to most:**
- [ ] Can you explain consensus algorithms?
- [ ] Do you know how to handle distributed transactions?
- [ ] Can you optimize systems for global scale?
- [ ] Have you designed systems for millions of concurrent users?

---

## Study Tips

### How to Learn Effectively

1. **Understand, Don't Memorize**
   - Focus on *why* solutions work, not just *what* they are
   - Practice explaining concepts to others

2. **Draw Everything**
   - Always sketch architecture diagrams
   - Use boxes, arrows, and labels
   - Visualize data flow

3. **Calculate Numbers**
   - Estimate storage, bandwidth, throughput
   - Practice back-of-envelope calculations
   - Example: 1M users × 100KB profile = 100GB storage

4. **Study Real Systems**
   - Read engineering blogs (Netflix, Uber, Twitter)
   - Understand why they made specific choices
   - See: [Case Studies](case-studies/index.md)

5. **Practice Out Loud**
   - Explain your designs to friends
   - Record yourself and listen back
   - Practice interviewing

### Common Mistakes to Avoid

❌ Jumping straight to implementation details
✅ Start with requirements and high-level design

❌ Not considering scale and constraints
✅ Always ask about scale, users, data size

❌ Over-engineering simple problems
✅ Start simple, then optimize

❌ Ignoring trade-offs
✅ Discuss pros/cons of each choice

❌ Not practicing enough
✅ Design 2-3 systems per week minimum

---

## Practice Problem Bank

### Easy (Beginner)
1. Design a URL shortener
2. Design a pastebin
3. Design a rate limiter
4. Design a key-value store
5. Design a simple blog platform

### Medium (Intermediate)
1. Design Twitter/Instagram feed
2. Design WhatsApp/Messenger
3. Design Uber/Lyft
4. Design Dropbox/Google Drive
5. Design YouTube (basic features)
6. Design Notification system
7. Design Web crawler
8. Design Autocomplete system

### Hard (Advanced)
1. Design Netflix
2. Design payment system (Stripe)
3. Design search engine (Google)
4. Design ad system (Google Ads)
5. Design stock trading platform
6. Design global database (Spanner)

---

## Learning Resources

### Books
- **Designing Data-Intensive Applications** by Martin Kleppmann (best overall)
- **System Design Interview** by Alex Xu (interview focused)
- **Web Scalability for Startup Engineers** by Artur Ejsmont (practical)

### Online
- Engineering blogs: Netflix, Uber, Twitter, LinkedIn
- [High Scalability](http://highscalability.com/)
- [SystemDesign Primer](https://github.com/donnemartin/system-design-primer)

### Practice
- [LeetCode System Design](https://leetcode.com/discuss/interview-question/system-design)
- Mock interviews with friends
- Design systems you use daily

---

## Next Steps

**Completed a path?**

- ✅ [Take mock interviews](interviews/index.md)
- ✅ Build a project from scratch
- ✅ Read engineering blogs
- ✅ Help others learn (teach = mastery)

**Still learning?**

- 📚 Study [fundamentals](fundamentals/index.md) if concepts are unclear
- 💼 Jump to [interview prep](interviews/index.md) if that's your goal
- 🏗️ Check [case studies](case-studies/index.md) to see real-world designs

---

**Remember:** System design is learned through practice. Design 2-3 systems every week, and you'll be amazed how quickly you improve.

[← Back to System Design Home](index.md)
