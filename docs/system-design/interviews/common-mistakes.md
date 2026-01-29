# Common Mistakes

Learn from these common interview mistakes so you can avoid them. Each mistake includes why it's a problem and how to fix it.

---

## Overview

| Mistake | Impact | Easy to Fix? |
|---------|--------|--------------|
| Not Asking Questions | Critical | Yes |
| Diving into Details Too Early | High | Yes |
| Not Discussing Trade-offs | High | Moderate |
| Over-Engineering | Moderate | Moderate |
| Not Considering Failures | High | Moderate |
| Working in Silence | High | Yes |

---

## Mistake 1: Not Asking Questions

### The Problem

Starting to design immediately without clarifying requirements.

**Example:**

```
Interviewer: "Design a URL shortener."

Bad response:
[Immediately starts drawing] "So we'll have a web server, a database..."
```

### Why It's Bad

- You might build the wrong system
- Shows lack of real-world experience (requirements are always unclear)
- Wastes time on wrong assumptions
- Misses opportunity to narrow scope

### How to Fix

**Always start with clarifying questions (5-10 minutes):**

```
Good response:
"Let me make sure I understand the requirements before diving in.

Functional requirements:
- Should the system just create short URLs and redirect, or also support
  custom aliases, analytics, expiration?
- What characters are allowed in short codes?
- How long should the short code be?

Non-functional requirements:
- What's the expected scale? Users? URLs per day?
- What's the read-to-write ratio?
- What are latency requirements?
- How important is consistency vs availability?

Let me confirm what I heard: [summarize]

Should I focus on any particular aspect?"
```

### Example Questions by System Type

**Social Media (Twitter, Instagram):**

```
✓ How many users?
✓ Follower graph scale (average followers per user)?
✓ Real-time updates required?
✓ Content types (text, images, videos)?
✓ Recommendation algorithm needed?
```

**Messaging (WhatsApp, Slack):**

```
✓ 1-on-1 only or group chats?
✓ Max group size?
✓ Online/offline status needed?
✓ Message history - how long to store?
✓ Encryption requirements?
```

**Video Streaming (YouTube, Netflix):**

```
✓ Upload or just playback?
✓ Live streaming or VOD?
✓ Quality options (4K, HD, SD)?
✓ Recommendation engine needed?
✓ Geographic distribution requirements?
```

---

## Mistake 2: Diving into Details Too Early

### The Problem

Starting with implementation details before establishing high-level architecture.

**Example:**

```
Interviewer: "Design Twitter."

Bad response:
"For the database schema, we need a Users table with columns:
id BIGINT, username VARCHAR(50), email VARCHAR(255)..."

[15 minutes later, still discussing database schema]
```

### Why It's Bad

- May waste time on wrong components
- Never gets to overall architecture
- Shows inability to think at right abstraction level
- Interview time is limited (45-60 minutes)

### How to Fix

**Follow the framework: High-level first, then deep dive.**

```
Good response:
"Let me start with a high-level architecture, then we can dive into
specific components.

At a high level, we need:
1. Client (mobile app, web)
2. Load balancer
3. API servers (create tweet, read timeline)
4. Database (tweets, users, relationships)
5. Cache (for hot tweets and timelines)
6. Object storage (for media)

[Draw simple diagram]

The main flows are:
- User posts tweet → API → Database
- User reads timeline → API → Cache/Database → Response

Does this high-level approach make sense before I dive into specifics?"
```

### Abstraction Levels

```
Level 1 (First 10 min): High-level components
  - Major services
  - Data flow
  - Get interviewer buy-in

Level 2 (Next 20 min): Component details
  - Database schema
  - API design
  - Key algorithms

Level 3 (If time): Implementation details
  - Specific technologies
  - Code examples
  - Edge cases
```

---

## Mistake 3: Not Discussing Trade-offs

### The Problem

Stating technology choices without explaining alternatives or trade-offs.

**Example:**

```
Bad: "We'll use MongoDB for the database."

Interviewer: "Why MongoDB?"

Bad: "Because it's fast and scalable."
```

### Why It's Bad

- Shows shallow understanding
- Doesn't demonstrate decision-making ability
- Misses opportunity to show experience
- Can't defend choices under questioning

### How to Fix

**Always discuss alternatives and explain your reasoning.**

```
Good:
"For the database, I'm considering two options:

Option 1: PostgreSQL (SQL)
  Pros: ACID transactions, strong consistency, complex queries
  Cons: Harder to scale horizontally

Option 2: MongoDB (NoSQL)
  Pros: Easy horizontal scaling, flexible schema
  Cons: Eventual consistency (by default), no multi-document ACID

For a URL shortener, I'd choose PostgreSQL because:
1. We need strong consistency to prevent duplicate short codes
2. Schema is simple and won't change often
3. Scale (40 writes/sec) doesn't require NoSQL's benefits
4. We can scale reads with replicas

If requirements change - say 100K writes/sec - I'd reconsider MongoDB
or sharding PostgreSQL."
```

### Common Trade-off Discussions

**1. SQL vs NoSQL:**

```
Trade-off: Consistency vs Scalability

Use SQL when:
✓ Need ACID transactions
✓ Complex relationships (JOINs)
✓ Structured, predictable schema
✓ Moderate scale

Use NoSQL when:
✓ Massive scale (sharding needed)
✓ Flexible schema
✓ Eventual consistency acceptable
✓ Simple queries (key-value, document)
```

**2. Monolith vs Microservices:**

```
Trade-off: Simplicity vs Flexibility

Use Monolith when:
✓ Small team
✓ Early stage (unclear what to split)
✓ Fast iteration needed
✓ Simple deployment

Use Microservices when:
✓ Large organization (multiple teams)
✓ Clear service boundaries
✓ Different scaling needs per service
✓ Can handle operational complexity
```

**3. Cache-Aside vs Write-Through:**

```
Trade-off: Read Performance vs Write Consistency

Cache-Aside:
✓ Only cache what's accessed (memory efficient)
✓ Reads fast after first access
✗ Cache miss penalty

Write-Through:
✓ Cache always consistent
✓ No cache miss
✗ Slower writes
✗ Cache unused data
```

### Framework for Any Decision

```
When choosing [X], I considered:

Option A: [Technology/Approach]
  Pros: [2-3 benefits]
  Cons: [2-3 drawbacks]

Option B: [Alternative]
  Pros: [2-3 benefits]
  Cons: [2-3 drawbacks]

I choose [A/B] because:
  [Reason tied to requirements]

If [requirement changes], I'd reconsider [alternative].
```

---

## Mistake 4: Over-Engineering

### The Problem

Proposing overly complex solutions for simple problems.

**Example:**

```
Interviewer: "Design a URL shortener with 100 users and 1000 requests/day."

Bad response:
"We'll use Kubernetes for orchestration, Kafka for message queues,
Cassandra for the database sharded across 100 nodes, Redis Cluster with
sentinel for caching, and deploy across 5 AWS regions..."
```

### Why It's Bad

- Shows poor judgment
- Wastes resources in real world
- Indicates lack of experience (experienced engineers keep it simple)
- Makes system harder to maintain

### How to Fix

**Start simple. Scale when needed.**

```
Good response:
"Given the scale - 1000 requests/day is about 0.01 QPS - we can start
very simple:

Phase 1 (Current scale):
- Single web server
- Single PostgreSQL database
- Simple in-memory caching

This handles 100x current load easily and costs ~$50/month.

Phase 2 (If we reach 100K requests/day):
- Add load balancer
- Add 2-3 API servers
- Add Redis cache
- Add database read replica

Phase 3 (If we reach 100M requests/day):
- Database sharding
- CDN for redirects
- Multi-region deployment

Does this phased approach make sense?"
```

### Red Flags for Over-Engineering

```
For small scale (<1000 QPS):
✗ Kubernetes (use simple VMs)
✗ Microservices (use monolith)
✗ Database sharding (use single DB with replicas)
✗ Message queues (use direct calls)
✗ Service mesh (use simple load balancer)

For medium scale (1000-10K QPS):
✓ Load balancer
✓ Database replication
✓ Redis cache
✓ Multiple app servers
✗ Still too early for: Kafka, sharding, multi-region
```

### Right-Sizing Decisions

| Scale | Appropriate Architecture |
|-------|-------------------------|
| <100 QPS | Single server, single DB |
| 100-1K QPS | Load balancer, 3-5 servers, DB replicas, cache |
| 1K-10K QPS | Above + sharding (if needed), CDN, message queue |
| 10K-100K QPS | Above + multi-region, advanced caching |
| >100K QPS | Full distributed system, microservices |

---

## Mistake 5: Not Considering Failures

### The Problem

Designing happy-path only without discussing what happens when things fail.

**Example:**

```
Bad: [Draws architecture with single database]

Interviewer: "What happens if the database goes down?"

Bad: "Um... the system stops working?"
```

### Why It's Bad

- Systems fail in production constantly
- Shows lack of operational experience
- High-availability is a key non-functional requirement
- Misses opportunity to show depth

### How to Fix

**Proactively discuss failure scenarios and mitigations.**

```
Good response:
"Let me identify potential failures and how to handle them:

1. API Server Failure:
   - Have multiple servers behind load balancer
   - Load balancer health checks
   - Auto-scaling to replace failed servers
   - Impact: No downtime (other servers handle load)

2. Database Master Failure:
   - Master-slave replication
   - Automatic failover (promote slave to master)
   - Takes 30-60 seconds
   - Impact: Brief downtime, no data loss

3. Cache Failure:
   - Cache is not source of truth
   - Falls back to database
   - Impact: Higher latency, but system still works

4. Load Balancer Failure:
   - Multiple load balancers with DNS failover
   - Impact: 1-2 minute failover time

5. Entire Datacenter Failure:
   - Multi-region deployment (Phase 3)
   - DNS failover to secondary region
   - Impact: 5-10 minute recovery
"
```

### Failure Scenarios Checklist

```
✓ Server crashes
✓ Database fails (master, slave)
✓ Cache goes down
✓ Network partition
✓ Disk full
✓ Datacenter outage
✓ DDoS attack
✓ Database connection pool exhausted
✓ Message queue backup
✓ Third-party API down
```

### Mitigation Strategies

**1. Redundancy:**

```
- Multiple servers
- Database replicas
- Multi-region deployment
- No single point of failure
```

**2. Health Checks:**

```
- Load balancer health checks
- Service monitoring
- Automatic recovery
- Alert on failures
```

**3. Graceful Degradation:**

```
- Cache miss → query database
- Service down → return cached/default data
- Timeout → retry with backoff
- Overload → rate limiting
```

**4. Data Safety:**

```
- Regular backups
- Write-ahead logging
- Replication (async or sync)
- Point-in-time recovery
```

---

## Mistake 6: Working in Silence

### The Problem

Thinking silently for long periods without explaining your thought process.

**Example:**

```
[10 minutes of silence while drawing]

Interviewer: "What are you thinking?"

Bad: "Just working on the design..."
```

### Why It's Bad

- Interviewer can't evaluate your thinking
- Looks like you're stuck
- Wastes time if you're going wrong direction
- Misses opportunity for hints
- Not collaborative

### How to Fix

**Think out loud constantly.**

```
Good:
"Let me start by drawing the main components...

[While drawing] So we have clients here at the top, they'll connect to
a load balancer... I'm adding the load balancer because we need multiple
API servers for high availability...

[Continues drawing] Behind the API servers, we need a database. I'm
thinking PostgreSQL for strong consistency... Actually, wait, let me
reconsider. With a 100:1 read-write ratio, should I add a cache layer?

Yes, I'll add Redis cache here between API servers and database. That
will handle most reads...

[Draws more] For the redirect flow, the request comes in, we check Redis
first, if miss then database, then update cache...

Does this flow make sense so far?"
```

### What to Verbalize

**1. Clarifying Requirements:**

```
"Let me make sure I understand. You said 100 million URLs per month, so
that's about... let me calculate... 100M divided by 30 days is 3.3M per
day, divided by 86,400 seconds is about 40 requests per second. That's
moderate scale."
```

**2. Exploring Options:**

```
"For storing URLs, I'm thinking about two approaches:
1. Use auto-increment ID and convert to base62
2. Generate random codes and check for collisions

Let me think about the trade-offs..."
```

**3. Making Decisions:**

```
"I'm going with base62 encoding because it's deterministic, no collision
checks needed, and gives us 3.5 trillion combinations with 7 characters.
That's enough for our 6 billion URLs over 5 years."
```

**4. Identifying Problems:**

```
"Hmm, I just realized if we use sequential IDs, someone could enumerate
all URLs by incrementing the code. We might need to add randomization or
make this a security consideration..."
```

**5. Responding to Feedback:**

```
"Oh, good point about celebrity users creating hotspots. Let me think...
We could handle that by... actually, maybe we treat high-follower users
differently? Pre-compute their timelines? Or use a different fan-out
strategy?"
```

### Practice Technique

**1. Record Yourself:**

```
- Pick a problem
- Solve it while recording audio
- Listen back and count silent gaps >30 seconds
- Goal: No gaps >30 seconds
```

**2. Practice with Partner:**

```
- Take turns being interviewer
- Interviewer notes when candidate goes silent
- Provide feedback on communication
```

**3. Mock Interviews:**

```
- Use Pramp, Interviewing.io, or similar
- Get real feedback on communication
- Practice with strangers (more realistic)
```

---

## How to Avoid These Mistakes

### Pre-Interview Checklist

```
Before interview:
✓ Review framework (4-step approach)
✓ Practice 5-10 problems
✓ Practice thinking out loud
✓ Review trade-off discussions
✓ Study failure scenarios
✓ Get good sleep

During interview:
✓ Start with clarifying questions (5-10 min)
✓ Draw high-level design first
✓ Think out loud constantly
✓ Discuss trade-offs for every decision
✓ Ask for feedback regularly
✓ Consider failure scenarios
✓ Start simple, then scale
```

### Red Flags During Interview

If you catch yourself doing these, stop and correct:

```
🚩 "I'll use [technology] because it's the best"
   → Fix: Discuss alternatives and trade-offs

🚩 [Silence for >30 seconds]
   → Fix: Verbalize what you're thinking

🚩 "We'll use Kubernetes..."  (for small scale)
   → Fix: Start simple, explain when to add complexity

🚩 [Immediately starts with database schema]
   → Fix: Step back to high-level design first

🚩 "The database is here" [moves on]
   → Fix: Explain failure handling

🚩 [Draws for 20 minutes without asking questions]
   → Fix: Start with requirements clarification
```

### Recovery Strategies

**If you realize you made a mistake:**

```
Good: "Actually, let me reconsider that. I said we'd use NoSQL, but
       thinking about the consistency requirements, SQL might be better.
       Can I revise my design?"
```

**If you're stuck:**

```
Good: "I'm thinking through two approaches for [problem]. Can you give
       me a hint about which direction is more interesting to you?"
```

**If interviewer challenges your design:**

```
Good: "That's a great point. You're right that [issue] could be a problem.
       Let me adjust the design to handle that..."
```

---

## Example: Good Interview vs Bad Interview

### Bad Interview Example

```
Interviewer: "Design a URL shortener."

Candidate: [Silent for 30 seconds, then draws]
           "We'll use MongoDB for the database."

Interviewer: "Why MongoDB?"

Candidate: "Because NoSQL is fast."

Interviewer: "What if the database goes down?"

Candidate: "Um... we'd need to restart it?"

Interviewer: "How do you generate short codes?"

Candidate: "We could use a hash function."

Interviewer: "What about collisions?"

Candidate: "Uh... I'm not sure."

[30 minutes in, still no clear design]
```

**Mistakes made:**

- Didn't ask clarifying questions
- Silent thinking
- No trade-off discussion
- Didn't consider failures
- No clear structure

---

### Good Interview Example

```
Interviewer: "Design a URL shortener."

Candidate: "Great, let me clarify the requirements first.

Functional requirements:
- Should we support custom aliases or just auto-generated codes?
- Do we need analytics or just redirect?

Non-functional:
- What's the expected scale? URLs per day?
- Latency requirements?

[After clarification]

Let me start with a high-level design.
[Draws while explaining]

We have:
1. Load balancer - for distributing requests
2. API servers - stateless, can scale horizontally
3. Database - I'm thinking PostgreSQL for ACID guarantees
4. Redis cache - for 100:1 read/write ratio

Let me walk through the flows...
[Explains create and redirect flows]

Does this high-level design make sense before I dive into details?

[After approval]

Let me design the database schema...
[Shows schema while explaining indexes]

For generating short codes, I'm considering:
1. Base62 encoding of auto-increment ID
2. Random generation with collision check

Base62 is better because... [explains trade-offs]

For failures:
- Multiple API servers (no SPOF)
- Database replication (master-slave)
- Cache can fail (falls back to DB)

[Discusses scaling, monitoring, etc.]

What aspects would you like me to dive deeper into?"
```

**What was done right:**

- Clarified requirements first
- High-level before details
- Thought out loud constantly
- Discussed trade-offs
- Considered failures
- Invited feedback regularly
- Clear structure and logic

---

## Practice Tips

### 1. Self-Review

After each practice problem:

```
Did I:
✓ Ask clarifying questions?
✓ Start with high-level design?
✓ Discuss trade-offs?
✓ Think out loud?
✓ Consider failures?
✓ Avoid over-engineering?

Time spent:
- Requirements: _____ min (target: 5-10 min)
- High-level: _____ min (target: 10-15 min)
- Deep dive: _____ min (target: 20-30 min)
- Wrap-up: _____ min (target: 5-10 min)
```

### 2. Record and Review

```
Week 1-2: Record 3-5 practice sessions
- Note silent gaps
- Note missing trade-off discussions
- Note times jumped to details too early

Week 3-4: Focus on weak areas
- Practice thinking out loud
- Practice trade-off discussions
- Time yourself on each phase

Week 5-6: Full mock interviews
- Practice with partners
- Get external feedback
- Aim for interview-ready performance
```

### 3. Mistake Journal

```
Keep a log:
Date | Problem | Mistake Made | How to Fix | Improved?
---------------------------------------------------------
1/15 | Twitter | Went silent  | Talk more  | [ ]
1/16 | URL     | No trade-off | Compare    | [ ]
1/17 | Uber    | Too complex  | Start sim  | [x]
```

---

## Related Resources

- [Framework](framework.md) - The right structure to avoid mistakes
- [Communication Guide](communication.md) - How to think out loud effectively
- [Practice Problems](practice-problems.md) - Problems to practice on
- [Calculations](calculations.md) - How to estimate scale correctly
- [Back to Interviews](index.md)
