# Communication Guide

System design interviews are 80% communication, 20% technical knowledge. This guide teaches you how to communicate effectively during the interview.

---

## What Interviewers Want to Hear

### Good vs Bad Examples

**Scenario: Choosing a Database**

**Bad:**

```
"We'll use MongoDB because it's NoSQL and NoSQL is faster."
```

Why it's bad:

- Vague reasoning
- No context for the choice
- Incorrect assumption (NoSQL isn't always faster)
- No trade-off discussion

**Good:**

```
"I'm considering two options for the database:

Option 1: PostgreSQL (SQL)
Pros: ACID transactions, strong consistency, mature, good for relational data
Cons: Harder to scale horizontally

Option 2: MongoDB (NoSQL)
Pros: Easy horizontal scaling, flexible schema
Cons: Eventual consistency by default, no ACID across documents

Given our requirement for strong consistency (URL shortener needs to avoid
duplicate short codes) and relatively simple schema (just URLs table), I'd
choose PostgreSQL. We can scale reads with replicas, and 40 writes/sec is
well within a single Postgres instance's capability.

Does this make sense, or do you see the requirements differently?"
```

Why it's good:

- Shows structured thinking
- Compares options
- Lists pros and cons
- Ties decision to requirements
- Invites feedback

---

## Template for Explanations

Use this structure for explaining any component or decision:

### The SPICE Framework

**S**tate the problem
**P**ropose solution
**I**dentify alternatives
**C**ompare trade-offs
**E**xplain choice

### Example: Caching Strategy

**State:**

```
"Our URL shortener has a 100:1 read-to-write ratio, so reads are the main
performance concern. We need to reduce database load."
```

**Propose:**

```
"I propose adding a Redis cache layer between API servers and database."
```

**Identify:**

```
"We could also:
1. Use in-memory caching on API servers
2. Use a CDN
3. Add database read replicas without caching"
```

**Compare:**

```
"Redis pros: Centralized, fast (sub-ms), shared across servers
Redis cons: Extra infrastructure, need to handle cache invalidation

In-memory pros: Simple, no extra infrastructure
In-memory cons: Cache not shared, each server has cold start

CDN pros: Distributed globally, very fast
CDN cons: Overkill for our use case, expensive

Read replicas pros: Simple, maintains consistency
Read replicas cons: Still hits database, slower than cache"
```

**Explain:**

```
"I choose Redis because:
1. Sub-ms latency handles our <100ms requirement
2. Shared cache means high hit rate
3. Cache-aside pattern is simple to implement
4. Redis is standard for this use case

We can start with a single Redis instance and add clustering if needed."
```

---

## Handling "I Don't Know"

### It's OK Not to Know Everything

Interviewers expect you to encounter unfamiliar topics. What matters is how you handle it.

### Good Responses to Unknown Topics

**Scenario: Interviewer asks about Bloom Filters**

**Bad:**

```
"I don't know what that is."
[silence]
```

**Good:**

```
"I'm not familiar with Bloom filters specifically, but let me think through
the problem. You mentioned we need to check if a URL was already crawled,
and we have billions of URLs, so memory is a concern.

A hash table would work but uses lots of memory. Could we use a compact
data structure that tells us 'definitely not seen' or 'possibly seen'?
We could accept some false positives if it saves memory.

Is that the idea behind Bloom filters?"
```

Why it's good:

- Admits lack of knowledge honestly
- Shows problem-solving ability
- Reasons through the use case
- Makes an intelligent guess
- Asks for validation

### Reasoning Through Unknown Concepts

**Framework:**

```
1. Acknowledge: "I haven't used [technology] before..."
2. Context: "...but based on the problem we're solving..."
3. Reason: "...I would need something that [key properties]..."
4. Guess: "...so my guess is [technology] provides [benefits]?"
5. Ask: "Could you give me a quick overview of how it works?"
```

**Example: Asked about CAP Theorem**

```
"I haven't studied CAP theorem formally, but I understand there are
trade-offs between consistency and availability in distributed systems.

For our URL shortener, we need to prevent duplicate short codes, which
sounds like a consistency requirement. But we also want high availability.

My guess is CAP theorem says we can't have both perfect consistency and
perfect availability at the same time during network failures?

Could you clarify what specific trade-off you're asking about?"
```

---

## Thinking Out Loud

### Why It Matters

Interviewers can't read your mind. Silent thinking looks like:

- You're stuck
- You don't know what you're doing
- You're not collaborative

Thinking out loud shows your thought process even if you make mistakes.

### What to Verbalize

**1. Clarifying Requirements:**

```
"Let me make sure I understand the scale. You said 100 million users and
1% are active daily, so that's 1 million DAU. And if each user creates
5 posts per day, that's 5 million posts daily. Let me write that down..."
```

**2. Exploring Options:**

```
"For the database, I'm thinking about a few options. We need to store
user profiles and posts. User profiles are pretty structured, but posts
might have varying fields like text, images, videos. Should I optimize
for structured queries or flexible schema?

Let me think about the access patterns first..."
```

**3. Making Decisions:**

```
"OK, so we have 10,000 reads per second. A single database can probably
handle that, but we need high availability. I'll add read replicas -
maybe 3 of them. That gives us 4x capacity and redundancy. Does that
seem reasonable?"
```

**4. Calculating Numbers:**

```
"Let me estimate storage. 100 million tweets per day, average tweet is
140 characters so maybe 200 bytes with metadata. That's 100M × 200 bytes
= 20GB per day. Over a year, that's 20GB × 365 = 7.3TB. Add some buffer
for media, so maybe 10TB per year. That's manageable."
```

**5. Identifying Issues:**

```
"Wait, I just realized a problem. If we generate short codes sequentially,
users could guess other people's URLs. We need either random generation
or we need to make URLs unguessable. Maybe we add a random salt?"
```

**6. Responding to Feedback:**

```
"Oh, good point about database hotspots. If we shard by user ID, popular
users would create hotspots. Maybe we should shard by tweet ID instead?
That would distribute the load more evenly."
```

### Practice Technique

**Record yourself:**

```
1. Pick a problem
2. Start a voice recorder
3. Solve the problem while explaining every thought
4. Play back and listen
5. Note where you went silent
6. Practice those moments
```

---

## Trade-Off Discussions

### Why Trade-offs Matter

Every design decision has pros and cons. Discussing them shows:

- You understand multiple solutions
- You can evaluate options objectively
- You make informed decisions
- You're not dogmatic

### Framework for Trade-offs

```
"For [decision], we're trading [X] for [Y]."
```

### Common Trade-offs

#### 1. SQL vs NoSQL

```
Trade-off: Consistency vs Scalability

SQL (PostgreSQL):
✓ ACID transactions
✓ Strong consistency
✓ Complex queries (JOINs)
✗ Harder to scale horizontally
✗ Schema migrations can be complex

NoSQL (MongoDB):
✓ Easy horizontal scaling
✓ Flexible schema
✓ High write throughput
✗ Eventual consistency (by default)
✗ No multi-document ACID (limited)

Decision:
"For our URL shortener, we need to prevent duplicate short codes, which
requires strong consistency. The scale (40 writes/sec) doesn't require
NoSQL's scalability benefits. Therefore, SQL is the better choice."
```

#### 2. Synchronous vs Asynchronous

```
Trade-off: Simplicity vs Performance

Synchronous:
✓ Simple to understand
✓ Easy to debug
✓ Immediate consistency
✗ Slower response times
✗ Blocked threads

Asynchronous:
✓ Non-blocking, better throughput
✓ Can handle more concurrent requests
✓ Better resource utilization
✗ Complex error handling
✗ Eventual consistency issues

Decision:
"For creating short URLs, sync is fine (40/sec is low). But for recording
analytics clicks, async makes sense (4000/sec is high and analytics can
be eventual)."
```

#### 3. Microservices vs Monolith

```
Trade-off: Flexibility vs Complexity

Monolith:
✓ Simple deployment
✓ Easy local development
✓ Better performance (no network calls)
✗ Scaling all-or-nothing
✗ Technology lock-in

Microservices:
✓ Independent scaling
✓ Technology flexibility
✓ Team autonomy
✗ Complex deployment
✗ Network latency
✗ Distributed debugging

Decision:
"For a new startup, I'd start with a monolith. We don't know what will
need to scale, and we want to iterate fast. We can split into
microservices later when we know the pain points."
```

#### 4. Cache-Aside vs Write-Through

```
Trade-off: Read Performance vs Write Consistency

Cache-Aside (Lazy Loading):
✓ Only cache what's accessed (efficient memory)
✓ Cache failures don't affect writes
✗ Cache miss penalty
✗ Potential inconsistency

Write-Through:
✓ Cache always consistent
✓ No cache miss penalty
✗ Cache what's not accessed (waste memory)
✗ Write latency increased

Decision:
"For URL redirects with 100:1 read/write ratio, cache-aside is better.
We optimize for read performance, and a cache miss occasionally is
acceptable."
```

#### 5. Database Replication: Sync vs Async

```
Trade-off: Consistency vs Availability

Synchronous:
✓ Strong consistency
✓ No data loss on master failure
✗ Higher write latency
✗ Writes fail if replica down

Asynchronous:
✓ Low write latency
✓ High availability (writes succeed even if replica down)
✗ Potential data loss
✗ Read replicas may lag

Decision:
"For URL creation, async replication is OK. If we lose a few seconds of
data during failover, users can retry. For a payment system, we'd need
synchronous replication."
```

#### 6. REST vs GraphQL

```
Trade-off: Simplicity vs Flexibility

REST:
✓ Simple, well-understood
✓ Easy to cache
✓ Good tooling
✗ Over-fetching/under-fetching
✗ Multiple requests for related data

GraphQL:
✓ Fetch exactly what you need
✓ Single request for complex data
✓ Strongly typed schema
✗ Harder to cache
✗ Complex queries can be expensive
✗ More complex to implement

Decision:
"For our URL shortener, REST is sufficient. We have simple entities
(URLs) and don't need complex nested queries."
```

### Template for Trade-off Discussion

```
When choosing [Component/Technology]:

Option A: [Name]
  Pros: [2-3 benefits]
  Cons: [2-3 drawbacks]

Option B: [Name]
  Pros: [2-3 benefits]
  Cons: [2-3 drawbacks]

Given our requirements:
- [Requirement 1]
- [Requirement 2]

I choose [Option] because [reason tied to requirements].

Alternative considered: [Other option] would be better if [different requirement].
```

---

## Structured Explanation Examples

### Example 1: Explaining API Design

```
"Let me design the API for creating short URLs.

Endpoint: POST /api/shorten

Why POST? Because we're creating a resource.

Request body:
{
  "long_url": "https://example.com/very/long/path",
  "custom_alias": "mylink",  // optional
  "expires_at": "2026-12-31"  // optional
}

Why JSON? Industry standard, easy to extend, human-readable.

Response (201 Created):
{
  "short_url": "https://short.ly/abc123",
  "short_code": "abc123",
  "long_url": "https://example.com/very/long/path",
  "created_at": "2026-01-29T10:00:00Z"
}

Why 201? Indicates resource created successfully.
Why include created_at? Useful for debugging and analytics.

Error (400 Bad Request):
{
  "error": "invalid_url",
  "message": "The provided URL is not valid"
}

Alternative considered: Could use GET with query params, but POST is more
RESTful for resource creation and doesn't have URL length limits."
```

### Example 2: Explaining Scaling Strategy

```
"Our current system handles 1000 requests/sec, but we need to scale to
10,000 requests/sec. Here's my approach:

Step 1: Identify bottleneck
- Profile the system
- Database is bottleneck (CPU at 80%)

Step 2: Vertical scaling first
- Scale up database (more CPU/memory)
- Quick, simple, handles 3-4x growth
- Takes us to ~3000 req/sec

Step 3: Add caching
- Redis cache for read-heavy workload
- 80% cache hit rate reduces DB load by 80%
- Takes us to ~15,000 req/sec capacity

Step 4: Horizontal scaling (if needed)
- Add read replicas for database
- Load balance across replicas
- Each replica adds capacity

Why this order?
1. Vertical scaling is simplest (do first)
2. Caching gives biggest bang for buck
3. Horizontal scaling is most complex (do last)

This approach gets us to 10,000 req/sec with manageable complexity."
```

---

## Asking Clarifying Questions

### When to Ask

```
✓ Beginning (clarify requirements)
✓ Before deep dives (confirm assumptions)
✓ When stuck (get hints)
✓ Before finalizing (validate approach)

✗ Too frequently (interrupting)
✗ Obvious things (shows lack of knowledge)
```

### How to Ask

**Good Questions:**

```
"What's more important for this system: consistency or availability?"

"Should I optimize for read performance or write performance?"

"Is it acceptable to have eventual consistency for analytics?"

"How real-time do the updates need to be? Seconds? Minutes?"
```

**Bad Questions:**

```
"What database should I use?" (too broad, you should propose)

"Is 100ms latency good?" (you should know this)

"How does Redis work?" (too basic)
```

### Question Templates

```
Clarifying scope:
"Should I focus on [aspect A] or [aspect B]?"

Validating approach:
"I'm thinking [approach]. Does that align with what you're looking for?"

Handling uncertainty:
"I see two paths: [option A] or [option B]. Which direction interests you?"

Getting unstuck:
"I'm weighing [trade-off]. What's more important in this context?"
```

---

## Iterating Based on Feedback

### Responding to Challenges

**Interviewer: "What if we have 1 million writes per second instead of 40?"**

```
Good response:

"OK, that changes things significantly. Let me reconsider...

At 1M writes/sec:
- Single database can't handle this (max ~10K writes/sec)
- Need to shard the database

Sharding strategy:
- Shard by hash of short_code
- Each shard handles 1M / 100 shards = 10K writes/sec (manageable)

New architecture:
[Draw updated diagram with sharded databases]

Trade-offs:
- More complex (need shard routing logic)
- Can't do transactions across shards
- Rebalancing is hard

But this handles 1M writes/sec, which is the new requirement.

What do you think of this approach?"
```

**Bad response:**

```
"Oh, then we need to use Cassandra."
```

Why bad:

- Jumps to solution without reasoning
- Doesn't explain why
- Doesn't show adapted thinking

### Incorporating Feedback

```
Interviewer: "Your caching strategy might have a thundering herd problem."

Good response:

"Ah, good point. When cache expires and we get many simultaneous requests,
they'd all hit the database. Let me add cache warming and locking:

1. Probabilistic early expiration: Refresh cache before it expires
2. Lock on cache miss: First request fetches, others wait
3. Async refresh: Background job refreshes popular items

This prevents thundering herd. Thanks for catching that!"
```

Shows:

- You understand the problem
- You can adapt quickly
- You appreciate feedback
- You know multiple solutions

---

## Red Flags to Avoid

### 1. Being Defensive

```
Interviewer: "What about database hotspots?"

Bad: "My design doesn't have hotspots."

Good: "You're right, celebrity users could create hotspots. Let me adjust
       the design to handle that..."
```

### 2. Over-Explaining

```
Bad: [15 minutes explaining how TCP works]

Good: "We'll use REST over HTTP for the API. Should I dive into details
       or move on to the database design?"
```

### 3. Analysis Paralysis

```
Bad: [45 minutes still discussing database choice]

Good: "Let me choose PostgreSQL for now. We can revisit if the requirements
       change. Moving on to API design..."
```

### 4. Ignoring Interviewer

```
Bad: [Drawing for 10 minutes without talking]

Good: [Drawing while explaining] "I'm adding the cache layer here because...
       and it connects to the database like this..."
```

---

## Practice Exercises

### Exercise 1: Explain a Decision

Pick a technology decision. Explain it in 2 minutes using SPICE framework.

Example: Why use Redis for caching?

### Exercise 2: Handle a Curveball

Practice responding to: "What if the scale is 100x bigger?"

### Exercise 3: Think Out Loud

Solve a problem while recording yourself. Listen back and note silent periods.

### Exercise 4: Trade-off Discussion

For each decision, list 3 pros and 3 cons. Practice explaining them.

---

## Related Resources

- [Framework](framework.md) - What to say during each interview phase
- [Practice Problems](practice-problems.md) - Problems to practice communicating
- [Calculations](calculations.md) - How to explain your calculations
- [Common Mistakes](common-mistakes.md) - Communication mistakes to avoid
- [Back to Interviews](index.md)
