# System Design & Architecture

Learn to design scalable, reliable systems that serve millions of users—from basic concepts to complex distributed architectures.

---

## What is System Design?

System design is the process of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. It's about making informed trade-offs between competing concerns like scalability, reliability, performance, and cost.

Whether you're building a startup MVP or architecting systems for millions of users, system design skills help you:
- Make better technical decisions
- Communicate architecture effectively
- Scale systems efficiently
- Anticipate and prevent failures

---

## I Want To...

Choose what best describes your goal:

=== "📖 Learn System Design"

    **Start from the beginning:**

    | Your Level | Start Here | Time |
    |------------|------------|------|
    | Complete Beginner | [Fundamentals](fundamentals/index.md) → [Learning Path](learning-path.md) | 8-12 weeks |
    | Some Experience | [Learning Path: Intermediate](learning-path.md#intermediate-path) | 6-8 weeks |
    | Refresh Knowledge | Browse [Topics](#topics) below | As needed |

=== "💼 Prepare for Interviews"

    **Getting ready for tech interviews:**

    1. Start: [Interview Preparation Guide](interviews/index.md)
    2. Study: [Core Topics](#essential-interview-topics) below
    3. Practice: Work through [Interview Problems](interviews/index.md#practice-problems)
    4. Timeline: 4-8 weeks intensive

=== "🏗️ Design a Specific System"

    **Need to build something now:**

    | System Type | Key Resources |
    |-------------|---------------|
    | **Web Application** | [Fundamentals](fundamentals/index.md) · [Databases](databases/index.md) · [Caching](caching/index.md) · [Load Balancing](load-balancing/index.md) |
    | **Real-Time App** | [Messaging](messaging/index.md) · [WebSockets](networking/index.md) · [Distributed Systems](distributed-systems/index.md) |
    | **Data Pipeline** | [Databases](databases/index.md) · [Messaging](messaging/index.md) · [Scalability](scalability/index.md) |
    | **API Service** | [Load Balancing](load-balancing/index.md) · [Caching](caching/index.md) · [Reliability](reliability-security/index.md) |

    Then check: [Case Studies](case-studies/index.md) for real-world examples

=== "🎯 Study a Specific Topic"

    **Jump to what you need:**

    Browse [Topics](#topics) below or use search (press `/`)

---

## Essential Interview Topics

If preparing for interviews, prioritize these:

| Priority | Topic | Why Important | Time to Learn |
|----------|-------|---------------|---------------|
| 🔴 Critical | [Databases](databases/index.md) | 80% of systems need data storage | 1 week |
| 🔴 Critical | [Caching](caching/index.md) | First optimization technique | 3 days |
| 🔴 Critical | [Scalability](scalability/index.md) | Core interview concept | 1 week |
| 🟡 Important | [Load Balancing](load-balancing/index.md) | Common scaling strategy | 2 days |
| 🟡 Important | [Distributed Systems](distributed-systems/index.md) | Advanced problems | 1 week |
| 🟢 Useful | [Messaging](messaging/index.md) | Async architectures | 3 days |
| 🟢 Useful | [Consistent Hashing](consistent-hashing/index.md) | Data distribution | 2 days |

**Full interview guide:** [Interview Preparation](interviews/index.md)

---

## Topics

### Foundation
Start here if you're new to system design.

- **[Fundamentals](fundamentals/index.md)** - Core concepts, trade-offs, design principles
- **[Scalability](scalability/index.md)** - Grow from 10 to 10 million users
- **[Performance](performance/index.md)** - Latency, throughput, optimization
- **[Reliability & Security](reliability-security/index.md)** - Build systems that don't fail

### Data & Storage
How to store and retrieve data efficiently.

- **[Databases](databases/index.md)** - SQL vs NoSQL, choosing the right database
- **[Caching](caching/index.md)** - Redis, Memcached, CDN strategies
- **[Consistent Hashing](consistent-hashing/index.md)** - Distribute data across servers

### Communication
How components talk to each other.

- **[Networking](networking/index.md)** - HTTP, TCP/IP, DNS, protocols
- **[Load Balancing](load-balancing/index.md)** - Distribute traffic across servers
- **[Messaging](messaging/index.md)** - Queues, pub/sub, event streaming
- **[Proxy](proxy/index.md)** - Forward and reverse proxies

### Advanced
Complex distributed system concepts.

- **[Distributed Systems](distributed-systems/index.md)** - CAP theorem, consistency, consensus
- **[Sessions](sessions/index.md)** - Stateful vs stateless, session management

### Practice & Application
Learn from real systems and prepare for interviews.

- **[Case Studies](case-studies/index.md)** - How Netflix, Uber, Twitter built their systems
- **[Interview Preparation](interviews/index.md)** - Framework, problems, tips

---

## Quick Reference

### The Trade-Off Triangle

Every system design decision involves trade-offs. You typically optimize for 2 of these 3:

```
        Consistency
           /\
          /  \
         /    \
        /______\
  Availability  Partition Tolerance
```

**The CAP Theorem:** In a distributed system, you can only guarantee 2 out of 3.

### Common Architecture Patterns

| Pattern | Use When | Examples |
|---------|----------|----------|
| **Monolith** | Small team, simple domain, MVP | Early Instagram, GitHub |
| **Microservices** | Large team, complex domain, need to scale independently | Netflix, Amazon, Uber |
| **Serverless** | Event-driven, variable load, want no ops | AWS Lambda apps |
| **Event-Driven** | Async workflows, loose coupling | Order processing, notifications |

### Technology Cheat Sheet

| Need | Use |
|------|-----|
| Relational data + ACID | PostgreSQL, MySQL |
| Document storage | MongoDB, DynamoDB |
| Caching | Redis, Memcached |
| Message queue | RabbitMQ, SQS |
| Event streaming | Kafka, Kinesis |
| Search | Elasticsearch |
| CDN | CloudFront, Cloudflare |

---

## Learning Resources

| Resource | Best For |
|----------|----------|
| [Structured Learning Path](learning-path.md) | Follow week-by-week program (8-12 weeks) |
| [Interview Guide](interviews/index.md) | Prepare for tech interviews (4-8 weeks) |
| [Case Studies](case-studies/index.md) | Learn from real-world systems |
| [Fundamentals](fundamentals/index.md) | Start with core concepts |

---

## Interview Readiness Checklist

**Before your interview, make sure you can:**

### Concepts
- [ ] Explain CAP theorem with real examples
- [ ] Discuss SQL vs NoSQL trade-offs
- [ ] Design multi-tier caching strategy
- [ ] Explain horizontal vs vertical scaling
- [ ] Understand consistent hashing
- [ ] Know when to use message queues

### Skills
- [ ] Follow a structured 4-step framework
- [ ] Draw clear architecture diagrams
- [ ] Do back-of-envelope calculations quickly
- [ ] Discuss trade-offs for every technical choice
- [ ] Handle "what if" questions confidently
- [ ] Think out loud effectively

### Practice
- [ ] Completed 5+ easy problems
- [ ] Completed 5+ medium problems
- [ ] Completed 2+ mock interviews
- [ ] Can design Twitter/WhatsApp/Uber confidently

**Full preparation guide:** [Interview Prep](interviews/index.md)

---

## Getting Started

1. **New to system design?** → Start with [Fundamentals](fundamentals/index.md)
2. **Have some experience?** → Follow the [Learning Path](learning-path.md)
3. **Preparing for interviews?** → Jump to [Interview Guide](interviews/index.md)
4. **Need specific info?** → Browse [Topics](#topics) above

**Questions?** The fundamentals guide explains all core concepts.
