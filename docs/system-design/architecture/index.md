# Architecture Patterns

**Choose the right architecture for your system** | 🏛️ Patterns | ⚖️ Trade-offs | 💼 Decision Guide

---

## Quick Decision Guide

```
Which architecture should you pick?
====================================

What is your team size and scale?
    |
    +-- 1-10 engineers, MVP/startup
    |     |
    |     +-- Variable/spiky traffic? --> Serverless
    |     +-- Steady traffic?         --> Monolithic
    |
    +-- 10-30 engineers, growing product
    |     |
    |     +-- Need independent teams? --> Modular Monolith → Microservices
    |     +-- Single team, scaling?   --> Monolithic + caching/replicas
    |
    +-- 30+ engineers, multiple teams
    |     |
    |     +-- Real-time/async needs?  --> Event-Driven + Microservices
    |     +-- Many polyglot services? --> Microservices + Service Mesh
    |     +-- Complex domain?         --> DDD-guided boundaries
    |
    +-- High read:write ratio, audit needs
          |
          +-- CQRS & Event Sourcing
```

---

## Architecture Patterns

=== "Comparison Table"

    | Pattern | Best For | Team Size | Trade-off |
    |---------|----------|-----------|-----------|
    | [**Monolithic**](monolithic.md) | Startups, MVPs, simple apps | 1-15 | Simple but hard to scale independently |
    | [**Microservices**](microservices.md) | Large orgs, independent scaling | 10+ | Flexible but operationally complex |
    | [**Event-Driven**](event-driven.md) | Real-time, async workflows | 5-20+ | Decoupled but eventually consistent |
    | [**Serverless**](serverless.md) | Variable traffic, cost-sensitive | 1-10 | Zero-ops but cold starts + vendor lock-in |
    | [**Service Mesh**](service-mesh.md) | Many polyglot microservices | 20+ | Uniform networking but resource overhead |
    | [**Domain-Driven Design**](domain-driven-design.md) | Complex domains, service boundaries | 10+ | Right boundaries but high upfront investment |
    | [**CQRS & Event Sourcing**](cqrs-event-sourcing.md) | Audit trails, high read:write ratio | 5+ | Full history but added complexity |

=== "When to Use"

    **Start Here: Monolithic**

    - Team < 15 engineers, unclear domain boundaries
    - Need to ship fast and iterate
    - Stack Overflow serves 1.3B page views/month on 9 servers

    **Scale Up: Microservices**

    - Team > 10 engineers, need independent deployment
    - Different components need different scaling
    - Netflix: 800+ services, 200M subscribers

    **Decouple: Event-Driven**

    - One action triggers multiple independent reactions
    - Need temporal decoupling and audit trails
    - LinkedIn: 7 trillion messages/day through Kafka

    **Optimize Cost: Serverless**

    - Bursty/unpredictable traffic, idle periods
    - Event-driven processing (file uploads, webhooks)
    - Coca-Cola: vending machine API scales to zero

    **Standardize: Service Mesh**

    - 50+ polyglot services, need uniform mTLS/observability
    - Inconsistent retry/circuit-breaking across teams
    - Airbnb: Istio across 1,000+ services

    **Model the Domain: DDD**

    - Complex business rules, multiple teams
    - Need to find the right service boundaries
    - Amazon: two-pizza teams aligned to bounded contexts

    **Separate Reads/Writes: CQRS & Event Sourcing**

    - 100:1 read-to-write ratio, need audit trail
    - Read and write models diverge significantly
    - Banking: transaction log as source of truth

=== "Evolution Path"

    Most systems evolve through stages rather than starting complex:

    ```
    Stage 1: Monolith
    (0-15 engineers, finding product-market fit)
        |
        v
    Stage 2: Modular Monolith
    (15-30 engineers, clear domain boundaries emerge)
        |
        v
    Stage 3: Selective Microservice Extraction
    (30+ engineers, extract by domain NOT by layer)
        |
        +---> Add Event-Driven for async workflows
        +---> Add Service Mesh for networking concerns
        +---> Add CQRS for read-heavy paths
    ```

    **Don't skip stages!** Monolith → microservices without modularizing first = distributed monolith.

    | Company | Path | Lesson |
    |---------|------|--------|
    | **Shopify** | Monolith → Modular monolith (2004-2015) | Ran a Rails monolith processing $B before extracting |
    | **Amazon** | Monolith → SOA → Microservices (2001-2006) | Bezos API mandate driven by org pain, not tech limits |
    | **Netflix** | Monolith → Microservices (2009-2012) | Data center outage forced migration to cloud + services |
    | **Uber** | Monolith → Microservices → DOMA (2014-2020) | 2,200 services became unmanageable, re-organized by domain |

---

## Common Architecture Mistakes

| Mistake | Why It Happens | Better Approach |
|---------|---------------|-----------------|
| Premature microservices | "Netflix does it" for a 3-person team | Start monolith, migrate when org needs it |
| Technical service boundaries | "Notification Service" called by everyone | Domain boundaries — each context owns its notifications |
| Shared database across services | "It's easier than APIs" | Database per service, communicate via APIs/events |
| No API gateway | Clients call services directly | Gateway for routing, auth, rate limiting |
| Big-bang rewrite | "Let's rewrite everything in microservices" | Strangler fig pattern — extract incrementally |
| Ignoring Conway's Law | Architecture doesn't match team structure | Align service boundaries with team boundaries |

---

## Further Reading

- [Scalability Patterns](../scalability/index.md) — How to scale your architecture
- [Distributed Systems](../distributed-systems/index.md) — Consistency, consensus, coordination
- [API Design](../communication/api-design/index.md) — How services communicate
- [Deployment Strategies](../deployment/index.md) — How to deploy different architectures
- [Architecture Interview Questions](interview-questions.md) — Practice for interviews

**Practice Problems:**

- [Design Instagram](../problems/instagram.md) — Monolith → microservices migration
- [Design Netflix](../problems/netflix.md) — Microservices at scale
- [Design Uber](../problems/uber.md) — Event-driven + domain-driven architecture
