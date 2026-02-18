# Architecture Interview Questions

**Common interview questions organized by pattern** | 💬 Questions | ✅ Strong Answers | ⚠️ Red Flags

---

## How to Use This Page

Architecture questions test whether you can **choose the right pattern for the constraints** and **articulate trade-offs clearly**. Interviewers don't want textbook definitions -- they want to hear you reason through trade-offs with concrete scenarios.

---

=== "Monolith vs Microservices"

    ### Q: When would you choose a monolith over microservices?

    **Strong answer:**
    > "For a team under 15 engineers where domain boundaries aren't clear yet. A monolith lets you iterate fast, debug in a single process, and deploy simply. Stack Overflow serves 1.3B page views/month on 9 servers with a monolith. I'd only extract services when team coordination becomes the bottleneck -- typically at 20+ engineers -- or when one component has fundamentally different scaling needs."

    **Red flag:** "Microservices are always better" or "Monoliths don't scale."

    ---

    ### Q: How would you migrate from a monolith to microservices?

    **Strong answer:**
    > "I'd use the Strangler Fig pattern -- extract one domain at a time behind an API gateway, so clients are unaware of the migration. I'd start with a service that has clear boundaries and different scaling needs. The gateway routes specific paths to the new service while the monolith handles everything else. Critically, I'd extract by business domain (payments, inventory) not by technical layer (validation, notifications)."

    **Key points:** Incremental, domain-based, gateway-routed, rollback-safe.

    ---

    ### Q: What is a distributed monolith and how do you avoid it?

    **Strong answer:**
    > "A distributed monolith has the deployment complexity of microservices but none of the benefits -- services must deploy together, share databases, or break when one changes. You avoid it by ensuring each service owns its data, has clear API contracts, and aligns with a bounded context. If changing service A always requires changing service B, they probably should be one service."

    ---

    ### Q: How do you determine service boundaries?

    **Strong answer:**
    > "I use Domain-Driven Design's bounded contexts. I look for where the ubiquitous language shifts -- when the same term means different things to different teams. I follow organizational boundaries (Conway's Law), identify independent lifecycles, and check if different parts need different consistency guarantees. A good service encapsulates a complete business capability."

=== "Event-Driven"

    ### Q: When would you use event-driven architecture vs request-driven?

    **Strong answer:**
    > "Event-driven when a single action triggers multiple independent reactions -- like an order placement that needs to reserve inventory, charge payment, send email, and update analytics simultaneously. Request-driven when the caller needs an immediate response, like fetching a user profile. Most real systems use both: synchronous APIs for user-facing reads, events for downstream fan-out."

    ---

    ### Q: Explain choreography vs orchestration. When would you pick each?

    **Strong answer:**
    > "In choreography, each service reacts to events independently -- no central coordinator. Great for simple flows with 3-4 participants where max decoupling matters. Netflix uses this for content encoding. In orchestration, a central coordinator directs each step -- easier to reason about for complex flows with many participants or branching logic. Uber uses orchestrated sagas via Temporal for ride processing. I'd pick choreography for simple fan-out, orchestration when I need explicit failure handling."

    ---

    ### Q: How do you ensure a database write and event publication happen atomically?

    **Strong answer:**
    > "The Transactional Outbox pattern. Write both the business data and the event to the same database in one transaction -- the event goes into an 'outbox' table. A separate process (or CDC tool like Debezium) reads the outbox and publishes to the message broker. This guarantees atomicity because both writes succeed or fail together."

    ---

    ### Q: What is a saga pattern? How do you handle failures?

    **Strong answer:**
    > "A saga is a sequence of local transactions across services. Each step publishes an event triggering the next. If step 3 fails, compensating transactions undo steps 2 and 1 in reverse. For example, if payment fails after inventory was reserved, a 'RefundRequired' event triggers stock release. The key design question is whether to use choreography (each service knows its compensation) or orchestration (central coordinator manages the sequence)."

=== "CQRS & Event Sourcing"

    ### Q: When would you use CQRS?

    **Strong answer:**
    > "When the read model and write model have fundamentally different shapes. For example, a social media feed: writes are simple (create post), but reads require joining user data, post content, like counts, and comment previews from multiple sources. CQRS lets me optimize each independently. Also when the read-to-write ratio is extreme -- say 100:1 -- I can scale read replicas with denormalized data while keeping the write side lean. I would NOT use it for simple CRUD where read and write shapes are identical."

    ---

    ### Q: What is event sourcing and when would you use it?

    **Strong answer:**
    > "Instead of storing current state, I store every state change as an immutable event. Current state is derived by replaying events. It's the same principle as bank transaction logs or Git. I'd use it when I need a complete audit trail (finance, healthcare), temporal queries ('what was the state last Tuesday?'), or the ability to build new projections from historical data without migration. The trade-offs are complexity, eventual consistency, and needing to handle schema evolution."

    ---

    ### Q: How do you handle the eventual consistency between read and write models in CQRS?

    **Strong answer:**
    > "The read model lags behind writes by milliseconds to seconds. I'd handle this with 'read your own writes' -- after a user submits a command, the UI optimistically shows the expected result rather than querying the read model immediately. For critical paths, the command handler can return the write result synchronously as an escape hatch. The key insight is that most reads don't need real-time consistency -- a dashboard can lag by seconds without harm."

=== "Serverless"

    ### Q: When would you choose serverless over containers?

    **Strong answer:**
    > "Serverless for bursty, unpredictable traffic where I'd otherwise pay for idle capacity. A startup API handling 100 requests/hour normally but 10K during a launch. Event-driven processing like image resizing on upload. Scheduled jobs that run for 30 seconds nightly. I'd choose containers for latency-sensitive workloads (cold starts hurt), long-running processes (Lambda's 15-min limit), high steady throughput (beyond ~50M requests/month, containers are cheaper), or anything needing persistent connections like WebSockets."

    ---

    ### Q: How do you handle cold starts?

    **Strong answer:**
    > "Cold starts range from 50ms (Go) to 3 seconds (Java). Mitigations: use lightweight runtimes (Python/Node over Java), keep packages small, minimize initialization outside the handler, and use provisioned concurrency for latency-critical paths. AWS SnapStart helps Java. In practice, cold starts affect only 1-5% of invocations with steady traffic. The architecture decision is whether that P95/P99 tail latency is acceptable for your use case."

    ---

    ### Q: What are the cost trade-offs?

    **Strong answer:**
    > "At 1M requests/month with 200ms avg duration, Lambda costs about $2 vs $60 for containers. At 500M requests, Lambda costs $1,050 vs $120 for containers. The crossover is roughly 50-100M steady requests/month. The key variable is traffic pattern -- if you have 10x traffic for 2 hours and near-zero for 22 hours, serverless wins even at high volume because you pay nothing during idle time."

=== "Service Mesh"

    ### Q: When would you introduce a service mesh?

    **Strong answer:**
    > "When I have 50+ services in multiple languages and I'm seeing inconsistent networking behavior -- some teams implement retries, others don't; mTLS exists in Java services but not Python. A mesh standardizes retries, circuit breaking, mTLS, and observability at the infrastructure layer, independent of language. I'd use Linkerd for simplicity or Istio for advanced routing needs. For fewer than 10 services, library-based approaches are simpler."

    ---

    ### Q: What's the overhead of a service mesh?

    **Strong answer:**
    > "Each sidecar adds 1-5ms latency per hop and ~20-50MB memory depending on the implementation. For 500 services with 3 replicas, that's 30-75GB of cluster RAM for sidecars alone. Newer ambient mesh models (Istio) reduce this with per-node proxies instead of per-pod. The trade-off is whether the uniform mTLS, observability, and traffic management justify that overhead -- at scale it almost always does."

=== "Domain-Driven Design"

    ### Q: How does DDD help with microservices?

    **Strong answer:**
    > "DDD's bounded contexts tell you where to draw service boundaries. Each microservice should align with one bounded context -- owning its data, its domain model, and its ubiquitous language. This prevents the distributed monolith where services are technically separate but share databases or must deploy together. Amazon's two-pizza teams are essentially bounded contexts enforced by organizational policy."

    ---

    ### Q: What's an Anti-Corruption Layer and when would you use it?

    **Strong answer:**
    > "An ACL is a translation layer between your internal domain model and an external system's model. When I integrate with Stripe, I don't let their PaymentIntent model leak into my order domain. The ACL translates 'amount: 9999 cents' into 'amount: $99.99' and 'succeeded' into my PAID status. I'd use it whenever integrating with external APIs, legacy systems, or any upstream whose model differs from mine."

    ---

    ### Q: What's the difference between an entity and a value object?

    **Strong answer:**
    > "An entity has identity -- two orders with the same items are still different orders because they have different IDs. A value object is defined by its attributes -- two Money(100, 'USD') instances are interchangeable. Value objects should be immutable. The practical impact: entities get their own table with a primary key; value objects are embedded in entity tables or serialized. Most objects in a domain model should be value objects -- entities are fewer but more important."

=== "General Architecture"

    ### Q: How do you choose between different architecture patterns?

    **Strong answer:**
    > "I start with the simplest architecture that meets the constraints. Team size drives the initial choice -- under 15 people, monolith almost always. Then I look at specific requirements: need independent scaling? Microservices. Event fan-out? EDA. Audit trail? Event sourcing. Variable traffic? Serverless. I never introduce complexity without a concrete problem it solves. The biggest mistake I see is premature microservices for a 3-person team."

    ---

    ### Q: Explain Conway's Law and why it matters for architecture.

    **Strong answer:**
    > "Conway's Law says systems mirror the communication structures of the organizations that build them. If I have four teams, I'll end up with a four-service architecture regardless of what I plan. This isn't a bug -- it's a tool. I align service boundaries with team boundaries intentionally. Amazon's two-pizza teams and Uber's DOMA both use this principle: organize teams by business domain, and the architecture follows. Fighting Conway's Law always fails."

    ---

    ### Q: Walk me through how you'd handle a service going down in your architecture.

    **Strong answer:**
    > "Multiple layers of defense. First, circuit breakers detect the failure and stop sending traffic to the downed service, returning fallback responses. Second, retries with exponential backoff and jitter handle transient failures without overwhelming the recovering service. Third, bulkheads isolate the failure so the rest of the system continues. Fourth, graceful degradation -- maybe I serve cached data or a simplified response. The key is designing for partial failure from day one rather than assuming everything works."

---

## Interview Preparation Tips

| Tip | Why |
|-----|-----|
| **Lead with trade-offs, not definitions** | "X is better when... but costs you..." shows depth |
| **Use real company examples** | "Netflix uses choreography for..." is more convincing than theory |
| **State your assumptions** | "Assuming 10M users and a team of 20..." grounds your answer |
| **Draw diagrams** | ASCII boxes and arrows > paragraphs of explanation |
| **Acknowledge what you'd NOT use** | "I'd avoid CQRS here because..." shows judgment |
| **Connect patterns** | "I'd combine EDA with saga pattern because..." shows systems thinking |

---

## Related Topics

- [Monolithic Architecture](monolithic.md)
- [Microservices Architecture](microservices.md)
- [Event-Driven Architecture](event-driven.md)
- [Serverless Architecture](serverless.md)
- [Service Mesh](service-mesh.md)
- [Domain-Driven Design](domain-driven-design.md)
- [CQRS & Event Sourcing](cqrs-event-sourcing.md)
- [Interview Framework](../interviews/framework.md)
