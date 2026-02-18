# Monolithic Architecture

A monolithic architecture packages an entire application -- its user interface, business logic, and data access -- into a single deployable unit running in one process. Despite the industry's obsession with microservices, the monolith remains the right starting point for the vast majority of software projects. Companies like Basecamp, Stack Overflow, and Shopify scaled to millions of users and billions in revenue on monolithic codebases before considering decomposition. The monolith is not a legacy pattern to be avoided; it is a deliberate, powerful architectural choice.

---

=== "What and Why"

    ## What is a Monolith?

    A monolith is an application where all functionality lives in a single codebase, compiles into a single artifact, and deploys as a single process. Every request -- whether it touches user authentication, product search, or payment processing -- is handled within that same running process.

    ```
    ┌──────────────────────────────────────────────────┐
    │              Monolithic Application               │
    │                                                  │
    │   ┌──────────────────────────────────────────┐   │
    │   │          Presentation Layer               │   │
    │   │    (HTTP handlers, API controllers)       │   │
    │   └──────────────────┬───────────────────────┘   │
    │                      │ function calls             │
    │   ┌──────────────────▼───────────────────────┐   │
    │   │          Business Logic Layer             │   │
    │   │    (Domain services, validation, rules)   │   │
    │   └──────────────────┬───────────────────────┘   │
    │                      │ function calls             │
    │   ┌──────────────────▼───────────────────────┐   │
    │   │           Data Access Layer               │   │
    │   │    (Repositories, ORM, query builders)    │   │
    │   └──────────────────┬───────────────────────┘   │
    │                      │                           │
    └──────────────────────┼───────────────────────────┘
                           │ TCP connection
                  ┌────────▼────────┐
                  │    Database     │
                  └─────────────────┘
    ```

    The defining trait is that communication between components is through in-process function calls -- microsecond latency, no serialization, full type safety. Compare this to microservices where every inter-service call crosses a network boundary with millisecond latency, serialization overhead, and potential failure modes.

    ## Why Monoliths are Right for Most Startups

    The reason is simple: **complexity is the enemy of shipping**. A startup's primary constraint is time-to-market, not scalability. A monolith eliminates an entire category of problems that microservices introduce -- service discovery, distributed transactions, network partitioning, data consistency across services, deployment orchestration, and observability across dozens of services.

    Consider the development experience. In a monolith, a developer can run the entire application on their laptop, set a breakpoint anywhere in the stack, and trace a request from HTTP handler to database query in a single debugger session. Refactoring is straightforward because the IDE can find every caller of a function. Cross-cutting changes like "add a `created_by` field to every entity" take hours, not weeks of coordinating across service teams.

    **Basecamp** has run on a Rails monolith for over 20 years, serving millions of users. Their team of roughly 20 developers ships features continuously without the coordination overhead that microservices would impose. DHH, Basecamp's CTO, has been vocal that their monolith is a competitive advantage -- fewer moving parts means more time building product.

    **Shopify** ran as a Rails monolith from 2004 through 2015, processing billions of dollars in merchant transactions. By the time they began extracting services, they had a deep understanding of their domain boundaries -- exactly the knowledge you need before decomposing.

    ## When a Monolith Fits

    | Scenario | Why It Works |
    |----------|-------------|
    | New product, unproven market | Ship fast, iterate, find product-market fit |
    | Team under 15-20 engineers | One codebase, one deploy pipeline, minimal coordination |
    | Domain boundaries unclear | Discover boundaries through iteration, not upfront guessing |
    | Read-heavy web application | Horizontal scaling + caching handles enormous read loads |

=== "Modular Monolith"

    ## Bounded Contexts Inside a Single Deployment

    The modular monolith is a middle ground that captures most of the organizational benefits of microservices while keeping the operational simplicity of a single deployment. The idea is to structure your monolith as a collection of well-defined modules, each owning its own domain logic and data access, communicating with other modules only through explicit public interfaces.

    ```
    ┌──────────────────────────────────────────────────────┐
    │                 Single Deployed Application           │
    │                                                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
    │  │   Catalog    │  │    Cart     │  │   Payment   │  │
    │  │   Module     │  │   Module    │  │   Module    │  │
    │  │             │  │             │  │             │  │
    │  │ - Products   │  │ - Cart ops  │  │ - Charges   │  │
    │  │ - Categories │  │ - Pricing   │  │ - Refunds   │  │
    │  │ - Search     │  │ - Discounts │  │ - Payouts   │  │
    │  │             │  │             │  │             │  │
    │  │ [Public API] │  │ [Public API]│  │ [Public API]│  │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
    │         │                │                │         │
    │    Module interfaces only -- no direct DB access     │
    │         │                │                │         │
    │  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  │
    │  │ catalog_*   │  │  cart_*     │  │ payment_*   │  │
    │  │  tables     │  │  tables    │  │  tables     │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  │
    │              Shared Database Server                   │
    └──────────────────────────────────────────────────────┘
    ```

    The critical discipline is that modules never reach into each other's database tables directly. The Cart module does not JOIN against the Catalog module's `products` table. Instead, it calls `CatalogModule.getProduct(id)` through a defined interface. This creates the same logical separation you would have with microservices, but without the network boundary.

    ## Shopify's Modular Monolith

    Shopify's journey is instructive. Rather than decomposing into hundreds of microservices, they restructured their massive Rails monolith into a modular monolith they call "componentized Rails." Each component defines its public interface, and automated tooling enforces that no component accesses another's internals. At their scale -- over 2 million merchants, $200+ billion in annual merchant sales -- this approach lets them maintain a single deployment while giving teams clear ownership boundaries.

    The key insight from Shopify's experience is that the pain of a monolith usually comes from poor internal structure, not from the deployment model itself. A well-structured monolith with enforced module boundaries gives you most of microservices' organizational benefits without the operational tax.

    ## When Modular Monolith Beats Microservices

    A modular monolith is often the better choice when your team is between 15 and 80 engineers, when you need strong data consistency across domains (ACID transactions within a single database), or when your operational maturity isn't ready for managing dozens of independent services, deployment pipelines, and observability stacks. It gives you a clean extraction path -- if a module truly needs to become a service later, its well-defined interface means the extraction is straightforward.

=== "Scaling a Monolith"

    ## Horizontal Scaling Behind a Load Balancer

    The most impactful scaling technique for a monolith is running multiple stateless copies behind a load balancer. This requires one key discipline: keep no state in the application process itself. Sessions go in Redis, file uploads go to object storage, and cached data lives in a shared cache layer.

    ```
    ┌─────────────────────────────┐
    │        Load Balancer        │
    │   (NGINX, ALB, HAProxy)     │
    └──────┬──────┬──────┬────────┘
           │      │      │
     ┌─────▼──┐ ┌─▼────┐ ┌▼──────┐
     │ App #1 │ │App #2│ │App #3 │    Stateless app instances
     │ (4 CPU)│ │(4CPU)│ │(4 CPU)│    (auto-scale based on load)
     └───┬────┘ └──┬───┘ └──┬────┘
         │         │        │
         └────┬────┴────────┘
              │
     ┌────────▼──────────┐
     │   Redis Cache      │    Sessions, hot data
     │   (100K+ QPS)      │
     └────────┬───────────┘
              │ cache miss
     ┌────────▼──────────┐
     │   Primary DB       │──── Write path
     │   (PostgreSQL)     │
     └───┬────────────────┘
         │ replication
     ┌───▼────────────────┐
     │   Read Replicas    │──── Read path (80% of traffic)
     │   (1-3 replicas)   │
     └────────────────────┘
    ```

    **Stack Overflow** serves over 1.3 billion page views per month with a monolithic ASP.NET application running on just 9 web servers. Their architecture is a textbook example: IIS load balancing across a handful of app servers, SQL Server with read replicas, Redis for caching, and Elasticsearch for search. No Kubernetes, no microservices, no container orchestration.

    ## The Database is Your Bottleneck

    In nearly every monolith scaling story, the database becomes the bottleneck long before the application tier does. Adding app servers is cheap and easy; the database is where the real engineering challenge lives.

    **Read replicas** are the first tool to reach for. Most web applications are 80-90% reads. Routing read queries to one or more replicas immediately frees your primary database to focus on writes. A typical setup uses 2-3 read replicas, which can handle 3-4x your original read throughput.

    **Caching** is the second tool. A Redis layer in front of your database can absorb the most frequently accessed data -- user profiles, product catalogs, configuration -- at 100,000+ QPS with sub-millisecond latency. If your hit rate is above 90%, you've effectively reduced database load by 10x.

    **Connection pooling** is often overlooked. Each app server instance opens connections to the database. With 10 app servers each holding 20 connections, your database is managing 200 concurrent connections. Tools like PgBouncer (for PostgreSQL) pool connections efficiently, letting you scale app instances without overwhelming the database.

    ## When You've Outgrown the Monolith

    The honest answer is: later than you think. Most applications never reach the scale where a monolith becomes truly inadequate. But there are real signals that indicate the monolith is holding you back:

    **Deploy velocity has collapsed.** When 50+ engineers are committing to the same codebase and every deploy requires coordinating across teams, when a single broken test blocks everyone's release, the monolith's single deploy pipeline has become a bottleneck.

    **One component's resource needs dominate.** If your video transcoding consumes 10x the CPU of your web-serving tier, scaling the entire monolith to meet transcoding demand wastes enormous resources. This is a signal that the transcoding component should be extracted.

    **Team autonomy is suffering.** When the payments team cannot ship a critical security patch because the catalog team's migration isn't ready, organizational coupling has become the constraint.

=== "Monolith to Microservices"

    ## The Strangler Fig Pattern

    Named after the strangler fig tree that slowly grows around its host tree, this pattern is the safest way to incrementally extract services from a monolith. Rather than a risky "big bang" rewrite, you route specific functionality to a new service while the monolith continues handling everything else.

    ```
    Phase 1: Route new traffic to extracted service
    ┌──────────────┐
    │ API Gateway / │
    │ Load Balancer │
    └──┬────────┬──┘
       │        │
       │   /payments/*
       │        │
       │   ┌────▼──────────┐
       │   │  Payment Svc   │ ← New service handles payments
       │   │  (extracted)   │
       │   └────┬───────────┘
       │        │ own database
    ┌──▼────────┼───────────┐
    │  Monolith │           │
    │  (all other traffic)  │ ← Monolith no longer handles payments
    │  ┌────────────────┐   │
    │  │ Catalog        │   │
    │  │ Cart           │   │
    │  │ Inventory      │   │
    │  └────────────────┘   │
    └───────────────────────┘

    Phase 2: Continue extracting by domain
    ┌──────────────┐
    │ API Gateway   │
    └──┬───┬───┬───┘
       │   │   │
    ┌──▼┐ ┌▼──┐ ┌▼─────────┐
    │Pay│ │Inv│ │ Monolith  │
    │Svc│ │Svc│ │ (Catalog, │
    │   │ │   │ │  Cart)    │
    └───┘ └───┘ └───────────┘
    ```

    The gateway (or a reverse proxy layer) makes this transparent to clients. They hit the same URLs; the routing layer decides whether the request goes to the monolith or the new service. You can roll back instantly by changing the routing rule.

    ## Extract by Domain, Not by Layer

    A common mistake is extracting by technical layer -- pulling out "the database layer" or "the API layer" into a service. This creates distributed monolith antipatterns where services are tightly coupled and must be deployed together, negating every benefit of microservices.

    Instead, extract by business domain. The payments module becomes the payment service, owning its own database, its own API, and its own deployment pipeline. It encapsulates a complete business capability.

    A good extraction candidate has these traits: it has a well-defined interface with the rest of the system (few, clear integration points), it has different scaling or deployment requirements, and the team owning it would benefit from independent release cycles.

    ## Amazon's Monolith-to-SOA Story

    Amazon's evolution is one of the most cited examples of monolith decomposition. In the early 2000s, Amazon ran on a massive C++ monolith called "Obidos." As the company grew, deploys became agonizing -- a single deploy could take hours, and a failure meant rolling back the entire application. Teams were blocked by each other constantly.

    In 2001-2002, Jeff Bezos issued the now-famous mandate: all teams must expose their functionality through service interfaces, all communication must happen over these interfaces, and there are no exceptions. This kicked off a multi-year migration that eventually produced hundreds of services. The result was that teams could deploy independently, scale independently, and choose their own technology stacks.

    But the critical detail often overlooked is that this migration took years, not months, and was driven by a specific organizational pain point -- team coordination at scale (thousands of engineers). Amazon did not decompose because of technical scaling limits; they decomposed because their organizational structure demanded it.

    ## Warning Signs That It's Time to Split

    Not every monolith needs decomposition. But when multiple of these signals appear simultaneously, it's worth planning a phased extraction:

    - Deploy frequency has dropped below once per week because of coordination overhead across teams
    - Build and test cycles exceed 30-60 minutes, slowing every developer's feedback loop
    - A single component's failure (e.g., a recommendation engine OOM) takes down the entire application
    - Teams of 50+ engineers are stepping on each other's changes daily
    - One domain needs fundamentally different infrastructure (GPU for ML, high-memory for analytics)

    The key word is "simultaneously." Any one of these in isolation can usually be addressed without decomposition. When three or four converge, the monolith's trade-offs have shifted.

---

## Key Takeaways

A monolith is not a compromise -- it is the optimal architecture for most teams most of the time. The simplicity of a single codebase, a single deployment, and in-process communication eliminates entire categories of distributed systems problems. Start monolithic, structure it well with clear module boundaries, and only extract services when you have concrete evidence that the monolith is the bottleneck. Companies like Stack Overflow (1.3 billion monthly page views, 9 web servers), Basecamp (20 years on a Rails monolith), and Shopify (processing billions in transactions before extracting services) demonstrate that a well-operated monolith can scale far beyond what most applications will ever need.

## Related Topics

- [Microservices Architecture](microservices.md) -- when and why to decompose
- [Event-Driven Architecture](event-driven.md) -- decouple components within a monolith
- [Domain-Driven Design](domain-driven-design.md) -- finding the right module/service boundaries
- [Database Sharding](../data/databases/sharding.md) -- scaling the database layer
- [Caching Strategies](../data/caching/strategies.md) -- reducing database load
- [Architecture Interview Questions](interview-questions.md) -- practice for interviews
