# Microservices Architecture

Microservices architecture structures an application as a collection of small, autonomous services organized around business capabilities. Each service owns its own data, runs in its own process, and communicates with other services through well-defined APIs. The approach trades the simplicity of a single deployable unit for the ability to scale, deploy, and evolve parts of a system independently. Netflix runs over 800 microservices handling millions of requests per second; Amazon decomposes its retail platform into hundreds of services, each owned by a small team. But microservices are not a starting point -- they are a response to organizational and technical scaling pressures that monoliths struggle to absorb.

---

=== "Core Concepts"

    ## Service Boundaries and Bounded Contexts

    The hardest part of microservices is deciding where one service ends and another begins. Domain-Driven Design provides the concept of a bounded context: a boundary within which a particular domain model applies consistently. A "customer" in the billing context (payment methods, invoices, credit limits) is not the same as a "customer" in the shipping context (addresses, delivery preferences, tracking history). Forcing both into a single shared model creates coupling that undermines the independence microservices promise.

    Good service boundaries align with business capabilities, not technical layers. A service should encapsulate a complete business function -- the data it needs, the logic it applies, and the API it exposes. When two services need to change together for most features, they probably belong as one service.

    | Service | Owns | Does NOT Own |
    |---------|------|--------------|
    | User Service | Authentication, profiles, preferences | Product catalog |
    | Product Service | Catalog, search, pricing | Order lifecycle |
    | Order Service | Order creation, status, history | Payment processing |
    | Payment Service | Charge, refund, ledger | Shipping logistics |

    ## Independent Deployment

    Each service can be built, tested, and deployed on its own schedule without coordinating with other teams. This is the single most valuable property of microservices. When the payments team fixes a bug, they ship it in minutes without waiting for the catalog team to finish their sprint. When a deployment fails, only that service rolls back -- the rest of the system continues operating.

    Independent deployment requires strict API contracts between services. A new version of the product service must not break the order service. This is typically enforced through contract testing and backward-compatible API evolution (adding fields rather than removing them, versioning endpoints when breaking changes are unavoidable).

    ## Technology Diversity

    Because services communicate only through APIs, each team can choose the technology that best fits their problem. A recommendation engine might use Python for its machine learning ecosystem while the payment service runs on Java for its mature financial libraries. The search service might store data in Elasticsearch while the user service uses PostgreSQL.

    | Service | Language | Database | Why |
    |---------|----------|----------|-----|
    | Search | Python | Elasticsearch | Full-text search, ML ranking |
    | Users | Java | PostgreSQL | Relational data, ACID guarantees |
    | Analytics | Go | Cassandra | High write throughput, time-series |
    | Sessions | Node.js | Redis | In-memory speed, low latency |

    This freedom comes with a cost: the organization must support multiple technology stacks, build pipelines, and on-call expertise. Most companies limit diversity to a handful of approved stacks rather than allowing unlimited choice.

    ## Netflix: Microservices at Scale

    Netflix is the canonical example of microservices done well. Their architecture handles over 200 million subscribers across 190 countries, serving personalized content through 800+ microservices.

    ```
    Netflix Microservices Architecture (simplified)
    ================================================

    Mobile/TV/Browser Clients
              |
              v
    +-------------------+
    |    API Gateway     |  (Zuul - handles 50B+ requests/day)
    |  Auth / Routing /  |
    |  Rate Limiting     |
    +--------+----------+
             |
      +------+------+------+------+--------+
      |      |      |      |      |        |
      v      v      v      v      v        v
    +----+ +----+ +----+ +----+ +------+ +------+
    |User| |Cat-| |Play| |Rec-| |Search| |Bill- |
    |Svc | |alog| |back| |omm.| | Svc  | |ing   |
    +----+ +----+ +----+ +----+ +------+ +------+
      |      |      |      |      |        |
      v      v      v      v      v        v
    [Cass] [Cass] [Cass] [Cass] [Elastic] [MySQL]
    ```

    Key architectural decisions at Netflix include Chaos Monkey (randomly killing service instances in production to ensure resilience), Hystrix for circuit breaking, and a strong culture of "you build it, you run it" where each team operates their own services. Their investment in tooling -- including Eureka for service discovery, Ribbon for client-side load balancing, and Zuul for API gateway -- was so significant that they open-sourced the entire Netflix OSS stack.

=== "Communication"

    ## Synchronous vs Asynchronous Communication

    Services must talk to each other, and the choice between synchronous and asynchronous communication shapes the entire system's behavior. Synchronous calls (REST, gRPC) are simpler to reason about: service A calls service B, waits for a response, and continues. Asynchronous communication (events, messages) decouples services in time: service A publishes an event and moves on without waiting.

    Neither approach is universally better. Most real systems use both -- synchronous for queries where a user is waiting for a response, asynchronous for workflows where multiple services need to react to a change.

    | Aspect | Synchronous (REST/gRPC) | Asynchronous (Events) |
    |--------|-------------------------|----------------------|
    | Coupling | Temporal -- caller waits | Decoupled -- fire and forget |
    | Latency | Additive across call chain | Independent per consumer |
    | Failure | Cascading (A down = B stuck) | Isolated (retry from queue) |
    | Debugging | Direct stack trace | Requires distributed tracing |
    | Best for | User-facing queries | Background workflows, fan-out |

    ## REST and gRPC

    REST over HTTP is the default choice for synchronous communication. It is universally understood, easy to debug with standard tools (curl, browser), and works across any language. The trade-off is performance: JSON serialization, HTTP overhead, and text-based parsing add latency.

    gRPC uses Protocol Buffers for binary serialization and HTTP/2 for transport, delivering roughly 7-10x better throughput than REST for internal service calls. It also provides strong typing through `.proto` definitions and supports bidirectional streaming. The trade-off is debuggability -- binary payloads are opaque without tooling, and browser support requires a proxy layer.

    ```
    REST vs gRPC decision
    =====================
    External API (browser/mobile clients) --> REST (broad compatibility)
    Internal service-to-service            --> gRPC (performance, typing)
    Streaming data (real-time feeds)       --> gRPC streaming
    Simple CRUD with low traffic           --> REST (simpler tooling)
    ```

    At Google, gRPC handles billions of internal RPCs per second across their infrastructure. Internally, most large-scale companies (Uber, Square, Netflix) use gRPC between services while exposing REST to external clients.

    ## Service Discovery

    In a dynamic environment where services scale up and down, hardcoding addresses is not viable. Service discovery solves this by maintaining a registry of available service instances. When service A needs to call service B, it queries the registry to find a healthy instance.

    Client-side discovery (using tools like Netflix Eureka or HashiCorp Consul) has the client query the registry directly and choose an instance. Server-side discovery (the Kubernetes approach) uses DNS or a load balancer so the client simply calls a stable hostname and the platform routes to a healthy instance.

    ```
    Server-Side Discovery (Kubernetes Model)
    =========================================

    Order Service                    Kubernetes DNS
         |                                |
         |-- GET product-service/api ---->|
         |                                |
         |                     +----------+----------+
         |                     |    kube-proxy /     |
         |                     |    service mesh     |
         |                     +---+-----+-----+----+
         |                         |     |     |
         |                         v     v     v
         |                       [P1]  [P2]  [P3]
         |                       Product Service Pods
         |<--- response ---------|
    ```

    ## API Gateway Pattern

    An API gateway sits between clients and the microservices, providing a single entry point that handles cross-cutting concerns: authentication, rate limiting, request routing, response aggregation, and protocol translation.

    Without a gateway, each client must know about every service, handle authentication separately with each one, and make multiple round trips for a single page load. A mobile app displaying an order summary might need data from the user service, order service, and product service -- the gateway aggregates these into a single response.

    ```
    Request Flow Through API Gateway
    =================================

    Mobile App
       |
       |  POST /api/orders  (single request)
       v
    +---------------------------+
    |       API Gateway         |
    |  1. Validate JWT token    |
    |  2. Check rate limit      |
    |  3. Route to service      |
    |  4. Aggregate responses   |
    +--+--------+--------+-----+
       |        |        |
       v        v        v
    +------+ +------+ +------+
    | User | |Order | | Prod |
    | Svc  | | Svc  | | Svc  |
    +------+ +------+ +------+
       |        |        |
       v        v        v
    [Users] [Orders] [Products]
       DB       DB       DB
    ```

    Popular API gateway implementations include Kong, AWS API Gateway, and Envoy (often paired with Istio). Netflix's Zuul gateway handles over 50 billion requests per day, performing dynamic routing, monitoring, and security filtering.

    ## Events Between Microservices

    When a state change in one service is relevant to others, events provide loose coupling. The order service publishes an "OrderCreated" event; the inventory, payment, email, and analytics services each consume it independently. Adding a new consumer (say, a loyalty points service) requires zero changes to the order service.

    Event-driven communication between microservices is covered briefly here since it is a communication mechanism. For deeper coverage of event patterns (event sourcing, CQRS, event notification vs event-carried state transfer), see the dedicated [Event-Driven Architecture](event-driven.md) page.

    ```python
    # Publisher: Order Service emits an event after creating an order
    event_bus.publish("OrderCreated", {
        "order_id": "ord-789",
        "user_id": "usr-456",
        "total": 149.99
    })
    ```

=== "Data Management"

    ## Database Per Service

    The database-per-service pattern gives each microservice exclusive ownership of its data store. No other service may read from or write to another service's database directly -- all access goes through the owning service's API. This rule feels restrictive, but it is the foundation that makes independent deployment possible. If two services share a database, a schema change in one can break the other, recreating the coupling that microservices exist to eliminate.

    ```
    Shared Database (anti-pattern)        Database Per Service
    =============================         ====================

    +------+ +------+ +------+            +------+ +------+ +------+
    | User | | Prod | |Order |            | User | | Prod | |Order |
    | Svc  | | Svc  | | Svc  |            | Svc  | | Svc  | | Svc  |
    +--+---+ +--+---+ +--+---+            +--+---+ +--+---+ +--+---+
       |        |        |                    |        |        |
       +--------+--------+                    v        v        v
                |                          [Users] [Products] [Orders]
                v                            DB       DB        DB
           [Shared DB]
           (coupled!)
    ```

    Each service can choose the database technology that fits its workload. The product search service might use Elasticsearch for full-text queries while the order service uses PostgreSQL for transactional guarantees. This is only possible when databases are isolated.

    The cost is that data that was once a simple JOIN across tables now requires API calls between services or denormalized copies maintained through events. This trade-off is worthwhile at scale but painful for small teams.

    ## Saga Pattern for Distributed Transactions

    In a monolith, placing an order might wrap inventory reservation, payment processing, and order creation in a single ACID transaction. In microservices, each step lives in a different service with its own database -- there is no shared transaction manager.

    The saga pattern solves this by breaking a distributed transaction into a sequence of local transactions, each followed by an event that triggers the next step. If any step fails, compensating transactions undo the previous steps in reverse order.

    ```
    Order Saga: Happy Path
    ======================
    Order Svc          Inventory Svc       Payment Svc
        |                   |                   |
        |-- Create Order -->|                   |
        |   (PENDING)       |                   |
        |                   |-- Reserve Stock -->|
        |                   |   (RESERVED)      |
        |                   |                   |-- Charge Card
        |                   |                   |   (PAID)
        |<-------- OrderConfirmed --------------|

    Order Saga: Payment Fails (Compensation)
    =========================================
    Order Svc          Inventory Svc       Payment Svc
        |                   |                   |
        |-- Create Order -->|                   |
        |   (PENDING)       |                   |
        |                   |-- Reserve Stock -->|
        |                   |   (RESERVED)      |
        |                   |                   |-- Charge Card
        |                   |                   |   FAILED!
        |                   |<-- Release Stock --|
        |<-- Cancel Order --|   (compensate)    |
        |   (CANCELLED)     |                   |
    ```

    There are two saga coordination styles. In choreography, each service listens for events and decides what to do next -- simple but hard to track across many services. In orchestration, a central saga coordinator directs each step -- easier to understand but introduces a single point of coordination.

    ## Uber's Saga Implementation

    Uber processes millions of rides daily, each involving driver matching, fare calculation, payment processing, and receipt generation across separate services. Their CADENCE workflow engine (now open-sourced as Temporal) orchestrates these sagas. When a payment fails after a ride completes, the system automatically triggers fare adjustment, driver compensation recalculation, and rider notification -- all as compensating actions within the saga. At Uber's scale (25 million rides per day at peak), choreography-based sagas became unmanageable, which drove their investment in orchestration tooling.

    ## Event-Driven Consistency

    When services own separate databases, the system cannot provide strong consistency across service boundaries. Instead, microservices embrace eventual consistency: after a state change in one service, other services will eventually reflect that change, but not instantaneously.

    Consider a product price update. The product service changes the price in its database and publishes a "PriceChanged" event. The order service, which caches product prices for display, receives the event and updates its local copy. During the propagation delay (typically milliseconds to seconds), a customer might see the old price on the order page. The system must be designed to handle this window gracefully -- for example, by validating the price at checkout time against the source of truth.

    ```
    Eventual Consistency Timeline
    =============================
    T=0ms   Product Svc updates price to $79.99
    T=5ms   "PriceChanged" event published to broker
    T=15ms  Order Svc receives event, updates local cache
    T=15ms+ All services consistent

    During T=0 to T=15ms, Order Svc still shows old price.
    Design for this: validate at checkout, not at browse time.
    ```

    The key insight is to identify which operations require strong consistency (keep them within a single service) and which can tolerate eventual consistency (spread them across services). Payment charging and order creation should happen in a saga with compensation logic. Updating a recommendation feed or analytics dashboard can lag by seconds without harm.

=== "Operational Complexity"

    ## Service Mesh

    As the number of microservices grows, each service needs retry logic, circuit breaking, mutual TLS, and traffic management. Implementing these in every service's application code leads to duplication and inconsistency. A service mesh moves this logic into a sidecar proxy (like Envoy) deployed alongside each service instance.

    ```
    Service Mesh Architecture (Istio)
    ==================================

    +------+  +-------+      +-------+  +------+
    | User |  | Envoy |<---->| Envoy |  | Prod |
    | Svc  |--| proxy |      | proxy |--| Svc  |
    +------+  +-------+      +-------+  +------+
                  ^               ^
                  |               |
              +---+---------------+---+
              |    Control Plane      |
              |  (Istio / Linkerd)    |
              |  - mTLS certificates  |
              |  - Traffic rules      |
              |  - Retry policies     |
              +-----------------------+
    ```

    The sidecar proxy intercepts all network traffic to and from the service, applying policies defined centrally in the control plane. This means a new service automatically gets mTLS encryption, circuit breaking, and observability without writing a single line of infrastructure code. Istio and Linkerd are the most widely adopted service mesh implementations. At Lyft, Envoy (which they created) handles all inter-service traffic for hundreds of services.

    ## Distributed Tracing

    When a single user request flows through five or more services, debugging a slow response or an error requires visibility across the entire call chain. Distributed tracing solves this by assigning a unique trace ID at the entry point and propagating it through every service call.

    ```
    Distributed Trace: GET /api/order/789
    =======================================
    Trace ID: abc-123

    API Gateway       |====|                          45ms
      User Svc          |==|                          20ms
      Order Svc            |========|                 80ms
        Product Svc           |====|                  40ms
        Payment Svc              |==|                 15ms
                     0   20  40  60  80  100  120ms

    Total: 120ms
    Bottleneck: Order Svc (80ms) --> investigate DB query
    ```

    Tools like Jaeger, Zipkin, and AWS X-Ray collect and visualize these traces. Combined with centralized logging (ELK stack, Datadog) and metrics (Prometheus, Grafana), they form the "three pillars of observability" that are non-negotiable for running microservices in production. Google's Dapper paper, which described their internal distributed tracing system, inspired most of these open-source tools.

    ## Deployment and Orchestration

    Microservices multiply the operational surface area. Instead of deploying one application, you deploy tens or hundreds. Container orchestration platforms like Kubernetes have become the standard solution, providing automated deployment, scaling, self-healing, and service discovery.

    A typical deployment pipeline for a microservice includes: build the container image, run unit and integration tests, push to a container registry, deploy to a staging environment, run end-to-end tests against other services, then gradually roll out to production using canary or blue-green deployment. Kubernetes handles the rollout, monitoring health checks and automatically rolling back if the new version fails.

    ```
    Deployment Pipeline
    ===================
    Code Push --> Build Image --> Unit Tests --> Push to Registry
                                                      |
                    Production  <-- Canary (5%)  <-- Staging
                    (if healthy)    (monitor)        (e2e tests)
    ```

    At Spotify, each of their 800+ microservices deploys independently through automated pipelines, with teams shipping multiple times per day. Their Backstage platform (now open-sourced) provides a service catalog so teams can discover, understand, and manage the growing number of services.

    ## When Microservices Are Wrong

    Microservices introduce distributed systems complexity: network failures, data consistency challenges, operational overhead, and debugging difficulty. This complexity is only justified when the benefits -- independent scaling, autonomous teams, fault isolation -- outweigh the costs.

    For teams smaller than about 10 people, a well-structured monolith (or modular monolith) is almost always the better choice. The coordination overhead that microservices solve simply does not exist in a small team. Shopify ran a monolithic Ruby on Rails application serving billions of dollars in transactions before selectively extracting services. Basecamp deliberately stays monolithic, arguing that their team size does not justify the distributed systems tax.

    The typical progression looks like this:

    ```
    Monolith-to-Microservices Evolution
    ====================================
    Year 1-2:  Monolith
               - Learn the domain, iterate fast
               - 5-15 developers, single deploy

    Year 2-3:  Modular Monolith
               - Clear module boundaries internally
               - Separate databases per module
               - 15-30 developers

    Year 3+:   Selective Microservice Extraction
               - Extract modules that need independent scaling
               - Extract when team ownership boundaries are clear
               - 30+ developers across multiple teams
    ```

    ## Amazon's Two-Pizza Teams

    Amazon pioneered the organizational model that makes microservices work. Jeff Bezos mandated in 2002 that all teams must communicate through service interfaces -- no direct database access, no shared memory, no backdoors. Each service is owned by a team small enough to be fed by two pizzas (typically 6-8 people).

    This organizational rule drove architectural decisions. A two-pizza team owns everything about their service: development, testing, deployment, monitoring, and on-call support. The team has full autonomy over technology choices within their service boundary. This model scaled Amazon from a single monolithic bookstore application to hundreds of services -- and eventually led to AWS, when they realized their internal infrastructure services could be offered externally.

    The lesson is that microservices are as much an organizational pattern as a technical one. Conway's Law states that system architecture mirrors team communication structures. If you want independent services, you need independent teams.

---

## Key Takeaways

Microservices decompose an application along business capability boundaries, giving each service its own database, deployment pipeline, and owning team. The architecture enables independent scaling (Netflix scales its recommendation engine separately from its streaming pipeline), fault isolation (a payment outage does not take down product browsing), and team autonomy (Amazon's two-pizza teams ship independently). The costs are substantial: distributed data consistency requires saga patterns, debugging requires distributed tracing, and operational overhead requires container orchestration and service mesh infrastructure. Start with a monolith, extract services when team coordination becomes the bottleneck, and invest heavily in observability from day one.

---

## Related Topics

- [Monolithic Architecture](monolithic.md) -- when and why to avoid microservices
- [Event-Driven Architecture](event-driven.md) -- asynchronous communication patterns in depth
- [Domain-Driven Design](domain-driven-design.md) -- bounded contexts for service boundaries
- [Service Mesh](service-mesh.md) -- infrastructure layer for service-to-service communication
- [API Design](../communication/api-design/index.md) -- designing service interfaces
- [Distributed Systems](../distributed-systems/index.md) -- consistency, failure, and coordination challenges
- [Architecture Interview Questions](interview-questions.md) -- practice for interviews
