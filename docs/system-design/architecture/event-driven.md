# Event-Driven Architecture

Event-driven architecture (EDA) is a design paradigm where components communicate by producing and consuming events rather than making direct requests to one another. An event represents an immutable fact -- something that has already happened -- like "OrderPlaced" or "PaymentProcessed." The producing service publishes the event without knowing or caring who will consume it, and any number of downstream services can react independently. This inversion of control is what distinguishes EDA from traditional request-driven systems: the producer announces what happened rather than instructing another service what to do.

The fundamental shift is philosophical. In request-driven systems, the caller owns the interaction and expects a response. In event-driven systems, the publisher relinquishes control the moment the event leaves. This makes EDA naturally suited for workflows where a single business action triggers multiple independent reactions -- placing an order that simultaneously reserves inventory, charges a credit card, sends a confirmation email, and updates an analytics dashboard.

## Event-Driven vs Request-Driven

| Aspect | Request-Driven | Event-Driven |
|---|---|---|
| Coupling | Caller knows the callee directly | Producer does not know its consumers |
| Communication | Synchronous (request/response) | Asynchronous (fire and forget) |
| Failure impact | Caller blocked if callee is down | Producer unaffected by consumer failure |
| Adding consumers | Requires modifying the caller | Zero changes to the producer |
| Data consistency | Immediate (within a transaction) | Eventual (consumers process asynchronously) |
| Debugging | Linear call chain, easy to trace | Distributed flow, requires correlation IDs |
| Scaling | Caller and callee scale together | Consumers scale independently |
| Best for | Simple CRUD, real-time queries | Multi-step workflows, decoupled reactions |

---

=== "Architecture Pattern"

    ## Producers, Consumers, and Channels

    Every event-driven system has three roles. Producers detect state changes and publish events. Consumers subscribe to event channels and react when relevant events arrive. Channels (topics, queues, or streams provided by a message broker) carry events from producers to consumers. The broker is the infrastructure that makes this decoupling possible -- for broker technology choices, delivery guarantees, and operational trade-offs, see [Messaging Patterns](../communication/messaging/patterns.md).

    Events should be named in the past tense ("OrderPlaced," not "PlaceOrder") to make the semantic distinction clear: they describe facts, not commands. A well-designed event is immutable, self-contained (carries enough data for consumers to act without calling back), and timestamped.

    ## Choreography vs Orchestration

    When multiple services must coordinate to complete a business workflow, there are two approaches to managing the flow.

    **Choreography** is decentralized. Each service listens for events, performs its work, and publishes the next event. There is no central coordinator -- the workflow emerges from the collective behavior of independent services, much like dancers following a shared rhythm without a conductor. Choreography keeps services fully decoupled and is easy to extend (add a new subscriber without modifying anything), but the overall flow becomes implicit and harder to reason about as the number of participants grows.

    ```
    Choreography -- each service reacts and emits the next event:

    OrderService             InventoryService          PaymentService
        |                         |                         |
        |-- OrderPlaced --------->|                         |
        |                         |-- InventoryReserved --->|
        |                         |                         |-- PaymentProcessed
        |                         |                         |       |
        |                    ShippingService <----------------------------
        |                         |
        |                    ShipmentDispatched
    ```

    **Orchestration** is centralized. A dedicated orchestrator service coordinates the workflow by sending commands to each participant and listening for their responses. The orchestrator owns the workflow definition explicitly -- it knows the sequence, handles branching logic, and manages compensation on failure. This makes complex workflows visible and testable but introduces a single point of coordination.

    ```
    Orchestration -- a central coordinator directs each step:

    OrderOrchestrator
        |
        |-- ReserveInventory ----> InventoryService
        |<-- InventoryReserved ---      |
        |
        |-- ProcessPayment ------> PaymentService
        |<-- PaymentProcessed ---       |
        |
        |-- DispatchShipment ----> ShippingService
        |<-- ShipmentDispatched --      |
        |
        v
    OrderConfirmed
    ```

    Neither approach is universally better. Choreography works well when the workflow has few participants (3-4 services), each step is independent, and you want maximum decoupling. Orchestration is preferable when the workflow has many steps, requires complex branching or conditional logic, or when visibility into the end-to-end flow matters more than decoupling.

    ## Netflix: Choreography at Scale

    Netflix's content encoding pipeline is a widely cited example of choreography in production. When new content is ingested, an "AssetIngested" event triggers a cascade of independent processing steps -- video transcoding into dozens of formats, subtitle extraction, thumbnail generation, quality analysis, and metadata enrichment. Each step is handled by a separate service that listens for the relevant upstream event and publishes its own completion event.

    At Netflix's scale (processing thousands of hours of new content weekly across 240+ countries), choreography provides the flexibility to add new processing steps -- say, a new HDR format or an AI-generated preview clip -- without modifying any existing service. The trade-off is that monitoring the end-to-end progress of a single asset requires correlating events across many services, which Netflix handles through a purpose-built pipeline tracking system.

=== "Event Sourcing"

    ## Store Events, Not State

    Traditional systems store the current state of an entity. An order record shows "status: SHIPPED" but says nothing about when it was created, when payment was confirmed, or whether the shipping address was changed between creation and dispatch. Event sourcing inverts this model: instead of storing the latest state, you store every event that ever happened to the entity as an immutable, append-only log.

    The current state is derived by replaying events from the beginning. This is the same principle behind database transaction logs, version control systems, and accounting ledgers -- the balance in your bank account is not stored independently but computed from the sequence of deposits and withdrawals.

    ```
    Traditional state storage (lossy):

    orders table
    +----------+--------+---------+
    | order_id | status | total   |
    +----------+--------+---------+
    | 42       | SHIPPED| $149.99 |   <-- How did we get here?
    +----------+--------+---------+

    Event sourcing (lossless):

    event_store
    +-----+-------------------+---------------------------------+
    | seq | event_type        | data                            |
    +-----+-------------------+---------------------------------+
    |  1  | OrderCreated      | {order:42, total:$149.99}       |
    |  2  | PaymentConfirmed  | {order:42, payment_id:"pay_7"}  |
    |  3  | AddressChanged    | {order:42, new_zip:"94105"}     |
    |  4  | OrderShipped      | {order:42, tracking:"1Z999"}    |
    +-----+-------------------+---------------------------------+

    Current state = replay(event 1 -> 2 -> 3 -> 4)
    State at any point = replay(events up to that point)
    ```

    ## Rebuilding State from the Event Log

    To get the current state of an entity, a projection function folds over the event sequence. Each event type applies a specific transformation to the running state. This is conceptually simple -- a left fold over a list -- but it has powerful implications.

    ```python
    # Pseudocode: rebuild order state from events
    def rebuild(events):
        state = {}
        for e in events:
            if e.type == "OrderCreated":
                state = {"status": "CREATED", "total": e.total}
            elif e.type == "PaymentConfirmed":
                state["status"] = "PAID"
            elif e.type == "OrderShipped":
                state["status"] = "SHIPPED"
        return state
    ```

    Since replaying the full event history for every read would be slow, production systems use **snapshots** -- periodic checkpoints of the computed state. To get the current state, you load the latest snapshot and replay only the events that occurred after it, keeping read latency bounded even as the event log grows to millions of entries.

    ## Benefits

    **Complete audit trail.** Every state transition is recorded with its timestamp, the actor who triggered it, and the full event payload. This is invaluable for financial systems, healthcare platforms, and any domain with regulatory compliance requirements. When an auditor asks "who changed this order and when?", the answer is in the event log.

    **Temporal queries.** Because the full history is preserved, you can reconstruct the state of any entity at any point in time. What was this customer's address when we shipped their order last Tuesday? Replay events up to that timestamp and read the result. This "time travel" capability is impossible with traditional state storage where updates overwrite previous values.

    **New projections without migration.** When business requirements change and you need a new view of existing data -- say, a report on average time between order creation and shipment -- you can build a new projection that replays the event history. No schema migration, no backfill job, no downtime.

    ## Challenges

    **Event schema evolution.** Events are immutable, but the schema of new events will inevitably change as the system evolves. Adding a new field is straightforward (consumers ignore unknown fields), but renaming or removing fields requires a versioning strategy. Common approaches include embedding a schema version in each event and maintaining upcasters that transform old event formats to new ones during replay.

    **Storage growth.** The event log grows indefinitely. While storage is cheap, replaying millions of events to rebuild state is not. Snapshots mitigate this for reads, but the raw log still needs retention policies. Some systems archive events older than a threshold to cold storage while keeping recent events in the hot path.

    **Eventual consistency.** Projections built from the event log are inherently eventually consistent. After a new event is published, there is a window during which the read model has not yet been updated. Systems must be designed to tolerate this delay, typically through optimistic UI updates or read-your-own-writes guarantees at the application layer.

=== "CQRS"

    ## Separate Read and Write Models

    Command Query Responsibility Segregation (CQRS) splits a system into two sides: a **command side** that handles writes (create, update, delete) and a **query side** that handles reads. Each side has its own data model optimized for its purpose. The write model enforces business rules and consistency constraints. The read model is denormalized, pre-joined, and shaped exactly for the queries the UI needs.

    In a traditional architecture, one data model serves both reads and writes, forcing compromises in both directions. The normalized schema that keeps writes consistent is expensive to query. The denormalized view that makes reads fast is hard to keep consistent during writes. CQRS eliminates this tension by letting each side optimize independently.

    ```
    Command side (writes):                Query side (reads):

    Client                                Client
      |                                     |
      |-- CreateOrder (command) -->         |-- GetOrderSummary (query) -->
      |                                     |
    Command Handler                       Query Handler
      |                                     |
      |-- validate business rules           |-- read from optimized view
      |-- write to event store              |
      |-- publish OrderCreated event        Read DB (denormalized)
      |                                     +---------------------------+
    Event Store (source of truth)           | order_id | customer | ... |
    +---+-------------------+---+           | 42       | Alice    | ... |
    | 1 | OrderCreated      |...|           +---------------------------+
    | 2 | PaymentConfirmed  |...|
    +---+-------------------+---+       <-- events projected into
                                            read-optimized views
    ```

    The command side publishes events after every write. A projection process consumes these events and updates the read model. Because the read model is built asynchronously from events, there is a brief delay between a write and its visibility on the read side -- this is the eventual consistency trade-off at the heart of CQRS.

    ## Eventual Consistency Between Models

    The gap between write and read is typically measured in milliseconds to low seconds, but it is never zero. Applications must account for this. The most common strategy is "read your own writes" -- after a user submits a command, the UI optimistically displays the expected result rather than immediately querying the read model. By the time the user navigates away and comes back, the projection has caught up.

    For cases where stronger consistency is required, the command handler can return the result of the write synchronously (bypassing the read model for that one request), or the client can poll until the read model reflects the change. These are pragmatic escape hatches, not violations of the pattern.

    ## When CQRS Is Worth the Complexity

    CQRS adds real complexity: two data models to maintain, a projection pipeline to build and monitor, and eventual consistency to reason about. It is not a default architectural choice -- it pays for itself only in specific scenarios.

    **High read-to-write ratio.** When the system handles 100 reads for every write (product catalogs, social media feeds, dashboards), CQRS lets you scale the read side independently with heavily denormalized, cache-friendly views while keeping the write side lean and consistent.

    **Divergent read and write shapes.** When the data needed for writes (normalized, constrained) looks nothing like the data needed for reads (joined, aggregated, flattened), maintaining a single model is a constant source of friction. CQRS eliminates the compromise.

    **Event-sourced systems.** CQRS is a natural companion to event sourcing. The event store is the write model. Projections built from the event stream are the read models. Together they provide a complete architecture where writes capture intent, events capture history, and projections serve queries.

    **When to avoid it.** Simple CRUD applications where the read and write models are nearly identical gain nothing from CQRS but inherit all the complexity. If your entities map cleanly to database tables and your queries map cleanly to those same tables, a single model is the right choice.

=== "Real-World Patterns"

    ## Saga Pattern (Choreography-Based)

    A saga is a sequence of local transactions across multiple services where each step publishes an event that triggers the next. If any step fails, compensating transactions are executed to undo the work of previous steps. In the choreography-based variant, there is no central coordinator -- each service knows its own compensating action and triggers it when it receives a failure event.

    ```
    Happy path:

    OrderService        PaymentService       InventoryService
        |                    |                     |
        |-- OrderPlaced ---->|                     |
        |                    |-- PaymentCharged -->|
        |                    |                     |-- InventoryReserved
        |                    |                     |       |
        v                    v                     v       v
    Order confirmed     Payment recorded      Stock decremented

    Failure with compensation:

    OrderService        PaymentService       InventoryService
        |                    |                     |
        |-- OrderPlaced ---->|                     |
        |                    |-- PaymentCharged -->|
        |                    |                     |-- OUT OF STOCK!
        |                    |                     |
        |                    |<-- RefundRequired --|
        |                    |-- PaymentRefunded   |
        |<-- OrderFailed ----|                     |
        |                                          |
    Order cancelled     Refund issued         (no reservation)
    ```

    The choreography-based saga keeps services fully decoupled -- each service only knows about its own domain events. However, as the number of participants grows, the compensation logic becomes scattered across services and the failure paths become difficult to test exhaustively. For sagas with more than 4-5 participants, an orchestrator-based saga (where a central coordinator manages the sequence and compensations) is usually easier to maintain.

    ## Transactional Outbox Pattern

    One of the trickiest problems in event-driven systems is ensuring that the database write and the event publication happen atomically. If the service writes to its database and then publishes an event, a crash between the two operations means the database is updated but no event is sent -- downstream services never learn about the change. If the service publishes the event first, a crash before the database write means consumers react to an event that never materialized.

    The outbox pattern solves this by writing the event to an "outbox" table in the same database transaction as the business data. A separate process (a log tailer or polling publisher) reads the outbox table and publishes the events to the broker. Because the business write and the outbox insert are in the same transaction, they either both succeed or both roll back.

    ```
    Without outbox (dual-write problem):

    Service ---> Write to DB     (succeeds)
            ---> Publish event   (fails -- crash!)
            Result: DB updated, but no event sent

    With outbox (atomic):

    Service ---> BEGIN TRANSACTION
            --->   INSERT INTO orders (...)
            --->   INSERT INTO outbox (event_type, payload)
            ---> COMMIT
                        |
    Outbox Publisher --->|  (polls outbox table or tails DB log)
            ---> Publish event to broker
            ---> Mark outbox row as published
    ```

    Debezium (an open-source change data capture tool) is commonly used to implement the outbox pattern by tailing the database transaction log and streaming changes directly to Kafka, eliminating the need for a custom polling process.

    ## Event Schema Registry

    As an event-driven system grows, dozens of services produce and consume hundreds of event types. Without a shared contract, producers can change event schemas in ways that break consumers. A schema registry is a centralized service that stores and validates event schemas, enforcing compatibility rules before a new schema version can be registered.

    Confluent Schema Registry (used with Kafka) is the most widely adopted implementation. It supports Avro, Protobuf, and JSON Schema formats and enforces compatibility modes: backward (new consumers can read old events), forward (old consumers can read new events), and full (both directions). When a producer attempts to register a schema that violates the compatibility rules, the registry rejects it, preventing breaking changes from reaching production.

    Schema evolution best practices include making all new fields optional with sensible defaults, never removing or renaming fields (deprecate them instead), and using a union or oneof type for fields whose meaning may change. These rules keep the event contract stable while allowing the schema to evolve over time.

    ## LinkedIn: Event-Driven Data Pipeline

    LinkedIn's data infrastructure is built around an event-driven pipeline with Kafka at its center. Every user action -- profile views, connection requests, job applications, feed interactions -- is published as an event to Kafka, which serves as the single source of truth for all downstream systems. Search indexes, recommendation engines, analytics dashboards, notification services, and compliance systems all consume from the same event streams independently.

    This architecture processes over 7 trillion messages per day across more than 100,000 topics. The key insight was treating the event pipeline as the integration layer rather than building point-to-point connections between systems. Adding a new downstream consumer (say, a new ML model for job recommendations) requires only subscribing to existing topics -- zero changes to any producer or existing consumer. For more on Kafka's internal architecture and how consumer groups enable this parallel consumption, see [Messaging Patterns -- Event Streaming](../communication/messaging/patterns.md).

    ## When to Use Event-Driven vs Request-Driven

    Event-driven architecture is not a replacement for request-driven communication -- they serve different purposes and most production systems use both.

    **Use event-driven when:**

    - A single action triggers reactions in multiple independent services (fan-out)
    - Services should evolve and deploy independently without coordinating releases
    - Temporal decoupling matters -- the producer should not wait for or depend on consumer availability
    - You need an audit trail or event history for compliance, debugging, or analytics
    - Workloads are bursty and consumers need to process at their own pace

    **Use request-driven when:**

    - The caller needs an immediate, synchronous response (user-facing API calls)
    - The workflow involves a simple, linear chain of 2-3 services
    - Strong consistency is required within a single request
    - The interaction is inherently query-based (read a user profile, fetch search results)

    Most large-scale systems land on a hybrid. Synchronous APIs handle user-facing reads and commands, while events handle the downstream reactions, cross-service coordination, and data pipeline fan-out behind those APIs.

---

## Key Takeaways

1. **Events are facts, not commands.** They describe what already happened ("OrderPlaced") rather than what should happen ("PlaceOrder"). This semantic distinction drives the entire architecture -- producers announce, consumers decide whether and how to react.

2. **Choreography vs orchestration is a spectrum.** Choreography maximizes decoupling but makes workflows implicit. Orchestration makes workflows explicit but introduces a coordinator. Choose based on the number of participants and the complexity of failure handling.

3. **Event sourcing preserves complete history** by storing events instead of current state. It enables audit trails, temporal queries, and new projections without migration, but requires careful handling of schema evolution and storage growth.

4. **CQRS pays for itself only at scale.** When read and write models diverge significantly or the read-to-write ratio is extreme, separating them eliminates painful compromises. For simple CRUD, it adds complexity without benefit.

5. **The outbox pattern solves the dual-write problem.** Writing business data and the outgoing event in the same database transaction guarantees atomicity, with a separate process handling the actual event publication.

6. **Event-driven and request-driven coexist.** Synchronous APIs serve user-facing interactions; events handle fan-out, cross-service coordination, and asynchronous data pipelines. The choice is "both, for different purposes," not "one or the other."

---

## Related Topics

- [Microservices Architecture](microservices.md) -- EDA enables loose coupling between microservices
- [Messaging Patterns](../communication/messaging/patterns.md) -- broker technologies, delivery guarantees, Kafka internals
- [Distributed Systems](../distributed-systems/index.md) -- consistency models and coordination challenges
