# Event-Driven Architecture

Event-driven architecture (EDA) is a design paradigm where components communicate by producing and consuming events rather than making direct requests. An event represents an immutable fact -- something that has already happened -- like "OrderPlaced" or "PaymentProcessed." The producing service publishes the event without knowing who will consume it, and any number of downstream services can react independently.

## Event-Driven vs Request-Driven

| Aspect | Request-Driven | Event-Driven |
|--------|---------------|-------------|
| Coupling | Caller knows the callee | Producer doesn't know consumers |
| Communication | Synchronous (request/response) | Asynchronous (fire and forget) |
| Failure impact | Cascading (A down = B stuck) | Isolated (retry from queue) |
| Adding consumers | Requires modifying the caller | Zero changes to producer |
| Consistency | Immediate (within a transaction) | Eventual (async processing) |
| Debugging | Linear call chain | Requires correlation IDs |
| Best for | User-facing queries, simple CRUD | Multi-step workflows, fan-out |

Most large-scale systems use **both** -- synchronous APIs for user-facing reads, events for downstream reactions and cross-service coordination.

---

=== "Architecture Pattern"

    ## Producers, Consumers, and Channels

    Every EDA system has three roles:

    - **Producers** detect state changes and publish events
    - **Consumers** subscribe and react when relevant events arrive
    - **Channels** (topics/queues/streams) carry events via a message broker

    Events should be named in the **past tense** ("OrderPlaced," not "PlaceOrder") -- they describe facts, not commands. A well-designed event is immutable, self-contained, and timestamped.

    For broker technologies, delivery guarantees, and operational trade-offs, see [Messaging Patterns](../communication/messaging/patterns.md).

    ## Choreography vs Orchestration

    When multiple services must coordinate a business workflow, there are two approaches:

    **Choreography** -- decentralized. Each service reacts and emits the next event. No central coordinator.

    ```
    OrderService             InventoryService          PaymentService
        |                         |                         |
        |-- OrderPlaced --------->|                         |
        |                         |-- InventoryReserved --->|
        |                         |                         |-- PaymentProcessed
        |                    ShippingService <---------------------------
    ```

    **Orchestration** -- centralized. A coordinator directs each step and handles failures.

    ```
    OrderOrchestrator
        |-- ReserveInventory ----> InventoryService
        |<-- InventoryReserved
        |-- ProcessPayment ------> PaymentService
        |<-- PaymentProcessed
        |-- DispatchShipment ----> ShippingService
        v
    OrderConfirmed
    ```

    | Aspect | Choreography | Orchestration |
    |--------|-------------|--------------|
    | Coupling | Fully decoupled | Central coordinator |
    | Visibility | Workflow is implicit | Workflow is explicit |
    | Adding steps | Subscribe without changes | Modify orchestrator |
    | Failure handling | Scattered across services | Centralized |
    | Best for | < 4 participants, simple flows | Complex flows, many steps |

    **Netflix** uses choreography for content encoding -- "AssetIngested" triggers dozens of parallel steps (transcoding, subtitles, thumbnails) across independent services. Adding a new step (e.g., AI preview clip) requires zero changes to existing services.

=== "Key Patterns"

    ## Saga Pattern (Distributed Transactions)

    A saga breaks a distributed transaction into local transactions, each followed by an event. If any step fails, compensating transactions undo previous steps.

    ```
    Happy path:
    OrderService      PaymentService     InventoryService
        |-- OrderPlaced -->|                    |
        |                  |-- PaymentCharged ->|
        |                  |                    |-- InventoryReserved
        v                  v                    v
    Confirmed          Recorded             Decremented

    Payment fails (compensation):
    OrderService      PaymentService     InventoryService
        |-- OrderPlaced -->|                    |
        |                  |-- PaymentCharged ->|
        |                  |                    |-- OUT OF STOCK!
        |                  |<-- RefundRequired -|
        |<-- OrderFailed --|-- PaymentRefunded  |
    Cancelled          Refunded            (no reservation)
    ```

    **Uber** processes millions of rides daily using orchestrated sagas via Cadence/Temporal. Each ride involves matching, fare calculation, payment, and receipt -- with compensations for every failure path.

    ## Transactional Outbox Pattern

    Solves the dual-write problem: ensuring the DB write and event publication happen atomically.

    ```
    Without outbox (dual-write problem):
    Service ---> Write to DB     (succeeds)
            ---> Publish event   (fails -- crash!)
    Result: DB updated but no event sent

    With outbox (atomic):
    Service ---> BEGIN TRANSACTION
            --->   INSERT INTO orders (...)
            --->   INSERT INTO outbox (event_type, payload)
            ---> COMMIT

    Outbox Publisher ---> polls outbox / tails DB log
            ---> Publish event to broker
            ---> Mark row as published
    ```

    **Debezium** (open-source CDC tool) implements this by tailing the DB transaction log and streaming changes directly to Kafka.

    ## Event Schema Registry

    As the system grows, dozens of services produce hundreds of event types. A schema registry stores and validates event schemas, enforcing compatibility rules.

    - **Backward compatible:** New consumers can read old events
    - **Forward compatible:** Old consumers can read new events
    - **Best practices:** All new fields optional with defaults, never remove/rename fields

    Confluent Schema Registry (with Kafka) is the most widely adopted implementation.

=== "Event Sourcing & CQRS"

    EDA pairs naturally with two advanced patterns. This section provides an overview -- for deep coverage, see [CQRS & Event Sourcing](cqrs-event-sourcing.md).

    ## Event Sourcing (Brief Overview)

    Instead of storing current state, store every event that happened as an immutable log. Current state = replay all events.

    ```
    Traditional (lossy):              Event sourced (lossless):

    orders table                      event_store
    | order_id | status  |           | seq | event_type       | data         |
    | 42       | SHIPPED |           |  1  | OrderCreated     | {total:$150} |
                                     |  2  | PaymentConfirmed | {pay:"p7"}   |
    How did we get here? 🤷          |  3  | OrderShipped     | {track:"1Z"} |
                                     Complete history ✓
    ```

    **Benefits:** Complete audit trail, temporal queries ("state at any point in time"), new projections without migration.

    **Challenges:** Event schema evolution, storage growth (mitigated by snapshots), eventual consistency.

    ## CQRS (Brief Overview)

    Separate the write model (commands, business rules, event store) from the read model (denormalized, query-optimized views).

    ```
    Commands → Write Model → Event Store → Projections → Read Model → Queries
    ```

    **Worth it when:** Read:write ratio is 100:1, read and write shapes diverge, or paired with event sourcing. **Overkill for:** Simple CRUD where read/write models are identical.

    **Full coverage:** [CQRS & Event Sourcing](cqrs-event-sourcing.md)

=== "Real-World Examples"

    ## LinkedIn: Event Pipeline at Scale

    LinkedIn's data infrastructure centers on Kafka as the single event backbone. Every user action (profile views, connections, job applications) is published as an event.

    - **7 trillion messages/day** across 100,000+ topics
    - Search, recommendations, analytics, notifications all consume independently
    - Adding a new consumer = subscribe to existing topics, zero producer changes

    ## When to Use Event-Driven vs Request-Driven

    **Use event-driven when:**

    - One action triggers reactions in multiple independent services (fan-out)
    - Services should deploy independently without coordinating releases
    - Producer should not wait for or depend on consumer availability
    - You need an audit trail or event history
    - Workloads are bursty -- consumers process at their own pace

    **Use request-driven when:**

    - Caller needs an immediate synchronous response
    - Simple linear chain of 2-3 services
    - Strong consistency required within a single request
    - Inherently query-based (fetch user profile, search results)

---

## Key Takeaways

1. **Events are facts, not commands.** Named in past tense ("OrderPlaced"), producers announce, consumers decide how to react.
2. **Choreography vs orchestration is a spectrum.** Choreography maximizes decoupling; orchestration makes workflows explicit. Choose by participant count and failure complexity.
3. **The outbox pattern solves dual-write.** Write business data + event in one DB transaction; a separate process publishes to the broker.
4. **Event sourcing preserves complete history** but requires handling schema evolution and storage growth.
5. **Event-driven and request-driven coexist.** Synchronous for user-facing; events for fan-out and async coordination.

---

## Related Topics

- [CQRS & Event Sourcing](cqrs-event-sourcing.md) -- Deep dive into separate read/write models and event stores
- [Microservices Architecture](microservices.md) -- EDA enables loose coupling between microservices
- [Messaging Patterns](../communication/messaging/patterns.md) -- Broker technologies, delivery guarantees, Kafka internals
- [Distributed Transactions](../distributed-systems/distributed-transactions.md) -- Saga patterns and consistency
