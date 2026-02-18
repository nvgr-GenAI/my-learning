# CQRS & Event Sourcing

CQRS (Command Query Responsibility Segregation) separates a system into a write side that handles commands and a read side optimized for queries. Event Sourcing stores state as an immutable sequence of events rather than current state. Together, they provide a powerful architecture for systems that need audit trails, high read-to-write ratios, or divergent read/write models.

---

=== "Event Sourcing"

    ## Store Events, Not State

    Traditional systems store the current state and overwrite previous values. Event sourcing stores every state change as an immutable, append-only event. The current state is derived by replaying events.

    ```
    Traditional (lossy):                Event Sourced (lossless):

    orders table                        event_store
    +----------+--------+---------+     +-----+-------------------+------------------+
    | order_id | status | total   |     | seq | event_type        | data             |
    | 42       | SHIPPED| $149.99 |     |  1  | OrderCreated      | {total:$149.99}  |
    +----------+--------+---------+     |  2  | PaymentConfirmed  | {pay_id:"pay_7"} |
                                        |  3  | AddressChanged    | {zip:"94105"}    |
    How did we get here? 🤷            |  4  | OrderShipped      | {track:"1Z999"}  |
                                        +-----+-------------------+------------------+
                                        State at any point = replay events up to that point
    ```

    This is the same principle behind database transaction logs, Git version control, and accounting ledgers -- your bank balance is computed from deposits and withdrawals, not stored independently.

    ## Rebuilding State

    A projection function folds over the event sequence, applying each event to build current state:

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

    **Snapshots** prevent replaying the full history for every read. Periodic checkpoints store computed state -- load the latest snapshot, then replay only events after it.

    ## Benefits and Challenges

    | Benefit | How It Helps |
    |---------|-------------|
    | **Complete audit trail** | Every state transition recorded with timestamp and actor |
    | **Temporal queries** | Reconstruct state at any point in time ("time travel") |
    | **New projections** | Build new views by replaying history -- no migration needed |
    | **Debugging** | Replay the exact sequence that led to a bug |

    | Challenge | Mitigation |
    |-----------|-----------|
    | **Schema evolution** | Version events, add fields as optional, use upcasters for old formats |
    | **Storage growth** | Snapshots for read performance, archive old events to cold storage |
    | **Eventual consistency** | Projections lag behind writes -- design for this (optimistic UI) |
    | **Complexity** | Only use for domains that genuinely benefit (see decision framework) |

=== "CQRS"

    ## Separate Read and Write Models

    CQRS splits a system into two sides, each with its own data model optimized for its purpose:

    - **Command side (writes):** Enforces business rules, validates constraints, writes to the event store
    - **Query side (reads):** Denormalized, pre-joined, shaped for the exact queries the UI needs

    ```
    Command Flow:                        Query Flow:

    Client                               Client
      |                                    |
      |-- CreateOrder (command) -->        |-- GetOrderSummary (query) -->
      |                                    |
    Command Handler                      Query Handler
      |                                    |
      |-- validate rules                   |-- read from optimized view
      |-- write to event store             |
      |-- publish event                  Read DB (denormalized)
      |                                  +----------+----------+-----+
    Event Store                          | order_id | customer | ... |
    (source of truth)                    | 42       | Alice    | ... |
                                         +----------+----------+-----+
                                            ^
                                            |
                                     Projection process
                                     (consumes events,
                                      updates read model)
    ```

    ## Eventual Consistency Between Models

    The gap between write and read is typically milliseconds to low seconds, but never zero.

    | Strategy | How It Works |
    |----------|-------------|
    | **Read your own writes** | After submit, UI optimistically shows expected result |
    | **Synchronous return** | Command handler returns result directly for that one request |
    | **Polling** | Client polls until read model reflects the change |

    ## When CQRS Is Worth the Complexity

    | Scenario | Worth It? | Why |
    |----------|----------|-----|
    | **100:1 read-to-write ratio** | Yes | Scale reads independently with denormalized views |
    | **Read and write shapes diverge** | Yes | Normalized writes, flattened reads -- no compromise |
    | **Paired with event sourcing** | Yes | Event store = write model, projections = read models |
    | **Simple CRUD (read ≈ write)** | No | Two models for identical shapes = complexity without benefit |
    | **Small team, early stage** | No | Operational overhead outweighs gains |

=== "Together: ES + CQRS"

    ## The Full Architecture

    Event sourcing and CQRS complement each other naturally. The event store is the write model. Projections built from the event stream are the read models.

    ```
    Full ES + CQRS Architecture
    ============================

    [Client]
       |
       +-- Commands -----> [Command Handler]
       |                        |
       |                   Validate rules
       |                        |
       |                   Write to Event Store
       |                        |
       |                   [Event Store] (append-only, source of truth)
       |                        |
       |                   Publish events
       |                        |
       |              +---------+---------+
       |              |                   |
       |         [Projection A]     [Projection B]
       |              |                   |
       |         [Read Model A]     [Read Model B]
       |         (order summary)    (analytics view)
       |              |                   |
       +-- Queries -> [Query Handler A]   [Query Handler B]
    ```

    **Key insight:** Write once to the event store, build as many read models as you need. Adding a new dashboard or report = add a new projection that replays the event history.

    ## Real-World Examples

    | Company | How They Use ES + CQRS | Scale |
    |---------|----------------------|-------|
    | **Banks** | Transaction log IS the event store; account balance is a projection | Every bank, by regulation |
    | **Stripe** | Payment events stored immutably; multiple read views for merchants, analytics, compliance | Billions of payments |
    | **Event stores** | EventStoreDB, Axon Framework, Marten (.NET) are purpose-built | Dedicated tooling |

    ### E-Commerce Order Flow Example

    ```
    Events stored:                Read models built:

    1. OrderCreated               Customer Dashboard:
    2. ItemAdded (x3)              - "Order #42: 3 items, $149.99, SHIPPED"
    3. PaymentConfirmed
    4. AddressChanged             Warehouse View:
    5. OrderShipped                - "Order #42: pick items A,B,C from bin 7"
    6. DeliveryConfirmed
                                  Finance Report:
                                   - "Order #42: $149.99 revenue, $12 shipping cost"

                                  Same events, different projections for different needs
    ```

=== "Decision Framework"

    ## Should You Use Event Sourcing? CQRS? Both?

    ```
    Do you need a complete audit trail?
        |
        +-- No --> Do reads and writes have very different shapes?
        |              |
        |              +-- No --> Simple CRUD (no ES, no CQRS)
        |              |
        |              +-- Yes --> CQRS only (separate read/write models)
        |
        +-- Yes --> Event Sourcing
                     |
                     +-- Is read:write ratio > 10:1?
                          |
                          +-- No --> Event Sourcing only
                          |          (single read model from projections)
                          |
                          +-- Yes --> Event Sourcing + CQRS
                                     (multiple optimized read models)
    ```

    ## Comparison with Traditional Architecture

    | Aspect | Traditional CRUD | CQRS Only | Event Sourcing Only | ES + CQRS |
    |--------|-----------------|-----------|-------------------|-----------|
    | Data model | Single | Separate read/write | Event log + projection | Event log + multiple projections |
    | History | Lost on update | Lost on write side | Complete | Complete |
    | Read performance | Compromised | Optimized | Good | Highly optimized |
    | Complexity | Low | Medium | Medium-High | High |
    | Best for | Simple apps | High read ratio | Audit/compliance | All of the above |

    ## Common Pitfalls

    | Pitfall | Problem | Solution |
    |---------|---------|---------|
    | ES for everything | Simple CRUD gets complex for no benefit | Reserve for core domains |
    | Huge events | Events carry too much data, hard to evolve | Small, focused events with just what changed |
    | No snapshots | Replaying millions of events per read | Snapshot every N events |
    | Ignoring eventual consistency | UI shows stale data, user confused | Optimistic UI, read-your-own-writes |
    | Event store as message bus | Conflating storage with messaging | Separate event store from message broker |

---

## Key Takeaways

| Concept | Summary |
|---------|---------|
| **Event Sourcing** | Store events, not state. Current state = replay events. Complete history preserved. |
| **CQRS** | Separate write model (commands + rules) from read model (denormalized views). |
| **Together** | Event store = write model; projections from events = read models. Write once, read many ways. |
| **When to use** | Audit trails, high read:write ratio, divergent read/write shapes, temporal queries. |
| **When to skip** | Simple CRUD, small team, read and write models are identical. |

---

## Related Topics

- [Event-Driven Architecture](event-driven.md) -- Events, choreography, sagas, outbox pattern
- [Data Consistency](../fundamentals/data-consistency.md) -- Eventual vs strong consistency models
- [Messaging Patterns](../communication/messaging/patterns.md) -- Kafka, delivery guarantees, consumer groups
- [Domain-Driven Design](domain-driven-design.md) -- Aggregates and domain events
