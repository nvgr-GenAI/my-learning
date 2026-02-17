# Distributed Transactions

When a single operation spans multiple services or databases, you face the hardest
problem in distributed systems: ensuring all participants either commit or abort
together. In a monolith, the database gives you ACID transactions for free. In a
distributed system, there is no single authority that can enforce atomicity across
network boundaries.

Consider booking a vacation: you need to reserve a flight, a hotel, and a rental car.
If the hotel booking fails after the flight is confirmed, you need to undo the flight
reservation. This is trivial with one database but extraordinarily complex when each
service owns its own data store. Distributed transactions are the set of patterns
that solve this coordination problem — each with very different trade-offs.

---

## Why Distributed Transactions Are Hard

The fundamental challenge is the absence of shared state. In a single database, the
transaction manager can lock rows, write to a log, and commit atomically. Across
services, you have:

- **No shared lock manager** — each service controls its own locks
- **No shared transaction log** — each database has its own WAL
- **Unreliable networks** — messages can be lost, delayed, or duplicated
- **Independent failure modes** — service A can crash while B succeeds

```
MONOLITH: Single database, single transaction
┌─────────────────────────────────────┐
│  BEGIN TRANSACTION                   │
│    UPDATE flights SET status='booked'│
│    UPDATE hotels SET status='booked' │
│    UPDATE cars SET status='booked'   │
│  COMMIT  ← one atomic operation     │
└─────────────────────────────────────┘

MICROSERVICES: Three databases, three potential outcomes
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Flight   │   │ Hotel    │   │ Car      │
│ Service  │   │ Service  │   │ Service  │
│ ✅ booked │   │ ❌ failed │   │ ✅ booked │
│          │   │          │   │          │
│ flights  │   │ hotels   │   │ cars     │
│ DB       │   │ DB       │   │ DB       │
└──────────┘   └──────────┘   └──────────┘
   Now what? Flight and car are booked but hotel failed.
   System is in an inconsistent state.
```

The CAP theorem tells us we cannot have consistency, availability, and partition
tolerance simultaneously. Distributed transaction patterns navigate this constraint
by choosing different trade-offs along the consistency-availability spectrum.

---

=== "Two-Phase Commit (2PC)"

    ## Two-Phase Commit

    The oldest and most straightforward approach. A central **coordinator** orchestrates
    all participants through two phases to ensure atomic commit or abort.

    ### How It Works

    ```
    PHASE 1: PREPARE (Voting)
    ┌─────────────┐
    │ Coordinator  │──── "Can you commit?" ────→ Participant A
    │              │──── "Can you commit?" ────→ Participant B
    │              │──── "Can you commit?" ────→ Participant C
    │              │
    │  Collects    │←─── "Yes, prepared" ──────  Participant A
    │  votes       │←─── "Yes, prepared" ──────  Participant B
    │              │←─── "Yes, prepared" ──────  Participant C
    └─────────────┘

    PHASE 2: COMMIT (Decision)
    ┌─────────────┐
    │ Coordinator  │──── "COMMIT" ─────────────→ Participant A
    │  (all voted  │──── "COMMIT" ─────────────→ Participant B
    │   yes)       │──── "COMMIT" ─────────────→ Participant C
    └─────────────┘

    If ANY participant votes "No" in Phase 1:
    ┌─────────────┐
    │ Coordinator  │──── "ABORT" ──────────────→ All participants
    └─────────────┘
    ```

    **Phase 1 — Prepare:** The coordinator asks each participant if it can commit. Each
    participant executes the transaction locally, acquires locks, writes to its WAL, but
    does not commit. It responds "prepared" (yes) or "abort" (no).

    **Phase 2 — Commit/Abort:** If all participants voted yes, the coordinator sends
    "commit" and each participant makes the change permanent. If any participant voted
    no (or timed out), the coordinator sends "abort" and all participants roll back.

    ### The Blocking Problem

    2PC's critical weakness is that it is a **blocking protocol**. If the coordinator
    crashes after Phase 1 but before sending Phase 2 decisions, participants that voted
    "prepared" are stuck — they hold locks and cannot independently decide to commit or
    abort. They must wait for the coordinator to recover.

    ```
    COORDINATOR FAILURE SCENARIO

    Phase 1 complete: all participants voted YES
    Coordinator writes "COMMIT" to its log... then CRASHES

    Participant A: "I voted YES, holding locks, waiting..."
    Participant B: "I voted YES, holding locks, waiting..."
    Participant C: "I voted YES, holding locks, waiting..."

    ⏳ All participants BLOCKED until coordinator recovers
       Locks held → other transactions cannot proceed
       Could be minutes, hours, or longer
    ```

    ### When to Use 2PC

    | Strength | Limitation |
    |---|---|
    | Strong consistency — all or nothing | Blocking on coordinator failure |
    | Simple to reason about | High latency (2 round trips minimum) |
    | Works with existing databases (XA) | Coordinator is single point of failure |
    | Guaranteed atomicity | Locks held for entire protocol duration |

    **Good fit:** Banking transfers between accounts in the same institution, database
    migrations, systems where consistency is non-negotiable and latency tolerance is high.

    **Poor fit:** Cross-service transactions in microservices, high-throughput systems,
    anything requiring sub-100ms latency.

    **Real-world usage:** MySQL and PostgreSQL support XA transactions (the standardized
    2PC protocol). Google Spanner uses a variant of 2PC combined with TrueTime to achieve
    globally consistent transactions across datacenters — but this requires specialized
    hardware (atomic clocks and GPS receivers) that most organizations cannot replicate.

=== "Three-Phase Commit (3PC)"

    ## Three-Phase Commit

    3PC adds a **pre-commit** phase between prepare and commit to solve the blocking
    problem of 2PC. The idea: if the coordinator crashes, participants that received the
    pre-commit message know the coordinator intended to commit, so they can proceed
    independently.

    ```
    2PC:  PREPARE ──────────────────→ COMMIT
          (can abort)                 (committed, no going back)
                      ↑ GAP: crash here = blocked

    3PC:  PREPARE ──→ PRE-COMMIT ──→ COMMIT
          (can abort)  (intent to     (committed)
                        commit)
                      ↑ crash here = participants can decide
    ```

    ### Why 3PC Is Rarely Used

    While 3PC solves the blocking problem in theory, it introduces a new problem: in the
    presence of **network partitions**, 3PC can lead to inconsistent decisions. One group
    of participants might commit while another aborts if they cannot communicate during
    the pre-commit phase.

    Since network partitions are inevitable in real systems (unlike coordinator crashes,
    which are rare), 3PC trades one problem for a worse one. This is why virtually all
    production systems use 2PC (for strong consistency) or Saga (for availability), and
    3PC remains primarily academic.

    | Property | 2PC | 3PC |
    |---|---|---|
    | Blocking on coordinator failure | Yes | No |
    | Safe under network partition | Yes | No — can diverge |
    | Round trips | 2 | 3 (higher latency) |
    | Practical adoption | Very high | Almost none |

=== "Saga Pattern"

    ## Saga Pattern

    Sagas take a fundamentally different approach: instead of trying to make a distributed
    transaction atomic, break it into a sequence of **local transactions** where each step
    has a **compensating transaction** that can undo it if a later step fails.

    The key insight: in many business scenarios, temporary inconsistency is acceptable as
    long as the system eventually reaches a consistent state. A customer seeing a brief
    "processing" status is fine; a permanently inconsistent booking is not.

    ### How Sagas Work

    ```
    FORWARD FLOW (happy path):
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ T1: Book │───→│ T2: Book │───→│ T3: Book │───→│ T4: Charge│
    │ Flight   │    │ Hotel    │    │ Car      │    │ Payment   │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    COMPENSATION FLOW (T3 fails):
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ C2: Cancel│←──│ C1: Cancel│←──│ T3: Book │ FAILED
    │ Hotel     │   │ Flight   │   │ Car      │
    └──────────┘    └──────────┘    └──────────┘
    Each completed step is undone in reverse order.
    ```

    Each step (T1, T2, T3...) has a compensating action (C1, C2, C3...) defined in
    advance. If step Tn fails, compensations are executed in reverse: C(n-1), C(n-2),
    ..., C1.

    ### Choreography vs Orchestration

    There are two ways to coordinate saga steps:

    ```
    CHOREOGRAPHY (event-driven, decentralized)
    ┌──────────┐  FlightBooked  ┌──────────┐  HotelBooked  ┌──────────┐
    │ Flight   │───────────────→│ Hotel    │───────────────→│ Car      │
    │ Service  │                │ Service  │                │ Service  │
    └──────────┘                └──────────┘                └──────────┘
    Each service listens for events and decides what to do next.
    Simple for 2-3 steps, becomes spaghetti for complex flows.

    ORCHESTRATION (centralized coordinator)
    ┌──────────────────────┐
    │   Saga Orchestrator   │
    │   (knows full flow)   │
    └───┬──────┬──────┬────┘
        │      │      │
        ▼      ▼      ▼
    ┌──────┐┌──────┐┌──────┐
    │Flight││Hotel ││Car   │
    └──────┘└──────┘└──────┘
    Orchestrator tells each service what to do.
    Easier to understand and debug complex flows.
    ```

    | Aspect | Choreography | Orchestration |
    |---|---|---|
    | Coupling | Loose — services only know events | Medium — orchestrator knows all services |
    | Complexity | Grows rapidly with steps | Linear growth |
    | Debugging | Hard — distributed event chain | Easy — centralized flow |
    | Single point of failure | None | Orchestrator (mitigated with replication) |
    | Best for | Simple 2-4 step flows | Complex multi-step business processes |

    ### Compensating Transactions in Practice

    Not all actions are easily reversible. The compensation strategy depends on the
    nature of the operation:

    | Action | Compensation | Difficulty |
    |---|---|---|
    | Reserve a hotel room | Cancel reservation | Easy — soft state |
    | Send an email notification | Send correction email | Medium — can't unsend |
    | Charge a credit card | Issue a refund | Medium — takes days |
    | Ship a physical package | Initiate return | Hard — real-world action |
    | Post to social media | Delete post | Medium — may have been seen |

    **Semantic compensation**: Sometimes the compensation is not an exact undo but a
    business-level correction. Charging then refunding a credit card is not the same as
    never charging — there is a temporary hold on the customer's funds. The business
    must decide if this is acceptable.

    ### Real-World Saga Implementations

    **Uber** uses an orchestrated saga for ride completion: authorize payment → complete
    ride → charge rider → pay driver. If charging the rider fails, the system
    compensates by reversing the driver payment and flagging the ride for manual review.

    **Amazon** order processing is a saga: validate order → reserve inventory → process
    payment → initiate shipping. Each step can be compensated. If payment fails after
    inventory is reserved, the reservation is released.

    **Temporal.io** and **Netflix Conductor** are popular open-source saga orchestration
    frameworks. They handle retries, timeouts, and compensation automatically, letting
    developers focus on business logic rather than distributed coordination.

=== "Event Sourcing + CQRS"

    ## Event Sourcing and CQRS

    Event sourcing takes a radically different approach to distributed consistency:
    instead of storing the current state, store the sequence of **events** that led to
    that state. The current state is derived by replaying events from the beginning.

    ### Event Sourcing

    ```
    TRADITIONAL (state-based):
    ┌─────────────────────────────┐
    │ Account #123                │
    │ Balance: $750               │  ← only current state stored
    │ Last updated: 2024-03-15   │
    └─────────────────────────────┘

    EVENT SOURCING (event-based):
    ┌─────────────────────────────────────────┐
    │ Event Log for Account #123              │
    │                                         │
    │ 1. AccountOpened   { balance: $0 }      │
    │ 2. MoneyDeposited  { amount: $1000 }    │
    │ 3. MoneyWithdrawn  { amount: $200 }     │
    │ 4. MoneyWithdrawn  { amount: $50 }      │
    │                                         │
    │ Current balance = replay events = $750  │
    └─────────────────────────────────────────┘
    ```

    **Why this helps distributed transactions:** Events are immutable facts. Instead of
    coordinating a distributed commit, services publish events to a shared log (like
    Kafka). Other services consume events and update their own state. If something goes
    wrong, you can replay events, publish corrective events, or rebuild state from any
    point in time.

    **Trade-offs:**

    - Complete audit trail — you know exactly how you reached the current state
    - Can rebuild any past state by replaying to a point in time
    - Event log grows indefinitely — requires snapshotting for performance
    - Eventual consistency — read models may lag behind the event log
    - Schema evolution is complex — old events must remain readable

    ### CQRS (Command Query Responsibility Segregation)

    CQRS separates the write model (commands) from the read model (queries). Writes go
    to the event store; reads come from denormalized views optimized for specific queries.

    ```
    ┌────────────┐         ┌───────────────┐
    │  Commands   │────────→│  Event Store   │
    │ (writes)    │         │ (append-only)  │
    └────────────┘         └───────┬───────┘
                                    │ events published
                                    ▼
                           ┌───────────────┐
                           │  Projections   │
                           │ (build views)  │
                           └───────┬───────┘
                                    │
    ┌────────────┐         ┌───────▼───────┐
    │  Queries    │←────────│  Read Models   │
    │ (reads)     │         │ (denormalized) │
    └────────────┘         └───────────────┘
    ```

    This pattern works well with distributed transactions because each service can
    maintain its own read model by consuming events from the shared log. There is no
    need for distributed locking — services are eventually consistent by design.

    **Real-world:** The UK banking system **Monzo** uses event sourcing for all account
    transactions. Every balance change is an event, enabling complete audit trails and
    the ability to reconstruct any account's history. **LinkedIn** uses CQRS to separate
    the high-write-throughput activity feed from the read-optimized profile pages.

---

## Pattern Comparison

Choosing the right pattern depends on your consistency requirements, latency tolerance,
and operational complexity budget.

| Property | 2PC | Saga | Event Sourcing |
|---|---|---|---|
| **Consistency** | Strong (ACID) | Eventual | Eventual |
| **Latency** | High (2+ round trips, locks held) | Low per step | Low (async) |
| **Availability** | Lower (blocking) | Higher | Higher |
| **Complexity** | Low concept, medium ops | Medium concept, high ops | High concept, high ops |
| **Failure handling** | Coordinator recovery | Compensating transactions | Event replay |
| **Audit trail** | None built-in | Partial (compensation log) | Complete (event log) |
| **Scalability** | Limited (lock contention) | Good | Excellent |
| **Best for** | Financial transactions, same-org DBs | Microservice workflows | Audit-critical, analytics-heavy |

### Decision Guide

```
Do you need STRONG consistency across services?
├── YES: Can you tolerate higher latency and lower availability?
│   ├── YES → Two-Phase Commit (2PC)
│   └── NO → Reconsider: can you redesign to avoid distributed transactions?
│
└── NO: Eventual consistency is acceptable?
    ├── Is the workflow a linear sequence of steps?
    │   ├── YES, 2-4 simple steps → Saga (Choreography)
    │   ├── YES, 5+ or complex steps → Saga (Orchestration)
    │   └── NO → Event Sourcing + CQRS
    │
    └── Do you need complete audit trail / time travel?
        ├── YES → Event Sourcing
        └── NO → Saga is simpler
```

---

## Real-World Architecture Examples

### Stripe Payment Processing

Stripe processes millions of payments daily using an orchestrated saga. When you call
their API to create a payment:

1. **Validate** — check card details, merchant account (local)
2. **Authorize** — place hold with card network (external)
3. **Capture** — transfer funds (may happen later)
4. **Settle** — reconcile with bank (batch, async)

If authorization succeeds but capture fails, Stripe automatically reverses the
authorization hold. Each step is idempotent — retrying a step produces the same result,
which is critical for reliable compensation.

### Airbnb Booking

Airbnb's booking flow involves multiple services that must coordinate:

1. Hold dates on the listing calendar
2. Calculate pricing with dynamic rates
3. Process payment through payment service
4. Confirm reservation and notify host

They use an orchestrated saga with Temporal as the workflow engine. If payment fails,
the calendar hold is released. If the host declines (after payment authorization), the
payment authorization is reversed and the guest is notified. Each step publishes events
that other systems (analytics, messaging, trust & safety) consume asynchronously.

### Banking: Inter-Bank Transfers

Traditional banks use 2PC (via XA transactions) for transfers within the same
institution. For inter-bank transfers, they use a saga-like pattern through the SWIFT
network or ACH system — funds are debited from one account and credited to another as
separate operations, with reconciliation processes handling failures. This is why
inter-bank transfers take 1-3 business days: the compensation and reconciliation
windows are built into the timeline.

---

## Key Takeaways

1. **Avoid distributed transactions when possible.** The best distributed transaction
   is the one you don't need. Redesign service boundaries so that operations that must
   be atomic live within a single service.

2. **2PC gives strong consistency at the cost of availability and latency.** Use it
   when correctness is non-negotiable and you control all participants (same
   organization, same datacenter).

3. **Sagas trade consistency for availability.** They allow each service to remain
   available and responsive, but the system may be temporarily inconsistent between
   steps. Design compensations carefully — some actions are difficult or impossible to
   undo.

4. **Event sourcing provides the strongest audit trail** but is the most complex to
   implement and operate. Use it when you genuinely need time-travel debugging, complete
   audit logs, or the ability to rebuild state from events.

5. **Idempotency is non-negotiable.** Regardless of the pattern, every operation must
   be safe to retry. Networks are unreliable, and messages will be delivered more than
   once. Design every handler to produce the same result whether called once or ten
   times.

6. **The choice is often "no distributed transaction."** In many cases, you can
   restructure your data model or service boundaries to eliminate the need entirely.
   Amazon processes orders by having a single Order service own the entire order
   lifecycle, coordinating with other services via async events rather than distributed
   transactions.

---

## Related Topics

- [Consensus Algorithms](consensus.md) — how nodes agree on state, the foundation underneath 2PC
- [Data Consistency](../fundamentals/data-consistency.md) — consistency models from strong to eventual
- [Messaging Patterns](../communication/messaging/patterns.md) — event buses and message queues that power sagas
- [CAP Theorem](../fundamentals/cap-theorem.md) — the theoretical framework for consistency trade-offs
- [Fault Tolerance](../reliability/fault-tolerance.md) — handling failures that trigger compensations
