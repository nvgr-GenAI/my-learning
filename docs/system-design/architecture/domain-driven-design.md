# Domain-Driven Design

Domain-Driven Design (DDD) places the business domain at the center of every architectural decision. For system design, DDD's strategic patterns -- bounded contexts, context mapping, and ubiquitous language -- answer the question that determines the success of any distributed system: **where do you draw the lines between services?**

```
The Core Problem DDD Solves
============================

Without DDD (technical boundaries):        With DDD (domain boundaries):

  +-------------+  +---------------+          +-------------------+
  |  UI Layer   |  |  Validation   |          |   Order Context   |
  | (all pages) |  |   Service     |          | UI + Logic + Data |
  +------+------+  +-------+-------+          +-------------------+
         |                 |
  +------+------+  +-------+-------+          +-------------------+
  | Business    |  |   Database    |          |  Shipping Context |
  | Logic Layer |  |   Service     |          | UI + Logic + Data |
  +-------------+  +---------------+          +-------------------+

  Every feature touches all layers         Each context is self-contained
```

---

=== "Strategic Design"

    ## Bounded Contexts

    A bounded context defines a boundary within which a domain model is consistent. The same real-world concept means different things in different parts of a business -- bounded contexts make those differences explicit.

    | Context | What "Product" Means | Key Attributes |
    |---------|---------------------|----------------|
    | **Catalog** | Something customers browse | Name, description, images, categories |
    | **Inventory** | Something in a warehouse | SKU, quantity, location, reorder threshold |
    | **Pricing** | Something with a price strategy | Base price, discounts, tax rules |
    | **Shipping** | Something with physical dimensions | Weight, dimensions, fragility |
    | **Recommendations** | Something with behavioral signals | View count, purchase frequency |

    A single "Product" model serving all five contexts produces a bloated entity every team must coordinate on. Bounded contexts let each team define exactly the model they need.

    ### How to Identify Bounded Contexts

    | Heuristic | What to Look For |
    |-----------|-----------------|
    | **Language shifts** | Same word means different things to different people |
    | **Independent lifecycles** | Pricing rules change weekly, catalog structure quarterly |
    | **Org boundaries** | Separate teams own separate business functions (Conway's Law) |
    | **Consistency requirements** | Catalog tolerates staleness; inventory cannot |

    ## Ubiquitous Language

    Within each bounded context, the team uses a single, precise vocabulary shared between developers, domain experts, and codebase.

    | Principle | Example |
    |-----------|---------|
    | Code mirrors business terms | `ShippingLabel`, not `OutputDoc47` |
    | One term, one meaning per context | "Shipment" means one thing in Shipping, another in Returns |
    | Domain experts can read the code | `order.confirm()`, not `order.setState(STATE_2)` |

    ## Context Mapping

    Bounded contexts must communicate. Context mapping defines how they relate -- who depends on whom and how models translate at boundaries.

    ```
    Context Map for E-Commerce
    ===========================

    +----------+                    +----------+
    | Catalog  |---[Published   ]---| Search   |
    | Context  |   [Language    ]   | Context  |
    +----+-----+                    +----------+
         |
         | [Customer-Supplier]
         |
    +----+-----+        +-----------+
    |  Order   |---[ACL]---| Payment  |
    | Context  |           | Gateway  |
    +----+-----+           +-----------+
         |
         | [Conformist]
         |
    +----+-----+
    |  Carrier  |
    |   API     |
    +-----------+
    ```

=== "Context Relationships"

    ## Relationship Patterns

    | Pattern | Power Dynamic | Coupling | Best For |
    |---------|--------------|----------|----------|
    | **Shared Kernel** | Equal partnership, co-ownership | High | Closely related teams sharing Address, Money |
    | **Customer-Supplier** | Upstream accommodates downstream | Medium | Internal teams with goodwill |
    | **Conformist** | Upstream dictates, take it or leave it | High | External APIs (FedEx, Stripe) you can't influence |
    | **Anti-Corruption Layer** | Downstream protects itself | Low | Legacy systems, unstable external APIs |
    | **Published Language** | Stable documented API for many consumers | Low | Platform services, public APIs |

    ## Anti-Corruption Layer (Most Important for Interviews)

    The ACL translates an external model into your internal model, isolating your domain from changes you don't control.

    ```
    External Payment Gateway             Your Order Context

    +--------------------+          +----------------------------+
    |  PaymentIntent     |          |  Anti-Corruption Layer     |
    |  {                 |   API    |                            |
    |    id: "pi_1abc",  +--------->|  "pi_1abc"   -> ref       |
    |    amount: 9999,   |          |  9999 cents  -> $99.99    |
    |    currency: "usd" |          |  "succeeded" -> PAID      |
    |  }                 |          +----------------------------+
    +--------------------+                    |
                                              v
                                    Payment (internal model)
                                    { reference, amount: $99.99, status: PAID }
    ```

    **When to use:** Integrating with any external system whose model is different from yours, unstable, or poorly designed. Almost always worth it for third-party integrations.

=== "Tactical Patterns"

    Tactical DDD provides patterns for modeling *within* a bounded context. Less critical for interviews than strategic patterns, but understanding aggregates is essential.

    ## Aggregates: Consistency Boundaries

    An aggregate is a cluster of domain objects treated as a single unit for data changes. The root entity enforces all business invariants.

    **Key insight:** An aggregate defines the boundary of a transaction. Changes within one aggregate = strongly consistent (ACID). Changes across aggregates = eventually consistent (via domain events).

    ```
    Aggregate: Order
    =================
    +-----------------------------------------------+
    |  Order (Aggregate Root)                       |
    |    - orderId, status, placedAt                |
    |                                               |
    |    +-------------------+  +----------------+  |
    |    | OrderLine         |  | OrderLine      |  |
    |    |  productId, qty   |  |  productId, qty|  |
    |    +-------------------+  +----------------+  |
    |                                               |
    |    +-------------------+                      |
    |    | ShippingAddress   |  (value object)      |
    |    +-------------------+                      |
    +-----------------------------------------------+

    Invariants enforced by root:
      - total = sum(line.qty * line.price)
      - Cannot add lines to a SHIPPED order
      - Max 50 lines per order
    ```

    **Sizing guideline:** Make aggregates as small as possible while enforcing invariants that must be immediately consistent.

    | Too Large | Right Sized |
    |-----------|-------------|
    | Customer contains Orders, Addresses, PaymentMethods | Customer = profile fields only |
    | Changing address locks all orders | Order, Address, PaymentMethod are separate aggregates |

    ## Entities vs Value Objects

    | Aspect | Entity | Value Object |
    |--------|--------|-------------|
    | Identity | Has unique ID | Defined by attributes |
    | Equality | Same ID = same entity | Same attributes = same value |
    | Mutability | Can change over time | Immutable (replace, don't modify) |
    | Examples | Order, Customer, Shipment | Money, Address, DateRange |

    ## Domain Events

    Past-tense facts that enable loose coupling: `OrderPlaced`, `PaymentConfirmed`, `ShipmentDispatched`. Primary mechanism for communication between aggregates and between bounded contexts.

=== "DDD + Microservices"

    ## Bounded Context = Service Boundary

    **Each microservice should align with one bounded context.** This prevents the distributed monolith -- services that are technically separate but so coupled they must deploy together.

    ```
    Domain Model (DDD)              Deployment (Microservices)

    +-------------------+           +-------------------+
    | Catalog Context   | ========> | Catalog Service   |
    | - Product model   |           | - Own database    |
    +-------------------+           +-------------------+

    +-------------------+           +-------------------+
    | Order Context     | ========> | Order Service     |
    | - Order model     |           | - Own database    |
    +-------------------+           +-------------------+

    +-------------------+           +-------------------+
    | Payment Context   | ========> | Payment Service   |
    | - Payment model   |           | - Own database    |
    +-------------------+           +-------------------+
    ```

    | Distributed Monolith Symptom | DDD Solution |
    |-----------------------------|-------------|
    | Changing one service requires changing three others | Each context owns its model completely |
    | Shared database between services | Database per bounded context |
    | Every feature requires cross-team coordination | Contexts aligned with teams (Conway's Law) |
    | Deploying A breaks B | ACL or Published Language at boundaries |

=== "Real-World Examples"

    ## Amazon: Two-Pizza Teams

    Bezos's 2002 API mandate effectively created bounded contexts by organizational policy. Each team (6-10 people) owns a context end-to-end.

    | Bounded Context | Scale |
    |----------------|-------|
    | Product Catalog | 350M+ products |
    | Search & Discovery | Billions of queries/day |
    | Pricing | Price changes every 10 minutes |
    | Fulfillment | 200+ fulfillment centers |
    | Payments | $600B+ annually |

    **Key insight:** Org structure drives bounded contexts, not the other way around.

    ## Netflix: Domain-Driven Decomposition

    800+ microservices grouped into domain umbrellas. A bounded context can contain multiple services.

    ```
    Content Domain: [Content Mgmt] [Encoding] [Delivery]
    Streaming Domain: [Playback] [Adaptive Bitrate] [Device Support]
    Member Domain: [Profiles] [Preferences] [Auth]
    Billing Domain: [Subscription] [Payment] [Invoice]
    ```

    ## Uber: DOMA (2020)

    Re-organized 2,200 services using DDD principles:

    - **Domain gateways** -- each domain has a single entry point (Open Host Service)
    - **Extension points** -- domains register logic via hooks, not direct calls (ACL)
    - **Clear ownership** -- every service belongs to exactly one domain

=== "Common Mistakes"

    | Mistake | Problem | Fix |
    |---------|---------|-----|
    | **Technical boundaries** | "Notification Service" called by everyone | Each context sends its own notifications |
    | **Nano-services** | Service per entity (CustomerService, AddressService) | Context = business capability, not DB table |
    | **Giant contexts** | "Commerce" context with 50 developers | Split when different parts have different release cadences |
    | **Ignoring ubiquitous language** | `txn_type` instead of `payment_method` | Code should match domain terms |
    | **DDD everywhere** | Aggregates for a simple CRUD admin panel | Reserve DDD for core domains with complex rules |

    ## When Is DDD Worth It?

    ```
    Is the domain complex? (many rules, edge cases)
        |
        +-- No → Simple CRUD / framework conventions
        |
        +-- Yes → Multiple teams?
                    |
                    +-- No → Tactical DDD only (aggregates in a monolith)
                    +-- Yes → Strategic DDD (bounded contexts, context mapping)
    ```

    | System Type | DDD Investment | Return |
    |------------|---------------|--------|
    | Internal tool / admin panel | Not worth it | Use simple CRUD |
    | E-commerce platform | Strategic DDD | High -- prevents distributed monolith |
    | Financial trading system | Full DDD | Very high -- domain rules ARE the product |
    | Simple REST API | Not worth it | Negative -- overhead exceeds benefit |

---

## Key Takeaways

| Concept | One-Sentence Summary |
|---------|---------------------|
| **Bounded Context** | Boundary where a domain model is consistent -- foundation of good service boundaries |
| **Context Mapping** | Explicitly define relationships and power dynamics between contexts |
| **Ubiquitous Language** | Shared vocabulary within a context used in code, conversations, and docs |
| **Aggregate** | Consistency boundary within a context -- unit of transactional integrity |
| **Anti-Corruption Layer** | Translation layer protecting your model from external models |
| **Domain Events** | Past-tense facts enabling loose coupling between aggregates and contexts |

**The fundamental lesson:** The most expensive mistake in distributed systems is drawing service boundaries in the wrong place. DDD provides discipline for finding boundaries that align with the business.

---

## Related Topics

- [Microservices Architecture](microservices.md) -- Service decomposition and communication
- [Event-Driven Architecture](event-driven.md) -- Domain events, choreography vs orchestration
- [Monolithic Architecture](monolithic.md) -- Why modular monolith is often the right first step
- [CQRS & Event Sourcing](cqrs-event-sourcing.md) -- Separate read/write models
