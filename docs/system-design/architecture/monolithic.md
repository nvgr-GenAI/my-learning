# Monolithic Architecture

**Single deployable unit** | 🏢 Traditional | 📦 All-in-One | ⚡ Simple Start

---

## Overview

A monolithic architecture is a traditional software design pattern where all components of an application are built as a single, unified codebase and deployed as one unit.

**Key Characteristic:** Everything runs in a single process - UI, business logic, and data access layer are tightly coupled.

---

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│       Monolithic Application            │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Presentation Layer          │   │
│  │  (UI Components, Controllers)   │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │      Business Logic Layer       │   │
│  │  (Services, Domain Logic)       │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │     Data Access Layer           │   │
│  │  (Repositories, ORMs)           │   │
│  └─────────────────────────────────┘   │
│                 ↓                       │
└─────────────────┼───────────────────────┘
                  ↓
         ┌────────────────┐
         │    Database    │
         └────────────────┘
```

---

## Characteristics

=== "Deployment"
    **Single Deployment Unit**

    - Entire application packaged as one artifact (JAR, WAR, EXE)
    - Deploy all or nothing - no partial updates
    - Single process handles all requests
    - Easier to deploy initially but harder to scale

    ```bash
    # Example deployment
    java -jar myapp.jar
    # or
    docker run myapp:latest
    ```

=== "Development"
    **Unified Codebase**

    - All code in one repository
    - Shared data models and libraries
    - Easy to refactor across boundaries
    - IDE-friendly with full code navigation

    ```
    myapp/
    ├── src/
    │   ├── controllers/
    │   ├── services/
    │   ├── models/
    │   └── repositories/
    ├── tests/
    └── config/
    ```

=== "Data"
    **Single Database**

    - One shared database for entire application
    - ACID transactions work naturally
    - No distributed data concerns
    - Schema changes affect entire app

    ```sql
    -- Direct JOINs work seamlessly
    SELECT o.*, c.name, p.title
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN products p ON o.product_id = p.id
    ```

=== "Communication"
    **In-Process Calls**

    - Method/function calls (no network overhead)
    - Shared memory space
    - Fast and reliable
    - Strong type safety

    ```java
    // Direct method call - microsecond latency
    User user = userService.getUserById(123);
    Order order = orderService.createOrder(user, items);
    ```

---

## When to Use Monolithic

| Scenario | Recommended? | Reason |
|----------|--------------|--------|
| **New startup/MVP** | ✅ Yes | Fast development, simple deployment |
| **Small team (< 10)** | ✅ Yes | Easier coordination, less complexity |
| **Simple domain** | ✅ Yes | Unnecessary to split into services |
| **Unclear boundaries** | ✅ Yes | Premature splitting adds risk |
| **Large team (> 50)** | ❌ No | Coordination becomes bottleneck |
| **Multiple products** | ❌ No | Different release cycles conflict |
| **Scale 10M+ users** | ⚠️ Maybe | Vertical scaling sufficient? |

---

## Advantages

### ✅ **Simplicity**
- **Easy to develop**: Standard MVC/layered architecture
- **Easy to test**: Run entire app locally
- **Easy to deploy**: Single artifact to production
- **Easy to debug**: Full stack trace in one place

### ✅ **Performance**
- **No network latency**: In-process method calls
- **No serialization overhead**: Direct object passing
- **ACID transactions**: Database guarantees consistency
- **Efficient**: Single process, shared resources

### ✅ **Development Speed (initially)**
- **Fast iteration**: Change multiple components together
- **IDE support**: Full refactoring capabilities
- **Shared code**: Reuse models and utilities
- **Single codebase**: One place to look

---

## Disadvantages

### ❌ **Scaling Challenges**
- **Vertical scaling only**: Must scale entire app, even if only one part is bottleneck
- **Resource waste**: Can't scale features independently
- **Limited by single machine**: Hardware ceiling

```
Problem: Only checkout needs more capacity

Monolith Solution: Scale entire app (wasteful)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Full App     │  │ Full App     │  │ Full App     │
│ - Catalog    │  │ - Catalog    │  │ - Catalog    │
│ - Cart       │  │ - Cart       │  │ - Cart       │
│ - Checkout   │  │ - Checkout   │  │ - Checkout   │
│ - Inventory  │  │ - Inventory  │  │ - Inventory  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### ❌ **Maintenance Burden**
- **Codebase grows**: 100K+ lines become hard to navigate
- **Tight coupling**: Changes ripple across modules
- **Slow builds**: 10+ minute compile times
- **Test suite slowdown**: Hours to run all tests

### ❌ **Deployment Risk**
- **All or nothing**: Small change requires full redeployment
- **Downtime risk**: Bug in one feature breaks entire app
- **Rollback complexity**: Must rollback everything
- **Coordination required**: All teams deploy together

### ❌ **Team Scaling**
- **Merge conflicts**: Multiple teams editing same codebase
- **Coordination overhead**: Cross-team dependencies
- **Technology lock-in**: Must use same language/framework
- **Bottlenecks**: Shared database, shared resources

---

## Real-World Examples

=== "E-commerce Monolith"
    **Typical Structure:**

    ```
    ecommerce-app/
    ├── modules/
    │   ├── catalog/
    │   │   ├── ProductController.java
    │   │   ├── ProductService.java
    │   │   └── ProductRepository.java
    │   ├── cart/
    │   │   ├── CartController.java
    │   │   └── CartService.java
    │   ├── checkout/
    │   │   ├── CheckoutController.java
    │   │   ├── PaymentService.java
    │   │   └── OrderService.java
    │   └── inventory/
    │       ├── InventoryService.java
    │       └── InventoryRepository.java
    └── shared/
        ├── models/
        ├── utils/
        └── config/
    ```

    **Data Flow:**
    ```java
    @RestController
    public class CheckoutController {
        @Autowired ProductService productService;
        @Autowired CartService cartService;
        @Autowired PaymentService paymentService;
        @Autowired InventoryService inventoryService;

        @PostMapping("/checkout")
        public Order checkout(String userId) {
            Cart cart = cartService.getCart(userId);
            List<Product> products = productService.getProducts(cart.getItemIds());

            // Direct method calls - all in same process
            inventoryService.reserve(products);
            Payment payment = paymentService.process(cart.getTotal());
            Order order = orderService.create(cart, payment);

            return order;
        }
    }
    ```

=== "WordPress"
    **Classic PHP Monolith:**

    - Single PHP application
    - All plugins run in same process
    - MySQL database handles everything
    - Powers 40% of the web

    **Why it works:**
    - Most sites don't need microservices
    - Simple hosting requirements
    - Rich ecosystem of plugins
    - Easy for non-developers to manage

=== "Shopify (Started as Monolith)"
    **Evolution:**

    ```
    2004-2015: Ruby on Rails Monolith
    ├── Handled millions of merchants
    ├── Single database (sharded)
    └── Vertical scaling + optimization

    2015+: Gradual decomposition
    ├── Extracted payment processing
    ├── Extracted shipping service
    └── Core remains monolithic
    ```

    **Key Insight:** Started monolithic, scaled to billions in revenue before needing microservices.

---

## Scaling Strategies

### 1. **Vertical Scaling**
Scale up the machine (more CPU, RAM, storage)

```
Before: 4 CPU, 8GB RAM  →  After: 32 CPU, 128GB RAM
```

**Pros:** Simple, no code changes
**Cons:** Hardware limits, expensive, single point of failure

### 2. **Horizontal Scaling (Replicas)**
Run multiple copies behind a load balancer

```
        ┌─────────────────┐
        │  Load Balancer  │
        └────────┬────────┘
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
    ┌──────┐ ┌──────┐ ┌──────┐
    │ App  │ │ App  │ │ App  │
    │ Copy │ │ Copy │ │ Copy │
    └───┬──┘ └───┬──┘ └───┬──┘
        └────────┼────────┘
                 ↓
          ┌────────────┐
          │  Database  │
          └────────────┘
```

**Requirements:**
- Stateless application (session in Redis/database)
- Database can handle increased connections
- Load balancer distributes traffic

### 3. **Database Optimization**
- **Caching:** Redis/Memcached for read-heavy workloads
- **Read replicas:** Route reads to replicas
- **Sharding:** Partition data across multiple databases
- **Indexing:** Optimize query performance

### 4. **Modular Monolith**
Structure code as independent modules within the monolith

```java
// Well-defined boundaries
package com.ecommerce.catalog;     // Catalog module
package com.ecommerce.cart;        // Cart module
package com.ecommerce.checkout;    // Checkout module

// Communication via interfaces
public interface CatalogService {
    Product getProduct(String id);
}

// Easier to extract later if needed
```

---

## Migration Patterns

### From Monolith to Microservices

**Strategy: Strangler Fig Pattern**

```
Phase 1: Identify boundaries
┌─────────────────────┐
│     Monolith        │
│ ┌─────────────────┐ │
│ │ Payment (heavy) │←─── Extract this
│ ├─────────────────┤ │
│ │ Cart            │ │
│ ├─────────────────┤ │
│ │ Catalog         │ │
│ └─────────────────┘ │
└─────────────────────┘

Phase 2: Extract service
┌─────────────────────┐      ┌────────────────┐
│     Monolith        │      │ Payment Service│
│ ┌─────────────────┐ │ ───▶ │ (Microservice) │
│ │ Cart            │ │ HTTP │                │
│ ├─────────────────┤ │      └────────────────┘
│ │ Catalog         │ │
│ └─────────────────┘ │
└─────────────────────┘

Phase 3: Gradual extraction
Continue extracting services one at a time
```

**When NOT to migrate:**
- Team < 20 developers
- Monolith performs well
- Clear module boundaries don't exist
- Business priorities are elsewhere

---

## Best Practices

### ✅ **Do's**

1. **Organize by features/modules**
   ```
   ✅ Good: modules/payment/, modules/inventory/
   ❌ Bad:  controllers/, services/, models/
   ```

2. **Use dependency injection**
   - Loose coupling between layers
   - Easier to test and refactor

3. **Implement caching early**
   - Redis for sessions
   - Cache frequently accessed data
   - Reduces database load

4. **Monitor from day one**
   - Application metrics (response time, error rate)
   - Infrastructure metrics (CPU, memory)
   - Business metrics (orders/second)

### ❌ **Don'ts**

1. **Don't create a distributed monolith**
   - Don't split just because "microservices are cool"
   - Split when you have a clear business reason

2. **Don't skip database optimization**
   - Indexes, query optimization matter
   - Vertical scaling goes far with tuning

3. **Don't tightly couple everything**
   - Use interfaces and abstraction layers
   - Prepare for potential future extraction

---

## Interview Talking Points

**Q: When would you choose a monolith over microservices?**

✅ **Strong Answer:**
> "I'd choose a monolith for early-stage startups or when the team is small (< 15 people). The simplicity accelerates development velocity, and you can always extract services later once you understand the domain better. Examples like Shopify and Stack Overflow show that monoliths can scale to millions of users with proper caching, database optimization, and horizontal scaling. I'd only migrate to microservices when team coordination becomes a bottleneck or when we need independent scaling of specific features."

**Q: How do you scale a monolithic application?**

✅ **Strong Answer:**
> "First, I'd optimize the existing system - add caching (Redis), database indexes, and connection pooling. Then horizontal scaling with load balancers and stateless app servers. For the database, read replicas handle read-heavy workloads, and sharding partitions data if needed. I'd also profile to find bottlenecks - often 80% of load comes from 20% of endpoints. This approach took companies like Stack Overflow to billions of page views without microservices."

---

## Related Topics

- [Microservices Architecture](microservices.md) - When to split the monolith
- [Event-Driven Architecture](event-driven.md) - Decouple within monolith
- [Database Scaling](../data/databases/scaling-patterns.md) - Scale the data layer
- [Caching Strategies](../data/caching/strategies.md) - Improve monolith performance

---

**Start simple, scale when needed! 🏢**
