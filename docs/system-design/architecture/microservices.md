# Microservices Architecture

**Independently deployable services** | 🔧 Modular | 🚀 Scalable | 🎯 Distributed

---

## Overview

Microservices architecture structures an application as a collection of small, autonomous services that are independently deployable and organized around business capabilities.

**Key Principle:** Each service is a separate unit that can be developed, deployed, and scaled independently.

---

## Architecture Diagram

```
                    ┌─────────────────┐
                    │  API Gateway    │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ↓                ↓                ↓
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   User        │ │   Product     │ │   Order       │
    │   Service     │ │   Service     │ │   Service     │
    │               │ │               │ │               │
    │ ┌──────────┐  │ │ ┌──────────┐  │ │ ┌──────────┐  │
    │ │   API    │  │ │ │   API    │  │ │ │   API    │  │
    │ ├──────────┤  │ │ ├──────────┤  │ │ ├──────────┤  │
    │ │ Business │  │ │ │ Business │  │ │ │ Business │  │
    │ │  Logic   │  │ │ │  Logic   │  │ │ │  Logic   │  │
    │ ├──────────┤  │ │ ├──────────┤  │ │ ├──────────┤  │
    │ │   DB     │  │ │ │   DB     │  │ │ │   DB     │  │
    │ └──────────┘  │ │ └──────────┘  │ │ └──────────┘  │
    └───────────────┘ └───────────────┘ └───────────────┘
         Users DB         Products DB        Orders DB
```

---

## Core Principles

=== "Single Responsibility"
    **Each service owns one business capability**

    | Service | Responsibility | NOT Responsible For |
    |---------|---------------|---------------------|
    | User Service | User authentication, profiles | Product catalog |
    | Product Service | Product catalog, search | Order processing |
    | Order Service | Order management, checkout | Inventory count |
    | Payment Service | Payment processing | Shipping rates |

    ```
    ✅ Good: Each service has clear boundaries
    ❌ Bad:  "Shared Service" that handles everything
    ```

=== "Independent Deployment"
    **Deploy services without affecting others**

    ```bash
    # Deploy only payment service
    kubectl apply -f payment-service-v2.yaml

    # Other services remain unchanged
    User Service:    v1.2.3 ✅ Running
    Product Service: v2.0.1 ✅ Running
    Payment Service: v1.5.0 → v1.5.1 🚀 Deploying
    Order Service:   v3.1.0 ✅ Running
    ```

    **Benefits:**
    - Faster deployments (minutes, not hours)
    - Reduced deployment risk
    - Independent release cycles
    - No "big bang" releases

=== "Own Database"
    **Each service manages its own data**

    ```
    ❌ Shared Database (Monolith)
    ┌────────┐  ┌────────┐  ┌────────┐
    │  User  │  │Product │  │ Order  │
    │Service │  │Service │  │Service │
    └───┬────┘  └───┬────┘  └───┬────┘
        └───────────┼───────────┘
                    ↓
            ┌──────────────┐
            │   Database   │
            │  (Coupled!)  │
            └──────────────┘

    ✅ Database Per Service
    ┌────────┐      ┌────────┐      ┌────────┐
    │  User  │      │Product │      │ Order  │
    │Service │      │Service │      │Service │
    └───┬────┘      └───┬────┘      └───┬────┘
        ↓               ↓               ↓
    ┌───────┐      ┌───────┐      ┌───────┐
    │User DB│      │Prod DB│      │Order  │
    └───────┘      └───────┘      │  DB   │
                                  └───────┘
    ```

    **Why?**
    - Service can change schema independently
    - No accidental coupling through database
    - Choose best database for each service

=== "API Communication"
    **Services communicate via well-defined APIs**

    **Synchronous (REST/gRPC):**
    ```javascript
    // Order Service calls Product Service
    async function createOrder(userId, productId) {
        // HTTP call to Product Service
        const product = await fetch(
            `http://product-service/api/products/${productId}`
        ).then(r => r.json());

        // HTTP call to Inventory Service
        const available = await fetch(
            `http://inventory-service/api/check/${productId}`
        ).then(r => r.json());

        if (!available) throw new Error('Out of stock');

        // Create order in local database
        return await Order.create({
            userId, productId, price: product.price
        });
    }
    ```

    **Asynchronous (Events):**
    ```javascript
    // Order Service publishes event
    await eventBus.publish('order.created', {
        orderId: '12345',
        userId: 'user-1',
        productId: 'prod-100',
        timestamp: Date.now()
    });

    // Payment Service listens to event
    eventBus.subscribe('order.created', async (event) => {
        await processPayment(event.orderId);
    });

    // Inventory Service listens to same event
    eventBus.subscribe('order.created', async (event) => {
        await reserveInventory(event.productId);
    });
    ```

---

## When to Use Microservices

| Factor | Monolith Better | Microservices Better |
|--------|----------------|---------------------|
| **Team Size** | < 15 developers | > 20 developers |
| **Domain Complexity** | Simple, unclear boundaries | Complex, clear domains |
| **Scale Requirements** | Uniform scaling | Selective scaling (e.g., only search) |
| **Release Frequency** | Weekly/monthly | Multiple times per day |
| **Fault Isolation** | Not critical | Critical (one failure ≠ total outage) |
| **Technology Diversity** | Single stack preferred | Different tech per service |
| **Organizational** | Single team | Multiple autonomous teams |

---

## Advantages

### ✅ **Independent Scalability**

**Scale only what needs scaling:**

```
Normal Load:
┌─────────┐ ┌─────────┐ ┌─────────┐
│ User x1 │ │Product  │ │ Order   │
│         │ │  x1     │ │  x1     │
└─────────┘ └─────────┘ └─────────┘

Black Friday (search traffic spike):
┌─────────┐ ┌─────────┐ ┌─────────┐
│ User x1 │ │Product  │ │ Order   │
│         │ │  x10    │ │  x3     │
└─────────┘ └─────────┘ └─────────┘
            Scale up     Scale up
            search       checkout
```

**Cost Savings:** Only pay for resources you need

### ✅ **Team Autonomy**

Each team owns a service end-to-end:

```
Team Payments:
├── Owns: Payment Service
├── Technology: Node.js, PostgreSQL
├── Deploys: When ready (no coordination)
└── On-call: For their service only

Team Catalog:
├── Owns: Product Service
├── Technology: Python, MongoDB
├── Deploys: Independently
└── On-call: For their service only
```

**Benefits:**
- Faster feature delivery
- No cross-team bottlenecks
- Clear ownership and accountability

### ✅ **Fault Isolation**

**Failure in one service doesn't crash entire system:**

```
Scenario: Payment Service is down

❌ Monolith: Entire website down

✅ Microservices:
   - Browsing products: ✅ Works
   - Adding to cart: ✅ Works
   - Checking out: ❌ Shows error but site remains up
   - User profile: ✅ Works
```

Implement circuit breakers to gracefully degrade.

### ✅ **Technology Flexibility**

**Choose the right tool for each job:**

| Service | Technology | Why? |
|---------|-----------|------|
| Product Search | Elasticsearch | Full-text search |
| User Service | PostgreSQL | Relational data, ACID |
| Analytics | Cassandra | Time-series, high writes |
| Cache Service | Redis | In-memory speed |
| ML Recommendations | Python | ML libraries |

---

## Disadvantages

### ❌ **Complexity**

**Distributed systems are inherently complex:**

```
Monolith (1 thing to debug):
┌──────────────┐
│     App      │
└──────────────┘

Microservices (10+ things to debug):
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ S1 │→│ S2 │→│ S3 │→│ S4 │
└────┘ └────┘ └────┘ └────┘
  ↓      ↓      ↓      ↓
 DB1    DB2    DB3    DB4
  ↓      ↓      ↓      ↓
Queue  Cache  Log    Trace
```

**Must handle:**
- Network failures
- Service discovery
- Load balancing
- Distributed tracing
- Monitoring across services

### ❌ **Data Consistency Challenges**

**No ACID transactions across services:**

```javascript
// Monolith: ACID transaction (easy)
await database.transaction(async (tx) => {
    await tx.orders.create(order);
    await tx.inventory.decrement(productId);
    await tx.payments.charge(userId, amount);
});
// Either all succeed or all rollback

// Microservices: Distributed transaction (hard)
try {
    await orderService.create(order);      // ✅ Success
    await inventoryService.reserve(productId); // ✅ Success
    await paymentService.charge(userId, amount); // ❌ Fails!
    // Now order is created but payment failed!
    // Need compensation logic...
} catch (error) {
    // Rollback order
    // Rollback inventory reservation
}
```

**Solutions:**
- Eventual consistency
- Saga pattern
- Event sourcing

### ❌ **Testing Difficulty**

**Must test interactions between services:**

```
Unit Test:     ✅ Test individual service
Integration:   ⚠️ Test service + database
E2E Test:      ❌ Test 10 services + infrastructure
               Very slow, flaky, hard to maintain
```

### ❌ **Operational Overhead**

| Task | Monolith | Microservices |
|------|----------|---------------|
| **Deploy** | 1 deployment | 10-100 deployments |
| **Monitor** | 1 dashboard | 10-100 dashboards |
| **Logs** | 1 log file | Distributed logging needed |
| **Debug** | 1 stack trace | Distributed tracing needed |
| **Versions** | 1 version | Version matrix (v1.2 + v2.1 + v3.0...) |

**Requires:**
- Kubernetes/orchestration
- Service mesh (Istio, Linkerd)
- Centralized logging (ELK)
- Distributed tracing (Jaeger)
- Monitoring (Prometheus, Grafana)

---

## Communication Patterns

=== "Synchronous (REST)"
    **Request-response pattern**

    ```javascript
    // Order Service → Product Service
    const response = await axios.get(
        'http://product-service/api/products/123'
    );
    const product = response.data;
    ```

    **Pros:**
    - Simple to understand
    - Immediate response
    - Easy to debug

    **Cons:**
    - Tight coupling
    - Cascading failures
    - Higher latency

    **Use when:** Need immediate response, synchronous flow

=== "Synchronous (gRPC)"
    **Binary protocol, faster than REST**

    ```protobuf
    // product.proto
    service ProductService {
        rpc GetProduct (ProductRequest) returns (ProductResponse);
    }

    message ProductRequest {
        string product_id = 1;
    }

    message ProductResponse {
        string id = 1;
        string name = 2;
        double price = 3;
    }
    ```

    ```javascript
    // Client code
    const client = new ProductServiceClient('product-service:50051');
    const product = await client.GetProduct({ product_id: '123' });
    ```

    **Pros:**
    - 7x faster than REST
    - Strong typing with protobuf
    - Bi-directional streaming

    **Cons:**
    - Harder to debug (binary)
    - Browser support limited

    **Use when:** High performance needed, internal services

=== "Asynchronous (Events)"
    **Publish-subscribe pattern**

    ```javascript
    // Publisher: Order Service
    await messageBroker.publish('OrderCreated', {
        orderId: '123',
        userId: 'user-1',
        amount: 99.99
    });

    // Subscriber: Email Service
    messageBroker.subscribe('OrderCreated', async (event) => {
        await sendEmail(event.userId, 'Order confirmed!');
    });

    // Subscriber: Analytics Service
    messageBroker.subscribe('OrderCreated', async (event) => {
        await trackRevenue(event.amount);
    });
    ```

    **Pros:**
    - Loose coupling
    - Services don't need to know about each other
    - Easy to add new subscribers
    - Resilient (retry on failure)

    **Cons:**
    - Eventual consistency
    - Harder to debug
    - Message broker is single point of failure

    **Use when:** Don't need immediate response, multiple services interested in same event

---

## Data Management Patterns

### 1. **Database Per Service**

Each service owns its data:

```
✅ Correct:
User Service    → Users DB
Product Service → Products DB
Order Service   → Orders DB

❌ Wrong:
User Service    ↘
Product Service → Shared DB (creates coupling!)
Order Service   ↗
```

### 2. **Saga Pattern**

Manage distributed transactions across services:

```javascript
// Saga: Create Order
async function createOrderSaga(order) {
    try {
        // Step 1: Create order
        const orderId = await orderService.create(order);

        // Step 2: Reserve inventory
        await inventoryService.reserve(order.productId);

        // Step 3: Process payment
        await paymentService.charge(order.userId, order.amount);

        // Step 4: Confirm order
        await orderService.confirm(orderId);

    } catch (error) {
        // Compensation: Rollback in reverse order
        await paymentService.refund(order.userId);
        await inventoryService.release(order.productId);
        await orderService.cancel(orderId);
    }
}
```

### 3. **CQRS (Command Query Responsibility Segregation)**

Separate read and write models:

```
Write Side (Commands):
┌─────────────┐
│   Command   │
│   Service   │
└──────┬──────┘
       ↓
  ┌─────────┐
  │Write DB │
  └────┬────┘
       ↓
    Events

Read Side (Queries):
  Events
    ↓
┌──────────┐
│ Read DB  │ (Denormalized, optimized for queries)
│(Redis)   │
└─────┬────┘
      ↓
┌──────────┐
│  Query   │
│ Service  │
└──────────┘
```

---

## Service Discovery

**Services need to find each other dynamically:**

=== "Client-Side Discovery"
    ```javascript
    // Service Registry (e.g., Consul, Eureka)
    const serviceRegistry = new ServiceRegistry();

    // Product Service registers itself
    serviceRegistry.register('product-service', {
        host: '10.0.1.23',
        port: 8080
    });

    // Order Service discovers Product Service
    const productService = await serviceRegistry.lookup('product-service');
    const response = await fetch(`http://${productService.host}:${productService.port}/api/products`);
    ```

=== "Server-Side Discovery"
    ```
    Kubernetes Service Discovery:

    Order Service → kubernetes.default.svc.cluster.local
                    ↓
                    DNS lookup
                    ↓
                    "product-service" → 10.0.1.23:8080
    ```

    ```yaml
    # Kubernetes Service
    apiVersion: v1
    kind: Service
    metadata:
      name: product-service
    spec:
      selector:
        app: product
      ports:
        - port: 80
          targetPort: 8080
    ```

---

## Best Practices

### ✅ **Start with Monolith**

Don't build microservices from day one:

```
Phase 1: Monolith (Year 1)
- Learn domain
- Fast development
- Validate business model

Phase 2: Modular Monolith (Year 2)
- Clear module boundaries
- Separate databases internally
- Prepare for extraction

Phase 3: Microservices (Year 3+)
- Extract high-value services
- Extract when team grows > 20
- Extract when independent scaling needed
```

### ✅ **Design for Failure**

Assume services will fail:

```javascript
// Circuit Breaker Pattern
const circuitBreaker = new CircuitBreaker(productService.get, {
    timeout: 3000,        // Fail after 3s
    errorThreshold: 50,   // Open circuit if 50% fail
    resetTimeout: 30000   // Try again after 30s
});

try {
    const product = await circuitBreaker.fire(productId);
} catch (error) {
    // Fallback: Return cached data or default
    return getCachedProduct(productId) || DEFAULT_PRODUCT;
}
```

### ✅ **API Gateway**

Single entry point for clients:

```
           ┌─────────────┐
           │ API Gateway │
           │  - Auth     │
           │  - Routing  │
           │  - Rate     │
           │    Limiting │
           └──────┬──────┘
                  │
     ┌────────────┼────────────┐
     ↓            ↓            ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│  User   │ │ Product │ │  Order  │
│ Service │ │ Service │ │ Service │
└─────────┘ └─────────┘ └─────────┘
```

**Benefits:**
- Client doesn't need to know about all services
- Centralized authentication
- Response aggregation
- Rate limiting

### ✅ **Observability**

Must have end-to-end visibility:

```
Request Flow with Trace ID:

Client
  ↓ trace-id: abc123
API Gateway
  ↓ trace-id: abc123
User Service (logs: abc123, took 50ms)
  ↓ trace-id: abc123
Product Service (logs: abc123, took 120ms)
  ↓ trace-id: abc123
Order Service (logs: abc123, took 200ms)

Total latency: 370ms (easy to debug!)
```

---

## Real-World Examples

=== "Netflix"
    **Scale:**
    - 800+ microservices
    - Millions of requests/second
    - Global deployment

    **Key Decisions:**
    - Async communication (event-driven)
    - Chaos engineering (Chaos Monkey)
    - Hystrix for circuit breaking
    - Service mesh for observability

=== "Uber"
    **Evolution:**
    ```
    2012: Python monolith
    2014: Started microservices migration
    2016: 1000+ microservices
    2020: 4000+ microservices
    ```

    **Challenges:**
    - Distributed tracing essential
    - Service mesh for traffic management
    - Strong API contracts (gRPC)

=== "Amazon"
    **Two-Pizza Team Rule:**
    - Each service owned by small team (< 10 people)
    - Team can sustain on 2 pizzas
    - Full autonomy: build, deploy, operate

    **API-First:**
    - All teams expose APIs
    - Internal services communicate only via APIs
    - Led to AWS (internal services → external products)

---

## Interview Talking Points

**Q: When would you choose microservices over a monolith?**

✅ **Strong Answer:**
> "I'd choose microservices when we have clear business domain boundaries, a team larger than 20 developers, and a need for independent scaling or deployment. For example, if our search traffic spikes 10x during sales but checkout traffic only doubles, microservices let us scale them independently. However, I'd start with a well-structured monolith first - companies like Shopify scaled to billions in revenue before moving to microservices. The complexity of distributed systems isn't worth it until the coordination cost of a monolith becomes the bottleneck."

**Q: How do you handle data consistency across microservices?**

✅ **Strong Answer:**
> "I'd use the Saga pattern for distributed transactions. For example, in an order workflow: (1) create order, (2) reserve inventory, (3) process payment. If payment fails, we execute compensating transactions in reverse - refund payment, release inventory, cancel order. I'd also embrace eventual consistency where appropriate - it's okay if the analytics dashboard shows yesterday's numbers. For critical consistency needs, I'd consider keeping that functionality within a single service rather than splitting it."

---

## Related Topics

- [Monolithic Architecture](monolithic.md) - When to avoid microservices
- [Event-Driven Architecture](event-driven.md) - Async communication pattern
- [API Design](../communication/api-design/index.md) - Design service APIs
- [Distributed Systems](../distributed-systems/index.md) - Challenges and solutions

---

**Microservices aren't a goal, they're a consequence of scaling teams! 🚀**
