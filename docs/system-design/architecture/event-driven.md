# Event-Driven Architecture

**Asynchronous communication via events** | 📨 Decoupled | ⚡ Reactive | 🔄 Scalable

---

## Overview

Event-Driven Architecture (EDA) is a design pattern where components communicate by producing and consuming events. An event represents a significant change in state that other parts of the system may care about.

**Key Concept:** "Something happened" - not "do this"

---

## Architecture Diagram

```
┌──────────────┐         ┌─────────────────┐        ┌──────────────┐
│   Producer   │         │  Event Broker   │        │   Consumer   │
│  (Service A) │────────▶│  (Kafka/RabbitMQ│───────▶│  (Service B) │
└──────────────┘ Publish │   /SNS/SQS)     │Subscribe└──────────────┘
                         └─────────────────┘               │
                                  │                         │
                                  │                         ↓
                                  │                  ┌──────────────┐
                                  └─────────────────▶│   Consumer   │
                                           Subscribe │  (Service C) │
                                                     └──────────────┘

Flow:
1. Producer publishes event (doesn't know who consumes)
2. Event Broker routes to interested consumers
3. Consumers process event independently
```

---

## Core Concepts

=== "Events"
    **Immutable facts about what happened**

    ```javascript
    // ✅ Good Event (describes what happened)
    {
        "eventType": "OrderCreated",
        "eventId": "evt_123",
        "timestamp": "2025-01-30T10:30:00Z",
        "data": {
            "orderId": "order_789",
            "userId": "user_456",
            "items": [
                { "productId": "prod_001", "quantity": 2 }
            ],
            "totalAmount": 199.98
        }
    }

    // ❌ Bad Event (sounds like a command)
    {
        "eventType": "CreateOrder",  // This is a command, not event!
        ...
    }

    // ❌ Bad Event (mutable)
    {
        "eventType": "OrderStatus",
        "status": "PENDING"  // Status will change - not immutable!
    }
    ```

    **Characteristics:**
    - **Past tense:** "OrderCreated" not "CreateOrder"
    - **Immutable:** Once published, never changes
    - **Self-contained:** Contains all relevant data
    - **Timestamped:** Know when it happened

=== "Event Producers"
    **Components that publish events**

    ```javascript
    // Order Service (Producer)
    class OrderService {
        async createOrder(order) {
            // 1. Perform business logic
            const savedOrder = await this.orderRepository.save(order);

            // 2. Publish event (fire and forget)
            await this.eventBus.publish('OrderCreated', {
                orderId: savedOrder.id,
                userId: savedOrder.userId,
                amount: savedOrder.totalAmount,
                timestamp: new Date()
            });

            // 3. Return immediately (don't wait for consumers)
            return savedOrder;
        }
    }
    ```

    **Producer doesn't:**
    - Know who consumes the event
    - Wait for consumers to process
    - Care if consumers succeed or fail

=== "Event Consumers"
    **Components that react to events**

    ```javascript
    // Email Service (Consumer 1)
    eventBus.subscribe('OrderCreated', async (event) => {
        await sendEmail(event.userId, 'Order Confirmation', {
            orderId: event.orderId,
            amount: event.amount
        });
    });

    // Analytics Service (Consumer 2)
    eventBus.subscribe('OrderCreated', async (event) => {
        await analytics.track('order_created', {
            amount: event.amount,
            timestamp: event.timestamp
        });
    });

    // Inventory Service (Consumer 3)
    eventBus.subscribe('OrderCreated', async (event) => {
        await inventory.reserveItems(event.orderId);
    });
    ```

    **Multiple consumers can react to same event independently**

=== "Event Broker"
    **Middleware that routes events**

    | Broker | Type | Use Case |
    |--------|------|----------|
    | **Kafka** | Event Streaming | High throughput, event replay |
    | **RabbitMQ** | Message Queue | Complex routing, low latency |
    | **AWS SNS/SQS** | Pub/Sub + Queue | Serverless, AWS ecosystem |
    | **Redis Streams** | Lightweight | Simple use cases, low cost |
    | **Google Pub/Sub** | Pub/Sub | GCP ecosystem |

---

## Event Patterns

=== "Event Notification"
    **Notify others something happened**

    ```javascript
    // User Service: Publishes event
    await eventBus.publish('UserRegistered', {
        userId: '123',
        email: 'user@example.com'
    });

    // Multiple services react:
    // - Email Service: Send welcome email
    // - Analytics: Track signup
    // - Marketing: Add to mailing list
    // - CRM: Create customer record
    ```

    **Use when:** Multiple services need to know about the event

    **Pros:**
    - Loose coupling
    - Easy to add new consumers

    **Cons:**
    - Hard to track end-to-end flow

=== "Event-Carried State Transfer"
    **Event contains full state (no need to query)**

    ```javascript
    // Include enough data so consumers don't need to call back
    {
        "eventType": "ProductPriceChanged",
        "productId": "prod_123",
        "oldPrice": 99.99,
        "newPrice": 79.99,
        "productName": "Wireless Mouse",  // Include for convenience
        "category": "Electronics",
        "inStock": true,
        "timestamp": "2025-01-30T10:30:00Z"
    }

    // Consumer doesn't need to call Product Service
    eventBus.subscribe('ProductPriceChanged', async (event) => {
        // Has all data needed
        await updateCache(event.productId, {
            price: event.newPrice,
            name: event.productName
        });
    });
    ```

    **Use when:** Reduce service dependencies, improve performance

    **Trade-off:** Larger events, data duplication

=== "Event Sourcing"
    **Store all state changes as events**

    ```javascript
    // Traditional: Store current state
    {
        "orderId": "123",
        "status": "DELIVERED",  // Lost history!
        "total": 99.99
    }

    // Event Sourcing: Store all events
    [
        { "event": "OrderCreated", "amount": 99.99, "timestamp": "..." },
        { "event": "OrderPaid", "paymentId": "pay_1", "timestamp": "..." },
        { "event": "OrderShipped", "trackingId": "trk_1", "timestamp": "..." },
        { "event": "OrderDelivered", "timestamp": "..." }
    ]

    // Reconstruct current state by replaying events
    function getCurrentState(events) {
        let state = { status: null, amount: 0 };
        for (const event of events) {
            switch(event.event) {
                case 'OrderCreated':
                    state.amount = event.amount;
                    state.status = 'CREATED';
                    break;
                case 'OrderPaid':
                    state.status = 'PAID';
                    break;
                case 'OrderShipped':
                    state.status = 'SHIPPED';
                    break;
                case 'OrderDelivered':
                    state.status = 'DELIVERED';
                    break;
            }
        }
        return state;
    }
    ```

    **Pros:**
    - Complete audit trail
    - Time travel (replay to any point)
    - Easy to add new projections

    **Cons:**
    - Complex to implement
    - Storage grows indefinitely
    - Eventual consistency

=== "CQRS"
    **Command Query Responsibility Segregation**

    ```
    Write Side (Commands):
    ┌──────────────┐
    │   Command    │  CreateOrder
    │   Handler    │ ────────────▶ Write DB
    └──────────────┘              (Event Store)
            │                           │
            └───────────────────────────┘
                       Publish Event

    Read Side (Queries):
         Event ──────▶ Projection ──────▶ Read DB
                                       (Optimized views)
                                              │
    ┌──────────────┐                         │
    │    Query     │ ◀───────────────────────┘
    │   Handler    │  GetOrder
    └──────────────┘
    ```

    ```javascript
    // Write Side: Handle commands
    commandBus.handle('CreateOrder', async (command) => {
        const event = { type: 'OrderCreated', ...command };
        await eventStore.append(event);
        await eventBus.publish(event);
    });

    // Read Side: Build projections
    eventBus.subscribe('OrderCreated', async (event) => {
        // Update read model (denormalized for queries)
        await readDB.orders.insert({
            orderId: event.orderId,
            customerName: event.customerName,
            total: event.total,
            status: 'CREATED'
        });
    });

    // Query uses read model (fast!)
    async function getOrder(orderId) {
        return await readDB.orders.findOne({ orderId });
    }
    ```

    **Use when:**
    - Read and write patterns are very different
    - Need optimized read views
    - High read:write ratio (100:1)

---

## Advantages

### ✅ **Loose Coupling**

Services don't need to know about each other:

```javascript
// Order Service publishes event
await eventBus.publish('OrderCreated', orderData);

// Easy to add new consumer later (no change to Order Service!)
eventBus.subscribe('OrderCreated', async (event) => {
    await newFeatureService.process(event);
});
```

**Benefit:** Add features without modifying existing services

### ✅ **Scalability**

Consumers process events at their own pace:

```
Producer (fast):
   Publishes 10,000 events/sec
            ↓
   Event Broker (buffer)
            ↓
Consumer (slow):
   Processes 100 events/sec (catches up over time)
```

**Benefit:** Natural load leveling, peak handling

### ✅ **Resilience**

Consumer failures don't affect producers:

```
Order Service: Publishes OrderCreated ✅
                      ↓
Email Service: Tries to send email ❌ (SMTP server down)
                      ↓
Event stays in queue, retries later ✅
                      ↓
Analytics Service: Tracks order ✅ (independent)
```

**Benefit:** Partial system failures don't cascade

### ✅ **Audit Trail**

Events provide history:

```javascript
// Query: "What happened to order 123?"
const events = await eventStore.get('order_123');
// [
//   { event: 'OrderCreated', timestamp: '...' },
//   { event: 'PaymentProcessed', timestamp: '...' },
//   { event: 'OrderShipped', timestamp: '...' },
//   { event: 'OrderDelivered', timestamp: '...' }
// ]
```

**Benefit:** Debugging, compliance, analytics

---

## Disadvantages

### ❌ **Eventual Consistency**

Changes aren't immediately visible:

```javascript
// Time 0: Create user
await userService.create({ id: '123', name: 'Alice' });
// Publishes UserCreated event

// Time 1ms: Try to query (might not be ready!)
const user = await queryService.getUser('123');
// Returns null (projection hasn't updated yet)

// Time 100ms: Query again
const user = await queryService.getUser('123');
// Returns Alice (projection updated)
```

**Challenge:** UI might show stale data temporarily

### ❌ **Complexity**

Harder to understand and debug:

```
Monolith (direct call):
  Controller → Service → Database
  (Easy to trace)

Event-Driven:
  ServiceA → Event → ServiceB
                  → ServiceC
                  → ServiceD → ServiceE
  (Hard to trace without tooling)
```

**Need:** Distributed tracing, correlation IDs

### ❌ **Message Ordering**

Events may arrive out of order:

```javascript
// Published in order:
1. OrderCreated (t=100ms)
2. OrderPaid (t=101ms)
3. OrderShipped (t=102ms)

// Consumer receives out of order due to network:
1. OrderCreated (t=100ms)
3. OrderShipped (t=102ms) ← Arrives before payment!
2. OrderPaid (t=101ms)

// Need to handle: Can't ship before payment
if (event.type === 'OrderShipped' && !order.isPaid) {
    // Reject or requeue
}
```

**Solution:** Use timestamps, version numbers, or ordered partitions

### ❌ **Testing Difficulty**

Must test asynchronous flows:

```javascript
// Hard to test
test('order creates and sends email', async () => {
    await orderService.create(order);
    // Email sent asynchronously - when to check?
    await sleep(1000); // Flaky!
    expect(emailService.sent).toBe(true);
});
```

**Solution:** Integration tests, test event handlers independently

---

## Best Practices

### ✅ **Use Correlation IDs**

Track requests across services:

```javascript
// Generate ID at entry point
const correlationId = uuid();

// Include in all events
await eventBus.publish('OrderCreated', {
    correlationId,
    orderId: '123',
    ...
});

// Log with correlation ID
logger.info({ correlationId, msg: 'Order created' });

// All logs for this request have same ID (easy to search!)
```

### ✅ **Idempotent Consumers**

Handle duplicate events:

```javascript
// ❌ Not idempotent
eventBus.subscribe('OrderCreated', async (event) => {
    await inventory.decrement(event.productId, event.quantity);
    // Problem: If event processed twice, inventory wrong!
});

// ✅ Idempotent
eventBus.subscribe('OrderCreated', async (event) => {
    const processed = await checkIfProcessed(event.eventId);
    if (processed) return; // Skip duplicate

    await inventory.decrement(event.productId, event.quantity);
    await markAsProcessed(event.eventId);
});
```

### ✅ **Dead Letter Queues**

Handle poison messages:

```javascript
eventBus.subscribe('OrderCreated', {
    maxRetries: 3,
    deadLetterQueue: 'orders-dlq',
    handler: async (event) => {
        await processOrder(event);
    }
});

// After 3 failures, moves to DLQ
// Monitor DLQ, investigate failures
```

### ✅ **Schema Versioning**

Handle event evolution:

```javascript
// v1 Event
{
    "version": 1,
    "eventType": "OrderCreated",
    "orderId": "123"
}

// v2 Event (added field)
{
    "version": 2,
    "eventType": "OrderCreated",
    "orderId": "123",
    "priority": "high"  // New field
}

// Consumer handles both
eventBus.subscribe('OrderCreated', async (event) => {
    const priority = event.priority || 'normal'; // Default for v1
    await process(event.orderId, priority);
});
```

---

## Technology Choices

=== "Apache Kafka"
    **Best for: Event streaming, high throughput**

    ```javascript
    // Producer
    await kafka.send({
        topic: 'orders',
        messages: [{
            key: order.id,
            value: JSON.stringify(order)
        }]
    });

    // Consumer
    await consumer.run({
        eachMessage: async ({ topic, message }) => {
            const order = JSON.parse(message.value);
            await processOrder(order);
        }
    });
    ```

    **Pros:**
    - 1M+ messages/sec
    - Event replay (keep events for days/weeks)
    - Partitioning for scaling
    - Strong durability

    **Cons:**
    - Complex to operate
    - Overkill for simple use cases

    **Use when:** High volume, need event history

=== "RabbitMQ"
    **Best for: Traditional message queuing**

    ```javascript
    // Publish
    await channel.publish('exchange', 'orders.created',
        Buffer.from(JSON.stringify(order))
    );

    // Subscribe
    await channel.consume('order-queue', async (msg) => {
        const order = JSON.parse(msg.content);
        await processOrder(order);
        channel.ack(msg);
    });
    ```

    **Pros:**
    - Easy to set up
    - Flexible routing
    - Low latency (< 1ms)

    **Cons:**
    - Lower throughput than Kafka
    - Messages deleted after consumption

    **Use when:** Lower volume, need flexible routing

=== "AWS SNS/SQS"
    **Best for: Serverless, AWS-native**

    ```javascript
    // SNS: Publish
    await sns.publish({
        TopicArn: 'arn:aws:sns:us-east-1:123456789:orders',
        Message: JSON.stringify(order)
    });

    // SQS: Consume
    const messages = await sqs.receiveMessage({
        QueueUrl: 'https://sqs.us-east-1.amazonaws.com/123456789/orders'
    });
    ```

    **Pros:**
    - Fully managed (no ops)
    - Integrates with Lambda
    - Auto-scaling

    **Cons:**
    - AWS lock-in
    - Higher latency than self-hosted

    **Use when:** Serverless, AWS infrastructure

---

## Real-World Example

**E-commerce Order Flow:**

```javascript
// 1. User places order
POST /api/orders
    ↓
OrderService.create()
    ↓
Publish: OrderCreated
    ↓
┌───────────────┬───────────────┬──────────────┬──────────────┐
↓               ↓               ↓              ↓              ↓
Inventory    Payment       Shipping       Email        Analytics
Service      Service       Service        Service      Service
  ↓            ↓             ↓              ↓            ↓
Reserve    Process       Calculate      Send         Track
items      payment       shipping       confirmation  revenue
  ↓            ↓             ↓              ↓            ↓
Publish:   Publish:      Publish:       (Done)       (Done)
Inventory  Payment       Shipping
Reserved   Processed     Calculated
  ↓            ↓             ↓
  └────────────┴─────────────┘
               ↓
         Order Service
         (Listens to all)
               ↓
         Update order
         status: CONFIRMED
               ↓
         Publish: OrderConfirmed
```

**Code:**

```javascript
// Order Service
class OrderService {
    async createOrder(orderData) {
        const order = await this.db.orders.insert(orderData);

        await this.eventBus.publish('OrderCreated', {
            orderId: order.id,
            userId: order.userId,
            items: order.items,
            total: order.total
        });

        return order;
    }

    // Listen for completion events
    async handlePaymentProcessed(event) {
        await this.db.orders.update(event.orderId, {
            paymentStatus: 'PAID'
        });
        await this.checkIfOrderComplete(event.orderId);
    }

    async handleInventoryReserved(event) {
        await this.db.orders.update(event.orderId, {
            inventoryStatus: 'RESERVED'
        });
        await this.checkIfOrderComplete(event.orderId);
    }

    async checkIfOrderComplete(orderId) {
        const order = await this.db.orders.findOne(orderId);
        if (order.paymentStatus === 'PAID' &&
            order.inventoryStatus === 'RESERVED') {
            await this.eventBus.publish('OrderConfirmed', {
                orderId: order.id
            });
        }
    }
}
```

---

## Interview Talking Points

**Q: When would you use event-driven architecture?**

✅ **Strong Answer:**
> "I'd use event-driven architecture when I need loose coupling between services or when multiple services need to react to the same event. For example, when a user places an order, I might need to update inventory, process payment, send email, and track analytics. Instead of the order service calling each service directly (tight coupling), I'd publish an 'OrderCreated' event that each service consumes independently. This makes it easy to add new features later - like adding a loyalty points service - without modifying the order service. However, I'd avoid it for simple use cases since the eventual consistency and debugging complexity isn't worth it unless you need the decoupling."

**Q: How do you handle failures in event-driven systems?**

✅ **Strong Answer:**
> "I'd implement retry logic with exponential backoff and use dead letter queues for poison messages. For example, if an email service fails to send, I'd retry 3 times with increasing delays (1s, 5s, 25s). If it still fails, move the event to a dead letter queue for manual investigation. I'd also make consumers idempotent - using event IDs to track which events have been processed so duplicate events don't cause issues. For critical workflows, I'd add monitoring and alerts on DLQ depth so we're notified of persistent failures."

---

## Related Topics

- [Microservices Architecture](microservices.md) - EDA enables microservices
- [Messaging Patterns](../communication/messaging/patterns.md) - Event broker details
- [CQRS Pattern](../distributed-systems/index.md) - Read/write separation
- [Distributed Systems](../distributed-systems/index.md) - Consistency challenges

---

**Events describe what happened, not what should happen! 📨**
