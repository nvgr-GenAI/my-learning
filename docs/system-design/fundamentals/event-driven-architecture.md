# Event-Driven Architecture: Building Reactive Systems

## 🎯 What is Event-Driven Architecture?

Event-Driven Architecture (EDA) is a design pattern where system components communicate through the production and consumption of events. Instead of direct service-to-service calls, components react to events that represent state changes or significant occurrences in the system.

## 🔄 Events vs Messages vs Commands

### Events
**Something that has already happened**

**Characteristics**:
- Past tense naming (OrderCreated, UserRegistered)
- Immutable - cannot be changed
- Broadcasted to multiple consumers
- Represents a fact about the system

**Example**:
```json
{
  "eventType": "OrderCreated",
  "eventId": "evt-123",
  "timestamp": "2024-01-15T10:30:00Z",
  "aggregateId": "order-456",
  "payload": {
    "orderId": "order-456",
    "userId": "user-789",
    "items": [...],
    "total": 99.99
  }
}
```

### Messages
**Information passed between systems**

**Characteristics**:
- Can be events, commands, or queries
- May be processed asynchronously
- Point-to-point or publish-subscribe
- Can be persistent or transient

### Commands
**Instructions to perform an action**

**Characteristics**:
- Imperative naming (CreateOrder, UpdateUser)
- Sent to specific handlers
- Can be rejected or validated
- Represents an intent to change system state

**Example**:
```json
{
  "commandType": "CreateOrder",
  "commandId": "cmd-789",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "userId": "user-789",
    "items": [...],
    "shippingAddress": {...}
  }
}
```

## 🏗️ Event-Driven Patterns

### Publish-Subscribe Pattern
**One-to-many communication**

**How it works**:
- Publishers emit events without knowing who will consume them
- Subscribers register interest in specific event types
- Event broker manages delivery to all interested subscribers

**Example Flow**:
```
Order Service → OrderCreated Event → Event Broker
                                          ├── Inventory Service
                                          ├── Payment Service
                                          ├── Shipping Service
                                          └── Analytics Service
```

**Benefits**:
- Loose coupling between services
- Easy to add new consumers
- Scalable event processing
- Fault tolerance through retries

### Event Sourcing
**Store all changes as events**

**Concept**:
Instead of storing current state, store the sequence of events that led to that state.

**Example - User Account**:
```
Events:
1. UserRegistered(userId: 123, email: "user@example.com")
2. UserEmailChanged(userId: 123, newEmail: "new@example.com")
3. UserPasswordChanged(userId: 123, hashedPassword: "...")

Current State (derived from events):
{
  "userId": 123,
  "email": "new@example.com",
  "hashedPassword": "...",
  "registrationDate": "2024-01-15T10:30:00Z"
}
```

**Advantages**:
- Complete audit trail
- Ability to replay events
- Time travel debugging
- Natural fit for distributed systems

**Challenges**:
- Event schema evolution
- Snapshot management for performance
- Increased storage requirements
- Complex query patterns

### CQRS (Command Query Responsibility Segregation)
**Separate read and write models**

**Traditional Model**:
```
Application → Single Model → Database
```

**CQRS Model**:
```
Commands → Write Model → Event Store
Events → Read Model Builder → Read Database
Queries → Read Model → Read Database
```

**Benefits**:
- Optimized read and write operations
- Independent scaling of read/write sides
- Flexibility in data models
- Better performance for complex queries

**Use Cases**:
- High-read, low-write applications
- Complex reporting requirements
- Different consistency requirements for reads/writes
- Collaborative applications

### Saga Pattern
**Manage distributed transactions with events**

**Choreography-based Saga**:
```
1. Order Service → OrderCreated
2. Payment Service → PaymentProcessed
3. Inventory Service → InventoryReserved
4. Shipping Service → ShipmentCreated
5. Order Service → OrderCompleted
```

**Orchestration-based Saga**:
```
Order Saga Orchestrator:
1. Send PaymentCommand → Payment Service
2. Send ReserveInventoryCommand → Inventory Service
3. Send CreateShipmentCommand → Shipping Service
4. Send CompleteOrderCommand → Order Service
```

**Compensation Events**:
```
If shipping fails:
1. ShippingFailed event
2. ReleaseInventory event
3. RefundPayment event
4. CancelOrder event
```

## 🚀 Event Streaming Platforms

### Apache Kafka
**High-throughput distributed streaming platform**

**Core Concepts**:
- **Topics**: Categories of events
- **Partitions**: Parallel processing units
- **Producers**: Event publishers
- **Consumers**: Event subscribers
- **Brokers**: Kafka servers

**Example Usage**:
```java
// Producer
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

Producer<String, String> producer = new KafkaProducer<>(props);
producer.send(new ProducerRecord<>("orders", "order-123", orderJson));

// Consumer
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-processors");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("orders"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        processOrder(record.value());
    }
}
```

### Amazon EventBridge
**Serverless event bus service**

**Features**:
- Built-in event sources (AWS services)
- Custom event buses
- Event filtering and routing
- Schema registry
- Automatic scaling

**Event Pattern**:
```json
{
  "source": ["mycompany.orders"],
  "detail-type": ["Order Placed"],
  "detail": {
    "state": ["CA", "NY"],
    "total": [{"numeric": [">", 100]}]
  }
}
```

### Google Cloud Pub/Sub
**Messaging service for event-driven systems**

**Features**:
- Global message routing
- At-least-once delivery
- Message ordering
- Dead letter queues
- Push and pull subscriptions

## 📊 Event Schema Design

### Schema Evolution
**Handle changes over time**

**Backward Compatibility**:
- Add optional fields only
- Don't remove existing fields
- Don't change field types
- Use default values

**Example Evolution**:
```json
// Version 1
{
  "eventType": "OrderCreated",
  "version": 1,
  "orderId": "order-123",
  "userId": "user-456",
  "total": 99.99
}

// Version 2 (backward compatible)
{
  "eventType": "OrderCreated",
  "version": 2,
  "orderId": "order-123",
  "userId": "user-456",
  "total": 99.99,
  "currency": "USD",  // New optional field
  "discountCode": null  // New optional field
}
```

### Schema Registry
**Centralized schema management**

**Benefits**:
- Schema versioning
- Compatibility checking
- Code generation
- Documentation

**Avro Schema Example**:
```json
{
  "type": "record",
  "name": "OrderCreated",
  "namespace": "com.example.events",
  "fields": [
    {"name": "orderId", "type": "string"},
    {"name": "userId", "type": "string"},
    {"name": "total", "type": "double"},
    {"name": "currency", "type": "string", "default": "USD"},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"}
  ]
}
```

## 🔧 Event Processing Patterns

### Stream Processing
**Real-time event processing**

**Windowing**:
```
Fixed Windows:
[00:00-00:05] [00:05-00:10] [00:10-00:15]

Sliding Windows:
[00:00-00:05] [00:01-00:06] [00:02-00:07]

Session Windows:
[UserSession1] [Gap] [UserSession2]
```

**Aggregations**:
```sql
-- Calculate order totals per hour
SELECT 
  HOUR(timestamp) as hour,
  COUNT(*) as order_count,
  SUM(total) as total_revenue
FROM orders
GROUP BY HOUR(timestamp)
```

### Complex Event Processing (CEP)
**Pattern detection across event streams**

**Example - Fraud Detection**:
```
Pattern: Multiple failed login attempts followed by successful login
1. LoginFailed (user: john, timestamp: 10:00)
2. LoginFailed (user: john, timestamp: 10:01)
3. LoginFailed (user: john, timestamp: 10:02)
4. LoginSuccess (user: john, timestamp: 10:03)
→ Trigger FraudAlert
```

### Event Replay
**Reprocess historical events**

**Use Cases**:
- Bug fixes requiring data correction
- New feature requiring historical data
- Disaster recovery
- Testing with production data

**Implementation**:
```
1. Stop current event processing
2. Reset consumer offset to desired point
3. Start replay from historical position
4. Process events with updated logic
5. Resume real-time processing
```

## 🛡️ Reliability and Error Handling

### Delivery Guarantees
**Different levels of reliability**

**At-most-once**:
- Messages may be lost
- Messages never duplicated
- Lowest latency
- Use case: Metrics, logging

**At-least-once**:
- Messages never lost
- Messages may be duplicated
- Moderate latency
- Use case: Most business events

**Exactly-once**:
- Messages delivered exactly once
- Highest latency
- Complex implementation
- Use case: Financial transactions

### Idempotency
**Handle duplicate messages safely**

**Idempotent Operations**:
```java
// Using unique event IDs
public void processOrderCreated(OrderCreatedEvent event) {
    if (processedEvents.contains(event.getEventId())) {
        return; // Already processed
    }
    
    // Process the event
    createOrder(event.getPayload());
    processedEvents.add(event.getEventId());
}
```

### Dead Letter Queues
**Handle failed message processing**

**Process**:
1. Message processing fails
2. Retry according to policy
3. After max retries, send to dead letter queue
4. Manual investigation and reprocessing

**Example Configuration**:
```json
{
  "retryPolicy": {
    "maxRetries": 3,
    "backoffMultiplier": 2,
    "initialDelaySeconds": 1
  },
  "deadLetterQueue": {
    "topic": "failed-orders",
    "maxDeliveryAttempts": 5
  }
}
```

## 🔍 Monitoring and Observability

### Event Metrics
**Key performance indicators**

**Throughput Metrics**:
- Events published per second
- Events consumed per second
- Consumer lag (events behind)
- Processing time per event

**Error Metrics**:
- Failed event processing rate
- Dead letter queue size
- Retry attempts
- Schema validation errors

### Distributed Tracing
**Track events across services**

**Example Trace**:
```
TraceID: abc123
├── OrderService.createOrder (SpanID: span1)
├── EventBroker.publish (SpanID: span2)
├── PaymentService.processPayment (SpanID: span3)
├── InventoryService.reserveItems (SpanID: span4)
└── ShippingService.createShipment (SpanID: span5)
```

### Event Lineage
**Track event flow through system**

**Example Flow**:
```
UserClickedBuy → OrderCreated → PaymentProcessed → InventoryReserved → ShipmentCreated → OrderCompleted
```

## 🎯 Best Practices

### Event Design
1. **Use descriptive names**: OrderCreated, not OrderEvent
2. **Include context**: Add metadata like userId, timestamp
3. **Keep events small**: Only include necessary data
4. **Version events**: Plan for schema evolution
5. **Make events immutable**: Never change published events

### System Design
1. **Design for failure**: Handle missing events gracefully
2. **Implement circuit breakers**: Prevent cascade failures
3. **Use event sourcing judiciously**: Not every domain needs it
4. **Monitor consumer lag**: Detect processing bottlenecks
5. **Test with real data**: Use production-like event volumes

### Operational Considerations
1. **Capacity planning**: Plan for event volume growth
2. **Backup and recovery**: Protect against data loss
3. **Security**: Encrypt sensitive event data
4. **Compliance**: Handle data retention requirements
5. **Performance testing**: Validate under load

## 🚀 When to Use Event-Driven Architecture

### Good Use Cases
- **Real-time analytics**: Process events as they happen
- **Microservices integration**: Loose coupling between services
- **Audit trails**: Complete history of system changes
- **Notification systems**: React to system events
- **Workflow orchestration**: Complex business processes

### Challenges to Consider
- **Eventual consistency**: Data may be temporarily inconsistent
- **Debugging complexity**: Harder to trace event flows
- **Message ordering**: Ensuring correct event sequence
- **Schema evolution**: Managing event format changes
- **Operational complexity**: More infrastructure to manage

Event-driven architecture enables building scalable, resilient systems that can react to changes in real-time while maintaining loose coupling between components.
