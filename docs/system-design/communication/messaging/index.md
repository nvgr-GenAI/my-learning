# Messaging

**Async communication between services** | 📨 Queues | 📡 Streaming | 🔔 Pub/Sub

## Quick Decision Guide

| Pattern | Use Case | Delivery | Order | Examples |
|---------|----------|----------|-------|----------|
| **Message Queue** | Task processing, decoupling | At-least-once | FIFO | RabbitMQ, SQS |
| **Event Streaming** | Real-time data, analytics | At-least-once | Ordered | Kafka, Kinesis |
| **Pub/Sub** | Event notification, fanout | At-most-once | No order | SNS, Redis Pub/Sub |

---

## Topics

| Topic | Status |
|-------|--------|
| [Message Queues](message-queues.md) | 📝 Planned |
| [Event Streaming](event-streaming.md) | 📝 Planned |
| [Pub/Sub](pub-sub.md) | 📝 Planned |

---

**Decouple services with messaging! 📨**
