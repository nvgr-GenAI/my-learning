# Messaging Patterns

Modern distributed systems rarely communicate through synchronous request-response alone. As services scale independently, they need a way to exchange information without requiring both sides to be available at the same instant. Messaging patterns solve this by introducing an intermediary -- a broker or queue -- that decouples producers from consumers in both time and space. A producer can fire off a message and move on; the consumer processes it whenever it is ready.

This decoupling buys three things. First, temporal decoupling: the producer and consumer do not need to be running at the same time. Second, load leveling: a queue absorbs traffic spikes so consumers can process at a steady rate. Third, failure isolation: if a consumer crashes, messages wait in the queue rather than being lost. These properties are the foundation of resilient, scalable architectures, and they explain why virtually every large-scale system -- from ride-hailing to payment processing to social media feeds -- relies on messaging infrastructure.

The patterns on this page cover the infrastructure layer: how messages travel from point A to point B, what delivery guarantees they carry, and which broker technology fits which workload. For higher-level architectural patterns like event sourcing, CQRS, and choreography vs orchestration, see the Event-Driven Architecture page linked at the bottom.

---

=== "Message Queues"

    ## Point-to-Point vs Publish-Subscribe

    The two fundamental messaging models differ in who receives each message. In a point-to-point queue, every message is delivered to exactly one consumer. In publish-subscribe, every message is broadcast to all subscribers on that topic. Understanding this distinction is the first step in designing any messaging system.

    **Point-to-Point (Competing Consumers)**

    ```
    Producer ---> [ Queue ] ---> Consumer A
                            \--> Consumer B  (only one receives each message)
                            \--> Consumer C
    ```

    In this model, the queue acts as a load balancer. When a producer enqueues a message, only one consumer from the pool dequeues and processes it. This is ideal for work distribution -- tasks like sending emails, processing orders, or resizing images where each job should be done exactly once. Adding more consumers increases throughput linearly, and the broker handles the distribution automatically.

    The competing consumers pattern also provides natural backpressure. When the queue fills up, producers can be throttled or messages can be rejected, preventing the system from being overwhelmed. The queue depth itself becomes a useful metric -- a growing queue signals that consumers cannot keep up, and more instances need to be added.

    **Publish-Subscribe (Fan-Out)**

    ```
    Publisher ---> [ Topic: order.placed ] ---> Subscriber: Billing Service
                                           \--> Subscriber: Inventory Service
                                           \--> Subscriber: Analytics Service
    ```

    In pub/sub, every subscriber receives a copy of every message. When an order is placed, billing charges the card, inventory reserves stock, and analytics records the event -- all independently, all from the same message. This is the pattern behind event notification: one event, many reactions. Adding a new subscriber requires no changes to the publisher or existing subscribers, making the system highly extensible.

    The choice between the two often depends on intent. If you are asking "please do this work," use a queue. If you are announcing "this thing happened," use a topic. Many systems use both: an order service publishes an "order.placed" event to a topic (pub/sub), and one of the subscribers enqueues individual fulfillment tasks into a work queue (point-to-point).

    ## The Producer-Consumer Model

    Regardless of point-to-point or pub/sub, all messaging follows the same lifecycle. A producer creates a message and sends it to the broker. The broker persists the message (in memory or on disk, depending on configuration) until a consumer retrieves it. The consumer processes the message and sends an acknowledgment back to the broker. Only after acknowledgment does the broker remove the message or mark it consumed.

    ```
    Producer           Broker              Consumer
       |                  |                    |
       |--- send msg ---->|                    |
       |                  |  (persist to disk) |
       |                  |--- deliver msg --->|
       |                  |                    |-- process
       |                  |<--- ack -----------|
       |                  |                    |
       |                  | (msg removed)      |
    ```

    This acknowledgment step is critical. If the consumer crashes before sending the ack, the broker redelivers the message to another consumer. This is how at-least-once delivery works in practice, and it is why consumers must be designed to handle duplicate messages gracefully (see the Patterns tab for idempotent consumer design).

    The visibility timeout is an important detail in this model. When a consumer receives a message, the broker hides it from other consumers for a configurable period. If the consumer does not acknowledge within that window, the broker assumes the consumer failed and makes the message visible again for redelivery. Setting this timeout correctly matters: too short, and a slow consumer causes duplicate processing; too long, and a crashed consumer delays reprocessing.

    ## When to Use Each Model

    Point-to-point queues shine for task distribution: background jobs, order processing, file uploads, and any workflow where each unit of work should be processed once. Uber uses point-to-point queuing for ride matching -- when a rider requests a ride, the system enqueues the request, and exactly one matching worker picks it up, evaluates nearby drivers, and assigns the ride. At peak, Uber processes millions of ride requests per day through this pipeline. The queue absorbs the burst of requests during rush hour while the matching workers process them at a sustainable rate.

    Pub/sub topics are the right fit for event notification: audit logging, real-time dashboards, cross-service coordination, and any scenario where multiple independent systems need to react to the same event. Netflix uses pub/sub extensively -- when a user starts streaming, a single event fans out to the recommendation engine (to update viewing history), the CDN prefetch system (to cache the next episode), the billing tracker (to log usage), and the quality-of-experience monitor (to detect buffering). Each system reacts independently without knowing about the others, and Netflix can add new subscribers (say, a parental controls service) without modifying any existing code.

    ## RabbitMQ vs Amazon SQS

    | Aspect | RabbitMQ | Amazon SQS |
    |---|---|---|
    | Model | Both point-to-point and pub/sub (exchanges) | Point-to-point (with SNS for pub/sub) |
    | Delivery | Push-based to consumers | Pull-based (long polling) |
    | Ordering | Guarantees FIFO per queue | Best-effort (FIFO queues available at extra cost) |
    | Throughput | ~50K msgs/sec per node | Virtually unlimited (managed) |
    | Latency | Sub-millisecond | 1-10ms typical |
    | Operations | Self-managed (clustering, monitoring) | Fully managed by AWS |
    | Protocol | AMQP, MQTT, STOMP | HTTP/HTTPS (AWS SDK) |
    | Best for | Complex routing, low latency, on-prem | Serverless workloads, AWS-native apps |

    RabbitMQ offers fine-grained routing through exchanges (direct, topic, fanout, headers) and supports multiple wire protocols. A single RabbitMQ cluster can implement sophisticated routing rules -- for example, routing messages to different queues based on message attributes, geographic region, or priority level. SQS trades that flexibility for zero operational overhead -- no clusters to manage, no disks to monitor, automatic scaling. For teams running on AWS who need simple queuing without operational burden, SQS is the default choice. For teams who need sub-millisecond latency, complex routing, or on-premise deployment, RabbitMQ is the better fit.

=== "Event Streaming"

    ## Kafka Architecture

    Apache Kafka is not a traditional message queue -- it is a distributed commit log. Messages are appended to an immutable, ordered log and retained for a configurable period (often days or weeks), regardless of whether any consumer has read them. This fundamental difference enables capabilities that queues cannot offer: replay, time-travel debugging, and multiple independent consumer groups reading the same data at different speeds.

    ```
    Producers          Kafka Cluster                         Consumers
                   +--------------------------------+
    Producer A --> | Topic: payments                |  --> Consumer Group: billing
                   |   Partition 0: [0][1][2][3]    |  --> Consumer Group: analytics
    Producer B --> |   Partition 1: [0][1][2]       |  --> Consumer Group: fraud
                   |   Partition 2: [0][1][2][3][4] |
    Producer C --> |                                |
                   +--------------------------------+
                     (each partition replicated 3x
                      across different brokers)
    ```

    A Kafka **topic** is a named stream of records, analogous to a database table. Each topic is divided into **partitions** -- ordered, immutable sequences of records, each identified by a sequential offset number. Partitions are the unit of parallelism: within a single partition, messages are strictly ordered by offset, but across partitions, there is no global order. Producers choose which partition receives a message, typically by hashing a message key (such as a user ID or order ID), ensuring all messages with the same key land in the same partition and maintain their relative order.

    Each partition is replicated across multiple brokers for fault tolerance. One replica is the leader (handles all reads and writes), and the others are followers that replicate the data. If a broker hosting a leader partition fails, one of the followers is promoted automatically. The replication factor (typically 3) determines how many broker failures the system can tolerate.

    **Consumer groups** are Kafka's mechanism for parallel consumption. Each partition is assigned to exactly one consumer within a group, so the maximum parallelism equals the number of partitions. Different consumer groups read the same topic independently -- billing can process payments at its own pace while fraud detection reads the same stream separately. If a consumer in a group fails, Kafka rebalances the partitions among the remaining consumers automatically. This rebalancing is one of Kafka's most operationally sensitive behaviors -- during rebalance, consumption pauses briefly, which is why partition assignment strategies (range, round-robin, sticky) matter at scale.

    ## How Streaming Differs from Traditional Queues

    The most important distinction is retention. A traditional queue deletes messages after consumption. Kafka retains them. This means a new consumer group can start reading from the beginning of a topic and replay the entire history -- useful for rebuilding state, backfilling a new service, or reprocessing after a bug fix. It also means consumers manage their own position (offset) in the log rather than the broker tracking delivery state per consumer.

    This design also changes the scaling model. Traditional queues scale by adding consumers that compete for messages. Kafka scales by adding partitions, and each partition can be consumed by only one member of a consumer group. Adding more consumers than partitions means some consumers sit idle. The result is more predictable performance but requires upfront thought about partition count -- and changing partition count on a live topic is operationally disruptive because it changes key-to-partition mappings.

    Another key difference is consumption semantics. With a traditional queue, reading a message removes it. With Kafka, reading is non-destructive -- consumers simply advance their offset. This means a slow consumer does not block other consumer groups, and a consumer can "rewind" to reprocess messages by resetting its offset. This property makes Kafka especially valuable for data pipelines where the same events feed multiple downstream systems at different processing speeds.

    ## LinkedIn: Event Streaming at Scale

    Kafka was born at LinkedIn to solve a specific problem: connecting dozens of backend systems that all needed access to the same streams of data -- user activity, system metrics, log events. Today, LinkedIn's Kafka clusters handle over 7 trillion messages per day across more than 100,000 topics. Every profile view, connection request, job application, and feed impression flows through Kafka before reaching its destination systems -- search indexes, recommendation engines, analytics pipelines, and notification services.

    The key design decision was treating Kafka as the central nervous system rather than point-to-point connections between services. Instead of N services each maintaining M connections (an N*M problem), every service publishes to Kafka and consumes from Kafka, reducing the integration complexity to N+M. This hub-and-spoke model also means adding a new service requires only subscribing to the relevant topics, with zero changes to existing producers.

    LinkedIn also pioneered the concept of compacted topics -- topics where Kafka retains only the latest value for each key rather than the full history. This is used for maintaining the current state of entities (like user profiles) as a stream, enabling new services to bootstrap their state by reading the compacted topic from the beginning.

    ## Queues vs Streams Comparison

    | Characteristic | Message Queue (RabbitMQ, SQS) | Event Stream (Kafka, Kinesis) |
    |---|---|---|
    | Message lifecycle | Deleted after consumption | Retained for configured period |
    | Consumer model | Competing consumers (one gets it) | Consumer groups (each group gets all) |
    | Replay | Not possible after ack | Replay by resetting offset |
    | Ordering | Per-queue FIFO | Per-partition ordering |
    | Throughput | Thousands to tens of thousands/sec | Millions/sec (partitioned) |
    | Latency | Sub-ms to low ms | Low ms (batched writes) |
    | Best for | Task queues, work distribution | Data pipelines, log aggregation, CDC |
    | Scaling | Add consumers | Add partitions |
    | Backpressure | Queue depth / rejection | Consumer lag (offset behind) |

    Choose a queue when messages represent work to be done and can be discarded after processing. Choose a stream when messages represent facts that happened and may be needed by multiple consumers now or in the future. When in doubt, consider whether you would ever want to replay the data -- if yes, you want a stream.

=== "Patterns"

    ## Dead Letter Queues

    Not every message can be processed successfully. A dead letter queue (DLQ) is a holding area for messages that have failed processing after all retry attempts are exhausted. Without a DLQ, poison messages -- messages that consistently cause processing failures -- can block an entire queue, creating a head-of-line blocking problem that stalls all downstream work.

    ```
    Main Queue          Consumer              Dead Letter Queue
        |                  |                        |
        |--- message A --->|                        |
        |                  |-- fail (attempt 1)     |
        |--- message A --->|                        |
        |                  |-- fail (attempt 2)     |
        |--- message A --->|                        |
        |                  |-- fail (attempt 3)     |
        |                  |--- move to DLQ ------->|
        |                  |                        |
        |--- message B --->|                        |
        |                  |-- success              |  (DLQ reviewed by ops)
    ```

    In practice, teams set up alerts on DLQ depth and periodically review failed messages. Some are genuine bugs (fix the consumer and replay), some are malformed data (fix the producer), and some are transient failures that resolve on manual retry. Amazon SQS has built-in DLQ support through redrive policies, where you configure the maximum receive count before automatic transfer. RabbitMQ supports DLQs through dead-letter exchanges, which can route failed messages based on rejection reason.

    A well-designed DLQ workflow includes metadata about why the message failed: the exception type, the consumer that processed it, the number of attempts, and the timestamp of the last failure. This metadata makes triage dramatically faster when you are staring at thousands of failed messages trying to identify the root cause.

    ## Retry with Exponential Backoff

    When a message fails, retrying immediately is usually counterproductive -- the same transient condition (a database timeout, a downstream service restart) is likely still present. Exponential backoff spaces retries apart with increasing delays, giving the system time to recover.

    ```python
    # Retry delay calculation with jitter
    def retry_delay(attempt, base=1, max_delay=60):
        delay = min(base * (2 ** attempt), max_delay)
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
    # attempt 0: ~1s, attempt 1: ~2s, attempt 2: ~4s, attempt 3: ~8s
    ```

    The jitter component is essential. Without it, if a downstream service goes down and 10,000 messages all fail at once, they will all retry at exactly the same intervals, creating a "thundering herd" that hammers the recovering service. Adding randomized jitter spreads retries across the time window and gives the system a chance to recover gradually. AWS recommends "full jitter" (randomizing the entire delay, not just a small fraction) for the best distribution of retries.

    Stripe uses this pattern for webhook delivery. When a merchant's endpoint is unreachable, Stripe retries with exponential backoff over 72 hours, attempting delivery up to 30 times before giving up and marking the webhook as failed. This approach balances persistence (the merchant eventually gets their data) with courtesy (not overwhelming their recovering server).

    Some systems implement tiered retry queues as an alternative to delayed redelivery. Instead of one queue with configurable delays, they use separate queues for each retry tier (retry-1min, retry-5min, retry-30min), with consumers on each queue moving failed messages to the next tier. This approach works well when the broker does not natively support delayed messages.

    ## Ordering Guarantees

    Global ordering across a distributed system is expensive and often unnecessary. Most systems only need ordering within a logical grouping -- all events for a specific user, or all updates to a specific account. Understanding the ordering requirements upfront prevents over-engineering.

    Kafka solves this with partition-level ordering. By hashing the message key to determine the partition, all messages with the same key are guaranteed to arrive in order. A banking system can hash on account ID, ensuring that deposits and withdrawals for account 12345 are always processed in sequence, even though events for different accounts may interleave freely across partitions.

    ```
    Messages: [A:user1] [B:user2] [C:user1] [D:user3] [E:user2]

    Partition 0 (user1): [A] [C]         <-- ordered
    Partition 1 (user2): [B] [E]         <-- ordered
    Partition 2 (user3): [D]             <-- ordered

    Cross-partition order: undefined (B might process before A)
    ```

    The trade-off is clear: more partitions mean more parallelism but weaker global ordering. Fewer partitions mean stronger ordering but lower throughput. Most production systems find the sweet spot by partitioning on a natural business key and accepting that cross-key ordering is undefined. SQS FIFO queues offer a similar concept through message group IDs, where messages within the same group are processed in order while different groups can be processed in parallel.

    ## Exactly-Once Semantics

    Exactly-once delivery is the holy grail of messaging, and it is notoriously difficult to achieve in a distributed system. The fundamental problem is that the network can fail between the broker confirming receipt of a message and the producer receiving that confirmation. Did the message make it or not? The producer cannot know, so it retries -- and now there might be a duplicate.

    Most brokers provide one of two weaker guarantees. At-most-once means fire and forget: the producer sends the message once and never retries, accepting the possibility of loss. At-least-once means the producer retries on failure, accepting the possibility of duplicates. True exactly-once requires coordination between the broker and the consumer's storage, typically through transactions or deduplication.

    Kafka achieves exactly-once within its own ecosystem through two mechanisms: idempotent producers (each message carries a producer ID and sequence number, allowing the broker to detect and discard duplicates) and transactional writes (a producer can atomically write to multiple partitions and commit consumer offsets in a single transaction). But this only works end-to-end when the consumer's side effects stay within Kafka -- for example, reading from one topic, transforming the data, and writing to another topic. The moment a consumer writes to an external database, you are back to needing application-level idempotency.

    ## Idempotent Consumers

    Since most systems operate with at-least-once delivery, consumers must be prepared to handle duplicates gracefully. An idempotent consumer produces the same result whether it processes a message once or ten times. This is the practical solution to the exactly-once problem.

    ```python
    # Idempotent consumer: check-then-process in one transaction
    def process_payment(message):
        idempotency_key = message["payment_id"]
        if db.exists("processed_payments", idempotency_key):
            return  # Already processed, skip
        db.execute_transaction(
            insert("processed_payments", idempotency_key),
            update("accounts", balance=balance - message["amount"])
        )
    ```

    The pattern is straightforward: store a unique identifier for each processed message and check it before processing. The critical detail is that the deduplication check and the business logic must execute within the same transaction -- otherwise a crash between the two could leave the system in an inconsistent state where the payment was processed but the idempotency key was never stored, leading to a duplicate charge on retry.

    Shopify processes millions of payment events daily using this pattern, storing idempotency keys alongside the payment records to ensure no customer is charged twice even when messages are redelivered. They retain idempotency keys for 24 to 48 hours, long enough to cover any realistic redelivery window, then expire them to keep the deduplication table from growing without bound.

    Natural idempotency is even better than explicit deduplication. Operations like "set balance to $100" are inherently idempotent -- doing it twice has the same effect as doing it once. Operations like "add $50 to balance" are not -- doing it twice doubles the effect. Where possible, design messages as absolute state rather than relative changes.

=== "Choosing a System"

    ## Decision Tree

    Selecting a messaging system starts with understanding the workload characteristics. The following decision tree covers the most common scenarios, though many production systems end up running multiple messaging technologies to handle different workloads.

    ```
    What is your primary need?
    |
    +-- Task distribution (background jobs, work queues)
    |   |
    |   +-- Running on AWS? --------------------> Amazon SQS
    |   +-- Need complex routing / priorities? --> RabbitMQ
    |   +-- Simple, ultra-low latency? ---------> Redis Streams
    |
    +-- Event streaming (data pipelines, log aggregation)
    |   |
    |   +-- Self-managed is OK? --------> Apache Kafka
    |   +-- Managed on AWS? ------------> Amazon Kinesis
    |   +-- Managed on GCP? ------------> Google Pub/Sub
    |   +-- Managed, multi-cloud? ------> Confluent Cloud
    |
    +-- Real-time pub/sub (notifications, chat, IoT)
    |   |
    |   +-- Need persistence + replay? -> Google Pub/Sub
    |   +-- Ephemeral fan-out is OK? ---> Redis Pub/Sub
    |
    +-- Exactly-once processing (financial, transactional)
        |
        +-- Within Kafka ecosystem? ----> Kafka Transactions
        +-- Cross-system? --------------> Idempotent consumers + dedup
    ```

    ## Comparison Table

    | Feature | Kafka | RabbitMQ | SQS | Redis Streams | Google Pub/Sub |
    |---|---|---|---|---|---|
    | Throughput | Millions/sec | ~50K/sec | Unlimited (managed) | ~100K/sec | Millions/sec |
    | Latency (p99) | 5-15ms | <1ms | 10-50ms | <1ms | 10-50ms |
    | Ordering | Per-partition | Per-queue | FIFO queues available | Per-stream | Per-key (ordering key) |
    | Retention | Days to forever | Until consumed | 4-14 days | Configurable | 7-31 days |
    | Replay | Yes (offset reset) | No | No | Yes (ID-based) | Yes (seek to timestamp) |
    | Delivery | At-least-once, exactly-once | At-least-once, at-most-once | At-least-once | At-least-once | At-least-once |
    | Ops burden | High (ZooKeeper/KRaft, brokers) | Medium (clustering, Erlang) | None (managed) | Low (part of Redis) | None (managed) |
    | Cost model | Infrastructure | Infrastructure | Per-request | Infrastructure | Per-message + storage |
    | DLQ support | Manual (separate topic) | Built-in (dead-letter exchange) | Built-in (redrive policy) | Manual | Built-in (dead-letter topic) |

    ## Company Examples at Scale

    **Uber -- RabbitMQ to Kafka Migration.** Uber initially used RabbitMQ for inter-service messaging but hit scaling limits around 2016 when they reached hundreds of microservices. RabbitMQ struggled with the fan-out pattern: multiple services needed the same event data, requiring either message duplication or complex exchange topologies. They migrated their core event pipeline to Kafka, which now handles over 1 trillion messages per day across more than 1,000 topics. The key driver was Kafka's native support for multiple consumer groups reading the same data without duplication of the underlying stream.

    **Slack -- Job Queues with Redis and Kafka.** Slack uses Redis-based queues for latency-sensitive operations like message delivery (where sub-millisecond dequeue time matters for real-time chat) and Kafka for the durable event pipeline that feeds search indexing, analytics, and compliance archival. This dual-system approach uses each technology where it excels rather than forcing one tool to handle both real-time delivery and durable event processing. The Redis queues handle transient work that can be retried if lost, while Kafka provides the durability and replay guarantees needed for data pipelines.

    **Airbnb -- SQS for Decoupled Workflows.** Airbnb uses Amazon SQS extensively for asynchronous workflows: booking confirmations, host notifications, price calculations, and search index updates. At their scale of over 7 million active listings and 150 million users, SQS handles the variable load without any queue infrastructure management, letting teams focus on business logic rather than broker operations. The per-request pricing model means they pay nothing during quiet periods and scale seamlessly during peak booking seasons.

    **Spotify -- Google Pub/Sub for Event Delivery.** Spotify migrated from self-managed Kafka to Google Pub/Sub for a large portion of their event pipeline, processing billions of events per day (play events, ad impressions, user interactions). The managed nature of Pub/Sub eliminated the operational overhead of maintaining Kafka clusters across multiple regions while still providing the at-least-once delivery guarantees and multi-subscriber semantics they needed. They still use Kafka for workloads requiring exact offset control and long retention, demonstrating that the choice is often "both" rather than "either/or."

---

## Key Takeaways

1. **Point-to-point queues** distribute work across competing consumers where each message should be processed once. **Pub/sub topics** broadcast events to all interested subscribers. Choose based on whether you are assigning work or announcing facts.

2. **Kafka is a commit log, not a queue.** Messages are retained and replayable, consumer groups track their own offsets, and partitions are the unit of parallelism and ordering. This makes it ideal for event pipelines but overkill for simple task queues.

3. **Dead letter queues are not optional** in production systems. Without them, poison messages can block entire processing pipelines. Monitor DLQ depth, enrich failed messages with failure metadata, and review them regularly.

4. **Exactly-once delivery is a coordination problem**, not a transport guarantee. Most systems achieve effective exactly-once through at-least-once delivery combined with idempotent consumers that deduplicate on the receiving side.

5. **Exponential backoff with jitter** is the standard retry strategy. Without jitter, synchronized retries from many consumers can overwhelm a recovering service -- the thundering herd problem.

6. **No single messaging system fits every workload.** Companies like Uber, Slack, and Spotify run multiple systems in parallel -- low-latency queues for real-time work, Kafka for durable event streaming, managed services for operational simplicity. Choose based on your ordering, retention, throughput, and operational requirements.

---

## Related Topics

- [Event-Driven Architecture](../../architecture/event-driven.md) -- event sourcing, CQRS, choreography vs orchestration
- [Session Management](../session-management/sessions.md) -- managing state in asynchronous systems
- [API Design](../api-design/index.md) -- synchronous communication patterns that messaging complements
