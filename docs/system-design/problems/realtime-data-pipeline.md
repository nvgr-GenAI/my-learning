# Design Real-time Data Pipeline

A distributed streaming data processing platform that ingests, processes, and analyzes millions of events per second with exactly-once semantics, windowing, state management, and low-latency delivery to multiple sinks.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M+ events/sec, petabytes/day, millisecond latency, billions of state entries |
| **Key Challenges** | Exactly-once processing, windowing, watermarks, late data, back-pressure, state management, fault tolerance |
| **Core Concepts** | Stream processing, event time vs processing time, windowing (tumbling, sliding, session), checkpointing, stateful operators |
| **Companies** | Kafka, Apache Flink, Spark Streaming, AWS Kinesis, Confluent, Dataflow, LinkedIn, Uber, Netflix, Airbnb |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Stream Ingestion** | Ingest from Kafka, Kinesis, Pub/Sub at millions of events/sec | P0 (Must have) |
    | **Stream Processing** | Stateless and stateful transformations (map, filter, aggregate) | P0 (Must have) |
    | **Windowing** | Tumbling, sliding, and session windows | P0 (Must have) |
    | **Exactly-Once Semantics** | Guarantee each event processed exactly once | P0 (Must have) |
    | **State Management** | Distributed state store (RocksDB) with checkpointing | P0 (Must have) |
    | **Watermarks** | Handle late data and trigger window computations | P0 (Must have) |
    | **Back-pressure** | Flow control when downstream is slow | P0 (Must have) |
    | **Multiple Sinks** | Output to databases, data lakes, search, analytics | P0 (Must have) |
    | **Stream Joins** | Join multiple streams or stream-to-table joins | P1 (Should have) |
    | **Event Time Processing** | Process by event timestamp, not arrival time | P1 (Should have) |
    | **Schema Evolution** | Handle schema changes without pipeline restart | P1 (Should have) |
    | **Monitoring & Alerting** | Lag metrics, throughput, error rates, SLA alerts | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Batch processing (use Spark batch jobs instead)
    - Data warehouse queries (use separate OLAP system)
    - ML model training (use separate ML platform)
    - Complex event processing (CEP) rules engine
    - Data catalog/governance (use DataHub/Amundsen)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Throughput** | 10M events/sec | Support high-volume streams (IoT, clickstream, logs) |
    | **Latency (E2E)** | < 100ms p95 | Real-time analytics and alerting |
    | **Availability** | 99.99% uptime | Critical business applications |
    | **Durability** | Zero data loss | Exactly-once with checkpointing |
    | **Scalability** | Horizontal scaling (add workers) | Handle traffic spikes and growth |
    | **State Size** | Petabytes of state | Large aggregations (daily counts, session data) |
    | **Recovery Time** | < 5 minutes | Fast checkpoint restoration |
    | **Late Data** | Handle up to 1 hour late | Accommodate data delays and timezone issues |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Event ingestion:
    - Events/sec: 10M events/sec (peak)
    - Events/day: 10M × 86,400 = 864B events/day
    - Average event size: 1 KB (JSON payload)
    - Daily volume: 864B × 1 KB = 864 TB/day ≈ 1 PB/day

    Event sources:
    - User clickstream: 5M events/sec (50%)
    - Application logs: 2M events/sec (20%)
    - IoT sensors: 2M events/sec (20%)
    - Database CDC: 1M events/sec (10%)

    Processing stages:
    - Parse/validate: 10M events/sec
    - Enrich (lookup): 8M events/sec (80% need enrichment)
    - Aggregate (windows): 10M events/sec
    - Filter (drop): 3M events/sec dropped (30% spam/invalid)
    - Output: 7M events/sec to sinks

    Windowing workload:
    - Tumbling windows: 5M events/sec (1-min, 5-min windows)
    - Sliding windows: 3M events/sec (10-min window, 1-min slide)
    - Session windows: 2M events/sec (30-min gap timeout)

    State operations:
    - Stateful aggregations: 8M events/sec
    - State lookups: 8M lookups/sec
    - State updates: 8M updates/sec
    - State entries: 10B active keys (user sessions, counters)
    ```

    ### Storage Estimates

    ```
    State storage (RocksDB):

    Session state:
    - Active sessions: 100M concurrent users
    - State per session: 10 KB (session attributes, partial aggregations)
    - Total: 100M × 10 KB = 1 TB
    - With replication (3x): 3 TB

    Aggregation state (tumbling windows):
    - Active windows: 1,000 window keys × 24 hours × 60 min = 1.44M windows
    - State per window: 1 MB (counters, top-K, histograms)
    - Total: 1.44M × 1 MB = 1.44 TB
    - With replication (3x): 4.32 TB

    Sliding window state:
    - Windows: 10 windows × 1,000 keys = 10,000 windows
    - State per window: 5 MB
    - Total: 10,000 × 5 MB = 50 GB
    - With replication (3x): 150 GB

    Lookup tables (broadcast state):
    - User profiles: 1B users × 5 KB = 5 TB
    - Product catalog: 100M products × 2 KB = 200 GB
    - Geolocation: 10M locations × 1 KB = 10 GB
    - Total: ~5.2 TB (replicated to all workers)

    Changelog storage (Kafka topics):
    - State changes: 8M updates/sec × 2 KB = 16 GB/sec = 1.38 PB/day
    - Retention: 7 days → 9.7 PB
    - With compaction: ~3 TB (only latest per key)

    Checkpoints (S3/HDFS):
    - Full state snapshot: 10 TB (all state)
    - Checkpoint frequency: every 5 minutes
    - Checkpoints/day: 288 checkpoints
    - Daily checkpoint storage: 10 TB × 288 = 2.88 PB
    - With compression (5:1): 576 TB
    - Retention: 2 checkpoints → 1.15 PB

    Total storage:
    - State backends: 10 TB (hot state)
    - Kafka changelogs: 3 TB (compacted)
    - Checkpoints: 1.15 PB (S3)
    - Total: ~1.16 PB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (from Kafka):
    - 10M events/sec × 1 KB = 10 GB/sec = 80 Gbps
    - With replication (fetch from replicas): 80 Gbps

    State backend I/O (RocksDB):
    - State lookups: 8M/sec × 10 KB = 80 GB/sec = 640 Gbps
    - State updates: 8M/sec × 10 KB = 80 GB/sec = 640 Gbps
    - LSM compaction: 20% overhead → 128 Gbps
    - Total state I/O: ~1.4 Tbps (local SSD)

    Checkpoint bandwidth (to S3):
    - Checkpoint: 10 TB / 5 min = 33.3 GB/sec = 267 Gbps
    - Incremental checkpoints: ~10% of state = 26.7 Gbps

    Egress (to sinks):
    - Database writes: 2M events/sec × 1 KB = 2 GB/sec = 16 Gbps
    - S3 data lake: 3M events/sec × 1 KB = 3 GB/sec = 24 Gbps
    - Elasticsearch: 1M events/sec × 1 KB = 1 GB/sec = 8 Gbps
    - Analytics systems: 1M events/sec × 1 KB = 1 GB/sec = 8 Gbps
    - Total egress: 56 Gbps

    Network requirements:
    - Per worker: 10-40 Gbps (ingress + state + egress)
    - Total cluster: 10 Tbps (1,000 workers × 10 Gbps)
    ```

    ### Server Estimates

    ```
    Kafka cluster (stream storage):
    - Brokers: 100 nodes (handle 10M events/sec)
    - Per broker: 32 cores, 256 GB RAM, 20 TB NVMe SSD, 40 Gbps network
    - Partitions: 10,000 partitions (100 per broker)

    Flink cluster (stream processing):

    Job Managers (coordination):
    - Nodes: 3 (HA setup)
    - Per node: 16 cores, 64 GB RAM
    - Responsibilities: checkpoint coordination, scheduling, REST API

    Task Managers (data processing):
    - Nodes: 1,000 task managers
    - Per node: 32 cores, 128 GB RAM, 2 TB NVMe SSD (RocksDB state), 40 Gbps
    - Parallelism: 8 slots per TM × 1,000 TM = 8,000 parallel tasks
    - Throughput: 10M events/sec / 8,000 tasks = 1,250 events/sec/task

    State backend storage (RocksDB):
    - Local SSD: 2 TB per worker × 1,000 workers = 2 PB
    - Working set (hot data): 10 TB fits in RAM (page cache)
    - Cold data: 2 PB on SSD

    Checkpoint storage (S3/HDFS):
    - Managed service: 1.15 PB (no dedicated nodes)

    Monitoring & metadata:
    - Prometheus: 3 nodes (16 cores, 64 GB RAM)
    - ZooKeeper/Etcd: 3 nodes (8 cores, 32 GB RAM)
    - Schema Registry: 3 nodes (8 cores, 32 GB RAM)

    Total infrastructure:
    - Kafka brokers: 100 nodes
    - Flink Job Managers: 3 nodes
    - Flink Task Managers: 1,000 nodes
    - Monitoring: 9 nodes
    - Total: ~1,112 nodes
    ```

    ---

    ## Key Assumptions

    1. 10M events/sec peak throughput (864B events/day)
    2. Average event size: 1 KB (mix of small and large events)
    3. 70% of events need enrichment (lookup joins)
    4. 30% of events filtered out (spam, invalid, duplicates)
    5. Exactly-once semantics with 5-minute checkpoints
    6. Late data up to 1 hour handled by watermarks
    7. State size: 10 TB active state (100M sessions + aggregations)
    8. Recovery time < 5 minutes from last checkpoint

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Event time processing:** Process by event timestamp, not arrival time
    2. **Exactly-once semantics:** Two-phase commit with checkpointing
    3. **Stateful stream processing:** Distributed state with RocksDB
    4. **Windowing:** Tumbling, sliding, session windows for aggregations
    5. **Watermarks:** Handle late data and trigger window computations
    6. **Back-pressure:** Flow control using credit-based mechanism
    7. **Fault tolerance:** Checkpointing with incremental snapshots

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Event Sources"
            Producer1[Web Servers<br/>Clickstream<br/>5M events/sec]
            Producer2[App Servers<br/>Logs<br/>2M events/sec]
            Producer3[IoT Devices<br/>Sensors<br/>2M events/sec]
            Producer4[Databases<br/>CDC<br/>1M events/sec]
        end

        subgraph "Stream Ingestion Layer - Kafka"
            subgraph "Kafka Cluster"
                Topic1[events.clicks<br/>1000 partitions<br/>7-day retention]
                Topic2[events.logs<br/>500 partitions<br/>3-day retention]
                Topic3[events.iot<br/>500 partitions<br/>7-day retention]
                Topic4[events.cdc<br/>200 partitions<br/>30-day retention]
            end

            SchemaReg[Schema Registry<br/>Avro/Protobuf<br/>Schema validation]
        end

        subgraph "Stream Processing Layer - Flink"
            subgraph "Flink Cluster"
                JobMgr[Job Manager<br/>Checkpoint coordinator<br/>Task scheduler<br/>HA: 3 nodes]

                subgraph "Task Manager 1"
                    TM1_Source[Source<br/>Kafka consumer<br/>Deserialization]
                    TM1_Parse[Parse<br/>Validation<br/>Filtering]
                    TM1_State[(RocksDB State<br/>2 TB local SSD<br/>Session data)]
                end

                subgraph "Task Manager 2"
                    TM2_Enrich[Enrich<br/>Lookup join<br/>User profile]
                    TM2_Window[Windowing<br/>Tumbling: 1-min<br/>Sliding: 10-min]
                    TM2_State[(RocksDB State<br/>Window buffers<br/>Aggregations)]
                end

                subgraph "Task Manager N"
                    TMN_Agg[Aggregate<br/>Count, Sum, Avg<br/>Top-K, Percentiles]
                    TMN_Sink[Sink<br/>At-least-once<br/>Transactional]
                    TMN_State[(RocksDB State<br/>Counter state<br/>Top-K heap)]
                end
            end

            Checkpoint[Checkpoint Store<br/>S3/HDFS<br/>Incremental snapshots<br/>1.15 PB storage]
        end

        subgraph "State Management"
            StateBackend[State Backend<br/>RocksDB<br/>10 TB state<br/>8M ops/sec]
            Changelog[Changelog Topics<br/>Kafka compacted<br/>State replication<br/>3 TB]
        end

        subgraph "Watermark & Time"
            Watermark[Watermark Generator<br/>Periodic watermarks<br/>1-hour late data<br/>Out-of-order handling]
            EventTime[Event Time Extractor<br/>Parse timestamps<br/>Timezone handling]
        end

        subgraph "Output Sinks"
            DB[(PostgreSQL<br/>Aggregated metrics<br/>2M writes/sec)]
            ES[Elasticsearch<br/>Search index<br/>1M docs/sec]
            S3[S3 Data Lake<br/>Parquet files<br/>3M events/sec<br/>Hourly partitions]
            Redshift[(Redshift<br/>Analytics<br/>Batch loads<br/>1M rows/sec)]
            Alerts[Alert Service<br/>Anomaly detection<br/>SLA breaches]
        end

        subgraph "Monitoring & Observability"
            Metrics[Prometheus<br/>Lag, throughput<br/>Checkpoint duration<br/>Back-pressure]
            Logs[ELK Stack<br/>Task logs<br/>Error traces]
            Grafana[Grafana<br/>Dashboards<br/>Alerting]
            Lineage[Data Lineage<br/>Event provenance<br/>Impact analysis]
        end

        Producer1 -->|Produce| Topic1
        Producer2 -->|Produce| Topic2
        Producer3 -->|Produce| Topic3
        Producer4 -->|Produce| Topic4

        Producer1 -.->|Validate schema| SchemaReg

        Topic1 -->|Consume| TM1_Source
        Topic2 -->|Consume| TM1_Source
        Topic3 -->|Consume| TM1_Source
        Topic4 -->|Consume| TM1_Source

        TM1_Source --> TM1_Parse
        TM1_Parse --> TM2_Enrich
        TM1_Parse -.->|State ops| TM1_State

        TM2_Enrich --> TM2_Window
        TM2_Window --> TMN_Agg
        TM2_Enrich -.->|State ops| TM2_State
        TM2_Window -.->|State ops| TM2_State

        TMN_Agg --> TMN_Sink
        TMN_Agg -.->|State ops| TMN_State

        TMN_Sink --> DB
        TMN_Sink --> ES
        TMN_Sink --> S3
        TMN_Sink --> Redshift
        TMN_Sink --> Alerts

        JobMgr -.->|Schedule tasks| TM1_Source
        JobMgr -.->|Schedule tasks| TM2_Enrich
        JobMgr -.->|Schedule tasks| TMN_Agg

        JobMgr -->|Trigger checkpoint| Checkpoint
        TM1_State -->|Snapshot| Checkpoint
        TM2_State -->|Snapshot| Checkpoint
        TMN_State -->|Snapshot| Checkpoint

        TM1_State -->|Replicate| Changelog
        TM2_State -->|Replicate| Changelog
        TMN_State -->|Replicate| Changelog

        StateBackend -.->|Embedded in| TM1_State
        StateBackend -.->|Embedded in| TM2_State
        StateBackend -.->|Embedded in| TMN_State

        Watermark -.->|Emit watermarks| TM2_Window
        EventTime -.->|Extract time| TM1_Parse

        TM1_Source -->|Metrics| Metrics
        TM2_Enrich -->|Metrics| Metrics
        TMN_Agg -->|Metrics| Metrics
        JobMgr -->|Metrics| Metrics

        TM1_Source -->|Logs| Logs
        TM2_Enrich -->|Logs| Logs

        Metrics --> Grafana
        Grafana --> Alerts

        TM1_Parse -.->|Track lineage| Lineage

        style JobMgr fill:#90EE90
        style TM1_State fill:#FFE4B5
        style TM2_State fill:#FFE4B5
        style TMN_State fill:#FFE4B5
        style Checkpoint fill:#E6E6FA
        style Changelog fill:#FFF0F5
        style Watermark fill:#B0E0E6
        style Metrics fill:#FFE4E1
        style SchemaReg fill:#F0E68C
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Kafka** | Durable stream storage, replay capability, decoupling | Kinesis (vendor lock-in), Pulsar (less mature), RabbitMQ (no replay) |
    | **Flink** | Exactly-once semantics, stateful processing, low latency | Spark Streaming (micro-batch, higher latency), Storm (at-most-once), Samza (tied to Kafka) |
    | **RocksDB** | Embedded state backend, billions of keys, LSM-tree efficiency | In-memory (too expensive), Remote DB (network latency), LevelDB (single-threaded) |
    | **Watermarks** | Handle late data, trigger window computations | Drop late data (data loss), unbounded windows (memory explosion), fixed delay (inaccurate) |
    | **Checkpointing** | Exactly-once guarantee, fault tolerance, fast recovery | Logs replay (slow), external transactions (2PC overhead), no checkpoints (at-most-once) |
    | **Windowing** | Time-based aggregations, sessionization, real-time analytics | Global state (no time boundaries), micro-batching (higher latency), external aggregator (no ordering) |
    | **Back-pressure** | Prevent OOM, flow control, graceful degradation | Drop events (data loss), unbounded buffers (OOM), rate limiting (underutilization) |

    ---

    ## Data Flow

    **Example: Real-time Click Analytics Pipeline**

    ```
    1. Event Generation:
       - User clicks button → Web server logs event
       - Event: {user_id: "U123", page: "/checkout", timestamp: "2026-02-05T10:30:45Z"}

    2. Ingestion (Kafka):
       - Producer serializes to Avro using Schema Registry
       - Partitioned by user_id (maintain user ordering)
       - Published to "events.clicks" topic
       - Replicated to 3 brokers

    3. Consumption (Flink Source):
       - Flink consumer fetches from assigned partitions
       - Deserializes Avro → POJO
       - Extracts event timestamp (event time)
       - Emits watermark (current_time - 1 minute)

    4. Parsing & Validation:
       - Filter invalid events (missing fields, future timestamps)
       - Parse user agent, geo-location
       - Enrich with session_id from state

    5. Enrichment (Stateful):
       - Lookup user profile from broadcast state (1B users)
       - Join with product catalog (async I/O)
       - Add computed fields (device_type, country)

    6. Windowing:
       - Tumbling window: 1-minute counts per page
       - Sliding window: 10-minute moving average
       - Session window: User sessions with 30-min timeout

    7. Aggregation (Stateful):
       - Count clicks per page per window
       - Calculate percentiles (P50, P95, P99)
       - Top-K pages (heap-based algorithm)
       - Update state in RocksDB (keyed by window + page)

    8. Late Data Handling:
       - Event arrives 30 minutes late
       - Watermark allows up to 1 hour late
       - Re-open closed window, update aggregate, re-emit result
       - Downstream handles updates (upsert to database)

    9. Checkpointing (every 5 minutes):
       - Flink triggers checkpoint barrier
       - All operators snapshot state to S3 (incremental)
       - Kafka offsets committed transactionally
       - State changelog compacted in Kafka

    10. Output (Transactional Sinks):
        - PostgreSQL: Aggregated metrics (2-phase commit)
        - Elasticsearch: Searchable events (idempotent writes)
        - S3: Raw events in Parquet (atomic rename)
        - Alert service: Anomaly detection (at-least-once)

    11. Failure Recovery:
        - Task manager crashes → Job manager detects failure
        - All operators restore from last checkpoint (S3)
        - Kafka offsets reset to checkpoint
        - Processing resumes (exactly-once maintained)

    Processing time: 50-100ms end-to-end (event to sink)
    ```

=== "🔌 Step 3: API Design"

    ## Producer API (Kafka)

    ```java
    // Producer configuration
    Properties props = new Properties();
    props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
    props.put("value.serializer", "io.confluent.kafka.serializers.KafkaAvroSerializer");
    props.put("schema.registry.url", "http://schema-registry:8081");
    props.put("acks", "all");  // Wait for all replicas
    props.put("enable.idempotence", "true");  // Exactly-once producer
    props.put("compression.type", "snappy");
    props.put("batch.size", 16384);  // 16 KB batches
    props.put("linger.ms", 10);  // Wait 10ms to batch

    KafkaProducer<String, ClickEvent> producer = new KafkaProducer<>(props);

    // Produce event
    public void publishClickEvent(ClickEvent event) {
        ProducerRecord<String, ClickEvent> record = new ProducerRecord<>(
            "events.clicks",         // topic
            event.getUserId(),       // key (for partitioning)
            event                    // value
        );

        // Add headers
        record.headers().add("source", "web-server".getBytes());
        record.headers().add("version", "v2".getBytes());

        // Async send with callback
        producer.send(record, (metadata, exception) -> {
            if (exception != null) {
                log.error("Failed to send event: {}", event, exception);
                // Retry or dead-letter queue
            } else {
                log.debug("Event sent: topic={}, partition={}, offset={}, timestamp={}",
                    metadata.topic(), metadata.partition(),
                    metadata.offset(), metadata.timestamp());
            }
        });
    }

    // Flush and close
    producer.flush();
    producer.close(Duration.ofSeconds(10));
    ```

    ## Consumer API (Kafka)

    ```java
    // Consumer configuration
    Properties props = new Properties();
    props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");
    props.put("group.id", "flink-job-1");
    props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
    props.put("value.deserializer", "io.confluent.kafka.serializers.KafkaAvroDeserializer");
    props.put("schema.registry.url", "http://schema-registry:8081");
    props.put("enable.auto.commit", "false");  // Manual offset management
    props.put("isolation.level", "read_committed");  // Read only committed (for transactions)
    props.put("max.poll.records", 500);  // Fetch 500 records per poll
    props.put("fetch.min.bytes", 1048576);  // 1 MB minimum fetch

    KafkaConsumer<String, ClickEvent> consumer = new KafkaConsumer<>(props);
    consumer.subscribe(Arrays.asList("events.clicks"));

    // Poll loop
    while (true) {
        ConsumerRecords<String, ClickEvent> records = consumer.poll(Duration.ofMillis(100));

        for (ConsumerRecord<String, ClickEvent> record : records) {
            processEvent(record.key(), record.value(), record.timestamp());
        }

        // Manual commit after processing
        consumer.commitSync();
    }
    ```

    ## Flink DataStream API

    ```java
    // Flink execution environment
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
    env.enableCheckpointing(300000);  // Checkpoint every 5 minutes
    env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
    env.getCheckpointConfig().setMinPauseBetweenCheckpoints(60000);  // 1 min between checkpoints
    env.getCheckpointConfig().setCheckpointTimeout(600000);  // 10 min timeout
    env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

    // Configure state backend (RocksDB)
    env.setStateBackend(new RocksDBStateBackend("s3://checkpoints/flink", true));
    env.getCheckpointConfig().enableIncrementalCheckpointing(true);

    // Kafka source
    FlinkKafkaConsumer<ClickEvent> kafkaSource = new FlinkKafkaConsumer<>(
        "events.clicks",
        new AvroDeserializationSchema<>(ClickEvent.class),
        kafkaProps
    );

    // Assign timestamp and watermarks
    kafkaSource.assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<ClickEvent>forBoundedOutOfOrderness(Duration.ofMinutes(5))  // 5-min out-of-order
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
            .withIdleness(Duration.ofMinutes(1))  // Handle idle partitions
    );

    // Data stream pipeline
    DataStream<ClickEvent> clicks = env.addSource(kafkaSource);

    // Stateless transformation: filter and map
    DataStream<EnrichedClick> enrichedClicks = clicks
        .filter(event -> event.getUserId() != null)
        .map(new EnrichFunction());

    // Stateful transformation: keyed state for sessionization
    DataStream<SessionClick> sessionClicks = enrichedClicks
        .keyBy(event -> event.getUserId())
        .process(new SessionProcessFunction());

    // Windowing: tumbling window aggregation
    DataStream<PageStats> pageStats = sessionClicks
        .keyBy(event -> event.getPage())
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .allowedLateness(Time.minutes(5))  // Accept late data up to 5 minutes
        .sideOutputLateData(lateDataTag)   // Capture very late data
        .aggregate(new PageStatsAggregator());

    // Sliding window: moving average
    DataStream<PageStats> movingAvg = sessionClicks
        .keyBy(event -> event.getPage())
        .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
        .aggregate(new MovingAverageAggregator());

    // Session window: user sessions with 30-min timeout
    DataStream<UserSession> userSessions = sessionClicks
        .keyBy(event -> event.getUserId())
        .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
        .process(new SessionAggregationFunction());

    // Sinks
    pageStats.addSink(new JdbcSink<>(...));  // PostgreSQL
    pageStats.addSink(new ElasticsearchSink<>(...));  // Elasticsearch
    pageStats.addSink(new StreamingFileSink<>(...));  // S3 Parquet

    env.execute("Real-time Click Analytics");
    ```

    ## Stateful Processing

    ```java
    // Keyed process function with state
    public class SessionProcessFunction extends KeyedProcessFunction<String, EnrichedClick, SessionClick> {

        // Value state: store session metadata
        private ValueState<SessionMetadata> sessionState;

        // List state: store recent events
        private ListState<EnrichedClick> recentEvents;

        // Map state: store page visit counts
        private MapState<String, Long> pageVisitCounts;

        @Override
        public void open(Configuration parameters) {
            // Initialize state descriptors
            ValueStateDescriptor<SessionMetadata> sessionDesc =
                new ValueStateDescriptor<>("session", SessionMetadata.class);
            sessionState = getRuntimeContext().getState(sessionDesc);

            ListStateDescriptor<EnrichedClick> eventsDesc =
                new ListStateDescriptor<>("recent-events", EnrichedClick.class);
            recentEvents = getRuntimeContext().getListState(eventsDesc);

            MapStateDescriptor<String, Long> countsDesc =
                new MapStateDescriptor<>("page-counts", String.class, Long.class);
            pageVisitCounts = getRuntimeContext().getMapState(countsDesc);
        }

        @Override
        public void processElement(EnrichedClick event, Context ctx, Collector<SessionClick> out)
                throws Exception {

            SessionMetadata session = sessionState.value();

            // Create new session if needed
            if (session == null || isSessionExpired(session, event.getTimestamp())) {
                session = new SessionMetadata(
                    UUID.randomUUID().toString(),
                    event.getTimestamp()
                );
                sessionState.update(session);

                // Clear state for new session
                recentEvents.clear();
                pageVisitCounts.clear();

                // Register timer for session timeout
                ctx.timerService().registerEventTimeTimer(
                    event.getTimestamp() + Duration.ofMinutes(30).toMillis()
                );
            }

            // Update session state
            session.setLastEventTime(event.getTimestamp());
            session.incrementEventCount();
            sessionState.update(session);

            // Update page visit counts
            String page = event.getPage();
            Long count = pageVisitCounts.get(page);
            pageVisitCounts.put(page, (count == null ? 0 : count) + 1);

            // Add to recent events (keep last 100)
            recentEvents.add(event);
            if (getListSize(recentEvents) > 100) {
                removeOldestElement(recentEvents);
            }

            // Emit enriched event with session
            SessionClick sessionClick = new SessionClick(
                event,
                session.getSessionId(),
                session.getEventCount()
            );
            out.collect(sessionClick);
        }

        @Override
        public void onTimer(long timestamp, OnTimerContext ctx, Collector<SessionClick> out)
                throws Exception {
            // Session timeout: close session and clear state
            SessionMetadata session = sessionState.value();
            if (session != null) {
                log.info("Session {} expired for user {}", session.getSessionId(), ctx.getCurrentKey());

                // Emit session summary
                SessionSummary summary = new SessionSummary(
                    session.getSessionId(),
                    session.getStartTime(),
                    session.getLastEventTime(),
                    session.getEventCount(),
                    pageVisitCounts
                );
                // Send to summary stream...

                // Clear state
                sessionState.clear();
                recentEvents.clear();
                pageVisitCounts.clear();
            }
        }

        private boolean isSessionExpired(SessionMetadata session, long eventTime) {
            return eventTime - session.getLastEventTime() > Duration.ofMinutes(30).toMillis();
        }
    }
    ```

    ## Window Aggregation

    ```java
    // Tumbling window aggregator
    public class PageStatsAggregator implements AggregateFunction<
            SessionClick,         // Input type
            PageStatsAccumulator, // Accumulator type
            PageStats             // Output type
        > {

        @Override
        public PageStatsAccumulator createAccumulator() {
            return new PageStatsAccumulator();
        }

        @Override
        public PageStatsAccumulator add(SessionClick click, PageStatsAccumulator acc) {
            acc.count++;
            acc.uniqueUsers.add(click.getUserId());
            acc.totalDuration += click.getDuration();
            acc.bounces += (click.isBounce() ? 1 : 0);
            return acc;
        }

        @Override
        public PageStats getResult(PageStatsAccumulator acc) {
            return new PageStats(
                acc.page,
                acc.windowStart,
                acc.windowEnd,
                acc.count,
                acc.uniqueUsers.size(),
                acc.totalDuration / acc.count,  // avg duration
                (double) acc.bounces / acc.count  // bounce rate
            );
        }

        @Override
        public PageStatsAccumulator merge(PageStatsAccumulator a, PageStatsAccumulator b) {
            a.count += b.count;
            a.uniqueUsers.addAll(b.uniqueUsers);
            a.totalDuration += b.totalDuration;
            a.bounces += b.bounces;
            return a;
        }
    }

    // Accumulator class
    public static class PageStatsAccumulator {
        String page;
        long windowStart;
        long windowEnd;
        long count = 0;
        Set<String> uniqueUsers = new HashSet<>();
        long totalDuration = 0;
        long bounces = 0;
    }
    ```

    ## Watermark Strategy

    ```java
    // Custom watermark generator
    public class BoundedOutOfOrdernessWatermarks implements WatermarkGenerator<ClickEvent> {

        private final long maxOutOfOrderness = 60_000; // 1 minute
        private long currentMaxTimestamp = Long.MIN_VALUE;

        @Override
        public void onEvent(ClickEvent event, long eventTimestamp, WatermarkOutput output) {
            currentMaxTimestamp = Math.max(currentMaxTimestamp, eventTimestamp);
        }

        @Override
        public void onPeriodicEmit(WatermarkOutput output) {
            // Emit watermark = max timestamp - max out-of-orderness
            output.emitWatermark(new Watermark(currentMaxTimestamp - maxOutOfOrderness));
        }
    }

    // Punctuated watermarks (per-event)
    public class PunctuatedWatermarks implements WatermarkGenerator<ClickEvent> {

        @Override
        public void onEvent(ClickEvent event, long eventTimestamp, WatermarkOutput output) {
            // Emit watermark if event is end-of-stream marker
            if (event.isEndOfStream()) {
                output.emitWatermark(new Watermark(eventTimestamp));
            }
        }

        @Override
        public void onPeriodicEmit(WatermarkOutput output) {
            // Don't emit periodic watermarks
        }
    }
    ```

=== "🔍 Step 4: Deep Dive"

    ## 4.1 Exactly-Once Processing

    **Challenge:** Ensure each event is processed exactly once, even with failures.

    **Components:**

    1. **Idempotent Source:**
       - Kafka: Offsets are part of checkpoint
       - Can replay from last checkpoint offset
       - No duplicates from source

    2. **Checkpointing (Chandy-Lamport Algorithm):**
       ```
       Step 1: Job Manager triggers checkpoint (every 5 min)

       Step 2: Source operator receives checkpoint barrier
           - Snapshot Kafka offsets
           - Emit barrier downstream

       Step 3: Intermediate operators receive barrier
           - Snapshot state to RocksDB
           - Flush in-flight records
           - Emit barrier downstream

       Step 4: Sink operator receives barrier
           - Pre-commit transactions (2PC phase 1)
           - Acknowledge checkpoint to Job Manager

       Step 5: Job Manager receives all acknowledgments
           - Mark checkpoint complete
           - Commit transactions (2PC phase 2)
           - Advance checkpoint ID

       On failure:
           - Restore all operators from last checkpoint
           - Reset Kafka offsets to checkpoint
           - Abort uncommitted transactions
           - Resume processing
       ```

    3. **Transactional Sinks (2-Phase Commit):**
       ```java
       // Two-phase commit sink
       public class TwoPhaseCommitSink<IN> extends TwoPhaseCommitSinkFunction<IN, Transaction, Void> {

           @Override
           protected Transaction beginTransaction() {
               // Phase 1: Start transaction
               return database.beginTransaction();
           }

           @Override
           protected void invoke(Transaction txn, IN value, Context ctx) {
               // Write to transaction buffer (not committed)
               txn.write(value);
           }

           @Override
           protected void preCommit(Transaction txn) {
               // Pre-commit: prepare transaction
               // (happens before checkpoint completes)
               txn.prepare();
           }

           @Override
           protected void commit(Transaction txn) {
               // Commit: finalize transaction
               // (happens after checkpoint completes)
               txn.commit();
           }

           @Override
           protected void abort(Transaction txn) {
               // Abort: rollback on failure
               txn.rollback();
           }
       }
       ```

    4. **Changelog-based State Backend:**
       ```
       Every state update:
       1. Write to local RocksDB (fast)
       2. Append to Kafka changelog topic (durable)
       3. Checkpoint: Snapshot RocksDB to S3 (incremental)

       On recovery:
       1. Restore from last S3 snapshot
       2. Replay changelog from snapshot offset
       3. Rebuild full state
       ```

    **Performance:**
    - Checkpoint overhead: 5-10% throughput reduction
    - Recovery time: 2-5 minutes for 10 TB state
    - Incremental checkpoints: Only changed keys (10-30% of state)

    ---

    ## 4.2 Windowing

    **Window Types:**

    ### 1. Tumbling Windows (Non-overlapping)
    ```
    Time:  00:00  00:01  00:02  00:03  00:04  00:05
    Win 1: [-----------)
    Win 2:             [-----------)
    Win 3:                         [-----------)

    Use case: 1-minute aggregations (event counts, sums)
    State: 1 accumulator per window per key
    Memory: O(keys × windows) = O(keys) (only current window)
    ```

    ```java
    // Tumbling window
    stream
        .keyBy(event -> event.getPage())
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .aggregate(new CountAggregator());
    ```

    ### 2. Sliding Windows (Overlapping)
    ```
    Time:  00:00  00:01  00:02  00:03  00:04  00:05
    Win 1: [---------------------)
    Win 2:       [---------------------)
    Win 3:             [---------------------)

    Use case: Moving average (10-min window, 1-min slide)
    State: Multiple accumulators (window_size / slide)
    Memory: O(keys × (window_size / slide))
    ```

    ```java
    // Sliding window
    stream
        .keyBy(event -> event.getPage())
        .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
        .aggregate(new AverageAggregator());
    ```

    ### 3. Session Windows (Gap-based)
    ```
    Events:  |  |     |     |        |  |  |
    Session1: [---]
    Session2:         [-----]
    Session3:                      [--------]

    Gap: 30 minutes of inactivity ends session

    Use case: User sessions, clickstream analysis
    State: Dynamic windows per key (created on-the-fly)
    Memory: O(active_sessions)
    ```

    ```java
    // Session window
    stream
        .keyBy(event -> event.getUserId())
        .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
        .process(new SessionAggregator());
    ```

    **Window Lifecycle:**
    ```
    1. Window created: First event in window arrives
    2. Events accumulated: Buffer events in state
    3. Watermark advances: Time progresses
    4. Watermark passes window end: Trigger computation
    5. Aggregation computed: Process buffered events
    6. Result emitted: Send to downstream
    7. Late data arrives: Re-open window (if allowed)
    8. Window purged: Clear state after retention period
    ```

    **Late Data Handling:**
    ```java
    stream
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .allowedLateness(Time.minutes(5))     // Reprocess up to 5 min late
        .sideOutputLateData(lateDataTag)      // Capture very late data
        .aggregate(new Aggregator());

    // Very late data (> 5 min) sent to side output
    DataStream<Event> veryLateData = stream.getSideOutput(lateDataTag);
    veryLateData.addSink(new DeadLetterSink());
    ```

    ---

    ## 4.3 State Management

    **State Types:**

    | State Type | Use Case | Example | Memory |
    |------------|----------|---------|--------|
    | **ValueState** | Single value per key | Session metadata, last event | O(keys) |
    | **ListState** | List of values | Recent events buffer | O(keys × list_size) |
    | **MapState** | Key-value map | Page visit counts, user attributes | O(keys × map_size) |
    | **ReducingState** | Incrementally reduced | Running sum, count | O(keys) |
    | **AggregatingState** | Custom aggregation | Average, percentiles, top-K | O(keys) |

    **State Backends:**

    ### 1. MemoryStateBackend
    ```
    Pros: Fastest (no serialization)
    Cons: Limited by heap size, no persistence
    Use case: Testing, small state (< 1 GB)
    ```

    ### 2. FsStateBackend
    ```
    Pros: Larger state (100s GB), checkpoints to HDFS/S3
    Cons: Heap-based (GC pressure), slower than memory
    Use case: Medium state (1-100 GB)
    ```

    ### 3. RocksDBStateBackend (Recommended)
    ```
    Pros: Massive state (TBs), off-heap, incremental checkpoints
    Cons: Serialization overhead, disk I/O
    Use case: Large state (> 100 GB), production workloads

    Storage:
    - Working set: In-memory (page cache)
    - Cold data: Local SSD (RocksDB files)
    - Checkpoints: S3/HDFS (snapshots)

    LSM Tree structure:
    - Writes: MemTable (in-memory) → flush to SSTable (disk)
    - Reads: Check MemTable → L0 → L1 → ... → L6 (levels)
    - Compaction: Merge SSTables to reduce read amplification
    ```

    **State Configuration:**
    ```java
    // RocksDB tuning
    RocksDBStateBackend stateBackend = new RocksDBStateBackend("s3://checkpoints", true);

    // Incremental checkpoints (only changed keys)
    stateBackend.enableIncrementalCheckpointing(true);

    // Predefined options (best practices)
    stateBackend.setPredefinedOptions(PredefinedOptions.SPINNING_DISK_OPTIMIZED);

    // Custom RocksDB options
    stateBackend.setRocksDBOptions(new RocksDBOptionsFactory() {
        @Override
        public DBOptions createDBOptions(DBOptions currentOptions,
                                          Collection<AutoCloseable> handlesToClose) {
            return currentOptions
                .setIncreaseParallelism(4)
                .setMaxBackgroundJobs(4);
        }

        @Override
        public ColumnFamilyOptions createColumnOptions(ColumnFamilyOptions currentOptions,
                                                        Collection<AutoCloseable> handlesToClose) {
            return currentOptions
                .setCompactionStyle(CompactionStyle.LEVEL)
                .setWriteBufferSize(64 * 1024 * 1024)  // 64 MB
                .setMaxWriteBufferNumber(3)
                .setMinWriteBufferNumberToMerge(1);
        }
    });

    env.setStateBackend(stateBackend);
    ```

    **State TTL (Time-To-Live):**
    ```java
    // Clean up old state automatically
    StateTtlConfig ttlConfig = StateTtlConfig
        .newBuilder(Time.hours(24))
        .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
        .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
        .cleanupFullSnapshot()  // Clean up during checkpoint
        .build();

    ValueStateDescriptor<String> descriptor =
        new ValueStateDescriptor<>("my-state", String.class);
    descriptor.enableTimeToLive(ttlConfig);

    ValueState<String> state = getRuntimeContext().getState(descriptor);
    ```

    ---

    ## 4.4 Watermarks and Event Time

    **Watermark Definition:**
    ```
    Watermark(t) = "All events with timestamp ≤ t have been received"

    Example:
    - Current watermark: 10:30:00
    - Meaning: All events up to 10:30:00 have arrived
    - Action: Can trigger windows ending at or before 10:30:00
    ```

    **Watermark Propagation:**
    ```
    Source1 (Watermark: 10:30:00) ─┐
                                      ├─→ Operator (min(10:30:00, 10:29:00) = 10:29:00)
    Source2 (Watermark: 10:29:00) ─┘

    Rule: Use minimum watermark from all inputs
    Reason: Can't trigger window until all inputs confirm events arrived
    ```

    **Watermark Strategies:**

    ### 1. Periodic Watermarks (Most Common)
    ```java
    // Emit watermark every 200ms
    WatermarkStrategy
        .<Event>forBoundedOutOfOrderness(Duration.ofMinutes(1))
        .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    ```

    ### 2. Punctuated Watermarks
    ```java
    // Emit watermark on special events
    public class SpecialEventWatermarks implements WatermarkGenerator<Event> {
        @Override
        public void onEvent(Event event, long eventTimestamp, WatermarkOutput output) {
            if (event.isEndOfBatch()) {
                output.emitWatermark(new Watermark(eventTimestamp));
            }
        }
    }
    ```

    ### 3. Idle Source Handling
    ```java
    // Mark source as idle if no events for 1 minute
    WatermarkStrategy
        .<Event>forBoundedOutOfOrderness(Duration.ofMinutes(1))
        .withIdleness(Duration.ofMinutes(1))
    // Prevents idle sources from blocking watermark progress
    ```

    **Out-of-Order Events:**
    ```
    Events arrive:  e1(10:00) → e2(10:02) → e3(10:01) → e4(10:03)

    Watermark strategy: 1-minute bounded out-of-orderness

    Timeline:
    - e1(10:00): Watermark = 10:00 - 1 min = 09:59
    - e2(10:02): Watermark = 10:02 - 1 min = 10:01
    - e3(10:01): Late by 0 seconds (< 1 min), accepted
    - e4(10:03): Watermark = 10:03 - 1 min = 10:02

    If e5(09:58) arrives now: Late by 4 minutes (> 1 min), rejected or side output
    ```

    ---

    ## 4.5 Back-pressure

    **Problem:** Downstream operator slower than upstream → buffer overflow → OOM

    **Flink's Credit-Based Flow Control:**
    ```
    Upstream Task:   [Buffer Pool] ──→ Network Buffer ──→ Socket
                           ↑                               |
                           └─────── Credit feedback ───────┘

    Downstream Task: Socket ──→ Network Buffer ──→ [Buffer Pool]
                                        |
                                    Send credit (available buffers)

    1. Downstream has N available buffers
    2. Send "credit = N" to upstream
    3. Upstream sends ≤ N buffers
    4. Downstream processes buffers, sends new credit

    If credit = 0: Upstream blocks, propagates back-pressure
    ```

    **Back-pressure Indicators:**
    ```
    Metrics:
    - outPoolUsage: Output buffer pool usage (0-1)
    - inPoolUsage: Input buffer pool usage (0-1)
    - isBackPressured: Boolean flag

    Alert if:
    - outPoolUsage > 0.9 (upstream blocked)
    - inPoolUsage > 0.9 (downstream slow)
    ```

    **Handling Back-pressure:**

    1. **Scale Out:**
       ```
       Increase parallelism: 100 → 200 tasks
       More capacity to process events
       ```

    2. **Async I/O:**
       ```java
       // Non-blocking enrichment
       AsyncDataStream.unorderedWait(
           stream,
           new AsyncDatabaseLookup(),
           5000,  // timeout: 5 seconds
           TimeUnit.MILLISECONDS,
           100    // max concurrent requests
       );
       ```

    3. **Buffer Tuning:**
       ```java
       // Increase network buffers
       taskmanager.network.memory.fraction: 0.2  // 20% of heap
       taskmanager.network.memory.min: 256mb
       taskmanager.network.memory.max: 2gb
       ```

    4. **Source Throttling:**
       ```java
       // Limit ingestion rate
       kafkaSource.setStartupMode(StartupMode.GROUP_OFFSETS);
       kafkaSource.setCommitOffsetsOnCheckpoints(true);
       // Kafka partition lag increases → natural throttling
       ```

    ---

    ## 4.6 Stream Joins

    ### 1. Window Join (Both Streams Windowed)
    ```java
    // Join clicks and purchases within 10-minute window
    DataStream<Click> clicks = ...;
    DataStream<Purchase> purchases = ...;

    DataStream<ClickPurchase> joined = clicks
        .join(purchases)
        .where(click -> click.getUserId())        // Key from clicks
        .equalTo(purchase -> purchase.getUserId())  // Key from purchases
        .window(TumblingEventTimeWindows.of(Time.minutes(10)))
        .apply((click, purchase) -> new ClickPurchase(click, purchase));
    ```

    ### 2. Interval Join (Event-Time Range)
    ```java
    // Join clicks with purchases that happen within 1 hour after click
    clicks
        .keyBy(click -> click.getUserId())
        .intervalJoin(purchases.keyBy(purchase -> purchase.getUserId()))
        .between(Time.seconds(0), Time.hours(1))  // -0s to +1h
        .process(new JoinFunction());

    // Example:
    // Click at 10:00 → joins with purchases from 10:00 to 11:00
    ```

    ### 3. Temporal Table Join (Stream-to-Table)
    ```java
    // Join stream with versioned table (e.g., product catalog)
    // Table updates with timestamp → temporal versions

    DataStream<Click> clicks = ...;

    // Create temporal table from changelog stream
    TemporalTableFunction productCatalog = productUpdates
        .createTemporalTableFunction("updateTime", "productId");

    // Join: Lookup product at click time
    Table result = clicks
        .joinLateral(productCatalog(click.timestamp),
                     "productId = " + click.productId)
        .select("click.*, product.name, product.price");
    ```

=== "📈 Step 5: Scalability & Optimization"

    ## Scaling Strategies

    ### 1. Horizontal Scaling

    **Kafka Partitioning:**
    ```
    Scale out: Add more partitions
    - 1,000 partitions → 10,000 partitions
    - Repartition topic (create new topic, migrate data)
    - Trade-off: More partitions = more files, longer recovery

    Partitioning strategy:
    - Hash by key: Hash(user_id) % num_partitions
    - Round-robin: No key → balanced load
    - Custom partitioner: Geography-aware, range-based
    ```

    **Flink Parallelism:**
    ```java
    // Operator-level parallelism
    stream
        .filter(...).setParallelism(100)      // 100 parallel filter tasks
        .map(...).setParallelism(200)         // 200 parallel map tasks
        .keyBy(...)
        .window(...)
        .aggregate(...).setParallelism(500);  // 500 parallel aggregation tasks

    // Job-level parallelism
    env.setParallelism(1000);  // Default for all operators

    // Rebalance (redistribute data evenly)
    stream.rebalance().map(...);  // Avoid skew

    // Rescale (local redistribution within task manager)
    stream.rescale().map(...);  // Reduce network shuffle
    ```

    **Task Manager Sizing:**
    ```
    Scenario: 10M events/sec, 1 KB events, 1,000 events/sec per task

    Required parallelism: 10M / 1,000 = 10,000 tasks

    Task Manager config:
    - Slots per TM: 8 (CPU cores)
    - Task Managers: 10,000 / 8 = 1,250 TMs

    Resources per TM:
    - CPU: 32 cores (for 8 slots + OS)
    - Memory: 128 GB (16 GB per slot)
    - Disk: 2 TB NVMe SSD (RocksDB state)
    - Network: 40 Gbps
    ```

    ---

    ### 2. State Sharding

    **Key-based Sharding:**
    ```
    State distributed by key:
    - user_id "U123" → Task 1
    - user_id "U456" → Task 2
    - user_id "U789" → Task 3

    Hash function: Hash(key) % parallelism

    Benefits:
    - Parallel state access (no contention)
    - Co-located computation (no network)

    Challenges:
    - Data skew (hot keys)
    - Rebalancing (rescaling changes hash)
    ```

    **State Size Optimization:**
    ```java
    // Use efficient serializers
    env.registerTypeWithKryoSerializer(MyClass.class, MySerializer.class);

    // Enable compression
    stateBackend.enableCompression(true);

    // State TTL (remove old data)
    StateTtlConfig ttl = StateTtlConfig.newBuilder(Time.days(7)).build();
    stateDescriptor.enableTimeToLive(ttl);

    // Incremental checkpoints (only deltas)
    stateBackend.enableIncrementalCheckpointing(true);
    ```

    ---

    ### 3. Checkpoint Optimization

    **Incremental Checkpointing:**
    ```
    Full checkpoint: Snapshot entire state (10 TB)
    Incremental checkpoint: Only changed keys (1-3 TB)

    RocksDB incremental:
    - Checkpoint 1: Full snapshot → 10 TB
    - Checkpoint 2: SSTable files changed → 1.5 TB
    - Checkpoint 3: SSTable files changed → 2 TB
    - ...
    - Recovery: Base + deltas = full state

    Savings: 70-90% less data written to S3
    ```

    **Asynchronous Checkpoints:**
    ```
    Timeline:
    1. Checkpoint barrier arrives
    2. Operator snapshots state (sync, fast)
    3. State uploaded to S3 (async, slow)
    4. Operator continues processing

    Benefit: No blocking during S3 upload
    ```

    **Checkpoint Configuration:**
    ```java
    CheckpointConfig config = env.getCheckpointConfig();

    // Checkpoint interval: 5 minutes
    config.setCheckpointInterval(300_000);

    // Min pause between checkpoints: 1 minute
    config.setMinPauseBetweenCheckpoints(60_000);

    // Checkpoint timeout: 10 minutes
    config.setCheckpointTimeout(600_000);

    // Max concurrent checkpoints: 1
    config.setMaxConcurrentCheckpoints(1);

    // Alignment timeout: Use unaligned checkpoints if > 10s
    config.setAlignmentTimeout(Duration.ofSeconds(10));
    config.setUnalignedCheckpointsEnabled(true);

    // Retain checkpoints on cancellation
    config.enableExternalizedCheckpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    );
    ```

    ---

    ### 4. Network Optimization

    **Network Buffers:**
    ```
    Buffer per channel: 32 KB (default)
    Channels: parallelism × parallelism (full mesh)

    Example:
    - Parallelism: 1,000 tasks
    - Channels: 1,000 × 1,000 = 1M channels
    - Buffers: 2 per channel × 1M = 2M buffers
    - Memory: 2M × 32 KB = 64 GB

    Optimization:
    - Increase buffer size: 32 KB → 64 KB (fewer syscalls)
    - Limit parallelism: Use rescale() instead of rebalance()
    - Batch writes: Flush every 10ms or 64 KB
    ```

    **Data Serialization:**
    ```java
    // Fast serializers
    1. Flink's PojoSerializer (reflection-based, schema evolution)
    2. Kryo (general-purpose, slower than Pojo)
    3. Avro (schema registry, versioning, efficient)
    4. Protobuf (compact, fast, code generation)

    // Register types
    env.registerType(MyClass.class);  // Use PojoSerializer
    env.addDefaultKryoSerializer(MyClass.class, MySerializer.class);  // Custom Kryo

    // Disable Kryo (force explicit registration)
    env.getConfig().disableGenericTypes();
    ```

    ---

    ## Performance Tuning

    **Kafka Consumer Tuning:**
    ```properties
    # Fetch more data per request
    fetch.min.bytes=1048576          # 1 MB
    fetch.max.wait.ms=500            # Wait 500ms to batch
    max.partition.fetch.bytes=10485760  # 10 MB per partition

    # Consumer parallelism
    max.poll.records=500             # Process 500 records per poll

    # Commit strategy
    enable.auto.commit=false         # Manual commit (Flink controls)
    isolation.level=read_committed   # Exactly-once
    ```

    **Flink Memory Tuning:**
    ```yaml
    taskmanager.memory.process.size: 32gb
    taskmanager.memory.flink.size: 28gb
    taskmanager.memory.managed.fraction: 0.4    # RocksDB: 40%
    taskmanager.memory.network.fraction: 0.2    # Network: 20%
    taskmanager.memory.jvm-overhead.fraction: 0.1  # JVM: 10%
    ```

    **RocksDB Tuning:**
    ```yaml
    # More write buffers (reduce LSM compaction)
    state.backend.rocksdb.writebuffer.size: 64mb
    state.backend.rocksdb.writebuffer.count: 4

    # Block cache (read performance)
    state.backend.rocksdb.block.cache-size: 512mb

    # Compaction threads
    state.backend.rocksdb.thread.num: 4
    ```

    **Watermark Tuning:**
    ```java
    // Increase out-of-order tolerance (reduce reprocessing)
    WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofMinutes(5));

    // Emit watermarks less frequently (reduce overhead)
    env.getConfig().setAutoWatermarkInterval(1000);  // 1 second
    ```

    ---

    ## Monitoring Metrics

    **Key Metrics:**
    ```
    Throughput:
    - numRecordsInPerSecond: Events ingested per second
    - numRecordsOutPerSecond: Events emitted per second

    Latency:
    - latency: End-to-end processing latency (P50, P95, P99)
    - eventTimeLag: Difference between event time and processing time

    State:
    - rocksdbStateBackendBytesRead: RocksDB read throughput
    - rocksdbStateBackendBytesWritten: RocksDB write throughput
    - rocksdbStateBackendMemtableFlushes: MemTable flush rate

    Checkpoints:
    - lastCheckpointDuration: Time to complete checkpoint
    - lastCheckpointSize: Checkpoint size in bytes
    - numberOfFailedCheckpoints: Failed checkpoint count

    Back-pressure:
    - isBackPressured: Boolean flag per task
    - outPoolUsage: Output buffer pool usage (0-1)

    Kafka:
    - records-lag-max: Consumer lag per partition
    - fetch-latency-avg: Fetch latency from brokers
    ```

    **Alerting Rules:**
    ```yaml
    # High latency
    alert: HighProcessingLatency
    expr: flink_taskmanager_job_latency_p95 > 5000  # > 5 seconds

    # Consumer lag
    alert: HighConsumerLag
    expr: kafka_consumer_records_lag_max > 1000000  # > 1M events

    # Checkpoint failures
    alert: CheckpointFailures
    expr: rate(flink_jobmanager_job_numberOfFailedCheckpoints[5m]) > 0

    # Back-pressure
    alert: BackPressure
    expr: flink_taskmanager_job_task_isBackPressured == 1
    ```

=== "💡 Step 6: Interview Tips"

    ## What Interviewers Look For

    1. **Understanding of stream semantics:**
       - Difference between event time and processing time
       - Out-of-order events and late data handling
       - Watermarks and window triggering

    2. **State management knowledge:**
       - When to use keyed state vs. operator state
       - State backends trade-offs (Memory vs. RocksDB)
       - Checkpointing and recovery

    3. **Scalability thinking:**
       - How to scale ingestion (Kafka partitions)
       - How to scale processing (Flink parallelism)
       - Handling data skew (hot keys)

    4. **Fault tolerance:**
       - Exactly-once vs. at-least-once semantics
       - How checkpointing works (Chandy-Lamport)
       - Recovery time and state restoration

    5. **Performance optimization:**
       - Back-pressure handling
       - Network buffer tuning
       - Async I/O for enrichment

    ---

    ## Common Follow-up Questions

    ### Q1: How do you handle late data?
    ```
    Answer:
    1. Watermarks: Allow bounded out-of-orderness (e.g., 1 hour)
    2. allowedLateness(): Reprocess windows for late events
    3. Side outputs: Capture very late data to dead-letter queue
    4. Update downstream: Use upsert to correct previous results

    Code:
    stream
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .allowedLateness(Time.hours(1))
        .sideOutputLateData(lateDataTag)
        .aggregate(new Aggregator());
    ```

    ### Q2: How do you achieve exactly-once semantics?
    ```
    Answer:
    1. Idempotent source: Kafka offsets in checkpoint
    2. Checkpointing: Snapshot state + offsets every 5 minutes
    3. Transactional sinks: 2-phase commit (prepare, commit)
    4. Recovery: Restore from checkpoint, reset offsets, abort transactions

    Requirements:
    - Kafka: Transactions enabled, isolation.level=read_committed
    - Flink: CheckpointingMode.EXACTLY_ONCE
    - Sinks: TwoPhaseCommitSinkFunction or idempotent writes
    ```

    ### Q3: How do you handle data skew (hot keys)?
    ```
    Answer:
    1. Pre-aggregation: Reduce data before shuffle
    2. Salting: Add random suffix to hot keys → distribute load
    3. Two-phase aggregation: Local agg + global agg
    4. Custom partitioner: Distribute hot keys to multiple tasks

    Example (two-phase):
    stream
        .keyBy(event -> event.getKey() + "-" + random.nextInt(10))  // Salt
        .window(...)
        .aggregate(new LocalAggregator())
        .keyBy(event -> event.getKey())  // Original key
        .window(...)
        .aggregate(new GlobalAggregator());
    ```

    ### Q4: How do you monitor pipeline health?
    ```
    Answer:
    1. Lag metrics: Kafka consumer lag (events behind)
    2. Latency metrics: End-to-end processing time (P95, P99)
    3. Checkpoint metrics: Duration, size, failures
    4. Back-pressure: Detect slow operators
    5. Throughput: Events per second (in/out)
    6. Errors: Task failures, restarts

    Tools:
    - Prometheus + Grafana (metrics)
    - Flink Web UI (topology, back-pressure)
    - Kafka Manager (lag, partition distribution)
    - ELK Stack (logs, error traces)
    ```

    ### Q5: Session window vs. tumbling window?
    ```
    Answer:

    Tumbling:
    - Fixed-size, non-overlapping windows
    - Use: Regular aggregations (hourly counts)
    - Memory: O(keys) (only current window)

    Session:
    - Dynamic windows based on inactivity gap
    - Use: User sessions, clickstream analysis
    - Memory: O(active_sessions) (varies over time)

    Example:
    - Tumbling: Count events every 5 minutes
    - Session: Group user clicks with 30-min timeout
    ```

    ### Q6: How do you scale Kafka and Flink independently?
    ```
    Answer:

    Scale Kafka:
    - Add brokers: Rebalance partition leaders
    - Add partitions: Create new topic, migrate data
    - Increase replication: More fault tolerance

    Scale Flink:
    - Increase parallelism: Redeploy job with higher parallelism
    - Add task managers: More slots for tasks
    - Rescale state: Flink redistributes state automatically

    Independence:
    - Kafka stores data (durable)
    - Flink processes data (stateless to Kafka)
    - Can restart Flink without losing data
    ```

    ### Q7: What happens during Flink failure?
    ```
    Answer:

    Failure scenarios:

    1. Task Manager crash:
       - Job Manager detects failure (heartbeat timeout)
       - Trigger job restart from last checkpoint
       - All operators restore state from S3
       - Kafka offsets reset to checkpoint
       - Resume processing (no data loss)

    2. Job Manager crash:
       - ZooKeeper/Kubernetes detects failure
       - Start new Job Manager (HA setup)
       - Load checkpoint metadata from ZooKeeper
       - Reconnect to Task Managers
       - Resume job (leadership transfer)

    3. Kafka broker crash:
       - Kafka elects new partition leader
       - Flink consumer reconnects to new leader
       - Continue fetching (no data loss due to replication)

    Recovery time: 2-5 minutes (depends on state size)
    ```

    ### Q8: How do you join a stream with a slowly changing dimension table?
    ```
    Answer:

    Option 1: Broadcast State (Small table < 10 GB)
    ```java
    // Broadcast dimension table to all tasks
    MapStateDescriptor<String, Product> productState = ...;
    BroadcastStream<Product> products = productStream.broadcast(productState);

    // Join clicks with products
    clicks.connect(products)
          .process(new BroadcastJoinFunction());
    ```

    Option 2: Temporal Table Join (Versioned table)
    ```java
    // Create temporal table with update time
    TemporalTableFunction productCatalog =
        productUpdates.createTemporalTableFunction("updateTime", "productId");

    // Join: Lookup product as of click time
    clicks.joinLateral(productCatalog(click.timestamp), ...);
    ```

    Option 3: Async I/O (External DB)
    ```java
    // Non-blocking lookup from database
    AsyncDataStream.unorderedWait(
        clicks,
        new AsyncDatabaseLookup(),
        5000, TimeUnit.MILLISECONDS, 100
    );
    ```

    ---

    ## Real-World Examples

    **Uber (Real-time Pricing):**
    ```
    Pipeline:
    1. Ingest: Driver locations (GPS) + rider requests → Kafka
    2. Process: Calculate surge pricing based on supply/demand
    3. Window: 1-minute tumbling windows per geohash
    4. State: Driver availability map, ride request queue
    5. Output: Pricing multipliers to rider app (< 100ms)

    Scale:
    - 10M location updates/sec
    - 1M ride requests/sec
    - 1,000 cities worldwide
    - 10 TB state (driver/rider positions)
    ```

    **Netflix (Real-time Recommendations):**
    ```
    Pipeline:
    1. Ingest: User viewing events (play, pause, stop) → Kafka
    2. Process: Update user profile, trending titles
    3. Window: Session windows (viewing sessions)
    4. State: User preferences, title popularity scores
    5. Output: Recommendations API, A/B testing metrics

    Scale:
    - 5M events/sec (200M users × 5 hours/day / 86,400)
    - 500 TB state (user profiles)
    - < 50ms recommendation latency
    ```

    **LinkedIn (Real-time Analytics):**
    ```
    Pipeline:
    1. Ingest: Profile views, messages, job applications → Kafka
    2. Process: Aggregate metrics (impressions, clicks, conversions)
    3. Window: 5-minute tumbling windows
    4. State: Aggregation counters, top-K trending jobs
    5. Output: Dashboards (Tableau), alerts (PagerDuty)

    Scale:
    - 2M events/sec
    - 100 TB state (aggregations)
    - 1-minute dashboard latency
    ```

    ---

    ## Key Takeaways

    1. **Event time > processing time:** Always use event timestamps
    2. **Watermarks are critical:** Handle late data properly
    3. **State management:** RocksDB for large state, checkpointing for fault tolerance
    4. **Exactly-once semantics:** Checkpointing + transactional sinks
    5. **Windowing:** Choose right window type (tumbling/sliding/session)
    6. **Back-pressure:** Monitor and handle gracefully
    7. **Scalability:** Partition (Kafka) + parallelism (Flink)
    8. **Monitoring:** Lag, latency, checkpoint health

    ---

    ## Additional Resources

    - [Apache Flink Documentation](https://flink.apache.org/)
    - [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
    - [Stream Processing with Apache Flink (Book)](https://www.oreilly.com/library/view/stream-processing-with/9781491974285/)
    - [Designing Data-Intensive Applications (Chapter 11: Stream Processing)](https://dataintensive.net/)
    - [Google Dataflow/Beam Model](https://beam.apache.org/documentation/programming-guide/)
    - [Uber's Real-time Data Infrastructure](https://eng.uber.com/real-time-data-infrastructure/)
    - [Netflix's Keystone Real-time Stream Processing](https://netflixtechblog.com/keystone-real-time-stream-processing-platform-a3ee651812a)
