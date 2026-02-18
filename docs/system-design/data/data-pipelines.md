# Data Processing Pipelines

Every modern system generates far more data than any single machine can process. User clicks, transactions, sensor readings, log events — they all need to be collected, transformed, and delivered to the right place at the right time. Data pipelines are the plumbing that makes this happen: they move data from where it's produced to where it's consumed, applying transformations along the way.

The fundamental question in pipeline design is **when** you process data. Process it in large, scheduled chunks and you get **batch processing** — high throughput, high latency. Process it as it arrives and you get **stream processing** — lower throughput per event, but near-real-time results. Most production systems use both.

```
                        Data Processing Spectrum

  Batch Processing                                    Stream Processing
  ◄──────────────────────────────────────────────────────────────────►

  ┌─────────────────────┐                    ┌─────────────────────────┐
  │  Collect all data   │                    │  Process each event     │
  │  Process in bulk    │                    │  as it arrives          │
  │  Results in hours   │                    │  Results in seconds     │
  │                     │                    │                         │
  │  ETL jobs           │                    │  Real-time dashboards   │
  │  Daily reports      │                    │  Fraud detection        │
  │  ML model training  │                    │  Surge pricing          │
  └─────────────────────┘                    └─────────────────────────┘
         │                                              │
         │            ┌──────────────────┐              │
         └───────────>│  Hybrid / Micro  │<─────────────┘
                      │  batch (seconds) │
                      │  Spark Streaming │
                      └──────────────────┘
```

---

## Batch Processing

Batch processing is the oldest and most straightforward model: accumulate data over a period (an hour, a day, a week), then process the entire collection at once. The key insight is that by deferring processing, you can optimize for throughput — reading and writing data sequentially is orders of magnitude faster than random access.

### MapReduce: The Foundation

MapReduce, introduced by Google in 2004, established the pattern that all modern batch frameworks build on. The idea is simple: break a large computation into two phases that can run in parallel across hundreds of machines.

```
Input Data (distributed across machines)
─────────────────────────────────────────

  File 1            File 2            File 3
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ "hello"  │      │ "world"  │      │ "hello"  │
  │ "world"  │      │ "hello"  │      │ "hello"  │
  └─────┬────┘      └─────┬────┘      └─────┬────┘
        │                  │                  │
        v                  v                  v
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │  MAP 1  │       │  MAP 2  │       │  MAP 3  │
   │(hello,1)│       │(world,1)│       │(hello,1)│
   │(world,1)│       │(hello,1)│       │(hello,1)│
   └─────┬───┘       └────┬────┘       └────┬────┘
         │                 │                  │
         └────────┬────────┴──────────────────┘
                  │
            SHUFFLE & SORT
         (group by key across network)
                  │
         ┌────────┴────────┐
         v                  v
   ┌───────────┐     ┌───────────┐
   │ REDUCE 1  │     │ REDUCE 2  │
   │ "hello"   │     │ "world"   │
   │ [1,1,1,1] │     │ [1,1]     │
   │  => 4     │     │  => 2     │
   └───────────┘     └───────────┘
```

**Map phase:** Each machine processes its local chunk of data independently, emitting key-value pairs. No coordination needed — this is what makes it massively parallel.

**Shuffle phase:** The framework redistributes data across the network so all values for the same key end up on the same machine. This is the most expensive phase — it's a distributed sort.

**Reduce phase:** Each machine aggregates the values for its assigned keys, producing the final output.

!!! tip "Why MapReduce changed everything"
    Before MapReduce, processing a petabyte of data required writing custom distributed programs. MapReduce let developers express their logic as simple map and reduce functions while the framework handled parallelism, fault tolerance, and data distribution automatically.

### When to Use Batch Processing

| Use Case | Why Batch Works | Example |
|----------|----------------|---------|
| **ETL pipelines** | Data extracted from sources doesn't change during processing | Nightly load from OLTP to data warehouse |
| **Reporting & analytics** | Humans consume reports on daily/weekly cadence | Daily revenue dashboards, monthly churn reports |
| **ML model training** | Training needs the full dataset, not incremental updates | Retraining recommendation models on yesterday's clicks |
| **Data backfills** | Reprocessing historical data doesn't need real-time | Recomputing user scores after algorithm change |
| **Large aggregations** | Counting, summing, joining over billions of rows | Computing annual tax summaries for all customers |

### Batch Technology Landscape

**Hadoop MapReduce** was the first widely adopted framework, but its disk-heavy approach (writing intermediate results to HDFS between stages) made multi-step pipelines painfully slow.

**Apache Spark** replaced MapReduce for most workloads by keeping intermediate data in memory. A pipeline that took 30 minutes on Hadoop often completes in 2-3 minutes on Spark. Spark also introduced a richer API — instead of just map and reduce, you get joins, aggregations, window functions, and ML libraries built in.

```
Performance comparison (illustrative):

Hadoop MapReduce:   Disk ──► Map ──► Disk ──► Reduce ──► Disk ──► Map ──► Disk
                    (every stage reads/writes disk)

Spark:              Disk ──► Map ──► Reduce ──► Map ──► Disk
                          └── in memory ──┘└── in memory ──┘
                    (only reads input, writes final output)
```

!!! note "Spark doesn't replace Hadoop entirely"
    Spark replaces the MapReduce compute engine, but most Spark deployments still use HDFS (Hadoop's distributed file system) or cloud object storage (S3, GCS) for persistent data. Hadoop the ecosystem is alive and well; Hadoop MapReduce the compute engine is largely obsolete.

---

## Stream Processing

Stream processing flips the batch model: instead of waiting for data to accumulate, you process each event the moment it arrives. The trade-off is lower throughput per event but dramatically lower latency — from hours to milliseconds.

### Processing Models

There are two fundamental approaches to stream processing, and the distinction matters for both performance and correctness.

**Event-at-a-time** processing handles each event individually as it arrives. This gives the lowest latency (single-digit milliseconds) but requires the framework to maintain state carefully across events.

**Micro-batch** processing collects events into tiny batches (typically 0.5-2 seconds) and processes each mini-batch as a small batch job. Latency is slightly higher but the programming model is simpler and throughput is better.

```
Event-at-a-time (Flink, Kafka Streams):

  event ──► process ──► output
  event ──► process ──► output         Latency: ~1-10ms per event
  event ──► process ──► output         Each event handled individually
  event ──► process ──► output

Micro-batch (Spark Streaming):

  event ─┐
  event ──┤  batch    process    output
  event ──┤  (0.5s) ────────► ────────►   Latency: 500ms-2s
  event ─┘                                Events grouped into mini-batches
  event ─┐
  event ──┤  batch    process    output
  event ──┤  (0.5s) ────────► ────────►
  event ─┘
```

!!! info "Which model to choose?"
    If you need sub-second latency (fraud detection, real-time bidding), use event-at-a-time. If second-level latency is acceptable (dashboards, alerting), micro-batch is simpler to reason about and often easier to operate.

### Windowing: Grouping Unbounded Data

Batch processing has a natural boundary — the file or time range you're processing. Streams are unbounded: data flows forever. **Windowing** solves this by slicing the infinite stream into finite chunks for aggregation.

#### Tumbling Windows

Fixed-size, non-overlapping. Every event belongs to exactly one window.

```
Timeline:  ──────────────────────────────────────────────►
Events:     A  B    C  D  E    F    G  H    I  J

            ┌──────────┐┌──────────┐┌──────────┐
Window 1:   │ A  B     ││          ││          │
Window 2:   │          ││ C  D  E  ││          │
Window 3:   │          ││          ││ F  G  H  │
            └──────────┘└──────────┘└──────────┘
            0s       5s  5s      10s 10s     15s

Use case: "Count page views every 5 minutes"
Each 5-minute period is independent.
```

#### Sliding Windows

Fixed-size, overlapping. A single event can belong to multiple windows. Defined by a window **size** and a **slide** interval.

```
Timeline:  ──────────────────────────────────────────────►
Events:     A  B    C  D  E    F    G  H

Window size: 10s, slide: 5s

  ┌────────────────────┐
  │  A  B    C  D  E   │                    Window 1: 0-10s
  └────────────────────┘
       ┌────────────────────┐
       │  C  D  E    F    G │               Window 2: 5-15s
       └────────────────────┘
            ┌────────────────────┐
            │  F    G  H        │           Window 3: 10-20s
            └────────────────────┘

Use case: "Average CPU over the last 10 minutes, updated every 5 minutes"
Events C, D, E appear in both Window 1 and Window 2.
```

#### Session Windows

Dynamic size, non-overlapping. Defined by a **gap** — if no events arrive within the gap, the window closes. Perfect for user-activity modeling.

```
Timeline:  ──────────────────────────────────────────────────────►
Events:     A B C          D E              F      G H I

Gap timeout: 3 seconds

  ┌─────────┐    ┌──────┐         ┌────┐  ┌───────────┐
  │ A  B  C │    │ D  E │         │ F  │  │ G  H  I   │
  └─────────┘    └──────┘         └────┘  └───────────┘
   Session 1      Session 2       Sess 3   Session 4
                                  (solo)

Use case: "Measure user session duration on website"
Each burst of activity is one session. Long pauses create new sessions.
```

### The Exactly-Once Challenge

In distributed systems, failures are inevitable. A machine crashes, a network partition occurs, a consumer restarts. The critical question is: **how many times does each event get processed?**

| Delivery Guarantee | Meaning | Trade-off |
|---|---|---|
| **At-most-once** | Events may be lost, never duplicated | Fastest, simplest. Acceptable for metrics where small gaps are OK |
| **At-least-once** | Events never lost, but may be duplicated | Requires downstream deduplication. Most common default |
| **Exactly-once** | Each event processed exactly once | Hardest to achieve. Requires coordination between source, processor, and sink |

True exactly-once processing across distributed systems is extremely difficult. What most frameworks actually provide is **effectively exactly-once** — they combine at-least-once delivery with idempotent writes or transactional commits so the *observable effect* is exactly-once.

!!! warning "Exactly-once is an end-to-end property"
    A framework advertising "exactly-once" only guarantees it within its own boundary. If your pipeline reads from Kafka, processes in Flink, and writes to PostgreSQL, you need exactly-once guarantees at every boundary — Kafka's consumer offsets, Flink's checkpointing, and idempotent/transactional writes to PostgreSQL. A break in any link breaks the guarantee.

### Stream Technology Landscape

| Technology | Model | Latency | Strengths | Weaknesses |
|---|---|---|---|---|
| **Apache Kafka Streams** | Event-at-a-time | ~1-10ms | Lightweight library (no cluster needed), tight Kafka integration | Only works with Kafka as source/sink |
| **Apache Flink** | Event-at-a-time | ~1-10ms | Advanced windowing, exactly-once, savepoints for reprocessing | Operational complexity, steep learning curve |
| **Spark Structured Streaming** | Micro-batch / continuous | 100ms-2s | Unified batch and stream API, mature ecosystem | Higher latency than native stream processors |
| **Apache Samza** | Event-at-a-time | ~1-10ms | Local state management, Kafka-native | Smaller community, LinkedIn-centric |

---

## Architecture Patterns

The tension between batch and stream processing has produced two influential architectural patterns for building data systems that need both historical accuracy and real-time responsiveness.

### Lambda Architecture

Proposed by Nathan Marz (creator of Apache Storm), Lambda runs **two parallel pipelines**: a batch layer for accuracy and a speed layer for freshness. A serving layer merges their outputs.

```
                         ┌────────────────────────────┐
                         │       DATA SOURCES          │
                         │  (events, logs, clicks...)  │
                         └─────────────┬──────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                          v                         v
                ┌──────────────────┐     ┌──────────────────┐
                │   BATCH LAYER    │     │   SPEED LAYER    │
                │                  │     │                  │
                │  All historical  │     │  Only recent     │
                │  data            │     │  data (stream)   │
                │  Hadoop/Spark    │     │  Flink/Storm     │
                │  High latency    │     │  Low latency     │
                │  High accuracy   │     │  Approximate     │
                │                  │     │                  │
                │  Runs every      │     │  Runs            │
                │  few hours       │     │  continuously    │
                └────────┬─────────┘     └────────┬─────────┘
                         │                         │
                         │    ┌──────────────┐     │
                         └───►│   SERVING    │◄────┘
                              │    LAYER     │
                              │              │
                              │ Merges batch │
                              │ + speed views│
                              │              │
                              │ Query:       │
                              │ batch_view + │
                              │ speed_view   │
                              └──────────────┘
```

**How it works:**

1. All incoming data goes to both the batch and speed layers
2. The **batch layer** reprocesses the entire dataset periodically (e.g., every 6 hours), producing highly accurate "batch views"
3. The **speed layer** processes only the data that arrived since the last batch run, producing approximate "real-time views"
4. The **serving layer** answers queries by combining the batch view (accurate but stale) with the speed view (fresh but approximate)
5. Each time the batch layer completes, its results replace the speed layer's approximations for that time period

**Advantage:** The batch layer acts as a self-correcting mechanism. If the speed layer produces slightly inaccurate results (due to approximation or late data), the next batch run corrects them.

**Drawback:** You maintain two separate codebases — one for batch logic, one for stream logic — that must produce compatible results. This is the primary criticism of Lambda.

### Kappa Architecture

Proposed by Jay Kreps (co-creator of Apache Kafka), Kappa simplifies Lambda by eliminating the batch layer entirely. Everything goes through a **single stream processing pipeline**, and reprocessing is handled by replaying the event log.

```
                         ┌────────────────────────────┐
                         │       DATA SOURCES          │
                         │  (events, logs, clicks...)  │
                         └─────────────┬──────────────┘
                                       │
                                       v
                         ┌────────────────────────────┐
                         │      IMMUTABLE EVENT LOG    │
                         │        (Apache Kafka)       │
                         │                             │
                         │  Retains all events for     │
                         │  days / weeks / forever     │
                         └─────────────┬──────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                          v                         v
                ┌──────────────────┐     ┌──────────────────┐
                │  STREAM JOB v2   │     │  STREAM JOB v1   │
                │  (new logic)     │     │  (current)       │
                │                  │     │                  │
                │  Replays full    │     │  Processes live   │
                │  event log from  │     │  events           │
                │  beginning       │     │                  │
                └────────┬─────────┘     └────────┬─────────┘
                         │                         │
                         v                         v
                ┌──────────────┐        ┌──────────────┐
                │  New output  │        │ Current      │
                │  (building)  │        │ output       │
                └──────────────┘        └──────────────┘

                When v2 catches up, swap v2 → current, retire v1.
```

**How it works:**

1. All data flows into an immutable, append-only log (Kafka)
2. A single stream processing job consumes events and maintains the serving views
3. To reprocess (bug fix, algorithm change), deploy a new version of the stream job that replays the entire log from the beginning
4. Once the new job catches up to real-time, switch traffic to it and shut down the old one

**Advantage:** One codebase, one processing paradigm. No batch/stream code divergence.

**Drawback:** Replaying a massive log can be slow and expensive. Not practical if your event log spans years of data.

### Lambda vs Kappa Comparison

| Dimension | Lambda | Kappa |
|---|---|---|
| **Codebases** | Two (batch + stream) | One (stream only) |
| **Reprocessing** | Batch layer reruns automatically | Replay event log with new job |
| **Accuracy** | Batch layer guarantees correctness | Stream must be correct from the start |
| **Complexity** | Higher (two systems to maintain) | Lower (one system) |
| **Late data handling** | Batch layer absorbs late data naturally | Must handle in stream logic |
| **Cost** | Higher (two compute clusters) | Lower (one cluster + log storage) |
| **Best for** | Massive historical datasets + real-time needs | Event-sourced systems, moderate data volumes |
| **Used by** | Twitter (pre-2020), many traditional data teams | LinkedIn, newer Kafka-native architectures |

??? tip "How to choose between Lambda and Kappa"
    **Choose Lambda** when you have petabytes of historical data, complex aggregations that are hard to get right in streaming, or when batch accuracy is non-negotiable (financial reporting).

    **Choose Kappa** when your system is event-sourced, your event log is manageable (weeks to months, not years), and you want operational simplicity with a single codebase.

---

## Key Concepts in Stream Processing

### Backpressure Handling

When a downstream consumer can't keep up with the rate of incoming events, you need **backpressure** — a mechanism to slow down producers rather than dropping data or crashing.

```
Without backpressure:

  Producer           Consumer
  (1000 evt/s)  ──►  (500 evt/s)
                      ┌─────────┐
                      │ Buffer  │ ◄── Grows unbounded
                      │ ▓▓▓▓▓▓ │     until OOM crash
                      │ ▓▓▓▓▓▓ │
                      └─────────┘

With backpressure:

  Producer           Consumer
  (1000 evt/s)       (500 evt/s)
       │                  │
       │   "slow down!"   │
       │◄─────────────────│     Consumer signals it's overwhelmed
       │                  │
  (500 evt/s)  ──►  (500 evt/s)  Producer throttles to match
```

Common backpressure strategies:

- **Buffer and spill:** Write overflow to disk (Flink's approach). Handles bursts, but prolonged overload still causes issues.
- **Drop oldest:** Discard the oldest buffered events. Acceptable for metrics, catastrophic for financial data.
- **Reactive pull:** Consumer explicitly requests the next N events (Reactive Streams / Project Reactor pattern). Producer never sends more than requested.
- **Dynamic scaling:** Auto-scale consumers when lag exceeds a threshold (Kafka consumer groups + Kubernetes HPA).

### Watermarks and Late Data

In the real world, events don't arrive in order. A mobile app might buffer events during a subway ride and send them 10 minutes later. **Watermarks** track the progress of event time through the system, telling the processor "you've probably received all events up to time T."

```
Event time vs Processing time:

Processing time ──►
  │
  │    Events arrive out of order:
  │
  │    ○ Event at t=10:03 arrives at 10:03:01  (on time)
  │    ○ Event at t=10:01 arrives at 10:03:05  (2 min late!)
  │    ○ Event at t=10:04 arrives at 10:04:02  (on time)
  │    ○ Event at t=10:02 arrives at 10:06:00  (4 min late!)
  │
  │    Watermark says: "I believe all events before t=10:03
  │    have arrived." Events before the watermark are
  │    processed normally. Events after are "late."
  │
  │    Timeline:
  │    ──────────────────────────────────────────────
  │    10:01  10:02  10:03  10:04  10:05
  │                    ▲
  │              Watermark position
  │
  │    Events with t < watermark: processed normally
  │    Events with t > watermark: late arrivals
```

What happens to late data depends on your configuration:

- **Drop it:** Simplest approach. Window results are final once watermark passes. Acceptable when late data is rare and imprecise results are OK.
- **Refire the window:** Recompute and emit an updated result. Downstream systems must handle updates/retractions.
- **Side output:** Route late events to a separate stream for manual review or delayed processing.

### State Management in Streaming

Unlike batch jobs that process static datasets, stream processors must maintain **state** across events: running counters, session data, machine learning model features, deduplication tables.

The challenge is making this state **fault-tolerant** without sacrificing performance. If a node crashes, its in-memory state is lost.

```
Stateful Stream Processing:

  ┌──────────┐     ┌──────────────────┐     ┌──────────┐
  │  Input   │────►│  Stream Processor │────►│  Output  │
  │  Events  │     │                  │     │  Events  │
  └──────────┘     │  ┌────────────┐  │     └──────────┘
                   │  │   STATE    │  │
                   │  │            │  │
                   │  │ count: 47  │  │
                   │  │ sum: 1234  │  │
                   │  │ session: { │  │
                   │  │   user: .. │  │
                   │  │ }          │  │
                   │  └─────┬──────┘  │
                   └────────┼─────────┘
                            │
                    Periodic checkpoint
                            │
                            v
                   ┌──────────────┐
                   │  Checkpoint  │
                   │  Storage     │
                   │  (HDFS, S3)  │
                   └──────────────┘

On failure: restore state from latest checkpoint, replay
events from that checkpoint's offset in the input log.
```

**Flink** manages state internally with RocksDB (embedded key-value store) and takes periodic snapshots to durable storage. Recovery means restoring the snapshot and replaying events from the corresponding Kafka offset.

**Kafka Streams** uses Kafka itself as the state store, backing local RocksDB instances with compacted Kafka topics. If a node fails, the new node rebuilds state by consuming the changelog topic.

### Idempotency and Deduplication

When a system retries (and retries are inevitable), you need to ensure that processing an event twice produces the same result as processing it once. This is **idempotency**.

Common deduplication strategies:

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Event ID tracking** | Store processed event IDs; skip duplicates | Requires a fast lookup store (Redis, RocksDB). Storage grows over time. |
| **Idempotent writes** | Use UPSERT instead of INSERT; set final value, don't increment | Only works for "set" operations. `balance = 100` is idempotent; `balance += 10` is not. |
| **Transactional outbox** | Write output + offset atomically in one transaction | Guarantees exactly-once but couples processing to a transactional store. |
| **Kafka transactions** | Producer writes to output topic + commits consumer offset atomically | Only works within the Kafka ecosystem. |

---

## Real-World Examples

### Netflix: Real-Time Viewing Analytics

**Scale:** 260M+ subscribers, 500+ million hours watched daily, 1.5 trillion events per day

Netflix processes viewing events in real-time to power:

- **Personalized recommendations** that update as you watch
- **"Trending Now"** lists that refresh every few minutes
- **Content performance dashboards** that show viewership within minutes of a title launch
- **A/B test scoring** that evaluates UI experiments in near-real-time

```
Netflix Data Pipeline (simplified):

  ┌──────────┐     ┌──────────┐     ┌──────────────┐
  │ 260M+    │────►│  Kafka   │────►│  Flink Jobs  │──► Real-time
  │ devices  │     │  Cluster │     │  (hundreds)  │    dashboards
  └──────────┘     │  1.5T    │     └──────┬───────┘
                   │  events/ │            │
                   │  day     │     ┌──────▼───────┐
                   └────┬─────┘     │  Spark Batch │──► ML training,
                        │           │  (nightly)   │    deep analytics
                        │           └──────────────┘
                        │
                        └──────────────────────────► S3 data lake
                                                     (long-term storage)
```

Netflix migrated from a pure batch system to a hybrid Lambda-like architecture with Flink for the speed layer and Spark for batch reprocessing. Their key insight: real-time viewing signals dramatically improve recommendation quality compared to daily batch updates.

### Uber: Surge Pricing with Stream Processing

**Scale:** 130M+ active riders, millions of trips per day, pricing recalculated every 30 seconds per geographic zone

Uber's surge pricing requires processing multiple real-time signals simultaneously:

- Current ride requests per zone
- Available drivers per zone
- Historical demand patterns
- Event data (concerts, sports games ending)
- Weather conditions

```
Uber Surge Pricing Pipeline:

  Ride requests ──────┐
  Driver locations ───┤
  Event feeds ────────┤──► Kafka ──► Flink ──┬──► Price Service
  Weather data ───────┤              │       │    (per zone,
  Historical model ───┘              │       │    every 30s)
                                     │       │
                              ┌──────▼─────┐ │
                              │  State:    │ │
                              │  demand[]  │ │
                              │  supply[]  │ │
                              │  by zone   │ │
                              └────────────┘ │
                                             │
                                             └──► Analytics
                                                  (trip patterns,
                                                  model retraining)
```

The system must be highly available — if surge pricing fails, Uber either underprices (losing revenue) or overprices (losing riders). They use Flink's checkpointing to recover from failures within seconds.

### LinkedIn: Unified Streaming with Samza

**Scale:** 1B+ members, 100B+ events per day across activity feeds, messaging, and analytics

LinkedIn built Apache Samza specifically for their needs: a stream processing framework deeply integrated with Kafka that emphasizes local state management.

Key pipelines include:

- **"People You May Know"** — processes connection graph changes in real-time to update recommendations within minutes
- **Feed ranking** — re-ranks content as engagement signals arrive
- **Standardization** — normalizes job titles, company names, and skills from free-text profile updates into structured data

LinkedIn's architecture is essentially Kappa — they use Kafka as the source of truth and process everything through stream jobs. When they need to reprocess, they replay the Kafka log with a new version of the Samza job.

### Spotify: Batch + Stream for Music Recommendations

**Scale:** 600M+ users, 100M+ tracks, billions of listening events daily

Spotify's recommendation system ("Discover Weekly," "Daily Mix") requires both:

- **Batch processing** for heavy ML model training on weeks of listening history (collaborative filtering, matrix factorization)
- **Stream processing** for real-time adjustments — if you just listened to jazz for an hour, your recommendations should shift immediately, not wait until tomorrow's batch run

```
Spotify Recommendation Pipeline:

                        ┌──────────────────┐
  Listening events ────►│  Google Cloud     │
                        │  Pub/Sub          │
                        └───────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                              │
                 v                              v
       ┌─────────────────┐           ┌──────────────────┐
       │  Batch (Spark)  │           │  Stream (Flink)  │
       │                 │           │                  │
       │  Weekly:        │           │  Real-time:      │
       │  - Train models │           │  - Session taste │
       │  - Compute user │           │    profile       │
       │    taste vectors│           │  - Adjust recs   │
       │  - Generate     │           │    based on      │
       │    Discover     │           │    current mood  │
       │    Weekly       │           │                  │
       └────────┬────────┘           └────────┬─────────┘
                │                              │
                └──────────┬───────────────────┘
                           v
                  ┌──────────────────┐
                  │  Recommendation  │
                  │  Service         │
                  │  (merges batch   │
                  │   + real-time)   │
                  └──────────────────┘
```

This is a classic Lambda architecture in practice: the batch layer produces high-quality weekly playlists, while the stream layer ensures recommendations feel responsive to your current listening session.

---

## Decision Framework: Batch vs Stream vs Hybrid

Start with the simplest approach that meets your latency requirements. You can always add complexity later.

```
What is your latency requirement?
│
├── Hours to days are acceptable
│   └── BATCH
│       Simpler, cheaper, easier to debug.
│       Spark or serverless (AWS Glue, BigQuery)
│
├── Seconds to minutes
│   │
│   ├── Do you also need historical reprocessing?
│   │   │
│   │   ├── Yes ──► HYBRID (Lambda or Kappa)
│   │   │          Lambda if data is massive, Kappa if event log is manageable
│   │   │
│   │   └── No ──► STREAM
│   │              Flink or Kafka Streams
│   │
│   └── Is approximate/eventual accuracy OK?
│       │
│       ├── Yes ──► Micro-batch (Spark Streaming)
│       │          Easier to build and operate
│       │
│       └── No ──► Event-at-a-time (Flink)
│                  Lower latency, exactly-once semantics
│
└── Sub-second (milliseconds)
    └── STREAM (event-at-a-time)
        Flink, Kafka Streams, or custom
```

### Key Questions to Ask

| Question | If Yes | If No |
|---|---|---|
| Can users wait hours for results? | Batch is sufficient | Need stream or hybrid |
| Do you need to reprocess historical data regularly? | Consider Lambda or Kappa with log replay | Pure stream is fine |
| Is your team experienced with streaming? | Flink / Kafka Streams | Start with micro-batch (Spark Streaming) |
| Do you need exactly-once guarantees? | Flink with checkpointing + transactional sinks | At-least-once + idempotent writes is simpler |
| Is your data volume > 1 PB? | Lambda (batch for historical, stream for recent) | Kappa (replay the log when needed) |
| Are you already heavily invested in Kafka? | Kafka Streams or Flink with Kafka connectors | Evaluate Spark Streaming for flexibility |

!!! warning "Don't over-engineer"
    Most applications don't need real-time processing. A batch job running every 15 minutes is operationally simpler, cheaper, and easier to debug than a streaming pipeline — and "15 minutes late" is fast enough for 80% of use cases. Start batch, add streaming only when latency requirements demand it.

---

## Summary

| Concept | Batch | Stream |
|---|---|---|
| **Data model** | Bounded (finite dataset) | Unbounded (infinite events) |
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Throughput** | Very high (optimized for bulk) | Moderate per-event |
| **State** | Stateless (each job reads full input) | Stateful (maintains across events) |
| **Failure recovery** | Re-run the job | Checkpoints + log replay |
| **Complexity** | Lower | Higher |
| **Cost** | Lower (on-demand compute) | Higher (always-on compute) |
| **Best for** | ETL, ML training, reports | Alerting, fraud, real-time dashboards |
| **Technologies** | Spark, Hadoop, BigQuery, AWS Glue | Flink, Kafka Streams, Spark Streaming |

The data processing landscape continues to converge: Spark added streaming, Flink added batch, and cloud services increasingly abstract the choice away. The trend is toward **unified APIs** that let you write logic once and execute it as either batch or stream depending on the source — exemplified by Apache Beam's portability model and Flink's unified DataStream API.

Understanding the fundamental trade-offs — latency vs throughput, simplicity vs flexibility, accuracy vs freshness — remains essential regardless of which framework you choose.
