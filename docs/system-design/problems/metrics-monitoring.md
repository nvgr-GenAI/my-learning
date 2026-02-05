# Design Metrics Monitoring System

A scalable, distributed metrics monitoring and alerting platform that collects, stores, aggregates, and visualizes time-series data from thousands of servers, supporting real-time queries, dashboards, and intelligent alerting.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M metrics/sec, billions of data points, <100ms query latency, 99.99% availability |
| **Key Challenges** | Time-series storage, high-cardinality handling, downsampling, query optimization, alerting rules |
| **Core Concepts** | Pull vs push model, time-series database, aggregation, retention policies, cardinality explosion |
| **Companies** | Datadog, Prometheus, Grafana, New Relic, Dynatrace, SignalFx, Splunk, AppDynamics |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Metrics Collection** | Ingest metrics from services (push/pull) | P0 (Must have) |
    | **Time-Series Storage** | Store metrics with timestamp, tags | P0 (Must have) |
    | **Query & Visualization** | Real-time queries, graphs, dashboards | P0 (Must have) |
    | **Aggregation** | Sum, avg, min, max, percentiles | P0 (Must have) |
    | **Alerting** | Rule-based alerts with thresholds | P0 (Must have) |
    | **Downsampling** | Reduce resolution for old data | P1 (Should have) |
    | **Multi-Tenancy** | Isolate metrics per customer/team | P1 (Should have) |
    | **Retention Policies** | Automatic data expiration | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Log aggregation (use ELK/Splunk instead)
    - Distributed tracing (use Jaeger/Zipkin)
    - Application performance monitoring (full APM)
    - Custom scripting/analysis (basic queries only)
    - Real-time streaming analytics (batch processing)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Ingestion Rate** | 10M metrics/sec | Support large-scale infrastructure |
    | **Query Latency** | < 100ms p99 | Real-time dashboard updates |
    | **Availability** | 99.99% uptime | Critical for production monitoring |
    | **Retention** | 15 months raw data | Compliance and trend analysis |
    | **Data Compression** | 10:1 ratio | Reduce storage costs |
    | **Cardinality** | 100M unique series | Support high-cardinality tags |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Monitored infrastructure:
    - Total servers: 10,000 servers
    - Containers/pods: 50,000 containers
    - Metrics per host: 200 metrics (CPU, memory, disk, network)
    - Metrics per container: 50 metrics
    - Collection interval: 15 seconds

    Total metrics ingestion:
    - Host metrics: 10,000 × 200 / 15s = 133K metrics/sec
    - Container metrics: 50,000 × 50 / 15s = 167K metrics/sec
    - Application metrics: 10M custom metrics/sec
    - Total: ~10M metrics/sec (peak: 30M during incidents)

    Query load:
    - Active dashboards: 10,000 dashboards
    - Queries per dashboard: 10 queries
    - Refresh interval: 30 seconds
    - Query QPS: 10,000 × 10 / 30s = 3,333 queries/sec

    Alert evaluations:
    - Active alert rules: 100,000 rules
    - Evaluation interval: 1 minute
    - Alert QPS: 100,000 / 60s = 1,667 evals/sec
    ```

    ### Storage Estimates

    ```
    Per metric data point:
    - Timestamp: 8 bytes (Unix timestamp)
    - Value: 8 bytes (float64)
    - Metric name: 50 bytes (avg)
    - Tags: 200 bytes (avg: host, service, region, etc.)
    - Total per data point: 266 bytes

    Raw storage (15 seconds resolution):
    - Data points per day: 10M metrics/sec × 86,400s = 864B data points
    - Storage per day: 864B × 266 bytes = 230 TB/day
    - With compression (10:1): 23 TB/day

    Retention tiers (multi-tier storage):

    Tier 1: Full resolution (15s) - 7 days
    - Storage: 23 TB/day × 7 = 161 TB
    - Use: Real-time monitoring, recent investigations

    Tier 2: Downsampled (1 min) - 30 days
    - Reduction: 4x fewer points
    - Storage: 23 TB/day × 30 / 4 = 172 TB
    - Use: Week-over-week comparisons, weekly reports

    Tier 3: Downsampled (5 min) - 90 days
    - Reduction: 20x fewer points
    - Storage: 23 TB/day × 90 / 20 = 103 TB
    - Use: Monthly trends, capacity planning

    Tier 4: Downsampled (1 hour) - 365 days
    - Reduction: 240x fewer points
    - Storage: 23 TB/day × 365 / 240 = 35 TB
    - Use: Yearly trends, compliance

    Total storage: 161 + 172 + 103 + 35 = 471 TB

    With replication (3x): 471 TB × 3 = 1.4 PB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (metrics collection):
    - 10M metrics/sec × 266 bytes = 2.66 GB/sec ≈ 21 Gbps
    - Compressed: 2.1 Gbps

    Egress (queries + dashboards):
    - 3,333 queries/sec × 1,000 data points × 16 bytes = 53 MB/sec ≈ 424 Mbps
    - Dashboard renders: 10,000 active × 10 charts × 100 KB / 30s = 333 MB/sec
    - Total egress: ~3 Gbps

    Total bandwidth: 2.1 Gbps (ingress) + 3 Gbps (egress) = 5.1 Gbps
    ```

    ### Server Estimates

    ```
    Ingestion layer (write path):
    - 10M metrics/sec / 100K metrics per node = 100 nodes
    - CPU: 4 cores per node (protocol parsing, validation)
    - Memory: 16 GB per node (buffering, batching)

    Storage layer (TSDB):
    - 471 TB / 4 TB per node = 118 storage nodes
    - CPU: 8 cores per node (compression, indexing)
    - Memory: 64 GB per node (indexes, hot data)
    - Disk: 4 TB SSD per node (fast writes)

    Query layer:
    - 3,333 queries/sec / 100 queries per node = 34 query nodes
    - CPU: 16 cores per node (aggregation, computation)
    - Memory: 128 GB per node (query cache, working set)

    Total servers:
    - Ingestion: 100 nodes
    - Storage: 118 nodes
    - Query: 34 nodes
    - Coordination: 10 nodes (metadata, routing)
    - Total: ~262 nodes
    ```

    ---

    ## Key Assumptions

    1. 10M metrics per second at peak (30M during incidents)
    2. Average metric cardinality: 10 tags per metric
    3. 15-second collection interval (standard)
    4. 95% write, 5% read workload (write-heavy)
    5. 10:1 compression ratio (time-series compression)
    6. 99% of queries access last 24 hours data

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Pull-based collection:** Prometheus-style scraping (vs. push-based)
    2. **Columnar time-series storage:** Efficient compression and queries
    3. **Downsampling & retention:** Multi-tier storage for cost optimization
    4. **Distributed aggregation:** Parallel query execution
    5. **Rule-based alerting:** Continuous evaluation of alert conditions

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Monitored Infrastructure"
            Host1[Host 1<br/>Node Exporter]
            Host2[Host 2<br/>Node Exporter]
            App1[Application 1<br/>Custom Metrics]
            App2[Application 2<br/>Custom Metrics]
            K8s[Kubernetes<br/>cAdvisor]
        end

        subgraph "Metrics Collection Layer"
            Collector1[Metrics Collector 1<br/>Prometheus/Telegraf]
            Collector2[Metrics Collector 2<br/>Prometheus/Telegraf]
            CollectorN[Metrics Collector N<br/>Prometheus/Telegraf]
            ServiceDiscovery[Service Discovery<br/>Consul/K8s API]
        end

        subgraph "Ingestion & Processing"
            Gateway[Ingestion Gateway<br/>Load balancer]
            Validator[Validator<br/>Tag validation<br/>Rate limiting]
            Buffer[Write Buffer<br/>Kafka/Kinesis]
        end

        subgraph "Time-Series Database (TSDB)"
            subgraph "Write Path"
                Ingester1[Ingester 1<br/>Write to WAL]
                Ingester2[Ingester 2<br/>Write to WAL]
                IngesterN[Ingester N<br/>Write to WAL]
            end

            subgraph "Storage Tier"
                Hot[(Hot Storage<br/>SSD<br/>Last 7 days<br/>15s resolution)]
                Warm[(Warm Storage<br/>SSD<br/>30 days<br/>1min resolution)]
                Cold[(Cold Storage<br/>S3/GCS<br/>1 year<br/>1hr resolution)]
            end

            subgraph "Query Path"
                Querier1[Querier 1<br/>PromQL/SQL]
                Querier2[Querier 2<br/>PromQL/SQL]
                QuerierN[Querier N<br/>PromQL/SQL]
                QueryCache[Query Cache<br/>Redis]
            end
        end

        subgraph "Aggregation & Downsampling"
            Compactor1[Compactor 1<br/>Downsample<br/>Compress]
            Compactor2[Compactor 2<br/>Downsample<br/>Compress]
        end

        subgraph "Alerting Engine"
            RuleEvaluator[Rule Evaluator<br/>Alert rules<br/>Thresholds]
            AlertManager[Alert Manager<br/>Deduplication<br/>Routing]
            Notifications[Notifications<br/>PagerDuty/Slack/Email]
        end

        subgraph "Visualization Layer"
            Grafana[Grafana<br/>Dashboards<br/>Charts]
            APIGateway[API Gateway<br/>REST/GraphQL]
        end

        subgraph "Metadata & Coordination"
            MetaStore[(Metadata Store<br/>PostgreSQL<br/>Series index)]
            Coordinator[Coordinator<br/>etcd/ZooKeeper<br/>Cluster state)]
        end

        ServiceDiscovery --> Collector1
        ServiceDiscovery --> Collector2
        Host1 -.->|Pull| Collector1
        Host2 -.->|Pull| Collector1
        App1 -.->|Pull| Collector2
        App2 -.->|Pull| Collector2
        K8s -.->|Pull| CollectorN

        Collector1 --> Gateway
        Collector2 --> Gateway
        CollectorN --> Gateway

        Gateway --> Validator
        Validator --> Buffer

        Buffer --> Ingester1
        Buffer --> Ingester2
        Buffer --> IngesterN

        Ingester1 --> Hot
        Ingester2 --> Hot
        IngesterN --> Hot

        Hot --> Compactor1
        Hot --> Compactor2
        Compactor1 --> Warm
        Compactor2 --> Warm
        Warm --> Compactor1
        Compactor1 --> Cold

        Querier1 --> Hot
        Querier1 --> Warm
        Querier1 --> Cold
        Querier1 --> QueryCache
        Querier2 --> Hot
        QuerierN --> Cold

        Grafana --> APIGateway
        APIGateway --> Querier1
        APIGateway --> Querier2
        APIGateway --> QuerierN

        RuleEvaluator --> Querier1
        RuleEvaluator --> AlertManager
        AlertManager --> Notifications

        Ingester1 --> MetaStore
        Querier1 --> MetaStore
        Compactor1 --> Coordinator
        Ingester1 --> Coordinator

        style Gateway fill:#e1f5ff
        style Hot fill:#ffe1e1
        style Warm fill:#fff4e1
        style Cold fill:#f0f0f0
        style RuleEvaluator fill:#e8f5e9
        style Grafana fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Pull-based Collection** | Targets aware of collectors (service discovery), easier debugging | Push-based (StatsD) - harder to debug, client dependencies |
    | **Write-Ahead Log (WAL)** | Durability, replay on crash, batch writes | Direct writes (data loss on crash), no buffering |
    | **Columnar Storage** | 10x compression, efficient range queries | Row-based (MySQL) - poor compression, slow queries |
    | **Multi-Tier Storage** | 90% cost savings vs. all-SSD | Single-tier SSD (expensive), S3-only (slow queries) |
    | **Query Cache** | 80% cache hit rate, 10x faster queries | No cache (slow repeated queries), cache-aside (complex) |
    | **Downsampling** | Reduce storage 20x, maintain long retention | Keep full resolution (expensive), aggressive deletion (data loss) |
    | **Service Discovery** | Auto-detect targets, no manual config | Static config (manual updates), DNS (stale records) |

    **Key Trade-off:** We chose **eventual consistency** for writes (async replication) to achieve high ingestion rates. Recent writes may be lost during failures, but this is acceptable for monitoring use cases.

    ---

    ## API Design

    ### 1. Metrics Ingestion (Push Model)

    **Request:**
    ```bash
    POST /api/v1/metrics
    Content-Type: application/json

    {
      "metrics": [
        {
          "name": "http_requests_total",
          "type": "counter",
          "value": 1543,
          "timestamp": 1735819200,
          "tags": {
            "host": "web-server-01",
            "service": "api",
            "method": "GET",
            "status": "200",
            "region": "us-east-1"
          }
        },
        {
          "name": "cpu_usage_percent",
          "type": "gauge",
          "value": 67.5,
          "timestamp": 1735819200,
          "tags": {
            "host": "web-server-01",
            "cpu": "cpu0"
          }
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "accepted": 2,
      "rejected": 0,
      "errors": []
    }
    ```

    **Design Notes:**

    - Batch multiple metrics in single request (reduce network overhead)
    - Tags allow high-dimensional querying
    - Timestamp in Unix epoch (seconds or milliseconds)
    - Metric types: counter, gauge, histogram, summary

    ---

    ### 2. Query Metrics (PromQL-style)

    **Request:**
    ```bash
    POST /api/v1/query
    Content-Type: application/json

    {
      "query": "rate(http_requests_total{service='api',status='200'}[5m])",
      "start": "2025-01-01T00:00:00Z",
      "end": "2025-01-01T23:59:59Z",
      "step": "1m"
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "data": {
        "resultType": "matrix",
        "result": [
          {
            "metric": {
              "service": "api",
              "status": "200",
              "host": "web-server-01"
            },
            "values": [
              [1735689600, "125.5"],
              [1735689660, "132.1"],
              [1735689720, "128.7"]
            ]
          }
        ]
      }
    }
    ```

    **Query Functions:**

    - `rate()` - Per-second rate over time window
    - `sum()` - Aggregate across dimensions
    - `avg()`, `min()`, `max()` - Statistical aggregations
    - `histogram_quantile()` - Percentiles (p50, p95, p99)
    - `increase()` - Total increase over time

    ---

    ### 3. Create Alert Rule

    **Request:**
    ```bash
    POST /api/v1/alerts/rules
    Content-Type: application/json

    {
      "name": "HighCPUUsage",
      "query": "avg(cpu_usage_percent{service='api'}) > 80",
      "duration": "5m",
      "severity": "warning",
      "labels": {
        "team": "platform",
        "component": "compute"
      },
      "annotations": {
        "summary": "High CPU usage on {{ $labels.host }}",
        "description": "CPU usage is {{ $value }}% (threshold: 80%)"
      },
      "notifications": ["slack", "pagerduty"]
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "rule_id": "alert-12345",
      "message": "Alert rule created successfully"
    }
    ```

    **Design Notes:**

    - `duration` - How long condition must be true before firing
    - `severity` - Critical, warning, info
    - `annotations` - Templated alert messages
    - `labels` - Metadata for routing and grouping

    ---

    ### 4. Dashboard Query

    **Request:**
    ```bash
    POST /api/v1/dashboard/query
    Content-Type: application/json

    {
      "queries": [
        {
          "id": "cpu",
          "query": "avg(cpu_usage_percent) by (host)",
          "timeRange": "last_1h"
        },
        {
          "id": "memory",
          "query": "avg(memory_usage_bytes / memory_total_bytes * 100) by (host)",
          "timeRange": "last_1h"
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "results": {
        "cpu": {
          "web-server-01": [65.2, 67.3, 69.1],
          "web-server-02": [45.1, 43.8, 46.5]
        },
        "memory": {
          "web-server-01": [78.5, 79.2, 80.1],
          "web-server-02": [62.3, 63.1, 64.0]
        }
      },
      "timestamps": [1735819200, 1735819260, 1735819320]
    }
    ```

    ---

    ## Database Schema

    ### Time-Series Data Model (Prometheus-style)

    **Metric Identifier:**

    ```
    metric_name{label1="value1", label2="value2", ...}

    Example:
    http_requests_total{host="web-01", method="GET", status="200"}
    ```

    **Storage Format (Columnar):**

    ```
    Block format (compressed, immutable):

    ┌───────────────────────────────────────────┐
    │ Block Header                               │
    │ - Block ID (UUID)                          │
    │ - Min time, Max time                       │
    │ - Series count                             │
    │ - Chunk count                              │
    └───────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │ Series Index                               │
    │ - Series ID (hash of metric + labels)     │
    │ - Metric name                              │
    │ - Label key-value pairs                    │
    │ - Chunk offsets                            │
    └───────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │ Chunks (time + value pairs)                │
    │                                            │
    │ Chunk 1: Series A, 2h time range          │
    │   Timestamps: [t1, t2, t3, ...]           │
    │   Values: [v1, v2, v3, ...]               │
    │   Compression: Gorilla/Delta-of-delta     │
    │                                            │
    │ Chunk 2: Series A, next 2h                │
    │   ...                                     │
    └───────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │ Inverted Index (fast label queries)        │
    │                                            │
    │ Label: "host"                              │
    │   "web-01" → [SeriesID1, SeriesID5, ...]  │
    │   "web-02" → [SeriesID2, SeriesID8, ...]  │
    │                                            │
    │ Label: "status"                            │
    │   "200" → [SeriesID1, SeriesID2, ...]     │
    │   "500" → [SeriesID7, SeriesID9, ...]     │
    └───────────────────────────────────────────┘
    ```

    **Why Columnar Storage:**

    - **High compression:** Similar values compress 10x better
    - **Fast scans:** Read only needed columns
    - **Efficient aggregation:** Sum/avg without decompressing all data

    ---

    ### Metadata Store (PostgreSQL)

    **Series Metadata:**

    ```sql
    CREATE TABLE metric_series (
        series_id BIGINT PRIMARY KEY,           -- Hash of metric name + labels
        metric_name VARCHAR(255) NOT NULL,
        labels JSONB NOT NULL,                  -- Key-value pairs
        first_seen TIMESTAMP NOT NULL,
        last_seen TIMESTAMP NOT NULL,
        sample_count BIGINT DEFAULT 0,
        INDEX idx_metric_name (metric_name),
        INDEX idx_labels GIN (labels)           -- GIN index for JSON queries
    );

    -- Example:
    INSERT INTO metric_series VALUES (
        12345678901,
        'http_requests_total',
        '{"host":"web-01","method":"GET","status":"200"}',
        '2025-01-01 00:00:00',
        '2025-01-02 23:59:59',
        86400
    );
    ```

    **Alert Rules:**

    ```sql
    CREATE TABLE alert_rules (
        rule_id UUID PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        query TEXT NOT NULL,                    -- PromQL query
        duration INTERVAL NOT NULL,             -- e.g., '5 minutes'
        severity VARCHAR(50) NOT NULL,          -- critical, warning, info
        labels JSONB,
        annotations JSONB,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_active_rules ON alert_rules(is_active) WHERE is_active = true;
    ```

    **Alert State:**

    ```sql
    CREATE TABLE alert_states (
        alert_id UUID PRIMARY KEY,
        rule_id UUID REFERENCES alert_rules(rule_id),
        state VARCHAR(20) NOT NULL,             -- pending, firing, resolved
        labels JSONB NOT NULL,                  -- Instance-specific labels
        value DOUBLE PRECISION,
        fired_at TIMESTAMP,
        resolved_at TIMESTAMP,
        last_evaluated TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_firing_alerts ON alert_states(state) WHERE state = 'firing';
    ```

    ---

    ## Data Flow Diagrams

    ### Write Path (Metrics Ingestion)

    ```mermaid
    sequenceDiagram
        participant Exporter as Node Exporter
        participant Collector as Prometheus<br/>Collector
        participant Gateway as Ingestion<br/>Gateway
        participant Validator
        participant Kafka
        participant Ingester
        participant WAL
        participant TSDB as Hot Storage<br/>(TSDB)

        Collector->>Exporter: Scrape metrics (HTTP GET)<br/>/metrics endpoint
        Exporter-->>Collector: Metrics (Prometheus format)

        Collector->>Gateway: Batch push (1000 metrics)
        Gateway->>Validator: Validate format, tags

        alt Invalid metrics
            Validator-->>Gateway: Reject with errors
        else Valid metrics
            Validator->>Kafka: Write to Kafka topic
            Kafka-->>Validator: Ack
            Validator-->>Gateway: Success
        end

        Kafka->>Ingester: Consume batch
        Ingester->>WAL: Append to Write-Ahead Log
        Ingester->>Ingester: Buffer in memory<br/>(time-series chunks)

        Note over Ingester: Every 2 hours or<br/>when buffer full

        Ingester->>TSDB: Flush compressed block<br/>to disk
        TSDB-->>Ingester: Block ID
        Ingester->>WAL: Checkpoint (safe to delete)
    ```

    ---

    ### Query Path (Dashboard/API)

    ```mermaid
    sequenceDiagram
        participant Grafana
        participant API as API Gateway
        participant Cache as Query Cache<br/>(Redis)
        participant Querier
        participant Hot as Hot Storage<br/>(7 days)
        participant Warm as Warm Storage<br/>(30 days)
        participant Cold as Cold Storage<br/>(S3)

        Grafana->>API: Query: rate(http_requests[5m])<br/>Time: last 24h
        API->>Cache: Check cache (query hash)

        alt Cache hit (80% of queries)
            Cache-->>API: Cached result
            API-->>Grafana: Return data
        else Cache miss
            API->>Querier: Execute query

            Querier->>Querier: Parse PromQL<br/>Identify time range

            Note over Querier: Query last 24h<br/>(only hit hot storage)

            Querier->>Hot: Query range [now-24h, now]
            Hot-->>Querier: Time-series data

            Querier->>Querier: Aggregate<br/>(rate calculation)
            Querier->>Cache: Store result (TTL: 30s)
            Querier-->>API: Result
            API-->>Grafana: Return data
        end
    ```

    ---

    ### Alert Evaluation Path

    ```mermaid
    sequenceDiagram
        participant Timer as Cron Scheduler<br/>(every 1 min)
        participant Evaluator as Rule Evaluator
        participant Querier
        participant TSDB
        participant AlertMgr as Alert Manager
        participant StateDB as Alert State DB
        participant Notifier as Notifications<br/>(PagerDuty/Slack)

        Timer->>Evaluator: Trigger evaluation cycle

        loop For each active rule
            Evaluator->>Querier: Execute rule query<br/>avg(cpu) > 80
            Querier->>TSDB: Fetch metrics (last 5m)
            TSDB-->>Querier: Data
            Querier-->>Evaluator: Result: 85.2 > 80 ✓

            alt Condition met
                Evaluator->>StateDB: Check alert state

                alt Alert already firing
                    StateDB-->>Evaluator: State: firing (5 mins)
                    Note over Evaluator: Already notified,<br/>update last_seen
                else Alert not firing
                    StateDB-->>Evaluator: State: pending (2 mins)

                    Note over Evaluator: Check duration:<br/>2 min < 5 min required

                    alt Duration met
                        Evaluator->>AlertMgr: Fire alert!
                        AlertMgr->>AlertMgr: Deduplication<br/>Grouping
                        AlertMgr->>Notifier: Send notification
                        Notifier-->>AlertMgr: Sent
                        AlertMgr->>StateDB: Update state: firing
                    else Duration not met
                        Evaluator->>StateDB: Update state: pending
                    end
                end

            else Condition not met
                Evaluator->>StateDB: Check state

                alt Was firing
                    StateDB-->>Evaluator: State: firing
                    Evaluator->>AlertMgr: Resolve alert
                    AlertMgr->>Notifier: Send resolved notification
                    AlertMgr->>StateDB: Update state: resolved
                else Not firing
                    Note over Evaluator: No action needed
                end
            end
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Time-Series Database (TSDB) Internals

    ### Pull vs. Push Collection Models

    **Pull Model (Prometheus):**

    ```python
    class PrometheusCollector:
        """
        Pull-based metrics collection (Prometheus style)

        Advantages:
        - Centralized configuration (collectors know targets)
        - Easy debugging (can manually curl /metrics)
        - Failed scrapes visible (monitoring the monitor)
        - No client dependencies (standard HTTP)

        Disadvantages:
        - Firewall complexity (collectors must reach targets)
        - Service discovery required (find new targets)
        """

        def __init__(self, service_discovery):
            self.targets = []
            self.service_discovery = service_discovery
            self.scrape_interval = 15  # seconds

        def discover_targets(self):
            """Get list of targets from service discovery"""
            # Query Consul, Kubernetes API, DNS, etc.
            targets = self.service_discovery.list_services()

            # Example: [
            #   {"host": "web-01.prod.com", "port": 9100, "labels": {"env": "prod"}},
            #   {"host": "web-02.prod.com", "port": 9100, "labels": {"env": "prod"}},
            # ]

            self.targets = targets

        def scrape_target(self, target):
            """Scrape metrics from a single target"""
            url = f"http://{target['host']}:{target['port']}/metrics"

            try:
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    # Parse Prometheus text format
                    metrics = self.parse_prometheus_format(response.text)

                    # Add target labels
                    for metric in metrics:
                        metric['labels'].update(target['labels'])

                    return metrics
                else:
                    # Record scrape failure
                    self.record_scrape_error(target, response.status_code)
                    return []

            except requests.Timeout:
                self.record_scrape_error(target, 'timeout')
                return []

        def parse_prometheus_format(self, text):
            """
            Parse Prometheus exposition format:

            # HELP http_requests_total Total HTTP requests
            # TYPE http_requests_total counter
            http_requests_total{method="GET",status="200"} 1543 1735819200000
            http_requests_total{method="POST",status="201"} 432 1735819200000
            """
            metrics = []

            for line in text.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue

                # Parse: metric_name{labels} value timestamp
                match = re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*){(.+)} ([\d.]+) (\d+)', line)

                if match:
                    metric_name, labels_str, value, timestamp = match.groups()

                    # Parse labels: key1="value1",key2="value2"
                    labels = dict(re.findall(r'(\w+)="([^"]+)"', labels_str))

                    metrics.append({
                        'name': metric_name,
                        'labels': labels,
                        'value': float(value),
                        'timestamp': int(timestamp) // 1000  # ms to seconds
                    })

            return metrics

        def run(self):
            """Main scrape loop"""
            while True:
                # Refresh targets from service discovery
                self.discover_targets()

                # Scrape all targets in parallel
                with ThreadPoolExecutor(max_workers=100) as executor:
                    results = executor.map(self.scrape_target, self.targets)

                # Flatten and send to ingestion pipeline
                all_metrics = [metric for batch in results for metric in batch]
                self.send_to_ingestion(all_metrics)

                time.sleep(self.scrape_interval)
    ```

    **Push Model (Graphite/StatsD):**

    ```python
    class StatsDCollector:
        """
        Push-based metrics collection (StatsD style)

        Advantages:
        - No service discovery needed (clients push)
        - Works behind firewalls (outbound only)
        - Short-lived processes (batch jobs)

        Disadvantages:
        - Client dependencies (StatsD library)
        - Failed pushes invisible (silent failures)
        - No centralized configuration
        - Network load on clients
        """

        def __init__(self, port=8125):
            self.port = port
            self.buffer = []
            self.flush_interval = 10  # seconds

        def listen(self):
            """Listen for UDP packets"""
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('0.0.0.0', self.port))

            while True:
                data, addr = sock.recvfrom(1024)

                # Parse StatsD format: metric_name:value|type|@sample_rate
                # Example: http_requests:1|c|@0.1
                #          response_time:234|ms
                #          active_users:150|g

                metrics = self.parse_statsd_format(data.decode('utf-8'))
                self.buffer.extend(metrics)

        def parse_statsd_format(self, data):
            """Parse StatsD protocol"""
            metrics = []

            for line in data.split('\n'):
                if not line.strip():
                    continue

                parts = line.split('|')
                if len(parts) < 2:
                    continue

                name_value = parts[0].split(':')
                if len(name_value) != 2:
                    continue

                metric_name, value = name_value
                metric_type = parts[1]  # c=counter, g=gauge, ms=timer, h=histogram
                sample_rate = 1.0

                if len(parts) > 2 and parts[2].startswith('@'):
                    sample_rate = float(parts[2][1:])

                metrics.append({
                    'name': metric_name,
                    'value': float(value),
                    'type': metric_type,
                    'sample_rate': sample_rate,
                    'timestamp': time.time()
                })

            return metrics

        def flush(self):
            """Periodically flush buffer"""
            while True:
                time.sleep(self.flush_interval)

                if self.buffer:
                    # Aggregate counters, calculate stats
                    aggregated = self.aggregate_metrics(self.buffer)
                    self.send_to_ingestion(aggregated)
                    self.buffer.clear()
    ```

    **Recommendation:** Use **pull-based (Prometheus)** for most cases. It's easier to debug, has better visibility, and simplifies client instrumentation.

    ---

    ### Time-Series Compression

    **Gorilla Compression (Facebook/Prometheus):**

    ```python
    class GorillaCompression:
        """
        Gorilla time-series compression algorithm

        Key ideas:
        1. Delta-of-delta encoding for timestamps (predictable intervals)
        2. XOR encoding for values (small changes between adjacent values)
        3. Achieves 10:1 compression for typical metrics

        Reference: Facebook's Gorilla paper (VLDB 2015)
        """

        def compress_timestamps(self, timestamps):
            """
            Delta-of-delta encoding for timestamps

            Example:
            Timestamps: [1000, 1015, 1030, 1045, 1060]
            Deltas:     [  15,   15,   15,   15,   15]  (regular 15s intervals)
            Delta-of-deltas: [0, 0, 0, 0]  (all zeros!)

            Result: Store only first timestamp + interval (15s)
            Compression: 5 × 8 bytes = 40 bytes → 8 + 2 bytes = 10 bytes (4x)
            """
            if not timestamps:
                return []

            compressed = []

            # Store first timestamp (full 64 bits)
            compressed.append(('FULL', timestamps[0]))

            if len(timestamps) < 2:
                return compressed

            # Calculate deltas
            prev_delta = timestamps[1] - timestamps[0]
            compressed.append(('DELTA', prev_delta))

            # Delta-of-deltas for remaining timestamps
            for i in range(2, len(timestamps)):
                delta = timestamps[i] - timestamps[i-1]
                delta_of_delta = delta - prev_delta

                # Encode delta-of-delta with variable bit width
                if delta_of_delta == 0:
                    # Most common case (regular intervals)
                    compressed.append(('DOD_ZERO', 0))  # 1 bit: 0
                elif -63 <= delta_of_delta <= 64:
                    # Small deviation: 7 bits
                    compressed.append(('DOD_SMALL', delta_of_delta))
                elif -255 <= delta_of_delta <= 256:
                    # Medium deviation: 9 bits
                    compressed.append(('DOD_MEDIUM', delta_of_delta))
                else:
                    # Large deviation: full 64 bits
                    compressed.append(('DOD_LARGE', delta_of_delta))

                prev_delta = delta

            return compressed

        def compress_values(self, values):
            """
            XOR encoding for float values

            Key insight: Metric values change slowly, so XOR of adjacent
            values has many leading/trailing zeros

            Example:
            Value 1: 65.2  →  0x4050333333333333 (binary: 0100000001010000...)
            Value 2: 65.3  →  0x40504CCCCCCCCCCD (binary: 0100000001010000...)
            XOR:              0x0000799999999999 (binary: 0000000000000111...)

            Leading zeros: 13 bits (can skip)
            Trailing zeros: 0 bits
            Meaningful bits: 51 bits (only store these)

            Compression: 64 bits → ~20 bits (3x)
            """
            if not values:
                return []

            compressed = []

            # Store first value (full 64 bits)
            compressed.append(('FULL', values[0]))

            prev_value_bits = struct.unpack('>Q', struct.pack('>d', values[0]))[0]
            prev_leading_zeros = 0
            prev_trailing_zeros = 0

            for value in values[1:]:
                # Convert float to 64-bit integer representation
                value_bits = struct.unpack('>Q', struct.pack('>d', value))[0]

                # XOR with previous value
                xor_result = value_bits ^ prev_value_bits

                if xor_result == 0:
                    # Value unchanged (common for gauges)
                    compressed.append(('SAME', 0))  # 1 bit: 0
                else:
                    # Count leading and trailing zeros
                    leading_zeros = self._count_leading_zeros(xor_result)
                    trailing_zeros = self._count_trailing_zeros(xor_result)

                    if (leading_zeros >= prev_leading_zeros and
                        trailing_zeros >= prev_trailing_zeros):
                        # Use previous control bits (2 bits: 10)
                        meaningful_bits = 64 - prev_leading_zeros - prev_trailing_zeros
                        compressed.append(('CONTROL_PREV',
                            (xor_result >> prev_trailing_zeros) & ((1 << meaningful_bits) - 1)))
                    else:
                        # Store new control bits (2 bits: 11 + 5 bits leading + 6 bits meaningful)
                        meaningful_bits = 64 - leading_zeros - trailing_zeros
                        compressed.append(('CONTROL_NEW', {
                            'leading_zeros': leading_zeros,
                            'meaningful_bits': meaningful_bits,
                            'value': (xor_result >> trailing_zeros) & ((1 << meaningful_bits) - 1)
                        }))

                        prev_leading_zeros = leading_zeros
                        prev_trailing_zeros = trailing_zeros

                prev_value_bits = value_bits

            return compressed

        def _count_leading_zeros(self, n):
            """Count leading zero bits in 64-bit integer"""
            if n == 0:
                return 64
            return 63 - n.bit_length()

        def _count_trailing_zeros(self, n):
            """Count trailing zero bits"""
            if n == 0:
                return 64
            count = 0
            while (n & 1) == 0:
                count += 1
                n >>= 1
            return count

    # Compression Results (typical metrics):
    # - Timestamps: 8 bytes → 1-2 bits per sample (50x compression)
    # - Values: 8 bytes → 8-16 bits per sample (4-8x compression)
    # - Overall: 16 bytes per sample → 1-3 bytes (10x compression)
    ```

    ---

    ### Downsampling & Retention

    ```python
    class Downsampler:
        """
        Downsample high-resolution data to save storage

        Strategy:
        - Keep full resolution (15s) for recent data (7 days)
        - Downsample to 1 minute for older data (30 days)
        - Downsample to 5 minutes for archive (90 days)
        - Downsample to 1 hour for long-term (1 year)

        Storage savings: 90% reduction vs. keeping all raw data
        """

        def downsample_block(self, block, target_resolution):
            """
            Downsample a time-series block to lower resolution

            Args:
                block: Raw time-series data (15s resolution)
                target_resolution: Target interval (60s, 300s, 3600s)

            Returns:
                Downsampled block with aggregated values
            """
            downsampled_series = []

            for series in block.series:
                # Group samples into time windows
                windows = self._group_by_time_window(
                    series.samples,
                    target_resolution
                )

                # Aggregate each window
                aggregated_samples = []

                for window_start, samples in windows.items():
                    if not samples:
                        continue

                    # Calculate aggregations
                    values = [s.value for s in samples]

                    aggregated = {
                        'timestamp': window_start,
                        'count': len(values),
                        'sum': sum(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'first': values[0],
                        'last': values[-1]
                    }

                    # For counters: use last value (cumulative)
                    # For gauges: use average value
                    # For histograms: merge buckets

                    if series.type == 'counter':
                        final_value = aggregated['last']
                    elif series.type == 'gauge':
                        final_value = aggregated['avg']
                    else:
                        final_value = aggregated['sum']

                    aggregated_samples.append({
                        'timestamp': window_start,
                        'value': final_value,
                        'aggregations': aggregated  # Keep for rollups
                    })

                downsampled_series.append({
                    'series_id': series.id,
                    'samples': aggregated_samples,
                    'resolution': target_resolution
                })

            return downsampled_series

        def _group_by_time_window(self, samples, window_size):
            """Group samples into time windows"""
            windows = {}

            for sample in samples:
                # Calculate window start
                window_start = (sample.timestamp // window_size) * window_size

                if window_start not in windows:
                    windows[window_start] = []

                windows[window_start].append(sample)

            return windows

        def run_compaction_job(self):
            """
            Background job to compact and downsample blocks

            Runs hourly to:
            1. Merge small blocks into larger blocks
            2. Downsample old blocks
            3. Delete expired data
            4. Upload cold blocks to S3
            """
            current_time = time.time()

            # Get all blocks
            blocks = self.list_blocks()

            for block in blocks:
                block_age_days = (current_time - block.max_time) / 86400

                if block_age_days > 365:
                    # Delete blocks older than 1 year
                    self.delete_block(block.id)

                elif block_age_days > 90:
                    # Move to cold storage (S3), downsample to 1 hour
                    if block.resolution < 3600:
                        downsampled = self.downsample_block(block, 3600)
                        self.write_block(downsampled, 'cold')
                        self.delete_block(block.id)

                elif block_age_days > 30:
                    # Warm storage, downsample to 5 minutes
                    if block.resolution < 300:
                        downsampled = self.downsample_block(block, 300)
                        self.write_block(downsampled, 'warm')
                        self.delete_block(block.id)

                elif block_age_days > 7:
                    # Still in hot storage, but downsample to 1 minute
                    if block.resolution < 60:
                        downsampled = self.downsample_block(block, 60)
                        self.write_block(downsampled, 'hot')
                        self.delete_block(block.id)
    ```

    ---

    ### Query Optimization

    ```python
    class QueryOptimizer:
        """
        Optimize time-series queries for performance

        Key optimizations:
        1. Query cache (Redis) - 80% hit rate
        2. Query rewriting (push down filters)
        3. Parallel execution (fan-out to shards)
        4. Result streaming (don't buffer all data)
        """

        def execute_query(self, query_str, start_time, end_time):
            """
            Execute PromQL query with optimizations

            Example query:
            rate(http_requests_total{service="api",status="200"}[5m])
            """
            # 1. Check query cache
            cache_key = self._generate_cache_key(query_str, start_time, end_time)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                return cached_result

            # 2. Parse query
            query_ast = self.parse_promql(query_str)

            # 3. Optimize query plan
            optimized_plan = self.optimize_query_plan(query_ast)

            # 4. Determine which storage tiers to query
            storage_tiers = self.determine_storage_tiers(start_time, end_time)

            # 5. Fan out to shards (parallel execution)
            shard_queries = self.partition_query_by_shards(optimized_plan)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []

                for shard_id, shard_query in shard_queries.items():
                    future = executor.submit(
                        self.execute_shard_query,
                        shard_id,
                        shard_query,
                        storage_tiers
                    )
                    futures.append(future)

                # 6. Collect results from all shards
                shard_results = [f.result() for f in futures]

            # 7. Merge and aggregate results
            final_result = self.merge_results(shard_results, optimized_plan)

            # 8. Cache result (TTL: 30s for recent queries, 5min for old queries)
            cache_ttl = 30 if end_time > time.time() - 3600 else 300
            self.cache.set(cache_key, final_result, ttl=cache_ttl)

            return final_result

        def optimize_query_plan(self, query_ast):
            """
            Optimize query execution plan

            Optimizations:
            1. Push down filters (label selectors) to storage layer
            2. Eliminate redundant aggregations
            3. Reorder operations (filters before aggregations)
            4. Use pre-aggregated data when possible
            """
            optimized = query_ast

            # Push filters down to storage
            optimized = self._push_down_filters(optimized)

            # Use downsampled data for long time ranges
            time_range_hours = (query_ast.end_time - query_ast.start_time) / 3600

            if time_range_hours > 24:
                # Use 1-minute resolution (4x faster)
                optimized.target_resolution = 60
            elif time_range_hours > 168:  # 1 week
                # Use 5-minute resolution (20x faster)
                optimized.target_resolution = 300
            else:
                # Use full resolution (15s)
                optimized.target_resolution = 15

            return optimized

        def _push_down_filters(self, query_ast):
            """
            Push label filters down to storage layer

            Example:
            Before: rate(http_requests[5m]) AND service="api"
            After:  rate(http_requests{service="api"}[5m])

            This reduces data scanned by 10-100x
            """
            # Extract label matchers from query
            label_filters = query_ast.label_matchers

            # Apply filters at storage layer (inverted index lookup)
            query_ast.storage_filters = label_filters

            return query_ast
    ```

    ---

    ## 3.2 High Cardinality Problem

    **Problem:** Too many unique time series = memory explosion

    ```python
    class CardinalityManager:
        """
        Manage high cardinality metrics

        Problem Example:
        Metric: http_requests_total
        Labels: {user_id="12345", session_id="abc123", request_id="xyz789"}

        If 1M active users, cardinality = 1M unique series
        Each series: 1 KB metadata = 1 GB memory just for metadata!

        Solutions:
        1. Drop high-cardinality labels
        2. Use sampling
        3. Aggregate at collection time
        4. Use separate system for high-cardinality (exemplars)
        """

        def validate_cardinality(self, metric):
            """
            Reject metrics with dangerous high cardinality

            Rules:
            1. No user IDs in labels (use histogram instead)
            2. No session IDs in labels
            3. No UUIDs in labels
            4. Limited set of label values (< 10K per label)
            """
            high_cardinality_patterns = [
                r'user_id',
                r'session_id',
                r'request_id',
                r'trace_id',
                r'uuid',
                r'guid'
            ]

            for label_key in metric.labels.keys():
                for pattern in high_cardinality_patterns:
                    if re.search(pattern, label_key, re.IGNORECASE):
                        return False, f"High cardinality label detected: {label_key}"

            # Check label value cardinality
            for label_key, label_value in metric.labels.items():
                cardinality = self.get_label_cardinality(metric.name, label_key)

                if cardinality > 10000:
                    return False, f"Label {label_key} has too many values: {cardinality}"

            return True, None

        def get_label_cardinality(self, metric_name, label_key):
            """Get number of unique values for a label"""
            query = """
                SELECT COUNT(DISTINCT labels->>%s) as cardinality
                FROM metric_series
                WHERE metric_name = %s
            """
            result = self.db.execute(query, [label_key, metric_name])
            return result[0]['cardinality']

        def monitor_cardinality(self):
            """
            Monitor and alert on cardinality growth

            Alert thresholds:
            - Per metric: > 10K unique series
            - Per label: > 1K unique values
            - Total: > 10M unique series
            """
            # Total active series
            total_series = self.db.execute(
                "SELECT COUNT(*) FROM metric_series WHERE last_seen > NOW() - INTERVAL '1 hour'"
            )[0]['count']

            if total_series > 10_000_000:
                self.alert("High cardinality: {} total series".format(total_series))

            # Top cardinality metrics
            top_metrics = self.db.execute("""
                SELECT metric_name, COUNT(*) as series_count
                FROM metric_series
                WHERE last_seen > NOW() - INTERVAL '1 hour'
                GROUP BY metric_name
                ORDER BY series_count DESC
                LIMIT 10
            """)

            for metric in top_metrics:
                if metric['series_count'] > 100_000:
                    self.alert(f"High cardinality metric: {metric['metric_name']} " +
                              f"has {metric['series_count']} series")
    ```

    ---

    ## 3.3 Alerting Engine

    ```python
    class AlertEvaluator:
        """
        Evaluate alert rules and fire notifications

        Architecture:
        1. Load active rules from database
        2. Execute each rule query (PromQL)
        3. Check if condition met for duration
        4. Fire alert or resolve alert
        5. Send to Alert Manager for deduplication
        """

        def __init__(self, querier, alert_manager):
            self.querier = querier
            self.alert_manager = alert_manager
            self.rule_cache = {}
            self.evaluation_interval = 60  # seconds

        def run(self):
            """Main evaluation loop"""
            while True:
                # Load active rules (cached for 5 minutes)
                rules = self.load_active_rules()

                # Evaluate all rules in parallel
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = [
                        executor.submit(self.evaluate_rule, rule)
                        for rule in rules
                    ]

                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Rule evaluation failed: {e}")

                time.sleep(self.evaluation_interval)

        def evaluate_rule(self, rule):
            """
            Evaluate a single alert rule

            Example rule:
            {
                "name": "HighCPU",
                "query": "avg(cpu_usage) > 80",
                "duration": "5m",
                "severity": "warning"
            }
            """
            # Execute rule query
            try:
                result = self.querier.execute_query(
                    rule.query,
                    start_time=time.time() - rule.lookback_window,
                    end_time=time.time()
                )
            except Exception as e:
                logger.error(f"Query failed for rule {rule.name}: {e}")
                return

            # Check if condition met for each series
            for series in result.series:
                alert_key = self._generate_alert_key(rule.id, series.labels)

                # Get current alert state
                alert_state = self.get_alert_state(alert_key)

                if self._condition_met(result, series):
                    # Condition is true
                    self._handle_firing_condition(rule, series, alert_state)
                else:
                    # Condition is false
                    self._handle_resolved_condition(rule, series, alert_state)

        def _handle_firing_condition(self, rule, series, alert_state):
            """Handle case where alert condition is true"""
            current_time = time.time()

            if alert_state is None:
                # New alert - create pending state
                self.create_alert_state(
                    rule.id,
                    series.labels,
                    state='pending',
                    started_at=current_time,
                    value=series.value
                )
                logger.info(f"Alert {rule.name} pending for {series.labels}")

            elif alert_state.state == 'pending':
                # Check if duration threshold met
                time_pending = current_time - alert_state.started_at

                if time_pending >= rule.duration_seconds:
                    # Duration met - fire alert!
                    self.update_alert_state(
                        alert_state.id,
                        state='firing',
                        fired_at=current_time,
                        value=series.value
                    )

                    # Send to Alert Manager
                    self.alert_manager.fire_alert({
                        'rule_name': rule.name,
                        'severity': rule.severity,
                        'labels': series.labels,
                        'annotations': self._render_annotations(rule, series),
                        'value': series.value,
                        'fired_at': current_time
                    })

                    logger.warning(f"Alert {rule.name} FIRING for {series.labels}")

            elif alert_state.state == 'firing':
                # Alert already firing - update value
                self.update_alert_state(
                    alert_state.id,
                    value=series.value,
                    last_evaluated=current_time
                )

        def _handle_resolved_condition(self, rule, series, alert_state):
            """Handle case where alert condition is false"""
            if alert_state and alert_state.state == 'firing':
                # Alert was firing, now resolved
                self.update_alert_state(
                    alert_state.id,
                    state='resolved',
                    resolved_at=time.time()
                )

                # Send resolved notification
                self.alert_manager.resolve_alert({
                    'rule_name': rule.name,
                    'labels': series.labels,
                    'resolved_at': time.time()
                })

                logger.info(f"Alert {rule.name} RESOLVED for {series.labels}")

            elif alert_state and alert_state.state == 'pending':
                # Alert was pending, but condition no longer met
                self.delete_alert_state(alert_state.id)

        def _render_annotations(self, rule, series):
            """
            Render templated annotations with actual values

            Template: "CPU usage on {{ $labels.host }} is {{ $value }}%"
            Result: "CPU usage on web-01 is 85.2%"
            """
            annotations = {}

            for key, template in rule.annotations.items():
                # Replace {{ $labels.key }} with actual label values
                rendered = template

                for label_key, label_value in series.labels.items():
                    rendered = rendered.replace(
                        f"{{{{ $labels.{label_key} }}}}",
                        str(label_value)
                    )

                # Replace {{ $value }} with metric value
                rendered = rendered.replace("{{ $value }}", f"{series.value:.2f}")

                annotations[key] = rendered

            return annotations


    class AlertManager:
        """
        Alert Manager - deduplication, grouping, routing

        Responsibilities:
        1. Deduplicate alerts (same alert firing multiple times)
        2. Group related alerts (batch notifications)
        3. Route to correct channels (Slack, PagerDuty, email)
        4. Implement silences (mute alerts during maintenance)
        """

        def fire_alert(self, alert):
            """Process new firing alert"""
            # Check if alert is silenced
            if self.is_silenced(alert):
                logger.info(f"Alert {alert['rule_name']} is silenced")
                return

            # Deduplicate (check if already firing)
            existing = self.find_existing_alert(alert)

            if existing:
                # Update existing alert timestamp
                self.update_alert_timestamp(existing.id)
                return

            # Group with similar alerts (wait 30s for grouping)
            self.add_to_pending_group(alert)

        def flush_pending_groups(self):
            """Send grouped alerts (runs every 30s)"""
            groups = self.get_pending_groups()

            for group in groups:
                if time.time() - group.created_at > 30:
                    # Group has waited long enough
                    self.send_notification(group)
                    self.mark_group_sent(group.id)

        def send_notification(self, alert_group):
            """Send notification to configured channels"""
            # Determine routing based on labels
            for alert in alert_group.alerts:
                channels = self.get_notification_channels(alert)

                for channel in channels:
                    if channel == 'slack':
                        self.send_slack_notification(alert)
                    elif channel == 'pagerduty':
                        self.send_pagerduty_notification(alert)
                    elif channel == 'email':
                        self.send_email_notification(alert)

        def send_slack_notification(self, alert):
            """Send alert to Slack"""
            message = {
                "text": f"🔥 Alert: {alert['rule_name']}",
                "attachments": [{
                    "color": "danger" if alert['severity'] == 'critical' else "warning",
                    "fields": [
                        {"title": "Severity", "value": alert['severity'], "short": True},
                        {"title": "Value", "value": f"{alert['value']:.2f}", "short": True},
                        {"title": "Labels", "value": str(alert['labels']), "short": False},
                        {"title": "Description", "value": alert['annotations']['description'], "short": False}
                    ],
                    "footer": "Metrics Monitoring System",
                    "ts": alert['fired_at']
                }]
            }

            requests.post(
                os.getenv('SLACK_WEBHOOK_URL'),
                json=message
            )
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Ingestion Layer (Stateless):
    - Scale collectors based on scrape targets
    - Each collector: 100-1000 targets
    - 10,000 targets = 10-100 collector instances
    - Auto-scale based on lag

    Storage Layer (Stateful):
    - Shard by metric name hash
    - Each shard: 10-50 GB data
    - Replication factor: 3x
    - Scale by adding shards (consistent hashing)

    Query Layer (Stateless):
    - Scale based on query load
    - Each querier: 100-500 qps
    - Auto-scale based on p99 latency

    Alert Evaluation (Stateless):
    - Partition rules across evaluators
    - Each evaluator: 1000-5000 rules
    - Scale based on rule count
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Query cache (Redis)** | 10x faster queries (80% hit rate) | Stale data (30s) |
    | **Downsampling** | 10x storage reduction | Lower resolution |
    | **Multi-tier storage** | 5x cost reduction | Query complexity |
    | **Compression (Gorilla)** | 10x compression | CPU overhead |
    | **Inverted index** | 100x faster label queries | Memory overhead |
    | **Pre-aggregation** | 50x faster aggregations | Less flexibility |

    ---

    ## Monitoring the Monitor

    ```python
    # Key metrics for monitoring system health

    # Ingestion metrics
    metrics_ingested_per_second
    ingestion_lag_seconds
    invalid_metrics_dropped

    # Storage metrics
    active_time_series_count
    cardinality_per_metric
    disk_usage_per_tier
    block_compaction_duration

    # Query metrics
    query_latency_p99
    query_cache_hit_rate
    queries_per_second
    slow_query_count

    # Alert metrics
    alert_evaluation_duration
    alerts_firing_count
    alert_notification_failures
    ```

    ---

    ## Cost Optimization

    ```
    Monthly Cost (10M metrics/sec):

    Compute:
    - 100 ingestion nodes × $50 = $5,000
    - 118 storage nodes × $200 = $23,600
    - 34 query nodes × $150 = $5,100
    - Total compute: $33,700/month

    Storage:
    - Hot tier (161 TB SSD): 161 × $100 = $16,100
    - Warm tier (172 TB SSD): 172 × $100 = $17,200
    - Cold tier (371 TB S3): 371 × $23 = $8,533
    - Total storage: $41,833/month

    Network:
    - 5 Gbps × $0.08/GB × 330 TB/month = $26,400

    Total: ~$102K/month ≈ $1.2M/year

    Optimizations:
    1. Use spot instances (50% savings): -$16K
    2. Aggressive downsampling (reduce cold storage): -$5K
    3. Reserved instances (30% savings): -$10K
    4. Compression tuning (reduce storage 20%): -$8K

    Optimized Total: ~$63K/month ≈ $756K/year
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"Pull vs. Push metrics collection?"**
   - **Pull (Prometheus):** Centralized config, easier debugging, visible failures
   - **Push (StatsD):** Works behind firewalls, short-lived jobs, no service discovery
   - **Recommendation:** Pull for most cases (better visibility)

2. **"How do you handle high cardinality?"**
   - Validate labels at ingestion (reject user IDs, session IDs)
   - Monitor cardinality growth (alert on > 10K series per metric)
   - Use sampling for high-cardinality data
   - Separate system for traces (Jaeger) vs. metrics (Prometheus)
   - Pre-aggregate at collection time

3. **"How do you optimize query performance?"**
   - Query cache (Redis) - 80% hit rate, 10x faster
   - Push down filters to storage (inverted index)
   - Use downsampled data for long ranges (4-20x faster)
   - Parallel execution (fan-out to shards)
   - Result streaming (don't buffer all data)

4. **"How do you handle data retention?"**
   - Multi-tier storage (hot/warm/cold)
   - Automatic downsampling (full → 1min → 5min → 1hr)
   - Retention policies (7 days full, 30 days 1min, 90 days 5min, 1 year 1hr)
   - Compressed blocks uploaded to S3 (90% cost savings)

5. **"How do you prevent alert fatigue?"**
   - Deduplication (same alert from multiple sources)
   - Grouping (batch related alerts)
   - Silences (mute during maintenance)
   - Duration threshold (condition must be true for 5 minutes)
   - Severity levels (critical, warning, info)

6. **"How do you scale the system?"**
   - Horizontal scaling (add more nodes)
   - Sharding (consistent hashing by metric name)
   - Replication (3x for availability)
   - Auto-scaling (based on lag, latency)
   - Multi-region deployment (reduce latency)

7. **"How do you ensure data durability?"**
   - Write-Ahead Log (WAL) - replay on crash
   - Replication (async, 3x)
   - Periodic snapshots (RDB-style)
   - Multi-tier storage (S3 for cold data)
   - **Trade-off:** Async replication means < 1s of data loss on failure

**Key Points to Mention:**

- Time-series database optimized for metrics (columnar storage, compression)
- Multi-tier storage for cost optimization (90% savings)
- Query cache for performance (80% hit rate)
- Cardinality management (biggest challenge at scale)
- Downsampling for long retention (20x storage reduction)
- Pull-based collection (Prometheus-style)
- Rule-based alerting with deduplication
- Distributed architecture (horizontal scaling)

---

## Real-World Examples

**Prometheus Architecture:**
- Pull-based collection (scrape targets)
- Local TSDB (no distributed storage)
- PromQL query language
- Alertmanager for routing
- Federation for scaling

**Datadog Architecture:**
- Push-based collection (agents)
- Distributed TSDB (proprietary)
- 15-month retention
- High-cardinality support (100M series)
- SaaS platform

**Grafana Mimir:**
- Prometheus-compatible
- Object storage (S3) for blocks
- Horizontal scaling (1B+ series)
- Multi-tenancy support
- Open-source

---

## Summary

**System Characteristics:**

- **Ingestion:** 10M metrics/sec (30M peak)
- **Storage:** 471 TB (multi-tier), 10:1 compression
- **Query Latency:** < 100ms p99
- **Availability:** 99.99% uptime
- **Retention:** 15 months (with downsampling)

**Core Components:**

1. **Metrics Collectors:** Pull-based scraping (Prometheus)
2. **Ingestion Gateway:** Validation, rate limiting
3. **Time-Series Database:** Columnar storage, compression
4. **Query Engine:** Parallel execution, caching
5. **Alert Evaluator:** Rule-based alerting
6. **Alert Manager:** Deduplication, routing

**Key Design Decisions:**

- Pull-based collection (better visibility)
- Columnar time-series storage (10x compression)
- Multi-tier storage (90% cost savings)
- Downsampling (20x storage reduction)
- Query cache (10x faster queries)
- Cardinality validation (prevent explosions)
- Distributed architecture (horizontal scaling)
- Eventual consistency (high ingestion rate)

This design provides a scalable, cost-effective metrics monitoring system capable of handling billions of time-series data points with low query latency and intelligent alerting.
