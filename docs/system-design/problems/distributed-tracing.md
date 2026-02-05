# Design Distributed Tracing System

A distributed tracing system that captures, stores, and analyzes request flows across microservices, enabling developers to identify performance bottlenecks, debug distributed transactions, and understand service dependencies in complex architectures.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐ Medium | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10B+ spans/day, 100K+ traces/sec, petabyte-scale storage, <50ms query latency |
| **Key Challenges** | Context propagation, sampling strategies, high cardinality, storage optimization, service dependency graphs |
| **Core Concepts** | Trace, span, trace context, sampling (head-based, tail-based), span storage, waterfall visualization |
| **Companies** | Jaeger, Zipkin, Datadog APM, New Relic, AWS X-Ray, Google Cloud Trace, Lightstep, Honeycomb |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Span Collection** | Ingest spans from instrumented services | P0 (Must have) |
    | **Trace Context Propagation** | Propagate trace_id and span_id across services | P0 (Must have) |
    | **Trace Storage** | Store complete traces with all spans | P0 (Must have) |
    | **Trace Query** | Search traces by trace_id, tags, duration | P0 (Must have) |
    | **Waterfall Visualization** | Display trace timeline with spans | P0 (Must have) |
    | **Service Dependency Graph** | Map service-to-service calls | P0 (Must have) |
    | **Sampling** | Intelligently sample traces to reduce volume | P0 (Must have) |
    | **Performance Analysis** | Identify slow spans, bottlenecks | P1 (Should have) |
    | **Error Tracking** | Flag traces with errors | P1 (Should have) |
    | **Span Attributes/Tags** | Custom metadata on spans | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Full APM capabilities (metrics, profiling)
    - Log aggregation (use ELK/Splunk)
    - Real-time anomaly detection
    - Custom analytics/ML models
    - Code-level profiling

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Ingestion Rate** | 100K spans/sec (10B/day) | Support large microservices architecture |
    | **Query Latency** | < 100ms p95 for trace lookup | Fast debugging experience |
    | **Availability** | 99.9% uptime | Non-critical (monitoring tool) |
    | **Retention** | 30 days full data, 90 days sampled | Balance cost and debugging needs |
    | **Sampling Efficiency** | 0.1-1% sampling rate | Reduce volume while preserving insights |
    | **Context Overhead** | < 1KB per request | Minimal impact on application performance |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Microservices architecture:
    - Total services: 500 microservices
    - Requests per second: 500K requests/sec
    - Spans per request: 20 spans (average call depth)
    - Total spans: 500K × 20 = 10M spans/sec

    With sampling (1%):
    - Sampled spans: 10M × 0.01 = 100K spans/sec
    - Daily spans: 100K × 86,400 = 8.64B spans/day

    Peak traffic (3x):
    - Peak spans: 300K spans/sec
    - Peak daily: 26B spans/day

    Query load:
    - Active developers: 200 developers
    - Queries per developer per day: 50
    - Total queries: 10,000 queries/day ≈ 0.12 queries/sec
    - Peak queries: 10 queries/sec (during incidents)
    ```

    ### Storage Estimates

    ```
    Per span size:
    - Trace ID: 16 bytes (128-bit UUID)
    - Span ID: 8 bytes (64-bit)
    - Parent Span ID: 8 bytes
    - Service name: 50 bytes (avg)
    - Operation name: 50 bytes (avg)
    - Start time: 8 bytes (timestamp)
    - Duration: 8 bytes
    - Tags/attributes: 200 bytes (avg: 10 tags × 20 bytes)
    - Status: 1 byte
    - Total per span: ~350 bytes

    Raw storage (with sampling):
    - Spans per day: 8.64B spans
    - Storage per day: 8.64B × 350 bytes = 3 TB/day
    - With compression (5:1): 600 GB/day

    Retention storage:

    Tier 1: Full data (30 days)
    - Storage: 600 GB/day × 30 = 18 TB
    - Use: Recent debugging, root cause analysis

    Tier 2: Sampled (90 days)
    - Additional sampling: 10x (keep 0.1%)
    - Storage: 600 GB/day × 90 / 10 = 5.4 TB
    - Use: Historical trends, long-term patterns

    Indexes:
    - Trace ID index: 8.64B traces/day × 16 bytes = 138 GB/day
    - Tag index: 8.64B spans × 200 bytes / 10 (compression) = 173 GB/day
    - Total index: ~300 GB/day

    Total storage: 18 TB (full) + 5.4 TB (sampled) + 9 TB (indexes) = 32.4 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (span collection):
    - 100K spans/sec × 350 bytes = 35 MB/sec ≈ 280 Mbps
    - Compressed: 70 Mbps
    - Peak (3x): 210 Mbps

    Egress (trace queries):
    - 10 queries/sec × 20 spans × 350 bytes = 70 KB/sec ≈ 0.56 Mbps
    - Dashboard traffic: 100 dashboards × 1 MB/min = 1.67 MB/sec ≈ 13 Mbps
    - Total egress: ~15 Mbps

    Total bandwidth: 70 Mbps (ingress) + 15 Mbps (egress) = 85 Mbps
    ```

    ### Server Estimates

    ```
    Collection layer:
    - 100K spans/sec / 10K spans per collector = 10 collectors
    - CPU: 4 cores per collector (parsing, validation)
    - Memory: 8 GB per collector (buffering)

    Storage layer (Cassandra/Elasticsearch):
    - 32 TB / 2 TB per node = 16 storage nodes
    - Replication factor: 3x = 48 nodes total
    - CPU: 8 cores per node (indexing, compaction)
    - Memory: 32 GB per node (caches, bloom filters)
    - Disk: 2 TB SSD per node

    Query layer:
    - 10 queries/sec / 50 queries per node = 1-2 query nodes
    - CPU: 16 cores per node (aggregation)
    - Memory: 64 GB per node (query cache)

    Total servers:
    - Collection: 10 nodes
    - Storage: 48 nodes
    - Query: 2 nodes
    - Coordination: 3 nodes (Kafka, metadata)
    - Total: ~63 nodes
    ```

    ---

    ## Key Assumptions

    1. 1% sampling rate reduces volume while preserving critical traces
    2. Average trace has 20 spans (typical microservices depth)
    3. 30-day retention for full data, 90-day for sampled data
    4. Traces are immutable once created (append-only)
    5. Most queries are for recent traces (last 24 hours)
    6. W3C Trace Context standard for context propagation
    7. OpenTelemetry as instrumentation standard

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Context propagation:** Pass trace context through all service calls
    2. **Async collection:** Non-blocking span reporting via queues
    3. **Columnar storage:** Optimize for trace reconstruction and tag queries
    4. **Intelligent sampling:** Capture interesting traces, drop boring ones
    5. **Inverted indexes:** Fast tag-based search
    6. **Service mesh integration:** Automatic instrumentation

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Application Layer (Instrumented Services)"
            Service1[Service A<br/>OpenTelemetry SDK]
            Service2[Service B<br/>OpenTelemetry SDK]
            Service3[Service C<br/>OpenTelemetry SDK]
            ServiceN[Service N<br/>OpenTelemetry SDK]
        end

        subgraph "Context Propagation"
            HTTP[HTTP Headers<br/>traceparent/tracestate]
            gRPC[gRPC Metadata<br/>grpc-trace-bin]
            Kafka[Kafka Headers<br/>trace-id]
        end

        subgraph "Collection Layer"
            Agent1[OpenTelemetry<br/>Agent 1]
            Agent2[OpenTelemetry<br/>Agent 2]
            AgentN[OpenTelemetry<br/>Agent N]
            LoadBalancer[Load Balancer]
        end

        subgraph "Ingestion Pipeline"
            Collector1[Collector 1<br/>Jaeger Collector]
            Collector2[Collector 2<br/>Jaeger Collector]
            Sampler[Sampling Processor<br/>Head-based sampling]
            Kafka_Queue[Kafka Queue<br/>span-ingestion]
        end

        subgraph "Processing Layer"
            SpanProcessor1[Span Processor 1<br/>Validation<br/>Enrichment]
            SpanProcessor2[Span Processor 2<br/>Validation<br/>Enrichment]
            TailSampler[Tail Sampler<br/>Intelligent sampling]
        end

        subgraph "Storage Layer"
            subgraph "Hot Storage (7 days)"
                Cassandra1[(Cassandra 1<br/>Spans)]
                Cassandra2[(Cassandra 2<br/>Spans)]
                Cassandra3[(Cassandra 3<br/>Spans)]
            end

            subgraph "Index Storage"
                ES1[(Elasticsearch 1<br/>Trace index<br/>Tag index)]
                ES2[(Elasticsearch 2<br/>Trace index<br/>Tag index)]
                ES3[(Elasticsearch 3<br/>Trace index<br/>Tag index)]
            end

            subgraph "Cold Storage (90 days)"
                S3[(S3 / GCS<br/>Parquet files<br/>Compressed)]
            end
        end

        subgraph "Query Layer"
            QueryService1[Query Service 1<br/>Trace reconstruction]
            QueryService2[Query Service 2<br/>Trace reconstruction]
            Cache[Query Cache<br/>Redis]
        end

        subgraph "Analysis Services"
            DependencyGraph[Service Dependency<br/>Graph Builder]
            Analytics[Analytics Service<br/>Latency percentiles<br/>Error rates]
        end

        subgraph "Visualization & API"
            UI[Tracing UI<br/>Waterfall view<br/>Service map]
            API[API Gateway<br/>REST/gRPC]
        end

        subgraph "Metadata Store"
            MetaDB[(PostgreSQL<br/>Service catalog<br/>Sampling config)]
        end

        Service1 -->|Span export| Agent1
        Service2 -->|Span export| Agent2
        Service3 -->|Span export| Agent2
        ServiceN -->|Span export| AgentN

        Service1 -.->|Propagate| HTTP
        Service2 -.->|Propagate| gRPC
        Service3 -.->|Propagate| Kafka

        Agent1 --> LoadBalancer
        Agent2 --> LoadBalancer
        AgentN --> LoadBalancer

        LoadBalancer --> Collector1
        LoadBalancer --> Collector2

        Collector1 --> Sampler
        Collector2 --> Sampler
        Sampler --> Kafka_Queue

        Kafka_Queue --> SpanProcessor1
        Kafka_Queue --> SpanProcessor2

        SpanProcessor1 --> TailSampler
        SpanProcessor2 --> TailSampler

        TailSampler --> Cassandra1
        TailSampler --> Cassandra2
        TailSampler --> Cassandra3

        TailSampler --> ES1
        TailSampler --> ES2
        TailSampler --> ES3

        Cassandra1 -.->|Archive| S3
        Cassandra2 -.->|Archive| S3
        Cassandra3 -.->|Archive| S3

        QueryService1 --> Cassandra1
        QueryService1 --> ES1
        QueryService1 --> S3
        QueryService1 --> Cache

        QueryService2 --> Cassandra2
        QueryService2 --> ES2
        QueryService2 --> Cache

        DependencyGraph --> QueryService1
        Analytics --> QueryService2

        UI --> API
        API --> QueryService1
        API --> QueryService2
        API --> DependencyGraph
        API --> Analytics

        Sampler --> MetaDB
        TailSampler --> MetaDB
        QueryService1 --> MetaDB

        style Service1 fill:#e1f5ff
        style Service2 fill:#e1f5ff
        style Service3 fill:#e1f5ff
        style Cassandra1 fill:#ffe1e1
        style ES1 fill:#fff4e1
        style S3 fill:#f0f0f0
        style TailSampler fill:#e8f5e9
        style UI fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **OpenTelemetry SDK** | Industry standard, auto-instrumentation, vendor-neutral | Custom instrumentation (reinvent wheel), Zipkin client (less features) |
    | **Kafka Queue** | Decouple ingestion from processing, handle spikes | Direct to storage (no buffering), RabbitMQ (less throughput) |
    | **Cassandra** | Write-optimized, time-series friendly, horizontally scalable | PostgreSQL (poor write performance), MongoDB (no time-series optimization) |
    | **Elasticsearch** | Inverted indexes for tag search, full-text search | Cassandra-only (slow tag queries), custom index (complex) |
    | **Tail Sampling** | Sample after seeing full trace (better signal) | Head sampling only (miss interesting traces) |
    | **S3 Cold Storage** | 90% cost savings for old data | All-SSD (expensive), delete data (lose long-term trends) |

    **Key Trade-off:** We chose **eventual consistency** for span ingestion (async queues) to achieve high throughput. Traces may take 5-10 seconds to become queryable, which is acceptable for a monitoring tool.

    ---

    ## API Design

    ### 1. Export Spans (OTLP - OpenTelemetry Protocol)

    **Request:**
    ```protobuf
    // OpenTelemetry Protocol (gRPC)
    service TraceService {
      rpc Export(ExportTraceServiceRequest) returns (ExportTraceServiceResponse) {}
    }

    message ExportTraceServiceRequest {
      repeated ResourceSpans resource_spans = 1;
    }

    message ResourceSpans {
      Resource resource = 1;  // Service info (name, version, host)
      repeated ScopeSpans scope_spans = 2;
    }

    message ScopeSpans {
      InstrumentationScope scope = 1;
      repeated Span spans = 2;
    }

    message Span {
      bytes trace_id = 1;        // 16 bytes (128-bit)
      bytes span_id = 2;         // 8 bytes (64-bit)
      bytes parent_span_id = 3;  // 8 bytes (optional)
      string name = 4;           // Operation name (e.g., "GET /api/users")
      SpanKind kind = 5;         // CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER
      fixed64 start_time_unix_nano = 6;
      fixed64 end_time_unix_nano = 7;
      repeated KeyValue attributes = 8;  // Tags
      repeated Event events = 9;         // Logs within span
      repeated Link links = 10;          // Links to other spans
      Status status = 11;                // OK, ERROR, UNSET
      TraceState trace_state = 12;       // W3C trace state
    }
    ```

    **Example (HTTP JSON):**
    ```bash
    POST /v1/traces
    Content-Type: application/json

    {
      "resource_spans": [{
        "resource": {
          "attributes": [
            {"key": "service.name", "value": {"string_value": "order-service"}},
            {"key": "service.version", "value": {"string_value": "1.2.3"}},
            {"key": "host.name", "value": {"string_value": "pod-abc123"}}
          ]
        },
        "scope_spans": [{
          "spans": [{
            "trace_id": "5B8EFFF798038103D269B633813FC60C",
            "span_id": "EEE19B7EC3C1B174",
            "parent_span_id": "EEE19B7EC3C1B173",
            "name": "POST /orders",
            "kind": "SPAN_KIND_SERVER",
            "start_time_unix_nano": "1735819200000000000",
            "end_time_unix_nano": "1735819200250000000",
            "attributes": [
              {"key": "http.method", "value": {"string_value": "POST"}},
              {"key": "http.route", "value": {"string_value": "/orders"}},
              {"key": "http.status_code", "value": {"int_value": 201}},
              {"key": "user.id", "value": {"string_value": "user-12345"}}
            ],
            "status": {"code": "STATUS_CODE_OK"}
          }]
        }]
      }]
    }
    ```

    **Response:**
    ```json
    {
      "partial_success": {
        "rejected_spans": 0,
        "error_message": ""
      }
    }
    ```

    ---

    ### 2. Query Trace by ID

    **Request:**
    ```bash
    GET /api/v1/traces/5B8EFFF798038103D269B633813FC60C
    ```

    **Response:**
    ```json
    {
      "trace_id": "5B8EFFF798038103D269B633813FC60C",
      "spans": [
        {
          "trace_id": "5B8EFFF798038103D269B633813FC60C",
          "span_id": "EEE19B7EC3C1B173",
          "parent_span_id": null,
          "service_name": "api-gateway",
          "operation_name": "POST /orders",
          "start_time": "2025-01-02T10:00:00.000Z",
          "duration_ms": 523,
          "tags": {
            "http.method": "POST",
            "http.status_code": 201,
            "component": "go-http"
          },
          "status": "ok"
        },
        {
          "trace_id": "5B8EFFF798038103D269B633813FC60C",
          "span_id": "EEE19B7EC3C1B174",
          "parent_span_id": "EEE19B7EC3C1B173",
          "service_name": "order-service",
          "operation_name": "CreateOrder",
          "start_time": "2025-01-02T10:00:00.050Z",
          "duration_ms": 287,
          "tags": {
            "order.id": "order-98765",
            "user.id": "user-12345"
          },
          "status": "ok"
        },
        {
          "trace_id": "5B8EFFF798038103D269B633813FC60C",
          "span_id": "EEE19B7EC3C1B175",
          "parent_span_id": "EEE19B7EC3C1B174",
          "service_name": "payment-service",
          "operation_name": "ProcessPayment",
          "start_time": "2025-01-02T10:00:00.100Z",
          "duration_ms": 234,
          "tags": {
            "payment.amount": "99.99",
            "payment.method": "credit_card"
          },
          "status": "ok"
        }
      ],
      "process_map": {
        "api-gateway": {"service_name": "api-gateway", "host": "gateway-01"},
        "order-service": {"service_name": "order-service", "host": "order-pod-123"},
        "payment-service": {"service_name": "payment-service", "host": "payment-pod-456"}
      }
    }
    ```

    ---

    ### 3. Search Traces

    **Request:**
    ```bash
    POST /api/v1/search
    Content-Type: application/json

    {
      "service_name": "order-service",
      "operation_name": "CreateOrder",
      "tags": {
        "http.status_code": "500",
        "error": "true"
      },
      "min_duration": "1s",
      "max_duration": "10s",
      "start_time": "2025-01-01T00:00:00Z",
      "end_time": "2025-01-02T00:00:00Z",
      "limit": 20
    }
    ```

    **Response:**
    ```json
    {
      "traces": [
        {
          "trace_id": "ABC123...",
          "root_service_name": "api-gateway",
          "root_operation_name": "POST /orders",
          "start_time": "2025-01-01T10:30:45.123Z",
          "duration_ms": 2341,
          "span_count": 15,
          "error_count": 1,
          "services": ["api-gateway", "order-service", "payment-service"]
        },
        // ... more traces
      ],
      "total": 234,
      "limit": 20,
      "offset": 0
    }
    ```

    ---

    ### 4. Service Dependency Graph

    **Request:**
    ```bash
    GET /api/v1/dependencies?lookback=24h
    ```

    **Response:**
    ```json
    {
      "dependencies": [
        {
          "parent": "api-gateway",
          "child": "order-service",
          "call_count": 1234567,
          "error_rate": 0.002,
          "avg_duration_ms": 45.3,
          "p95_duration_ms": 123.4,
          "p99_duration_ms": 234.5
        },
        {
          "parent": "order-service",
          "child": "payment-service",
          "call_count": 987654,
          "error_rate": 0.001,
          "avg_duration_ms": 78.2,
          "p95_duration_ms": 156.7,
          "p99_duration_ms": 289.3
        },
        {
          "parent": "order-service",
          "child": "inventory-service",
          "call_count": 1234567,
          "error_rate": 0.005,
          "avg_duration_ms": 23.1,
          "p95_duration_ms": 67.8,
          "p99_duration_ms": 123.4
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### Cassandra (Span Storage)

    **Spans Table (Time-Series Optimized):**

    ```sql
    CREATE TABLE spans (
        trace_id blob,                  -- Partition key (group spans by trace)
        span_id blob,                   -- Clustering key (unique within trace)
        parent_span_id blob,
        service_name text,
        operation_name text,
        start_time_ns bigint,           -- Clustering key (time-ordered)
        duration_ns bigint,
        kind text,                      -- CLIENT, SERVER, INTERNAL
        status_code text,               -- OK, ERROR
        tags map<text, text>,           -- Key-value attributes
        events list<frozen<event>>,     -- Span events (logs)
        PRIMARY KEY ((trace_id), start_time_ns, span_id)
    ) WITH CLUSTERING ORDER BY (start_time_ns ASC)
      AND compaction = {'class': 'TimeWindowCompactionStrategy', 'compaction_window_unit': 'HOURS', 'compaction_window_size': 1}
      AND default_time_to_live = 2592000;  -- 30 days

    -- Query patterns:
    -- 1. Get all spans for a trace: SELECT * FROM spans WHERE trace_id = ?
    -- 2. Time-ordered spans: Clustering by start_time_ns
    -- 3. Automatic expiration: TTL = 30 days

    CREATE TYPE event (
        time_ns bigint,
        name text,
        attributes map<text, text>
    );
    ```

    **Trace Index (for tag queries - use Elasticsearch instead):**

    ```sql
    CREATE TABLE trace_index (
        service_name text,              -- Partition key
        start_time_bucket text,         -- Partition key (e.g., "2025-01-02-10")
        trace_id blob,
        operation_name text,
        start_time_ns bigint,           -- Clustering key
        duration_ns bigint,
        status_code text,
        tags map<text, text>,
        PRIMARY KEY ((service_name, start_time_bucket), start_time_ns, trace_id)
    ) WITH CLUSTERING ORDER BY (start_time_ns DESC)
      AND default_time_to_live = 2592000;

    -- Query patterns:
    -- 1. Find traces by service and time: WHERE service_name = ? AND start_time_bucket = ?
    -- 2. Filter by duration: WHERE duration_ns > ?
    -- 3. Filter by status: WHERE status_code = 'ERROR'
    ```

    ---

    ### Elasticsearch (Tag Index)

    **Trace Index:**

    ```json
    {
      "mappings": {
        "properties": {
          "trace_id": {"type": "keyword"},
          "service_name": {"type": "keyword"},
          "operation_name": {"type": "keyword"},
          "start_time": {"type": "date"},
          "duration_ms": {"type": "integer"},
          "span_count": {"type": "integer"},
          "error": {"type": "boolean"},
          "status_code": {"type": "keyword"},
          "tags": {"type": "object", "dynamic": true}
        }
      }
    }
    ```

    **Tag Search:**

    ```json
    POST /traces/_search
    {
      "query": {
        "bool": {
          "must": [
            {"term": {"service_name": "order-service"}},
            {"range": {"duration_ms": {"gte": 1000}}},
            {"term": {"tags.http.status_code": "500"}},
            {"range": {"start_time": {"gte": "2025-01-01", "lte": "2025-01-02"}}}
          ]
        }
      },
      "sort": [{"start_time": "desc"}],
      "size": 20
    }
    ```

    ---

    ### PostgreSQL (Metadata)

    **Service Catalog:**

    ```sql
    CREATE TABLE services (
        service_id SERIAL PRIMARY KEY,
        service_name VARCHAR(255) UNIQUE NOT NULL,
        description TEXT,
        owner_team VARCHAR(100),
        repository_url VARCHAR(500),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_service_name ON services(service_name);
    ```

    **Sampling Configuration:**

    ```sql
    CREATE TABLE sampling_config (
        config_id SERIAL PRIMARY KEY,
        service_name VARCHAR(255),
        operation_name VARCHAR(255),
        sampling_strategy VARCHAR(50) NOT NULL,  -- probabilistic, rate_limiting, tail_based
        sampling_rate FLOAT,                     -- 0.0 to 1.0
        priority INT DEFAULT 0,                  -- Higher priority = more likely to sample
        enabled BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    -- Example: Always sample errors
    INSERT INTO sampling_config VALUES
        (1, '*', '*', 'tail_based', 1.0, 100, true),  -- Sample all errors (high priority)
        (2, 'payment-service', '*', 'probabilistic', 0.1, 50, true),  -- 10% of payment traces
        (3, '*', 'healthcheck', 'probabilistic', 0.001, 1, true);  -- 0.1% of healthchecks

    CREATE INDEX idx_sampling_service ON sampling_config(service_name, operation_name);
    ```

    ---

    ## Data Flow Diagrams

    ### Trace Creation & Propagation

    ```mermaid
    sequenceDiagram
        participant Client
        participant Gateway as API Gateway<br/>(Service A)
        participant Order as Order Service<br/>(Service B)
        participant Payment as Payment Service<br/>(Service C)

        Client->>Gateway: POST /orders<br/>(no trace context)

        Note over Gateway: 1. Create new trace<br/>trace_id = generate_id()<br/>span_id = generate_id()

        Gateway->>Gateway: Start span: "POST /orders"

        Note over Gateway: 2. Inject trace context<br/>into HTTP headers

        Gateway->>Order: POST /api/orders<br/>traceparent: 00-{trace_id}-{span_id}-01

        Note over Order: 3. Extract trace context<br/>parent_span_id = span_id<br/>span_id = generate_new_id()

        Order->>Order: Start span: "CreateOrder"

        Note over Order: 4. Propagate context

        Order->>Payment: POST /api/payments<br/>traceparent: 00-{trace_id}-{span_id}-01

        Note over Payment: 5. Extract & continue trace

        Payment->>Payment: Start span: "ProcessPayment"
        Payment->>Payment: End span (duration: 234ms)

        Payment-->>Order: 201 Created

        Order->>Order: End span (duration: 287ms)

        Order-->>Gateway: 201 Created

        Gateway->>Gateway: End span (duration: 523ms)

        Gateway-->>Client: 201 Created

        Note over Gateway,Payment: All spans exported<br/>asynchronously to collector
    ```

    **W3C Trace Context Format:**

    ```
    traceparent: 00-{trace-id}-{parent-span-id}-{trace-flags}

    Example:
    traceparent: 00-5B8EFFF798038103D269B633813FC60C-EEE19B7EC3C1B173-01

    - Version: 00
    - Trace ID: 5B8EFFF798038103D269B633813FC60C (128-bit, 32 hex chars)
    - Parent Span ID: EEE19B7EC3C1B173 (64-bit, 16 hex chars)
    - Trace Flags: 01 (sampled)

    tracestate: vendor1=value1,vendor2=value2
    ```

    ---

    ### Span Ingestion & Storage

    ```mermaid
    sequenceDiagram
        participant SDK as OpenTelemetry<br/>SDK
        participant Agent as OTel Agent<br/>(on host)
        participant Collector as Collector
        participant Sampler as Head Sampler
        participant Kafka
        participant Processor as Span Processor
        participant TailSampler as Tail Sampler
        participant Cassandra
        participant ES as Elasticsearch

        SDK->>Agent: Export spans (batch)<br/>gRPC OTLP

        Agent->>Collector: Forward spans<br/>(load balanced)

        Collector->>Sampler: Head-based sampling<br/>(1% probabilistic)

        alt Sampled
            Sampler->>Kafka: Write to span-ingestion topic
            Kafka-->>Sampler: Ack
        else Dropped
            Note over Sampler: Drop span<br/>(not sampled)
        end

        Kafka->>Processor: Consume span batch

        Processor->>Processor: 1. Validate span<br/>2. Enrich with metadata<br/>3. Group by trace_id

        Processor->>TailSampler: Send span<br/>(buffer by trace_id)

        Note over TailSampler: Wait for trace completion<br/>(5 second window)

        alt Trace complete + interesting
            TailSampler->>Cassandra: Write spans<br/>(by trace_id)
            TailSampler->>ES: Index trace metadata<br/>(tags, duration, status)
        else Trace boring (all ok, fast)
            Note over TailSampler: Drop trace<br/>(additional sampling)
        end

        Note over Cassandra,ES: Spans stored,<br/>ready for query
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical distributed tracing subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Trace Context Propagation** | How to maintain trace identity across services? | W3C Trace Context standard with automatic injection/extraction |
    | **Sampling Strategies** | How to reduce volume while preserving insights? | Multi-stage sampling: head-based + tail-based + adaptive |
    | **Trace Storage** | How to store and query high-cardinality trace data? | Cassandra for spans + Elasticsearch for tag indexing |
    | **Service Dependency Graphs** | How to build real-time service maps? | Aggregate span relationships into directed graph |

    ---

    === "🔗 Trace Context Propagation"

        ## The Challenge

        **Problem:** Maintain trace identity as requests flow through 20+ microservices using HTTP, gRPC, Kafka, etc.

        **Requirements:**

        - Propagate trace_id and span_id across all service boundaries
        - Support multiple protocols (HTTP, gRPC, message queues)
        - Minimize overhead (< 1KB per request)
        - Vendor-neutral format (avoid lock-in)
        - Backward compatible (gracefully handle missing context)

        ---

        ## W3C Trace Context Implementation

        ```python
        import uuid
        import random
        from typing import Optional, Dict

        class TraceContext:
            """
            W3C Trace Context implementation

            Spec: https://www.w3.org/TR/trace-context/

            Format:
            traceparent: 00-{trace-id}-{parent-id}-{flags}
            tracestate: vendor1=value1,vendor2=value2
            """

            def __init__(
                self,
                trace_id: Optional[str] = None,
                span_id: Optional[str] = None,
                parent_span_id: Optional[str] = None,
                trace_flags: int = 0x00,
                trace_state: Optional[str] = None
            ):
                self.trace_id = trace_id or self._generate_trace_id()
                self.span_id = span_id or self._generate_span_id()
                self.parent_span_id = parent_span_id
                self.trace_flags = trace_flags  # 0x01 = sampled
                self.trace_state = trace_state or ""

            def _generate_trace_id(self) -> str:
                """Generate 128-bit trace ID (32 hex characters)"""
                return uuid.uuid4().hex + uuid.uuid4().hex[:16]

            def _generate_span_id(self) -> str:
                """Generate 64-bit span ID (16 hex characters)"""
                return uuid.uuid4().hex[:16]

            def is_sampled(self) -> bool:
                """Check if trace is sampled"""
                return (self.trace_flags & 0x01) == 0x01

            def set_sampled(self, sampled: bool):
                """Set sampling flag"""
                if sampled:
                    self.trace_flags |= 0x01
                else:
                    self.trace_flags &= ~0x01

            def to_traceparent_header(self) -> str:
                """
                Serialize to traceparent header

                Format: 00-{trace-id}-{span-id}-{flags}
                Example: 00-5b8efff798038103d269b633813fc60c-eee19b7ec3c1b174-01
                """
                return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

            def to_tracestate_header(self) -> str:
                """
                Serialize to tracestate header

                Format: vendor1=value1,vendor2=value2
                """
                return self.trace_state

            @classmethod
            def from_traceparent_header(cls, traceparent: str) -> 'TraceContext':
                """
                Parse traceparent header

                Format: 00-{trace-id}-{span-id}-{flags}
                """
                parts = traceparent.split('-')

                if len(parts) != 4:
                    raise ValueError(f"Invalid traceparent format: {traceparent}")

                version, trace_id, parent_span_id, flags = parts

                if version != '00':
                    raise ValueError(f"Unsupported version: {version}")

                if len(trace_id) != 32:
                    raise ValueError(f"Invalid trace_id length: {len(trace_id)}")

                if len(parent_span_id) != 16:
                    raise ValueError(f"Invalid span_id length: {len(parent_span_id)}")

                return cls(
                    trace_id=trace_id,
                    parent_span_id=parent_span_id,
                    trace_flags=int(flags, 16)
                )

            @classmethod
            def from_tracestate_header(cls, tracestate: str, context: 'TraceContext') -> 'TraceContext':
                """Parse tracestate header"""
                context.trace_state = tracestate
                return context


        class Span:
            """
            Represents a span in a distributed trace
            """

            def __init__(
                self,
                context: TraceContext,
                operation_name: str,
                service_name: str,
                kind: str = "INTERNAL"
            ):
                self.trace_id = context.trace_id
                self.span_id = context.span_id
                self.parent_span_id = context.parent_span_id
                self.operation_name = operation_name
                self.service_name = service_name
                self.kind = kind  # CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER
                self.start_time = None
                self.end_time = None
                self.tags = {}
                self.events = []
                self.status = "UNSET"

            def start(self):
                """Start span timer"""
                self.start_time = time.time_ns()

            def end(self):
                """End span timer"""
                self.end_time = time.time_ns()

            def set_tag(self, key: str, value: str):
                """Add tag/attribute to span"""
                self.tags[key] = value

            def set_error(self, error: Exception):
                """Mark span as error"""
                self.status = "ERROR"
                self.tags['error'] = True
                self.tags['error.message'] = str(error)
                self.tags['error.type'] = type(error).__name__

            def add_event(self, name: str, attributes: Dict[str, str] = None):
                """Add event to span (log)"""
                self.events.append({
                    'timestamp': time.time_ns(),
                    'name': name,
                    'attributes': attributes or {}
                })


        class Tracer:
            """
            Tracer manages trace context and span creation
            """

            def __init__(self, service_name: str, exporter):
                self.service_name = service_name
                self.exporter = exporter

            def start_trace(self, operation_name: str) -> Span:
                """
                Start a new trace (root span)

                Called when request enters system
                """
                context = TraceContext()

                # Apply sampling decision (head-based)
                sampled = self._should_sample(operation_name)
                context.set_sampled(sampled)

                span = Span(
                    context=context,
                    operation_name=operation_name,
                    service_name=self.service_name,
                    kind="SERVER"
                )
                span.start()

                return span

            def start_span_from_context(
                self,
                context: TraceContext,
                operation_name: str,
                kind: str = "INTERNAL"
            ) -> Span:
                """
                Start a child span from existing trace context

                Called when continuing an existing trace
                """
                # Create new span ID, keep trace ID
                child_context = TraceContext(
                    trace_id=context.trace_id,
                    parent_span_id=context.span_id,  # Current span becomes parent
                    trace_flags=context.trace_flags,
                    trace_state=context.trace_state
                )

                span = Span(
                    context=child_context,
                    operation_name=operation_name,
                    service_name=self.service_name,
                    kind=kind
                )
                span.start()

                return span

            def inject_http_headers(self, span: Span, headers: Dict[str, str]):
                """
                Inject trace context into HTTP headers

                Called before making outbound HTTP request
                """
                context = TraceContext(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    trace_flags=1 if span.status != "DROP" else 0
                )

                headers['traceparent'] = context.to_traceparent_header()

                if context.trace_state:
                    headers['tracestate'] = context.to_tracestate_header()

            def extract_http_headers(self, headers: Dict[str, str]) -> Optional[TraceContext]:
                """
                Extract trace context from HTTP headers

                Called when receiving inbound HTTP request
                """
                traceparent = headers.get('traceparent')

                if not traceparent:
                    return None

                try:
                    context = TraceContext.from_traceparent_header(traceparent)

                    tracestate = headers.get('tracestate')
                    if tracestate:
                        context = TraceContext.from_tracestate_header(tracestate, context)

                    return context
                except ValueError as e:
                    logger.warning(f"Invalid trace context: {e}")
                    return None

            def finish_span(self, span: Span):
                """
                Finish span and export to collector
                """
                span.end()

                # Only export if sampled
                if span.status != "DROP":
                    self.exporter.export_span(span)

            def _should_sample(self, operation_name: str) -> bool:
                """
                Head-based sampling decision

                Simple probabilistic sampling (1%)
                """
                if operation_name == "healthcheck":
                    return random.random() < 0.001  # 0.1% for healthchecks
                else:
                    return random.random() < 0.01  # 1% for other operations
        ```

        ---

        ## Multi-Protocol Propagation

        ```python
        class ContextPropagation:
            """Handle context propagation across different protocols"""

            @staticmethod
            def inject_grpc_metadata(span: Span, metadata: dict):
                """
                Inject trace context into gRPC metadata

                gRPC uses binary headers for efficiency
                """
                context = TraceContext(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    trace_flags=1
                )

                # Binary format (more efficient than text)
                trace_context_bytes = context.to_binary()
                metadata['grpc-trace-bin'] = trace_context_bytes

            @staticmethod
            def extract_grpc_metadata(metadata: dict) -> Optional[TraceContext]:
                """Extract trace context from gRPC metadata"""
                trace_context_bytes = metadata.get('grpc-trace-bin')

                if trace_context_bytes:
                    return TraceContext.from_binary(trace_context_bytes)

                return None

            @staticmethod
            def inject_kafka_headers(span: Span, headers: list):
                """
                Inject trace context into Kafka message headers

                Kafka headers are list of (key, value) tuples
                """
                context = TraceContext(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    trace_flags=1
                )

                headers.append(('traceparent', context.to_traceparent_header().encode('utf-8')))

                if context.trace_state:
                    headers.append(('tracestate', context.trace_state.encode('utf-8')))

            @staticmethod
            def extract_kafka_headers(headers: list) -> Optional[TraceContext]:
                """Extract trace context from Kafka message headers"""
                headers_dict = {k: v.decode('utf-8') for k, v in headers}

                traceparent = headers_dict.get('traceparent')

                if traceparent:
                    context = TraceContext.from_traceparent_header(traceparent)

                    tracestate = headers_dict.get('tracestate')
                    if tracestate:
                        context.trace_state = tracestate

                    return context

                return None
        ```

        ---

        ## Usage Example

        ```python
        # Service A: Start trace (API Gateway)
        tracer = Tracer(service_name="api-gateway", exporter=collector)

        # Receive HTTP request (no trace context)
        def handle_request(request):
            # Try to extract context from headers
            context = tracer.extract_http_headers(request.headers)

            if context:
                # Continue existing trace
                span = tracer.start_span_from_context(
                    context,
                    operation_name="POST /orders",
                    kind="SERVER"
                )
            else:
                # Start new trace
                span = tracer.start_trace(operation_name="POST /orders")

            span.set_tag("http.method", "POST")
            span.set_tag("http.route", "/orders")

            try:
                # Call downstream service
                order = create_order_downstream(span, request.body)

                span.set_tag("http.status_code", "201")
                span.set_tag("order.id", order.id)

                return Response(201, order)

            except Exception as e:
                span.set_error(e)
                span.set_tag("http.status_code", "500")
                raise

            finally:
                tracer.finish_span(span)


        # Service B: Continue trace (Order Service)
        def create_order_downstream(parent_span, order_data):
            # Start child span for HTTP call
            http_span = tracer.start_span_from_context(
                TraceContext(
                    trace_id=parent_span.trace_id,
                    span_id=parent_span.span_id
                ),
                operation_name="HTTP POST",
                kind="CLIENT"
            )

            # Inject context into headers
            headers = {}
            tracer.inject_http_headers(http_span, headers)

            # Make HTTP request
            response = requests.post(
                "http://order-service/api/orders",
                json=order_data,
                headers=headers
            )

            http_span.set_tag("http.status_code", str(response.status_code))
            tracer.finish_span(http_span)

            return response.json()
        ```

    === "🎯 Sampling Strategies"

        ## The Challenge

        **Problem:** Tracing 100% of requests generates 10M spans/sec (too expensive). Need to sample intelligently.

        **Naive approach:** Random 1% sampling. **Problem:** Miss important traces (errors, slow requests, rare endpoints).

        **Requirements:**

        - Reduce volume 100x (1% sampling)
        - Preserve all errors (100% sampling)
        - Preserve slow requests (>1s)
        - Preserve rare operations
        - Make sampling decision early (head-based) or late (tail-based)

        ---

        ## Sampling Strategies

        ```python
        from abc import ABC, abstractmethod
        import time
        import random
        from collections import defaultdict

        class SamplingStrategy(ABC):
            """Base class for sampling strategies"""

            @abstractmethod
            def should_sample(self, trace_context: dict) -> bool:
                """Decide if trace should be sampled"""
                pass


        class ProbabilisticSampler(SamplingStrategy):
            """
            Probabilistic sampling (head-based)

            Sample X% of all traces randomly

            Pros:
            - Simple, predictable volume
            - Low CPU overhead
            - Stateless (no coordination)

            Cons:
            - May miss important traces (errors, slow requests)
            - No adaptivity
            """

            def __init__(self, sampling_rate: float):
                self.sampling_rate = sampling_rate  # 0.0 to 1.0

            def should_sample(self, trace_context: dict) -> bool:
                """
                Sample based on trace_id hash (consistent sampling)

                All spans in same trace get same decision
                """
                trace_id = trace_context.get('trace_id', '')

                # Hash trace_id to [0, 1)
                hash_value = int(trace_id[:16], 16) / (2 ** 64)

                return hash_value < self.sampling_rate


        class RateLimitingSampler(SamplingStrategy):
            """
            Rate limiting sampling (head-based)

            Sample up to N traces per second

            Use case: Protect backend from overload
            """

            def __init__(self, traces_per_second: int):
                self.traces_per_second = traces_per_second
                self.bucket_tokens = traces_per_second
                self.last_refill = time.time()

            def should_sample(self, trace_context: dict) -> bool:
                """
                Token bucket rate limiting
                """
                current_time = time.time()

                # Refill tokens
                elapsed = current_time - self.last_refill
                tokens_to_add = elapsed * self.traces_per_second
                self.bucket_tokens = min(
                    self.traces_per_second,
                    self.bucket_tokens + tokens_to_add
                )
                self.last_refill = current_time

                # Consume token
                if self.bucket_tokens >= 1:
                    self.bucket_tokens -= 1
                    return True
                else:
                    return False


        class AdaptiveSampler(SamplingStrategy):
            """
            Adaptive sampling (head-based)

            Adjust sampling rate based on traffic patterns

            Rules:
            - Sample more when traffic is low (capture more)
            - Sample less when traffic is high (reduce volume)
            - Always sample errors (100%)
            - Always sample slow requests (>1s)
            """

            def __init__(self, target_spans_per_second: int):
                self.target_spans_per_second = target_spans_per_second
                self.current_sampling_rate = 0.01  # Start at 1%
                self.span_count_window = 0
                self.window_start = time.time()

            def should_sample(self, trace_context: dict) -> bool:
                """
                Adjust sampling rate every 10 seconds
                """
                current_time = time.time()

                # Update sampling rate every 10 seconds
                if current_time - self.window_start > 10:
                    self._update_sampling_rate()
                    self.span_count_window = 0
                    self.window_start = current_time

                # Always sample errors
                if trace_context.get('has_error'):
                    return True

                # Always sample slow requests
                if trace_context.get('duration_ms', 0) > 1000:
                    return True

                # Probabilistic sampling
                sampled = random.random() < self.current_sampling_rate

                if sampled:
                    self.span_count_window += trace_context.get('span_count', 1)

                return sampled

            def _update_sampling_rate(self):
                """
                Adjust sampling rate based on recent traffic

                If span rate too high: reduce sampling
                If span rate too low: increase sampling
                """
                actual_rate = self.span_count_window / 10  # Spans per second

                if actual_rate > self.target_spans_per_second * 1.2:
                    # Too high, reduce sampling
                    self.current_sampling_rate *= 0.8
                elif actual_rate < self.target_spans_per_second * 0.8:
                    # Too low, increase sampling
                    self.current_sampling_rate *= 1.2

                # Clamp to [0.001, 1.0]
                self.current_sampling_rate = max(0.001, min(1.0, self.current_sampling_rate))

                logger.info(f"Adjusted sampling rate to {self.current_sampling_rate:.4f}")


        class TailBasedSampler:
            """
            Tail-based sampling (sample after seeing full trace)

            Pros:
            - Sample based on complete trace information
            - Can sample errors, slow requests, rare operations
            - More intelligent than head-based

            Cons:
            - Requires buffering spans (memory overhead)
            - Higher latency (wait for trace completion)
            - Stateful (need to group spans by trace_id)
            """

            def __init__(self, buffer_timeout: int = 10):
                self.buffer_timeout = buffer_timeout  # seconds
                self.trace_buffers = defaultdict(list)
                self.trace_first_seen = {}

            def add_span(self, span: dict):
                """
                Buffer span until trace is complete or timeout
                """
                trace_id = span['trace_id']

                # Add to buffer
                self.trace_buffers[trace_id].append(span)

                # Track first seen time
                if trace_id not in self.trace_first_seen:
                    self.trace_first_seen[trace_id] = time.time()

            def evaluate_traces(self) -> list:
                """
                Evaluate buffered traces and decide which to keep

                Returns list of spans to persist
                """
                current_time = time.time()
                traces_to_persist = []
                traces_to_delete = []

                for trace_id, spans in self.trace_buffers.items():
                    first_seen = self.trace_first_seen[trace_id]
                    age = current_time - first_seen

                    # Trace complete (root span finished) or timeout
                    trace_complete = self._is_trace_complete(spans)

                    if trace_complete or age > self.buffer_timeout:
                        # Make sampling decision
                        if self._should_keep_trace(spans):
                            traces_to_persist.extend(spans)

                        # Clean up buffer
                        traces_to_delete.append(trace_id)

                # Clean up
                for trace_id in traces_to_delete:
                    del self.trace_buffers[trace_id]
                    del self.trace_first_seen[trace_id]

                return traces_to_persist

            def _is_trace_complete(self, spans: list) -> bool:
                """
                Check if trace is complete (root span has ended)
                """
                # Find root span (no parent_span_id)
                root_spans = [s for s in spans if not s.get('parent_span_id')]

                if not root_spans:
                    return False

                # Check if root span has end_time
                return root_spans[0].get('end_time') is not None

            def _should_keep_trace(self, spans: list) -> bool:
                """
                Tail-based sampling decision

                Keep trace if:
                1. Has errors (status_code = ERROR)
                2. Is slow (duration > 1 second)
                3. Touches rare service (< 100 traces/hour)
                4. Random sample (1% of remaining)
                """
                # Rule 1: Keep all errors
                has_error = any(s.get('status_code') == 'ERROR' for s in spans)
                if has_error:
                    logger.info(f"Keeping trace {spans[0]['trace_id']}: has error")
                    return True

                # Rule 2: Keep slow traces
                root_span = next((s for s in spans if not s.get('parent_span_id')), None)
                if root_span:
                    duration_ms = (root_span['end_time'] - root_span['start_time']) / 1_000_000
                    if duration_ms > 1000:
                        logger.info(f"Keeping trace {spans[0]['trace_id']}: slow ({duration_ms}ms)")
                        return True

                # Rule 3: Keep traces from rare operations
                operations = set(s['operation_name'] for s in spans)
                if any(self._is_rare_operation(op) for op in operations):
                    logger.info(f"Keeping trace {spans[0]['trace_id']}: rare operation")
                    return True

                # Rule 4: Random sample (1%)
                keep = random.random() < 0.01
                if keep:
                    logger.info(f"Keeping trace {spans[0]['trace_id']}: random sample")
                return keep

            def _is_rare_operation(self, operation_name: str) -> bool:
                """
                Check if operation is rare (< 100 traces/hour)

                In production: query from metadata store
                """
                # Simplified: hardcode rare operations
                rare_operations = ['DELETE /users', 'POST /admin', 'GET /export']
                return operation_name in rare_operations


        class HybridSampler:
            """
            Hybrid sampling (head-based + tail-based)

            Strategy:
            1. Head-based: Sample 10% at ingestion (fast path)
            2. Tail-based: Buffer remaining 90%, evaluate after completion
            3. Final sample rate: 1% (10% head + 0.11% tail ≈ 1%)

            Benefits:
            - Low latency for head-sampled traces
            - Intelligent sampling for tail-sampled traces
            - Best of both worlds
            """

            def __init__(self):
                self.head_sampler = ProbabilisticSampler(sampling_rate=0.1)
                self.tail_sampler = TailBasedSampler(buffer_timeout=10)

            def process_span(self, span: dict) -> str:
                """
                Process span through hybrid sampling

                Returns: 'PERSIST' or 'BUFFER' or 'DROP'
                """
                # Always keep errors (fast path)
                if span.get('status_code') == 'ERROR':
                    return 'PERSIST'

                # Head-based sampling
                trace_context = {
                    'trace_id': span['trace_id'],
                    'service_name': span['service_name'],
                    'operation_name': span['operation_name']
                }

                if self.head_sampler.should_sample(trace_context):
                    return 'PERSIST'

                # Not head-sampled, buffer for tail-based evaluation
                return 'BUFFER'
        ```

        ---

        ## Sampling Decision Flow

        ```python
        class SamplingDecider:
            """Centralized sampling decision logic"""

            def __init__(self, config):
                self.config = config
                self.strategies = {
                    'probabilistic': ProbabilisticSampler(0.01),
                    'adaptive': AdaptiveSampler(target_spans_per_second=100_000),
                    'tail_based': TailBasedSampler(buffer_timeout=10),
                    'hybrid': HybridSampler()
                }

            def make_decision(self, span: dict) -> tuple:
                """
                Make sampling decision for span

                Returns: (action, reason)
                - action: 'PERSIST', 'BUFFER', 'DROP'
                - reason: explanation for decision
                """
                # Priority rules (checked in order)

                # 1. Always sample errors
                if span.get('status_code') == 'ERROR':
                    return ('PERSIST', 'error_trace')

                # 2. Always sample slow requests
                if span.get('duration_ms', 0) > 1000:
                    return ('PERSIST', 'slow_trace')

                # 3. Check service-specific rules
                service_config = self.config.get(span['service_name'])
                if service_config:
                    if service_config['sample_rate'] == 1.0:
                        return ('PERSIST', 'service_rule')

                # 4. Use configured strategy
                strategy_name = self.config.get('default_strategy', 'hybrid')
                strategy = self.strategies[strategy_name]

                if strategy_name == 'hybrid':
                    action = strategy.process_span(span)
                    return (action, f'hybrid_{action.lower()}')
                else:
                    trace_context = {
                        'trace_id': span['trace_id'],
                        'service_name': span['service_name']
                    }

                    if strategy.should_sample(trace_context):
                        return ('PERSIST', f'{strategy_name}_sampled')
                    else:
                        return ('DROP', f'{strategy_name}_dropped')
        ```

    === "💾 Trace Storage Optimization"

        ## The Challenge

        **Problem:** Store 8.64B spans/day (3 TB/day) with fast query performance.

        **Requirements:**

        - Fast trace reconstruction (all spans by trace_id)
        - Fast tag search (e.g., "http.status_code=500")
        - Fast time-range queries
        - Support high cardinality tags (1M+ unique values)
        - 30-day retention (90 TB total)
        - < 100ms query latency

        ---

        ## Cassandra Schema Design

        ```python
        class SpanStorage:
            """
            Cassandra-based span storage

            Design decisions:
            1. Partition by trace_id (all spans in same trace co-located)
            2. Cluster by start_time (time-ordered spans)
            3. TTL for automatic expiration (30 days)
            4. Use frozen collections for tags (no individual updates)
            """

            def __init__(self, cassandra_session):
                self.session = cassandra_session

            def create_schema(self):
                """
                Create Cassandra tables
                """
                # Main spans table
                create_table = """
                CREATE TABLE IF NOT EXISTS spans (
                    trace_id blob,                  -- Partition key
                    start_time_ns bigint,           -- Clustering key
                    span_id blob,                   -- Clustering key
                    parent_span_id blob,
                    service_name text,
                    operation_name text,
                    duration_ns bigint,
                    kind text,
                    status_code text,
                    tags map<text, text>,           -- Frozen for efficiency
                    events list<frozen<event>>,     -- Span events
                    PRIMARY KEY ((trace_id), start_time_ns, span_id)
                ) WITH CLUSTERING ORDER BY (start_time_ns ASC, span_id ASC)
                  AND compaction = {
                    'class': 'TimeWindowCompactionStrategy',
                    'compaction_window_unit': 'HOURS',
                    'compaction_window_size': 1
                  }
                  AND default_time_to_live = 2592000  -- 30 days
                  AND bloom_filter_fp_chance = 0.01;
                """

                self.session.execute(create_table)

                # Service index (for listing traces by service)
                create_service_index = """
                CREATE TABLE IF NOT EXISTS spans_by_service (
                    service_name text,
                    hour_bucket text,               -- Format: "2025-01-02-10"
                    start_time_ns bigint,
                    trace_id blob,
                    operation_name text,
                    duration_ns bigint,
                    status_code text,
                    PRIMARY KEY ((service_name, hour_bucket), start_time_ns, trace_id)
                ) WITH CLUSTERING ORDER BY (start_time_ns DESC, trace_id ASC)
                  AND default_time_to_live = 2592000;
                """

                self.session.execute(create_service_index)

            def write_span(self, span: dict):
                """
                Write span to Cassandra

                Writes to:
                1. Main spans table (by trace_id)
                2. Service index (for querying by service)
                """
                # Convert to bytes
                trace_id = bytes.fromhex(span['trace_id'])
                span_id = bytes.fromhex(span['span_id'])
                parent_span_id = bytes.fromhex(span['parent_span_id']) if span.get('parent_span_id') else None

                # Write to main table
                insert_span = """
                INSERT INTO spans (
                    trace_id, span_id, parent_span_id,
                    service_name, operation_name,
                    start_time_ns, duration_ns,
                    kind, status_code, tags, events
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                self.session.execute(insert_span, (
                    trace_id,
                    span_id,
                    parent_span_id,
                    span['service_name'],
                    span['operation_name'],
                    span['start_time_ns'],
                    span['duration_ns'],
                    span['kind'],
                    span['status_code'],
                    span.get('tags', {}),
                    span.get('events', [])
                ))

                # Write to service index (for root spans only)
                if not span.get('parent_span_id'):
                    hour_bucket = self._get_hour_bucket(span['start_time_ns'])

                    insert_service_index = """
                    INSERT INTO spans_by_service (
                        service_name, hour_bucket, start_time_ns,
                        trace_id, operation_name, duration_ns, status_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """

                    self.session.execute(insert_service_index, (
                        span['service_name'],
                        hour_bucket,
                        span['start_time_ns'],
                        trace_id,
                        span['operation_name'],
                        span['duration_ns'],
                        span['status_code']
                    ))

            def get_trace(self, trace_id: str) -> list:
                """
                Get all spans for a trace

                Query pattern: SELECT * FROM spans WHERE trace_id = ?

                Performance: O(1) partition lookup, sequential scan
                Latency: < 10ms (all spans co-located in single partition)
                """
                trace_id_bytes = bytes.fromhex(trace_id)

                query = "SELECT * FROM spans WHERE trace_id = ?"
                rows = self.session.execute(query, (trace_id_bytes,))

                spans = []
                for row in rows:
                    spans.append({
                        'trace_id': row.trace_id.hex(),
                        'span_id': row.span_id.hex(),
                        'parent_span_id': row.parent_span_id.hex() if row.parent_span_id else None,
                        'service_name': row.service_name,
                        'operation_name': row.operation_name,
                        'start_time_ns': row.start_time_ns,
                        'duration_ns': row.duration_ns,
                        'kind': row.kind,
                        'status_code': row.status_code,
                        'tags': dict(row.tags),
                        'events': list(row.events)
                    })

                return spans

            def query_traces_by_service(
                self,
                service_name: str,
                start_time: int,
                end_time: int,
                limit: int = 20
            ) -> list:
                """
                Query traces by service and time range

                Query pattern: Multi-partition query (one per hour bucket)
                Performance: O(num_hours) partition lookups
                """
                # Generate hour buckets
                hour_buckets = self._get_hour_buckets(start_time, end_time)

                traces = []

                for hour_bucket in hour_buckets:
                    query = """
                    SELECT trace_id, operation_name, start_time_ns, duration_ns, status_code
                    FROM spans_by_service
                    WHERE service_name = ? AND hour_bucket = ?
                      AND start_time_ns >= ? AND start_time_ns <= ?
                    LIMIT ?
                    """

                    rows = self.session.execute(query, (
                        service_name,
                        hour_bucket,
                        start_time,
                        end_time,
                        limit
                    ))

                    for row in rows:
                        traces.append({
                            'trace_id': row.trace_id.hex(),
                            'operation_name': row.operation_name,
                            'start_time_ns': row.start_time_ns,
                            'duration_ns': row.duration_ns,
                            'status_code': row.status_code
                        })

                # Sort and limit
                traces.sort(key=lambda t: t['start_time_ns'], reverse=True)
                return traces[:limit]

            def _get_hour_bucket(self, timestamp_ns: int) -> str:
                """Convert timestamp to hour bucket: YYYY-MM-DD-HH"""
                dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
                return dt.strftime('%Y-%m-%d-%H')

            def _get_hour_buckets(self, start_time_ns: int, end_time_ns: int) -> list:
                """Generate list of hour buckets for time range"""
                buckets = []

                current = datetime.fromtimestamp(start_time_ns / 1_000_000_000, tz=timezone.utc)
                end = datetime.fromtimestamp(end_time_ns / 1_000_000_000, tz=timezone.utc)

                while current <= end:
                    buckets.append(current.strftime('%Y-%m-%d-%H'))
                    current += timedelta(hours=1)

                return buckets


        class TagIndex:
            """
            Elasticsearch-based tag index for complex queries

            Use cases:
            - Search by tags: http.status_code=500
            - Full-text search: operation_name contains "payment"
            - Aggregations: count by service, p95 latency
            """

            def __init__(self, elasticsearch_client):
                self.es = elasticsearch_client

            def create_index(self):
                """Create Elasticsearch index"""
                index_config = {
                    "settings": {
                        "number_of_shards": 10,
                        "number_of_replicas": 2,
                        "index.lifecycle.name": "traces_policy",  # ILM for automatic deletion
                        "refresh_interval": "5s"  # Batch refreshes
                    },
                    "mappings": {
                        "properties": {
                            "trace_id": {"type": "keyword"},
                            "service_name": {"type": "keyword"},
                            "operation_name": {"type": "keyword"},
                            "start_time": {"type": "date"},
                            "duration_ms": {"type": "integer"},
                            "span_count": {"type": "integer"},
                            "error": {"type": "boolean"},
                            "status_code": {"type": "keyword"},
                            "tags": {"type": "object", "dynamic": true}  # Dynamic tags
                        }
                    }
                }

                self.es.indices.create(index="traces", body=index_config, ignore=400)

            def index_trace(self, trace: dict):
                """
                Index trace metadata for searching

                Only index root span metadata (not all spans)
                """
                doc = {
                    "trace_id": trace['trace_id'],
                    "service_name": trace['service_name'],
                    "operation_name": trace['operation_name'],
                    "start_time": trace['start_time'],
                    "duration_ms": trace['duration_ns'] / 1_000_000,
                    "span_count": trace['span_count'],
                    "error": trace['status_code'] == 'ERROR',
                    "status_code": trace['status_code'],
                    "tags": trace.get('tags', {})
                }

                self.es.index(index="traces", id=trace['trace_id'], body=doc)

            def search_traces(self, query: dict) -> list:
                """
                Search traces using Elasticsearch DSL

                Example query:
                {
                    "service_name": "order-service",
                    "tags.http.status_code": "500",
                    "duration_ms": {"gte": 1000}
                }
                """
                # Build Elasticsearch query
                es_query = {
                    "query": {
                        "bool": {
                            "must": []
                        }
                    },
                    "sort": [{"start_time": "desc"}],
                    "size": query.get('limit', 20)
                }

                # Add filters
                if 'service_name' in query:
                    es_query['query']['bool']['must'].append({
                        "term": {"service_name": query['service_name']}
                    })

                if 'operation_name' in query:
                    es_query['query']['bool']['must'].append({
                        "term": {"operation_name": query['operation_name']}
                    })

                if 'error' in query:
                    es_query['query']['bool']['must'].append({
                        "term": {"error": query['error']}
                    })

                if 'duration_ms' in query:
                    es_query['query']['bool']['must'].append({
                        "range": {"duration_ms": query['duration_ms']}
                    })

                if 'tags' in query:
                    for key, value in query['tags'].items():
                        es_query['query']['bool']['must'].append({
                            "term": {f"tags.{key}": value}
                        })

                # Execute search
                response = self.es.search(index="traces", body=es_query)

                traces = []
                for hit in response['hits']['hits']:
                    traces.append(hit['_source'])

                return traces
        ```

    === "🗺️ Service Dependency Graph"

        ## The Challenge

        **Problem:** Visualize service-to-service dependencies in real-time from traces.

        **Requirements:**

        - Build directed graph (service A calls service B)
        - Calculate edge metrics (call count, latency, error rate)
        - Update in near real-time (1-minute lag)
        - Support time-range filters (last 1h, last 24h)
        - Detect critical paths (slowest dependencies)

        ---

        ## Dependency Graph Builder

        ```python
        import networkx as nx
        from collections import defaultdict
        from dataclasses import dataclass
        from typing import Dict, List, Set

        @dataclass
        class ServiceEdge:
            """Represents a dependency edge between services"""
            parent_service: str
            child_service: str
            call_count: int
            total_duration_ns: int
            error_count: int
            p95_latency_ns: int
            p99_latency_ns: int

            @property
            def avg_latency_ns(self) -> float:
                return self.total_duration_ns / self.call_count if self.call_count > 0 else 0

            @property
            def error_rate(self) -> float:
                return self.error_count / self.call_count if self.call_count > 0 else 0


        class DependencyGraphBuilder:
            """
            Build service dependency graph from spans

            Algorithm:
            1. For each trace, extract parent-child service relationships
            2. Aggregate metrics per edge (call count, latency, errors)
            3. Build directed graph
            4. Calculate graph metrics (critical path, betweenness centrality)
            """

            def __init__(self, storage):
                self.storage = storage
                self.graph = nx.DiGraph()
                self.edges: Dict[tuple, ServiceEdge] = {}

            def build_graph(self, start_time: int, end_time: int):
                """
                Build dependency graph for time range

                Steps:
                1. Query all traces in time range
                2. Extract service dependencies from each trace
                3. Aggregate metrics per edge
                4. Build graph
                """
                # Reset graph
                self.graph.clear()
                self.edges.clear()

                # Get traces (use Elasticsearch for efficient querying)
                traces = self.storage.query_traces_by_time_range(start_time, end_time)

                # Process each trace
                for trace in traces:
                    self._process_trace(trace)

                # Build NetworkX graph
                for (parent, child), edge in self.edges.items():
                    self.graph.add_edge(
                        parent,
                        child,
                        call_count=edge.call_count,
                        avg_latency_ms=edge.avg_latency_ns / 1_000_000,
                        error_rate=edge.error_rate
                    )

            def _process_trace(self, trace_id: str):
                """
                Extract service dependencies from a single trace

                For each span:
                1. If parent exists, create edge (parent_service -> service)
                2. Aggregate metrics (duration, errors)
                """
                spans = self.storage.get_trace(trace_id)

                # Build span map (span_id -> span)
                span_map = {s['span_id']: s for s in spans}

                # Extract edges
                for span in spans:
                    if not span.get('parent_span_id'):
                        # Root span, no edge
                        continue

                    parent_span = span_map.get(span['parent_span_id'])
                    if not parent_span:
                        # Parent not found (incomplete trace)
                        continue

                    # Create edge
                    parent_service = parent_span['service_name']
                    child_service = span['service_name']

                    if parent_service == child_service:
                        # Intra-service span, skip
                        continue

                    # Update edge metrics
                    edge_key = (parent_service, child_service)

                    if edge_key not in self.edges:
                        self.edges[edge_key] = ServiceEdge(
                            parent_service=parent_service,
                            child_service=child_service,
                            call_count=0,
                            total_duration_ns=0,
                            error_count=0,
                            p95_latency_ns=0,
                            p99_latency_ns=0
                        )

                    edge = self.edges[edge_key]
                    edge.call_count += 1
                    edge.total_duration_ns += span['duration_ns']

                    if span['status_code'] == 'ERROR':
                        edge.error_count += 1

            def get_dependencies(self) -> List[dict]:
                """
                Get list of service dependencies with metrics

                Returns list of edges sorted by call count
                """
                dependencies = []

                for (parent, child), edge in self.edges.items():
                    dependencies.append({
                        'parent': parent,
                        'child': child,
                        'call_count': edge.call_count,
                        'avg_latency_ms': edge.avg_latency_ns / 1_000_000,
                        'error_rate': edge.error_rate,
                        'p95_latency_ms': edge.p95_latency_ns / 1_000_000,
                        'p99_latency_ms': edge.p99_latency_ns / 1_000_000
                    })

                # Sort by call count (most frequent first)
                dependencies.sort(key=lambda d: d['call_count'], reverse=True)

                return dependencies

            def find_critical_path(self, start_service: str, end_service: str) -> List[str]:
                """
                Find critical path (slowest path) between two services

                Use Dijkstra's algorithm with latency as weight
                """
                try:
                    # Use latency as edge weight
                    path = nx.dijkstra_path(
                        self.graph,
                        start_service,
                        end_service,
                        weight='avg_latency_ms'
                    )
                    return path
                except nx.NetworkXNoPath:
                    return []

            def get_upstream_services(self, service: str) -> Set[str]:
                """Get all services that call this service"""
                return set(self.graph.predecessors(service))

            def get_downstream_services(self, service: str) -> Set[str]:
                """Get all services that this service calls"""
                return set(self.graph.successors(service))

            def detect_cycles(self) -> List[List[str]]:
                """
                Detect circular dependencies (service A calls B calls A)

                Circular dependencies can cause:
                - Infinite loops
                - Cascading failures
                - Difficult debugging
                """
                try:
                    cycles = list(nx.simple_cycles(self.graph))
                    return cycles
                except:
                    return []

            def calculate_service_metrics(self) -> Dict[str, dict]:
                """
                Calculate per-service metrics

                Metrics:
                - Indegree: number of services calling this service
                - Outdegree: number of services this service calls
                - Betweenness centrality: how critical is this service
                - PageRank: overall importance
                """
                metrics = {}

                # Betweenness centrality (how often service is on shortest path)
                betweenness = nx.betweenness_centrality(self.graph)

                # PageRank (importance based on call graph)
                pagerank = nx.pagerank(self.graph)

                for service in self.graph.nodes():
                    metrics[service] = {
                        'upstream_count': len(list(self.graph.predecessors(service))),
                        'downstream_count': len(list(self.graph.successors(service))),
                        'betweenness_centrality': betweenness.get(service, 0),
                        'pagerank': pagerank.get(service, 0)
                    }

                return metrics
        ```

        ---

        ## Real-Time Dependency Updates

        ```python
        class RealtimeDependencyAggregator:
            """
            Aggregate service dependencies in real-time using stream processing

            Architecture:
            1. Consume spans from Kafka
            2. Extract service relationships
            3. Aggregate in time windows (1 minute)
            4. Publish to dependency graph service
            """

            def __init__(self, kafka_consumer, redis_client):
                self.kafka = kafka_consumer
                self.redis = redis_client
                self.window_size = 60  # 1 minute

            def run(self):
                """Main processing loop"""
                for message in self.kafka.consume('spans-topic'):
                    span = json.loads(message.value)
                    self._process_span(span)

            def _process_span(self, span: dict):
                """
                Process span and update dependency graph

                For client spans (outbound calls):
                - Extract parent service -> child service relationship
                - Update metrics in Redis (time-windowed)
                """
                if span['kind'] != 'CLIENT':
                    # Only process client spans (outbound calls)
                    return

                parent_service = span['service_name']

                # Infer child service from tags
                child_service = span['tags'].get('peer.service') or \
                               span['tags'].get('http.url', '').split('/')[2]  # Extract from URL

                if not child_service:
                    return

                # Calculate time window
                timestamp = span['start_time_ns'] / 1_000_000_000  # Convert to seconds
                window_start = int(timestamp / self.window_size) * self.window_size

                # Update Redis metrics
                edge_key = f"dep:{parent_service}:{child_service}:{window_start}"

                # Increment call count
                self.redis.hincrby(edge_key, 'call_count', 1)

                # Add duration to total
                self.redis.hincrby(edge_key, 'total_duration_ns', span['duration_ns'])

                # Increment error count
                if span['status_code'] == 'ERROR':
                    self.redis.hincrby(edge_key, 'error_count', 1)

                # Set expiry (retain for 1 hour)
                self.redis.expire(edge_key, 3600)

            def get_dependencies_from_redis(self, lookback_seconds: int = 3600) -> List[dict]:
                """
                Get aggregated dependencies from Redis

                Merge data from multiple time windows
                """
                current_time = int(time.time())
                start_window = current_time - lookback_seconds

                # Aggregate by edge
                edge_metrics = defaultdict(lambda: {
                    'call_count': 0,
                    'total_duration_ns': 0,
                    'error_count': 0
                })

                # Scan Redis keys
                for key in self.redis.scan_iter(match='dep:*'):
                    parts = key.split(':')
                    if len(parts) != 4:
                        continue

                    _, parent, child, window_start = parts
                    window_start = int(window_start)

                    if window_start < start_window:
                        # Outside lookback window
                        continue

                    # Get metrics
                    metrics = self.redis.hgetall(key)

                    edge_key = (parent, child)
                    edge_metrics[edge_key]['call_count'] += int(metrics.get('call_count', 0))
                    edge_metrics[edge_key]['total_duration_ns'] += int(metrics.get('total_duration_ns', 0))
                    edge_metrics[edge_key]['error_count'] += int(metrics.get('error_count', 0))

                # Convert to list
                dependencies = []
                for (parent, child), metrics in edge_metrics.items():
                    call_count = metrics['call_count']
                    if call_count == 0:
                        continue

                    avg_latency_ms = (metrics['total_duration_ns'] / call_count) / 1_000_000
                    error_rate = metrics['error_count'] / call_count

                    dependencies.append({
                        'parent': parent,
                        'child': child,
                        'call_count': call_count,
                        'avg_latency_ms': avg_latency_ms,
                        'error_rate': error_rate
                    })

                return dependencies
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Span Ingestion** | ✅ Yes | Kafka queue (buffer spikes), batch writes (1000 spans/batch) |
    | **Cassandra Writes** | ✅ Yes | Partition by trace_id (parallel writes), async commits |
    | **Elasticsearch Indexing** | ✅ Yes | Bulk indexing (refresh_interval: 5s), index lifecycle management |
    | **Query Performance** | 🟡 Moderate | Query cache (Redis), pre-computed aggregations, partition pruning |
    | **Context Propagation** | 🟢 No | Minimal overhead (< 1KB headers), W3C standard |

    ---

    ## Cluster Sizing (for 10B spans/day with 1% sampling)

    **Ingestion Layer:**
    ```
    Throughput: 100K spans/sec
    Per collector: 10K spans/sec (parsing, validation)
    Collectors needed: 100K / 10K = 10 collectors

    Instance: c5.xlarge (4 vCPU, 8 GB)
    Cost: 10 × $140/month = $1,400/month
    ```

    **Kafka Queue:**
    ```
    Throughput: 100K spans/sec × 350 bytes = 35 MB/sec
    Daily ingress: 35 MB/sec × 86,400 = 3 TB/day
    Retention: 1 day (buffer only)

    Brokers: 3 brokers (replication factor 3)
    Instance: m5.large (2 vCPU, 8 GB, 500 GB SSD)
    Cost: 3 × $150/month = $450/month
    ```

    **Cassandra Cluster:**
    ```
    Storage: 32 TB (with replication)
    Per node: 2 TB SSD
    Nodes: 32 TB / 2 TB = 16 nodes × 3 (replication) = 48 nodes

    Instance: i3.xlarge (4 vCPU, 30 GB, 950 GB NVMe)
    Cost: 48 × $190/month = $9,120/month
    ```

    **Elasticsearch Cluster:**
    ```
    Index size: 9 TB (metadata only)
    Per node: 1 TB SSD
    Nodes: 9 TB / 1 TB = 9 nodes × 2 (replica) = 18 nodes

    Instance: r5.xlarge (4 vCPU, 32 GB, 1 TB EBS)
    Cost: 18 × $250/month = $4,500/month
    ```

    **Total Cost:**
    ```
    Monthly: $1,400 (ingestion) + $450 (Kafka) + $9,120 (Cassandra) + $4,500 (ES) = $15,470/month
    Yearly: ~$186K/year

    Per span cost: $15,470 / (8.64B spans/month) = $0.0000018 per span
    ```

    ---

    ## Performance Optimizations

    ### 1. Async Span Export (Non-Blocking)

    ```python
    class AsyncSpanExporter:
        """
        Export spans asynchronously (non-blocking)

        Benefits:
        - No impact on application latency
        - Buffer spikes (handle bursts)
        - Batch exports (reduce network overhead)
        """

        def __init__(self, collector_endpoint: str):
            self.collector_endpoint = collector_endpoint
            self.buffer = []
            self.buffer_lock = threading.Lock()
            self.batch_size = 1000
            self.flush_interval = 5  # seconds

            # Start background thread
            self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self.flush_thread.start()

        def export_span(self, span: dict):
            """
            Export span (non-blocking)

            Adds span to buffer, returns immediately
            """
            with self.buffer_lock:
                self.buffer.append(span)

                # Flush if buffer full
                if len(self.buffer) >= self.batch_size:
                    self._flush()

        def _flush_loop(self):
            """Background thread to flush buffer periodically"""
            while True:
                time.sleep(self.flush_interval)
                self._flush()

        def _flush(self):
            """Flush buffer to collector"""
            with self.buffer_lock:
                if not self.buffer:
                    return

                spans_to_send = self.buffer.copy()
                self.buffer.clear()

            # Send batch to collector
            try:
                response = requests.post(
                    f"{self.collector_endpoint}/v1/traces",
                    json={"resource_spans": spans_to_send},
                    timeout=10
                )

                if response.status_code != 200:
                    logger.error(f"Failed to export spans: {response.status_code}")

            except Exception as e:
                logger.error(f"Failed to export spans: {e}")
                # TODO: Retry logic, dead letter queue
    ```

    ### 2. Query Cache

    ```python
    class QueryCache:
        """
        Cache trace queries in Redis

        Cache keys:
        - Trace by ID: trace:{trace_id}
        - Trace search: search:{query_hash}
        - Dependencies: deps:{lookback_hours}
        """

        def __init__(self, redis_client):
            self.redis = redis_client
            self.ttl = 300  # 5 minutes

        def get_trace(self, trace_id: str) -> Optional[dict]:
            """Get trace from cache"""
            key = f"trace:{trace_id}"
            data = self.redis.get(key)

            if data:
                return json.loads(data)

            return None

        def set_trace(self, trace_id: str, trace: dict):
            """Cache trace"""
            key = f"trace:{trace_id}"
            self.redis.setex(key, self.ttl, json.dumps(trace))

        def get_search_results(self, query: dict) -> Optional[list]:
            """Get cached search results"""
            query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()
            key = f"search:{query_hash}"

            data = self.redis.get(key)
            if data:
                return json.loads(data)

            return None

        def set_search_results(self, query: dict, results: list):
            """Cache search results"""
            query_hash = hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()
            key = f"search:{query_hash}"

            # Short TTL for search results (1 minute)
            self.redis.setex(key, 60, json.dumps(results))
    ```

    ### 3. Compression

    ```
    Span compression (before storage):

    - gzip: 5:1 compression ratio
    - Reduces storage: 3 TB/day → 600 GB/day
    - Trade-off: CPU overhead (compress on write, decompress on read)

    Columnar compression (Parquet for cold storage):

    - 10:1 compression ratio
    - Column-based layout (better compression for similar values)
    - Reduced cold storage: 5.4 TB → 540 GB
    ```

    ---

    ## Monitoring the Tracer

    ```python
    # Key metrics for tracing system health

    # Ingestion metrics
    spans_ingested_per_second
    ingestion_lag_seconds
    sampling_rate_actual
    dropped_spans_count

    # Storage metrics
    cassandra_write_latency_p99
    cassandra_disk_usage_percent
    elasticsearch_index_size_gb
    active_trace_count

    # Query metrics
    query_latency_p95
    query_cache_hit_rate
    trace_reconstruction_duration_ms
    elasticsearch_query_count

    # Sampling metrics
    head_sampled_traces_percent
    tail_sampled_traces_percent
    error_traces_sampled_percent
    slow_traces_sampled_percent
    ```

---

## Interview Tips

**Common Follow-Up Questions:**

1. **"How do you propagate trace context?"**
   - **W3C Trace Context standard:** traceparent header (trace_id, span_id, flags)
   - **Automatic injection/extraction:** OpenTelemetry SDK handles propagation
   - **Multi-protocol support:** HTTP, gRPC, Kafka, message queues
   - **Minimal overhead:** < 1KB per request
   - **Backward compatible:** Gracefully handle missing context

2. **"How do you sample traces?"**
   - **Head-based sampling:** Decide at ingestion (fast, predictable volume)
   - **Tail-based sampling:** Decide after seeing full trace (intelligent, preserves errors)
   - **Hybrid approach:** 10% head-based + tail-based for remaining
   - **Adaptive sampling:** Adjust rate based on traffic patterns
   - **Priority rules:** Always sample errors, slow requests, rare operations

3. **"How do you store spans efficiently?"**
   - **Cassandra for spans:** Partition by trace_id (fast trace reconstruction)
   - **Elasticsearch for tags:** Inverted index (fast tag search)
   - **Columnar storage:** Better compression (5-10x)
   - **TTL-based expiration:** Automatic cleanup (30 days)
   - **Cold storage (S3):** 90% cost savings for old data

4. **"How do you build service dependency graphs?"**
   - **Extract from spans:** Parent-child service relationships
   - **Real-time aggregation:** Stream processing (Kafka + Redis)
   - **Graph algorithms:** Critical path, betweenness centrality
   - **Time-windowed metrics:** Last 1h, last 24h
   - **Detect issues:** Circular dependencies, high latency edges

5. **"How do you handle high cardinality?"**
   - **Problem:** User IDs, session IDs in tags = millions of unique series
   - **Solution 1:** Sampling (reduce volume)
   - **Solution 2:** Tag validation (reject high-cardinality tags)
   - **Solution 3:** Separate storage (exemplars for high-cardinality)
   - **Solution 4:** Aggregate at query time (don't index all combinations)

6. **"How do you scale the system?"**
   - **Ingestion:** Horizontal scaling (add collectors), Kafka for buffering
   - **Storage:** Shard by trace_id (Cassandra), partition by hour (Elasticsearch)
   - **Query:** Query cache (Redis), pre-computed aggregations
   - **Tail sampling:** Distributed processing (Kafka Streams, Flink)
   - **Multi-region:** Replicate data for low latency queries

7. **"Distributed tracing vs. logging vs. metrics?"**
   - **Tracing:** Request flow, latency breakdown, service dependencies
   - **Logging:** Detailed events, error messages, debugging
   - **Metrics:** Aggregated time-series, alerting, capacity planning
   - **Use together:** Unified observability (traces link to logs and metrics)

**Key Points to Mention:**

- W3C Trace Context for propagation (industry standard)
- OpenTelemetry for instrumentation (vendor-neutral)
- Intelligent sampling to reduce volume (1% of traffic)
- Cassandra for span storage (time-series optimized)
- Elasticsearch for tag indexing (fast search)
- Service dependency graphs for visualizing call patterns
- Tail-based sampling to preserve interesting traces
- Query cache for performance (80% hit rate)
- Real-time aggregation for dependency graphs

---

## Real-World Examples

**Jaeger (Uber):**
- OpenTelemetry compatible
- Cassandra for storage
- Elasticsearch for tag indexing
- Adaptive sampling
- Open-source

**Zipkin (Twitter):**
- Simple architecture (single binary)
- MySQL, Cassandra, or Elasticsearch storage
- Push-based collection
- Probabilistic sampling
- Open-source

**Datadog APM:**
- SaaS platform
- Intelligent sampling (1%)
- 15-month retention
- Real-time service maps
- Integrated with metrics and logs

**AWS X-Ray:**
- Managed service
- Sampling rules (configurable)
- Integration with AWS services
- Trace analytics
- Pay-per-use pricing

---

## Summary

**System Characteristics:**

- **Ingestion:** 100K spans/sec (10B/day with 1% sampling)
- **Storage:** 32 TB (Cassandra + Elasticsearch), 30-day retention
- **Query Latency:** < 100ms p95
- **Availability:** 99.9% uptime
- **Sampling:** 1% of traffic (intelligent sampling)

**Core Components:**

1. **OpenTelemetry SDK:** Auto-instrumentation, context propagation
2. **Kafka Queue:** Buffer spikes, decouple ingestion from processing
3. **Tail Sampler:** Intelligent sampling after seeing full trace
4. **Cassandra:** Span storage (time-series optimized)
5. **Elasticsearch:** Tag indexing (fast search)
6. **Query Service:** Trace reconstruction, aggregations
7. **Dependency Graph:** Real-time service map

**Key Design Decisions:**

- W3C Trace Context for propagation (standard)
- Hybrid sampling (head-based + tail-based)
- Cassandra for spans (partition by trace_id)
- Elasticsearch for tag search (inverted index)
- Async span export (non-blocking)
- Query cache (Redis) for performance
- Real-time dependency aggregation (Kafka + Redis)
- Multi-tier storage (hot SSD + cold S3)

This design provides a scalable, cost-effective distributed tracing system capable of handling billions of spans per day with intelligent sampling, fast queries, and real-time service dependency visualization.
