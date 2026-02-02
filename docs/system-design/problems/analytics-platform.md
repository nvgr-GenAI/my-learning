# Design Analytics Platform (Google Analytics)

A web analytics platform that tracks user behavior across millions of websites, processes billions of events daily, and provides real-time insights through dashboards, funnels, and custom reports.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10B events/day, 100M websites tracked, 1M queries/day, 50M concurrent sessions |
| **Key Challenges** | Real-time aggregation, historical queries, funnel analysis, session tracking, data retention |
| **Core Concepts** | Event streaming, OLAP, pre-aggregation, lambda architecture, sessionization, time-series |
| **Companies** | Google Analytics, Mixpanel, Amplitude, Segment, Heap, Adobe Analytics |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Event Tracking** | Track pageviews, clicks, custom events from websites/apps | P0 (Must have) |
    | **Real-time Dashboard** | Show live visitor count, active pages, traffic sources | P0 (Must have) |
    | **Funnel Analysis** | Track user journey through conversion funnels | P0 (Must have) |
    | **User Segmentation** | Group users by behavior, demographics, properties | P0 (Must have) |
    | **Custom Reports** | Ad-hoc queries with filters, dimensions, metrics | P0 (Must have) |
    | **Session Tracking** | Group events into user sessions with timeout | P1 (Should have) |
    | **Retention Analysis** | Cohort analysis, user retention over time | P1 (Should have) |
    | **A/B Testing** | Compare metrics between experiment variants | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Machine learning predictions (churn prediction, LTV)
    - Heat maps and session replay
    - Ad campaign management
    - Data warehouse integration (ETL pipelines)
    - Custom alerting and anomaly detection

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Events must not be lost, analytics always accessible |
    | **Latency (Event Ingestion)** | < 100ms p95 | Fast tracking SDK, minimal impact on website performance |
    | **Latency (Query)** | < 5s p95 | Users expect fast dashboard loads, interactive exploration |
    | **Event Delivery** | 99.9% delivery | At-least-once delivery, idempotent processing |
    | **Data Retention** | 13 months raw, 2 years aggregated | Compliance with privacy laws, historical analysis |
    | **Real-time** | Events appear within 10 seconds | Real-time dashboard, immediate feedback |
    | **Consistency** | Eventual consistency | Brief delays acceptable (metrics may lag by seconds) |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Websites tracked: 100M
    Daily Active Users (DAU) across all sites: 5B

    Event ingestion:
    - Average events per user: 2 events/session
    - Average session duration: 5 minutes
    - Sessions per DAU: 1.5 sessions/day
    - Daily events: 5B × 1.5 × 2 = 15B events/day
    - Event QPS: 15B / 86,400 = ~173,600 events/sec
    - Peak QPS: 3x average = ~520,800 events/sec (peak hours)

    Active sessions tracking:
    - Concurrent sessions: 50M (1% of DAU)
    - Session updates: every 30 seconds
    - Session QPS: 50M / 30 = ~1.67M updates/sec

    Query traffic:
    - Active websites: 10M (10% check analytics daily)
    - Queries per website: 5-10 queries/day
    - Daily queries: 10M × 7 = 70M queries/day
    - Query QPS: 70M / 86,400 = ~810 queries/sec
    - Peak QPS: 5x average = ~4,050 queries/sec (business hours)

    Total Write QPS: ~175K (events + session updates)
    Total Read QPS: ~5K (dashboard queries + reports)
    Read/Write ratio: 1:35 (write-heavy system)
    ```

    ### Storage Estimates

    ```
    Event storage (raw):
    - Event payload: 2 KB (user_id, session_id, event_type, properties, timestamp)
    - Daily events: 15B × 2 KB = 30 TB/day
    - 13 months retention: 30 TB × 365 × 1.08 = 11,826 TB ≈ 11.5 PB

    Pre-aggregated metrics (OLAP):
    - Hourly aggregations: 24 hours × 100M sites × 1 KB = 2.4 TB/day
    - Daily aggregations: 100M sites × 10 KB = 1 TB/day
    - 2 years retention: (2.4 TB + 1 TB) × 730 = 2,482 TB ≈ 2.5 PB

    Session data (active):
    - 50M concurrent sessions × 5 KB = 250 GB (in-memory)
    - Session history (30 days): 50M sessions × 365/12 × 10 KB = 1.5 TB

    User profiles:
    - 5B unique users × 5 KB = 25 TB

    Funnel definitions:
    - 100M sites × 10 funnels × 2 KB = 2 TB

    Total storage: 11.5 PB (raw events) + 2.5 PB (aggregations) + 1.5 TB (sessions) + 25 TB (users) ≈ 14 PB
    ```

    ### Bandwidth Estimates

    ```
    Event ingress:
    - 173,600 events/sec × 2 KB = 347 MB/sec ≈ 2.8 Gbps
    - Peak: 520,800 events/sec × 2 KB = 1.04 GB/sec ≈ 8.3 Gbps

    Dashboard egress:
    - 810 queries/sec × 100 KB (avg response) = 81 MB/sec ≈ 650 Mbps
    - Peak: 4,050 queries/sec × 100 KB = 405 MB/sec ≈ 3.2 Gbps

    Total ingress: ~2.8 Gbps (peak: 8.3 Gbps)
    Total egress: ~650 Mbps (peak: 3.2 Gbps)
    ```

    ### Memory Estimates (Caching)

    ```
    Active sessions (in-memory):
    - 50M sessions × 5 KB = 250 GB

    Real-time aggregations (5-minute windows):
    - 100M sites × 5 minutes × 1 KB = 500 GB

    Query result cache:
    - Hot dashboards: 10M sites × 50 KB = 500 GB
    - TTL: 60 seconds

    User segment cache:
    - Common segments: 100M users × 1 KB = 100 GB

    Total cache: 250 GB + 500 GB + 500 GB + 100 GB ≈ 1.35 TB
    ```

    ---

    ## Key Assumptions

    1. Average event size: 2 KB (includes all properties and metadata)
    2. 5B daily active users across all tracked websites
    3. Write-heavy workload (35:1 write-to-read ratio)
    4. 13 months raw data retention (privacy compliance)
    5. Real-time is critical (< 10 seconds for events to appear)
    6. Most queries hit pre-aggregated data (90% cache hit rate)
    7. Session timeout: 30 minutes of inactivity
    8. Peak traffic: 3x average during business hours (varies by timezone)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Lambda architecture:** Real-time (speed layer) + batch processing (batch layer) for accuracy
    2. **Event streaming:** Kafka for reliable, high-throughput event ingestion
    3. **Pre-aggregation:** Compute common metrics in background, serve from cache
    4. **Time-series optimized:** Partition data by time for fast queries
    5. **Sessionization:** Track and group events into user sessions
    6. **OLAP cubes:** Multi-dimensional analysis with pre-computed aggregations

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Website[Website/App<br/>Analytics SDK]
            Dashboard[Dashboard UI<br/>Admin portal]
        end

        subgraph "Edge Layer"
            CDN[CDN<br/>SDK delivery]
            LB[Load Balancer<br/>Event ingestion]
            Query_LB[Load Balancer<br/>Query API]
        end

        subgraph "Ingestion Layer"
            Ingest_API[Ingestion Service<br/>Event validation]
            Beacon_API[Beacon Endpoint<br/>Lightweight tracking]
        end

        subgraph "Stream Processing (Speed Layer)"
            Kafka[Kafka<br/>Event streaming]
            Stream_Processor[Stream Processor<br/>Flink/Kafka Streams]
            Session_Tracker[Session Tracker<br/>Sessionization]
            Real_Time_Agg[Real-time Aggregator<br/>5-min windows]
        end

        subgraph "Batch Processing (Batch Layer)"
            Spark[Spark Jobs<br/>Daily aggregations]
            Funnel_Processor[Funnel Calculator<br/>Conversion rates]
            Retention_Calc[Retention Calculator<br/>Cohort analysis]
        end

        subgraph "Query Layer"
            Query_API[Query Service<br/>REST API]
            Dashboard_API[Dashboard Service<br/>Real-time metrics]
            Funnel_API[Funnel Service<br/>Conversion analysis]
            Segment_API[Segment Service<br/>User filtering]
        end

        subgraph "Caching"
            Redis_RT[Redis<br/>Real-time metrics]
            Redis_Query[Redis<br/>Query cache]
            Redis_Session[Redis<br/>Session data]
        end

        subgraph "Storage (OLAP)"
            ClickHouse[ClickHouse<br/>OLAP database<br/>Columnar storage]
            Druid[Apache Druid<br/>Time-series OLAP]
        end

        subgraph "Storage (Raw Events)"
            Kafka_Storage[Kafka<br/>Event buffer<br/>7 days]
            S3_Cold[S3<br/>Cold storage<br/>13 months]
        end

        subgraph "Metadata"
            Postgres[(PostgreSQL<br/>Sites, users<br/>funnels, segments)]
        end

        Website --> CDN
        Website --> LB
        Dashboard --> Query_LB

        LB --> Ingest_API
        LB --> Beacon_API

        Ingest_API --> Kafka
        Beacon_API --> Kafka

        Kafka --> Stream_Processor
        Kafka --> Session_Tracker
        Kafka --> Real_Time_Agg
        Kafka --> Kafka_Storage

        Stream_Processor --> ClickHouse
        Session_Tracker --> Redis_Session
        Real_Time_Agg --> Redis_RT

        Kafka_Storage --> Spark
        Kafka_Storage --> S3_Cold

        Spark --> ClickHouse
        Spark --> Funnel_Processor
        Spark --> Retention_Calc

        Funnel_Processor --> ClickHouse
        Retention_Calc --> ClickHouse

        Query_LB --> Query_API
        Query_LB --> Dashboard_API
        Query_LB --> Funnel_API
        Query_LB --> Segment_API

        Query_API --> Redis_Query
        Query_API --> ClickHouse
        Query_API --> Druid

        Dashboard_API --> Redis_RT
        Dashboard_API --> ClickHouse

        Funnel_API --> ClickHouse
        Segment_API --> ClickHouse

        Query_API --> Postgres
        Dashboard_API --> Postgres

        style CDN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Query_LB fill:#e1f5ff
        style Redis_RT fill:#fff4e1
        style Redis_Query fill:#fff4e1
        style Redis_Session fill:#fff4e1
        style ClickHouse fill:#ffe1e1
        style Druid fill:#ffe1e1
        style Postgres fill:#e8eaf6
        style Kafka fill:#e8eaf6
        style S3_Cold fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Kafka** | High-throughput event ingestion (175K QPS), reliable delivery, replay capability | Kinesis (vendor lock-in), direct DB writes (no buffering, can't handle spikes) |
    | **ClickHouse** | Columnar OLAP database, 100x faster than row-based for analytics, excellent compression | PostgreSQL (too slow for billions of events), Redshift (expensive, slower) |
    | **Apache Druid** | Time-series OLAP, sub-second queries, real-time ingestion | Elasticsearch (not optimized for aggregations), TimescaleDB (slower at scale) |
    | **Flink/Kafka Streams** | Real-time stream processing, stateful operations (sessions), exactly-once semantics | Spark Streaming (higher latency), custom code (complex state management) |
    | **Redis** | Real-time metrics cache (<10ms reads), session tracking | No cache (OLAP DB can't handle 5K QPS), Memcached (no data structures) |
    | **S3** | Cost-effective cold storage, infinite scalability, data lake | HDFS (complex operations), keep in DB (expensive) |

    **Key Trade-off:** We chose **lambda architecture** (speed + batch layers) for both real-time visibility and accurate historical analysis. Real-time may have minor inaccuracies, batch layer provides source of truth.

    ---

    ## API Design

    ### 1. Track Event

    **Request:**
    ```http
    POST /api/v1/events
    Content-Type: application/json

    {
      "tracking_id": "UA-123456-1",
      "client_id": "abc123.def456",
      "session_id": "sess_xyz789",
      "event_type": "pageview",
      "timestamp": "2026-02-02T10:30:00.000Z",
      "page": {
        "url": "https://example.com/products/shoes",
        "title": "Running Shoes - Example Store",
        "referrer": "https://google.com/search?q=running+shoes"
      },
      "user": {
        "user_id": "user_12345",
        "country": "US",
        "device": "desktop",
        "browser": "Chrome"
      },
      "custom_properties": {
        "product_id": "prod_789",
        "category": "footwear",
        "price": 99.99
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 204 No Content
    ```

    **Design Notes:**

    - Return immediately (fire-and-forget, don't wait for processing)
    - Support batch ingestion (up to 500 events per request)
    - SDK handles retries with exponential backoff
    - Deduplicate events by (client_id, session_id, timestamp, event_type)
    - Rate limit: 1000 events/sec per tracking_id

    ---

    ### 2. Get Real-time Dashboard

    **Request:**
    ```http
    GET /api/v1/realtime?tracking_id=UA-123456-1
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "tracking_id": "UA-123456-1",
      "timestamp": "2026-02-02T10:30:00.000Z",
      "active_users": 1234,
      "active_sessions": 890,
      "events_per_minute": 5670,
      "top_pages": [
        {
          "url": "/products/shoes",
          "active_users": 234,
          "pageviews": 456
        },
        {
          "url": "/checkout",
          "active_users": 123,
          "pageviews": 189
        }
      ],
      "traffic_sources": {
        "direct": 450,
        "organic_search": 320,
        "social": 180,
        "referral": 150
      },
      "devices": {
        "desktop": 670,
        "mobile": 520,
        "tablet": 100
      }
    }
    ```

    **Design Notes:**

    - Served from Redis (real-time aggregations)
    - Updated every 5 seconds via stream processing
    - Active user = user with event in last 5 minutes
    - Auto-refresh dashboard every 10 seconds

    ---

    ### 3. Query Custom Report

    **Request:**
    ```http
    POST /api/v1/query
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "tracking_id": "UA-123456-1",
      "date_range": {
        "start": "2026-01-01",
        "end": "2026-01-31"
      },
      "metrics": ["pageviews", "unique_users", "bounce_rate"],
      "dimensions": ["date", "country", "device"],
      "filters": {
        "device": ["desktop", "mobile"],
        "country": "US"
      },
      "order_by": [{"field": "pageviews", "direction": "desc"}],
      "limit": 100
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "query_id": "q_abc123",
      "execution_time_ms": 234,
      "rows": [
        {
          "date": "2026-01-01",
          "country": "US",
          "device": "desktop",
          "pageviews": 123456,
          "unique_users": 45678,
          "bounce_rate": 0.42
        },
        // ... more rows
      ],
      "total_rows": 92,
      "is_sampled": false
    }
    ```

    **Design Notes:**

    - Check cache first (Redis, TTL: 60 seconds)
    - Query pre-aggregated OLAP tables when possible
    - Fall back to raw events for custom dimensions
    - Sample data (10%) if query spans > 90 days and no pre-aggregation

    ---

    ### 4. Create Funnel

    **Request:**
    ```http
    POST /api/v1/funnels
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "tracking_id": "UA-123456-1",
      "name": "Purchase Funnel",
      "steps": [
        {
          "name": "View Product",
          "event_type": "pageview",
          "filters": {"page_path": "/products/*"}
        },
        {
          "name": "Add to Cart",
          "event_type": "click",
          "filters": {"element_id": "add-to-cart"}
        },
        {
          "name": "Checkout",
          "event_type": "pageview",
          "filters": {"page_path": "/checkout"}
        },
        {
          "name": "Purchase",
          "event_type": "purchase"
        }
      ],
      "window": "7d"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "funnel_id": "funnel_xyz789",
      "name": "Purchase Funnel",
      "steps": [...],
      "window": "7d",
      "created_at": "2026-02-02T10:30:00.000Z"
    }
    ```

    ---

    ### 5. Get Funnel Analysis

    **Request:**
    ```http
    GET /api/v1/funnels/funnel_xyz789/analysis?start_date=2026-01-01&end_date=2026-01-31
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "funnel_id": "funnel_xyz789",
      "date_range": {
        "start": "2026-01-01",
        "end": "2026-01-31"
      },
      "total_users": 100000,
      "steps": [
        {
          "name": "View Product",
          "users": 100000,
          "conversion_rate": 1.0,
          "drop_off_rate": 0.0
        },
        {
          "name": "Add to Cart",
          "users": 45000,
          "conversion_rate": 0.45,
          "drop_off_rate": 0.55
        },
        {
          "name": "Checkout",
          "users": 15000,
          "conversion_rate": 0.33,
          "drop_off_rate": 0.67
        },
        {
          "name": "Purchase",
          "users": 8000,
          "conversion_rate": 0.53,
          "drop_off_rate": 0.47
        }
      ],
      "overall_conversion_rate": 0.08,
      "avg_time_to_convert": "2h 34m"
    }
    ```

    **Design Notes:**

    - Pre-compute funnel metrics daily (batch job)
    - Cache results for 5 minutes
    - Support breakdown by dimensions (device, country, etc.)

    ---

    ## Database Schema

    ### Events (ClickHouse - Columnar OLAP)

    ```sql
    -- Raw events table (partitioned by day)
    CREATE TABLE events (
        tracking_id String,
        event_id String,
        client_id String,
        session_id String,
        user_id Nullable(String),
        event_type String,
        event_timestamp DateTime,
        page_url Nullable(String),
        page_title Nullable(String),
        referrer Nullable(String),
        utm_source Nullable(String),
        utm_medium Nullable(String),
        utm_campaign Nullable(String),
        country String,
        city String,
        device String,
        browser String,
        os String,
        custom_properties String,  -- JSON
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMMDD(event_timestamp)
    ORDER BY (tracking_id, event_timestamp, client_id)
    TTL event_timestamp + INTERVAL 13 MONTH;

    -- Hourly aggregations (for fast queries)
    CREATE MATERIALIZED VIEW events_hourly
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMMDD(hour)
    ORDER BY (tracking_id, hour, event_type, country, device)
    AS SELECT
        tracking_id,
        toStartOfHour(event_timestamp) as hour,
        event_type,
        country,
        device,
        count() as event_count,
        uniqExact(client_id) as unique_users,
        uniqExact(session_id) as sessions
    FROM events
    GROUP BY tracking_id, hour, event_type, country, device;

    -- Daily aggregations (for historical analysis)
    CREATE MATERIALIZED VIEW events_daily
    ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(day)
    ORDER BY (tracking_id, day, country, device)
    AS SELECT
        tracking_id,
        toDate(event_timestamp) as day,
        country,
        device,
        count() as pageviews,
        uniqExact(client_id) as unique_users,
        uniqExact(session_id) as sessions,
        countIf(session_id IN (
            SELECT session_id FROM events e2
            WHERE e2.tracking_id = events.tracking_id
            GROUP BY session_id HAVING count() = 1
        )) as bounces
    FROM events
    WHERE event_type = 'pageview'
    GROUP BY tracking_id, day, country, device;

    -- Sessions table (aggregated from events)
    CREATE TABLE sessions (
        tracking_id String,
        session_id String,
        client_id String,
        user_id Nullable(String),
        session_start DateTime,
        session_end DateTime,
        session_duration UInt32,
        event_count UInt32,
        pageview_count UInt32,
        entry_page String,
        exit_page String,
        referrer Nullable(String),
        utm_source Nullable(String),
        utm_medium Nullable(String),
        country String,
        device String,
        browser String,
        converted Boolean DEFAULT false
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMMDD(session_start)
    ORDER BY (tracking_id, session_start, client_id)
    TTL session_start + INTERVAL 13 MONTH;

    -- Funnel results (pre-computed)
    CREATE TABLE funnel_results (
        tracking_id String,
        funnel_id String,
        date Date,
        step_index UInt8,
        step_name String,
        user_count UInt64,
        conversion_rate Float64,
        avg_time_to_next_step UInt32
    ) ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (tracking_id, funnel_id, date, step_index);
    ```

    **Why ClickHouse:**

    - **Columnar storage:** 10-100x compression, only read needed columns
    - **Fast aggregations:** Optimized for GROUP BY, COUNT, SUM queries
    - **Parallel processing:** Utilizes all CPU cores
    - **Materialized views:** Pre-compute common aggregations automatically
    - **TTL support:** Automatic data expiration after 13 months

    ---

    ### Metadata (PostgreSQL)

    ```sql
    -- Website tracking accounts
    CREATE TABLE tracking_accounts (
        tracking_id VARCHAR(50) PRIMARY KEY,
        owner_user_id BIGINT NOT NULL,
        website_url VARCHAR(255) NOT NULL,
        website_name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_owner (owner_user_id)
    );

    -- Funnel definitions
    CREATE TABLE funnels (
        funnel_id VARCHAR(50) PRIMARY KEY,
        tracking_id VARCHAR(50) NOT NULL,
        name VARCHAR(255) NOT NULL,
        steps JSONB NOT NULL,  -- Array of step definitions
        window_days INT DEFAULT 7,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (tracking_id) REFERENCES tracking_accounts(tracking_id),
        INDEX idx_tracking (tracking_id)
    );

    -- User segments
    CREATE TABLE segments (
        segment_id VARCHAR(50) PRIMARY KEY,
        tracking_id VARCHAR(50) NOT NULL,
        name VARCHAR(255) NOT NULL,
        filters JSONB NOT NULL,  -- Complex filter conditions
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (tracking_id) REFERENCES tracking_accounts(tracking_id),
        INDEX idx_tracking (tracking_id)
    );

    -- Custom dashboards
    CREATE TABLE dashboards (
        dashboard_id VARCHAR(50) PRIMARY KEY,
        tracking_id VARCHAR(50) NOT NULL,
        name VARCHAR(255) NOT NULL,
        widgets JSONB NOT NULL,  -- Array of widget configurations
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (tracking_id) REFERENCES tracking_accounts(tracking_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Event Ingestion Flow

    ```mermaid
    sequenceDiagram
        participant Website
        participant SDK
        participant LB
        participant Ingest_API
        participant Kafka
        participant Stream_Processor
        participant ClickHouse
        participant Redis

        Website->>SDK: User action (pageview, click)
        SDK->>SDK: Queue event locally

        alt Batch send (every 10s or 500 events)
            SDK->>LB: POST /api/v1/events (batch)
            LB->>Ingest_API: Forward events
            Ingest_API->>Ingest_API: Validate, enrich (IP->country)
            Ingest_API->>Kafka: Publish events (async)
            Ingest_API-->>SDK: 204 No Content (< 50ms)
        else Beacon (page unload)
            SDK->>LB: POST /api/v1/beacon (1px GIF)
            LB->>Ingest_API: Forward final events
            Ingest_API->>Kafka: Publish events
            Ingest_API-->>SDK: 204 No Content
        end

        Kafka->>Stream_Processor: Consume events (micro-batches)

        par Real-time processing
            Stream_Processor->>Stream_Processor: Sessionization (30-min window)
            Stream_Processor->>Redis: Update active user count
            Stream_Processor->>Redis: Update real-time metrics
        and Persist to OLAP
            Stream_Processor->>ClickHouse: Bulk insert (every 5s)
        end

        Note over ClickHouse: Materialized views<br/>auto-update aggregations
    ```

    **Flow Explanation:**

    1. **SDK batching** - Collect events locally, send in batches (reduce requests)
    2. **Fast response** - Return immediately after Kafka publish (< 50ms)
    3. **Stream processing** - Flink processes events in micro-batches (5s windows)
    4. **Sessionization** - Group events by user with 30-minute timeout
    5. **Real-time metrics** - Update Redis for dashboard (< 10s latency)
    6. **OLAP storage** - Bulk insert to ClickHouse, materialized views update automatically

    ---

    ### Query Execution Flow

    ```mermaid
    sequenceDiagram
        participant Dashboard
        participant Query_API
        participant Redis
        participant ClickHouse
        participant Druid

        Dashboard->>Query_API: GET /api/v1/query
        Query_API->>Query_API: Parse query, validate

        alt Simple metric (last 24h)
            Query_API->>Redis: GET cached result
            alt Cache HIT (90% of queries)
                Redis-->>Query_API: Cached metrics
                Query_API-->>Dashboard: 200 OK (< 20ms)
            else Cache MISS
                Redis-->>Query_API: null
                Query_API->>Druid: Query real-time OLAP
                Druid-->>Query_API: Aggregated results
                Query_API->>Redis: SET cache (TTL: 60s)
                Query_API-->>Dashboard: 200 OK (< 500ms)
            end
        else Historical report (last 30 days)
            Query_API->>ClickHouse: Query pre-aggregated table
            ClickHouse-->>Query_API: Results from materialized view
            Query_API->>Redis: SET cache (TTL: 300s)
            Query_API-->>Dashboard: 200 OK (< 2s)
        else Complex custom query (90 days, many dimensions)
            Query_API->>Query_API: Check if sampling needed
            alt Large dataset (>1B events)
                Query_API->>ClickHouse: Query with SAMPLE 0.1 (10%)
                ClickHouse-->>Query_API: Sampled results
                Query_API-->>Dashboard: 200 OK + is_sampled=true (< 5s)
            else Manageable dataset
                Query_API->>ClickHouse: Query raw events
                ClickHouse-->>Query_API: Full results
                Query_API-->>Dashboard: 200 OK (< 5s)
            end
        end
    ```

    **Flow Explanation:**

    1. **Cache first** - Check Redis for cached results (90% hit rate)
    2. **Pre-aggregated tables** - Use materialized views for common queries
    3. **Sampling** - For large datasets, sample 10% of events (5-10x faster)
    4. **Real-time vs historical** - Druid for last 24h, ClickHouse for history
    5. **Query optimization** - Push down filters, only select needed columns

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical analytics platform subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Lambda Architecture** | How to provide both real-time and accurate historical analytics? | Speed layer (Flink) + batch layer (Spark) + serving layer (Redis + ClickHouse) |
    | **Sessionization** | How to group events into user sessions in real-time? | Stateful stream processing with session windows (30-min timeout) |
    | **Pre-aggregation** | How to make queries fast on billions of events? | Materialized views, OLAP cubes, dimensionality reduction |
    | **Funnel Analysis** | How to calculate conversion funnels efficiently? | Pre-compute with batch jobs, store intermediate results |

    ---

    === "🏛️ Lambda Architecture"

        ## The Challenge

        **Problem:** Provide both real-time insights (< 10s latency) AND accurate historical analysis on 10B+ events/day.

        **Conflicting requirements:**

        - **Real-time:** Users want live dashboard updates (active users, current traffic)
        - **Accuracy:** Historical reports must be 100% accurate (no missing events)
        - **Scale:** Can't run complex queries on raw events (too slow)

        **Naive approach:** Query raw events in real-time. **Doesn't scale** (10B events/day = 23 TB, queries take minutes).

        ---

        ## Lambda Architecture Components

        **Three layers:**

        1. **Speed Layer (Real-time):** Process events immediately, approximate results
        2. **Batch Layer (Historical):** Process all events daily, accurate results
        3. **Serving Layer (Query):** Merge real-time + batch, serve to users

        ```
        Events → Speed Layer (Flink)     → Redis (real-time metrics)
              ↓                              ↓
              → Batch Layer (Spark)      → ClickHouse (accurate aggregations)
                                             ↓
                                          Query API (merge + serve)
        ```

        ---

        ## Speed Layer Implementation (Flink)

        ```python
        from pyflink.datastream import StreamExecutionEnvironment
        from pyflink.datastream.functions import MapFunction, ReduceFunction
        from pyflink.datastream.window import TumblingEventTimeWindows
        from datetime import timedelta

        class EventCounter(MapFunction):
            """Map event to (tracking_id, count) tuple"""
            def map(self, event):
                return (event['tracking_id'], 1)

        class SumCounts(ReduceFunction):
            """Reduce counts by summing"""
            def reduce(self, count1, count2):
                return (count1[0], count1[1] + count2[1])

        def speed_layer_pipeline():
            """
            Real-time event processing with Flink

            - Consume from Kafka
            - Window by 5-second tumbling windows
            - Aggregate metrics (event count, unique users)
            - Write to Redis for real-time dashboard
            """
            env = StreamExecutionEnvironment.get_execution_environment()
            env.set_parallelism(32)  # Scale across 32 cores

            # Kafka source
            events = env.add_source(
                FlinkKafkaConsumer(
                    topics=['events'],
                    deserialization_schema=JSONKeyValueDeserializationSchema(),
                    properties={
                        'bootstrap.servers': 'kafka:9092',
                        'group.id': 'speed-layer'
                    }
                )
            )

            # Extract event time for windowing
            events = events.assign_timestamps_and_watermarks(
                WatermarkStrategy
                    .for_bounded_out_of_orderness(timedelta(seconds=5))
                    .with_timestamp_assigner(lambda event, ts: event['event_timestamp'])
            )

            # Real-time aggregations (5-second windows)
            event_counts = (events
                .key_by(lambda event: event['tracking_id'])
                .window(TumblingEventTimeWindows.of(timedelta(seconds=5)))
                .reduce(
                    lambda acc, event: {
                        'tracking_id': event['tracking_id'],
                        'window_start': acc.get('window_start', event['event_timestamp']),
                        'event_count': acc.get('event_count', 0) + 1,
                        'unique_users': acc.get('unique_users', set()).union({event['client_id']})
                    }
                )
            )

            # Write to Redis (real-time metrics)
            event_counts.add_sink(RedisSink({
                'host': 'redis-host',
                'port': 6379,
                'key_template': 'rt:metrics:{tracking_id}',
                'ttl': 300  # 5-minute expiration
            }))

            # Execute pipeline
            env.execute("Speed Layer - Real-time Aggregations")

        # Sessionization (stateful processing)
        def sessionization_pipeline():
            """
            Group events into sessions using session windows

            Session = sequence of events with < 30 min gaps
            """
            env = StreamExecutionEnvironment.get_execution_environment()

            events = env.add_source(kafka_source)

            # Session window (30-minute inactivity timeout)
            sessions = (events
                .key_by(lambda e: (e['tracking_id'], e['client_id']))
                .window(EventTimeSessionWindows.with_gap(timedelta(minutes=30)))
                .process(SessionAggregator())
            )

            sessions.add_sink(kafka_sink('sessions'))
            env.execute("Sessionization Pipeline")

        class SessionAggregator(ProcessWindowFunction):
            """Aggregate events into session object"""

            def process(self, key, context, events):
                tracking_id, client_id = key
                event_list = list(events)
                event_list.sort(key=lambda e: e['event_timestamp'])

                session = {
                    'session_id': f"{client_id}_{context.window().start}",
                    'tracking_id': tracking_id,
                    'client_id': client_id,
                    'session_start': event_list[0]['event_timestamp'],
                    'session_end': event_list[-1]['event_timestamp'],
                    'duration': (event_list[-1]['event_timestamp'] - event_list[0]['event_timestamp']).seconds,
                    'event_count': len(event_list),
                    'pageviews': [e for e in event_list if e['event_type'] == 'pageview'],
                    'entry_page': event_list[0].get('page_url'),
                    'exit_page': event_list[-1].get('page_url'),
                    'converted': any(e['event_type'] == 'purchase' for e in event_list)
                }

                yield session
        ```

        ---

        ## Batch Layer Implementation (Spark)

        ```python
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, count, countDistinct, sum, avg, when
        from pyspark.sql.window import Window

        def batch_layer_daily_job():
            """
            Daily batch job for accurate aggregations

            - Read all events from previous day
            - Compute accurate metrics (no approximations)
            - Write to ClickHouse OLAP database
            - Backfill any missing data from speed layer
            """
            spark = SparkSession.builder \
                .appName("Batch Layer - Daily Aggregations") \
                .config("spark.executor.memory", "16g") \
                .config("spark.executor.cores", "4") \
                .getOrCreate()

            # Read yesterday's events from S3/Kafka
            yesterday = date.today() - timedelta(days=1)
            events = spark.read.parquet(f"s3://analytics-events/{yesterday.strftime('%Y/%m/%d')}/*")

            # Daily aggregations by tracking_id, country, device
            daily_stats = events.groupBy(
                'tracking_id',
                col('event_timestamp').cast('date').alias('date'),
                'country',
                'device'
            ).agg(
                count('*').alias('total_events'),
                countDistinct('client_id').alias('unique_users'),
                countDistinct('session_id').alias('sessions'),
                count(when(col('event_type') == 'pageview', 1)).alias('pageviews'),
                count(when(col('event_type') == 'purchase', 1)).alias('purchases')
            )

            # Calculate bounce rate (sessions with only 1 pageview)
            session_pageviews = events \
                .filter(col('event_type') == 'pageview') \
                .groupBy('tracking_id', 'session_id') \
                .agg(count('*').alias('pageview_count'))

            bounce_rate = session_pageviews \
                .groupBy('tracking_id') \
                .agg(
                    count('*').alias('total_sessions'),
                    count(when(col('pageview_count') == 1, 1)).alias('bounced_sessions')
                ) \
                .withColumn('bounce_rate', col('bounced_sessions') / col('total_sessions'))

            # Join and write to ClickHouse
            final_stats = daily_stats.join(bounce_rate, on='tracking_id', how='left')

            final_stats.write \
                .format("jdbc") \
                .option("url", "jdbc:clickhouse://clickhouse:8123/analytics") \
                .option("dbtable", "events_daily") \
                .option("batchsize", 100000) \
                .mode("append") \
                .save()

            spark.stop()

        def batch_layer_funnel_job():
            """
            Compute funnel conversions (batch job)

            - Read funnel definitions from PostgreSQL
            - For each funnel, calculate conversion rates
            - Store results in ClickHouse
            """
            spark = SparkSession.builder.appName("Funnel Calculations").getOrCreate()

            # Load funnel definitions
            funnels = spark.read \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://postgres:5432/metadata") \
                .option("dbtable", "funnels") \
                .load()

            # Load yesterday's events
            yesterday = date.today() - timedelta(days=1)
            events = spark.read.parquet(f"s3://analytics-events/{yesterday.strftime('%Y/%m/%d')}/*")

            # For each funnel, calculate conversions
            for funnel_row in funnels.collect():
                funnel_id = funnel_row['funnel_id']
                tracking_id = funnel_row['tracking_id']
                steps = json.loads(funnel_row['steps'])
                window_days = funnel_row['window_days']

                # Filter events for this tracking_id
                funnel_events = events.filter(col('tracking_id') == tracking_id)

                # Calculate funnel (SQL-like approach)
                funnel_results = calculate_funnel(funnel_events, steps, window_days)

                # Write results to ClickHouse
                funnel_results.write \
                    .format("jdbc") \
                    .option("url", "jdbc:clickhouse://clickhouse:8123/analytics") \
                    .option("dbtable", "funnel_results") \
                    .mode("append") \
                    .save()

            spark.stop()

        def calculate_funnel(events, steps, window_days):
            """
            Calculate funnel conversion rates

            Args:
                events: Spark DataFrame of events
                steps: List of funnel step definitions
                window_days: Time window for conversion (e.g., 7 days)

            Returns:
                DataFrame with conversion rates per step
            """
            # Step 1: Filter events matching first step
            step1_events = events.filter(
                (col('event_type') == steps[0]['event_type']) &
                # Apply step filters...
            )

            step1_users = step1_events.select('client_id').distinct()

            # For each subsequent step, find users who completed it
            results = []
            previous_users = step1_users

            for i, step in enumerate(steps):
                step_events = events.filter(
                    # Match step criteria...
                )

                # Users who completed this step (within window)
                step_users = step_events.join(
                    previous_users,
                    on='client_id'
                ).select('client_id').distinct()

                conversion_rate = step_users.count() / previous_users.count() if previous_users.count() > 0 else 0

                results.append({
                    'step_index': i,
                    'step_name': step['name'],
                    'user_count': step_users.count(),
                    'conversion_rate': conversion_rate
                })

                previous_users = step_users

            return spark.createDataFrame(results)
        ```

        ---

        ## Serving Layer (Query API)

        ```python
        from fastapi import FastAPI, HTTPException
        from redis import Redis
        from clickhouse_driver import Client

        app = FastAPI()
        redis_client = Redis(host='redis', port=6379)
        clickhouse_client = Client(host='clickhouse')

        @app.get("/api/v1/realtime")
        async def get_realtime_metrics(tracking_id: str):
            """
            Serve real-time metrics from speed layer

            - Read from Redis (updated every 5s by Flink)
            - If cache miss, fall back to Druid
            """
            cache_key = f"rt:metrics:{tracking_id}"
            cached = redis_client.get(cache_key)

            if cached:
                return json.loads(cached)

            # Cache miss - query Druid (real-time OLAP)
            query = f"""
                SELECT
                    COUNT(*) as active_users,
                    COUNT(DISTINCT session_id) as active_sessions
                FROM events
                WHERE tracking_id = '{tracking_id}'
                    AND event_timestamp > NOW() - INTERVAL 5 MINUTE
            """

            result = druid_client.query(query)

            # Cache for 10 seconds
            redis_client.setex(cache_key, 10, json.dumps(result))

            return result

        @app.post("/api/v1/query")
        async def execute_custom_query(query_request: QueryRequest):
            """
            Execute custom analytics query

            - Check cache first (Redis)
            - Route to appropriate data source:
                - Real-time (< 24h): Druid
                - Historical: ClickHouse pre-aggregated tables
                - Custom: ClickHouse raw events (with sampling)
            """
            # Generate cache key from query
            cache_key = f"query:{hash(json.dumps(query_request.dict()))}"
            cached = redis_client.get(cache_key)

            if cached:
                return json.loads(cached)

            # Determine query route
            date_range_days = (query_request.date_range.end - query_request.date_range.start).days

            if date_range_days <= 1:
                # Real-time query (Druid)
                result = execute_druid_query(query_request)
                ttl = 60  # Cache for 1 minute
            elif can_use_preaggregated(query_request):
                # Use materialized view (fast)
                result = execute_clickhouse_aggregated_query(query_request)
                ttl = 300  # Cache for 5 minutes
            else:
                # Query raw events (slow, may need sampling)
                result = execute_clickhouse_raw_query(query_request)
                ttl = 600  # Cache for 10 minutes

            # Cache result
            redis_client.setex(cache_key, ttl, json.dumps(result))

            return result

        def can_use_preaggregated(query_request):
            """
            Check if query can be satisfied by pre-aggregated tables

            Pre-aggregated dimensions: date, hour, country, device, event_type
            """
            allowed_dimensions = {'date', 'hour', 'country', 'device', 'event_type'}
            requested_dimensions = set(query_request.dimensions)

            return requested_dimensions.issubset(allowed_dimensions)

        def execute_clickhouse_aggregated_query(query_request):
            """Query pre-aggregated materialized view"""
            query = f"""
                SELECT
                    {', '.join(query_request.dimensions)},
                    SUM(event_count) as events,
                    SUM(unique_users) as users
                FROM events_hourly
                WHERE tracking_id = '{query_request.tracking_id}'
                    AND hour BETWEEN '{query_request.date_range.start}'
                    AND '{query_request.date_range.end}'
                GROUP BY {', '.join(query_request.dimensions)}
                ORDER BY events DESC
                LIMIT {query_request.limit}
            """

            return clickhouse_client.execute(query)

        def execute_clickhouse_raw_query(query_request):
            """
            Query raw events table

            Apply sampling if dataset is large (>1B events)
            """
            # Estimate query size
            estimated_events = estimate_event_count(query_request)

            # Apply sampling for large datasets
            sample_rate = 1.0
            if estimated_events > 1_000_000_000:
                sample_rate = 0.1  # Sample 10%

            query = f"""
                SELECT
                    {', '.join(query_request.dimensions)},
                    COUNT(*) * {1/sample_rate} as events,
                    uniqExact(client_id) * {1/sample_rate} as users
                FROM events
                SAMPLE {sample_rate}
                WHERE tracking_id = '{query_request.tracking_id}'
                    AND event_timestamp BETWEEN '{query_request.date_range.start}'
                    AND '{query_request.date_range.end}'
                GROUP BY {', '.join(query_request.dimensions)}
                ORDER BY events DESC
                LIMIT {query_request.limit}
            """

            result = clickhouse_client.execute(query)

            return {
                'rows': result,
                'is_sampled': sample_rate < 1.0,
                'sample_rate': sample_rate
            }
        ```

        ---

        ## Lambda Architecture Trade-offs

        | Aspect | Speed Layer | Batch Layer |
        |--------|-------------|-------------|
        | **Latency** | < 10 seconds | 24 hours |
        | **Accuracy** | ~99% (approximate) | 100% (exact) |
        | **Complexity** | High (stateful streaming) | Medium (batch processing) |
        | **Cost** | Higher (always running) | Lower (scheduled jobs) |
        | **Use Case** | Real-time dashboards | Historical reports, compliance |

        **Why both layers:**

        - Speed layer: Satisfy user need for real-time visibility
        - Batch layer: Ensure data accuracy for business decisions
        - Serving layer: Hide complexity, present unified view

    === "🔗 Sessionization"

        ## The Challenge

        **Problem:** Group events into user sessions in real-time. A session = sequence of events with < 30 min gaps.

        **Why difficult:**

        - **Out-of-order events:** Network delays cause events to arrive late
        - **High throughput:** 175K events/sec, must maintain state for 50M concurrent sessions
        - **Session timeout:** 30 minutes of inactivity ends session
        - **Distributed processing:** State must be partitioned across many workers

        ---

        ## Sessionization Algorithm

        **Approach: Session windows with state management**

        ```
        Event stream → Key by (tracking_id, client_id) → Session window (30-min gap) → Aggregate → Session object
        ```

        **Session window behavior:**

        ```
        Events:    E1---E2----E3-----------E4 (> 30 min gap)
        Sessions:  [  Session 1   ]         [Session 2]
        ```

        ---

        ## Implementation (Flink Session Windows)

        ```python
        from pyflink.datastream import StreamExecutionEnvironment
        from pyflink.datastream.window import EventTimeSessionWindows
        from pyflink.datastream.state import ValueStateDescriptor
        from datetime import timedelta

        def sessionization_pipeline():
            """
            Real-time sessionization using Flink session windows

            - Group events by (tracking_id, client_id)
            - Session window with 30-minute gap
            - Handle late events (up to 5 minutes late)
            - Emit session object when window closes
            """
            env = StreamExecutionEnvironment.get_execution_environment()
            env.set_parallelism(32)

            # Enable checkpointing for fault tolerance
            env.enable_checkpointing(60000)  # Checkpoint every 60 seconds

            # Kafka source
            events = env.add_source(FlinkKafkaConsumer(
                topics=['events'],
                deserialization_schema=JSONDeserializationSchema(),
                properties={'bootstrap.servers': 'kafka:9092'}
            ))

            # Assign timestamps and watermarks (for event time processing)
            events = events.assign_timestamps_and_watermarks(
                WatermarkStrategy
                    .for_bounded_out_of_orderness(timedelta(minutes=5))  # Allow 5-min late events
                    .with_timestamp_assigner(lambda e, ts: e['event_timestamp'])
            )

            # Key by (tracking_id, client_id) for sessionization
            keyed_events = events.key_by(
                lambda e: f"{e['tracking_id']}:{e['client_id']}"
            )

            # Session window (30-minute inactivity gap)
            sessions = keyed_events.window(
                EventTimeSessionWindows.with_gap(timedelta(minutes=30))
            ).process(SessionProcessor())

            # Sink to Kafka (for downstream processing) and ClickHouse (for storage)
            sessions.add_sink(FlinkKafkaProducer(
                topic='sessions',
                serialization_schema=JSONSerializationSchema()
            ))

            sessions.add_sink(ClickHouseSink(
                table='sessions',
                batch_size=1000
            ))

            env.execute("Sessionization Pipeline")

        class SessionProcessor(ProcessWindowFunction):
            """
            Process all events in a session window

            - Aggregate events into session object
            - Calculate session metrics (duration, pageviews, conversion)
            - Enrich with first/last page, referrer, etc.
            """

            def process(self, key, context, events):
                """
                Args:
                    key: Composite key (tracking_id:client_id)
                    context: Window context (start/end time)
                    events: Iterable of all events in session
                """
                tracking_id, client_id = key.split(':')
                event_list = list(events)

                # Sort events by timestamp
                event_list.sort(key=lambda e: e['event_timestamp'])

                # Extract session attributes
                first_event = event_list[0]
                last_event = event_list[-1]

                session_id = f"{client_id}_{context.window().start}"
                session_start = first_event['event_timestamp']
                session_end = last_event['event_timestamp']
                duration = (session_end - session_start).seconds

                # Count event types
                pageviews = [e for e in event_list if e['event_type'] == 'pageview']
                clicks = [e for e in event_list if e['event_type'] == 'click']
                purchases = [e for e in event_list if e['event_type'] == 'purchase']

                # Determine entry/exit pages
                entry_page = first_event.get('page_url')
                exit_page = last_event.get('page_url')

                # Extract traffic source
                referrer = first_event.get('referrer')
                utm_source = first_event.get('utm_source')
                utm_medium = first_event.get('utm_medium')
                utm_campaign = first_event.get('utm_campaign')

                # Detect conversion
                converted = len(purchases) > 0

                # Build session object
                session = {
                    'session_id': session_id,
                    'tracking_id': tracking_id,
                    'client_id': client_id,
                    'user_id': first_event.get('user_id'),
                    'session_start': session_start.isoformat(),
                    'session_end': session_end.isoformat(),
                    'session_duration': duration,
                    'event_count': len(event_list),
                    'pageview_count': len(pageviews),
                    'click_count': len(clicks),
                    'purchase_count': len(purchases),
                    'entry_page': entry_page,
                    'exit_page': exit_page,
                    'referrer': referrer,
                    'utm_source': utm_source,
                    'utm_medium': utm_medium,
                    'utm_campaign': utm_campaign,
                    'country': first_event.get('country'),
                    'device': first_event.get('device'),
                    'browser': first_event.get('browser'),
                    'os': first_event.get('os'),
                    'converted': converted,
                    'bounced': len(pageviews) == 1,  # Single pageview = bounce
                    'revenue': sum(p.get('revenue', 0) for p in purchases)
                }

                yield session

        # Alternative: Stateful sessionization (more control)
        class StatefulSessionProcessor(KeyedProcessFunction):
            """
            Stateful sessionization with manual timeout management

            More control than session windows, but more complex
            """

            def __init__(self):
                self.session_state = None
                self.timeout_timer = None

            def open(self, runtime_context):
                # Initialize state
                self.session_state = runtime_context.get_state(
                    ValueStateDescriptor("session", Types.STRING())
                )

            def process_element(self, event, ctx):
                """Process incoming event"""
                # Get current session from state
                session_json = self.session_state.value()

                if session_json is None:
                    # Start new session
                    session = {
                        'session_id': f"{event['client_id']}_{event['event_timestamp']}",
                        'events': [event],
                        'session_start': event['event_timestamp']
                    }
                else:
                    # Add event to existing session
                    session = json.loads(session_json)
                    session['events'].append(event)

                # Update state
                self.session_state.update(json.dumps(session))

                # Set/reset timeout timer (30 minutes from now)
                timeout_time = event['event_timestamp'] + timedelta(minutes=30)
                ctx.timer_service().register_event_time_timer(timeout_time.timestamp() * 1000)

            def on_timer(self, timestamp, ctx):
                """Called when timeout expires (30 min of inactivity)"""
                # Retrieve session from state
                session_json = self.session_state.value()

                if session_json:
                    session = json.loads(session_json)

                    # Finalize session
                    finalized = self.finalize_session(session)

                    # Emit session
                    yield finalized

                    # Clear state
                    self.session_state.clear()

            def finalize_session(self, session):
                """Aggregate events into final session object"""
                events = session['events']
                events.sort(key=lambda e: e['event_timestamp'])

                # Same aggregation logic as SessionProcessor...
                return {...}
        ```

        ---

        ## Handling Late Events

        **Problem:** Events may arrive out-of-order due to network delays, client buffering, etc.

        **Solution: Watermarks + allowed lateness**

        ```python
        # Allow events up to 5 minutes late
        events = events.assign_timestamps_and_watermarks(
            WatermarkStrategy
                .for_bounded_out_of_orderness(timedelta(minutes=5))
                .with_timestamp_assigner(lambda e, ts: e['event_timestamp'])
        )

        # Configure window to accept late data
        sessions = keyed_events.window(
            EventTimeSessionWindows.with_gap(timedelta(minutes=30))
        ).allowed_lateness(timedelta(minutes=5))  # Accept late events for 5 min after window closes
        ```

        **Trade-off:**

        - **Longer lateness:** More accurate sessions, but higher latency (wait longer before finalizing)
        - **Shorter lateness:** Faster results, but may miss late events

        ---

        ## Session Enrichment

        **Add derived attributes:**

        ```python
        def enrich_session(session):
            """
            Enrich session with derived attributes

            - Traffic source category
            - Device category
            - Session quality score
            """
            # Categorize traffic source
            if session['utm_source']:
                session['traffic_category'] = 'paid'
            elif 'google.com' in (session['referrer'] or ''):
                session['traffic_category'] = 'organic'
            elif session['referrer']:
                session['traffic_category'] = 'referral'
            else:
                session['traffic_category'] = 'direct'

            # Device category
            device_categories = {
                'mobile': ['iphone', 'android', 'mobile'],
                'tablet': ['ipad', 'tablet'],
                'desktop': ['windows', 'mac', 'linux']
            }

            for category, keywords in device_categories.items():
                if any(kw in (session['device'] or '').lower() for kw in keywords):
                    session['device_category'] = category
                    break

            # Session quality score (0-100)
            # Based on: duration, pageviews, engagement
            score = 0
            score += min(session['session_duration'] / 600 * 30, 30)  # Max 30 points for 10+ min
            score += min(session['pageview_count'] * 10, 40)  # Max 40 points for 4+ pageviews
            score += 30 if session['converted'] else 0  # 30 points for conversion

            session['quality_score'] = min(score, 100)

            return session
        ```

    === "📊 Pre-aggregation Strategy"

        ## The Challenge

        **Problem:** Queries on billions of raw events are too slow (10s of seconds).

        **Example query:**

        ```sql
        SELECT country, COUNT(*) as users
        FROM events
        WHERE tracking_id = 'UA-123'
          AND event_timestamp BETWEEN '2026-01-01' AND '2026-01-31'
        GROUP BY country
        ```

        On 15B events/day × 31 days = 465B events → Query takes 30+ seconds!

        **Users expect:** < 5 seconds for all queries

        ---

        ## Pre-aggregation Approach

        **Strategy: Compute common aggregations in advance**

        ```
        Raw events → Materialized views (hourly, daily) → Fast queries
        ```

        **Benefits:**

        - **100x faster queries:** Read pre-computed counts instead of scanning billions of rows
        - **Lower query cost:** Fewer CPU cycles, less I/O
        - **Predictable performance:** Query time proportional to aggregation table size, not raw events

        **Trade-off:**

        - **Storage overhead:** Store both raw events and aggregations (2-3x storage)
        - **Reduced flexibility:** Can only query pre-aggregated dimensions

        ---

        ## Materialized Views (ClickHouse)

        ```sql
        -- Hourly aggregations (updated automatically by ClickHouse)
        CREATE MATERIALIZED VIEW events_hourly
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMMDD(hour)
        ORDER BY (tracking_id, hour, country, device, event_type)
        AS SELECT
            tracking_id,
            toStartOfHour(event_timestamp) as hour,
            country,
            device,
            event_type,
            count() as event_count,
            uniqExact(client_id) as unique_users,
            uniqExact(session_id) as unique_sessions,
            sum(revenue) as total_revenue
        FROM events
        GROUP BY tracking_id, hour, country, device, event_type;

        -- Daily aggregations (more dimensions)
        CREATE MATERIALIZED VIEW events_daily
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(day)
        ORDER BY (tracking_id, day, country, device, browser, utm_source)
        AS SELECT
            tracking_id,
            toDate(event_timestamp) as day,
            country,
            device,
            browser,
            utm_source,
            utm_medium,
            count() as pageviews,
            uniqExact(client_id) as unique_users,
            uniqExact(session_id) as sessions,
            countIf(event_type = 'purchase') as purchases,
            sum(revenue) as total_revenue,
            avg(session_duration) as avg_session_duration
        FROM events
        LEFT JOIN sessions USING (session_id)
        WHERE event_type IN ('pageview', 'purchase')
        GROUP BY tracking_id, day, country, device, browser, utm_source, utm_medium;

        -- Query aggregated data (100x faster)
        SELECT
            country,
            SUM(unique_users) as total_users,
            SUM(pageviews) as total_pageviews
        FROM events_daily
        WHERE tracking_id = 'UA-123'
          AND day BETWEEN '2026-01-01' AND '2026-01-31'
        GROUP BY country
        ORDER BY total_users DESC;
        -- Query time: < 100ms (vs 30+ seconds on raw events)
        ```

        ---

        ## OLAP Cube (Multi-dimensional Aggregations)

        **Concept: Pre-compute all combinations of dimensions**

        **Example dimensions:** date, country, device, browser, utm_source

        **Naive approach:** Pre-compute all 2^5 = 32 combinations

        **Problem:** Combinatorial explosion! With 10 dimensions → 2^10 = 1,024 combinations

        **Solution: Partial cube (only common combinations)**

        ```python
        # Define aggregation levels
        aggregation_levels = [
            # Level 1: Most granular (hourly)
            ['tracking_id', 'hour', 'country', 'device'],

            # Level 2: Daily with multiple dimensions
            ['tracking_id', 'day', 'country', 'device', 'browser'],
            ['tracking_id', 'day', 'country', 'utm_source', 'utm_medium'],
            ['tracking_id', 'day', 'device', 'utm_source'],

            # Level 3: Weekly rollups
            ['tracking_id', 'week', 'country'],
            ['tracking_id', 'week', 'device'],

            # Level 4: Monthly summary
            ['tracking_id', 'month', 'country'],
            ['tracking_id', 'month']
        ]

        def build_olap_cubes():
            """
            Build OLAP cubes using Spark

            - Read daily events
            - For each aggregation level, compute and store
            """
            spark = SparkSession.builder.appName("OLAP Cubes").getOrCreate()

            events = spark.read.parquet("s3://events/2026-02-01/*")

            for dimensions in aggregation_levels:
                # Build aggregation
                cube = events.groupBy(*dimensions).agg(
                    count('*').alias('event_count'),
                    countDistinct('client_id').alias('unique_users'),
                    countDistinct('session_id').alias('sessions'),
                    sum('revenue').alias('total_revenue')
                )

                # Store in ClickHouse
                table_name = f"cube_{'_'.join(dimensions)}"
                cube.write \
                    .format("jdbc") \
                    .option("url", "jdbc:clickhouse://clickhouse:8123/analytics") \
                    .option("dbtable", table_name) \
                    .mode("append") \
                    .save()
        ```

        ---

        ## Query Router (Automatic Aggregation Selection)

        ```python
        class QueryRouter:
            """
            Route queries to appropriate aggregation level

            - Parse query dimensions
            - Find matching pre-aggregated table
            - Fall back to raw events if no match
            """

            def __init__(self):
                # Pre-computed aggregation levels (sorted by granularity)
                self.aggregation_levels = [
                    ('events_hourly', ['tracking_id', 'hour', 'country', 'device', 'event_type']),
                    ('events_daily', ['tracking_id', 'day', 'country', 'device', 'browser', 'utm_source', 'utm_medium']),
                    ('events_weekly', ['tracking_id', 'week', 'country', 'device']),
                    ('events_monthly', ['tracking_id', 'month', 'country'])
                ]

            def route_query(self, query_request):
                """
                Find optimal table for query

                Args:
                    query_request: User query with dimensions, filters

                Returns:
                    (table_name, needs_post_processing)
                """
                requested_dims = set(query_request.dimensions)

                # Find matching aggregation (dimensions must be subset)
                for table_name, available_dims in self.aggregation_levels:
                    available_set = set(available_dims)

                    if requested_dims.issubset(available_set):
                        # Found match!
                        return (table_name, False)

                # No match - need to query raw events
                return ('events', True)

            def execute_query(self, query_request):
                """Execute query on optimal table"""
                table_name, needs_sampling = self.route_query(query_request)

                if table_name == 'events' and self.estimate_size(query_request) > 1_000_000_000:
                    # Large query on raw events - apply sampling
                    sample_rate = 0.1
                    query_request.sample_rate = sample_rate
                else:
                    sample_rate = 1.0

                # Build SQL query
                sql = self.build_sql(table_name, query_request, sample_rate)

                # Execute on ClickHouse
                result = clickhouse_client.execute(sql)

                return {
                    'rows': result,
                    'table_used': table_name,
                    'is_sampled': sample_rate < 1.0
                }

            def build_sql(self, table_name, query_request, sample_rate):
                """Generate optimized SQL query"""
                sample_clause = f"SAMPLE {sample_rate}" if sample_rate < 1.0 else ""
                scale_factor = 1 / sample_rate

                sql = f"""
                    SELECT
                        {', '.join(query_request.dimensions)},
                        SUM(event_count) * {scale_factor} as events,
                        SUM(unique_users) * {scale_factor} as users
                    FROM {table_name}
                    {sample_clause}
                    WHERE tracking_id = '{query_request.tracking_id}'
                      AND {self.build_time_filter(query_request)}
                      {self.build_filters(query_request.filters)}
                    GROUP BY {', '.join(query_request.dimensions)}
                    ORDER BY events DESC
                    LIMIT {query_request.limit}
                """
                return sql
        ```

        ---

        ## Aggregation Trade-offs

        | Approach | Query Speed | Storage Cost | Flexibility |
        |----------|-------------|--------------|-------------|
        | **Raw events only** | Slow (10-30s) | 1x | Full (any dimensions) |
        | **Partial pre-agg** | Fast (< 1s) | 2-3x | High (common dimensions) |
        | **Full OLAP cube** | Fastest (< 100ms) | 5-10x | Limited (pre-defined dimensions) |

        **Recommended: Partial pre-aggregation** (sweet spot of speed, cost, flexibility)

    === "🔀 Funnel Analysis"

        ## The Challenge

        **Problem:** Calculate conversion funnels on billions of events.

        **Example funnel:** Product View → Add to Cart → Checkout → Purchase

        **Requirements:**

        - Track user progression through steps
        - Calculate conversion rates between steps
        - Support time windows (e.g., 7 days to convert)
        - Handle multiple paths (users may skip steps)

        **Naive approach:** Join events for each step in SQL. **Doesn't scale** (multi-way joins on billions of events).

        ---

        ## Funnel Calculation Approaches

        ### Approach 1: User-level Funnel (Most Accurate)

        **Strategy: Track each user's journey through funnel**

        ```python
        def calculate_user_level_funnel(events, funnel_definition):
            """
            Calculate funnel by tracking each user's progression

            - For each user, find events matching funnel steps
            - Check if events occur in order within time window
            - Aggregate conversion rates

            Args:
                events: Spark DataFrame of events
                funnel_definition: Funnel steps and criteria

            Returns:
                Funnel conversion stats
            """
            spark = SparkSession.builder.getOrCreate()

            # Step 1: Filter events matching any funnel step
            funnel_steps = funnel_definition['steps']
            step_filters = []

            for i, step in enumerate(funnel_steps):
                step_filter = (
                    (col('event_type') == step['event_type']) &
                    # Apply additional step filters...
                )
                step_filters.append(
                    when(step_filter, i).otherwise(None).alias(f'step_{i}')
                )

            # Add step indicators to events
            events_with_steps = events.select(
                'tracking_id',
                'client_id',
                'event_timestamp',
                *step_filters
            )

            # Step 2: Find users who completed each step
            window_days = funnel_definition['window_days']

            # Users who completed step 0 (funnel entry)
            step0_users = events_with_steps \
                .filter(col('step_0').isNotNull()) \
                .select('client_id', col('event_timestamp').alias('step0_time')) \
                .distinct()

            funnel_results = [
                {
                    'step_index': 0,
                    'step_name': funnel_steps[0]['name'],
                    'user_count': step0_users.count(),
                    'conversion_rate': 1.0
                }
            ]

            # For each subsequent step, find users who completed it after previous step
            prev_step_users = step0_users

            for i in range(1, len(funnel_steps)):
                # Find users who completed step i within window after step i-1
                step_i_events = events_with_steps \
                    .filter(col(f'step_{i}').isNotNull()) \
                    .select('client_id', col('event_timestamp').alias(f'step{i}_time'))

                # Join with previous step users
                step_i_users = prev_step_users.join(
                    step_i_events,
                    on='client_id',
                    how='inner'
                ).filter(
                    # Step i occurs after previous step
                    (col(f'step{i}_time') > col(f'step{i-1}_time')) &
                    # Within time window
                    (col(f'step{i}_time') <= col(f'step{i-1}_time') + expr(f'INTERVAL {window_days} DAYS'))
                ).select('client_id', f'step{i}_time').distinct()

                user_count = step_i_users.count()
                conversion_rate = user_count / prev_step_users.count() if prev_step_users.count() > 0 else 0

                funnel_results.append({
                    'step_index': i,
                    'step_name': funnel_steps[i]['name'],
                    'user_count': user_count,
                    'conversion_rate': conversion_rate
                })

                prev_step_users = step_i_users

            return funnel_results
        ```

        ---

        ### Approach 2: Session-level Funnel (Faster)

        **Strategy: Pre-compute funnel steps per session**

        ```python
        def calculate_session_level_funnel(sessions, funnel_definition):
            """
            Calculate funnel using pre-aggregated session data

            - For each session, mark which funnel steps were completed
            - Aggregate conversion rates
            - Much faster than user-level (fewer records)

            Args:
                sessions: DataFrame of session data with events
                funnel_definition: Funnel steps
            """
            funnel_steps = funnel_definition['steps']

            # For each session, determine which steps were completed
            def mark_completed_steps(session_events):
                """
                Check which funnel steps were completed in session

                Returns: List[bool] (True if step completed)
                """
                completed = [False] * len(funnel_steps)

                for event in session_events:
                    for i, step in enumerate(funnel_steps):
                        if matches_step(event, step):
                            completed[i] = True

                return completed

            # Add funnel completion markers to sessions
            sessions_with_funnel = sessions.map(
                lambda session: {
                    **session,
                    'funnel_steps_completed': mark_completed_steps(session['events'])
                }
            )

            # Calculate conversion rates
            funnel_results = []

            for i, step in enumerate(funnel_steps):
                # Users who completed this step
                step_completed = sessions_with_funnel.filter(
                    lambda s: s['funnel_steps_completed'][i]
                )

                # Users who completed this step AND all previous steps (in order)
                full_funnel_to_step = sessions_with_funnel.filter(
                    lambda s: all(s['funnel_steps_completed'][j] for j in range(i + 1))
                )

                user_count = full_funnel_to_step.count()

                if i == 0:
                    conversion_rate = 1.0
                else:
                    prev_step_count = funnel_results[i-1]['user_count']
                    conversion_rate = user_count / prev_step_count if prev_step_count > 0 else 0

                funnel_results.append({
                    'step_index': i,
                    'step_name': step['name'],
                    'user_count': user_count,
                    'conversion_rate': conversion_rate
                })

            return funnel_results
        ```

        ---

        ### Approach 3: Pre-computed Funnel Tables

        **Strategy: Store funnel progression in dedicated table**

        ```sql
        -- Funnel progress table (one row per user per funnel)
        CREATE TABLE funnel_progress (
            tracking_id String,
            funnel_id String,
            client_id String,
            date Date,
            step0_completed Boolean,
            step0_timestamp Nullable(DateTime),
            step1_completed Boolean,
            step1_timestamp Nullable(DateTime),
            step2_completed Boolean,
            step2_timestamp Nullable(DateTime),
            step3_completed Boolean,
            step3_timestamp Nullable(DateTime),
            fully_converted Boolean,
            time_to_convert Nullable(UInt32)
        ) ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (tracking_id, funnel_id, date, client_id);

        -- Daily batch job populates this table
        -- Query is extremely fast (pre-computed)
        SELECT
            step_index,
            countIf(step0_completed) as step0_users,
            countIf(step1_completed) as step1_users,
            countIf(step2_completed) as step2_users,
            countIf(step3_completed) as step3_users,
            countIf(fully_converted) as converted_users,
            avg(time_to_convert) as avg_time_to_convert
        FROM funnel_progress
        WHERE tracking_id = 'UA-123'
          AND funnel_id = 'funnel_xyz'
          AND date BETWEEN '2026-01-01' AND '2026-01-31'
        GROUP BY step_index;
        ```

        ---

        ## Funnel Visualization API

        ```python
        @app.get("/api/v1/funnels/{funnel_id}/analysis")
        async def get_funnel_analysis(
            funnel_id: str,
            tracking_id: str,
            start_date: date,
            end_date: date,
            breakdown_by: Optional[str] = None  # e.g., 'device', 'country'
        ):
            """
            Get funnel conversion analysis

            - Read from pre-computed funnel_results table
            - Support breakdown by dimension
            - Calculate drop-off rates
            """
            # Check cache
            cache_key = f"funnel:{funnel_id}:{start_date}:{end_date}:{breakdown_by}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Query pre-computed funnel results
            if breakdown_by:
                query = f"""
                    SELECT
                        {breakdown_by},
                        step_index,
                        step_name,
                        SUM(user_count) as users,
                        AVG(conversion_rate) as conversion_rate
                    FROM funnel_results
                    WHERE tracking_id = '{tracking_id}'
                      AND funnel_id = '{funnel_id}'
                      AND date BETWEEN '{start_date}' AND '{end_date}'
                    GROUP BY {breakdown_by}, step_index, step_name
                    ORDER BY {breakdown_by}, step_index
                """
            else:
                query = f"""
                    SELECT
                        step_index,
                        step_name,
                        SUM(user_count) as users,
                        AVG(conversion_rate) as conversion_rate
                    FROM funnel_results
                    WHERE tracking_id = '{tracking_id}'
                      AND funnel_id = '{funnel_id}'
                      AND date BETWEEN '{start_date}' AND '{end_date}'
                    GROUP BY step_index, step_name
                    ORDER BY step_index
                """

            result = clickhouse_client.execute(query)

            # Calculate drop-off rates
            funnel_data = []
            prev_users = None

            for row in result:
                users = row['users']
                drop_off_rate = 1 - row['conversion_rate'] if row['conversion_rate'] < 1 else 0

                funnel_data.append({
                    'step_index': row['step_index'],
                    'step_name': row['step_name'],
                    'users': users,
                    'conversion_rate': row['conversion_rate'],
                    'drop_off_rate': drop_off_rate
                })

                prev_users = users

            # Cache for 5 minutes
            redis_client.setex(cache_key, 300, json.dumps(funnel_data))

            return funnel_data
        ```

        ---

        ## Funnel Optimization

        | Technique | Speedup | Trade-off |
        |-----------|---------|-----------|
        | **Session-level funnel** | 10x | May miss cross-session conversions |
        | **Pre-computed tables** | 100x | Daily latency (not real-time) |
        | **Sampling** | 10x | Less accurate for small funnels |
        | **Incremental updates** | 5x | Complex to implement |

        **Recommended:** Pre-compute funnels daily, use session-level for custom/ad-hoc funnels

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling analytics platform from 1M to 10B events/day.

    **Scaling challenges at 10B events/day:**

    - **Event ingestion:** 175K events/sec (peak: 520K/sec)
    - **Storage:** 14 PB of data (raw + aggregations)
    - **Query throughput:** 5K queries/sec (peak hours)
    - **Real-time processing:** 50M concurrent sessions
    - **Cost:** $500K+/month infrastructure

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Event ingestion** | ✅ Yes | Scale Kafka to 50 brokers, partition by tracking_id, compression (Snappy) |
    | **ClickHouse writes** | ✅ Yes | Bulk inserts (10K events/batch), async replication, SSD storage |
    | **Query latency** | ✅ Yes | Pre-aggregated tables (100x faster), Redis cache (90% hit rate), query sampling |
    | **Session tracking** | ✅ Yes | Flink parallelism (32 workers), checkpoint to S3 every 60s, state TTL (2 hours) |
    | **Storage costs** | ✅ Yes | Compress old data (Zstandard 5x), move to S3 after 30 days, TTL (13 months) |

    ---

    ## Horizontal Scaling

    ### Kafka Cluster

    ```
    50 brokers
    - 1,000 partitions (20 per broker)
    - Partition key: tracking_id (ensures events from same site go to same partition)
    - Replication factor: 3
    - Retention: 7 days (buffer before S3)
    - Compression: Snappy (3x reduction)

    Throughput: 520K events/sec (peak) × 2 KB = 1 GB/sec
    Storage: 1 GB/sec × 86,400 × 7 = 604 TB
    ```

    ### ClickHouse Cluster

    ```
    100 nodes (sharded + replicated)
    - 20 shards (5 nodes each with 1 replica)
    - Shard key: tracking_id (co-locate data from same site)
    - Distributed table for queries across shards
    - MergeTree engine (optimized for inserts)
    - Materialized views (auto-compute aggregations)

    Inserts: 175K events/sec / 100 nodes = 1,750 events/sec per node
    Storage per node: 14 PB / 100 = 140 TB per node
    ```

    ### Flink Cluster

    ```
    32 workers (parallelism = 32)
    - Session state: 250 GB (in-memory with RocksDB overflow)
    - Checkpoint to S3 every 60 seconds
    - Restart from checkpoint on failure
    - Backpressure handling (slow down if ClickHouse can't keep up)

    Throughput: 175K events/sec / 32 workers = 5,468 events/sec per worker
    ```

    ---

    ## Query Optimization

    ### 1. Pre-aggregation

    ```sql
    -- Instead of scanning 465B raw events
    SELECT country, COUNT(*) as users
    FROM events
    WHERE tracking_id = 'UA-123' AND event_timestamp BETWEEN '2026-01-01' AND '2026-01-31'
    GROUP BY country;
    -- Query time: 30+ seconds

    -- Query pre-aggregated table (1M rows instead of 465B)
    SELECT country, SUM(unique_users) as users
    FROM events_daily
    WHERE tracking_id = 'UA-123' AND day BETWEEN '2026-01-01' AND '2026-01-31'
    GROUP BY country;
    -- Query time: < 1 second (30x speedup)
    ```

    ### 2. Query Sampling

    ```sql
    -- For custom queries on large datasets, sample 10% of events
    SELECT country, COUNT(*) * 10 as users  -- Scale up by sample rate
    FROM events
    SAMPLE 0.1  -- Random 10% sample
    WHERE tracking_id = 'UA-123' AND event_timestamp BETWEEN '2026-01-01' AND '2026-01-31'
    GROUP BY country;
    -- Query time: < 5 seconds (6x speedup, 90% accuracy)
    ```

    ### 3. Partition Pruning

    ```sql
    -- ClickHouse only scans relevant partitions (daily partitions)
    -- Query for 31 days → scan 31 partitions (not all 390 partitions in 13 months)
    SELECT COUNT(*) FROM events
    WHERE tracking_id = 'UA-123'
      AND event_timestamp BETWEEN '2026-01-01' AND '2026-01-31';
    -- Scans: 31 partitions (not 390) → 12x speedup
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 10B events/day:**

    | Component | Cost |
    |-----------|------|
    | **Kafka cluster (50 brokers)** | $43,200 (50 × r5.2xlarge) |
    | **ClickHouse cluster (100 nodes)** | $172,800 (100 × i3.4xlarge with SSD) |
    | **Flink cluster (32 workers)** | $27,648 (32 × m5.2xlarge) |
    | **Redis cache (20 nodes)** | $8,640 (20 × r5.xlarge) |
    | **PostgreSQL (metadata)** | $2,160 (2 × db.r5.2xlarge) |
    | **S3 storage (cold data)** | $35,000 (14 PB × $0.021/GB-month + requests) |
    | **Data transfer** | $9,000 (300 TB egress/month) |
    | **Ingestion API servers (50)** | $21,600 (50 × m5.2xlarge) |
    | **Query API servers (100)** | $43,200 (100 × m5.xlarge) |
    | **Total** | **$363,248/month** |

    **Cost reduction strategies:**

    1. **Compression:** Zstandard compression (5x) reduces storage from 70 PB → 14 PB (save $1M+/month)
    2. **Tiered storage:** Move data older than 30 days to S3 (10x cheaper than ClickHouse SSD)
    3. **Auto-scaling:** Scale down query API servers during off-peak hours (save 30%)
    4. **Reserved instances:** 3-year commitment (save 40% vs on-demand)
    5. **Query caching:** 90% cache hit rate eliminates 90% of ClickHouse queries

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Event Ingestion Latency (P95)** | < 100ms | > 500ms |
    | **Event Delivery Rate** | > 99.9% | < 99% |
    | **Query Latency (P95)** | < 5s | > 10s |
    | **Real-time Dashboard Latency** | < 10s | > 30s |
    | **Kafka Consumer Lag** | < 10,000 events | > 100,000 events |
    | **ClickHouse Insert QPS** | 175K/sec | < 100K/sec (backlog building) |
    | **Cache Hit Rate** | > 90% | < 80% |
    | **Session Processing Lag** | < 1 minute | > 5 minutes |

    **Dashboards:**

    1. **Event Pipeline Health:** Ingestion QPS, Kafka lag, Flink checkpoints
    2. **Query Performance:** P50/P95/P99 latency, cache hit rate, queries/sec
    3. **Storage Utilization:** ClickHouse disk usage, S3 storage growth
    4. **Cost Tracking:** Daily spend per component, query cost attribution

    ---

    ## Disaster Recovery

    **Backup strategy:**

    ```
    1. Kafka: 7-day retention + S3 backup (can replay)
    2. ClickHouse: Daily snapshots to S3 + continuous replication
    3. PostgreSQL: Point-in-time recovery (PITR) + read replicas
    4. Flink state: Checkpoints to S3 every 60 seconds
    ```

    **Recovery scenarios:**

    | Scenario | RTO | RPO | Recovery Steps |
    |----------|-----|-----|----------------|
    | **Single node failure** | 0 (auto-failover) | 0 | Replicas take over automatically |
    | **Cluster failure** | 1 hour | 0 (Kafka replay) | Restore from backup, replay Kafka events |
    | **Region outage** | 4 hours | 5 minutes | Failover to standby region, restore from S3 |
    | **Data corruption** | 2 hours | 1 day | Restore from daily snapshot, replay missing events |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Lambda architecture:** Speed layer (Flink) for real-time + batch layer (Spark) for accuracy
    2. **ClickHouse for OLAP:** Columnar storage, 100x faster aggregations than row-based databases
    3. **Kafka for ingestion:** High-throughput event streaming, reliable delivery, replay capability
    4. **Pre-aggregation strategy:** Materialized views for common queries (100x speedup)
    5. **Sessionization:** Flink session windows with 30-minute timeout for real-time grouping
    6. **Query optimization:** Cache (90% hit rate), pre-aggregated tables, sampling for large datasets

    ---

    ## Interview Tips

    ✅ **Start with requirements** - Clarify scale (events/day), latency (real-time vs historical), data retention

    ✅ **Discuss lambda architecture** - Explain trade-offs between real-time and batch processing

    ✅ **Pre-aggregation is critical** - Queries on raw events don't scale, must pre-compute common metrics

    ✅ **Sessionization complexity** - Out-of-order events, stateful processing, timeout handling

    ✅ **Cost awareness** - Storage is major cost driver, compression and tiering are essential

    ✅ **OLAP database choice** - ClickHouse vs Druid vs Redshift trade-offs

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle late-arriving events?"** | Watermarks (5-min lateness), allowed lateness window, batch layer for corrections |
    | **"How to make queries fast on billions of events?"** | Pre-aggregated tables (materialized views), query sampling (10%), partition pruning |
    | **"How to calculate funnels efficiently?"** | Pre-compute daily (batch job), session-level funnels (faster), store in dedicated table |
    | **"How to handle spike in traffic (e.g., Black Friday)?"** | Auto-scale Kafka/ClickHouse, backpressure in Flink, queue events in Kafka (7-day buffer) |
    | **"How to ensure data accuracy?"** | Lambda architecture (batch layer is source of truth), idempotent processing, deduplication |
    | **"How to support custom dimensions?"** | Fall back to raw events, apply sampling, inform user about slower query time |

    ---

    ## Extended Topics (If Time Permits)

    ### Real-time Anomaly Detection

    **Detect traffic spikes, conversion drops:**

    ```python
    # Flink CEP (Complex Event Processing)
    pattern = Pattern.begin("start").where(lambda e: e['event_type'] == 'pageview')
    pattern = pattern.next("drop").where(
        lambda e: e['conversion_rate'] < previous_avg * 0.5
    ).within(timedelta(minutes=15))

    # Alert if conversion rate drops 50% in 15 minutes
    alerts = pattern.select(lambda events: {
        'alert_type': 'conversion_drop',
        'severity': 'high',
        'current_rate': events['drop']['conversion_rate'],
        'expected_rate': events['start']['conversion_rate']
    })
    ```

    ### User Segmentation

    **Create dynamic user segments:**

    ```sql
    -- High-value users (multiple purchases, high revenue)
    CREATE VIEW high_value_users AS
    SELECT client_id
    FROM sessions
    WHERE converted = true
    GROUP BY client_id
    HAVING COUNT(*) >= 3 AND SUM(revenue) > 500;

    -- Query metrics for segment
    SELECT
        date,
        COUNT(DISTINCT client_id) as users,
        SUM(revenue) as total_revenue
    FROM sessions
    WHERE client_id IN (SELECT client_id FROM high_value_users)
      AND date BETWEEN '2026-01-01' AND '2026-01-31'
    GROUP BY date;
    ```

    ### Retention Analysis

    **Cohort retention over time:**

    ```python
    def calculate_retention(events):
        """
        Calculate N-day retention for cohorts

        Cohort = users who first visited on same day
        Retention = % who returned after N days
        """
        # Define cohorts (first visit date)
        cohorts = events.groupBy('client_id').agg(
            min('event_timestamp').alias('first_visit')
        )

        # For each cohort, calculate retention
        retention_data = []

        for cohort_date in cohorts.select('first_visit').distinct().collect():
            cohort_users = cohorts.filter(col('first_visit') == cohort_date).select('client_id')

            for days_after in [1, 7, 14, 30, 60, 90]:
                # Users who returned N days after first visit
                returned = events.join(cohort_users, on='client_id').filter(
                    (col('event_timestamp') >= cohort_date + timedelta(days=days_after)) &
                    (col('event_timestamp') < cohort_date + timedelta(days=days_after + 1))
                ).select('client_id').distinct().count()

                retention_rate = returned / cohort_users.count()

                retention_data.append({
                    'cohort_date': cohort_date,
                    'days_after': days_after,
                    'retention_rate': retention_rate
                })

        return retention_data
    ```

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Google Analytics, Mixpanel, Amplitude, Segment, Heap, Adobe Analytics

---

*Master this problem and you'll be ready for: Real-time analytics systems, data pipelines, OLAP systems, event streaming platforms*
