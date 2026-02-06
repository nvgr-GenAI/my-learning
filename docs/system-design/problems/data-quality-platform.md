# Design a Data Quality Platform

A comprehensive data quality monitoring and validation system that ensures data accuracy, completeness, and reliability through automated validation rules, anomaly detection, data profiling, lineage tracking, and intelligent alert routing across data pipelines and warehouses.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10,000+ datasets, 1M+ validation checks/day, 100K+ data quality rules, PB-scale data |
| **Key Challenges** | Real-time anomaly detection, scalable validation, false positive reduction, lineage tracking, alert deduplication |
| **Core Concepts** | Validation rules, statistical anomaly detection, data profiling, schema drift, SLA monitoring, incident management |
| **Companies** | Great Expectations, Monte Carlo, Soda, Bigeye, Datadog, Collibra, AWS Deequ, Google Cloud DQ |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Validation Rules Engine** | Define and execute SQL/Python validation rules | P0 (Must have) |
    | **Anomaly Detection** | Statistical and ML-based detection of data anomalies | P0 (Must have) |
    | **Data Profiling** | Generate statistics (histograms, percentiles, distributions) | P0 (Must have) |
    | **Schema Validation** | Detect schema changes, type mismatches | P0 (Must have) |
    | **SLA Monitoring** | Track data freshness, completeness, accuracy SLAs | P0 (Must have) |
    | **Alert Routing** | Intelligent alert routing with deduplication | P0 (Must have) |
    | **Data Lineage Tracking** | Track data dependencies and impact analysis | P1 (Should have) |
    | **Incident Management** | Track, prioritize, and resolve data quality issues | P1 (Should have) |
    | **Custom Metrics** | Define custom quality metrics per dataset | P1 (Should have) |
    | **Historical Trending** | Track quality metrics over time | P1 (Should have) |
    | **Root Cause Analysis** | Identify source of data quality issues | P2 (Nice to have) |
    | **Auto-remediation** | Automatically fix common issues | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Data catalog and discovery (use Amundsen/DataHub)
    - Data transformation/ETL (use Airflow/dbt)
    - Master data management (MDM)
    - Data governance and access control
    - Real-time streaming validation (focus on batch)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Validation Throughput** | 1M+ checks/day | Support large-scale data validation |
    | **Latency** | < 5 seconds for validation, < 1 min for anomaly detection | Near real-time feedback |
    | **Accuracy** | < 5% false positive rate for anomalies | Avoid alert fatigue |
    | **Availability** | 99.9% uptime | Critical data pipelines depend on it |
    | **Scalability** | Handle 10K+ datasets, PB-scale data | Enterprise-scale support |
    | **Freshness** | < 5 minutes detection lag | Quick issue identification |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Dataset scale:
    - Total monitored datasets: 10,000 (tables, files, APIs)
    - Active validation rules per dataset: 10 rules
    - Total validation rules: 100,000 rules
    - Validation frequency: 24 times/day (hourly)
    - Total validations: 10,000 × 10 × 24 = 2.4M checks/day
    - Average check QPS: 2.4M / 86,400 = 28 checks/sec

    Anomaly detection:
    - Datasets with anomaly detection: 5,000 (50%)
    - Metrics per dataset: 20 (row count, nulls, distributions)
    - Detection frequency: 12 times/day (every 2 hours)
    - Total anomaly checks: 5,000 × 20 × 12 = 1.2M checks/day
    - Average detection QPS: 1.2M / 86,400 = 14 checks/sec

    Data profiling:
    - New datasets profiled per day: 100
    - Re-profiling frequency: Weekly
    - Daily profiling jobs: 100 + (10,000 / 7) = 1,530 jobs/day
    - Columns per dataset: 50
    - Total column profiles: 1,530 × 50 = 76,500 profiles/day

    Alerts:
    - Validation failure rate: 5%
    - Anomaly detection rate: 2%
    - Daily alerts: (2.4M × 5%) + (1.2M × 2%) = 120K + 24K = 144K alerts/day
    - After deduplication (90% reduction): 14,400 alerts/day
    - Alert QPS: 14,400 / 86,400 = 0.17 alerts/sec

    Lineage queries:
    - Lineage lookups per day: 10,000 (per incident investigation)
    - Average query latency: 200ms
    ```

    ### Storage Estimates

    ```
    Validation rules metadata:
    - 100,000 rules × 5 KB = 500 MB

    Validation execution history:
    - 2.4M checks/day × 365 days = 876M checks/year
    - Per check: 1 KB (rule_id, dataset, timestamp, result, error)
    - Total: 876 GB/year
    - With 2-year retention: 1.75 TB

    Anomaly detection results:
    - 1.2M checks/day × 365 days = 438M checks/year
    - Per check: 2 KB (metrics, z-scores, thresholds)
    - Total: 876 GB/year
    - With 2-year retention: 1.75 TB

    Data profiles:
    - 10,000 datasets × 50 columns = 500K columns
    - Per column profile: 20 KB (histogram, percentiles, stats)
    - Total: 10 GB current profiles
    - Historical profiles (weekly snapshots, 2 years): 10 GB × 104 weeks = 1 TB

    Lineage graph:
    - 10,000 datasets × 10 dependencies = 100K edges
    - Per edge: 500 bytes (source, target, transformation)
    - Total: 50 MB (graph data)
    - With metadata: 500 MB

    Time-series metrics:
    - 10,000 datasets × 50 metrics = 500K time series
    - Data points per series: 24/day × 365 days × 2 years = 17,520 points
    - Per point: 16 bytes (timestamp, value)
    - Total: 500K × 17,520 × 16 bytes = 140 GB

    Alerts and incidents:
    - 14,400 alerts/day × 365 days = 5.26M alerts/year
    - Per alert: 5 KB (rule, dataset, context, status)
    - Total: 26.3 GB/year
    - With 2-year retention: 52.6 GB

    Total storage: 1.75 TB (validation) + 1.75 TB (anomaly) + 1 TB (profiles) +
                   140 GB (metrics) + 52.6 GB (alerts) + 0.5 GB (lineage) ≈ 4.7 TB
    With replication (3x): ~14 TB
    ```

    ### Bandwidth Estimates

    ```
    Validation queries (read from data warehouse):
    - Average validation query: 10 MB result set
    - 28 checks/sec × 10 MB = 280 MB/sec = 2.24 Gbps
    - Peak (5x): 11.2 Gbps

    Metrics ingestion:
    - Write validation results: 28 checks/sec × 1 KB = 28 KB/sec = 224 Kbps
    - Write anomaly results: 14 checks/sec × 2 KB = 28 KB/sec = 224 Kbps
    - Write time-series metrics: 500K series × 1 point/hour × 16 bytes / 3600 = 2.2 KB/sec
    - Total write: ~500 Kbps (negligible)

    Alert delivery:
    - 0.17 alerts/sec × 5 KB = 0.85 KB/sec = 6.8 Kbps (negligible)
    ```

    ### Server Estimates

    ```
    Validation engine:
    - Concurrent validation jobs: 100
    - CPU per job: 2 cores (SQL query execution)
    - Total: 200 cores → 25 nodes (8 cores each)
    - Memory per node: 32 GB (query buffering)

    Anomaly detection engine:
    - Statistical models: 5 nodes (16 cores, 64 GB RAM each)
    - ML models (Prophet, LSTM): 10 nodes (GPU-enabled for training)
    - Total: 15 nodes

    Data profiling workers:
    - Profiling jobs per hour: 1,530 / 24 = 64 jobs/hour
    - Job duration: 10 minutes average
    - Concurrent jobs: (64 × 10) / 60 = 11 jobs
    - Total: 15 nodes (2 jobs per node, 8 cores, 64 GB RAM each)

    Metadata store (PostgreSQL):
    - Primary: 1 node (32 cores, 128 GB RAM, 5 TB SSD)
    - Read replicas: 3 nodes
    - Total: 4 database nodes

    Time-series database (InfluxDB/Prometheus):
    - 3 nodes (16 cores, 64 GB RAM, 1 TB SSD each)
    - Write throughput: 50K points/sec
    - Query latency: < 100ms

    Lineage graph store (Neo4j):
    - 3 nodes (16 cores, 64 GB RAM each)
    - Graph query latency: < 200ms

    Alert manager:
    - 3 nodes (8 cores, 32 GB RAM)
    - Handle 1,000 alerts/sec (with deduplication)

    Message queue (Kafka):
    - 3 nodes (16 cores, 64 GB RAM)
    - Topic: validation-results, anomaly-detections, alerts

    Object storage (S3/GCS):
    - Managed service
    - 15 TB storage

    Total infrastructure:
    - Validation: 25 nodes
    - Anomaly detection: 15 nodes
    - Profiling: 15 nodes
    - Databases: 4 + 3 + 3 = 10 nodes
    - Alert manager: 3 nodes
    - Message queue: 3 nodes
    - Total: ~71 nodes
    ```

    ---

    ## Key Assumptions

    1. 10,000 datasets monitored (tables, files, APIs)
    2. 10 validation rules per dataset on average
    3. Hourly validation frequency (24 times/day)
    4. 5% validation failure rate, 2% anomaly rate
    5. 90% alert deduplication (reduce noise)
    6. 2-year retention for historical metrics
    7. 50 columns per dataset on average
    8. Weekly data profiling for established datasets

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Rule-based + ML hybrid:** Combine deterministic rules with statistical anomaly detection
    2. **Declarative validation:** Define quality checks as code (SQL, Python, YAML)
    3. **Distributed execution:** Scale validation across multiple workers
    4. **Alert intelligence:** Deduplication, prioritization, root cause grouping
    5. **Lineage-aware:** Impact analysis based on data dependencies
    6. **Self-service:** Teams define their own quality rules and metrics

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Data Sources"
            Warehouse[(Data Warehouse<br/>Snowflake, BigQuery<br/>Redshift)]
            DataLake[(Data Lake<br/>S3, GCS<br/>Parquet files)]
            Databases[(Databases<br/>PostgreSQL, MySQL)]
            APIs[REST APIs<br/>Third-party data]
        end

        subgraph "Rule Definition"
            RuleUI[Rule Builder UI<br/>SQL/Python editor<br/>Template library]
            RuleRepo[Rule Repository<br/>Git-based config<br/>YAML definitions]
            RuleRegistry[(Rule Registry<br/>PostgreSQL<br/>Rules, thresholds)]
        end

        subgraph "Validation Layer"
            Scheduler[Validation Scheduler<br/>Cron/Event-driven<br/>Job orchestration]

            subgraph "Validation Engines"
                SQLValidator[SQL Validator<br/>Execute SQL checks<br/>Aggregate queries]
                PythonValidator[Python Validator<br/>Custom logic<br/>Great Expectations]
                SchemaValidator[Schema Validator<br/>Type checks<br/>Column validation]
            end

            ValidationQueue[Validation Queue<br/>Kafka<br/>Job distribution]
        end

        subgraph "Anomaly Detection"
            Profiler[Data Profiler<br/>Statistics computation<br/>Histograms, percentiles]

            subgraph "Detection Engines"
                StatDetector[Statistical Detector<br/>Z-score, IQR<br/>Threshold-based]
                MLDetector[ML Detector<br/>Prophet, LSTM<br/>Seasonal patterns]
                DriftDetector[Drift Detector<br/>Schema changes<br/>Distribution shifts]
            end

            AnomalyQueue[Anomaly Queue<br/>Kafka<br/>Detection results]
        end

        subgraph "Metadata & State"
            MetaDB[(Metadata Store<br/>PostgreSQL<br/>Rules, results<br/>Incidents)]
            TimeSeriesDB[(Time-Series DB<br/>InfluxDB/Prometheus<br/>Metrics over time)]
            LineageGraph[(Lineage Graph<br/>Neo4j<br/>Dataset dependencies)]
            ProfileStore[(Profile Store<br/>S3/GCS<br/>Parquet files<br/>Statistics)]
        end

        subgraph "Alert & Incident Management"
            AlertEngine[Alert Engine<br/>Rule evaluation<br/>Threshold checks]
            Deduplicator[Alert Deduplicator<br/>Group similar alerts<br/>Suppress duplicates]
            Prioritizer[Alert Prioritizer<br/>Severity scoring<br/>Impact analysis]
            IncidentMgr[Incident Manager<br/>Track issues<br/>Resolution workflow]
            Notifier[Notification Service<br/>Slack, Email, PagerDuty<br/>Webhook delivery]
        end

        subgraph "Observability"
            Dashboard[Quality Dashboard<br/>Metrics visualization<br/>Trend analysis]
            Lineage[Lineage Explorer<br/>Impact analysis<br/>Root cause tracing]
            Reports[Quality Reports<br/>SLA compliance<br/>Health scores]
        end

        Warehouse --> SQLValidator
        DataLake --> SQLValidator
        Databases --> SQLValidator
        APIs --> PythonValidator

        RuleUI -->|Define rules| RuleRepo
        RuleRepo -->|Sync| RuleRegistry

        Scheduler -->|Schedule jobs| ValidationQueue
        RuleRegistry -.->|Load rules| Scheduler

        ValidationQueue --> SQLValidator
        ValidationQueue --> PythonValidator
        ValidationQueue --> SchemaValidator

        SQLValidator --> MetaDB
        PythonValidator --> MetaDB
        SchemaValidator --> MetaDB

        SQLValidator --> AnomalyQueue
        PythonValidator --> AnomalyQueue

        Warehouse --> Profiler
        DataLake --> Profiler
        Profiler --> ProfileStore
        Profiler --> StatDetector
        Profiler --> MLDetector

        ProfileStore --> DriftDetector

        StatDetector --> AnomalyQueue
        MLDetector --> AnomalyQueue
        DriftDetector --> AnomalyQueue

        AnomalyQueue --> AlertEngine
        MetaDB --> AlertEngine

        AlertEngine --> Deduplicator
        Deduplicator --> Prioritizer
        LineageGraph -.->|Impact| Prioritizer

        Prioritizer --> IncidentMgr
        IncidentMgr --> Notifier

        MetaDB --> TimeSeriesDB
        AnomalyQueue --> TimeSeriesDB

        TimeSeriesDB --> Dashboard
        LineageGraph --> Lineage
        MetaDB --> Reports

        Notifier -->|Slack| Users[Data Teams]
        Dashboard --> Users
        Lineage --> Users

        style SQLValidator fill:#e1f5ff
        style StatDetector fill:#fff9c4
        style AlertEngine fill:#ffe1e1
        style MetaDB fill:#f3e5f5
        style TimeSeriesDB fill:#e8f5e9
        style LineageGraph fill:#fce4ec
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Rule Registry (PostgreSQL)** | Centralized rule management, versioning, ACID | NoSQL (no complex queries), files (no concurrency) |
    | **Kafka for job queue** | High throughput, persistent queue, replay capability | Redis (limited persistence), RabbitMQ (lower throughput) |
    | **Time-series DB (InfluxDB)** | Optimized for metrics over time, efficient compression | PostgreSQL (slow for time-series), Cassandra (complex) |
    | **Graph DB (Neo4j)** | Fast lineage traversal, impact analysis queries | PostgreSQL (slow recursive queries), separate lineage service |
    | **Hybrid anomaly detection** | Statistical for simple cases, ML for seasonal patterns | Statistical only (misses patterns), ML only (expensive) |
    | **Alert deduplication** | Reduce alert fatigue (90% reduction in noise) | No deduplication (alert storm), simple grouping (loses context) |

    **Key Trade-off:** We chose **eventual consistency** for validation results (via Kafka) over synchronous writes to enable high throughput. Results may arrive out of order, but idempotent writes to time-series DB handle this gracefully.

    ---

    ## API Design

    ### 1. Define Validation Rule

    **Request (YAML config):**
    ```yaml
    # Great Expectations-style expectation suite
    rule_id: "user_table_freshness_check"
    dataset: "analytics.users"
    rule_type: "freshness"
    schedule: "0 * * * *"  # Hourly

    expectations:
      - expectation_type: "expect_table_row_count_to_be_between"
        kwargs:
          min_value: 10000
          max_value: 1000000000

      - expectation_type: "expect_column_values_to_not_be_null"
        column: "user_id"
        kwargs:
          mostly: 1.0  # 100% not null

      - expectation_type: "expect_column_values_to_be_unique"
        column: "user_id"
        kwargs:
          mostly: 0.999  # 99.9% unique

      - expectation_type: "expect_column_values_to_match_regex"
        column: "email"
        kwargs:
          regex: "^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"
          mostly: 0.99

      - expectation_type: "expect_column_max_to_be_between"
        column: "created_at"
        kwargs:
          min_value: "{{ now - 1 hour }}"  # Data not older than 1 hour
          max_value: "{{ now }}"

    alert_config:
      severity: "high"
      channels: ["slack:data-team", "email:data-team@company.com"]
      sla_threshold: 2  # Alert if 2+ consecutive failures
      dedupe_window: "1 hour"

    tags:
      team: "data-platform"
      criticality: "high"
      sla: "99.9%"
    ```

    **Response:**
    ```json
    {
      "rule_id": "user_table_freshness_check",
      "version": 1,
      "status": "active",
      "next_run": "2025-02-05T10:00:00Z",
      "created_at": "2025-02-05T09:30:00Z"
    }
    ```

    ---

    ### 2. Execute Validation Check

    **Request:**
    ```bash
    POST /api/v1/validations/execute
    Content-Type: application/json

    {
      "rule_id": "user_table_freshness_check",
      "dataset": "analytics.users",
      "execution_mode": "full",  # full or sample
      "async": true
    }
    ```

    **Response:**
    ```json
    {
      "validation_run_id": "vr_20250205_100000_abc123",
      "status": "running",
      "rule_id": "user_table_freshness_check",
      "dataset": "analytics.users",
      "started_at": "2025-02-05T10:00:00Z",
      "estimated_completion": "2025-02-05T10:00:05Z"
    }
    ```

    **Validation Result (async callback):**
    ```json
    {
      "validation_run_id": "vr_20250205_100000_abc123",
      "status": "completed",
      "overall_result": "failure",
      "execution_time_ms": 4523,
      "results": [
        {
          "expectation": "expect_table_row_count_to_be_between",
          "success": true,
          "observed_value": 5428392,
          "expected_range": [10000, 1000000000]
        },
        {
          "expectation": "expect_column_values_to_not_be_null",
          "column": "user_id",
          "success": true,
          "observed_value": 1.0,
          "expected_value": 1.0
        },
        {
          "expectation": "expect_column_max_to_be_between",
          "column": "created_at",
          "success": false,
          "observed_value": "2025-02-05T08:30:00Z",
          "expected_range": ["2025-02-05T09:00:00Z", "2025-02-05T10:00:00Z"],
          "error": "Data is stale by 30 minutes"
        }
      ],
      "alert_triggered": true,
      "alert_id": "alert_abc123"
    }
    ```

    ---

    ### 3. Get Data Profile

    **Request:**
    ```bash
    GET /api/v1/datasets/analytics.users/profile
    ```

    **Response:**
    ```json
    {
      "dataset": "analytics.users",
      "profile_timestamp": "2025-02-05T10:00:00Z",
      "row_count": 5428392,
      "column_count": 25,
      "size_bytes": 2147483648,

      "columns": [
        {
          "name": "user_id",
          "type": "bigint",
          "nullable": false,
          "unique_count": 5428392,
          "unique_ratio": 1.0,
          "null_count": 0,
          "null_ratio": 0.0,
          "statistics": {
            "min": 1,
            "max": 10583921,
            "mean": 5291960.5,
            "median": 5291960,
            "stddev": 1568231.2
          }
        },
        {
          "name": "email",
          "type": "string",
          "nullable": false,
          "unique_count": 5428392,
          "unique_ratio": 1.0,
          "null_count": 0,
          "null_ratio": 0.0,
          "pattern_distribution": {
            "gmail.com": 0.42,
            "yahoo.com": 0.18,
            "outlook.com": 0.15,
            "other": 0.25
          },
          "value_length": {
            "min": 8,
            "max": 120,
            "mean": 24.5
          }
        },
        {
          "name": "created_at",
          "type": "timestamp",
          "nullable": false,
          "null_count": 0,
          "statistics": {
            "min": "2020-01-01T00:00:00Z",
            "max": "2025-02-05T08:30:00Z",
            "mean": "2023-03-15T12:00:00Z"
          },
          "histogram": [
            {"bucket": "2020", "count": 458231},
            {"bucket": "2021", "count": 892341},
            {"bucket": "2022", "count": 1283921},
            {"bucket": "2023", "count": 1458392},
            {"bucket": "2024", "count": 1235507},
            {"bucket": "2025", "count": 100000}
          ]
        }
      ],

      "quality_score": 0.94,
      "issues": [
        {
          "type": "stale_data",
          "severity": "medium",
          "description": "Max created_at is 30 minutes old"
        }
      ]
    }
    ```

    ---

    ### 4. Query Data Lineage

    **Request:**
    ```bash
    GET /api/v1/lineage/dataset?dataset=analytics.user_daily_metrics&direction=upstream
    ```

    **Response:**
    ```json
    {
      "dataset": "analytics.user_daily_metrics",
      "lineage": {
        "upstream": [
          {
            "dataset": "analytics.users",
            "type": "table",
            "distance": 1,
            "transformation": "JOIN ON user_id",
            "quality_score": 0.94,
            "last_updated": "2025-02-05T10:00:00Z"
          },
          {
            "dataset": "analytics.events",
            "type": "table",
            "distance": 1,
            "transformation": "GROUP BY user_id",
            "quality_score": 0.89,
            "last_updated": "2025-02-05T09:55:00Z"
          },
          {
            "dataset": "raw.user_signups",
            "type": "table",
            "distance": 2,
            "transformation": "ETL pipeline",
            "quality_score": 0.98,
            "last_updated": "2025-02-05T09:00:00Z"
          }
        ],
        "downstream": [
          {
            "dataset": "dashboards.executive_metrics",
            "type": "dashboard",
            "distance": 1,
            "consumers": ["CEO Dashboard", "Weekly Report"],
            "affected_users": 50
          }
        ]
      },
      "impact_score": 0.85,
      "critical_path": true
    }
    ```

    ---

    ## Database Schema

    ### Validation Rules

    ```sql
    CREATE TABLE validation_rules (
        rule_id VARCHAR(255) PRIMARY KEY,
        dataset_id VARCHAR(500) NOT NULL,
        rule_type VARCHAR(100) NOT NULL,  -- freshness, completeness, accuracy, schema
        rule_definition JSONB NOT NULL,   -- Expectations, SQL, Python code
        schedule_cron VARCHAR(100),
        enabled BOOLEAN DEFAULT TRUE,
        version INT DEFAULT 1,
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        tags JSONB,
        alert_config JSONB  -- Severity, channels, SLA
    );

    CREATE INDEX idx_rules_dataset ON validation_rules(dataset_id);
    CREATE INDEX idx_rules_type ON validation_rules(rule_type);
    CREATE INDEX idx_rules_enabled ON validation_rules(enabled) WHERE enabled = TRUE;
    ```

    ### Validation Runs

    ```sql
    CREATE TABLE validation_runs (
        validation_run_id VARCHAR(255) PRIMARY KEY,
        rule_id VARCHAR(255) REFERENCES validation_rules(rule_id),
        dataset_id VARCHAR(500) NOT NULL,
        status VARCHAR(50) NOT NULL,  -- running, completed, failed
        overall_result VARCHAR(50),   -- success, failure, warning
        execution_time_ms INT,
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        results JSONB,  -- Array of expectation results
        row_count BIGINT,
        sample_size BIGINT,
        alert_triggered BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX idx_runs_rule ON validation_runs(rule_id);
    CREATE INDEX idx_runs_dataset ON validation_runs(dataset_id);
    CREATE INDEX idx_runs_started ON validation_runs(started_at DESC);
    CREATE INDEX idx_runs_status ON validation_runs(status);
    ```

    ### Data Profiles

    ```sql
    CREATE TABLE dataset_profiles (
        profile_id BIGSERIAL PRIMARY KEY,
        dataset_id VARCHAR(500) NOT NULL,
        profile_timestamp TIMESTAMP NOT NULL,
        row_count BIGINT,
        column_count INT,
        size_bytes BIGINT,
        quality_score DECIMAL(5, 4),  -- 0.0 to 1.0
        column_profiles JSONB,  -- Array of column statistics
        issues JSONB,  -- Array of detected issues
        profile_storage_path VARCHAR(1000),  -- S3/GCS path for detailed stats
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_profiles_dataset ON dataset_profiles(dataset_id);
    CREATE INDEX idx_profiles_timestamp ON dataset_profiles(profile_timestamp DESC);
    CREATE UNIQUE INDEX idx_profiles_dataset_timestamp ON dataset_profiles(dataset_id, profile_timestamp);
    ```

    ### Anomaly Detections

    ```sql
    CREATE TABLE anomaly_detections (
        anomaly_id BIGSERIAL PRIMARY KEY,
        dataset_id VARCHAR(500) NOT NULL,
        metric_name VARCHAR(255) NOT NULL,  -- row_count, null_ratio, avg_value
        observed_value DOUBLE PRECISION NOT NULL,
        expected_value DOUBLE PRECISION,
        z_score DOUBLE PRECISION,
        threshold DOUBLE PRECISION,
        anomaly_type VARCHAR(100),  -- spike, drop, drift, outlier
        severity VARCHAR(50),  -- low, medium, high, critical
        detection_method VARCHAR(100),  -- z_score, iqr, prophet, lstm
        confidence DECIMAL(5, 4),  -- 0.0 to 1.0
        context JSONB,  -- Historical values, seasonal pattern
        detected_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_anomaly_dataset ON anomaly_detections(dataset_id);
    CREATE INDEX idx_anomaly_detected ON anomaly_detections(detected_at DESC);
    CREATE INDEX idx_anomaly_severity ON anomaly_detections(severity);
    ```

    ### Data Lineage (Graph edges)

    ```sql
    CREATE TABLE dataset_lineage (
        lineage_id BIGSERIAL PRIMARY KEY,
        source_dataset_id VARCHAR(500) NOT NULL,
        target_dataset_id VARCHAR(500) NOT NULL,
        lineage_type VARCHAR(100),  -- direct, indirect, inferred
        transformation_logic TEXT,
        pipeline_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        UNIQUE(source_dataset_id, target_dataset_id)
    );

    CREATE INDEX idx_lineage_source ON dataset_lineage(source_dataset_id);
    CREATE INDEX idx_lineage_target ON dataset_lineage(target_dataset_id);
    ```

    ### Alerts and Incidents

    ```sql
    CREATE TABLE alerts (
        alert_id VARCHAR(255) PRIMARY KEY,
        rule_id VARCHAR(255) REFERENCES validation_rules(rule_id),
        validation_run_id VARCHAR(255) REFERENCES validation_runs(validation_run_id),
        anomaly_id BIGINT REFERENCES anomaly_detections(anomaly_id),
        dataset_id VARCHAR(500) NOT NULL,
        alert_type VARCHAR(100),  -- validation_failure, anomaly, schema_change
        severity VARCHAR(50),
        status VARCHAR(50),  -- open, acknowledged, resolved, suppressed
        title TEXT,
        description TEXT,
        context JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        acknowledged_at TIMESTAMP,
        resolved_at TIMESTAMP,
        assigned_to VARCHAR(100)
    );

    CREATE INDEX idx_alerts_dataset ON alerts(dataset_id);
    CREATE INDEX idx_alerts_status ON alerts(status);
    CREATE INDEX idx_alerts_severity ON alerts(severity);
    CREATE INDEX idx_alerts_created ON alerts(created_at DESC);

    CREATE TABLE incidents (
        incident_id VARCHAR(255) PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        severity VARCHAR(50),
        status VARCHAR(50),  -- open, investigating, resolved
        root_cause TEXT,
        impact_score DECIMAL(5, 4),
        affected_datasets TEXT[],
        alert_ids TEXT[],  -- Multiple alerts grouped into one incident
        created_at TIMESTAMP DEFAULT NOW(),
        resolved_at TIMESTAMP,
        assigned_to VARCHAR(100)
    );

    CREATE INDEX idx_incidents_status ON incidents(status);
    CREATE INDEX idx_incidents_severity ON incidents(severity);
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. Validation Rule Engine

    **SQL-Based Validation:**

    ```python
    class SQLValidator:
        """
        Execute SQL-based validation rules

        Supports common patterns:
        - Row count checks
        - Null checks
        - Uniqueness checks
        - Value range checks
        - Referential integrity
        """

        def __init__(self, db_connection):
            self.db = db_connection

        def execute_validation(self, rule):
            """Execute SQL validation rule"""
            rule_type = rule['rule_type']
            dataset = rule['dataset']
            config = rule['rule_definition']

            if rule_type == 'row_count':
                return self.validate_row_count(dataset, config)
            elif rule_type == 'null_check':
                return self.validate_nulls(dataset, config)
            elif rule_type == 'unique_check':
                return self.validate_uniqueness(dataset, config)
            elif rule_type == 'custom_sql':
                return self.validate_custom_sql(dataset, config)

        def validate_row_count(self, dataset, config):
            """Validate table row count is within expected range"""
            query = f"""
                SELECT COUNT(*) as row_count
                FROM {dataset}
            """

            result = self.db.execute(query)
            observed = result['row_count']

            min_value = config.get('min_value', 0)
            max_value = config.get('max_value', float('inf'))

            success = min_value <= observed <= max_value

            return {
                'expectation': 'row_count_in_range',
                'success': success,
                'observed_value': observed,
                'expected_range': [min_value, max_value],
                'error': None if success else f"Row count {observed} outside range [{min_value}, {max_value}]"
            }

        def validate_nulls(self, dataset, config):
            """Validate null percentage is below threshold"""
            column = config['column']
            threshold = config.get('max_null_ratio', 0.0)

            query = f"""
                SELECT
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as null_count,
                    SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as null_ratio
                FROM {dataset}
            """

            result = self.db.execute(query)
            observed_ratio = result['null_ratio']

            success = observed_ratio <= threshold

            return {
                'expectation': 'null_ratio_below_threshold',
                'column': column,
                'success': success,
                'observed_value': observed_ratio,
                'expected_value': threshold,
                'null_count': result['null_count'],
                'total_rows': result['total_rows']
            }

        def validate_uniqueness(self, dataset, config):
            """Validate column uniqueness ratio"""
            column = config['column']
            threshold = config.get('min_unique_ratio', 0.99)

            query = f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT {column}) as unique_count,
                    COUNT(DISTINCT {column})::FLOAT / COUNT(*) as unique_ratio
                FROM {dataset}
            """

            result = self.db.execute(query)
            observed_ratio = result['unique_ratio']

            success = observed_ratio >= threshold

            return {
                'expectation': 'unique_ratio_above_threshold',
                'column': column,
                'success': success,
                'observed_value': observed_ratio,
                'expected_value': threshold,
                'unique_count': result['unique_count'],
                'total_rows': result['total_rows']
            }

        def validate_custom_sql(self, dataset, config):
            """Execute custom SQL validation query"""
            # SQL should return rows that FAIL validation
            sql = config['sql']

            # Example: SELECT * FROM users WHERE age < 0 OR age > 150
            result = self.db.execute(sql)

            failure_count = len(result)
            success = failure_count == 0

            return {
                'expectation': 'custom_sql',
                'success': success,
                'failure_count': failure_count,
                'failed_rows': result[:10] if failure_count > 0 else []  # Sample
            }
    ```

    **Great Expectations Integration:**

    ```python
    import great_expectations as gx

    class GreatExpectationsValidator:
        """
        Integration with Great Expectations framework

        Provides 50+ built-in expectations:
        - expect_column_values_to_be_in_set
        - expect_column_mean_to_be_between
        - expect_table_columns_to_match_ordered_list
        """

        def __init__(self, data_context_config):
            self.context = gx.get_context(project_config=data_context_config)

        def execute_expectation_suite(self, dataset, suite_name):
            """Execute Great Expectations suite"""
            # Get data source
            datasource = self.context.get_datasource('my_datasource')

            # Create batch request
            batch_request = datasource.get_batch_request(
                data_asset_name=dataset,
                options={}
            )

            # Get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )

            # Run validation
            results = validator.validate()

            return self.parse_results(results)

        def parse_results(self, results):
            """Parse GE results to standard format"""
            return {
                'success': results['success'],
                'statistics': results['statistics'],
                'results': [
                    {
                        'expectation': r['expectation_config']['expectation_type'],
                        'success': r['success'],
                        'observed_value': r['result'].get('observed_value'),
                        'expected_value': r['expectation_config']['kwargs']
                    }
                    for r in results['results']
                ]
            }
    ```

    ---

    ## 2. Anomaly Detection Engines

    ### Statistical Anomaly Detection

    ```python
    import numpy as np
    from scipy import stats

    class StatisticalAnomalyDetector:
        """
        Statistical anomaly detection using:
        - Z-score (standard deviations from mean)
        - IQR (Interquartile Range)
        - Moving average with confidence bands
        """

        def detect_anomalies_zscore(self, metric_name, time_series, threshold=3.0):
            """
            Detect anomalies using Z-score

            Z-score = (value - mean) / std_dev
            Anomaly if |Z-score| > threshold (default: 3.0)
            """
            values = np.array([point['value'] for point in time_series])
            timestamps = [point['timestamp'] for point in time_series]

            # Compute statistics
            mean = np.mean(values)
            std = np.std(values)

            # Compute Z-scores
            z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)

            # Detect anomalies
            anomalies = []
            for i, (ts, value, z_score) in enumerate(zip(timestamps, values, z_scores)):
                if abs(z_score) > threshold:
                    anomalies.append({
                        'timestamp': ts,
                        'metric_name': metric_name,
                        'observed_value': value,
                        'expected_value': mean,
                        'z_score': z_score,
                        'threshold': threshold,
                        'anomaly_type': 'spike' if z_score > 0 else 'drop',
                        'severity': self.compute_severity(z_score, threshold),
                        'detection_method': 'z_score'
                    })

            return anomalies

        def detect_anomalies_iqr(self, metric_name, time_series, multiplier=1.5):
            """
            Detect anomalies using IQR (Interquartile Range)

            Outliers: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            """
            values = np.array([point['value'] for point in time_series])
            timestamps = [point['timestamp'] for point in time_series]

            # Compute quartiles
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            # Compute bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Detect anomalies
            anomalies = []
            for ts, value in zip(timestamps, values):
                if value < lower_bound or value > upper_bound:
                    anomalies.append({
                        'timestamp': ts,
                        'metric_name': metric_name,
                        'observed_value': value,
                        'expected_range': [lower_bound, upper_bound],
                        'anomaly_type': 'outlier',
                        'detection_method': 'iqr'
                    })

            return anomalies

        def compute_severity(self, z_score, threshold):
            """Compute severity based on Z-score magnitude"""
            abs_z = abs(z_score)
            if abs_z < threshold:
                return 'normal'
            elif abs_z < threshold * 1.5:
                return 'low'
            elif abs_z < threshold * 2:
                return 'medium'
            elif abs_z < threshold * 3:
                return 'high'
            else:
                return 'critical'
    ```

    ### ML-Based Anomaly Detection

    ```python
    from prophet import Prophet
    import pandas as pd

    class MLAnomalyDetector:
        """
        ML-based anomaly detection using:
        - Prophet (Facebook): Seasonal patterns, trends
        - LSTM: Complex temporal dependencies
        """

        def detect_anomalies_prophet(self, metric_name, time_series,
                                     confidence_interval=0.95):
            """
            Detect anomalies using Facebook Prophet

            Prophet decomposes time series into:
            - Trend
            - Seasonal patterns (daily, weekly, yearly)
            - Holidays/events
            """
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': [point['timestamp'] for point in time_series],
                'y': [point['value'] for point in time_series]
            })

            # Train Prophet model
            model = Prophet(
                interval_width=confidence_interval,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            model.fit(df)

            # Generate forecast
            forecast = model.predict(df)

            # Detect anomalies (values outside confidence interval)
            anomalies = []
            for i, row in df.iterrows():
                observed = row['y']
                expected = forecast.loc[i, 'yhat']
                lower_bound = forecast.loc[i, 'yhat_lower']
                upper_bound = forecast.loc[i, 'yhat_upper']

                if observed < lower_bound or observed > upper_bound:
                    # Compute confidence (how far outside bounds)
                    if observed < lower_bound:
                        deviation = (lower_bound - observed) / (expected - lower_bound + 1e-9)
                    else:
                        deviation = (observed - upper_bound) / (upper_bound - expected + 1e-9)

                    confidence = min(1.0, deviation / 3.0)  # Normalize to [0, 1]

                    anomalies.append({
                        'timestamp': row['ds'],
                        'metric_name': metric_name,
                        'observed_value': observed,
                        'expected_value': expected,
                        'expected_range': [lower_bound, upper_bound],
                        'anomaly_type': 'drop' if observed < lower_bound else 'spike',
                        'confidence': confidence,
                        'detection_method': 'prophet',
                        'context': {
                            'trend': forecast.loc[i, 'trend'],
                            'seasonal': forecast.loc[i, 'seasonal'],
                            'weekly': forecast.loc[i, 'weekly'] if 'weekly' in forecast else None
                        }
                    })

            return anomalies
    ```

    ---

    ## 3. Data Profiling Engine

    ```python
    class DataProfiler:
        """
        Generate comprehensive data profiles

        Profiles include:
        - Row count, column count, size
        - Column statistics (min, max, mean, median, stddev)
        - Histograms and percentiles
        - Null ratios, unique ratios
        - Data type distribution
        - Pattern analysis (strings)
        """

        def profile_dataset(self, dataset):
            """Generate complete dataset profile"""
            profile = {
                'dataset': dataset,
                'timestamp': datetime.utcnow(),
                'table_stats': self.profile_table(dataset),
                'column_profiles': []
            }

            # Get columns
            columns = self.get_columns(dataset)

            for column in columns:
                col_profile = self.profile_column(dataset, column)
                profile['column_profiles'].append(col_profile)

            # Compute quality score
            profile['quality_score'] = self.compute_quality_score(profile)

            return profile

        def profile_table(self, dataset):
            """Profile table-level statistics"""
            query = f"""
                SELECT
                    COUNT(*) as row_count,
                    pg_total_relation_size('{dataset}') as size_bytes
                FROM {dataset}
            """

            result = self.db.execute(query)

            return {
                'row_count': result['row_count'],
                'size_bytes': result['size_bytes'],
                'column_count': len(self.get_columns(dataset))
            }

        def profile_column(self, dataset, column):
            """Profile individual column"""
            col_type = self.get_column_type(dataset, column)

            if col_type in ['int', 'bigint', 'numeric', 'float', 'double']:
                return self.profile_numeric_column(dataset, column)
            elif col_type in ['varchar', 'text', 'string']:
                return self.profile_string_column(dataset, column)
            elif col_type in ['timestamp', 'date', 'datetime']:
                return self.profile_datetime_column(dataset, column)
            else:
                return self.profile_generic_column(dataset, column)

        def profile_numeric_column(self, dataset, column):
            """Profile numeric column with statistics"""
            query = f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT({column}) as non_null_count,
                    SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) as null_count,
                    COUNT(DISTINCT {column}) as unique_count,
                    MIN({column}) as min_value,
                    MAX({column}) as max_value,
                    AVG({column}) as mean_value,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) as median_value,
                    STDDEV({column}) as stddev_value,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as p25,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as p75,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {column}) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {column}) as p99
                FROM {dataset}
            """

            result = self.db.execute(query)

            # Generate histogram
            histogram = self.generate_histogram(dataset, column, num_bins=20)

            return {
                'name': column,
                'type': 'numeric',
                'total_rows': result['total_rows'],
                'null_count': result['null_count'],
                'null_ratio': result['null_count'] / result['total_rows'],
                'unique_count': result['unique_count'],
                'unique_ratio': result['unique_count'] / result['total_rows'],
                'statistics': {
                    'min': result['min_value'],
                    'max': result['max_value'],
                    'mean': result['mean_value'],
                    'median': result['median_value'],
                    'stddev': result['stddev_value'],
                    'p25': result['p25'],
                    'p75': result['p75'],
                    'p95': result['p95'],
                    'p99': result['p99']
                },
                'histogram': histogram
            }

        def profile_string_column(self, dataset, column):
            """Profile string column with patterns"""
            query = f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT({column}) as non_null_count,
                    COUNT(DISTINCT {column}) as unique_count,
                    MIN(LENGTH({column})) as min_length,
                    MAX(LENGTH({column})) as max_length,
                    AVG(LENGTH({column})) as avg_length
                FROM {dataset}
            """

            result = self.db.execute(query)

            # Top values
            top_values = self.get_top_values(dataset, column, limit=10)

            # Pattern detection (e.g., email domains)
            patterns = self.detect_patterns(dataset, column)

            return {
                'name': column,
                'type': 'string',
                'total_rows': result['total_rows'],
                'null_count': result['total_rows'] - result['non_null_count'],
                'null_ratio': (result['total_rows'] - result['non_null_count']) / result['total_rows'],
                'unique_count': result['unique_count'],
                'unique_ratio': result['unique_count'] / result['total_rows'],
                'value_length': {
                    'min': result['min_length'],
                    'max': result['max_length'],
                    'mean': result['avg_length']
                },
                'top_values': top_values,
                'patterns': patterns
            }

        def compute_quality_score(self, profile):
            """
            Compute overall quality score (0.0 to 1.0)

            Factors:
            - Null ratio (lower is better)
            - Unique ratio for key columns
            - Data freshness
            - Completeness
            """
            score = 1.0

            for col in profile['column_profiles']:
                # Penalize high null ratios
                null_penalty = col['null_ratio'] * 0.1
                score -= null_penalty

            # Penalize stale data
            # ... (check max timestamp)

            return max(0.0, min(1.0, score))
    ```

    ---

    ## 4. Alert Deduplication and Routing

    ```python
    class AlertDeduplicator:
        """
        Reduce alert noise through intelligent deduplication

        Strategies:
        - Time-based grouping (same alert within N minutes)
        - Root cause grouping (upstream failure causes downstream alerts)
        - Severity-based suppression (low alerts during high alert storm)
        """

        def deduplicate_alerts(self, new_alerts):
            """Deduplicate new alerts against recent alerts"""
            deduplicated = []

            for alert in new_alerts:
                # Check if similar alert exists in last 1 hour
                similar = self.find_similar_alerts(
                    alert,
                    time_window=timedelta(hours=1)
                )

                if similar:
                    # Update existing alert (increment count)
                    self.update_alert_count(similar[0]['alert_id'])
                else:
                    # New unique alert
                    deduplicated.append(alert)

            return deduplicated

        def find_similar_alerts(self, alert, time_window):
            """Find alerts with same dataset and rule within time window"""
            query = """
                SELECT alert_id, created_at, occurrence_count
                FROM alerts
                WHERE dataset_id = %s
                  AND rule_id = %s
                  AND status IN ('open', 'acknowledged')
                  AND created_at > %s
                ORDER BY created_at DESC
                LIMIT 1
            """

            cutoff_time = datetime.utcnow() - time_window

            return self.db.execute(query, [
                alert['dataset_id'],
                alert['rule_id'],
                cutoff_time
            ])

        def group_by_root_cause(self, alerts):
            """
            Group alerts by root cause using lineage

            Example:
            - Table A fails freshness check
            - Tables B, C, D (downstream of A) also fail
            - Group all 4 alerts under root cause: Table A
            """
            # Build alert graph based on lineage
            alert_graph = self.build_alert_graph(alerts)

            # Find root nodes (no upstream failures)
            root_alerts = [a for a in alerts if not alert_graph[a['alert_id']]['upstream']]

            # Group alerts by root
            incidents = []
            for root in root_alerts:
                # Find all downstream alerts
                downstream = self.find_downstream_alerts(root, alert_graph)

                incidents.append({
                    'incident_id': self.generate_incident_id(),
                    'root_alert': root,
                    'related_alerts': downstream,
                    'severity': root['severity'],
                    'impact_score': self.compute_impact_score(root, downstream)
                })

            return incidents

        def compute_impact_score(self, root_alert, related_alerts):
            """
            Compute impact score (0.0 to 1.0)

            Factors:
            - Number of affected datasets
            - Criticality of datasets
            - Number of downstream consumers
            """
            num_affected = 1 + len(related_alerts)

            # Get downstream consumers
            total_consumers = 0
            for alert in [root_alert] + related_alerts:
                consumers = self.get_dataset_consumers(alert['dataset_id'])
                total_consumers += len(consumers)

            # Normalize to [0, 1]
            impact = min(1.0, (num_affected / 10) * 0.5 + (total_consumers / 100) * 0.5)

            return impact
    ```

    ```python
    class AlertRouter:
        """
        Intelligent alert routing to appropriate teams/channels

        Routing logic:
        - Dataset ownership (from metadata)
        - Alert severity (critical → PagerDuty, low → Email)
        - Business hours (off-hours → on-call only)
        - Escalation (no acknowledgment after 30 min → escalate)
        """

        def route_alert(self, alert):
            """Route alert to appropriate channels"""
            # Get dataset owner
            owner = self.get_dataset_owner(alert['dataset_id'])

            # Determine channels based on severity
            channels = self.determine_channels(alert['severity'])

            # Check business hours
            if not self.is_business_hours() and alert['severity'] in ['low', 'medium']:
                # Suppress low/medium alerts off-hours
                channels = []

            # Send notifications
            for channel in channels:
                self.send_notification(channel, alert, owner)

            # Schedule escalation if critical
            if alert['severity'] == 'critical':
                self.schedule_escalation(alert, owner, delay_minutes=30)

        def determine_channels(self, severity):
            """Map severity to notification channels"""
            channel_map = {
                'low': ['email'],
                'medium': ['email', 'slack'],
                'high': ['email', 'slack', 'webhook'],
                'critical': ['email', 'slack', 'pagerduty']
            }

            return channel_map.get(severity, ['email'])

        def send_notification(self, channel, alert, owner):
            """Send notification to specific channel"""
            if channel == 'slack':
                self.send_slack_notification(alert, owner)
            elif channel == 'email':
                self.send_email_notification(alert, owner)
            elif channel == 'pagerduty':
                self.send_pagerduty_alert(alert, owner)
            elif channel == 'webhook':
                self.send_webhook(alert, owner)

        def send_slack_notification(self, alert, owner):
            """Format and send Slack message"""
            message = {
                "text": f"🚨 Data Quality Alert: {alert['title']}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{self.severity_emoji(alert['severity'])} {alert['title']}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Dataset:*\n{alert['dataset_id']}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n{alert['severity'].upper()}"},
                            {"type": "mrkdwn", "text": f"*Owner:*\n<@{owner['slack_user_id']}>"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{alert['created_at']}"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Description:*\n{alert['description']}"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Acknowledge"},
                                "value": alert['alert_id'],
                                "action_id": "acknowledge_alert"
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "View Lineage"},
                                "url": f"{self.base_url}/lineage/{alert['dataset_id']}",
                                "action_id": "view_lineage"
                            }
                        ]
                    }
                ]
            }

            self.slack_client.post_message(
                channel=owner['slack_channel'],
                message=message
            )

        def severity_emoji(self, severity):
            """Map severity to emoji"""
            emoji_map = {
                'low': '🔵',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }
            return emoji_map.get(severity, '⚪')
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Distributed Validation

    ```python
    class DistributedValidator:
        """
        Distribute validation across multiple workers

        Strategies:
        - Partition by dataset (shard validation rules)
        - Parallel execution per dataset
        - Sampling for large tables (validate 1M rows instead of 1B)
        """

        def schedule_validations(self, rules, num_workers=100):
            """Distribute validation rules across workers"""
            # Partition rules by dataset hash
            partitions = [[] for _ in range(num_workers)]

            for rule in rules:
                worker_id = hash(rule['dataset_id']) % num_workers
                partitions[worker_id].append(rule)

            # Dispatch to workers (via Kafka)
            for worker_id, rule_batch in enumerate(partitions):
                self.kafka_producer.send(
                    topic=f'validation-worker-{worker_id}',
                    value={'rules': rule_batch}
                )
    ```

    ### Incremental Profiling

    ```python
    class IncrementalProfiler:
        """
        Avoid full table scans by profiling incrementally

        Strategies:
        - Profile only new partitions (dt > last_profiled_date)
        - Update statistics incrementally (merge with existing)
        - Sample large tables (profile 1M rows, extrapolate)
        """

        def incremental_profile(self, dataset):
            """Profile only new data since last profile"""
            last_profile = self.get_last_profile(dataset)

            if not last_profile:
                # First profile, full scan
                return self.full_profile(dataset)

            # Profile only new partition
            new_partition = self.get_new_partition(dataset, last_profile['timestamp'])
            partition_profile = self.profile_partition(dataset, new_partition)

            # Merge with existing profile
            merged_profile = self.merge_profiles(last_profile, partition_profile)

            return merged_profile
    ```

    ### Query Optimization

    ```python
    class OptimizedValidator:
        """
        Optimize validation queries for large datasets

        Techniques:
        - Push down predicates (filter early)
        - Use approximate algorithms (HyperLogLog for COUNT DISTINCT)
        - Partition pruning
        - Materialize common subqueries
        """

        def validate_with_sampling(self, dataset, rule, sample_ratio=0.01):
            """Validate using sample for large tables"""
            if self.get_table_size(dataset) < 1_000_000:
                # Small table, full validation
                return self.validate_full(dataset, rule)

            # Large table, sample validation
            query = f"""
                SELECT *
                FROM {dataset}
                TABLESAMPLE BERNOULLI ({sample_ratio * 100})  -- Sample 1%
            """

            result = self.execute_validation_query(query, rule)

            # Extrapolate to full table
            result['estimated_failures'] = result['failure_count'] / sample_ratio

            return result
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Sampling validation** | 100x faster for large tables | Lower accuracy, may miss rare issues |
    | **Incremental profiling** | 10x faster for partitioned data | More complex logic, requires metadata |
    | **Alert deduplication** | 90% reduction in alert volume | May suppress legitimate unique alerts |
    | **Approximate algorithms** | 10x faster COUNT DISTINCT | ±2% accuracy loss |
    | **Query result caching** | 5x faster repeated validations | Stale results if data changes |
    | **Parallel execution** | 10x throughput with 10 workers | Higher infrastructure cost |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (10K datasets, 1M checks/day):

    Compute:
    - 25 validation nodes × $100 = $2,500
    - 15 anomaly detection nodes × $150 = $2,250
    - 15 profiling nodes × $100 = $1,500
    - 10 database nodes × $200 = $2,000
    - 3 alert manager nodes × $50 = $150
    - Total compute: $8,400/month

    Storage:
    - Metadata (15 TB): 15 × $23 (S3) = $345
    - Time-series (150 GB): InfluxDB Cloud $500
    - Graph DB (Neo4j): $1,000
    - Total storage: $1,845/month

    Data warehouse query costs:
    - 1M queries/day × $5/TB scanned = $5,000/month
    - With sampling (90% reduction): $500/month

    Total: ~$10,745/month

    Optimizations:
    1. Sample large tables (reduce query cost 90%): -$4,500
    2. Incremental profiling (reduce compute 50%): -$750
    3. Spot instances for profiling (70% discount): -$1,050
    4. Cache validation results (reduce queries 30%): -$1,350

    Optimized Total: ~$3,095/month (70% reduction)
    ```

    ---

    ## Monitoring Metrics

    ```python
    # Key metrics for data quality platform health

    # Validation metrics
    validation_runs_total{dataset, rule, status}  # status: success, failure
    validation_duration_seconds{dataset, rule}
    validation_failure_rate{dataset}

    # Anomaly detection metrics
    anomalies_detected_total{dataset, metric, severity}
    anomaly_detection_latency_seconds
    false_positive_rate{detection_method}

    # Alert metrics
    alerts_generated_total{severity, status}
    alerts_deduplicated_total
    alert_acknowledgment_time_seconds
    alert_resolution_time_seconds

    # Data quality metrics
    data_quality_score{dataset}  # 0.0 to 1.0
    sla_compliance_rate{dataset}  # % of time within SLA
    data_freshness_seconds{dataset}

    # System metrics
    validation_queue_depth
    profiling_queue_depth
    validation_worker_utilization
    ```

=== "💡 Step 5: Additional Considerations"

    ## Data Lineage Integration

    ```python
    class LineageIntegration:
        """
        Integration with data lineage for impact analysis

        Use cases:
        - Root cause: Find upstream source of quality issue
        - Impact analysis: Which dashboards are affected?
        - Prioritization: Critical path datasets get higher priority
        """

        def analyze_impact(self, failed_dataset):
            """Analyze downstream impact of failed dataset"""
            # Get downstream dependencies
            downstream = self.lineage_graph.get_downstream(failed_dataset)

            impact = {
                'affected_datasets': len(downstream),
                'affected_dashboards': [],
                'affected_ml_models': [],
                'estimated_users_impacted': 0
            }

            for dataset in downstream:
                # Check consumers
                consumers = self.get_consumers(dataset)

                for consumer in consumers:
                    if consumer['type'] == 'dashboard':
                        impact['affected_dashboards'].append(consumer)
                        impact['estimated_users_impacted'] += consumer['user_count']
                    elif consumer['type'] == 'ml_model':
                        impact['affected_ml_models'].append(consumer)

            return impact
    ```

    ---

    ## Auto-Remediation

    ```python
    class AutoRemediator:
        """
        Automatically fix common data quality issues

        Safe remediations:
        - Re-run failed ETL jobs
        - Delete and reload stale partitions
        - Refresh materialized views
        - Trigger data backfill
        """

        def attempt_remediation(self, alert):
            """Attempt to auto-fix issue"""
            if alert['alert_type'] == 'stale_data':
                # Trigger ETL pipeline to refresh data
                return self.trigger_etl_refresh(alert['dataset_id'])

            elif alert['alert_type'] == 'schema_change':
                # Cannot auto-fix, requires human review
                return {'auto_fix': False, 'reason': 'Schema changes require manual review'}

            elif alert['alert_type'] == 'validation_failure':
                # Check if transient issue
                retry_result = self.retry_validation(alert['rule_id'])
                if retry_result['success']:
                    return {'auto_fix': True, 'action': 'transient_issue_resolved'}

        def trigger_etl_refresh(self, dataset_id):
            """Trigger ETL pipeline to refresh dataset"""
            # Call Airflow API to trigger DAG
            dag_id = self.get_etl_dag_for_dataset(dataset_id)

            self.airflow_client.trigger_dag(
                dag_id=dag_id,
                conf={'reason': 'auto_remediation', 'dataset': dataset_id}
            )

            return {'auto_fix': True, 'action': 'etl_refresh_triggered'}
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you reduce false positives in anomaly detection?"**
   - Use adaptive thresholds (learn from historical data)
   - Combine multiple detection methods (Z-score + ML)
   - Incorporate domain knowledge (expected seasonal patterns)
   - Feedback loop (users mark false positives → adjust model)
   - Confidence scoring (only alert on high-confidence anomalies)

2. **"How do you handle schema changes?"**
   - Track schema history (version control)
   - Detect breaking changes (column removed, type changed)
   - Alert on unexpected schema drift
   - Provide schema evolution recommendations
   - Integration with schema registry (Confluent Schema Registry)

3. **"How do you scale validation to 10,000+ datasets?"**
   - Distributed validation workers (100+ nodes)
   - Sampling for large tables (validate 1% of rows)
   - Incremental profiling (only profile new partitions)
   - Query optimization (push down predicates, materialize)
   - Priority queue (critical datasets first)

4. **"How do you prioritize alerts?"**
   - Severity (critical > high > medium > low)
   - Impact score (downstream dependencies × consumer count)
   - Dataset criticality (SLA tier, business value)
   - Historical failure rate (frequent failures → lower priority)
   - Time of day (business hours vs. off-hours)

5. **"How do you track data lineage?"**
   - Parse SQL queries (extract table references)
   - Integration with ETL tools (Airflow, dbt metadata)
   - Manual registration (API for custom pipelines)
   - Graph database for efficient traversal (Neo4j)
   - Visualize lineage (Mermaid diagrams, interactive graphs)

6. **"How do you handle large-scale data profiling?"**
   - Sample data (profile 1M rows, extrapolate)
   - Incremental updates (only profile new data)
   - Approximate algorithms (HyperLogLog for cardinality)
   - Distributed profiling (Spark for PB-scale data)
   - Schedule during off-peak hours

7. **"How do you measure data quality over time?"**
   - Quality score (0.0 to 1.0 per dataset)
   - SLA compliance rate (% of time within SLA)
   - Trend analysis (quality improving or degrading?)
   - Historical validation results (pass rate over time)
   - Dashboard with time-series charts

8. **"How do you integrate with existing tools?"**
   - Great Expectations (validation framework)
   - dbt tests (SQL-based validation)
   - Airflow (trigger validation after ETL)
   - Slack/PagerDuty (alert delivery)
   - DataHub/Amundsen (metadata integration)

**Key Points to Mention:**

- Rule-based + ML hybrid approach
- Statistical (Z-score, IQR) + ML (Prophet, LSTM) anomaly detection
- Comprehensive profiling (histograms, percentiles, patterns)
- Intelligent alert deduplication (90% noise reduction)
- Lineage-aware impact analysis
- Distributed execution for scalability
- Sampling and incremental profiling for performance
- Auto-remediation for common issues

---

## Real-World Examples

**Great Expectations:**
- Open-source validation framework
- 50+ built-in expectations
- Integration with Airflow, dbt, Spark
- Declarative validation as code

**Monte Carlo Data:**
- ML-based anomaly detection
- Automated lineage discovery
- Incident management workflow
- SaaS platform (no infrastructure)

**AWS Deequ:**
- Spark-based validation at scale
- Incremental metric computation
- Integration with AWS Glue
- Open-source (Apache 2.0)

**Soda:**
- SQL-based data testing
- YAML configuration
- Integration with data warehouses
- dbt-native support

---

## Summary

**System Characteristics:**

- **Scale:** 10,000+ datasets, 1M+ checks/day, PB-scale data
- **Throughput:** 1,000 checks/sec (distributed), 100 profiles/day
- **Latency:** < 5 seconds validation, < 1 minute anomaly detection
- **Accuracy:** < 5% false positive rate (with ML tuning)
- **Availability:** 99.9% uptime

**Core Components:**

1. **Validation Engine:** SQL/Python rules, Great Expectations integration
2. **Anomaly Detection:** Statistical (Z-score, IQR) + ML (Prophet, LSTM)
3. **Data Profiler:** Statistics, histograms, patterns, quality scores
4. **Alert Manager:** Deduplication, prioritization, intelligent routing
5. **Lineage Tracker:** Graph DB for impact analysis, root cause tracing
6. **Time-Series Store:** Historical metrics, trend analysis

**Key Design Decisions:**

- Hybrid detection (rules + ML)
- Distributed execution (100+ workers)
- Sampling for large tables (100x speedup)
- Alert deduplication (90% noise reduction)
- Lineage-aware prioritization
- Incremental profiling (10x faster)
- Auto-remediation for safe fixes

This design provides a comprehensive, scalable data quality platform capable of monitoring thousands of datasets, detecting anomalies in near real-time, and intelligently routing alerts to reduce noise while ensuring critical issues are addressed promptly.
