# Design ETL Pipeline System

A distributed, scalable data pipeline orchestration system that extracts data from multiple sources, transforms it using SQL/Python, and loads it into data warehouses/lakes with DAG-based scheduling, dependency management, and comprehensive monitoring.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10,000+ pipelines, 10M+ tasks/day, 100K+ data sources, petabytes of data |
| **Key Challenges** | DAG scheduling, distributed execution, data lineage, incremental loading, SLA monitoring, fault tolerance |
| **Core Concepts** | Directed Acyclic Graph (DAG), workflow orchestration, idempotent tasks, backfill, data partitioning |
| **Companies** | Airflow, dbt, Fivetran, Airbyte, Prefect, Dagster, Luigi, AWS Glue, Azure Data Factory |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **DAG Definition** | Define pipelines as directed acyclic graphs | P0 (Must have) |
    | **Multi-Source Extract** | Pull from databases, APIs, files, streams | P0 (Must have) |
    | **Transform Engine** | SQL/Python transformations (dbt, Spark) | P0 (Must have) |
    | **Data Loading** | Load to warehouses (Snowflake, BigQuery, Redshift) | P0 (Must have) |
    | **Task Scheduling** | Cron-based and event-driven scheduling | P0 (Must have) |
    | **Dependency Resolution** | Execute tasks based on dependencies | P0 (Must have) |
    | **Error Handling** | Automatic retries with backoff | P0 (Must have) |
    | **Incremental Loading** | Only process new/changed data | P0 (Must have) |
    | **Data Lineage** | Track data flow from source to destination | P1 (Should have) |
    | **SLA Monitoring** | Alert on missed deadlines | P1 (Should have) |
    | **Backfill Support** | Re-run historical data loads | P1 (Should have) |
    | **Data Quality Checks** | Validation rules and data testing | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Real-time streaming (use Kafka/Flink instead)
    - Data catalog/discovery (use DataHub/Amundsen)
    - Advanced ML pipeline orchestration (use Kubeflow/MLflow)
    - Data governance/security (use external tools)
    - Visual DAG builder (code-first approach)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Task Throughput** | 10M tasks/day | Support large-scale data operations |
    | **DAG Count** | 10,000+ active DAGs | Multi-tenant platform |
    | **Scheduling Latency** | < 1 second | Fast task dispatch |
    | **Availability** | 99.9% uptime | Critical data pipelines |
    | **Concurrency** | 10,000+ parallel tasks | Efficient resource utilization |
    | **Scalability** | Horizontal scaling of workers | Handle growing workloads |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Pipeline scale:
    - Total active DAGs: 10,000 pipelines
    - Average tasks per DAG: 20 tasks
    - Average DAG runs per day: 50 runs (some run hourly, some daily)
    - Total task executions: 10,000 × 20 × 50 = 10M tasks/day
    - Peak throughput: 10M / 86,400 = 116 tasks/sec (avg), 1,000 tasks/sec (peak)

    Data volume:
    - Average data per task: 10 GB extracted
    - Total daily data: 10M tasks × 10 GB = 100 PB/day processed
    - Incremental loading: ~90% reduction → 10 PB/day actual transfer

    Scheduling:
    - Cron-based DAGs: 8,000 DAGs (80%)
    - Event-driven DAGs: 2,000 DAGs (20%)
    - Scheduler evaluations: 10,000 DAGs × 1 eval/sec = 10,000 evals/sec
    - Task state updates: 10M tasks × 10 state changes = 100M updates/day

    Worker capacity:
    - Worker nodes: 1,000 nodes
    - Tasks per worker: 10 concurrent tasks
    - Total capacity: 10,000 parallel tasks
    - Average task duration: 5 minutes
    - Throughput: 10,000 tasks / 5 min = 2,000 tasks/min = 33 tasks/sec
    ```

    ### Storage Estimates

    ```
    Metadata storage (PostgreSQL):

    DAG metadata:
    - 10,000 DAGs × 50 KB = 500 MB

    Task instances (execution history):
    - 10M tasks/day × 365 days = 3.65B task instances/year
    - Per task: 2 KB (task_id, dag_id, state, timestamps, logs pointer)
    - Total: 3.65B × 2 KB = 7.3 TB/year

    Task logs:
    - 10M tasks/day × 100 KB logs = 1 TB/day logs
    - Retention: 90 days → 90 TB logs
    - Compressed (5:1): 18 TB

    XCom (task communication):
    - 10M tasks/day × 10% use XCom × 10 KB = 10 GB/day
    - Retention: 7 days → 70 GB

    Data lineage:
    - 10M tasks × 5 lineage edges × 500 bytes = 25 GB/day
    - Retention: 1 year → 9 TB

    Total metadata: 7.3 TB (tasks) + 18 TB (logs) + 9 TB (lineage) = 34.3 TB
    With replication (3x): ~100 TB

    Data storage (raw/processed):
    - Raw data (S3/GCS): 10 PB/day × 30 days = 300 PB
    - Processed data (warehouse): 1 PB/day × 365 days = 365 PB
    - Total data storage: ~700 PB (separate from metadata)
    ```

    ### Bandwidth Estimates

    ```
    Data transfer:
    - Extract from sources: 10 PB/day = 115 GB/sec = 920 Gbps
    - Load to warehouse: 1 PB/day = 11.5 GB/sec = 92 Gbps
    - Total bandwidth: ~1 Tbps (split across 1,000 workers)
    - Per worker: ~1 Gbps

    Metadata traffic:
    - Task state updates: 100M updates/day × 1 KB = 100 GB/day
    - Scheduler heartbeats: 1,000 workers × 1 KB/sec = 1 MB/sec = 86 GB/day
    - Total metadata: ~200 GB/day = 2.3 MB/sec (negligible)
    ```

    ### Server Estimates

    ```
    Scheduler nodes:
    - DAG parsing: 10,000 DAGs / 1,000 DAGs per node = 10 nodes
    - Task scheduling: 1,000 tasks/sec / 100 tasks per node = 10 nodes
    - Total schedulers: 20 nodes (with HA)
    - CPU: 16 cores per node (parsing, dependency resolution)
    - Memory: 64 GB per node (DAG objects, task queue)

    Worker nodes (task execution):
    - 10,000 parallel tasks / 10 tasks per worker = 1,000 workers
    - CPU: 8 cores per worker (data processing)
    - Memory: 32 GB per worker (data buffering)
    - Network: 1 Gbps per worker

    Metadata database (PostgreSQL):
    - Primary: 1 node (64 cores, 256 GB RAM, 10 TB SSD)
    - Read replicas: 5 nodes (for queries, monitoring)
    - Total: 6 database nodes

    Message queue (RabbitMQ/Redis):
    - 3 nodes for HA (task queue, pub/sub)
    - 16 cores, 64 GB RAM per node

    Object storage (S3/GCS):
    - Managed service (no dedicated nodes)
    - 700 PB storage

    Total infrastructure:
    - Schedulers: 20 nodes
    - Workers: 1,000 nodes
    - Databases: 6 nodes
    - Message queue: 3 nodes
    - Total: ~1,030 nodes
    ```

    ---

    ## Key Assumptions

    1. 10M tasks executed per day (10,000 DAGs × 20 tasks × 50 runs)
    2. 90% of data transfers are incremental (not full refreshes)
    3. Average task duration: 5 minutes
    4. Task failure rate: 5% (with retries)
    5. Peak load is 10x average (during business hours)
    6. Metadata retention: 1 year for task history
    7. 80% of DAGs are scheduled (cron), 20% are event-driven

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **DAG-based orchestration:** Tasks organized in directed acyclic graphs
    2. **Distributed execution:** Horizontal scaling of worker nodes
    3. **Idempotent tasks:** Safe to retry without side effects
    4. **Event-driven scheduling:** React to data availability, not just time
    5. **Separation of concerns:** Scheduler, executor, metadata store
    6. **Data lineage tracking:** End-to-end visibility of data flow

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Data Sources"
            DB1[(MySQL<br/>Database)]
            DB2[(PostgreSQL<br/>Database)]
            API1[REST APIs]
            S3Source[S3 Buckets]
            Kafka1[Kafka Topics]
        end

        subgraph "Orchestration Layer"
            WebUI[Web UI<br/>DAG Monitoring<br/>Logs/Metrics]
            DAGAPI[DAG API<br/>Submit DAGs<br/>Trigger runs]

            subgraph "Scheduler Cluster"
                Scheduler1[Scheduler 1<br/>DAG parsing<br/>Task scheduling]
                Scheduler2[Scheduler 2<br/>DAG parsing<br/>Task scheduling]
                SchedulerN[Scheduler N<br/>DAG parsing<br/>Task scheduling]
            end

            LB[Load Balancer]
        end

        subgraph "Execution Layer"
            subgraph "Worker Pool 1 - Extract"
                Worker1[Worker 1<br/>Extract tasks<br/>DB connectors]
                Worker2[Worker 2<br/>Extract tasks<br/>API connectors]
            end

            subgraph "Worker Pool 2 - Transform"
                Worker3[Worker 3<br/>dbt transform<br/>SQL engine]
                Worker4[Worker 4<br/>Spark jobs<br/>Python transform]
            end

            subgraph "Worker Pool 3 - Load"
                Worker5[Worker 5<br/>Load to warehouse<br/>Bulk insert]
                Worker6[Worker 6<br/>Load to warehouse<br/>Upsert/merge]
            end
        end

        subgraph "Metadata & State"
            MetaDB[(Metadata DB<br/>PostgreSQL<br/>DAGs, Tasks, Runs<br/>Lineage, SLAs)]
            TaskQueue[Task Queue<br/>Redis/RabbitMQ<br/>Pending tasks]
            ResultBackend[Result Backend<br/>Redis<br/>XCom, State]
        end

        subgraph "Data Storage"
            RawData[(Raw Data Lake<br/>S3/GCS<br/>Extracted data<br/>Partitioned)]
            StagingData[(Staging Area<br/>S3/GCS<br/>Transformed data<br/>Temporary)]
        end

        subgraph "Data Warehouse"
            Snowflake[(Snowflake)]
            BigQuery[(BigQuery)]
            Redshift[(Redshift)]
        end

        subgraph "Monitoring & Observability"
            Metrics[Metrics<br/>Prometheus<br/>Task duration<br/>Success rate]
            Logs[Logs<br/>ELK Stack<br/>Task logs<br/>Error traces]
            Alerts[Alerts<br/>SLA breaches<br/>Task failures]
            Lineage[Lineage Tracker<br/>Data provenance<br/>Impact analysis]
        end

        DB1 -->|JDBC| Worker1
        DB2 -->|JDBC| Worker1
        API1 -->|HTTP| Worker2
        S3Source -->|S3 API| Worker2
        Kafka1 -->|Consumer| Worker2

        Worker1 --> RawData
        Worker2 --> RawData

        RawData --> Worker3
        RawData --> Worker4
        Worker3 --> StagingData
        Worker4 --> StagingData

        StagingData --> Worker5
        StagingData --> Worker6

        Worker5 --> Snowflake
        Worker5 --> BigQuery
        Worker6 --> Redshift

        WebUI --> LB
        DAGAPI --> LB
        LB --> Scheduler1
        LB --> Scheduler2
        LB --> SchedulerN

        Scheduler1 --> MetaDB
        Scheduler2 --> MetaDB
        SchedulerN --> MetaDB

        Scheduler1 --> TaskQueue
        Scheduler2 --> TaskQueue
        SchedulerN --> TaskQueue

        TaskQueue --> Worker1
        TaskQueue --> Worker2
        TaskQueue --> Worker3
        TaskQueue --> Worker4
        TaskQueue --> Worker5
        TaskQueue --> Worker6

        Worker1 --> ResultBackend
        Worker3 --> ResultBackend
        Worker5 --> ResultBackend

        Worker1 --> Lineage
        Worker3 --> Lineage
        Worker5 --> Lineage

        Scheduler1 --> Metrics
        Worker1 --> Metrics
        Worker3 --> Metrics

        Worker1 --> Logs
        Worker3 --> Logs

        Scheduler1 --> Alerts
        MetaDB --> Alerts

        style Scheduler1 fill:#e1f5ff
        style Worker1 fill:#e8f5e9
        style Worker3 fill:#fff9c4
        style Worker5 fill:#fce4ec
        style MetaDB fill:#ffe1e1
        style RawData fill:#f3e5f5
        style Snowflake fill:#e0f2f1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **DAG Definition (Python)** | Programmatic pipeline creation, version control | YAML config (less flexible), GUI builder (not code-first) |
    | **Distributed Scheduler** | Scale to 10K+ DAGs, HA for critical pipelines | Single scheduler (SPOF, limited scale) |
    | **Worker Pools** | Isolate workload types, resource optimization | Single pool (noisy neighbors, inefficient) |
    | **Task Queue (Redis)** | Decouple scheduler from workers, buffering | Direct RPC (tight coupling), database queue (slow) |
    | **Metadata DB (PostgreSQL)** | ACID guarantees, complex queries for lineage | NoSQL (no joins), file-based (no concurrency) |
    | **XCom / Result Backend** | Share small data between tasks | External storage (slow), database (not optimized) |
    | **S3 for staging** | Cheap storage, decoupled from compute | EBS volumes (expensive), HDFS (operational overhead) |

    **Key Trade-off:** We chose **eventually consistent task state** (via message queue) over strong consistency to achieve high throughput. This means tasks may be dispatched twice during failures, so tasks must be idempotent.

    ---

    ## API Design

    ### 1. Define DAG (Programmatic)

    **Airflow-style DAG definition:**

    ```python
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.sql import SQLExecuteQueryOperator
    from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
    from datetime import datetime, timedelta

    # Define default arguments
    default_args = {
        'owner': 'data-team',
        'depends_on_past': False,
        'start_date': datetime(2025, 1, 1),
        'email': ['data-team@company.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'sla': timedelta(hours=2)  # Must complete within 2 hours
    }

    # Create DAG
    dag = DAG(
        dag_id='user_analytics_pipeline',
        default_args=default_args,
        description='Extract user data, transform, load to Snowflake',
        schedule_interval='0 2 * * *',  # Daily at 2 AM
        catchup=False,  # Don't backfill missed runs
        max_active_runs=1,
        tags=['analytics', 'users', 'snowflake']
    )

    # Task 1: Extract from MySQL
    extract_users = PythonOperator(
        task_id='extract_users_from_mysql',
        python_callable=extract_users_function,
        op_kwargs={
            'source_db': 'mysql://prod-db:3306/users',
            'query': 'SELECT * FROM users WHERE updated_at >= {{ ds }}',
            'output_path': 's3://data-lake/raw/users/{{ ds }}/'
        },
        dag=dag
    )

    # Task 2: Extract from API
    extract_events = PythonOperator(
        task_id='extract_events_from_api',
        python_callable=extract_events_function,
        op_kwargs={
            'api_endpoint': 'https://api.company.com/events',
            'date': '{{ ds }}',
            'output_path': 's3://data-lake/raw/events/{{ ds }}/'
        },
        dag=dag
    )

    # Task 3: Transform with dbt
    transform_user_metrics = SQLExecuteQueryOperator(
        task_id='transform_user_metrics',
        conn_id='dbt_connection',
        sql='''
            dbt run --models user_daily_metrics
            --vars '{"execution_date": "{{ ds }}"}'
        ''',
        dag=dag
    )

    # Task 4: Data quality check
    def check_data_quality(**context):
        # Read transformed data
        df = read_from_s3(f"s3://data-lake/transformed/user_metrics/{context['ds']}/")

        # Assertions
        assert len(df) > 0, "No data found"
        assert df['user_id'].nunique() > 1000, "Too few users"
        assert df['revenue'].sum() > 0, "Revenue is zero"

        return True

    quality_check = PythonOperator(
        task_id='quality_check',
        python_callable=check_data_quality,
        dag=dag
    )

    # Task 5: Load to Snowflake
    load_to_warehouse = SnowflakeOperator(
        task_id='load_to_snowflake',
        snowflake_conn_id='snowflake_prod',
        sql='''
            COPY INTO analytics.user_daily_metrics
            FROM 's3://data-lake/transformed/user_metrics/{{ ds }}/'
            FILE_FORMAT = (TYPE = 'PARQUET')
            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
            ON_ERROR = 'ABORT_STATEMENT';
        ''',
        dag=dag
    )

    # Define dependencies (task execution order)
    [extract_users, extract_events] >> transform_user_metrics >> quality_check >> load_to_warehouse
    ```

    **Design Notes:**

    - Python code allows version control (Git)
    - Templating with Jinja2 (`{{ ds }}` = execution date)
    - Explicit task dependencies with `>>` operator
    - Retry logic with exponential backoff
    - SLA monitoring (alert if > 2 hours)

    ---

    ### 2. Trigger DAG Run

    **Request:**
    ```bash
    POST /api/v1/dags/user_analytics_pipeline/dagRuns
    Content-Type: application/json

    {
      "logical_date": "2025-01-15T00:00:00Z",
      "conf": {
        "source_db": "mysql://prod-db:3306/users",
        "target_warehouse": "snowflake"
      },
      "note": "Manual backfill for 2025-01-15"
    }
    ```

    **Response:**
    ```json
    {
      "dag_run_id": "manual__2025-01-15T00:00:00+00:00",
      "dag_id": "user_analytics_pipeline",
      "logical_date": "2025-01-15T00:00:00Z",
      "state": "queued",
      "external_trigger": true,
      "conf": {
        "source_db": "mysql://prod-db:3306/users",
        "target_warehouse": "snowflake"
      }
    }
    ```

    ---

    ### 3. Get Task Instance Status

    **Request:**
    ```bash
    GET /api/v1/dags/user_analytics_pipeline/dagRuns/manual__2025-01-15T00:00:00+00:00/taskInstances
    ```

    **Response:**
    ```json
    {
      "task_instances": [
        {
          "task_id": "extract_users_from_mysql",
          "state": "success",
          "start_date": "2025-01-15T02:00:05Z",
          "end_date": "2025-01-15T02:05:23Z",
          "duration": 318.5,
          "try_number": 1,
          "max_tries": 3
        },
        {
          "task_id": "extract_events_from_api",
          "state": "success",
          "start_date": "2025-01-15T02:00:05Z",
          "end_date": "2025-01-15T02:03:12Z",
          "duration": 187.2,
          "try_number": 1,
          "max_tries": 3
        },
        {
          "task_id": "transform_user_metrics",
          "state": "running",
          "start_date": "2025-01-15T02:05:25Z",
          "end_date": null,
          "duration": null,
          "try_number": 1,
          "max_tries": 3
        },
        {
          "task_id": "quality_check",
          "state": "none",
          "start_date": null,
          "end_date": null,
          "duration": null,
          "try_number": 0,
          "max_tries": 3
        },
        {
          "task_id": "load_to_snowflake",
          "state": "none",
          "start_date": null,
          "end_date": null,
          "duration": null,
          "try_number": 0,
          "max_tries": 3
        }
      ],
      "total_entries": 5
    }
    ```

    ---

    ### 4. Query Data Lineage

    **Request:**
    ```bash
    GET /api/v1/lineage/dataset?dataset=analytics.user_daily_metrics
    ```

    **Response:**
    ```json
    {
      "dataset": "analytics.user_daily_metrics",
      "upstream_datasets": [
        {
          "dataset": "s3://data-lake/raw/users/",
          "source": "mysql://prod-db:3306/users.users",
          "last_updated": "2025-01-15T02:05:23Z"
        },
        {
          "dataset": "s3://data-lake/raw/events/",
          "source": "https://api.company.com/events",
          "last_updated": "2025-01-15T02:03:12Z"
        }
      ],
      "transformations": [
        {
          "dag_id": "user_analytics_pipeline",
          "task_id": "transform_user_metrics",
          "code": "dbt run --models user_daily_metrics"
        }
      ],
      "downstream_datasets": [
        {
          "dataset": "analytics.user_monthly_summary",
          "dag_id": "monthly_aggregation_pipeline"
        }
      ]
    }
    ```

    ---

    ## Database Schema

    ### DAG Metadata

    ```sql
    CREATE TABLE dags (
        dag_id VARCHAR(250) PRIMARY KEY,
        description TEXT,
        schedule_interval VARCHAR(100),  -- Cron expression or timedelta
        is_paused BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        owners TEXT[],                   -- Array of owner emails
        tags TEXT[],
        default_args JSONB,              -- Default task arguments
        max_active_runs INT DEFAULT 1,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_dags_is_active ON dags(is_active) WHERE is_active = TRUE;
    CREATE INDEX idx_dags_tags ON dags USING GIN(tags);
    ```

    ### DAG Runs (Pipeline Executions)

    ```sql
    CREATE TABLE dag_runs (
        dag_run_id VARCHAR(250) PRIMARY KEY,  -- Format: scheduled__2025-01-15T00:00:00
        dag_id VARCHAR(250) REFERENCES dags(dag_id),
        execution_date TIMESTAMP NOT NULL,    -- Logical date for this run
        start_date TIMESTAMP,
        end_date TIMESTAMP,
        state VARCHAR(50) NOT NULL,           -- queued, running, success, failed
        run_type VARCHAR(50) NOT NULL,        -- scheduled, manual, backfill
        external_trigger BOOLEAN DEFAULT FALSE,
        conf JSONB,                            -- Runtime configuration
        data_interval_start TIMESTAMP,
        data_interval_end TIMESTAMP,

        UNIQUE(dag_id, execution_date)
    );

    CREATE INDEX idx_dag_runs_dag_id ON dag_runs(dag_id);
    CREATE INDEX idx_dag_runs_state ON dag_runs(state);
    CREATE INDEX idx_dag_runs_execution_date ON dag_runs(execution_date);
    ```

    ### Task Instances (Individual Task Executions)

    ```sql
    CREATE TABLE task_instances (
        task_id VARCHAR(250) NOT NULL,
        dag_id VARCHAR(250) NOT NULL,
        dag_run_id VARCHAR(250) REFERENCES dag_runs(dag_run_id),
        execution_date TIMESTAMP NOT NULL,
        state VARCHAR(50) NOT NULL,           -- none, scheduled, queued, running,
                                               -- success, failed, skipped, upstream_failed
        try_number INT DEFAULT 1,
        max_tries INT DEFAULT 3,
        start_date TIMESTAMP,
        end_date TIMESTAMP,
        duration FLOAT,                        -- Seconds
        hostname VARCHAR(255),                 -- Worker that executed task
        unixname VARCHAR(255),
        pool VARCHAR(256) NOT NULL,            -- Resource pool (e.g., 'default', 'high_memory')
        queue VARCHAR(256),                    -- Celery queue name
        priority_weight INT DEFAULT 1,
        operator VARCHAR(1000),                -- Operator class name
        queued_dttm TIMESTAMP,
        pid INT,
        log_url TEXT,                          -- S3 path to logs

        PRIMARY KEY (task_id, dag_id, dag_run_id, try_number)
    );

    CREATE INDEX idx_task_instances_state ON task_instances(state);
    CREATE INDEX idx_task_instances_pool ON task_instances(pool);
    CREATE INDEX idx_task_instances_dag_run_id ON task_instances(dag_run_id);
    ```

    ### Task Dependencies

    ```sql
    CREATE TABLE task_dependencies (
        task_id VARCHAR(250) NOT NULL,
        dag_id VARCHAR(250) NOT NULL,
        upstream_task_id VARCHAR(250) NOT NULL,
        upstream_dag_id VARCHAR(250) NOT NULL,

        PRIMARY KEY (task_id, dag_id, upstream_task_id, upstream_dag_id),
        FOREIGN KEY (task_id, dag_id) REFERENCES tasks(task_id, dag_id),
        FOREIGN KEY (upstream_task_id, upstream_dag_id) REFERENCES tasks(task_id, dag_id)
    );

    CREATE INDEX idx_task_deps_task ON task_dependencies(task_id, dag_id);
    ```

    ### Data Lineage

    ```sql
    CREATE TABLE dataset_lineage (
        lineage_id BIGSERIAL PRIMARY KEY,
        source_dataset VARCHAR(1000) NOT NULL,   -- e.g., "mysql://db/table"
        target_dataset VARCHAR(1000) NOT NULL,   -- e.g., "s3://bucket/path"
        dag_id VARCHAR(250),
        task_id VARCHAR(250),
        dag_run_id VARCHAR(250),
        transformation_code TEXT,
        created_at TIMESTAMP DEFAULT NOW(),

        FOREIGN KEY (task_id, dag_id, dag_run_id)
            REFERENCES task_instances(task_id, dag_id, dag_run_id)
    );

    CREATE INDEX idx_lineage_source ON dataset_lineage(source_dataset);
    CREATE INDEX idx_lineage_target ON dataset_lineage(target_dataset);
    CREATE INDEX idx_lineage_dag ON dataset_lineage(dag_id);
    ```

    ### SLA Misses

    ```sql
    CREATE TABLE sla_misses (
        task_id VARCHAR(250) NOT NULL,
        dag_id VARCHAR(250) NOT NULL,
        execution_date TIMESTAMP NOT NULL,
        expected_completion TIMESTAMP NOT NULL,
        actual_completion TIMESTAMP,
        duration FLOAT,
        notification_sent BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT NOW(),

        PRIMARY KEY (task_id, dag_id, execution_date)
    );

    CREATE INDEX idx_sla_misses_notification ON sla_misses(notification_sent)
        WHERE notification_sent = FALSE;
    ```

    ### XCom (Task Communication)

    ```sql
    CREATE TABLE xcom (
        xcom_id BIGSERIAL PRIMARY KEY,
        key VARCHAR(512) NOT NULL,
        value BYTEA,                      -- Pickled Python object
        task_id VARCHAR(250) NOT NULL,
        dag_id VARCHAR(250) NOT NULL,
        dag_run_id VARCHAR(250) NOT NULL,
        execution_date TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),

        UNIQUE(key, task_id, dag_id, dag_run_id)
    );

    CREATE INDEX idx_xcom_dag_run ON xcom(dag_run_id);
    ```

    ---

    ## Data Flow Diagrams

    ### DAG Scheduling Flow

    ```mermaid
    sequenceDiagram
        participant Timer as Scheduler Timer
        participant Scheduler as DAG Scheduler
        participant MetaDB as Metadata DB
        participant Queue as Task Queue
        participant Worker
        participant S3 as S3 Storage
        participant Warehouse as Snowflake

        Timer->>Scheduler: Tick (every 1 second)
        Scheduler->>MetaDB: Query DAGs due for execution<br/>WHERE schedule_interval matched
        MetaDB-->>Scheduler: List of DAGs to run

        loop For each DAG
            Scheduler->>Scheduler: Parse DAG file (Python)<br/>Build task graph
            Scheduler->>MetaDB: Create DAG run record<br/>state=RUNNING

            Scheduler->>Scheduler: Resolve dependencies<br/>Find tasks with no upstream

            loop For each ready task
                Scheduler->>MetaDB: Create task instance<br/>state=SCHEDULED
                Scheduler->>Queue: Enqueue task<br/>(task_id, dag_id, execution_date)
                MetaDB-->>Scheduler: Task queued
            end
        end

        Queue->>Worker: Dequeue task
        Worker->>MetaDB: Update task state=RUNNING<br/>Record start_date, hostname

        alt Extract Task
            Worker->>S3: Extract data from source<br/>Write to S3 raw layer
        else Transform Task
            Worker->>S3: Read raw data<br/>Transform (dbt/Spark)<br/>Write to S3 staging
        else Load Task
            Worker->>S3: Read staging data
            Worker->>Warehouse: COPY INTO warehouse
        end

        Worker->>MetaDB: Update task state=SUCCESS<br/>Record end_date, duration
        Worker->>MetaDB: Write XCom data<br/>(task output metadata)

        MetaDB->>Scheduler: Task completed event
        Scheduler->>Scheduler: Check downstream tasks<br/>Dependencies satisfied?

        alt All dependencies met
            Scheduler->>Queue: Enqueue downstream tasks
        else Dependencies not met
            Scheduler->>Scheduler: Wait for other tasks
        end

        Note over Scheduler: All tasks complete
        Scheduler->>MetaDB: Update DAG run state=SUCCESS
    ```

    ---

    ### Failure & Retry Flow

    ```mermaid
    sequenceDiagram
        participant Worker
        participant MetaDB as Metadata DB
        participant Queue as Task Queue
        participant Scheduler
        participant Alert as Alert System

        Worker->>Worker: Execute task<br/>ERROR: Connection timeout
        Worker->>MetaDB: Update task state=FAILED<br/>try_number=1, max_tries=3
        Worker->>MetaDB: Store error log

        MetaDB->>Scheduler: Task failed event

        alt try_number < max_tries
            Scheduler->>Scheduler: Calculate backoff delay<br/>delay = min(2^try_number * 60, 3600)
            Scheduler->>Scheduler: Sleep 2 minutes
            Scheduler->>Queue: Re-enqueue task<br/>try_number=2

            Queue->>Worker: Dequeue retry task
            Worker->>MetaDB: Update state=RUNNING<br/>try_number=2
            Worker->>Worker: Execute task<br/>SUCCESS!
            Worker->>MetaDB: Update state=SUCCESS

        else max_tries reached
            Scheduler->>MetaDB: Update task state=FAILED (final)
            Scheduler->>MetaDB: Mark downstream tasks<br/>state=UPSTREAM_FAILED
            Scheduler->>MetaDB: Update DAG run state=FAILED

            Scheduler->>Alert: Send failure notification
            Alert->>Alert: Email/Slack/PagerDuty
        end
    ```

    ---

    ### Incremental Loading Flow

    ```mermaid
    sequenceDiagram
        participant Task as Extract Task
        participant Source as Source DB
        participant S3 as S3 Data Lake
        participant MetaDB as Metadata DB
        participant Warehouse as Data Warehouse

        Task->>MetaDB: Get last successful run<br/>execution_date for this DAG
        MetaDB-->>Task: Last run: 2025-01-14

        Task->>Source: SELECT * FROM users<br/>WHERE updated_at > '2025-01-14'<br/>AND updated_at <= '2025-01-15'
        Source-->>Task: 15,000 rows (incremental)

        Note over Task: Instead of 10M rows (full)<br/>Only 15K rows (changed data)

        Task->>S3: Write Parquet file<br/>s3://lake/users/dt=2025-01-15/<br/>part-00001.parquet

        Task->>Task: Transform data<br/>(dbt incremental model)

        Task->>Warehouse: MERGE INTO users<br/>USING s3://lake/users/dt=2025-01-15/<br/>ON users.id = source.id<br/>WHEN MATCHED THEN UPDATE<br/>WHEN NOT MATCHED THEN INSERT

        Warehouse-->>Task: 15,000 rows upserted<br/>(10,000 updated, 5,000 inserted)

        Task->>MetaDB: Record lineage<br/>source: mysql://prod/users<br/>target: snowflake://analytics/users<br/>rows_processed: 15,000
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 DAG Scheduling & Execution

    ### Scheduler Algorithm

    ```python
    class DAGScheduler:
        """
        Core scheduler for orchestrating DAG execution

        Responsibilities:
        1. Parse DAG files (Python code)
        2. Determine which DAGs are due for execution
        3. Resolve task dependencies
        4. Enqueue tasks for execution
        5. Monitor task completion
        6. Handle failures and retries
        """

        def __init__(self, dag_folder, db, task_queue):
            self.dag_folder = dag_folder
            self.db = db
            self.task_queue = task_queue
            self.dags = {}  # In-memory DAG cache
            self.scheduler_loop_interval = 1  # seconds

        def run_scheduler_loop(self):
            """Main scheduler loop"""
            while True:
                try:
                    # 1. Refresh DAG definitions (re-parse Python files)
                    self.refresh_dags()

                    # 2. Create DAG runs for scheduled DAGs
                    self.create_scheduled_dag_runs()

                    # 3. Schedule tasks for active DAG runs
                    self.schedule_tasks_for_dag_runs()

                    # 4. Process task callbacks (on success/failure)
                    self.process_task_callbacks()

                    # 5. Check SLA misses
                    self.check_sla_misses()

                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")

                time.sleep(self.scheduler_loop_interval)

        def refresh_dags(self):
            """
            Parse DAG files and update in-memory cache

            This runs every loop to pick up new DAGs or changes
            Use file watcher (inotify) in production for efficiency
            """
            dag_files = glob.glob(f"{self.dag_folder}/**/*.py", recursive=True)

            for dag_file in dag_files:
                try:
                    # Execute Python file to create DAG object
                    # Uses restricted globals (no arbitrary code execution)
                    dag = self.parse_dag_file(dag_file)

                    if dag:
                        # Update cache
                        self.dags[dag.dag_id] = dag

                        # Persist to database
                        self.db.upsert_dag(
                            dag_id=dag.dag_id,
                            description=dag.description,
                            schedule_interval=dag.schedule_interval,
                            default_args=dag.default_args,
                            tags=dag.tags
                        )

                except Exception as e:
                    logger.error(f"Failed to parse DAG file {dag_file}: {e}")

        def create_scheduled_dag_runs(self):
            """
            Create DAG runs for DAGs that are due for execution

            Logic:
            1. Check each active DAG's schedule_interval
            2. Calculate next execution time
            3. If current_time >= next_execution_time, create DAG run
            """
            current_time = datetime.utcnow()

            for dag in self.dags.values():
                if dag.is_paused:
                    continue

                # Get last DAG run
                last_run = self.db.get_last_dag_run(dag.dag_id)

                # Calculate next execution time
                if dag.schedule_interval:
                    next_execution_time = self.calculate_next_execution(
                        dag.schedule_interval,
                        last_run.execution_date if last_run else dag.start_date
                    )

                    if current_time >= next_execution_time:
                        # Check if already exists (idempotency)
                        existing = self.db.get_dag_run(
                            dag.dag_id,
                            next_execution_time
                        )

                        if not existing:
                            # Create new DAG run
                            dag_run_id = f"scheduled__{next_execution_time.isoformat()}"

                            self.db.create_dag_run(
                                dag_run_id=dag_run_id,
                                dag_id=dag.dag_id,
                                execution_date=next_execution_time,
                                state='queued',
                                run_type='scheduled'
                            )

                            logger.info(f"Created DAG run: {dag.dag_id} @ {next_execution_time}")

        def calculate_next_execution(self, schedule_interval, last_execution):
            """
            Calculate next execution time based on schedule interval

            Formats:
            - Cron: "0 2 * * *" (daily at 2 AM)
            - Timedelta: "@daily", "@hourly", "0 */4 * * *" (every 4 hours)
            """
            if schedule_interval.startswith('@'):
                # Preset intervals
                if schedule_interval == '@hourly':
                    return last_execution + timedelta(hours=1)
                elif schedule_interval == '@daily':
                    return last_execution + timedelta(days=1)
                elif schedule_interval == '@weekly':
                    return last_execution + timedelta(weeks=1)
            else:
                # Cron expression
                from croniter import croniter
                cron = croniter(schedule_interval, last_execution)
                return cron.get_next(datetime)

        def schedule_tasks_for_dag_runs(self):
            """
            Schedule tasks for active DAG runs

            Algorithm:
            1. Get all DAG runs in 'queued' or 'running' state
            2. For each DAG run, build task dependency graph
            3. Find tasks ready to execute (all dependencies met)
            4. Enqueue tasks to task queue
            """
            active_runs = self.db.get_active_dag_runs()

            for dag_run in active_runs:
                dag = self.dags.get(dag_run.dag_id)
                if not dag:
                    continue

                # Update DAG run state to 'running'
                if dag_run.state == 'queued':
                    self.db.update_dag_run_state(dag_run.dag_run_id, 'running')

                # Get all task instances for this DAG run
                task_instances = self.db.get_task_instances(dag_run.dag_run_id)
                task_states = {ti.task_id: ti.state for ti in task_instances}

                # Find tasks ready to execute
                ready_tasks = self.get_ready_tasks(dag, task_states)

                for task in ready_tasks:
                    # Create task instance if not exists
                    if task.task_id not in task_states:
                        self.db.create_task_instance(
                            task_id=task.task_id,
                            dag_id=dag.dag_id,
                            dag_run_id=dag_run.dag_run_id,
                            execution_date=dag_run.execution_date,
                            state='scheduled',
                            try_number=1,
                            max_tries=task.retries,
                            pool=task.pool,
                            queue=task.queue,
                            priority_weight=task.priority_weight
                        )

                    # Enqueue task
                    self.task_queue.enqueue({
                        'task_id': task.task_id,
                        'dag_id': dag.dag_id,
                        'dag_run_id': dag_run.dag_run_id,
                        'execution_date': dag_run.execution_date,
                        'try_number': 1
                    })

                    # Update state to 'queued'
                    self.db.update_task_instance_state(
                        task.task_id,
                        dag.dag_id,
                        dag_run.dag_run_id,
                        'queued'
                    )

                # Check if DAG run is complete
                if self.is_dag_run_complete(dag_run.dag_run_id, task_states):
                    final_state = self.determine_dag_run_final_state(task_states)
                    self.db.update_dag_run_state(dag_run.dag_run_id, final_state)

        def get_ready_tasks(self, dag, task_states):
            """
            Find tasks that are ready to execute

            A task is ready if:
            1. It has no upstream dependencies, OR
            2. All upstream dependencies are in 'success' state
            3. Task is not already running/queued/success
            """
            ready_tasks = []

            for task in dag.tasks:
                # Skip if already processed
                if task.task_id in task_states:
                    state = task_states[task.task_id]
                    if state in ['queued', 'running', 'success', 'skipped']:
                        continue

                # Check dependencies
                upstream_tasks = task.upstream_list

                if not upstream_tasks:
                    # No dependencies, ready to run
                    ready_tasks.append(task)
                else:
                    # Check if all upstream tasks succeeded
                    all_upstream_success = all(
                        task_states.get(t.task_id) == 'success'
                        for t in upstream_tasks
                    )

                    if all_upstream_success:
                        ready_tasks.append(task)
                    else:
                        # Check if any upstream failed
                        any_upstream_failed = any(
                            task_states.get(t.task_id) == 'failed'
                            for t in upstream_tasks
                        )

                        if any_upstream_failed:
                            # Mark as upstream_failed
                            self.db.update_task_instance_state(
                                task.task_id,
                                dag.dag_id,
                                task_states.get('dag_run_id'),
                                'upstream_failed'
                            )

            return ready_tasks

        def is_dag_run_complete(self, dag_run_id, task_states):
            """Check if all tasks in DAG run are complete"""
            terminal_states = ['success', 'failed', 'skipped', 'upstream_failed']
            return all(state in terminal_states for state in task_states.values())

        def determine_dag_run_final_state(self, task_states):
            """Determine final state of DAG run based on task states"""
            if all(state == 'success' for state in task_states.values()):
                return 'success'
            elif any(state == 'failed' for state in task_states.values()):
                return 'failed'
            else:
                return 'success'  # Some tasks skipped but overall success
    ```

    ---

    ### Worker Execution Engine

    ```python
    class TaskExecutor:
        """
        Worker process that executes individual tasks

        Architecture:
        - Workers are stateless (can be scaled horizontally)
        - Pull tasks from queue (Celery/Redis)
        - Execute task operator (PythonOperator, SQLOperator, etc.)
        - Update task state in metadata DB
        """

        def __init__(self, task_queue, db, result_backend):
            self.task_queue = task_queue
            self.db = db
            self.result_backend = result_backend
            self.hostname = socket.gethostname()

        def run_worker_loop(self):
            """Main worker loop"""
            while True:
                try:
                    # 1. Dequeue task from queue
                    task_message = self.task_queue.dequeue(timeout=30)

                    if task_message:
                        # 2. Execute task
                        self.execute_task(task_message)

                except Exception as e:
                    logger.error(f"Worker error: {e}")

        def execute_task(self, task_message):
            """
            Execute a single task

            Steps:
            1. Load task definition from DAG
            2. Update state to 'running'
            3. Execute operator
            4. Handle success/failure
            5. Update state and store result
            """
            task_id = task_message['task_id']
            dag_id = task_message['dag_id']
            dag_run_id = task_message['dag_run_id']
            execution_date = task_message['execution_date']
            try_number = task_message['try_number']

            start_time = time.time()

            try:
                # 1. Update state to 'running'
                self.db.update_task_instance(
                    task_id=task_id,
                    dag_id=dag_id,
                    dag_run_id=dag_run_id,
                    try_number=try_number,
                    state='running',
                    start_date=datetime.utcnow(),
                    hostname=self.hostname,
                    pid=os.getpid()
                )

                # 2. Load DAG and task definition
                dag = self.load_dag(dag_id)
                task = dag.get_task(task_id)

                # 3. Build task context (available in templates)
                context = self.build_task_context(
                    task, dag, dag_run_id, execution_date, try_number
                )

                # 4. Execute operator
                logger.info(f"Executing task: {dag_id}.{task_id}")
                result = task.execute(context)

                # 5. Store result in XCom (if not None)
                if result is not None:
                    self.result_backend.set_xcom(
                        key='return_value',
                        value=result,
                        task_id=task_id,
                        dag_id=dag_id,
                        dag_run_id=dag_run_id
                    )

                # 6. Update state to 'success'
                duration = time.time() - start_time

                self.db.update_task_instance(
                    task_id=task_id,
                    dag_id=dag_id,
                    dag_run_id=dag_run_id,
                    try_number=try_number,
                    state='success',
                    end_date=datetime.utcnow(),
                    duration=duration
                )

                logger.info(f"Task succeeded: {dag_id}.{task_id} (duration: {duration:.2f}s)")

                # 7. Execute on_success_callback
                if task.on_success_callback:
                    task.on_success_callback(context)

            except Exception as e:
                # Task failed
                duration = time.time() - start_time

                logger.error(f"Task failed: {dag_id}.{task_id} - {str(e)}")

                # Update state to 'failed'
                self.db.update_task_instance(
                    task_id=task_id,
                    dag_id=dag_id,
                    dag_run_id=dag_run_id,
                    try_number=try_number,
                    state='failed',
                    end_date=datetime.utcnow(),
                    duration=duration
                )

                # Store error in logs
                self.store_task_log(
                    task_id, dag_id, dag_run_id, try_number,
                    f"ERROR: {str(e)}\n{traceback.format_exc()}"
                )

                # Check if should retry
                if try_number < task.max_retries:
                    # Calculate backoff delay
                    retry_delay = min(
                        task.retry_exponential_backoff ** try_number,
                        task.max_retry_delay
                    )

                    logger.info(f"Retrying task in {retry_delay}s (attempt {try_number + 1}/{task.max_retries})")

                    # Re-enqueue with delay
                    time.sleep(retry_delay)

                    self.task_queue.enqueue({
                        'task_id': task_id,
                        'dag_id': dag_id,
                        'dag_run_id': dag_run_id,
                        'execution_date': execution_date,
                        'try_number': try_number + 1
                    })
                else:
                    # Max retries reached
                    logger.error(f"Task failed permanently: {dag_id}.{task_id}")

                    # Execute on_failure_callback
                    if task.on_failure_callback:
                        task.on_failure_callback(context)

        def build_task_context(self, task, dag, dag_run_id, execution_date, try_number):
            """
            Build context dictionary for task execution

            Context is used for:
            1. Jinja templating ({{ ds }}, {{ dag_run.execution_date }})
            2. Accessing XCom (task_instance.xcom_pull)
            3. Task metadata
            """
            return {
                'task': task,
                'dag': dag,
                'dag_run': self.db.get_dag_run(dag_run_id),
                'execution_date': execution_date,
                'ds': execution_date.strftime('%Y-%m-%d'),  # Date string
                'ds_nodash': execution_date.strftime('%Y%m%d'),
                'prev_execution_date': self.get_prev_execution_date(dag, execution_date),
                'next_execution_date': self.get_next_execution_date(dag, execution_date),
                'try_number': try_number,
                'task_instance': self.db.get_task_instance(task.task_id, dag.dag_id, dag_run_id),
                'ti': self.db.get_task_instance(task.task_id, dag.dag_id, dag_run_id),
                'var': {
                    'value': self.db.get_variable,  # Access Airflow variables
                    'json': lambda key: json.loads(self.db.get_variable(key))
                },
                'task_instance_key_str': f"{dag.dag_id}__{task.task_id}__{execution_date.isoformat()}",
                'run_id': dag_run_id,
                'params': dag.params  # User-defined parameters
            }
    ```

    ---

    ## 3.2 Incremental Loading Strategies

    ### Time-Based Incremental (Most Common)

    ```python
    class IncrementalExtractor:
        """
        Extract only new/changed data based on timestamp

        Approach:
        1. Track last successful extraction timestamp
        2. Query only records updated since last extraction
        3. Use execution_date for partitioning
        """

        def extract_incremental_mysql(self, **context):
            """
            Extract incremental data from MySQL

            Example: Extract users modified in the last day
            """
            execution_date = context['execution_date']
            prev_execution_date = context['prev_execution_date']

            # Build incremental query
            query = f"""
                SELECT *
                FROM users
                WHERE updated_at >= '{prev_execution_date}'
                  AND updated_at < '{execution_date}'
            """

            # Execute query
            conn = mysql.connector.connect(**db_config)
            df = pd.read_sql(query, conn)

            logger.info(f"Extracted {len(df)} rows (incremental)")

            # Write to S3 partitioned by date
            output_path = f"s3://data-lake/raw/users/dt={execution_date.strftime('%Y-%m-%d')}/"
            df.to_parquet(output_path, index=False)

            # Store metadata in XCom
            return {
                'rows_extracted': len(df),
                'output_path': output_path,
                'execution_date': execution_date.isoformat()
            }
    ```

    ### Change Data Capture (CDC)

    ```python
    class CDCExtractor:
        """
        Use database transaction logs for incremental extraction

        Tools: Debezium, AWS DMS, Fivetran

        Advantages:
        - Near real-time updates
        - Capture deletes (not possible with timestamp-based)
        - No performance impact on source database

        Disadvantages:
        - Complex setup
        - Requires database permissions
        - Not supported by all databases
        """

        def process_cdc_events(self, **context):
            """
            Process CDC events from Kafka topic

            Event format:
            {
                "op": "c",  # c=create, u=update, d=delete
                "before": {"id": 123, "name": "Alice", ...},
                "after": {"id": 123, "name": "Alice Smith", ...},
                "ts_ms": 1735819200000
            }
            """
            execution_date = context['execution_date']

            # Read CDC events from Kafka
            consumer = KafkaConsumer(
                'mysql.users.cdc',
                bootstrap_servers='kafka:9092',
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )

            events = []
            for message in consumer:
                event = message.value

                # Filter events for this execution window
                event_time = datetime.fromtimestamp(event['ts_ms'] / 1000)
                if event_time >= context['prev_execution_date'] and event_time < execution_date:
                    events.append(event)

            # Convert to DataFrame
            df = self.cdc_events_to_dataframe(events)

            # Write to S3
            output_path = f"s3://data-lake/cdc/users/dt={execution_date.strftime('%Y-%m-%d')}/"
            df.to_parquet(output_path, index=False)

            return {
                'events_processed': len(events),
                'inserts': len([e for e in events if e['op'] == 'c']),
                'updates': len([e for e in events if e['op'] == 'u']),
                'deletes': len([e for e in events if e['op'] == 'd'])
            }

        def cdc_events_to_dataframe(self, events):
            """Convert CDC events to DataFrame with SCD Type 2"""
            rows = []

            for event in events:
                if event['op'] in ['c', 'u']:
                    # Insert or update
                    row = event['after']
                    row['_cdc_op'] = event['op']
                    row['_cdc_timestamp'] = event['ts_ms']
                    rows.append(row)
                elif event['op'] == 'd':
                    # Delete (mark as deleted)
                    row = event['before']
                    row['_cdc_op'] = 'd'
                    row['_cdc_timestamp'] = event['ts_ms']
                    row['_deleted'] = True
                    rows.append(row)

            return pd.DataFrame(rows)
    ```

    ### dbt Incremental Models

    ```sql
    -- dbt incremental model: models/user_daily_metrics.sql

    {{
        config(
            materialized='incremental',
            unique_key='user_id',
            incremental_strategy='merge',
            partition_by={
                "field": "date",
                "data_type": "date"
            }
        )
    }}

    WITH user_events AS (
        SELECT
            user_id,
            DATE(event_timestamp) AS date,
            COUNT(*) AS event_count,
            SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
            SUM(revenue) AS total_revenue
        FROM {{ source('raw', 'events') }}

        {% if is_incremental() %}
            -- Only process new data (incremental run)
            WHERE DATE(event_timestamp) = '{{ ds }}'
        {% else %}
            -- Full refresh (initial load)
            WHERE DATE(event_timestamp) >= '2024-01-01'
        {% endif %}

        GROUP BY user_id, DATE(event_timestamp)
    )

    SELECT
        user_id,
        date,
        event_count,
        purchase_count,
        total_revenue,
        CURRENT_TIMESTAMP() AS updated_at
    FROM user_events
    ```

    ```python
    # Airflow task to run dbt incremental model

    transform_dbt = BashOperator(
        task_id='transform_user_metrics_dbt',
        bash_command='''
            cd /opt/dbt &&
            dbt run --models user_daily_metrics --vars '{"ds": "{{ ds }}"}'
        ''',
        dag=dag
    )
    ```

    ---

    ## 3.3 Data Lineage Tracking

    ```python
    class LineageTracker:
        """
        Track data lineage (data flow from source to destination)

        Benefits:
        1. Impact analysis (which downstream depends on this table?)
        2. Root cause analysis (where did this data come from?)
        3. Compliance (GDPR, data auditing)
        4. Data discovery
        """

        def capture_lineage(self, task_id, dag_id, dag_run_id,
                           source_datasets, target_datasets,
                           transformation_code):
            """
            Capture lineage for a task execution

            Example:
            source_datasets = [
                "mysql://prod-db:3306/users.users",
                "s3://data-lake/raw/events/"
            ]
            target_datasets = [
                "s3://data-lake/transformed/user_metrics/",
                "snowflake://analytics/user_daily_metrics"
            ]
            """
            lineage_records = []

            for source in source_datasets:
                for target in target_datasets:
                    lineage_records.append({
                        'source_dataset': source,
                        'target_dataset': target,
                        'dag_id': dag_id,
                        'task_id': task_id,
                        'dag_run_id': dag_run_id,
                        'transformation_code': transformation_code,
                        'created_at': datetime.utcnow()
                    })

            # Bulk insert to lineage table
            self.db.bulk_insert('dataset_lineage', lineage_records)

        def get_upstream_datasets(self, dataset):
            """
            Get all upstream datasets that feed into this dataset

            Example: snowflake://analytics/user_daily_metrics
            -> s3://data-lake/transformed/user_metrics/
            -> s3://data-lake/raw/users/, s3://data-lake/raw/events/
            -> mysql://prod-db/users, https://api.company.com/events
            """
            query = """
                WITH RECURSIVE lineage_tree AS (
                    -- Base case: direct upstream
                    SELECT source_dataset, target_dataset, dag_id, task_id, 1 AS depth
                    FROM dataset_lineage
                    WHERE target_dataset = %s

                    UNION ALL

                    -- Recursive case: upstream of upstream
                    SELECT l.source_dataset, l.target_dataset, l.dag_id, l.task_id, lt.depth + 1
                    FROM dataset_lineage l
                    INNER JOIN lineage_tree lt ON l.target_dataset = lt.source_dataset
                    WHERE lt.depth < 10  -- Prevent infinite loops
                )
                SELECT DISTINCT source_dataset, target_dataset, dag_id, task_id, depth
                FROM lineage_tree
                ORDER BY depth, source_dataset;
            """

            return self.db.execute(query, [dataset])

        def get_downstream_datasets(self, dataset):
            """Get all downstream datasets that depend on this dataset"""
            query = """
                WITH RECURSIVE lineage_tree AS (
                    SELECT source_dataset, target_dataset, dag_id, task_id, 1 AS depth
                    FROM dataset_lineage
                    WHERE source_dataset = %s

                    UNION ALL

                    SELECT l.source_dataset, l.target_dataset, l.dag_id, l.task_id, lt.depth + 1
                    FROM dataset_lineage l
                    INNER JOIN lineage_tree lt ON l.source_dataset = lt.target_dataset
                    WHERE lt.depth < 10
                )
                SELECT DISTINCT source_dataset, target_dataset, dag_id, task_id, depth
                FROM lineage_tree
                ORDER BY depth, target_dataset;
            """

            return self.db.execute(query, [dataset])

        def visualize_lineage(self, dataset):
            """
            Generate visualization of data lineage

            Output: Mermaid diagram showing data flow
            """
            upstream = self.get_upstream_datasets(dataset)
            downstream = self.get_downstream_datasets(dataset)

            # Build Mermaid graph
            lines = ["graph LR"]

            for record in upstream:
                source = self.sanitize_node_name(record['source_dataset'])
                target = self.sanitize_node_name(record['target_dataset'])
                lines.append(f"    {source} -->|{record['dag_id']}.{record['task_id']}| {target}")

            for record in downstream:
                source = self.sanitize_node_name(record['source_dataset'])
                target = self.sanitize_node_name(record['target_dataset'])
                lines.append(f"    {source} -->|{record['dag_id']}.{record['task_id']}| {target}")

            return "\n".join(lines)
    ```

    **Example Lineage Visualization:**

    ```mermaid
    graph LR
        MySQL[(mysql://prod-db<br/>users)] -->|extract_users| RawUsers[s3://lake/raw/users/]
        API[api.company.com<br/>events] -->|extract_events| RawEvents[s3://lake/raw/events/]

        RawUsers --> Transform[dbt: user_daily_metrics]
        RawEvents --> Transform

        Transform --> Staging[s3://lake/staging/<br/>user_metrics/]
        Staging -->|load_to_warehouse| Snowflake[(snowflake://analytics<br/>user_daily_metrics)]

        Snowflake -->|monthly_aggregation| MonthlyMetrics[(user_monthly_summary)]
        Snowflake -->|dashboard_query| Dashboard[Tableau Dashboard]

        style MySQL fill:#e1f5ff
        style Snowflake fill:#e8f5e9
        style Transform fill:#fff9c4
    ```

    ---

    ## 3.4 SLA Monitoring

    ```python
    class SLAMonitor:
        """
        Monitor and alert on SLA violations

        SLA (Service Level Agreement):
        - Expected completion time for DAG/task
        - Example: "user_analytics_pipeline must complete by 6 AM"
        """

        def check_sla_misses(self):
            """
            Check for SLA violations

            Runs every minute in scheduler loop
            """
            current_time = datetime.utcnow()

            # Query tasks with SLA defined
            query = """
                SELECT
                    ti.task_id,
                    ti.dag_id,
                    ti.dag_run_id,
                    ti.execution_date,
                    ti.state,
                    ti.start_date,
                    ti.end_date,
                    t.sla,
                    dr.start_date + t.sla AS expected_completion
                FROM task_instances ti
                JOIN tasks t ON ti.task_id = t.task_id AND ti.dag_id = t.dag_id
                JOIN dag_runs dr ON ti.dag_run_id = dr.dag_run_id
                WHERE t.sla IS NOT NULL
                  AND ti.state IN ('running', 'queued')
                  AND dr.start_date + t.sla < %s
            """

            sla_misses = self.db.execute(query, [current_time])

            for miss in sla_misses:
                # Check if already recorded
                existing = self.db.get_sla_miss(
                    miss['task_id'],
                    miss['dag_id'],
                    miss['execution_date']
                )

                if not existing:
                    # Record SLA miss
                    self.db.insert_sla_miss(
                        task_id=miss['task_id'],
                        dag_id=miss['dag_id'],
                        execution_date=miss['execution_date'],
                        expected_completion=miss['expected_completion'],
                        actual_completion=miss.get('end_date')
                    )

                    # Send alert
                    self.send_sla_alert(miss)

        def send_sla_alert(self, sla_miss):
            """Send alert for SLA violation"""
            message = f"""
            🚨 SLA MISS ALERT

            DAG: {sla_miss['dag_id']}
            Task: {sla_miss['task_id']}
            Execution Date: {sla_miss['execution_date']}
            Expected Completion: {sla_miss['expected_completion']}
            Current State: {sla_miss['state']}

            This task has exceeded its SLA deadline!
            """

            # Send to alerting system
            self.alerting_system.send_alert(
                channel='slack',
                severity='warning',
                message=message,
                dag_id=sla_miss['dag_id'],
                task_id=sla_miss['task_id']
            )
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Scheduler Scaling:
    - Multiple scheduler instances (HA)
    - Each scheduler handles subset of DAGs
    - Coordination via database locks
    - Scale to 10K+ DAGs

    Worker Scaling:
    - Stateless workers (easy to scale)
    - Worker pools by resource type:
      * CPU-intensive: Spark, data processing
      * I/O-intensive: Database extracts
      * Memory-intensive: Large transformations
    - Auto-scaling based on queue depth
    - Scale to 10,000+ parallel tasks

    Database Scaling:
    - Read replicas for queries (5-10 replicas)
    - Connection pooling (PgBouncer)
    - Partitioning task_instances by execution_date
    - Archive old task instances (> 1 year)

    Task Queue Scaling:
    - Sharded Redis (multiple queues)
    - Celery supports multiple brokers
    - Priority queues (critical tasks first)
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Incremental loading** | 90% reduction in data transfer | Complex logic, requires timestamps |
    | **Task parallelism** | 10x faster DAG completion | Resource contention, costs |
    | **DAG parsing cache** | 100x faster scheduler loop | Stale DAGs (need refresh) |
    | **Connection pooling** | 5x more concurrent tasks | Connection limit on databases |
    | **XCom cleanup** | 80% reduction in DB size | Lose historical task outputs |
    | **Log archival (S3)** | 95% reduction in DB size | Slower log access |

    ---

    ## Fault Tolerance

    ```
    Task-Level Failures:
    - Automatic retries (exponential backoff)
    - Configurable max retries (default: 3)
    - Idempotent tasks (safe to retry)
    - Upstream failure propagation

    DAG-Level Failures:
    - Partial recovery (resume from failed task)
    - Manual rerun support
    - Backfill for historical data

    Infrastructure Failures:
    - Scheduler HA (multiple instances)
    - Task queue persistence (Redis AOF/RDB)
    - Database replication (primary + replicas)
    - Worker failure detection (heartbeat timeout)
    - Task timeout (kill runaway tasks)

    Data Failures:
    - Data quality checks (dbt tests, Great Expectations)
    - Rollback mechanisms (delete partition, truncate staging)
    - Data versioning (partition by execution_date)
    ```

    ---

    ## Cost Optimization

    ```
    Monthly Cost (10,000 DAGs, 10M tasks/day):

    Compute:
    - 20 scheduler nodes × $50 = $1,000
    - 1,000 worker nodes × $100 = $100,000
    - 6 database nodes × $500 = $3,000
    - 3 message queue nodes × $100 = $300
    - Total compute: $104,300/month

    Storage:
    - Metadata (100 TB): 100 × $23 (S3) = $2,300
    - Data lake (300 PB): 300,000 × $23 = $6.9M
    - Data warehouse (365 PB): Separate billing (Snowflake on-demand)
    - Total storage: ~$7M/month

    Network:
    - 1 Tbps × $0.08/GB × 330 TB/month = $26,400

    Total: ~$7.1M/month

    Optimizations:
    1. Spot instances for workers (70% savings): -$70K
    2. Incremental loading (reduce transfers 90%): -$6M data warehouse costs
    3. Compress Parquet files (reduce storage 50%): -$3.5M
    4. Task result caching (reduce reruns 20%): -$20K compute
    5. Archive old task logs to Glacier: -$2K/month

    Optimized Total: ~$700K/month (10x reduction)
    ```

    ---

    ## Monitoring Metrics

    ```python
    # Key metrics for ETL pipeline health

    # DAG metrics
    dag_run_duration_seconds{dag_id, state}
    dag_run_count{dag_id, state}  # state: success, failed, running
    dag_scheduling_latency_seconds  # Time from schedule to execution

    # Task metrics
    task_instance_duration_seconds{dag_id, task_id, state}
    task_instance_count{dag_id, task_id, state}
    task_retry_count{dag_id, task_id}
    task_queue_depth{queue_name}

    # Scheduler metrics
    scheduler_loop_duration_seconds
    dags_parsed_per_loop
    tasks_scheduled_per_loop

    # Worker metrics
    worker_active_tasks{hostname, pool}
    worker_cpu_usage_percent{hostname}
    worker_memory_usage_bytes{hostname}

    # Data metrics
    rows_extracted{dag_id, task_id, source}
    rows_loaded{dag_id, task_id, target}
    data_bytes_processed{dag_id, task_id}

    # SLA metrics
    sla_misses_count{dag_id, task_id}
    sla_miss_duration_seconds{dag_id, task_id}
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you handle task failures and retries?"**
   - Automatic retries with exponential backoff (2^n * 60 seconds)
   - Configurable max retries (default: 3)
   - Idempotent tasks (safe to retry without side effects)
   - Upstream failure propagation (skip downstream tasks)
   - Manual rerun support for debugging

2. **"How do you implement incremental loading?"**
   - Time-based: Track last successful run, query WHERE updated_at > last_run
   - CDC (Change Data Capture): Use transaction logs (Debezium, AWS DMS)
   - Partition-based: Process one partition at a time (dt=2025-01-15)
   - Checkpoint files: Store watermark in S3/database
   - dbt incremental models: merge/upsert strategy

3. **"How do you scale to 10,000+ DAGs?"**
   - Multiple scheduler instances (HA, load distribution)
   - DAG parsing cache (avoid re-parsing every loop)
   - Horizontal scaling of workers (1,000+ nodes)
   - Database read replicas (distribute query load)
   - Task queue sharding (multiple Redis instances)

4. **"How do you handle task dependencies?"**
   - DAG is directed acyclic graph (no cycles)
   - Scheduler resolves dependencies (topological sort)
   - Tasks execute when all upstream tasks succeed
   - Use `>>` operator: `task1 >> task2 >> task3`
   - Cross-DAG dependencies: ExternalTaskSensor

5. **"How do you track data lineage?"**
   - Capture source/target datasets for each task
   - Store in lineage table (recursive queries for upstream/downstream)
   - Use OpenLineage standard (Apache Marquez, DataHub)
   - Visualize with graph databases (Neo4j)
   - Impact analysis: "What breaks if I change this table?"

6. **"How do you ensure idempotency?"**
   - DELETE+INSERT (truncate partition, reload)
   - UPSERT/MERGE (update if exists, insert if not)
   - Partition by execution date (dt=2025-01-15)
   - Use unique task run IDs
   - Avoid operations with side effects (sending emails, charging credit cards)

7. **"How do you monitor SLAs?"**
   - Define SLA as timedelta from DAG start (e.g., 2 hours)
   - Scheduler checks for tasks exceeding deadline
   - Record SLA misses in database
   - Send alerts (Slack, PagerDuty)
   - Dashboard showing on-time completion rate

8. **"How do you handle backfills?"**
   - Airflow: `airflow dags backfill --start-date --end-date`
   - Creates multiple DAG runs for date range
   - Respects task dependencies
   - Can run in parallel (if max_active_runs > 1)
   - Use catchup=False to prevent automatic backfills

**Key Points to Mention:**

- DAG-based orchestration (directed acyclic graph)
- Separation of scheduler and workers (scalability)
- Idempotent tasks (safe retries)
- Incremental loading (90% reduction in data transfer)
- Data lineage tracking (impact analysis)
- SLA monitoring (alert on deadline misses)
- Distributed execution (horizontal scaling)
- Task queue for decoupling (Redis/RabbitMQ)
- Metadata database for state (PostgreSQL)

---

## Real-World Examples

**Apache Airflow (Airbnb):**
- 1,000+ DAGs, 10K+ tasks/day
- Python-based DAG definition
- Celery executor (distributed workers)
- PostgreSQL metadata store
- Used for ML pipelines, data warehouse ETL

**dbt (Fishtown Analytics):**
- SQL-based transformations
- Incremental models (merge strategy)
- Directed acyclic graph of models
- Data testing and documentation
- Integrates with Airflow/Prefect

**Netflix Data Platform:**
- 50,000+ data pipelines
- Batch + streaming (Kafka, Flink)
- Custom scheduler (Genie, Maestro)
- S3 data lake + Redshift warehouse
- Iceberg table format (ACID transactions)

**Uber Databook:**
- 100,000+ data pipelines
- Metadata-driven ETL
- Apache Spark for transformations
- Hive metastore for lineage
- Real-time + batch processing

---

## Summary

**System Characteristics:**

- **Scale:** 10,000+ DAGs, 10M tasks/day, 100K+ data sources
- **Throughput:** 1,000 tasks/sec (peak), 10 PB/day data processed
- **Latency:** < 1 second task scheduling, 5 min average task duration
- **Availability:** 99.9% uptime (HA scheduler, worker redundancy)

**Core Components:**

1. **DAG Scheduler:** Parse DAGs, resolve dependencies, enqueue tasks
2. **Task Queue:** Redis/RabbitMQ for decoupling scheduler and workers
3. **Worker Pool:** Distributed task execution (1,000+ nodes)
4. **Metadata DB:** PostgreSQL for DAG/task state, lineage, SLAs
5. **Data Lake:** S3/GCS for raw and staging data
6. **Data Warehouse:** Snowflake/BigQuery/Redshift for analytics

**Key Design Decisions:**

- DAG-based orchestration (explicit dependencies)
- Distributed execution (horizontal scaling)
- Idempotent tasks (safe retries)
- Incremental loading (90% reduction in data transfer)
- Partition-based storage (dt=YYYY-MM-DD)
- Data lineage tracking (impact analysis)
- SLA monitoring (alert on deadline misses)
- Task queue for decoupling (eventual consistency)

This design provides a scalable, fault-tolerant ETL pipeline orchestration system capable of handling millions of tasks per day across thousands of data sources with comprehensive monitoring, lineage tracking, and SLA enforcement.
