# Design Data Warehouse (Snowflake/BigQuery/Redshift)

A distributed, cloud-native data warehouse system that provides petabyte-scale storage, MPP (Massively Parallel Processing) query execution, and ACID transactions for analytical workloads with columnar storage, query optimization, and separation of compute and storage.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10 PB data, 10K concurrent queries, 100K queries/day, 1M tables, sub-second to minute-scale queries |
| **Key Challenges** | Query optimization at scale, MPP coordination, columnar compression, concurrency control, workload management |
| **Core Concepts** | MPP architecture, columnar storage (Parquet/ORC), cost-based optimizer, materialized views, micro-partitioning, zero-copy cloning |
| **Companies** | Snowflake, Google BigQuery, AWS Redshift, Azure Synapse, Databricks SQL, Teradata |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **SQL Queries** | Standard SQL (SELECT, JOIN, GROUP BY, window functions) | P0 (Must have) |
    | **DDL Operations** | CREATE/ALTER/DROP tables, schemas, views | P0 (Must have) |
    | **ACID Transactions** | Read committed isolation, atomic writes | P0 (Must have) |
    | **Materialized Views** | Pre-computed aggregations for fast queries | P0 (Must have) |
    | **Query Optimization** | Cost-based optimizer, predicate pushdown, partition pruning | P0 (Must have) |
    | **Workload Management** | Query priority, resource allocation, concurrency limits | P0 (Must have) |
    | **Semi-structured Data** | JSON, arrays, nested types (VARIANT, ARRAY, STRUCT) | P1 (Should have) |
    | **Time Travel** | Query historical data, restore to previous state | P1 (Should have) |
    | **Zero-copy Clone** | Instant table/schema clones without copying data | P1 (Should have) |
    | **Data Sharing** | Secure cross-account data sharing | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - ETL/ELT pipeline orchestration (use Airflow, dbt)
    - Real-time streaming (focus on batch analytics)
    - Machine learning model training (use separate ML platforms)
    - Data visualization (use BI tools: Tableau, Looker)
    - Object storage management (assume S3/GCS/ADLS)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Business-critical analytics, dashboard dependencies |
    | **Query Latency** | < 1s for simple, < 60s for complex (p95) | Interactive dashboards, ad-hoc exploration |
    | **Concurrency** | 10,000 concurrent queries | Many users, automated jobs, BI tools |
    | **Throughput** | 100,000 queries/day | High query workload across organization |
    | **Scalability** | 10 PB data, linear scaling | Store years of data, grow with business |
    | **Consistency** | Serializable for writes, snapshot isolation for reads | ACID guarantees, no dirty reads |
    | **Cost** | < $25/TB/month storage, $5/TB scanned | Competitive with competitors, cost-effective at scale |
    | **Compression** | 5-10x compression ratio | Reduce storage and scan costs |

    ---

    ## Capacity Estimation

    ### Storage Estimates

    ```
    Data size:
    - Total data: 10 PB (10,000 TB)
    - Daily ingestion: 50 TB/day
    - Growth rate: 2% per month (200 TB/month)
    - Data retention: 5 years (unlimited for aggregated)

    Table counts:
    - Total tables: 1,000,000 tables
    - Active tables (queried daily): 100,000 (10%)
    - Average table size: 10 GB
    - Large tables (>1 TB): 1,000 tables

    Compression:
    - Raw data: 50 PB (before compression)
    - Columnar compression (Parquet + Snappy): 10:1 ratio
    - Compressed storage: 5 PB
    - Additional 2x compression from encoding (RLE, dict): 2.5 PB
    - Actual storage: 2.5 PB (20% of raw)

    Micro-partitions (Snowflake approach):
    - Partition size: 16 MB compressed, 256 MB uncompressed
    - Total partitions: 2.5 PB / 16 MB = 160 million partitions
    - Metadata per partition: 1 KB
    - Total metadata: 160 GB

    Materialized views:
    - Views count: 50,000 views
    - Average view size: 100 GB
    - Total MV storage: 5 TB (0.2% of base data)

    Storage cost (S3):
    - Primary data (S3 Standard): 2.5 PB × $0.023/GB = $58,880/month
    - With Reserved Capacity: $40,000/month (~$20/TB)
    - With compression + tiering: $25,000/month (~$10/TB)
    ```

    ### Query Estimates

    ```
    Query workload:
    - Total queries: 100,000 queries/day
    - Peak queries: 3x average = 3.5 queries/sec
    - Average query: 3.5 queries/sec
    - Concurrent queries: 1,000 concurrent (at peak)
    - Max concurrent: 10,000 (burst capacity)

    Query types:
    - Simple SELECT (< 1s): 60% (60K queries)
    - Aggregations (1-10s): 30% (30K queries)
    - Complex joins (10-60s): 9% (9K queries)
    - Heavy scans (> 60s): 1% (1K queries)

    Data scanned per query:
    - Simple: 100 MB (indexed lookups)
    - Aggregations: 10 GB (single table scan)
    - Complex joins: 100 GB (multi-table joins)
    - Heavy scans: 1 TB (full table scans)
    - Average: 50 GB/query

    Total data scanned:
    - Daily scans: 100K queries × 50 GB = 5 PB/day
    - With partition pruning (90% reduction): 500 TB/day
    - With query cache (80% hit rate): 100 TB/day
    - Monthly scans: 3 PB/month

    Scan cost:
    - 3 PB/month × $5/TB = $15,000/month
    - Total cost: $25K (storage) + $15K (compute) = $40K/month
    ```

    ### Compute Estimates

    ```
    Cluster configuration:
    - Warehouse size: Medium (4 nodes × 8 cores = 32 cores)
    - Large warehouse: 128 cores
    - X-Large warehouse: 512 cores

    Compute allocation:
    - ETL jobs: 40% (dedicated large warehouse)
    - Interactive queries: 40% (auto-scaling warehouses)
    - Reporting: 20% (scheduled, small warehouse)

    Resource utilization:
    - Average CPU: 50% utilization
    - Peak CPU: 85% utilization
    - Memory per node: 64 GB
    - Total cluster memory: 256 GB (medium), 8 TB (X-Large)

    Compute cost:
    - Credits per hour: 4 credits/hour (medium)
    - Running time: 12 hours/day (auto-suspend when idle)
    - Daily credits: 4 × 12 = 48 credits/day
    - Monthly credits: 1,440 credits
    - Cost: 1,440 credits × $3/credit = $4,320/month
    ```

    ### Network Estimates

    ```
    Data ingress (loading):
    - Daily ingestion: 50 TB/day
    - Peak ingestion: 2 TB/hour
    - Throughput: 560 MB/sec
    - Ingress cost: Free (S3 to warehouse)

    Data egress (query results):
    - Average result size: 10 MB
    - Daily results: 100K × 10 MB = 1 TB/day
    - Egress cost: 1 TB × $0.09/GB = $90/day = $2,700/month

    Metadata queries:
    - Catalog queries: 10,000/day
    - Average size: 10 KB
    - Total: 100 MB/day (negligible)

    Total bandwidth: 50 TB/day ingress, 1 TB/day egress
    ```

    ---

    ## Key Assumptions

    1. 10:1 compression ratio from columnar storage (Parquet + Snappy)
    2. 90% partition pruning effectiveness (clustered data)
    3. 80% query result cache hit rate
    4. 10% of queries are complex (> 10s)
    5. Auto-suspend after 10 minutes of inactivity
    6. 5-year data retention (unlimited for aggregated tables)
    7. Star schema predominant (fact + dimension tables)
    8. Semi-structured data (JSON): 20% of total data
    9. Materialized views: 5% of tables
    10. Peak concurrency: 1,000 queries (10x average)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separation of compute and storage:** Independent scaling, pay only for what you use
    2. **MPP (Massively Parallel Processing):** Distribute queries across many nodes
    3. **Columnar storage:** Store columns separately for efficient analytics
    4. **Cost-based optimization:** Query planner chooses optimal execution strategy
    5. **Metadata-driven:** Use statistics for partition pruning and predicate pushdown
    6. **Multi-cluster:** Isolate workloads (ETL, reporting, ad-hoc)
    7. **Elastic compute:** Auto-scale based on query queue depth

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            BI[BI Tools<br/>Tableau, Looker]
            SQL_Client[SQL Clients<br/>DBeaver, DataGrip]
            JDBC[JDBC/ODBC<br/>Drivers]
            REST[REST API<br/>Programmatic access]
        end

        subgraph "API Gateway"
            Auth[Authentication<br/>OAuth, JWT]
            LB[Load Balancer<br/>Query routing]
            Session[Session Manager<br/>Connection pooling]
        end

        subgraph "Query Processing Layer"
            Parser[SQL Parser<br/>Syntax validation]
            Analyzer[Semantic Analyzer<br/>Schema resolution]
            Optimizer[Query Optimizer<br/>Cost-based planning]
            Planner[Query Planner<br/>Execution plan]
        end

        subgraph "Execution Layer (MPP)"
            Coordinator[Query Coordinator<br/>Distribute tasks]
            Compute1[Compute Node 1<br/>Execute fragments]
            Compute2[Compute Node 2<br/>Execute fragments]
            ComputeN[Compute Node N<br/>Execute fragments]
            Result_Cache[Result Cache<br/>Query result caching]
        end

        subgraph "Storage Layer (Columnar)"
            Metadata_Service[Metadata Service<br/>Table catalog]
            Partition_Index[Partition Index<br/>Min/max stats]
            S3_Data[Object Storage (S3)<br/>Parquet files]
            S3_Metadata[S3 Metadata<br/>Table definitions]
        end

        subgraph "Metadata Store"
            Catalog_DB[(Metadata DB<br/>PostgreSQL)]
            Stats_DB[(Statistics Store<br/>Column stats)]
            Version_DB[(Version Store<br/>Time travel)]
        end

        subgraph "Background Services"
            Compaction[Compaction Service<br/>Merge small files]
            Stats_Collector[Statistics Collector<br/>Analyze tables]
            Cluster_Manager[Clustering Service<br/>Sort/partition data]
            MV_Refresh[MV Refresh Service<br/>Incremental updates]
        end

        subgraph "Workload Management"
            Queue_Manager[Query Queue<br/>Priority scheduling]
            Resource_Manager[Resource Manager<br/>CPU/Memory limits]
            Auto_Scaler[Auto-scaler<br/>Add/remove nodes]
        end

        subgraph "Monitoring"
            Query_Monitor[Query Monitor<br/>Performance metrics]
            Cost_Monitor[Cost Monitor<br/>Credit tracking]
            Alerts[Alerting<br/>Slow queries, failures]
        end

        BI --> JDBC
        SQL_Client --> JDBC
        REST --> Auth
        JDBC --> Auth

        Auth --> LB
        LB --> Session

        Session --> Parser
        Parser --> Analyzer
        Analyzer --> Optimizer
        Optimizer --> Planner

        Planner --> Queue_Manager
        Queue_Manager --> Resource_Manager
        Resource_Manager --> Coordinator

        Coordinator --> Compute1
        Coordinator --> Compute2
        Coordinator --> ComputeN

        Coordinator --> Result_Cache

        Compute1 --> Metadata_Service
        Compute2 --> Metadata_Service
        ComputeN --> Metadata_Service

        Metadata_Service --> Catalog_DB
        Metadata_Service --> Stats_DB
        Metadata_Service --> Partition_Index

        Compute1 --> S3_Data
        Compute2 --> S3_Data
        ComputeN --> S3_Data

        Metadata_Service --> S3_Metadata

        Compaction --> S3_Data
        Stats_Collector --> Stats_DB
        Cluster_Manager --> S3_Data
        MV_Refresh --> S3_Data

        Auto_Scaler --> Coordinator
        Coordinator --> Query_Monitor
        Query_Monitor --> Alerts

        style Auth fill:#e1f5ff
        style LB fill:#e1f5ff
        style Optimizer fill:#ffe1e1
        style Coordinator fill:#ffe1e1
        style Compute1 fill:#fff4e1
        style S3_Data fill:#f3e5f5
        style Metadata_Service fill:#e8f5e9
        style Result_Cache fill:#e8eaf6
        style Queue_Manager fill:#e1f5ff
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **MPP Architecture** | Parallel query execution across nodes, linear scalability | Single-node DB (PostgreSQL - doesn't scale), SMP (limited scalability) |
    | **Columnar Storage (Parquet)** | 10x compression, fast column scans, predicate pushdown | Row storage (Avro - slow for analytics), ORC (less ecosystem support) |
    | **Cost-based Optimizer** | Choose optimal join order, scan strategy, significantly faster queries | Rule-based (suboptimal plans), no optimizer (always full scan) |
    | **S3 (Object Storage)** | Infinite scalability, 11 nines durability, $0.023/GB, decouples storage from compute | HDFS (expensive, complex ops), local disks (not scalable), EBS (costly) |
    | **Query Result Cache** | 80% cache hit rate, sub-second repeated queries | No cache (always recompute), client-side (inconsistent, stale) |
    | **Metadata Service** | Track partitions, statistics, avoid scanning S3 for metadata | Embedded in data files (slow), no metadata (full scans) |

    **Key Trade-off:** We chose **separation of compute and storage** (Snowflake model) over coupled architecture (traditional MPP). This enables independent scaling, lower costs, but adds network I/O latency.

    ---

    ## API Design

    ### 1. Execute SQL Query

    **Request:**
    ```sql
    -- Simple SELECT
    SELECT customer_id, order_date, total_amount
    FROM orders
    WHERE order_date >= '2026-01-01'
      AND region = 'US'
    ORDER BY order_date DESC
    LIMIT 1000;
    ```

    **Response:**
    ```json
    {
      "query_id": "019f8bc7-2341-ab89-0000-00000000dead",
      "status": "success",
      "execution_time_ms": 1234,
      "rows_returned": 1000,
      "bytes_scanned": 52428800,
      "partitions_scanned": 5,
      "partitions_total": 500,
      "cache_hit": false,
      "results": [
        {
          "customer_id": "cust_123",
          "order_date": "2026-02-05",
          "total_amount": 199.99
        },
        // ... more rows
      ]
    }
    ```

    **What happens internally:**

    1. **Parse & Validate:** SQL parser validates syntax
    2. **Semantic Analysis:** Resolve table/column references
    3. **Query Optimization:** Cost-based optimizer generates execution plan
    4. **Check Cache:** Query result cache lookup (hash of SQL + parameters)
    5. **Partition Pruning:** Filter partitions using metadata (WHERE clause)
       - 500 total partitions → 5 partitions (99% pruned)
    6. **Execute:** Distribute query to compute nodes (MPP)
    7. **Merge Results:** Coordinator aggregates results from all nodes
    8. **Cache Result:** Store in result cache (24-hour TTL)

    ---

    ### 2. Complex Aggregation Query

    **Request:**
    ```sql
    -- Multi-table join with aggregation
    SELECT
        d.region,
        d.product_category,
        DATE_TRUNC('month', f.order_date) AS month,
        COUNT(DISTINCT f.customer_id) AS unique_customers,
        SUM(f.total_amount) AS revenue,
        AVG(f.total_amount) AS avg_order_value
    FROM fact_orders f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    JOIN dim_products p ON f.product_id = p.product_id
    JOIN dim_dates d ON f.order_date = d.date
    WHERE d.year = 2026
      AND d.region IN ('US', 'CA', 'UK')
    GROUP BY d.region, d.product_category, month
    ORDER BY revenue DESC
    LIMIT 100;
    ```

    **Execution Plan:**
    ```
    Query Plan (cost-based optimizer):
    ┌─────────────────────────────────────────────┐
    │ Limit (n=100)                               │  Cost: 100
    │   Order By: revenue DESC                     │  Rows: 100
    │     ├─> Group By (region, category, month)  │  Cost: 1.2M
    │         Agg: COUNT(DISTINCT), SUM(), AVG()  │  Rows: 1,200
    │           ├─> Hash Join (f.product_id = p)  │  Cost: 500K
    │               Broadcast Join (p is small)   │  Rows: 10M
    │                 ├─> Hash Join (c.id = f)    │  Cost: 250K
    │                     Partitioned Join         │  Rows: 10M
    │                       ├─> Scan(fact_orders) │  Cost: 100K
    │                           Partition Pruning  │  Rows: 10M
    │                           Filter: year=2026  │  Partitions: 12/365
    │                       └─> Scan(dim_customers)│ Cost: 50K
    │                           Cache Hit          │  Rows: 1M
    │                 └─> Scan(dim_products)      │  Cost: 1K
    │                     Cache Hit               │  Rows: 10K
    └─────────────────────────────────────────────┘

    Optimizations applied:
    - Partition pruning: 365 partitions → 12 (97% reduction)
    - Predicate pushdown: year=2026 filter to storage layer
    - Broadcast join: Small dimension tables (products)
    - Partition join: Large tables co-located by customer_id
    - Column projection: Only read required columns (not all)
    ```

    **Performance:**
    ```
    Execution stats:
    - Planning time: 50ms
    - Execution time: 4.5 seconds
    - Bytes scanned: 120 GB (compressed: 12 GB)
    - Rows scanned: 10M (fact) + 1M (dims)
    - Rows returned: 100
    - Partitions scanned: 12 out of 365 (3.3%)
    - Cache: Dimension tables cached (1 GB)
    - Compute: 16 nodes × 8 cores = 128 parallel tasks
    ```

    ---

    ### 3. Create Materialized View

    **Request:**
    ```sql
    -- Create aggregated materialized view
    CREATE MATERIALIZED VIEW mv_daily_revenue AS
    SELECT
        order_date,
        region,
        product_category,
        COUNT(DISTINCT customer_id) AS unique_customers,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value
    FROM fact_orders f
    JOIN dim_products p ON f.product_id = p.product_id
    WHERE order_date >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY order_date, region, product_category;
    ```

    **Response:**
    ```json
    {
      "view_id": "mv_019f8bc7",
      "status": "created",
      "initial_build_time_ms": 45000,
      "rows_materialized": 50000,
      "storage_mb": 100,
      "refresh_schedule": "incremental_on_demand"
    }
    ```

    **What happens internally:**

    1. **Initial Build:** Execute query and store results
       - Time: 45 seconds
       - Storage: 100 MB (vs 10 TB base table)
       - Compression: 100x reduction

    2. **Incremental Refresh:** Update only changed partitions
       ```sql
       -- Refresh MV for yesterday's data
       REFRESH MATERIALIZED VIEW mv_daily_revenue
       WHERE order_date = CURRENT_DATE - 1;
       ```
       - Time: 1 second (only process 1 day)
       - Update strategy: DELETE + INSERT

    3. **Query Rewrite:** Optimizer automatically uses MV
       ```sql
       -- Original query
       SELECT region, SUM(total_revenue)
       FROM fact_orders
       WHERE order_date BETWEEN '2026-01-01' AND '2026-01-31'
       GROUP BY region;

       -- Rewritten query (automatic)
       SELECT region, SUM(total_revenue)
       FROM mv_daily_revenue
       WHERE order_date BETWEEN '2026-01-01' AND '2026-01-31'
       GROUP BY region;

       -- Performance: 10 TB scan → 100 MB scan (100,000x faster)
       ```

    ---

    ### 4. Load Data (COPY INTO)

    **Request:**
    ```sql
    -- Load data from S3 into table
    COPY INTO orders
    FROM 's3://my-bucket/data/orders/2026/02/05/'
    FILE_FORMAT = (
        TYPE = PARQUET
        COMPRESSION = SNAPPY
    )
    PATTERN = '.*\.parquet'
    ON_ERROR = CONTINUE
    VALIDATION_MODE = RETURN_ERRORS;
    ```

    **Response:**
    ```json
    {
      "load_id": "load_019f8bc7",
      "status": "success",
      "files_loaded": 100,
      "rows_loaded": 10000000,
      "bytes_loaded": 50000000000,
      "errors": 0,
      "execution_time_ms": 120000
    }
    ```

    **What happens internally:**

    1. **File Discovery:** List files in S3 matching pattern
       - 100 Parquet files found
       - Total size: 50 GB compressed

    2. **Parallel Load:** Distribute files across compute nodes
       - 16 nodes × 6 files/node = parallel load
       - Each node reads from S3 directly

    3. **Data Validation:** Check schema, constraints
       - Schema match: ✓
       - NOT NULL constraints: ✓
       - Data types: ✓

    4. **Micro-partitioning:** Split data into 16 MB partitions
       - 50 GB / 16 MB = 3,125 micro-partitions
       - Store min/max statistics per partition

    5. **Write to Storage:** Write to S3 in columnar format
       - Already Parquet, so minimal conversion
       - Update metadata catalog
       - Register partitions

    6. **Update Statistics:** Collect column statistics
       - Min/max values
       - Distinct count estimates (HyperLogLog)
       - Null count

    **Performance:**
    - Throughput: 50 GB / 120s = 416 MB/sec
    - Parallelism: 16 nodes
    - Per-node: 26 MB/sec (sustainable)

    ---

    ### 5. Time Travel Query

    **Request:**
    ```sql
    -- Query table as of 1 hour ago
    SELECT *
    FROM orders
    AT (TIMESTAMP => DATEADD(hour, -1, CURRENT_TIMESTAMP()))
    WHERE customer_id = 'cust_123';

    -- Query table as of specific time
    SELECT *
    FROM orders
    AT (TIMESTAMP => '2026-02-05 10:00:00')
    WHERE region = 'US';

    -- Show changes between two points in time
    SELECT *
    FROM orders
    CHANGES (INFORMATION => DEFAULT)
    AT (TIMESTAMP => DATEADD(hour, -24, CURRENT_TIMESTAMP()))
    END (TIMESTAMP => CURRENT_TIMESTAMP());
    ```

    **What happens internally:**

    1. **Version Lookup:** Find snapshot at requested time
       - Metadata stores version history
       - Each version tracks active partitions

    2. **Partition Resolution:** Build file list for that version
       - Files added before timestamp: Include
       - Files deleted before timestamp: Exclude
       - Files added after timestamp: Exclude

    3. **Execute Query:** Scan historical partitions
       - Read from S3 (files not deleted)
       - Same query execution as present

    **Use cases:**
    - Audit: "Show me sales data from last quarter end"
    - Debug: "What did the table look like when bug occurred?"
    - Rollback: "Restore table to yesterday's state"
    - Compliance: "Prove data lineage for regulatory audit"

    ---

    ### 6. Zero-copy Clone

    **Request:**
    ```sql
    -- Clone table instantly (no data copy)
    CREATE TABLE orders_dev
    CLONE orders;

    -- Clone database
    CREATE DATABASE analytics_dev
    CLONE analytics;

    -- Clone at specific time
    CREATE TABLE orders_snapshot
    CLONE orders
    AT (TIMESTAMP => '2026-02-05 00:00:00');
    ```

    **Response:**
    ```json
    {
      "clone_id": "clone_019f8bc7",
      "status": "success",
      "source_table": "orders",
      "target_table": "orders_dev",
      "bytes_cloned": 0,
      "execution_time_ms": 150
    }
    ```

    **What happens internally:**

    1. **Metadata Copy:** Copy table metadata only
       - Schema definition
       - Partition list (pointers to S3)
       - Statistics

    2. **Reference Counting:** Both tables reference same partitions
       - orders → partition_001 (ref_count=1)
       - orders_dev → partition_001 (ref_count=2)
       - Partition shared, not copied

    3. **Copy-on-Write:** When modified, create new partition
       ```sql
       -- Update clone (creates new partition)
       UPDATE orders_dev SET status = 'cancelled' WHERE id = 123;

       -- Original table unaffected
       -- New partition created only for modified data
       ```

    **Benefits:**
    - Instant: < 1 second for any size table
    - Storage efficient: Only store modifications
    - Use cases: Dev/test, backups, experimentation

    ---

    ## Database Schema Design

    ### Star Schema (Recommended)

    **Fact Table:**
    ```sql
    CREATE TABLE fact_orders (
        order_id BIGINT PRIMARY KEY,
        customer_id BIGINT,
        product_id BIGINT,
        date_id INT,
        quantity INT,
        unit_price DECIMAL(10, 2),
        total_amount DECIMAL(10, 2),
        discount DECIMAL(10, 2),
        tax DECIMAL(10, 2),
        order_timestamp TIMESTAMP,

        -- Clustered by date for partition pruning
        CLUSTER BY (date_id)
    )
    PARTITION BY DATE_TRUNC('day', order_timestamp);
    ```

    **Dimension Tables:**
    ```sql
    -- Customer dimension (slowly changing dimension type 2)
    CREATE TABLE dim_customers (
        customer_key BIGINT PRIMARY KEY,
        customer_id BIGINT,
        customer_name VARCHAR(255),
        email VARCHAR(255),
        country VARCHAR(50),
        region VARCHAR(50),
        segment VARCHAR(50),
        valid_from DATE,
        valid_to DATE,
        is_current BOOLEAN
    );

    -- Product dimension
    CREATE TABLE dim_products (
        product_id BIGINT PRIMARY KEY,
        product_name VARCHAR(255),
        category VARCHAR(100),
        subcategory VARCHAR(100),
        brand VARCHAR(100),
        unit_cost DECIMAL(10, 2)
    );

    -- Date dimension
    CREATE TABLE dim_dates (
        date_id INT PRIMARY KEY,
        date DATE,
        year INT,
        quarter INT,
        month INT,
        day INT,
        day_of_week INT,
        is_weekend BOOLEAN,
        is_holiday BOOLEAN
    );
    ```

    ### Snowflake Schema (Normalized)

    ```sql
    -- Fact table remains same
    CREATE TABLE fact_orders (...);

    -- Dimension with normalization
    CREATE TABLE dim_products (
        product_id BIGINT PRIMARY KEY,
        product_name VARCHAR(255),
        subcategory_id INT,
        brand_id INT
    );

    CREATE TABLE dim_subcategories (
        subcategory_id INT PRIMARY KEY,
        subcategory_name VARCHAR(100),
        category_id INT
    );

    CREATE TABLE dim_categories (
        category_id INT PRIMARY KEY,
        category_name VARCHAR(100)
    );
    ```

    **Star vs Snowflake trade-off:**
    - Star: Fewer joins (faster queries), denormalized (more storage)
    - Snowflake: More joins (slower), normalized (less storage, no duplicates)
    - Recommendation: **Star schema** for data warehouses (query performance > storage)

=== "🔧 Step 3: Deep Dive"

    ## 3.1 MPP (Massively Parallel Processing) Architecture

    **Challenge:** How to execute queries across thousands of nodes efficiently?

    **Solution:** Distribute data and computation using hash partitioning and shuffle operations

    ---

    ### Query Execution Model

    **Distributed Query Execution:**

    ```python
    class MPPQueryExecutor:
        """
        Massively Parallel Processing query executor
        """

        def __init__(self, num_nodes=16):
            self.coordinator = CoordinatorNode()
            self.compute_nodes = [ComputeNode(i) for i in range(num_nodes)]

        def execute_query(self, sql: str):
            """
            Execute SQL query using MPP
            """
            # 1. Parse and optimize query
            plan = self.coordinator.parse_and_optimize(sql)
            print(f"Execution plan: {plan}")

            # 2. Generate execution fragments
            fragments = self.coordinator.generate_fragments(plan)
            print(f"Generated {len(fragments)} fragments")

            # 3. Distribute fragments to compute nodes
            tasks = self.coordinator.schedule_fragments(fragments, self.compute_nodes)

            # 4. Execute in parallel
            results = self.execute_parallel(tasks)

            # 5. Merge results at coordinator
            final_result = self.coordinator.merge_results(results)

            return final_result

        def execute_parallel(self, tasks):
            """
            Execute tasks in parallel across compute nodes
            """
            results = []

            # Execute each fragment in parallel
            with ThreadPoolExecutor(max_workers=len(self.compute_nodes)) as executor:
                futures = []
                for task in tasks:
                    node = self.compute_nodes[task.node_id]
                    future = executor.submit(node.execute_fragment, task.fragment)
                    futures.append(future)

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            return results
    ```

    **Example: Distributed Join Execution**

    ```sql
    -- Query: Join orders with customers
    SELECT c.customer_name, COUNT(*) AS order_count, SUM(o.total_amount) AS revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_date >= '2026-01-01'
    GROUP BY c.customer_name;
    ```

    **Execution Plan (MPP):**

    ```
    Fragment 0 (Coordinator):
    ├─> Aggregate (final): SUM(order_count), SUM(revenue)
        Group By: customer_name
        ├─> Exchange: Gather results from all nodes

    Fragment 1 (Compute Nodes):
    ├─> Aggregate (partial): COUNT(*), SUM(total_amount)
        Group By: customer_name
        ├─> Hash Join: o.customer_id = c.customer_id
            ├─> Scan: orders (partitioned by customer_id)
            └─> Scan: customers (broadcast to all nodes)

    Data flow:
    1. Broadcast customers to all nodes (small table)
    2. Each node scans local orders partition
    3. Each node performs local join + partial aggregation
    4. Coordinator gathers and merges results
    ```

    **Join Strategies:**

    1. **Broadcast Join** (for small tables)
       - Send small table to all nodes
       - Each node joins with local partition
       - Use when: One table < 10 MB

    2. **Shuffle Hash Join** (for large tables)
       - Hash partition both tables by join key
       - Shuffle data across network
       - Co-locate matching rows on same node
       - Use when: Both tables large, evenly distributed

    3. **Bucket Join** (pre-partitioned)
       - Tables already partitioned by join key
       - No shuffle needed (data co-located)
       - Use when: Tables pre-clustered by join key

    **Performance:**
    ```
    Query: Join 1B orders with 10M customers

    Without MPP (single node):
    - Scan orders: 1B rows × 100 bytes = 100 GB
    - Scan customers: 10M rows × 200 bytes = 2 GB
    - Join time: 600 seconds (nested loop)

    With MPP (16 nodes):
    - Broadcast customers: 2 GB to 16 nodes = 2 GB network (1 sec)
    - Scan orders: 1B rows / 16 = 62.5M rows/node (parallel)
    - Join time: 600 / 16 = 37.5 seconds
    - Total: 38 seconds (16x speedup)
    ```

    ---

    ## 3.2 Columnar Storage (Parquet)

    **Why Columnar?**

    1. **Better compression:** Same data type in column compresses well
    2. **Column projection:** Read only needed columns
    3. **Predicate pushdown:** Filter before reading into memory
    4. **Vectorized execution:** SIMD operations on columns

    ---

    ### Parquet File Structure

    ```
    Parquet File Layout:
    ┌─────────────────────────────────────────┐
    │ Magic Number: PAR1                      │
    ├─────────────────────────────────────────┤
    │ Row Group 1 (128 MB)                    │
    │   ┌─────────────────────────────────┐   │
    │   │ Column Chunk: customer_id        │   │
    │   │   - Dictionary encoding         │   │
    │   │   - RLE/Bit-packing             │   │
    │   │   - Snappy compression          │   │
    │   │   - Min/max: [1000, 9999]       │   │
    │   └─────────────────────────────────┘   │
    │   ┌─────────────────────────────────┐   │
    │   │ Column Chunk: order_date         │   │
    │   │   - Delta encoding              │   │
    │   │   - Min/max: [2026-01-01, ...]  │   │
    │   └─────────────────────────────────┘   │
    │   ┌─────────────────────────────────┐   │
    │   │ Column Chunk: total_amount       │   │
    │   │   - Floating point encoding     │   │
    │   └─────────────────────────────────┘   │
    ├─────────────────────────────────────────┤
    │ Row Group 2 (128 MB)                    │
    │   ...                                   │
    ├─────────────────────────────────────────┤
    │ Footer                                  │
    │   - Schema                              │
    │   - Row group metadata                  │
    │   - Column chunk offsets                │
    └─────────────────────────────────────────┘
    ```

    **Encoding Schemes:**

    1. **Dictionary Encoding** (for low cardinality)
       ```
       Original: ["US", "US", "UK", "US", "UK"]
       Dictionary: {0: "US", 1: "UK"}
       Encoded: [0, 0, 1, 0, 1]
       Compression: 5 × 2 bytes = 10 bytes → 5 × 1 bit + 4 bytes dict = 5 bytes (50% savings)
       ```

    2. **Run-Length Encoding (RLE)**
       ```
       Original: [1, 1, 1, 1, 2, 2, 2, 3, 3]
       Encoded: [(1, 4), (2, 3), (3, 2)]
       Compression: 9 values → 3 tuples (67% savings)
       ```

    3. **Delta Encoding** (for sorted/monotonic)
       ```
       Original timestamps: [1000, 1001, 1002, 1003]
       Base: 1000
       Deltas: [0, 1, 1, 1]
       Compression: 4 × 8 bytes = 32 bytes → 8 + 4 × 1 byte = 12 bytes (62% savings)
       ```

    **Predicate Pushdown:**

    ```python
    def read_parquet_with_filter(file_path: str, filter_expr: str):
        """
        Read Parquet file with predicate pushdown
        """
        # Open Parquet file
        parquet_file = ParquetFile(file_path)

        # Read footer (metadata only, no data read yet)
        metadata = parquet_file.metadata

        # Check row group statistics
        row_groups_to_read = []
        for rg in metadata.row_groups:
            # Get column statistics (min/max)
            customer_id_stats = rg.columns['customer_id'].statistics

            # Filter: customer_id = 5000
            if customer_id_stats.min <= 5000 <= customer_id_stats.max:
                # This row group might contain matching rows
                row_groups_to_read.append(rg)
            else:
                # Skip this row group (no matches possible)
                print(f"Skipped row group {rg.ordinal} (out of range)")

        print(f"Reading {len(row_groups_to_read)} out of {len(metadata.row_groups)} row groups")

        # Read only matching row groups
        results = []
        for rg in row_groups_to_read:
            # Read only required columns (column projection)
            data = parquet_file.read_row_group(rg, columns=['customer_id', 'total_amount'])

            # Apply filter in-memory (for rows that passed min/max check)
            filtered = data[data['customer_id'] == 5000]
            results.append(filtered)

        return pd.concat(results)

    # Example performance
    # Without pushdown: Read 10 GB, filter to 10 MB (99.9% wasted I/O)
    # With pushdown: Read 100 MB, filter to 10 MB (99% I/O savings)
    ```

    **Compression Ratios:**
    ```
    Data type          | Raw size | Encoded | Compressed | Total ratio
    -------------------|----------|---------|------------|------------
    INT (sequential)   | 8 bytes  | 1 byte  | 0.5 bytes  | 16x
    STRING (repeating) | 20 bytes | 4 bytes | 2 bytes    | 10x
    FLOAT (random)     | 8 bytes  | 8 bytes | 6 bytes    | 1.3x
    TIMESTAMP (sorted) | 8 bytes  | 2 bytes | 1 byte     | 8x

    Average compression: 10x (Parquet + Snappy)
    ```

    ---

    ## 3.3 Cost-based Query Optimizer

    **Goal:** Choose optimal execution plan based on statistics

    ---

    ### Optimizer Components

    ```python
    class CostBasedOptimizer:
        """
        Cost-based query optimizer
        """

        def __init__(self, statistics_store):
            self.stats = statistics_store

        def optimize(self, logical_plan):
            """
            Transform logical plan to optimal physical plan
            """
            # 1. Apply logical optimizations
            plan = self.apply_logical_rules(logical_plan)

            # 2. Generate alternative physical plans
            alternatives = self.generate_alternatives(plan)

            # 3. Estimate cost of each alternative
            costs = [self.estimate_cost(alt) for alt in alternatives]

            # 4. Choose plan with lowest cost
            best_plan = alternatives[costs.index(min(costs))]

            return best_plan

        def apply_logical_rules(self, plan):
            """
            Logical optimizations (rule-based)
            """
            # Predicate pushdown
            plan = self.push_down_predicates(plan)

            # Column projection (only read needed columns)
            plan = self.push_down_projections(plan)

            # Constant folding
            plan = self.fold_constants(plan)

            # Join reordering (cost-based)
            plan = self.reorder_joins(plan)

            return plan

        def estimate_cost(self, plan):
            """
            Estimate execution cost

            Cost = I/O cost + CPU cost + Network cost
            """
            io_cost = self.estimate_io_cost(plan)
            cpu_cost = self.estimate_cpu_cost(plan)
            network_cost = self.estimate_network_cost(plan)

            total_cost = io_cost + cpu_cost + network_cost
            return total_cost

        def estimate_io_cost(self, plan):
            """
            Estimate I/O cost (bytes scanned)
            """
            if plan.type == 'Scan':
                table = plan.table
                stats = self.stats.get_table_stats(table)

                # Base cost: scan all data
                base_cost = stats.row_count * stats.avg_row_size

                # Apply selectivity from filters
                selectivity = self.estimate_selectivity(plan.filters, stats)
                cost = base_cost * selectivity

                # Partition pruning
                pruning_ratio = self.estimate_pruning(plan.filters, stats)
                cost = cost * pruning_ratio

                return cost

        def estimate_selectivity(self, filters, stats):
            """
            Estimate filter selectivity (fraction of rows matching)
            """
            selectivity = 1.0

            for filter in filters:
                if filter.op == '=':
                    # Equality: 1 / distinct_count
                    distinct = stats.columns[filter.column].distinct_count
                    selectivity *= 1.0 / distinct

                elif filter.op == '<' or filter.op == '>':
                    # Range: Estimate using histogram
                    histogram = stats.columns[filter.column].histogram
                    selectivity *= histogram.estimate_range(filter.value)

                elif filter.op == 'IN':
                    # IN list: count / distinct_count
                    distinct = stats.columns[filter.column].distinct_count
                    selectivity *= len(filter.values) / distinct

            return selectivity

        def reorder_joins(self, plan):
            """
            Reorder joins to minimize intermediate result size

            Dynamic programming algorithm (similar to Postgres)
            """
            if len(plan.tables) <= 1:
                return plan

            # Build join graph
            join_graph = self.build_join_graph(plan)

            # Find optimal join order using DP
            optimal_order = self.dynamic_programming_join(join_graph)

            # Reconstruct plan with optimal order
            return self.build_join_plan(optimal_order)

        def dynamic_programming_join(self, graph):
            """
            Dynamic programming join ordering

            Time: O(n * 2^n) where n = number of tables
            """
            n = len(graph.tables)

            # dp[subset] = (best_cost, best_plan)
            dp = {}

            # Base case: Single tables
            for i in range(n):
                subset = frozenset([i])
                cost = self.estimate_scan_cost(graph.tables[i])
                dp[subset] = (cost, [i])

            # Build up larger subsets
            for size in range(2, n + 1):
                for subset in itertools.combinations(range(n), size):
                    subset = frozenset(subset)
                    best_cost = float('inf')
                    best_plan = None

                    # Try all ways to split subset
                    for left_size in range(1, size):
                        for left in itertools.combinations(subset, left_size):
                            left = frozenset(left)
                            right = subset - left

                            # Check if join is valid (connected)
                            if not self.is_connected(left, right, graph):
                                continue

                            # Compute cost of this join
                            left_cost, left_plan = dp[left]
                            right_cost, right_plan = dp[right]
                            join_cost = self.estimate_join_cost(left, right, graph)

                            total_cost = left_cost + right_cost + join_cost

                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_plan = (left_plan, right_plan)

                    dp[subset] = (best_cost, best_plan)

            # Return optimal plan for all tables
            all_tables = frozenset(range(n))
            return dp[all_tables][1]
    ```

    **Example: Join Order Optimization**

    ```sql
    SELECT *
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE c.country = 'US' AND p.category = 'electronics';
    ```

    **Statistics:**
    ```
    Table          | Rows      | Avg row size | Filter selectivity
    ---------------|-----------|--------------|-------------------
    orders         | 1,000,000 | 100 bytes    | 1.0 (no filter)
    customers      | 100,000   | 200 bytes    | 0.3 (country = US)
    products       | 10,000    | 150 bytes    | 0.1 (category = electronics)
    ```

    **Alternative join orders:**

    ```
    Option 1: (orders ⋈ customers) ⋈ products
    - orders ⋈ customers: 1M × 100K = 100B comparisons
    - Intermediate: 1M × 0.3 = 300K rows
    - Result ⋈ products: 300K × 10K = 3B comparisons
    - Total: 103B comparisons

    Option 2: (orders ⋈ products) ⋈ customers
    - orders ⋈ products: 1M × 10K = 10B comparisons
    - Intermediate: 1M × 0.1 = 100K rows
    - Result ⋈ customers: 100K × 100K = 10B comparisons
    - Total: 20B comparisons (5x better!)

    Option 3: (customers ⋈ products) ⋈ orders
    - customers ⋈ products: No direct join key! (Cartesian product)
    - 100K × 10K = 1B rows (too large)
    - Not valid

    Optimizer chooses Option 2 (lowest cost)
    ```

    **Execution plan with costs:**
    ```
    Hash Join (o.customer_id = c.customer_id)  Cost: 10B
    ├─> Scan customers (country = 'US')        Cost: 100K × 200 = 20 MB
    └─> Hash Join (o.product_id = p.product_id)  Cost: 10B
        ├─> Scan orders                        Cost: 1M × 100 = 100 MB
        └─> Scan products (category = 'electronics')  Cost: 10K × 150 = 1.5 MB

    Total I/O: 121.5 MB (vs 220 MB for Option 1)
    ```

    ---

    ## 3.4 Micro-partitioning (Snowflake)

    **Concept:** Automatically partition data into small (16 MB) immutable files

    **Benefits:**
    1. Optimal for S3 (16 MB = good parallel read size)
    2. Fine-grained partition pruning
    3. Easy to add/remove partitions (immutable)
    4. Enable time travel (track partition versions)

    ---

    ### Micro-partition Implementation

    ```python
    class MicroPartitioner:
        """
        Snowflake-style micro-partitioning
        """

        def __init__(self, target_size_mb=16):
            self.target_size_mb = target_size_mb
            self.target_size_bytes = target_size_mb * 1024 * 1024

        def partition_data(self, data: DataFrame, cluster_keys: List[str]):
            """
            Partition data into micro-partitions
            """
            # 1. Sort by cluster keys (for better pruning)
            if cluster_keys:
                data = data.sort_values(by=cluster_keys)

            # 2. Split into 16 MB chunks
            partitions = []
            current_partition = []
            current_size = 0

            for row in data.itertuples():
                row_size = self.estimate_row_size(row)

                if current_size + row_size > self.target_size_bytes:
                    # Flush current partition
                    partition = self.create_partition(current_partition, cluster_keys)
                    partitions.append(partition)

                    # Start new partition
                    current_partition = [row]
                    current_size = row_size
                else:
                    current_partition.append(row)
                    current_size += row_size

            # Flush last partition
            if current_partition:
                partition = self.create_partition(current_partition, cluster_keys)
                partitions.append(partition)

            print(f"Created {len(partitions)} micro-partitions")
            return partitions

        def create_partition(self, rows, cluster_keys):
            """
            Create micro-partition with metadata
            """
            df = pd.DataFrame(rows)

            # Collect statistics (min/max for each column)
            stats = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'datetime64']:
                    stats[col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'null_count': df[col].isnull().sum(),
                        'distinct_count': df[col].nunique()
                    }

            # Write to Parquet
            partition_id = str(uuid.uuid4())
            file_path = f"s3://bucket/table/partitions/{partition_id}.parquet"
            write_parquet(df, file_path)

            # Return partition metadata
            return {
                'partition_id': partition_id,
                'file_path': file_path,
                'row_count': len(df),
                'compressed_size': get_file_size(file_path),
                'uncompressed_size': df.memory_usage(deep=True).sum(),
                'statistics': stats,
                'cluster_keys': cluster_keys
            }

        def prune_partitions(self, partitions, filters):
            """
            Prune partitions using min/max statistics
            """
            pruned = []

            for partition in partitions:
                if self.partition_matches_filters(partition, filters):
                    pruned.append(partition)

            print(f"Pruned {len(partitions)} → {len(pruned)} partitions " +
                  f"({len(pruned) / len(partitions) * 100:.1f}%)")

            return pruned

        def partition_matches_filters(self, partition, filters):
            """
            Check if partition can match filters based on min/max
            """
            stats = partition['statistics']

            for filter in filters:
                col = filter['column']
                op = filter['operator']
                value = filter['value']

                if col not in stats:
                    continue  # No stats, conservatively include

                col_min = stats[col]['min']
                col_max = stats[col]['max']

                # Check if filter can match partition
                if op == '=':
                    if value < col_min or value > col_max:
                        return False  # Value outside range

                elif op == '>':
                    if col_max <= value:
                        return False  # All values too small

                elif op == '<':
                    if col_min >= value:
                        return False  # All values too large

                elif op == 'BETWEEN':
                    if value[1] < col_min or value[0] > col_max:
                        return False  # No overlap

            return True  # Partition might match
    ```

    **Example: Partition Pruning**

    ```
    Table: orders (1 TB, 64,000 micro-partitions)
    Clustered by: order_date

    Query: SELECT * FROM orders WHERE order_date = '2026-02-05'

    Partition pruning:
    1. Check all 64,000 partitions' min/max statistics
    2. Filter partitions where order_date range overlaps '2026-02-05'
    3. Result: 100 partitions match (0.15% of total)
    4. Scan only 100 × 16 MB = 1.6 GB (vs 1 TB full scan)
    5. Speedup: 625x faster

    Without micro-partitions:
    - Coarse partitioning (daily): 365 partitions
    - Partition size: 2.7 GB each
    - Must scan entire day's partition (2.7 GB)
    - Less precise pruning

    With micro-partitions:
    - Fine-grained: 175 partitions/day
    - Only scan matching micro-partitions (1.6 GB)
    - 1.7x better pruning
    ```

=== "📊 Step 4: Scale & Optimize"

    ## Performance at Scale

    ### Query Performance Optimization

    **Optimization Techniques:**

    1. **Clustering Keys**
       ```sql
       -- Cluster table by commonly filtered column
       ALTER TABLE orders
       CLUSTER BY (order_date, region);

       -- Query benefits from clustering
       SELECT * FROM orders
       WHERE order_date >= '2026-01-01' AND region = 'US';

       -- Scan reduction: 99% (only scan matching clusters)
       ```

    2. **Materialized Views**
       ```sql
       -- Pre-aggregate expensive query
       CREATE MATERIALIZED VIEW mv_daily_sales AS
       SELECT order_date, region, SUM(total_amount) AS revenue
       FROM orders
       GROUP BY order_date, region;

       -- Query rewritten to use MV (automatic)
       SELECT order_date, SUM(revenue) FROM orders
       WHERE order_date >= '2026-01-01'
       GROUP BY order_date;

       -- Speedup: 1000x (scan 100 MB MV vs 100 GB base table)
       ```

    3. **Result Caching**
       ```sql
       -- Identical queries hit cache
       SELECT COUNT(*) FROM orders WHERE order_date = '2026-02-05';
       -- First run: 5 seconds (scan 10 GB)
       -- Cached run: 0.1 seconds (Redis lookup)
       -- Cache TTL: 24 hours
       ```

    4. **Query Hints**
       ```sql
       -- Force broadcast join for small table
       SELECT /*+ BROADCAST(customers) */ *
       FROM orders o
       JOIN customers c ON o.customer_id = c.customer_id;
       ```

    **Performance Benchmarks:**

    ```
    Query Type                      | Before Optimization | After Optimization | Speedup
    --------------------------------|---------------------|-------------------|--------
    Simple SELECT (indexed)         | 2s                  | 0.1s              | 20x
    Aggregation (single table)      | 30s                 | 1s                | 30x
    Join (2 large tables)           | 120s                | 10s               | 12x
    Complex join (3+ tables)        | 600s                | 45s               | 13x
    Full table scan (1 TB)          | 300s                | 30s (partitioned) | 10x

    Optimizations applied:
    - Partition pruning: 90% data skipped
    - Column projection: Read only needed columns (20% of columns)
    - Predicate pushdown: Filter at storage layer
    - Result caching: 80% cache hit rate
    - Materialized views: 1000x for aggregations
    ```

    ---

    ### Concurrency & Workload Management

    **Challenge:** Handle 10,000 concurrent queries without resource starvation

    **Solution:** Multi-cluster warehouses with workload isolation

    ```sql
    -- Create warehouses for different workloads
    CREATE WAREHOUSE ETL_WH WITH
        WAREHOUSE_SIZE = 'X-Large'
        AUTO_SUSPEND = 600
        AUTO_RESUME = TRUE
        MIN_CLUSTER_COUNT = 2
        MAX_CLUSTER_COUNT = 10
        SCALING_POLICY = 'STANDARD';

    CREATE WAREHOUSE REPORTING_WH WITH
        WAREHOUSE_SIZE = 'Small'
        AUTO_SUSPEND = 300
        MAX_CLUSTER_COUNT = 5;

    CREATE WAREHOUSE ADHOC_WH WITH
        WAREHOUSE_SIZE = 'Medium'
        AUTO_SUSPEND = 60
        AUTO_RESUME = TRUE
        MAX_CLUSTER_COUNT = 20
        SCALING_POLICY = 'ECONOMY';
    ```

    **Auto-scaling Logic:**

    ```python
    class AutoScaler:
        """
        Auto-scale compute clusters based on queue depth
        """

        def __init__(self, warehouse):
            self.warehouse = warehouse
            self.min_clusters = warehouse.min_cluster_count
            self.max_clusters = warehouse.max_cluster_count

        def scale(self):
            """
            Scale clusters based on queue depth
            """
            queue_depth = self.get_queue_depth()
            current_clusters = self.get_current_cluster_count()

            if queue_depth > 10 and current_clusters < self.max_clusters:
                # Scale up: Queue backing up
                new_count = min(current_clusters + 1, self.max_clusters)
                self.add_cluster()
                print(f"Scaled up to {new_count} clusters (queue depth: {queue_depth})")

            elif queue_depth == 0 and current_clusters > self.min_clusters:
                # Scale down: No queued queries
                idle_time = self.get_cluster_idle_time()
                if idle_time > 600:  # 10 minutes idle
                    self.remove_cluster()
                    print(f"Scaled down to {current_clusters - 1} clusters")

        def get_queue_depth(self):
            """
            Get number of queries waiting in queue
            """
            return len(self.warehouse.query_queue)
    ```

    **Query Priority:**

    ```sql
    -- High priority (ETL, critical reports)
    ALTER SESSION SET QUERY_PRIORITY = 'HIGH';

    -- Low priority (exploratory, ad-hoc)
    ALTER SESSION SET QUERY_PRIORITY = 'LOW';

    -- Priority scheduling:
    -- HIGH: Execute immediately (preempt LOW queries)
    -- NORMAL: Execute in order
    -- LOW: Execute when capacity available
    ```

    **Resource Limits:**

    ```sql
    -- Limit query execution time
    ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300;

    -- Limit memory per query
    ALTER SESSION SET MAX_MEMORY_MB = 10240;

    -- Kill runaway queries automatically
    ```

    ---

    ### Cost Optimization

    **Goal:** Minimize costs while maintaining performance

    **Strategies:**

    1. **Auto-suspend Warehouses**
       ```sql
       -- Suspend after 5 minutes of inactivity
       ALTER WAREHOUSE ETL_WH SET AUTO_SUSPEND = 300;

       -- Savings: 50-80% (run 4 hrs/day instead of 24 hrs/day)
       ```

    2. **Query Result Caching**
       ```
       Cache hit rate: 80%
       Without cache: 100K queries × $0.01/query = $1,000/day
       With cache: 20K queries × $0.01/query = $200/day
       Savings: $800/day = $24K/month
       ```

    3. **Materialized Views**
       ```
       Aggregation query: Scan 100 GB → $0.50/query
       With MV: Scan 100 MB → $0.0005/query (1000x cheaper)
       Run 1000x/day: $500/day → $0.50/day
       Savings: $499.50/day = $15K/month
       ```

    4. **Partition Pruning**
       ```
       Full table scan: 10 TB × $5/TB = $50/query
       With pruning (90% reduction): 1 TB × $5/TB = $5/query
       Savings: $45/query × 1000 queries/day = $45K/day
       ```

    5. **Query Optimization**
       ```sql
       -- Bad: Select all columns
       SELECT * FROM large_table;  -- Scan 100 GB

       -- Good: Select only needed columns
       SELECT id, name, amount FROM large_table;  -- Scan 10 GB

       -- Savings: 90% reduction in scan costs
       ```

    **Total Cost Breakdown:**

    ```
    Monthly costs (10 PB data, 100K queries/day):

    Storage:
    - S3 storage: 2.5 PB × $10/TB = $25,000
    - Metadata: 160 GB × $23/TB = $3.68
    - Total storage: $25,000/month

    Compute:
    - ETL warehouse: 1440 credits × $3 = $4,320
    - Reporting warehouse: 720 credits × $3 = $2,160
    - Ad-hoc warehouse: 2880 credits × $3 = $8,640
    - Total compute: $15,120/month

    Data transfer:
    - Egress: 1 TB/day × 30 × $0.09/GB = $2,700
    - Total transfer: $2,700/month

    Total cost: $42,820/month ≈ $514K/year
    Cost per TB: $514K / 10,000 TB = $51/TB/year = $4.25/TB/month

    With optimizations (cache, MVs, auto-suspend):
    - Storage: $25,000 (same)
    - Compute: $6,000 (60% reduction)
    - Transfer: $1,000 (63% reduction)
    - Total: $32,000/month ≈ $384K/year (25% savings)
    ```

    ---

    ### Scalability

    **Horizontal Scaling:**

    1. **Storage Scaling** (unlimited)
       - Store data in S3 (infinite scalability)
       - Add micro-partitions as data grows
       - No re-sharding or data movement

    2. **Compute Scaling** (elastic)
       - Auto-scale from 1 to 128 nodes
       - Add clusters in seconds
       - Linear performance scaling

    3. **Metadata Scaling**
       - Shard metadata by database/schema
       - Cache metadata in Redis (distributed)
       - Handle 1M+ tables

    **Performance Scaling:**

    ```
    Data size  | Compute nodes | Query time | Throughput
    -----------|---------------|------------|------------
    1 TB       | 1             | 10s        | 100 GB/s
    10 TB      | 10            | 10s        | 1 TB/s
    100 TB     | 100           | 10s        | 10 TB/s
    1 PB       | 1000          | 10s        | 100 TB/s

    Linear scaling achieved through:
    - MPP parallelism
    - Partition-level parallelism
    - No shared state between nodes
    ```

    ---

    ### Trade-offs

    | Aspect | Option A | Option B | Recommendation |
    |--------|----------|----------|----------------|
    | **Architecture** | Coupled (Redshift) | Separated (Snowflake) | **Separated** (better scalability, lower cost) |
    | **Clustering** | Manual | Auto-clustering | **Auto** (less maintenance, better performance) |
    | **Concurrency** | WLM queues | Multi-cluster | **Multi-cluster** (better isolation, auto-scaling) |
    | **Pricing** | Per-node-hour | Per-second | **Per-second** (pay only for usage) |
    | **Storage** | Node-attached | S3/GCS/ADLS | **Object storage** (cheaper, unlimited) |
    | **Caching** | Local SSD | Result cache | **Both** (local for data, result for queries) |

    ---

    ### Interview Tips

    **Common Follow-up Questions:**

    1. **"Why separate compute and storage?"**
       - **Storage:** Needs to be durable, cheap, infinite scale
       - **Compute:** Needs to be fast, elastic, pay-per-use
       - **Benefit:** Scale independently, pause compute, share data across warehouses
       - **Trade-off:** Network I/O latency (mitigated by caching)

    2. **"How does MPP differ from single-node DB?"**
       - **MPP:** Distribute data and queries across many nodes
       - **Single-node:** All data on one machine (limited by disk/memory)
       - **Benefit:** Linear scaling (10x nodes = 10x performance)
       - **Challenge:** Network overhead, data shuffling for joins

    3. **"How does cost-based optimizer work?"**
       - **Statistics:** Collect row counts, distinct values, min/max
       - **Cardinality estimation:** Predict result size for each operation
       - **Cost model:** Assign costs (I/O, CPU, network)
       - **Search:** Find plan with lowest cost (dynamic programming)
       - **Example:** Choose join order, scan vs index, broadcast vs shuffle

    4. **"What is micro-partitioning?"**
       - **Concept:** Automatically split data into small (16 MB) files
       - **Benefit:** Fine-grained pruning, optimal S3 read size, immutable
       - **vs Hive partitioning:** Automatic (no manual PARTITION BY), smaller granularity
       - **Statistics:** Store min/max per partition for pruning

    5. **"How do materialized views speed up queries?"**
       - **Pre-compute:** Execute query once, store results
       - **Incremental refresh:** Update only changed data (delta)
       - **Query rewrite:** Optimizer automatically uses MV
       - **Speedup:** 1000x for aggregations (scan 100 MB vs 100 GB)
       - **Trade-off:** Storage cost, refresh latency

    6. **"How does time travel work?"**
       - **Immutable files:** Never modify, only add/remove
       - **Version history:** Track file additions/removals per commit
       - **Query at time T:** Use files active at time T
       - **Cost:** Minimal (metadata-only, old files retained)
       - **Use cases:** Audit, debug, rollback, reproducibility

    7. **"How do you handle semi-structured data (JSON)?"**
       - **VARIANT type:** Store JSON as binary format
       - **Automatic schema detection:** Parse JSON structure
       - **Flatten on read:** Use JSON path expressions (e.g., `data:user.name`)
       - **Indexing:** Build statistics on commonly accessed paths
       - **Performance:** Slower than structured (no columnar), but flexible

    8. **"Snowflake vs BigQuery vs Redshift?"**
       - **Snowflake:** Best UX, auto-clustering, time travel, zero-copy clone
       - **BigQuery:** Serverless, no warehouses, best for ad-hoc, built-in ML
       - **Redshift:** AWS-native, cheapest, best for large committed workloads
       - **Recommendation:** BigQuery for simplicity, Snowflake for features, Redshift for cost

    **Key Points to Mention:**

    - Data warehouse built on MPP architecture for parallel query execution
    - Columnar storage (Parquet) provides 10x compression and fast analytics
    - Cost-based optimizer chooses optimal query plan using statistics
    - Separation of compute and storage enables elastic scaling and low cost
    - Micro-partitioning enables fine-grained pruning (99% data skipped)
    - Materialized views provide 1000x speedup for aggregations
    - Multi-cluster warehouses isolate workloads and auto-scale
    - S3 provides 11 nines durability and unlimited scalability

    ---

    ## Additional Considerations

    ### Security & Compliance

    ```
    Security layers:

    1. Encryption
       - At-rest: S3 SSE-KMS (all data encrypted)
       - In-transit: TLS 1.3 (all connections)
       - Column-level: Encrypt PII columns
       - Key rotation: Automatic every 90 days

    2. Access Control
       - Role-based access control (RBAC)
       - Object-level privileges (table, view, warehouse)
       - Row-level security (filter by user)
       - Column masking (hide sensitive columns)

    3. Auditing
       - Query history: Track all executed queries
       - Access logs: Who accessed what data
       - Data lineage: Track data flow
       - CloudTrail integration

    4. Compliance
       - GDPR: Right to erasure, data residency
       - SOC 2 Type II: Security controls
       - HIPAA: PHI data protection
       - PCI DSS: Credit card data
    ```

    ### Data Governance

    ```sql
    -- Row-level security
    CREATE ROW ACCESS POLICY region_filter
    AS (region STRING) RETURNS BOOLEAN ->
      CASE
        WHEN CURRENT_ROLE() = 'GLOBAL_ADMIN' THEN TRUE
        WHEN CURRENT_ROLE() = 'US_ANALYST' THEN region = 'US'
        ELSE FALSE
      END;

    ALTER TABLE orders
    ADD ROW ACCESS POLICY region_filter ON (region);

    -- Column masking
    CREATE MASKING POLICY email_mask
    AS (val STRING) RETURNS STRING ->
      CASE
        WHEN CURRENT_ROLE() IN ('ADMIN', 'COMPLIANCE') THEN val
        ELSE '***masked***'
      END;

    ALTER TABLE customers
    MODIFY COLUMN email SET MASKING POLICY email_mask;
    ```

    ### Monitoring & Observability

    ```sql
    -- Query performance monitoring
    SELECT
        query_id,
        user_name,
        warehouse_name,
        query_text,
        execution_time,
        bytes_scanned,
        bytes_spilled_to_disk,
        partitions_scanned,
        cache_hit
    FROM snowflake.account_usage.query_history
    WHERE execution_time > 60000  -- > 60 seconds
    ORDER BY execution_time DESC
    LIMIT 100;

    -- Cost monitoring
    SELECT
        warehouse_name,
        SUM(credits_used) AS total_credits,
        SUM(credits_used) * 3 AS total_cost_usd
    FROM snowflake.account_usage.warehouse_metering_history
    WHERE start_time >= DATEADD(day, -30, CURRENT_TIMESTAMP())
    GROUP BY warehouse_name
    ORDER BY total_credits DESC;

    -- Alert on expensive queries
    CREATE ALERT expensive_queries
    WAREHOUSE = monitoring_wh
    SCHEDULE = '5 minutes'
    IF (EXISTS (
        SELECT 1 FROM query_history
        WHERE execution_time > 300000  -- > 5 minutes
        AND start_time > DATEADD(minute, -5, CURRENT_TIMESTAMP())
    ))
    THEN CALL send_alert('Expensive queries detected');
    ```

    ---

    ## Summary

    **System Characteristics:**

    - **Scale:** 10 PB data, 10K concurrent queries, 100K queries/day
    - **Latency:** < 1s simple queries, < 60s complex queries (p95)
    - **Cost:** $4.25/TB/month (storage + compute)
    - **Compression:** 10x (columnar + encoding)

    **Core Components:**

    1. **MPP Architecture:** Parallel query execution across nodes
    2. **Columnar Storage (Parquet):** 10x compression, fast analytics
    3. **Cost-based Optimizer:** Optimal query plans using statistics
    4. **Metadata Service:** Track partitions, statistics, versions
    5. **Query Result Cache:** 80% hit rate, sub-second repeated queries
    6. **Workload Management:** Multi-cluster, auto-scaling, priority queues
    7. **Object Storage (S3):** Infinite scale, 11 nines durability

    **Key Design Decisions:**

    - **Separation of compute and storage:** Elastic scaling, pay-per-use
    - **Micro-partitioning:** Fine-grained pruning (16 MB partitions)
    - **Auto-clustering:** Automatic data layout optimization
    - **Materialized views:** 1000x speedup for aggregations
    - **Result caching:** 80% cache hit rate
    - **Cost optimizations:** Auto-suspend, query rewrite, partition pruning

    **Performance:**

    - **Query throughput:** 100,000 queries/day
    - **Concurrency:** 10,000 concurrent queries
    - **Scan performance:** 10 TB/sec with 1000 nodes
    - **Optimization:** 99% data pruned via partitioning + statistics

    This design provides enterprise-grade data warehouse capabilities with petabyte-scale storage, elastic compute, and cost-effective analytics at massive scale.
