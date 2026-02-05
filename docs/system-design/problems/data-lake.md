# Design Data Lake (Delta Lake / Apache Iceberg)

A distributed data lake platform that provides ACID transactions, schema evolution, time travel, and petabyte-scale storage for structured and semi-structured data on object storage (S3, ADLS, GCS).

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10 PB data, 10B files, 100K writes/sec, 50K queries/sec, millions of partitions |
| **Key Challenges** | ACID on object storage, schema evolution, query performance, metadata management, compaction |
| **Core Concepts** | Delta Lake/Iceberg, ACID transactions, time travel, Z-ordering, partition pruning, compaction |
| **Companies** | Databricks (Delta Lake), Netflix/Apple (Iceberg), Snowflake, AWS (S3), Dremio, Cloudera |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **ACID Transactions** | Support atomic, consistent, isolated, durable writes on object storage | P0 (Must have) |
    | **Schema Evolution** | Add/remove/rename columns without rewriting data | P0 (Must have) |
    | **Time Travel** | Query historical versions of data by timestamp or version | P0 (Must have) |
    | **Partition Management** | Automatic partitioning by date, region, etc. with partition pruning | P0 (Must have) |
    | **Metadata Catalog** | Centralized catalog for tables, schemas, partitions, statistics | P0 (Must have) |
    | **Concurrent Writes** | Multiple writers can write to same table without conflicts | P0 (Must have) |
    | **Data Compaction** | Merge small files, optimize file layout, remove deleted rows | P1 (Should have) |
    | **Data Quality** | Validate data on write, enforce constraints, reject bad records | P1 (Should have) |
    | **Query Optimization** | Data skipping, Z-ordering, statistics-based pruning | P1 (Should have) |
    | **Access Control** | Row/column-level security, audit logs, encryption | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Query engine implementation (assume Spark/Presto/Trino)
    - ETL orchestration (Airflow, Luigi)
    - Data discovery and lineage tracking
    - Machine learning model training
    - Real-time streaming analytics (focus on batch)
    - Cross-region replication

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Business-critical analytics workloads |
    | **Durability** | 99.999999999% (11 nines) | S3 durability guarantees, no data loss |
    | **Write Latency** | < 5s p95 for commits | Fast writes for batch jobs, ETL pipelines |
    | **Query Latency** | < 10s p95 for analytical queries | Interactive analytics, BI dashboards |
    | **Consistency** | Serializable isolation | ACID guarantees, no partial reads |
    | **Scalability** | 10 PB data, 10B files | Support massive datasets, thousands of partitions |
    | **Cost** | < $0.02/GB/month storage | Use S3 Standard-IA, optimize storage costs |
    | **Compaction** | < 10% storage overhead | Minimize small files, optimize read performance |

    ---

    ## Capacity Estimation

    ### Storage Estimates

    ```
    Data size:
    - Total data: 10 PB (10,000 TB)
    - Daily ingestion: 100 TB/day
    - Growth rate: 3% per month (36 TB/month)

    File counts:
    - Average file size: 1 GB (Parquet compressed)
    - Total files: 10 PB / 1 GB = 10 billion files
    - Daily new files: 100 TB / 1 GB = 100,000 files/day
    - After compaction: 50,000 files/day (2 GB avg)

    Partitions:
    - Partition scheme: year/month/day/region
    - Days: 365 × 2 years = 730 partitions (time)
    - Regions: 50 regions
    - Total partitions: 730 × 50 = 36,500 partitions
    - Files per partition: 10B / 36,500 = ~274,000 files/partition

    Metadata:
    - Transaction logs: 1 million commits × 10 KB = 10 GB
    - Partition metadata: 36,500 partitions × 1 KB = 36.5 MB
    - File metadata: 10B files × 100 bytes = 1 TB (stored in catalog)
    - Column statistics: 10B files × 10 columns × 50 bytes = 5 TB
    - Total metadata: ~6 TB (0.06% of data size)

    Storage cost (S3):
    - S3 Standard: 100 TB (hot data, last 30 days) × $0.023/GB = $2,300/month
    - S3 Standard-IA: 10 PB (warm data) × $0.0125/GB = $128,000/month
    - Total storage: ~$130K/month ≈ $1.56M/year
    ```

    ### Traffic Estimates

    ```
    Write traffic:
    - Batch ingestion: 100 TB/day = 1.16 GB/sec
    - Peak writes: 5x average = 5.8 GB/sec
    - Write operations: 100K files/day = 1.16 writes/sec
    - Peak write ops: 10 writes/sec

    Read traffic:
    - Query traffic: 50K queries/sec
    - Average query scans: 10 GB/query
    - Total read throughput: 50K × 10 GB = 500 TB/sec (peak)
    - Actual read (with partition pruning): 500 TB × 0.01 = 5 TB/sec
    - With caching (90% hit rate): 5 TB × 0.1 = 500 GB/sec = 4 Tbps

    S3 API calls:
    - LIST operations: 50K queries/sec × 10 partitions = 500K LIST/sec
    - GET operations: 50K queries × 100 files = 5M GET/sec
    - PUT operations: 100K files/day = 1.16 PUT/sec
    - Total API calls: ~5.5M/sec (read-heavy)

    Note: With metadata caching, actual S3 API calls reduced by 99%:
    - Cached LIST: 500K × 0.01 = 5K LIST/sec
    - Cached GET: 5M × 0.01 = 50K GET/sec
    ```

    ### Memory Estimates

    ```
    Metadata cache:
    - Partition list: 36,500 partitions × 1 KB = 36.5 MB
    - File list (recent): 1M files × 100 bytes = 100 MB
    - Column stats: 1M files × 10 columns × 50 bytes = 500 MB
    - Total metadata cache: ~650 MB per query engine

    Transaction log cache:
    - Recent commits: 10,000 commits × 10 KB = 100 MB
    - Snapshot cache: 1,000 snapshots × 50 KB = 50 MB

    Query result cache:
    - Hot queries: 10,000 queries × 10 MB = 100 GB
    - TTL: 1 hour

    Total memory: 100 GB (query cache) + 1 GB (metadata) ≈ 101 GB per cluster
    ```

    ---

    ## Key Assumptions

    1. Average file size after compaction: 1-2 GB (Parquet)
    2. Data retention: 2 years (730 days)
    3. Partition scheme: date-based (year/month/day) + region
    4. Query selectivity: 1% of data scanned (partition pruning)
    5. Cache hit rate: 90% (metadata) and 80% (query results)
    6. Compaction ratio: 10:1 (merge 10 small files into 1 large)
    7. Schema evolution: 5-10 columns added per month
    8. Time travel queries: 5% of total queries
    9. Concurrent writers: 100-1000 concurrent jobs
    10. S3 eventual consistency: < 1 second (S3 strong consistency now)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **ACID on object storage:** Delta Lake/Iceberg provide transaction logs for ACID guarantees
    2. **Metadata-driven:** Centralized catalog stores table/partition/file metadata
    3. **Optimistic concurrency:** Multiple writers with conflict resolution
    4. **Columnar storage:** Parquet format for efficient analytics (10x compression)
    5. **Lazy metadata loading:** Load partition metadata on-demand for scalability
    6. **Data skipping:** Min/max statistics for partition/file pruning
    7. **Time travel:** Snapshot isolation via immutable transaction logs

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Spark[Apache Spark<br/>ETL jobs]
            Presto[Presto/Trino<br/>Query engine]
            Flink[Apache Flink<br/>Streaming writes]
        end

        subgraph "API Layer"
            Write_API[Write API<br/>Delta/Iceberg writer]
            Read_API[Read API<br/>Delta/Iceberg reader]
            Catalog_API[Catalog API<br/>Metadata operations]
        end

        subgraph "Transaction Layer"
            TXN_Coordinator[Transaction Coordinator<br/>Optimistic concurrency]
            Commit_Service[Commit Service<br/>Atomic commits]
            Conflict_Resolver[Conflict Resolver<br/>Retry/abort]
        end

        subgraph "Metadata Layer"
            Metastore[Hive Metastore<br/>Table definitions]
            Catalog_DB[(Catalog DB<br/>PostgreSQL/Glue)]
            Version_Log[(Transaction Log<br/>S3: _delta_log/)]
        end

        subgraph "Caching Layer"
            Metadata_Cache[Metadata Cache<br/>Redis/Alluxio]
            File_Cache[File Cache<br/>Local SSD]
            Stats_Cache[Statistics Cache<br/>Min/max/count]
        end

        subgraph "Storage Layer (S3)"
            Data_Files[Data Files<br/>Parquet/ORC]
            Delta_Log[Delta Log<br/>JSON commits]
            Checkpoint[Checkpoints<br/>Snapshot metadata]
            Manifests[Manifest Files<br/>File lists]
        end

        subgraph "Optimization Layer"
            Compaction[Compaction Service<br/>Merge small files]
            Z_Order[Z-Ordering Service<br/>Data layout optimization]
            Vacuum[Vacuum Service<br/>Delete old files]
            Stats_Collector[Statistics Collector<br/>Column stats]
        end

        subgraph "Monitoring"
            Metrics[Metrics Service<br/>Prometheus]
            Audit_Log[(Audit Log<br/>CloudTrail)]
        end

        Spark --> Write_API
        Spark --> Read_API
        Presto --> Read_API
        Flink --> Write_API

        Write_API --> TXN_Coordinator
        Write_API --> Catalog_API
        Read_API --> Catalog_API
        Read_API --> Metadata_Cache

        TXN_Coordinator --> Commit_Service
        TXN_Coordinator --> Conflict_Resolver
        Commit_Service --> Version_Log
        Commit_Service --> Metastore

        Catalog_API --> Catalog_DB
        Catalog_API --> Metadata_Cache
        Catalog_API --> Version_Log

        Metadata_Cache --> Stats_Cache
        Read_API --> File_Cache

        Commit_Service --> Data_Files
        Commit_Service --> Delta_Log
        Commit_Service --> Manifests
        Commit_Service --> Checkpoint

        Read_API --> Data_Files
        Read_API --> Delta_Log
        Read_API --> Manifests

        Compaction --> Data_Files
        Z_Order --> Data_Files
        Vacuum --> Data_Files
        Stats_Collector --> Stats_Cache
        Stats_Collector --> Catalog_DB

        Write_API --> Metrics
        Read_API --> Metrics
        Commit_Service --> Audit_Log

        style Write_API fill:#e1f5ff
        style Read_API fill:#e1f5ff
        style TXN_Coordinator fill:#ffe1e1
        style Commit_Service fill:#ffe1e1
        style Version_Log fill:#fff4e1
        style Metadata_Cache fill:#e8f5e9
        style Data_Files fill:#f3e5f5
        style Compaction fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Delta Lake/Iceberg** | ACID transactions on object storage, time travel, schema evolution | Hive (no ACID), Hudi (less mature), custom solution (too complex) |
    | **S3** | Durable (11 nines), scalable (unlimited), cost-effective ($0.023/GB) | HDFS (expensive, complex ops), EBS (not scalable), Ceph (maintenance) |
    | **Parquet** | Columnar format (10x compression), efficient scans, predicate pushdown | ORC (less ecosystem support), Avro (row-based, slower), CSV (no compression) |
    | **Hive Metastore** | Industry standard, broad compatibility, centralized catalog | AWS Glue (vendor lock-in), custom DB (reinvent wheel), file-based (not scalable) |
    | **PostgreSQL** | ACID for metadata, complex queries, transactions | DynamoDB (no joins), MySQL (less ACID), MongoDB (not relational) |
    | **Redis/Alluxio** | Fast metadata lookup (<1ms), reduce S3 API calls by 99% | No cache (slow queries), Memcached (no persistence), local only (not shared) |

    **Key Trade-off:** We chose **Delta Lake** for ACID and time travel. Iceberg is also excellent (more mature open-source governance). Both provide similar features with minor differences in implementation.

    ---

    ## API Design

    ### 1. Create Table

    **Request:**
    ```scala
    // Delta Lake API (Scala/Python)
    spark.sql("""
      CREATE TABLE events (
        event_id STRING,
        user_id STRING,
        event_type STRING,
        timestamp TIMESTAMP,
        properties MAP<STRING, STRING>,
        country STRING,
        device STRING
      )
      USING delta
      PARTITIONED BY (date DATE, country STRING)
      LOCATION 's3://data-lake/events'
      TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'delta.deletedFileRetentionDuration' = 'interval 7 days',
        'delta.logRetentionDuration' = 'interval 30 days'
      )
    """)
    ```

    **Iceberg equivalent:**
    ```sql
    CREATE TABLE events (
      event_id STRING,
      user_id STRING,
      event_type STRING,
      timestamp TIMESTAMP,
      properties MAP<STRING, STRING>,
      country STRING,
      device STRING
    )
    USING iceberg
    PARTITIONED BY (days(timestamp), country)
    LOCATION 's3://data-lake/events'
    TBLPROPERTIES (
      'write.parquet.compression-codec' = 'snappy',
      'commit.retry.num-retries' = '10'
    )
    ```

    **What happens internally:**

    1. Create table entry in Hive Metastore (metadata)
    2. Initialize Delta Log at `s3://data-lake/events/_delta_log/00000000000000000000.json`
    3. Create empty transaction log (version 0)
    4. Set table properties (retention, compaction settings)

    ---

    ### 2. Write Data (INSERT/APPEND)

    **Request:**
    ```python
    # PySpark with Delta Lake
    from delta.tables import DeltaTable
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    # Read new data
    df = spark.read.parquet("s3://staging/events/2026-02-05/")

    # Write to Delta table (append)
    df.write \
        .format("delta") \
        .mode("append") \
        .partitionBy("date", "country") \
        .save("s3://data-lake/events")

    # Result: Atomic commit with transaction log
    ```

    **What happens internally:**

    1. **Stage files:** Write Parquet files to S3 (data files)
       - Files: `s3://data-lake/events/date=2026-02-05/country=US/part-00001.parquet`
       - Size: 1-2 GB each (configurable)

    2. **Collect metadata:** Generate file statistics
       - Min/max values for each column
       - Row count, null count, distinct count
       - File size, partition values

    3. **Optimistic transaction:** Attempt commit
       - Read latest version from Delta Log
       - Check for conflicts (same partitions modified?)
       - Generate commit JSON

    4. **Atomic commit:** Write transaction log entry
       - Create new version: `_delta_log/00000000000000000042.json`
       - Contains: add/remove file operations, statistics
       - Commit is atomic (either all files visible or none)

    5. **Update catalog:** Refresh partition metadata
       - Add new partitions to Hive Metastore
       - Update table statistics (row count, size)

    **Transaction log entry (JSON):**
    ```json
    {
      "commitInfo": {
        "timestamp": 1707134400000,
        "operation": "WRITE",
        "operationMetrics": {
          "numFiles": "10",
          "numOutputRows": "1000000",
          "numOutputBytes": "2147483648"
        }
      },
      "add": {
        "path": "date=2026-02-05/country=US/part-00001.parquet",
        "partitionValues": {"date": "2026-02-05", "country": "US"},
        "size": 214748364,
        "modificationTime": 1707134400000,
        "dataChange": true,
        "stats": "{\"numRecords\":100000,\"minValues\":{\"timestamp\":\"2026-02-05T00:00:00Z\"},\"maxValues\":{\"timestamp\":\"2026-02-05T23:59:59Z\"},\"nullCount\":{\"user_id\":0}}"
      }
    }
    ```

    ---

    ### 3. Read Data (Query)

    **Request:**
    ```python
    # Read from Delta table
    df = spark.read \
        .format("delta") \
        .load("s3://data-lake/events")

    # Query with filters (partition pruning)
    result = df.filter(
        (df.date >= "2026-02-01") &
        (df.date <= "2026-02-05") &
        (df.country == "US")
    ).select("event_id", "user_id", "event_type")

    result.show()
    ```

    **What happens internally:**

    1. **Read transaction log:** Get latest snapshot
       - Read all commits: `_delta_log/00000000000000000000.json` to `latest.json`
       - Or read checkpoint: `_delta_log/00000000000000000040.checkpoint.parquet` (faster)
       - Build file list: All files added (minus removed files)

    2. **Partition pruning:** Filter files by partition
       - Filter: `date >= 2026-02-01 AND date <= 2026-02-05 AND country = US`
       - Pruned files: 5 days × 1 country = 5 partitions
       - Reduction: 36,500 partitions → 5 partitions (99.99% pruned)

    3. **Data skipping:** Use column statistics
       - Check min/max values in stats
       - Skip files where `timestamp < 2026-02-01` (impossible range)
       - Reduction: 5,000 files → 500 files (90% pruned)

    4. **Read data files:** Parallel scan
       - Read 500 Parquet files in parallel
       - Use predicate pushdown (filter in Parquet reader)
       - Use column projection (read only needed columns)

    5. **Return results:** Combine results from all files
       - Merge data from 500 files
       - Apply remaining filters
       - Return result set

    **Performance:**
    - Without pruning: Scan 10 PB (all data)
    - With partition pruning: Scan 100 TB (1% of data)
    - With data skipping: Scan 10 TB (0.1% of data)
    - Query time: 10 seconds (vs 10,000 seconds without optimization)

    ---

    ### 4. Time Travel Query

    **Request:**
    ```python
    # Query data as of specific timestamp
    df_historical = spark.read \
        .format("delta") \
        .option("timestampAsOf", "2026-02-01 00:00:00") \
        .load("s3://data-lake/events")

    # Query data at specific version
    df_version = spark.read \
        .format("delta") \
        .option("versionAsOf", "42") \
        .load("s3://data-lake/events")

    # Get version history
    from delta.tables import DeltaTable
    deltaTable = DeltaTable.forPath(spark, "s3://data-lake/events")
    history = deltaTable.history()
    history.show()
    ```

    **Response (history):**
    ```
    +-------+-------------------+---------+----------+
    |version|timestamp          |operation|numFiles  |
    +-------+-------------------+---------+----------+
    |43     |2026-02-05 10:00:00|WRITE    |10        |
    |42     |2026-02-05 09:00:00|WRITE    |10        |
    |41     |2026-02-05 08:00:00|DELETE   |5         |
    |40     |2026-02-04 23:00:00|OPTIMIZE |50        |
    |39     |2026-02-04 22:00:00|WRITE    |10        |
    +-------+-------------------+---------+----------+
    ```

    **What happens internally:**

    1. **Find snapshot:** Binary search in transaction log
       - Search for version or timestamp in Delta Log
       - Read commits up to that version

    2. **Build file list:** Reconstruct state at that point
       - Apply all add/remove operations up to version 42
       - Ignore files added after version 42

    3. **Read historical data:** Query using historical file list
       - Read only files that existed at version 42
       - Same partition pruning and data skipping apply

    **Use cases:**
    - Debug: "What did the data look like when the bug happened?"
    - Audit: "Show me all transactions from last week"
    - Rollback: "Restore table to yesterday's version"
    - Reproducibility: "Re-run ML model on same training data"

    ---

    ### 5. Schema Evolution

    **Request:**
    ```python
    # Add new column (no data rewrite!)
    spark.sql("""
      ALTER TABLE events
      ADD COLUMNS (
        session_id STRING COMMENT 'User session identifier',
        platform_version STRING COMMENT 'App version'
      )
    """)

    # Rename column
    spark.sql("""
      ALTER TABLE events
      RENAME COLUMN properties TO event_properties
    """)

    # Change column type (requires rewrite)
    spark.sql("""
      ALTER TABLE events
      ALTER COLUMN event_id TYPE BIGINT
    """)
    ```

    **What happens internally:**

    1. **Add column (metadata-only):**
       - Update schema in Delta Log
       - New writes include new column
       - Old files: Assume NULL for new column (no rewrite!)
       - Cost: < 1 second, no data movement

    2. **Rename column (metadata-only):**
       - Update schema mapping in Delta Log
       - Use Parquet field IDs for compatibility
       - No data rewrite needed

    3. **Change type (requires rewrite):**
       - Read all data files
       - Cast column to new type
       - Rewrite all files with new schema
       - Cost: Hours for large tables

    **Schema evolution in Delta Log:**
    ```json
    {
      "metaData": {
        "schemaString": "{\"type\":\"struct\",\"fields\":[{\"name\":\"event_id\",\"type\":\"string\"},{\"name\":\"session_id\",\"type\":\"string\"}]}",
        "partitionColumns": ["date", "country"],
        "configuration": {},
        "createdTime": 1707134400000
      }
    }
    ```

    ---

    ### 6. MERGE (Upsert)

    **Request:**
    ```python
    from delta.tables import DeltaTable

    # Load Delta table
    deltaTable = DeltaTable.forPath(spark, "s3://data-lake/events")

    # Load updates
    updates = spark.read.parquet("s3://staging/event-updates/")

    # Perform MERGE (upsert)
    deltaTable.alias("target").merge(
        updates.alias("source"),
        "target.event_id = source.event_id"
    ).whenMatchedUpdate(set = {
        "event_type": "source.event_type",
        "properties": "source.properties"
    }).whenNotMatchedInsert(values = {
        "event_id": "source.event_id",
        "user_id": "source.user_id",
        "event_type": "source.event_type",
        "timestamp": "source.timestamp",
        "properties": "source.properties"
    }).execute()
    ```

    **What happens internally:**

    1. **Read matching files:** Find files with matching keys
       - Use partition pruning to narrow down files
       - Read files into memory

    2. **Perform merge:** Join target and source
       - Inner join to find matches (updates)
       - Left anti join to find non-matches (inserts)

    3. **Write new files:** Replace old files
       - Write updated records to new Parquet files
       - Mark old files as removed in Delta Log

    4. **Atomic commit:** Commit transaction
       - Add: New files with merged data
       - Remove: Old files that were merged
       - All-or-nothing commit

    **Performance:**
    - Small updates: Only rewrite affected files (not entire table)
    - Cost: 10x faster than full table rewrite
    - Example: Update 1M rows in 10B row table = rewrite 0.01% of data

    ---

    ### 7. OPTIMIZE (Compaction)

    **Request:**
    ```python
    # Compact small files into larger files
    spark.sql("""
      OPTIMIZE events
      WHERE date >= '2026-02-01'
    """)

    # Z-ordering for better data skipping
    spark.sql("""
      OPTIMIZE events
      ZORDER BY (user_id, event_type)
    """)
    ```

    **What happens internally (compaction):**

    1. **Identify small files:** Find files < 100 MB
       - List all files in Delta Log
       - Group by partition

    2. **Read and merge:** Combine small files
       - Read 10 small files (100 MB each)
       - Merge into 1 large file (1 GB)

    3. **Write compacted file:** Replace with larger file
       - Write new 1 GB Parquet file
       - Add: New large file
       - Remove: 10 small files

    4. **Commit transaction:** Atomic swap
       - Old files marked as removed (but not deleted yet)
       - VACUUM command deletes old files after retention period

    **What happens internally (Z-ordering):**

    1. **Read data:** Load partition data into memory
       - Read all files in partition

    2. **Sort by Z-order:** Multi-dimensional clustering
       - Interleave bits from multiple columns
       - Z-order curve maps multi-dimensional space to 1D

    3. **Write sorted files:** Replace with Z-ordered files
       - Data co-located by multiple dimensions
       - Better data skipping for multi-column filters

    **Performance improvement:**
    - Before compaction: 10,000 small files, 50s query time
    - After compaction: 100 large files, 5s query time (10x faster)
    - Z-ordering: 5s → 2s (2.5x faster for multi-column filters)

    ---

    ## Data Model

    ### Transaction Log Schema (Delta Lake)

    **Location:** `s3://data-lake/events/_delta_log/`

    **Files:**
    ```
    00000000000000000000.json  (initial commit)
    00000000000000000001.json  (commit 1)
    00000000000000000002.json  (commit 2)
    ...
    00000000000000000042.json  (commit 42)
    00000000000000000040.checkpoint.parquet  (checkpoint at version 40)
    ```

    **Commit JSON structure:**
    ```json
    {
      "commitInfo": {
        "timestamp": 1707134400000,
        "userId": "user123",
        "userName": "data-engineer",
        "operation": "WRITE",
        "operationParameters": {
          "mode": "Append",
          "partitionBy": "[\"date\",\"country\"]"
        },
        "operationMetrics": {
          "numFiles": "10",
          "numOutputRows": "1000000",
          "numOutputBytes": "2147483648"
        },
        "engineInfo": "Apache-Spark/3.5.0 Delta-Lake/3.0.0",
        "clusterId": "cluster-123"
      },
      "protocol": {
        "minReaderVersion": 1,
        "minWriterVersion": 2
      },
      "metaData": {
        "id": "table-uuid-12345",
        "name": "events",
        "description": "User events table",
        "format": {
          "provider": "parquet"
        },
        "schemaString": "{\"type\":\"struct\",\"fields\":[{\"name\":\"event_id\",\"type\":\"string\",\"nullable\":false,\"metadata\":{}},{\"name\":\"user_id\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"event_type\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"timestamp\",\"type\":\"timestamp\",\"nullable\":true,\"metadata\":{}},{\"name\":\"properties\",\"type\":{\"type\":\"map\",\"keyType\":\"string\",\"valueType\":\"string\"},\"nullable\":true,\"metadata\":{}},{\"name\":\"date\",\"type\":\"date\",\"nullable\":true,\"metadata\":{}},{\"name\":\"country\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}",
        "partitionColumns": ["date", "country"],
        "configuration": {
          "delta.autoOptimize.optimizeWrite": "true",
          "delta.deletedFileRetentionDuration": "interval 7 days"
        },
        "createdTime": 1707134400000
      },
      "add": {
        "path": "date=2026-02-05/country=US/part-00001-uuid123.snappy.parquet",
        "partitionValues": {
          "date": "2026-02-05",
          "country": "US"
        },
        "size": 214748364,
        "modificationTime": 1707134400000,
        "dataChange": true,
        "stats": "{\"numRecords\":100000,\"minValues\":{\"event_id\":\"evt_000001\",\"timestamp\":\"2026-02-05T00:00:00.000Z\"},\"maxValues\":{\"event_id\":\"evt_100000\",\"timestamp\":\"2026-02-05T23:59:59.999Z\"},\"nullCount\":{\"user_id\":0,\"event_type\":0}}"
      },
      "remove": {
        "path": "date=2026-02-04/country=US/part-00099-uuid456.snappy.parquet",
        "deletionTimestamp": 1707134400000,
        "dataChange": true
      }
    }
    ```

    **Checkpoint Parquet schema:**
    - Materializes transaction log state at specific version
    - Contains all active files at that version
    - Speeds up initialization (don't replay all commits)
    - Created every 10 commits (configurable)

    ---

    ### Manifest Files (Iceberg)

    **Location:** `s3://data-lake/events/metadata/`

    **Files:**
    ```
    v1.metadata.json  (table metadata)
    snap-123456789-1-abc.avro  (snapshot manifest list)
    abc123-m0.avro  (manifest file: list of data files)
    ```

    **Metadata JSON:**
    ```json
    {
      "format-version": 2,
      "table-uuid": "uuid-12345",
      "location": "s3://data-lake/events",
      "last-sequence-number": 42,
      "last-updated-ms": 1707134400000,
      "last-column-id": 10,
      "schema": {
        "type": "struct",
        "schema-id": 0,
        "fields": [
          {"id": 1, "name": "event_id", "required": true, "type": "string"},
          {"id": 2, "name": "user_id", "required": false, "type": "string"},
          {"id": 3, "name": "timestamp", "required": false, "type": "timestamp"}
        ]
      },
      "current-snapshot-id": 123456789,
      "snapshots": [
        {
          "snapshot-id": 123456789,
          "timestamp-ms": 1707134400000,
          "summary": {
            "operation": "append",
            "added-files-size": "2147483648",
            "total-records": "1000000"
          },
          "manifest-list": "s3://data-lake/events/metadata/snap-123456789-1-abc.avro"
        }
      ],
      "partition-spec": [
        {"name": "date", "transform": "day", "source-id": 3, "field-id": 1000},
        {"name": "country", "transform": "identity", "source-id": 7, "field-id": 1001}
      ]
    }
    ```

    **Manifest file (Avro):**
    ```
    File path: date=2026-02-05/country=US/data-001.parquet
    Partition: {date: 2026-02-05, country: US}
    File size: 214748364 bytes
    Record count: 100000
    Column stats:
      event_id: min=evt_000001, max=evt_100000, null_count=0
      timestamp: min=2026-02-05T00:00:00Z, max=2026-02-05T23:59:59Z
    ```

=== "🔧 Step 3: Deep Dive"

    ## 3.1 ACID Transactions on Object Storage

    **Challenge:** S3 is not ACID-compliant. How do we achieve ACID?

    **Solution:** Transaction log with optimistic concurrency control

    ---

    ### ACID Implementation (Delta Lake)

    **Key insight:** Object storage provides two critical guarantees:
    1. **Put-if-absent:** Can atomically create new file if it doesn't exist
    2. **List consistency:** List operations eventually consistent (S3 now strongly consistent)

    **Transaction protocol:**

    ```python
    class DeltaTransaction:
        """
        Delta Lake transaction implementation
        """

        def __init__(self, table_path, isolation_level="Serializable"):
            self.table_path = table_path
            self.delta_log_path = f"{table_path}/_delta_log"
            self.isolation_level = isolation_level
            self.read_version = None
            self.write_version = None

        def begin_transaction(self):
            """
            Start transaction by reading latest version
            """
            # Read latest version from Delta Log
            self.read_version = self._get_latest_version()
            print(f"Transaction started at version {self.read_version}")

            # Build snapshot (list of all files)
            self.snapshot = self._build_snapshot(self.read_version)

        def _get_latest_version(self) -> int:
            """
            Get latest version number from Delta Log

            LIST s3://table/_delta_log/ and find highest version number
            """
            log_files = s3.list_objects(self.delta_log_path)

            # Find latest commit file: 00000000000000000042.json
            versions = [
                int(f.split(".json")[0])
                for f in log_files
                if f.endswith(".json")
            ]

            return max(versions) if versions else -1

        def _build_snapshot(self, version: int) -> List[str]:
            """
            Build snapshot by replaying transaction log

            Two approaches:
            1. Replay all commits from 0 to version (slow)
            2. Load checkpoint + replay commits since checkpoint (fast)
            """
            # Find latest checkpoint before version
            checkpoint_version = (version // 10) * 10  # Every 10 commits
            checkpoint_file = f"{self.delta_log_path}/{checkpoint_version:020d}.checkpoint.parquet"

            if s3.exists(checkpoint_file):
                # Fast path: Load checkpoint (Parquet)
                files = read_parquet(checkpoint_file)["add"]["path"].tolist()
                start_version = checkpoint_version + 1
            else:
                # Slow path: Start from beginning
                files = []
                start_version = 0

            # Replay commits since checkpoint
            for v in range(start_version, version + 1):
                commit_file = f"{self.delta_log_path}/{v:020d}.json"
                commit = read_json(commit_file)

                # Apply add/remove operations
                for action in commit:
                    if "add" in action:
                        files.append(action["add"]["path"])
                    elif "remove" in action:
                        files.remove(action["remove"]["path"])

            print(f"Snapshot contains {len(files)} files")
            return files

        def write_data(self, data: DataFrame):
            """
            Write data to table

            1. Write Parquet files to S3 (data files)
            2. Collect file metadata (stats)
            3. Prepare commit (JSON)
            """
            # Write Parquet files
            new_files = []
            for partition, df in data.groupBy("date", "country"):
                file_path = f"date={partition.date}/country={partition.country}/part-{uuid4()}.parquet"
                full_path = f"{self.table_path}/{file_path}"

                # Write to S3
                write_parquet(df, full_path)

                # Collect metadata
                stats = {
                    "numRecords": len(df),
                    "minValues": df.min().to_dict(),
                    "maxValues": df.max().to_dict(),
                    "nullCount": df.isnull().sum().to_dict()
                }

                new_files.append({
                    "path": file_path,
                    "size": s3.get_object_size(full_path),
                    "partitionValues": {"date": partition.date, "country": partition.country},
                    "stats": json.dumps(stats)
                })

            self.new_files = new_files
            print(f"Wrote {len(new_files)} new files")

        def commit(self):
            """
            Attempt to commit transaction

            Optimistic concurrency control:
            1. Check for conflicts (other commits since read_version)
            2. If no conflicts, write commit file
            3. If conflicts, retry or abort
            """
            # Try to commit at next version
            self.write_version = self.read_version + 1
            commit_file = f"{self.delta_log_path}/{self.write_version:020d}.json"

            # Check for conflicts
            current_version = self._get_latest_version()
            if current_version > self.read_version:
                # Another commit happened! Check for conflicts
                if self._has_conflict(self.read_version + 1, current_version):
                    print(f"Conflict detected! Retrying...")
                    return self._retry_commit()
                else:
                    # No conflict, update write_version
                    self.write_version = current_version + 1
                    commit_file = f"{self.delta_log_path}/{self.write_version:020d}.json"

            # Prepare commit JSON
            commit_json = {
                "commitInfo": {
                    "timestamp": int(time.time() * 1000),
                    "operation": "WRITE",
                    "operationMetrics": {
                        "numFiles": str(len(self.new_files)),
                        "numOutputRows": str(sum(json.loads(f["stats"])["numRecords"] for f in self.new_files))
                    }
                }
            }

            # Add file operations
            for file in self.new_files:
                commit_json["add"] = file

            # Atomic commit: Write commit file (put-if-absent)
            try:
                s3.put_object_if_not_exists(commit_file, json.dumps(commit_json))
                print(f"Committed at version {self.write_version}")
                return True
            except FileExistsError:
                # Another writer committed at same version! Retry
                print(f"Commit failed (race condition). Retrying...")
                return self._retry_commit()

        def _has_conflict(self, start_version: int, end_version: int) -> bool:
            """
            Check if commits between start and end conflict with our write

            Conflict: Another writer modified same partitions
            """
            our_partitions = set(
                (f["partitionValues"]["date"], f["partitionValues"]["country"])
                for f in self.new_files
            )

            # Check commits since our read
            for v in range(start_version, end_version + 1):
                commit_file = f"{self.delta_log_path}/{v:020d}.json"
                commit = read_json(commit_file)

                # Check if any file touches our partitions
                for action in commit:
                    if "add" in action or "remove" in action:
                        file_partition = (
                            action.get("add", action.get("remove"))["partitionValues"]["date"],
                            action.get("add", action.get("remove"))["partitionValues"]["country"]
                        )
                        if file_partition in our_partitions:
                            # Conflict! Same partition modified
                            return True

            return False

        def _retry_commit(self):
            """
            Retry commit with exponential backoff
            """
            max_retries = 10
            for attempt in range(max_retries):
                time.sleep(2 ** attempt)  # Exponential backoff

                # Re-read latest version and retry
                self.read_version = self._get_latest_version()
                return self.commit()

            # Max retries exceeded
            raise Exception("Failed to commit after 10 retries")

    # Usage example
    txn = DeltaTransaction("s3://data-lake/events")
    txn.begin_transaction()
    txn.write_data(df)
    txn.commit()
    ```

    **ACID guarantees:**

    1. **Atomicity:** All-or-nothing
       - Either commit file is written (all files visible) or not (no files visible)
       - No partial writes

    2. **Consistency:** Schema validation
       - Validate schema on write (reject incompatible data)
       - Enforce constraints (NOT NULL, CHECK)

    3. **Isolation:** Serializable (snapshot isolation)
       - Readers see consistent snapshot at specific version
       - Writers use optimistic concurrency (detect conflicts)

    4. **Durability:** S3 durability (11 nines)
       - Once commit file written, data is durable
       - S3 replicates across multiple AZs

    ---

    ### Optimistic Concurrency Control

    **Scenario 1: No conflicts (happy path)**

    ```
    Time | Writer A                      | Writer B                      | Latest Version
    -----|-------------------------------|-------------------------------|---------------
    t0   | Read version 42               |                               | 42
    t1   |                               | Read version 42               | 42
    t2   | Write files (partition P1)    |                               | 42
    t3   |                               | Write files (partition P2)    | 42
    t4   | Commit version 43 (P1) ✓      |                               | 43
    t5   |                               | Commit version 44 (P2) ✓      | 44

    Result: Both commits succeed (no conflict, different partitions)
    ```

    **Scenario 2: Conflict detected**

    ```
    Time | Writer A                      | Writer B                      | Latest Version
    -----|-------------------------------|-------------------------------|---------------
    t0   | Read version 42               |                               | 42
    t1   |                               | Read version 42               | 42
    t2   | Write files (partition P1)    |                               | 42
    t3   |                               | Write files (partition P1)    | 42
    t4   | Commit version 43 (P1) ✓      |                               | 43
    t5   |                               | Try commit version 43 (P1) ✗  | 43
    t6   |                               | Detect conflict, retry        | 43
    t7   |                               | Re-read version 43            | 43
    t8   |                               | Commit version 44 (P1) ✓      | 44

    Result: Writer B retries and commits at version 44
    ```

    **Conflict resolution strategies:**

    1. **Retry (default):** Retry commit with exponential backoff
       - Works for most cases
       - Max retries: 10 (configurable)

    2. **Abort:** Fail fast, let caller handle
       - Use for interactive queries
       - User can manually retry

    3. **Merge:** Combine changes intelligently
       - Complex (requires application logic)
       - Example: Append both writes (no conflict)

    ---

    ## 3.2 Partition Pruning & Data Skipping

    **Goal:** Skip reading 99.99% of data by using metadata

    ---

    ### Partition Pruning

    **Concept:** Use partition columns to filter out entire partitions

    **Example:**

    ```python
    # Table partitioned by date and country
    # Total partitions: 730 days × 50 countries = 36,500 partitions
    # Files per partition: 1000 files
    # Total files: 36.5 million files

    # Query with partition filter
    df = spark.read.format("delta").load("s3://data-lake/events")
    result = df.filter(
        (df.date >= "2026-02-01") &
        (df.date <= "2026-02-05") &
        (df.country == "US")
    )

    # Partition pruning:
    # - Date range: 5 days (out of 730) = 0.68% of dates
    # - Country: 1 country (out of 50) = 2% of countries
    # - Combined: 5 × 1 = 5 partitions (out of 36,500) = 0.014%
    # - Files to scan: 5 × 1000 = 5,000 files (out of 36.5M) = 0.014%
    # - Data to scan: 5,000 × 1 GB = 5 TB (out of 10 PB) = 0.05%
    ```

    **Implementation:**

    ```python
    def prune_partitions(partitions: List[Partition], filters: List[Filter]) -> List[Partition]:
        """
        Prune partitions based on filters
        """
        pruned = []

        for partition in partitions:
            # Check if partition matches filters
            if matches_filters(partition.values, filters):
                pruned.append(partition)

        print(f"Pruned {len(partitions)} → {len(pruned)} partitions " +
              f"({len(pruned) / len(partitions) * 100:.2f}%)")

        return pruned

    def matches_filters(partition_values: Dict, filters: List[Filter]) -> bool:
        """
        Check if partition values match filters
        """
        for filter in filters:
            column = filter.column
            operator = filter.operator
            value = filter.value

            partition_value = partition_values.get(column)

            if operator == "=":
                if partition_value != value:
                    return False
            elif operator == ">=":
                if partition_value < value:
                    return False
            elif operator == "<=":
                if partition_value > value:
                    return False
            elif operator == "IN":
                if partition_value not in value:
                    return False

        return True
    ```

    ---

    ### Data Skipping (Min/Max Statistics)

    **Concept:** Use column statistics to skip files without reading them

    **Statistics stored per file:**

    ```json
    {
      "numRecords": 100000,
      "minValues": {
        "event_id": "evt_000001",
        "timestamp": "2026-02-05T00:00:00Z",
        "user_id": "user_000001"
      },
      "maxValues": {
        "event_id": "evt_100000",
        "timestamp": "2026-02-05T23:59:59Z",
        "user_id": "user_999999"
      },
      "nullCount": {
        "event_id": 0,
        "user_id": 100,
        "event_type": 0
      }
    }
    ```

    **Example:**

    ```python
    # Query with filter
    result = df.filter(df.timestamp >= "2026-02-05T12:00:00Z")

    # Check each file's min/max:
    # File 1: min=2026-02-05T00:00:00Z, max=2026-02-05T06:00:00Z
    #   → Skip (max < filter value, impossible to match)
    #
    # File 2: min=2026-02-05T10:00:00Z, max=2026-02-05T14:00:00Z
    #   → Read (overlaps with filter range)
    #
    # File 3: min=2026-02-05T13:00:00Z, max=2026-02-05T23:59:59Z
    #   → Read (overlaps with filter range)

    # Result: Skip 50% of files based on statistics
    ```

    **Implementation:**

    ```python
    def skip_files_by_stats(files: List[File], filters: List[Filter]) -> List[File]:
        """
        Skip files that cannot match filters based on min/max stats
        """
        files_to_read = []

        for file in files:
            stats = json.loads(file.stats)

            # Check if file can match filters
            if can_match_filters(stats, filters):
                files_to_read.append(file)

        print(f"Data skipping: {len(files)} → {len(files_to_read)} files " +
              f"({len(files_to_read) / len(files) * 100:.2f}%)")

        return files_to_read

    def can_match_filters(stats: Dict, filters: List[Filter]) -> bool:
        """
        Check if file can possibly match filters
        """
        for filter in filters:
            column = filter.column
            operator = filter.operator
            value = filter.value

            min_val = stats["minValues"].get(column)
            max_val = stats["maxValues"].get(column)

            # If column not in stats, conservatively include file
            if min_val is None or max_val is None:
                continue

            # Check if range overlaps with filter
            if operator == "=":
                if value < min_val or value > max_val:
                    return False  # Value outside range, skip file
            elif operator == ">=":
                if max_val < value:
                    return False  # All values too small, skip file
            elif operator == "<=":
                if min_val > value:
                    return False  # All values too large, skip file
            elif operator == ">":
                if max_val <= value:
                    return False
            elif operator == "<":
                if min_val >= value:
                    return False

        return True  # File might match, need to read it
    ```

    **Combined pruning example:**

    ```
    Query: SELECT * FROM events
           WHERE date = '2026-02-05'
             AND country = 'US'
             AND timestamp >= '2026-02-05T12:00:00Z'

    Step 1: Partition pruning
    - Total partitions: 36,500
    - After pruning: 1 partition (date=2026-02-05, country=US)
    - Reduction: 99.997%

    Step 2: Data skipping
    - Files in partition: 1,000
    - After data skipping: 500 files (timestamp range check)
    - Reduction: 50%

    Step 3: Read data
    - Files to read: 500 files
    - Data to scan: 500 GB (out of 10 PB)
    - Overall reduction: 99.995%
    - Query time: 10 seconds (vs 10,000 seconds without optimization)
    ```

    ---

    ### Z-Ordering (Multi-dimensional Clustering)

    **Problem:** Min/max stats only work well for one dimension

    **Example:**

    ```python
    # Query with multiple filters
    result = df.filter(
        (df.user_id >= "user_100000") &
        (df.user_id <= "user_200000") &
        (df.event_type == "purchase")
    )

    # Without Z-ordering:
    # - File 1: user_id=[user_000001, user_999999], event_type=[click, purchase]
    #   → Cannot skip (ranges too wide)
    # - All files have wide ranges for both columns
    # - Result: Read all files (no data skipping)

    # With Z-ordering by (user_id, event_type):
    # - File 1: user_id=[user_000001, user_050000], event_type=[click, click]
    #   → Skip (event_type doesn't match)
    # - File 2: user_id=[user_100000, user_150000], event_type=[purchase, purchase]
    #   → Read (matches both filters)
    # - Result: Skip 80% of files (excellent data skipping)
    ```

    **Z-order curve:**

    ```
    Z-order interleaves bits from multiple dimensions:

    user_id (binary):    001010  100011
    event_type (hash):   110001  011100

    Z-order (interleave): 011001011010  101000011110

    Sort by Z-order → Data co-located by both dimensions!
    ```

    **Implementation:**

    ```python
    def z_order_data(df: DataFrame, z_order_columns: List[str]) -> DataFrame:
        """
        Sort data by Z-order curve
        """
        # Generate Z-order key
        df = df.withColumn(
            "z_order_key",
            z_order_key_udf(*[df[col] for col in z_order_columns])
        )

        # Sort by Z-order key
        df = df.sortBy("z_order_key")

        # Drop Z-order key
        df = df.drop("z_order_key")

        return df

    def z_order_key_udf(*values):
        """
        Generate Z-order key by interleaving bits
        """
        # Hash each value to int
        hashes = [hash(v) % (2**32) for v in values]

        # Interleave bits
        z_key = 0
        for bit_pos in range(32):
            for i, h in enumerate(hashes):
                bit = (h >> bit_pos) & 1
                z_key |= (bit << (bit_pos * len(hashes) + i))

        return z_key
    ```

    **Performance:**

    ```
    Query: SELECT * FROM events
           WHERE user_id BETWEEN 'user_100000' AND 'user_200000'
             AND event_type = 'purchase'

    Without Z-ordering:
    - Files to read: 1,000 files (cannot skip based on stats)
    - Query time: 50 seconds

    With Z-ordering by (user_id, event_type):
    - Files to read: 200 files (80% skipped)
    - Query time: 10 seconds (5x faster)

    Trade-off:
    - Cost: Z-ordering requires sorting data (one-time cost)
    - Benefit: 5-10x faster queries on Z-ordered columns
    - Recommendation: Z-order by most common filter columns
    ```

    ---

    ## 3.3 Compaction & File Management

    **Problem:** Small files cause performance issues

    **Why small files are bad:**

    1. **S3 overhead:** LIST/GET operations have fixed cost
       - 1 file (1 GB): 1 LIST + 1 GET = 2 ops
       - 1000 files (1 MB each): 1 LIST + 1000 GET = 1001 ops (500x overhead!)

    2. **Query overhead:** More files = more tasks
       - Spark task per file (1000 files = 1000 tasks)
       - Task startup overhead dominates query time

    3. **Metadata overhead:** More files = larger transaction log
       - 1M files: 100 MB transaction log (slow to read)
       - 10K files: 1 MB transaction log (fast to read)

    ---

    ### Compaction Algorithm

    **Goal:** Merge small files into larger files (target: 1-2 GB)

    **Implementation:**

    ```python
    class CompactionService:
        """
        Compact small files into larger files
        """

        def __init__(self, table_path: str, target_file_size: int = 1_000_000_000):
            self.table_path = table_path
            self.target_file_size = target_file_size  # 1 GB
            self.small_file_threshold = target_file_size * 0.1  # 100 MB

        def compact_partition(self, partition: str):
            """
            Compact all small files in a partition
            """
            # Get all files in partition
            files = self._list_files_in_partition(partition)

            # Identify small files
            small_files = [
                f for f in files
                if f.size < self.small_file_threshold
            ]

            if not small_files:
                print(f"No small files in partition {partition}")
                return

            print(f"Compacting {len(small_files)} small files " +
                  f"(total size: {sum(f.size for f in small_files) / 1e9:.2f} GB)")

            # Group small files into bins (target: 1 GB per bin)
            bins = self._bin_packing(small_files, self.target_file_size)

            # Compact each bin
            for i, bin_files in enumerate(bins):
                self._compact_files(bin_files, f"{partition}/part-{i:05d}-compacted.parquet")

            print(f"Compaction complete: {len(small_files)} → {len(bins)} files")

        def _bin_packing(self, files: List[File], bin_size: int) -> List[List[File]]:
            """
            First-fit bin packing algorithm

            Pack small files into bins of target size
            """
            bins = []
            current_bin = []
            current_size = 0

            for file in sorted(files, key=lambda f: f.size, reverse=True):
                if current_size + file.size <= bin_size:
                    # Add to current bin
                    current_bin.append(file)
                    current_size += file.size
                else:
                    # Start new bin
                    bins.append(current_bin)
                    current_bin = [file]
                    current_size = file.size

            # Add last bin
            if current_bin:
                bins.append(current_bin)

            return bins

        def _compact_files(self, files: List[File], output_path: str):
            """
            Read multiple files and write to single output file
            """
            # Read all files
            dfs = [spark.read.parquet(f.path) for f in files]

            # Union all dataframes
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = merged_df.union(df)

            # Write to single file
            merged_df.coalesce(1).write.parquet(output_path)

            # Collect stats
            stats = self._collect_stats(merged_df)

            # Commit transaction (add new file, remove old files)
            self._commit_compaction(output_path, files, stats)

        def _commit_compaction(self, new_file: str, old_files: List[File], stats: Dict):
            """
            Atomic commit: Replace old files with new file
            """
            txn = DeltaTransaction(self.table_path)
            txn.begin_transaction()

            # Add new file
            txn.add_file(new_file, stats)

            # Remove old files
            for file in old_files:
                txn.remove_file(file.path)

            txn.commit()
            print(f"Compaction committed: {len(old_files)} old files → 1 new file")

    # Usage
    compaction = CompactionService("s3://data-lake/events")
    compaction.compact_partition("date=2026-02-05/country=US")
    ```

    **Compaction strategies:**

    1. **Scheduled compaction:** Run daily/weekly
       - Compact old partitions (data not changing)
       - Example: Compact yesterday's partitions every night

    2. **Auto-compaction:** Compact on write
       - Delta Lake: `delta.autoOptimize.autoCompact = true`
       - Automatically compact after writes exceed threshold

    3. **Manual compaction:** Run on-demand
       - Use for specific partitions with small file problem
       - Example: `OPTIMIZE events WHERE date = '2026-02-05'`

    ---

    ### VACUUM (Garbage Collection)

    **Problem:** Compaction creates new files but doesn't delete old files

    **Why?** Time travel needs old files!

    **Solution:** VACUUM command deletes old files after retention period

    **Implementation:**

    ```python
    class VacuumService:
        """
        Delete old files after retention period
        """

        def __init__(self, table_path: str, retention_days: int = 7):
            self.table_path = table_path
            self.retention_days = retention_days
            self.retention_seconds = retention_days * 86400

        def vacuum(self):
            """
            Delete files not needed for time travel
            """
            # Get current time
            current_time = time.time()
            cutoff_time = current_time - self.retention_seconds

            # Find all files in Delta Log
            all_files = self._get_all_files_in_history()

            # Find active files (in latest snapshot)
            active_files = self._get_active_files()

            # Find deleted files (in history but not in latest snapshot)
            deleted_files = set(all_files) - set(active_files)

            # Filter by retention period
            files_to_delete = [
                f for f in deleted_files
                if f.deletion_time < cutoff_time
            ]

            print(f"VACUUM: Deleting {len(files_to_delete)} files " +
                  f"(older than {self.retention_days} days)")

            # Delete files from S3
            for file in files_to_delete:
                s3.delete_object(f"{self.table_path}/{file.path}")

            print(f"VACUUM complete: Reclaimed {sum(f.size for f in files_to_delete) / 1e9:.2f} GB")

        def _get_all_files_in_history(self) -> List[File]:
            """
            Get all files ever added (from transaction log)
            """
            files = {}

            # Read all commits
            for version in range(self._get_latest_version() + 1):
                commit = self._read_commit(version)

                for action in commit:
                    if "add" in action:
                        file = File(
                            path=action["add"]["path"],
                            addition_time=action["add"]["modificationTime"]
                        )
                        files[file.path] = file
                    elif "remove" in action:
                        if action["remove"]["path"] in files:
                            files[action["remove"]["path"]].deletion_time = action["remove"]["deletionTimestamp"]

            return list(files.values())

        def _get_active_files(self) -> List[File]:
            """
            Get files in latest snapshot
            """
            txn = DeltaTransaction(self.table_path)
            txn.begin_transaction()
            return txn.snapshot

    # Usage
    vacuum = VacuumService("s3://data-lake/events", retention_days=7)
    vacuum.vacuum()
    ```

    **VACUUM considerations:**

    1. **Retention period:** Trade-off between storage cost and time travel
       - Short (7 days): Lower cost, limited time travel
       - Long (30 days): Higher cost, more time travel

    2. **Running jobs:** Don't VACUUM while queries are running
       - Could delete files being read (race condition)
       - Use locking or coordination

    3. **Audit trail:** Keep deletion logs for compliance
       - Track what was deleted and when
       - Recover files from S3 versioning if needed

    ---

    ## 3.4 Schema Evolution

    **Challenge:** Tables evolve over time (add columns, rename, change types)

    **Traditional approach (Hive):** Rewrite entire table (expensive!)

    **Delta Lake/Iceberg:** Metadata-only schema changes (instant!)

    ---

    ### Schema Evolution Examples

    **1. Add column (metadata-only):**

    ```python
    # Add column without rewriting data
    spark.sql("""
      ALTER TABLE events
      ADD COLUMNS (session_id STRING, app_version STRING)
    """)

    # What happens:
    # 1. Update schema in Delta Log (add columns to schema)
    # 2. New writes include new columns
    # 3. Old files: Assume NULL for new columns (no rewrite!)

    # Cost: < 1 second (metadata update only)
    # Old table: 10 PB, 10B files
    # New table: 10 PB, 10B files (same files!)
    ```

    **Transaction log update:**

    ```json
    {
      "metaData": {
        "schemaString": "{\"fields\":[..., {\"name\":\"session_id\",\"type\":\"string\"}, {\"name\":\"app_version\",\"type\":\"string\"}]}"
      }
    }
    ```

    **Reading data after schema evolution:**

    ```python
    # Read old file (no session_id column)
    df = spark.read.parquet("date=2026-01-01/part-0001.parquet")
    # Schema: event_id, user_id, event_type, timestamp

    # Delta reader adds missing columns as NULL
    # Result: event_id, user_id, event_type, timestamp, session_id=NULL, app_version=NULL
    ```

    **2. Rename column (metadata-only with field IDs):**

    ```python
    # Rename column
    spark.sql("""
      ALTER TABLE events
      RENAME COLUMN properties TO event_properties
    """)

    # What happens (Iceberg):
    # 1. Update schema mapping (field_id=5 renamed from "properties" to "event_properties")
    # 2. Parquet column still named "properties" (backward compatible)
    # 3. Reader translates: Parquet "properties" → Iceberg "event_properties"

    # Cost: < 1 second (metadata update only)
    ```

    **Iceberg field IDs (critical for schema evolution):**

    ```json
    {
      "schema": {
        "fields": [
          {"id": 1, "name": "event_id", "type": "string"},
          {"id": 2, "name": "user_id", "type": "string"},
          {"id": 5, "name": "event_properties", "type": "map<string,string>"}
        ]
      }
    }
    ```

    **Why field IDs matter:**
    - Decouple schema from physical column names
    - Rename columns without rewriting data
    - Add/remove columns without breaking compatibility

    **3. Change column type (requires rewrite):**

    ```python
    # Change type (requires rewriting data!)
    spark.sql("""
      ALTER TABLE events
      ALTER COLUMN event_id TYPE BIGINT
    """)

    # What happens:
    # 1. Read all data files (10 PB)
    # 2. Cast event_id STRING → BIGINT
    # 3. Write new Parquet files with new schema
    # 4. Update Delta Log (add new files, remove old files)

    # Cost: Hours (full table rewrite)
    # Use CAST in queries instead (cheaper)
    ```

    **4. Drop column (metadata-only):**

    ```python
    # Drop column (soft delete)
    spark.sql("""
      ALTER TABLE events
      DROP COLUMN session_id
    """)

    # What happens:
    # 1. Update schema in Delta Log (mark column as dropped)
    # 2. Reader ignores column (doesn't read from Parquet)
    # 3. Data still exists in Parquet files (can be recovered)

    # Cost: < 1 second (metadata update only)
    # Physical deletion: Run compaction to rewrite files without column
    ```

    ---

    ### Schema Compatibility

    **Delta Lake schema evolution modes:**

    ```python
    # Mode 1: Strict (default)
    # - No automatic schema evolution
    # - Write fails if schema doesn't match
    spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "false")

    # Mode 2: Merge schema (add new columns)
    # - Automatically add new columns from DataFrame
    # - Existing columns must match types
    df.write.format("delta").mode("append").option("mergeSchema", "true").save(path)

    # Mode 3: Overwrite schema (replace schema)
    # - Replace entire schema
    # - Use for major schema changes
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    ```

    **Iceberg schema evolution:**

    ```python
    # Enable schema evolution
    spark.conf.set("spark.sql.iceberg.handle-timestamp-without-timezone", "true")

    # Write with schema evolution
    df.writeTo("events").append()  # Automatically adds new columns
    ```

=== "📊 Step 4: Scale & Optimize"

    ## Performance at Scale

    ### Write Performance

    **Target:** 100 TB/day ingestion (1.16 GB/sec)

    **Bottlenecks:**

    1. **S3 write throughput:** 5,500 requests/sec per prefix
       - Solution: Use random prefixes (uuid)
       - Example: `s3://bucket/uuid1/file1.parquet` instead of `s3://bucket/date/file1.parquet`

    2. **Small files:** Too many files (100K files/day)
       - Solution: Enable auto-compaction
       - Delta Lake: `delta.autoOptimize.optimizeWrite = true`

    3. **Commit conflicts:** Multiple writers to same partition
       - Solution: Partition by hour (more fine-grained)
       - Reduces conflicts: 24 partitions/day instead of 1

    **Optimization:**

    ```python
    # Optimize write performance
    spark.conf.set("spark.sql.files.maxRecordsPerFile", 1000000)  # 1M records/file
    spark.conf.set("spark.sql.shuffle.partitions", 2000)  # Match parallelism
    spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")  # Auto-optimize
    spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")  # Auto-compact

    # Write with optimal settings
    df.repartition(2000, "date", "country") \
      .write \
      .format("delta") \
      .mode("append") \
      .save("s3://data-lake/events")
    ```

    **Write throughput:**

    ```
    Baseline: 100 TB/day, 100K files/day
    - Write time: 10 hours
    - Throughput: 2.8 GB/sec
    - Cost: $100/day (Spark cluster)

    Optimized: 100 TB/day, 50K files/day (2 GB/file)
    - Write time: 5 hours (2x faster)
    - Throughput: 5.6 GB/sec
    - Cost: $50/day (50% savings)
    ```

    ---

    ### Read Performance

    **Target:** 50K queries/sec, < 10s latency (p95)

    **Bottlenecks:**

    1. **Metadata overhead:** LIST operations slow
       - Solution: Cache partition metadata (Redis)
       - Reduction: 5,000 LIST/sec → 50 LIST/sec (99% cache hit)

    2. **S3 GET requests:** 5M GET/sec (too many)
       - Solution: File cache (Alluxio, local SSD)
       - Reduction: 5M GET/sec → 500K GET/sec (90% cache hit)

    3. **Query parsing:** Parse SQL, plan, optimize
       - Solution: Pre-compiled queries, result cache
       - Reduction: 10s → 1s (10x faster)

    **Optimization:**

    ```python
    # Enable aggressive caching
    spark.conf.set("spark.sql.inMemoryColumnarStorage.compressed", "true")
    spark.conf.set("spark.sql.inMemoryColumnarStorage.batchSize", 10000)
    spark.conf.set("spark.databricks.io.cache.enabled", "true")  # Delta cache

    # Cache hot tables
    spark.sql("CACHE TABLE events")

    # Enable result caching
    spark.conf.set("spark.databricks.delta.queryCache.enabled", "true")
    ```

    **Query performance:**

    ```
    Baseline: 50K queries/sec, 50s latency
    - Without optimization
    - Scan full table (10 PB)

    With partition pruning: 50K queries/sec, 10s latency
    - Scan 0.1% of data (10 TB)
    - 5x faster

    With data skipping: 50K queries/sec, 5s latency
    - Scan 0.01% of data (1 TB)
    - 10x faster

    With caching: 50K queries/sec, 1s latency
    - 90% cache hit rate
    - 50x faster
    ```

    ---

    ### Storage Optimization

    **Goal:** Minimize storage cost (< $0.02/GB/month)

    **Strategies:**

    1. **Compression:** Parquet with Snappy (10x compression)
       - Raw data: 1 TB → Compressed: 100 GB

    2. **S3 storage classes:** Move cold data to cheaper tiers
       - S3 Standard: $0.023/GB (hot data, last 30 days)
       - S3 Standard-IA: $0.0125/GB (warm data, 30-365 days)
       - S3 Glacier: $0.004/GB (cold data, >365 days)

    3. **Retention policy:** Delete old data after 2 years
       - Saves: 50% of storage cost

    4. **Deduplication:** Identify and remove duplicate files
       - Saves: 5-10% of storage

    **Cost calculation:**

    ```
    Total data: 10 PB

    Storage breakdown:
    - Hot (30 days): 100 TB × $0.023/GB = $2,300/month
    - Warm (1-2 years): 9.9 PB × $0.0125/GB = $127,700/month
    - Total: $130,000/month ≈ $1.56M/year

    Optimizations:
    - Compression (10x): $1.56M → $156K/year
    - Tiering (50% to Glacier): $156K → $100K/year
    - Retention (2 years): $100K → $80K/year

    Final cost: $80K/year for 10 PB = $0.008/GB/year = $0.0007/GB/month
    ```

    ---

    ### Availability & Durability

    **Target:** 99.9% availability, 99.999999999% durability

    **S3 guarantees:**
    - Durability: 99.999999999% (11 nines)
    - Availability: 99.99% (4 nines)

    **Additional measures:**

    1. **Multi-region replication:** Replicate to second region
       - Cost: 2x storage cost
       - Benefit: Disaster recovery

    2. **S3 versioning:** Keep multiple versions of files
       - Cost: 1.5x storage cost (only changed files)
       - Benefit: Recover from accidental deletes

    3. **Transaction log backup:** Backup Delta Log to second bucket
       - Cost: Negligible (logs are small)
       - Benefit: Recover metadata if primary corrupted

    4. **Health checks:** Monitor S3 availability
       - Alert if S3 API errors exceed threshold
       - Failover to replica region

    ---

    ### Trade-offs

    | Aspect | Option A | Option B | Recommendation |
    |--------|----------|----------|----------------|
    | **Format** | Delta Lake (Databricks) | Iceberg (Open-source) | **Delta Lake** (better tooling, easier) or **Iceberg** (vendor-neutral) |
    | **Storage** | S3 (AWS) | HDFS (on-prem) | **S3** (lower cost, managed, scalable) |
    | **Catalog** | Hive Metastore | AWS Glue | **Hive Metastore** (compatibility) or **Glue** (managed) |
    | **Compaction** | Auto (on write) | Manual (scheduled) | **Scheduled** (more control, lower cost) |
    | **Partitioning** | Fine-grained (hour) | Coarse-grained (day) | **Day** (fewer partitions, less overhead) |
    | **Caching** | Local SSD | Alluxio (distributed) | **Alluxio** (shared cache, better hit rate) |
    | **Retention** | 1 year | 2 years | **2 years** (compliance, historical analysis) |

    ---

    ### Interview Tips

    **Common Follow-up Questions:**

    1. **"Why use Delta Lake/Iceberg instead of plain Parquet?"**
       - ACID transactions (no partial reads/writes)
       - Time travel (query historical data)
       - Schema evolution (add columns without rewrite)
       - Better performance (metadata caching, data skipping)
       - Example: Update 1M rows in 10B row table (impossible with Parquet, easy with Delta)

    2. **"How do you handle concurrent writes?"**
       - Optimistic concurrency control (like database transactions)
       - Writers read latest version, write files, attempt commit
       - If conflict (same partition modified), retry with exponential backoff
       - Conflicts rare if partitioning is fine-grained

    3. **"How does time travel work?"**
       - Transaction log stores all commits (add/remove files)
       - Query at version N: Replay log up to version N
       - Checkpoint optimization: Snapshot at version N (no replay needed)
       - Cost: Minimal (metadata-only, no data movement)

    4. **"How do you optimize query performance?"**
       - Partition pruning (skip 99.99% of partitions)
       - Data skipping (skip 90% of files using min/max stats)
       - Z-ordering (co-locate data by multiple dimensions)
       - Caching (metadata in Redis, data in Alluxio/SSD)
       - Result: 10s → 1s query time (10x faster)

    5. **"How do you handle small files?"**
       - Compaction: Merge small files into large files (target: 1-2 GB)
       - Auto-compaction: On write (Delta Lake)
       - Scheduled compaction: Daily/weekly for old partitions
       - VACUUM: Delete old files after retention period

    6. **"Delta Lake vs Iceberg - which one?"**
       - Delta Lake: Better Databricks integration, easier to use, auto-optimization
       - Iceberg: Vendor-neutral, more mature governance, better multi-engine support
       - Both provide similar features (ACID, time travel, schema evolution)
       - Recommendation: Use what your ecosystem supports (Databricks → Delta, AWS → Iceberg)

    7. **"How do you ensure data quality?"**
       - Schema validation on write (reject bad data)
       - Constraints (NOT NULL, CHECK, UNIQUE)
       - Expectations (data quality rules)
       - Quarantine tables (bad data goes to separate table)
       - Delta Lake: `dataChange` flag tracks data vs metadata changes

    **Key Points to Mention:**

    - Data lake provides ACID transactions on object storage
    - Transaction log enables time travel and schema evolution
    - Partition pruning and data skipping critical for performance
    - Compaction required to avoid small file problem
    - Optimistic concurrency control handles concurrent writes
    - S3 provides 11 nines durability, infinite scalability
    - Cost-effective: $0.02/GB/month (100x cheaper than data warehouse)

    ---

    ## Additional Considerations

    ### Security & Governance

    ```
    Security layers:

    1. Encryption
       - At-rest: S3 SSE-KMS (encrypt all files)
       - In-transit: TLS 1.3 (HTTPS)
       - Column-level: Encrypt sensitive columns (PII)

    2. Access Control
       - IAM roles: S3 bucket policies
       - Table-level: Hive Metastore permissions
       - Row-level: Predicate pushdown (filter by user)
       - Column-level: View-based access control

    3. Audit Logging
       - S3 CloudTrail: Track all S3 API calls
       - Delta Log: Track all table operations
       - Query logs: Track who queried what

    4. Data Lineage
       - Track data flow (source → transformations → destination)
       - Metadata catalog (AWS Glue, Alation)
       - Impact analysis (which dashboards affected by schema change?)
    ```

    ### Cost Optimization

    ```
    Cost breakdown (10 PB data):

    1. Storage: $130K/month
       - S3 Standard: $23/TB × 100 TB = $2.3K
       - S3 Standard-IA: $12.50/TB × 9,900 TB = $123.75K
       - S3 Glacier: $4/TB × 0 TB = $0

    2. API Requests: $10K/month
       - LIST: 50K/sec × $0.005/1000 = $6.5K
       - GET: 500K/sec × $0.0004/1000 = $4.3K
       - PUT: 10/sec × $0.005/1000 = $0.01K

    3. Data Transfer: $5K/month
       - Egress: 500 GB/sec × 10% cross-region × $0.02/GB = $5K
       - Ingress: Free

    4. Compute: $50K/month
       - Spark clusters: 100 nodes × 24 hours × $0.5/hour = $50K

    Total: $195K/month ≈ $2.34M/year

    Optimizations:
    - Reserved instances: $50K → $25K (50% savings)
    - Spot instances: $25K → $7.5K (70% savings)
    - S3 Intelligent-Tiering: $130K → $80K (38% savings)
    - Caching (reduce API calls): $10K → $1K (90% savings)

    Optimized total: $88.5K/month ≈ $1.06M/year (55% savings)
    ```

    ---

    ## Summary

    **System Characteristics:**

    - **Scale:** 10 PB data, 10B files, 100K writes/sec, 50K queries/sec
    - **Latency:** < 5s write commits, < 10s query latency (p95)
    - **Durability:** 99.999999999% (S3 guarantees)
    - **Cost:** $0.008/GB/year ($80K/year for 10 PB)

    **Core Components:**

    1. **Delta Lake/Iceberg:** ACID transactions on object storage
    2. **S3:** Durable, scalable, cost-effective storage
    3. **Parquet:** Columnar format with 10x compression
    4. **Transaction Log:** ACID guarantees, time travel
    5. **Metadata Catalog:** Centralized table/partition metadata
    6. **Compaction Service:** Merge small files
    7. **Caching Layer:** Redis (metadata), Alluxio (data)

    **Key Design Decisions:**

    - **ACID on S3:** Transaction log with optimistic concurrency
    - **Time travel:** Snapshot isolation via immutable log
    - **Schema evolution:** Metadata-only changes (add columns)
    - **Query optimization:** Partition pruning + data skipping (99.99% reduction)
    - **Compaction:** Scheduled compaction (target: 1-2 GB files)
    - **Caching:** 90% cache hit rate (metadata + data)
    - **Cost optimization:** S3 tiering, compression, retention

    **Performance:**

    - **Write:** 100 TB/day (1.16 GB/sec)
    - **Read:** 50K queries/sec with < 10s latency
    - **Storage:** 10 PB with < $0.02/GB/month
    - **Optimization:** 99.99% data pruned via partitioning + statistics

    This design provides enterprise-grade data lake capabilities with ACID transactions, time travel, and petabyte-scale performance at low cost.
