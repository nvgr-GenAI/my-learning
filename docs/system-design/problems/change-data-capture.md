# Design a Change Data Capture (CDC) System

A distributed platform that captures database changes in real-time by parsing transaction logs (MySQL binlog, Postgres WAL) and streams them to multiple destinations with exactly-once delivery, schema evolution support, and minimal impact on source databases.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1,000+ tables, 100K+ writes/sec, millions of rows/day, petabytes of data |
| **Key Challenges** | Log parsing (binlog/WAL), schema evolution, initial snapshot + incremental sync, DDL handling, exactly-once delivery, minimizing source impact |
| **Core Concepts** | Transaction log tailing, write-ahead log (WAL), binlog replication, event streaming, schema registry, at-least-once delivery, eventual consistency |
| **Companies** | Debezium, AWS DMS, Airbyte, Fivetran, Striim, Oracle GoldenGate, Qlik Replicate |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Log-based CDC** | Parse database transaction logs (MySQL binlog, Postgres WAL) | P0 (Must have) |
    | **Initial Snapshot** | Full table snapshot before incremental replication | P0 (Must have) |
    | **Incremental Replication** | Stream only changed data (inserts, updates, deletes) | P0 (Must have) |
    | **Schema Evolution** | Handle DDL changes (ADD COLUMN, ALTER TABLE) | P0 (Must have) |
    | **Multiple Sources** | Support MySQL, PostgreSQL, MongoDB, Oracle, SQL Server | P0 (Must have) |
    | **Multiple Sinks** | Kafka, data warehouses, databases, data lakes | P0 (Must have) |
    | **At-least-once Delivery** | Guarantee no data loss | P0 (Must have) |
    | **Offset Management** | Track binlog position, LSN, checkpoint | P0 (Must have) |
    | **Table Filtering** | Select specific tables/columns to replicate | P1 (Should have) |
    | **Data Transformations** | Basic field mapping, masking, filtering | P1 (Should have) |
    | **Monitoring & Alerting** | Lag metrics, error rates, throughput | P1 (Should have) |
    | **Multi-region Replication** | Cross-region CDC for disaster recovery | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Complex ETL transformations (use separate pipeline)
    - Data quality validation (use Great Expectations)
    - Real-time queries (use separate query engine)
    - Conflict resolution for multi-master (out of scope for CDC)
    - Full data migration tools (Migrate existing data separately)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Throughput** | 100K writes/sec per table | Support high-volume transactional databases |
    | **Latency** | < 1 second (log to sink) | Near real-time data synchronization |
    | **Availability** | 99.9% uptime | Critical for data pipelines |
    | **Source Impact** | < 5% overhead | Minimal performance impact on production DB |
    | **Durability** | Zero data loss | At-least-once delivery guarantee |
    | **Scalability** | 1,000+ tables per connector | Multi-tenant platform |
    | **Recovery Time** | < 5 minutes | Fast offset restoration |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Database workload:
    - Tables monitored: 1,000 tables
    - Write operations: 100K writes/sec (total across all tables)
    - Average row size: 2 KB (before/after JSON)
    - Daily changes: 100K × 86,400 = 8.64B row changes/day
    - Daily data volume: 8.64B × 2 KB = 17.28 TB/day

    Change distribution:
    - INSERT: 40% (40K/sec)
    - UPDATE: 50% (50K/sec)
    - DELETE: 10% (10K/sec)

    Initial snapshot:
    - Total rows: 10B rows (across 1,000 tables)
    - Average row size: 1 KB
    - Total snapshot size: 10B × 1 KB = 10 TB
    - Snapshot duration: 10 TB / 1 GB/sec = 10,000 seconds ≈ 3 hours

    Schema changes:
    - DDL operations: 100 DDL changes/day (ALTER TABLE, ADD COLUMN)
    - Schema validation: 1,000 schemas × 1 validation/min = 1,000/min

    Event processing:
    - Binlog events: 100K events/sec
    - Schema lookups: 100K × 0.1 (10% new tables) = 10K/sec
    - Transformations: 100K × 0.5 (50% filtered) = 50K/sec
    - Sink writes: 100K events/sec to Kafka/DB
    ```

    ### Storage Estimates

    ```
    Binlog storage (MySQL):

    Binlog events:
    - 100K writes/sec × 500 bytes (binlog row format) = 50 MB/sec
    - Daily binlog: 50 MB/sec × 86,400 = 4.32 TB/day
    - Retention: 7 days → 30.24 TB
    - With compression (3:1): 10 TB

    Kafka topic storage (change events):
    - Events: 100K/sec × 2 KB = 200 MB/sec = 17.28 TB/day
    - Retention: 7 days → 120 TB
    - With compaction (log compacted): 20 TB (only latest per key)

    Snapshot storage (S3):
    - Initial snapshot: 10 TB
    - Incremental snapshots: 500 GB/day (changed partitions)
    - Retention: 30 days → 15 TB
    - Total snapshot storage: 25 TB

    Offset/checkpoint storage:
    - Binlog positions: 1,000 tables × 100 bytes = 100 KB
    - WAL LSN: 1,000 tables × 50 bytes = 50 KB
    - Kafka offsets: 10,000 partitions × 100 bytes = 1 MB
    - Total checkpoint metadata: 2 MB (negligible)

    Schema registry:
    - Schemas: 1,000 tables × 50 KB (Avro schema) = 50 MB
    - Schema versions: 1,000 × 10 versions = 10,000 versions × 50 KB = 500 MB
    - Total schema storage: 1 GB

    Total storage:
    - Binlog: 10 TB (compressed)
    - Kafka: 20 TB (compacted)
    - Snapshots: 25 TB
    - Schemas: 1 GB
    - Total: ~55 TB
    ```

    ### Bandwidth Estimates

    ```
    Database read (initial snapshot):
    - 10 TB / 3 hours = 926 MB/sec = 7.4 Gbps

    Binlog read (incremental):
    - 50 MB/sec = 400 Mbps (per MySQL instance)
    - 10 MySQL instances: 4 Gbps

    Kafka write (change events):
    - 200 MB/sec = 1.6 Gbps
    - With replication (3x): 4.8 Gbps

    Sink write (to data warehouse):
    - 200 MB/sec = 1.6 Gbps
    - Multiple sinks (3 destinations): 4.8 Gbps

    Network requirements:
    - Per CDC connector: 1-2 Gbps (read + write)
    - Total cluster: 10-20 Gbps
    ```

    ### Server Estimates

    ```
    CDC connectors (Debezium):
    - Connectors: 10 connectors (1 per MySQL instance)
    - CPU: 8 cores per connector (binlog parsing, transformations)
    - Memory: 16 GB per connector (buffering, schema cache)
    - Network: 2 Gbps per connector
    - Total: 10 nodes (1 per connector)

    Kafka cluster (change event storage):
    - Brokers: 20 nodes (handle 100K events/sec)
    - Per broker: 32 cores, 128 GB RAM, 10 TB SSD, 10 Gbps network
    - Partitions: 10,000 partitions (10 per table × 1,000 tables)

    Schema Registry:
    - Nodes: 3 (HA setup)
    - Per node: 8 cores, 32 GB RAM
    - Storage: 10 GB SSD

    Kafka Connect (CDC framework):
    - Workers: 10 nodes (distribute connectors)
    - Per worker: 16 cores, 64 GB RAM, 10 Gbps network

    Monitoring & coordination:
    - ZooKeeper: 3 nodes (8 cores, 32 GB RAM)
    - Prometheus: 3 nodes (16 cores, 64 GB RAM)
    - Grafana: 2 nodes (8 cores, 32 GB RAM)

    Total infrastructure:
    - CDC connectors: 10 nodes
    - Kafka brokers: 20 nodes
    - Kafka Connect: 10 nodes
    - Schema Registry: 3 nodes
    - Monitoring: 8 nodes
    - Total: 51 nodes
    ```

    ---

    ## Key Assumptions

    1. 1,000 tables monitored, 100K writes/sec total
    2. Average row size: 2 KB (before/after state)
    3. Initial snapshot: 10 TB (10B rows)
    4. 7-day retention for change events (Kafka)
    5. < 1 second lag from database commit to sink
    6. At-least-once delivery (duplicates handled by sink)
    7. Schema changes: 100 DDL operations/day
    8. 10 MySQL instances (100 tables each)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Log-based CDC:** Parse transaction logs (non-intrusive)
    2. **Exactly-once source:** Read each change exactly once from log
    3. **At-least-once delivery:** Guarantee no data loss to sink
    4. **Schema evolution:** Handle DDL changes without downtime
    5. **Offset management:** Track log position for recovery
    6. **Initial snapshot:** Full table copy before incremental sync
    7. **Event streaming:** Real-time change event propagation

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Source Databases"
            MySQL1[(MySQL DB 1<br/>Production<br/>100 tables)]
            MySQL2[(MySQL DB 2<br/>Production<br/>100 tables)]
            Postgres1[(PostgreSQL<br/>Production<br/>200 tables)]
            Mongo1[(MongoDB<br/>Replica Set<br/>50 collections)]
        end

        subgraph "Transaction Logs"
            Binlog1[MySQL Binlog<br/>ROW format<br/>7-day retention<br/>GTID enabled]
            WAL1[Postgres WAL<br/>Logical replication<br/>Replication slot]
            Oplog1[MongoDB Oplog<br/>Change streams<br/>Capped collection]
        end

        subgraph "CDC Connectors Layer - Debezium"
            subgraph "MySQL CDC Connector 1"
                BinlogReader1[Binlog Reader<br/>Parse binlog events<br/>Track GTID position]
                SnapshotEngine1[Snapshot Engine<br/>Consistent snapshot<br/>Lock-free]
                SchemaHistory1[(Schema History<br/>DDL tracking<br/>Table versions)]
            end

            subgraph "PostgreSQL CDC Connector"
                WALReader[WAL Reader<br/>Decode logical changes<br/>Track LSN]
                SnapshotEngine2[Snapshot Engine<br/>Export snapshot<br/>pg_dump]
                SchemaHistory2[(Schema History<br/>DDL tracking)]
            end

            subgraph "MongoDB CDC Connector"
                OplogReader[Oplog Tailer<br/>Watch change streams<br/>Resume token]
                SnapshotEngine3[Snapshot Engine<br/>Consistent backup]
            end

            Transformer[Event Transformer<br/>Envelope format<br/>Before/After state<br/>Metadata enrichment]
            Filter[Event Filter<br/>Table whitelist<br/>Column selection<br/>Operation filter]
        end

        subgraph "Kafka Connect Framework"
            ConnectCluster[Kafka Connect Cluster<br/>10 workers<br/>Distributed mode<br/>Auto-rebalancing]
            OffsetStore[(Offset Storage<br/>Kafka topic<br/>Binlog positions<br/>LSN, resume tokens)]
            ConfigStore[(Config Storage<br/>Connector configs<br/>Task assignments)]
        end

        subgraph "Event Streaming - Kafka"
            subgraph "Change Event Topics"
                Topic1[db1.users<br/>1000 partitions<br/>Key: user_id<br/>Compacted]
                Topic2[db1.orders<br/>500 partitions<br/>Key: order_id<br/>Compacted]
                Topic3[db2.products<br/>200 partitions<br/>Key: product_id]
                TopicN[... 1,000 topics]
            end

            SchemaRegistry[Schema Registry<br/>Avro schemas<br/>Compatibility checks<br/>Version management]
        end

        subgraph "CDC Event Format"
            EventEnvelope[Event Envelope<br/>Before: old row<br/>After: new row<br/>Op: INSERT/UPDATE/DELETE<br/>Source: table, timestamp<br/>Transaction: tx_id, LSN]
        end

        subgraph "Sink Connectors"
            subgraph "Data Warehouse Sink"
                SnowflakeSink[Snowflake Sink<br/>MERGE/UPSERT<br/>Staging tables<br/>Batch writes]
                BigQuerySink[BigQuery Sink<br/>Streaming inserts<br/>Table decorators]
            end

            subgraph "Database Sink"
                PostgresSink[PostgreSQL Sink<br/>JDBC connector<br/>Transactional writes]
                CassandraSink[Cassandra Sink<br/>Write-optimized<br/>Eventually consistent]
            end

            subgraph "Data Lake Sink"
                S3Sink[S3 Sink<br/>Parquet files<br/>Partitioned by date<br/>Hourly batches]
                IcebergSink[Iceberg Sink<br/>ACID transactions<br/>Time travel<br/>Schema evolution]
            end

            subgraph "Search Sink"
                ElasticSink[Elasticsearch Sink<br/>Bulk indexing<br/>Idempotent updates]
            end
        end

        subgraph "Monitoring & Orchestration"
            Prometheus[Prometheus<br/>Lag metrics<br/>Throughput<br/>Error rates]
            Grafana[Grafana Dashboards<br/>CDC lag per table<br/>Event processing rate<br/>Connector health]
            Alerting[Alert Manager<br/>Lag > 1 minute<br/>Connector failures<br/>Schema conflicts]
            Kafka_UI[Kafka UI<br/>Topic lag<br/>Consumer groups]
        end

        MySQL1 -->|Write operations| Binlog1
        MySQL2 -->|Write operations| Binlog1
        Postgres1 -->|Write operations| WAL1
        Mongo1 -->|Write operations| Oplog1

        Binlog1 -->|Tail binlog| BinlogReader1
        WAL1 -->|Decode WAL| WALReader
        Oplog1 -->|Watch oplog| OplogReader

        BinlogReader1 --> SnapshotEngine1
        SnapshotEngine1 -->|Full table scan| MySQL1

        BinlogReader1 --> Transformer
        WALReader --> Transformer
        OplogReader --> Transformer

        Transformer --> Filter
        Filter --> EventEnvelope

        EventEnvelope -->|Validate schema| SchemaRegistry

        EventEnvelope --> Topic1
        EventEnvelope --> Topic2
        EventEnvelope --> Topic3
        EventEnvelope --> TopicN

        BinlogReader1 -.->|Store GTID| OffsetStore
        WALReader -.->|Store LSN| OffsetStore
        OplogReader -.->|Store resume token| OffsetStore

        ConnectCluster -.->|Manage| BinlogReader1
        ConnectCluster -.->|Manage| WALReader
        ConnectCluster -.->|Manage| OplogReader
        ConnectCluster -.->|Read offsets| OffsetStore
        ConnectCluster -.->|Read config| ConfigStore

        Topic1 -->|Consume| SnowflakeSink
        Topic1 -->|Consume| PostgresSink
        Topic1 -->|Consume| S3Sink
        Topic1 -->|Consume| ElasticSink

        Topic2 -->|Consume| BigQuerySink
        Topic2 -->|Consume| CassandraSink
        Topic2 -->|Consume| IcebergSink

        SchemaHistory1 -.->|Track DDL| BinlogReader1
        SchemaHistory2 -.->|Track DDL| WALReader

        BinlogReader1 -->|Metrics| Prometheus
        WALReader -->|Metrics| Prometheus
        Topic1 -->|Lag metrics| Prometheus
        SnowflakeSink -->|Metrics| Prometheus

        Prometheus --> Grafana
        Prometheus --> Alerting

        Kafka_UI -.->|Monitor| Topic1

        style BinlogReader1 fill:#e1f5ff
        style WALReader fill:#e1f5ff
        style SnapshotEngine1 fill:#fff9c4
        style Transformer fill:#e8f5e9
        style SchemaRegistry fill:#fce4ec
        style OffsetStore fill:#ffe1e1
        style Topic1 fill:#f3e5f5
        style SnowflakeSink fill:#e0f2f1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Debezium** | Production-ready CDC, multi-DB support, schema tracking | AWS DMS (proprietary), custom binlog parser (complex), Airbyte (less mature CDC) |
    | **Kafka** | Durable event storage, replay capability, decoupling | Direct DB-to-DB sync (tight coupling), message queue (no replay), Kinesis (vendor lock-in) |
    | **Log-based CDC** | Non-intrusive, captures deletes, low source impact | Trigger-based (high overhead), Timestamp-based (no deletes), Query-based (polling, high load) |
    | **Kafka Connect** | Distributed framework, fault tolerance, offset management | Custom scripts (reinventing wheel), Airflow (batch-oriented), Lambda (stateless) |
    | **Schema Registry** | Schema versioning, compatibility checks, evolution | Embed in events (bloat), external DB (latency), no validation (errors) |
    | **At-least-once** | Simpler than exactly-once, idempotent sinks handle duplicates | Exactly-once (complex 2PC), At-most-once (data loss) |
    | **Initial Snapshot** | Consistent starting point, historical data | Skip snapshot (missing history), manual export (operational burden) |

    **Key Trade-off:** We chose **log-based CDC over query-based** because it's non-intrusive (< 5% overhead), captures deletes, and provides low latency (< 1 second). The trade-off is complexity in parsing binary logs and handling schema changes.

    ---

    ## Data Flow Overview

    **CDC Pipeline Flow:**

    1. **Database Write:** Application writes to MySQL (INSERT/UPDATE/DELETE)
    2. **Transaction Commit:** MySQL writes to binlog (row-based format)
    3. **Binlog Tailing:** Debezium connector reads binlog events
    4. **Schema Lookup:** Load table schema from schema history
    5. **Event Transformation:** Convert binlog to Kafka envelope format
    6. **Schema Validation:** Check against Schema Registry
    7. **Kafka Publish:** Write change event to Kafka topic (partitioned by key)
    8. **Offset Commit:** Store binlog position in offset topic
    9. **Sink Consumption:** Sink connectors consume from Kafka
    10. **Data Loading:** Write to Snowflake/S3/Elasticsearch (UPSERT/MERGE)

    **Latency Breakdown:**
    ```
    Database commit: 0ms (baseline)
    Binlog write: 1-5ms
    Binlog read: 10-50ms (tailing lag)
    Transformation: 10-20ms
    Kafka write: 10-30ms (3 replicas)
    Sink consumption: 50-200ms (batch size)
    Sink write: 100-500ms (network + DB)

    Total E2E latency: 200-800ms (p95 < 1 second)
    ```

    ---

    ## Event Format

    **Debezium Change Event Envelope:**

    ```json
    {
      "before": {
        "id": 123,
        "name": "John Doe",
        "email": "john@example.com",
        "updated_at": "2026-02-01T10:00:00Z"
      },
      "after": {
        "id": 123,
        "name": "John Smith",
        "email": "john.smith@example.com",
        "updated_at": "2026-02-05T15:30:00Z"
      },
      "source": {
        "version": "2.5.0.Final",
        "connector": "mysql",
        "name": "prod-mysql-1",
        "ts_ms": 1738767000000,
        "snapshot": "false",
        "db": "ecommerce",
        "table": "users",
        "server_id": 1,
        "gtid": "3e11fa47-71ca-11e1-9e33-c80aa9429562:23",
        "file": "mysql-bin.000003",
        "pos": 154,
        "row": 0,
        "thread": 7,
        "query": null
      },
      "op": "u",
      "ts_ms": 1738767000105,
      "transaction": {
        "id": "3e11fa47-71ca-11e1-9e33-c80aa9429562:23",
        "total_order": 1,
        "data_collection_order": 1
      }
    }
    ```

    **Operation Types:**
    - `c` = CREATE (INSERT)
    - `u` = UPDATE
    - `d` = DELETE
    - `r` = READ (initial snapshot)

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Log-Based CDC Implementation

    ### MySQL Binlog CDC

    **Binlog Configuration:**

    ```sql
    -- Enable binlog with ROW format (required for CDC)
    SET GLOBAL binlog_format = 'ROW';
    SET GLOBAL binlog_row_image = 'FULL';  -- Capture before/after state

    -- Enable GTID (Global Transaction ID)
    SET GLOBAL gtid_mode = ON;
    SET GLOBAL enforce_gtid_consistency = ON;

    -- Binlog retention
    SET GLOBAL expire_logs_days = 7;

    -- Verify configuration
    SHOW VARIABLES LIKE 'binlog%';
    SHOW VARIABLES LIKE 'gtid%';
    ```

    **Debezium MySQL Connector Configuration:**

    ```json
    {
      "name": "mysql-cdc-connector",
      "config": {
        "connector.class": "io.debezium.connector.mysql.MySqlConnector",
        "tasks.max": "1",
        "database.hostname": "prod-mysql-1.example.com",
        "database.port": "3306",
        "database.user": "debezium",
        "database.password": "${mysql_password}",
        "database.server.id": "12345",
        "database.server.name": "prod-mysql-1",
        "database.whitelist": "ecommerce,analytics",
        "table.whitelist": "ecommerce.users,ecommerce.orders,analytics.events",

        "snapshot.mode": "initial",
        "snapshot.locking.mode": "minimal",
        "snapshot.select.statement.overrides": "ecommerce.users:SELECT id, name, email FROM users WHERE active = 1",

        "binlog.buffer.size": "8192",
        "database.history.kafka.bootstrap.servers": "kafka1:9092,kafka2:9092",
        "database.history.kafka.topic": "schema-history.prod-mysql-1",

        "include.schema.changes": "true",
        "time.precision.mode": "connect",
        "decimal.handling.mode": "precise",
        "bigint.unsigned.handling.mode": "precise",

        "transforms": "unwrap,route",
        "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
        "transforms.unwrap.drop.tombstones": "false",
        "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
        "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
        "transforms.route.replacement": "$2.$3",

        "key.converter": "org.apache.kafka.connect.json.JsonConverter",
        "value.converter": "io.confluent.connect.avro.AvroConverter",
        "value.converter.schema.registry.url": "http://schema-registry:8081"
      }
    }
    ```

    **Binlog Reader Implementation:**

    ```java
    import com.github.shyiko.mysql.binlog.BinaryLogClient;
    import com.github.shyiko.mysql.binlog.event.*;

    public class MySQLBinlogReader {

        private final BinaryLogClient client;
        private final KafkaProducer<String, ChangeEvent> producer;
        private final OffsetManager offsetManager;

        public MySQLBinlogReader(String host, int port, String user, String password) {
            this.client = new BinaryLogClient(host, port, user, password);
            this.producer = createKafkaProducer();
            this.offsetManager = new OffsetManager();

            // Set GTID for resumption
            String lastGtid = offsetManager.getLastGtid();
            if (lastGtid != null) {
                client.setGtidSet(lastGtid);
            }

            // Register event listeners
            client.registerEventListener(this::handleEvent);
            client.registerLifecycleListener(this::handleLifecycle);
        }

        public void start() throws IOException {
            // Perform initial snapshot
            if (!offsetManager.hasSnapshot()) {
                performInitialSnapshot();
            }

            // Start binlog tailing
            client.connect();
        }

        private void performInitialSnapshot() {
            log.info("Starting initial snapshot...");

            try (Connection conn = getConnection()) {
                // Lock tables for consistent snapshot
                conn.setAutoCommit(false);
                conn.createStatement().execute("FLUSH TABLES WITH READ LOCK");

                // Get current binlog position
                ResultSet rs = conn.createStatement().executeQuery("SHOW MASTER STATUS");
                rs.next();
                String binlogFile = rs.getString("File");
                long binlogPos = rs.getLong("Position");
                String gtidSet = rs.getString("Executed_Gtid_Set");

                // Read all tables
                List<String> tables = getTablesToSnapshot();
                for (String table : tables) {
                    snapshotTable(conn, table);
                }

                // Store offset
                offsetManager.saveOffset(binlogFile, binlogPos, gtidSet);

                // Release lock
                conn.createStatement().execute("UNLOCK TABLES");
                conn.commit();

                log.info("Initial snapshot completed");
            } catch (SQLException e) {
                throw new RuntimeException("Snapshot failed", e);
            }
        }

        private void snapshotTable(Connection conn, String table) throws SQLException {
            String query = String.format("SELECT * FROM %s", table);

            try (Statement stmt = conn.createStatement();
                 ResultSet rs = stmt.executeQuery(query)) {

                ResultSetMetaData metadata = rs.getMetaData();
                int columnCount = metadata.getColumnCount();

                while (rs.next()) {
                    Map<String, Object> row = new HashMap<>();

                    for (int i = 1; i <= columnCount; i++) {
                        String columnName = metadata.getColumnName(i);
                        Object value = rs.getObject(i);
                        row.put(columnName, value);
                    }

                    // Create snapshot event
                    ChangeEvent event = ChangeEvent.builder()
                        .operation(Operation.READ)
                        .before(null)
                        .after(row)
                        .source(createSource(table, true))
                        .timestamp(System.currentTimeMillis())
                        .build();

                    // Publish to Kafka
                    String key = extractKey(row, table);
                    producer.send(new ProducerRecord<>(
                        getTopicName(table), key, event
                    ));
                }

                log.info("Snapshot completed for table: {}", table);
            }
        }

        private void handleEvent(Event event) {
            EventType eventType = event.getHeader().getEventType();

            switch (eventType) {
                case TABLE_MAP:
                    handleTableMap((TableMapEventData) event.getData());
                    break;

                case EXT_WRITE_ROWS:
                case WRITE_ROWS:
                    handleInsert((WriteRowsEventData) event.getData());
                    break;

                case EXT_UPDATE_ROWS:
                case UPDATE_ROWS:
                    handleUpdate((UpdateRowsEventData) event.getData());
                    break;

                case EXT_DELETE_ROWS:
                case DELETE_ROWS:
                    handleDelete((DeleteRowsEventData) event.getData());
                    break;

                case QUERY:
                    handleDDL((QueryEventData) event.getData());
                    break;

                case XID:
                    handleCommit((XidEventData) event.getData());
                    break;

                case GTID:
                    handleGtid((GtidEventData) event.getData());
                    break;

                default:
                    // Ignore other event types
                    break;
            }
        }

        private void handleInsert(WriteRowsEventData data) {
            String table = tableMap.get(data.getTableId()).getTable();

            for (Serializable[] row : data.getRows()) {
                Map<String, Object> after = convertRow(row, table);

                ChangeEvent event = ChangeEvent.builder()
                    .operation(Operation.CREATE)
                    .before(null)
                    .after(after)
                    .source(createSource(table, false))
                    .timestamp(System.currentTimeMillis())
                    .build();

                publishEvent(table, after, event);
            }
        }

        private void handleUpdate(UpdateRowsEventData data) {
            String table = tableMap.get(data.getTableId()).getTable();

            for (Map.Entry<Serializable[], Serializable[]> row : data.getRows()) {
                Map<String, Object> before = convertRow(row.getKey(), table);
                Map<String, Object> after = convertRow(row.getValue(), table);

                ChangeEvent event = ChangeEvent.builder()
                    .operation(Operation.UPDATE)
                    .before(before)
                    .after(after)
                    .source(createSource(table, false))
                    .timestamp(System.currentTimeMillis())
                    .build();

                publishEvent(table, after, event);
            }
        }

        private void handleDelete(DeleteRowsEventData data) {
            String table = tableMap.get(data.getTableId()).getTable();

            for (Serializable[] row : data.getRows()) {
                Map<String, Object> before = convertRow(row, table);

                ChangeEvent event = ChangeEvent.builder()
                    .operation(Operation.DELETE)
                    .before(before)
                    .after(null)
                    .source(createSource(table, false))
                    .timestamp(System.currentTimeMillis())
                    .build();

                publishEvent(table, before, event);
            }
        }

        private void handleDDL(QueryEventData data) {
            String sql = data.getSql();

            if (isDDL(sql)) {
                log.info("DDL detected: {}", sql);

                // Parse DDL
                DDLChange change = parseDDL(sql);

                // Update schema history
                schemaHistory.recordDDL(change);

                // Publish schema change event
                SchemaChangeEvent event = SchemaChangeEvent.builder()
                    .ddl(sql)
                    .database(data.getDatabase())
                    .timestamp(System.currentTimeMillis())
                    .build();

                producer.send(new ProducerRecord<>(
                    "schema-changes", event
                ));
            }
        }

        private void publishEvent(String table, Map<String, Object> row, ChangeEvent event) {
            String topic = getTopicName(table);
            String key = extractKey(row, table);

            ProducerRecord<String, ChangeEvent> record =
                new ProducerRecord<>(topic, key, event);

            producer.send(record, (metadata, exception) -> {
                if (exception != null) {
                    log.error("Failed to publish event", exception);
                    // Retry or dead-letter queue
                } else {
                    // Update offset after successful publish
                    offsetManager.updateOffset(currentGtid);
                }
            });
        }
    }
    ```

    ---

    ### PostgreSQL WAL CDC

    **PostgreSQL Configuration:**

    ```sql
    -- Enable logical replication
    ALTER SYSTEM SET wal_level = 'logical';
    ALTER SYSTEM SET max_replication_slots = 10;
    ALTER SYSTEM SET max_wal_senders = 10;

    -- Restart PostgreSQL
    SELECT pg_reload_conf();

    -- Create replication slot
    SELECT * FROM pg_create_logical_replication_slot('debezium_slot', 'pgoutput');

    -- Create publication
    CREATE PUBLICATION dbz_publication FOR ALL TABLES;

    -- Grant permissions
    CREATE ROLE debezium WITH REPLICATION LOGIN PASSWORD 'password';
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;
    GRANT USAGE ON SCHEMA public TO debezium;
    ```

    **Debezium PostgreSQL Connector:**

    ```json
    {
      "name": "postgres-cdc-connector",
      "config": {
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
        "tasks.max": "1",
        "database.hostname": "prod-postgres-1.example.com",
        "database.port": "5432",
        "database.user": "debezium",
        "database.password": "${postgres_password}",
        "database.dbname": "ecommerce",
        "database.server.name": "prod-postgres-1",
        "table.include.list": "public.users,public.orders",

        "slot.name": "debezium_slot",
        "publication.name": "dbz_publication",
        "plugin.name": "pgoutput",

        "snapshot.mode": "initial",
        "snapshot.select.statement.overrides": "public.users:SELECT * FROM users WHERE active = true",

        "time.precision.mode": "adaptive",
        "decimal.handling.mode": "precise",
        "hstore.handling.mode": "json",
        "interval.handling.mode": "string",

        "heartbeat.interval.ms": "10000",
        "heartbeat.action.query": "INSERT INTO heartbeat (timestamp) VALUES (NOW())",

        "value.converter": "io.confluent.connect.avro.AvroConverter",
        "value.converter.schema.registry.url": "http://schema-registry:8081"
      }
    }
    ```

    ---

    ## 3.2 Schema Evolution Handling

    **Schema Change Detection:**

    ```java
    public class SchemaEvolutionHandler {

        private final SchemaRegistry schemaRegistry;
        private final Map<String, TableSchema> schemaCache;

        public void handleDDL(DDLEvent ddlEvent) {
            String sql = ddlEvent.getSql();
            String table = ddlEvent.getTable();

            log.info("Processing DDL: {}", sql);

            if (sql.contains("ALTER TABLE") && sql.contains("ADD COLUMN")) {
                handleAddColumn(table, sql);
            } else if (sql.contains("ALTER TABLE") && sql.contains("DROP COLUMN")) {
                handleDropColumn(table, sql);
            } else if (sql.contains("ALTER TABLE") && sql.contains("MODIFY COLUMN")) {
                handleModifyColumn(table, sql);
            } else if (sql.contains("CREATE TABLE")) {
                handleCreateTable(table, sql);
            } else if (sql.contains("DROP TABLE")) {
                handleDropTable(table);
            }
        }

        private void handleAddColumn(String table, String sql) {
            // Parse DDL
            ColumnDefinition newColumn = parseDDL(sql);

            // Get current schema
            TableSchema currentSchema = schemaCache.get(table);

            // Create new schema version
            TableSchema newSchema = currentSchema.addColumn(newColumn);

            // Register with Schema Registry (backward compatible)
            try {
                schemaRegistry.register(
                    table + "-value",
                    newSchema.toAvroSchema(),
                    CompatibilityMode.BACKWARD
                );

                log.info("Schema evolved for {}: added column {}",
                    table, newColumn.getName());

                // Update cache
                schemaCache.put(table, newSchema);

            } catch (IncompatibleSchemaException e) {
                log.error("Schema evolution failed: incompatible change", e);
                throw new RuntimeException("Cannot add column: incompatible schema", e);
            }
        }

        private void handleDropColumn(String table, String sql) {
            // Get current schema
            TableSchema currentSchema = schemaCache.get(table);

            // Parse column to drop
            String columnName = parseColumnName(sql);

            // Option 1: Mark as deprecated (recommended)
            TableSchema newSchema = currentSchema.deprecateColumn(columnName);

            // Option 2: Remove from schema (breaking change)
            // TableSchema newSchema = currentSchema.removeColumn(columnName);

            // Register with Schema Registry (forward compatible)
            try {
                schemaRegistry.register(
                    table + "-value",
                    newSchema.toAvroSchema(),
                    CompatibilityMode.FORWARD
                );

                log.info("Schema evolved for {}: deprecated column {}",
                    table, columnName);

                schemaCache.put(table, newSchema);

            } catch (IncompatibleSchemaException e) {
                log.error("Schema evolution failed", e);
                throw new RuntimeException("Cannot drop column", e);
            }
        }

        private void handleModifyColumn(String table, String sql) {
            // Type changes are tricky - often incompatible
            // Best practice: Add new column, deprecate old column

            log.warn("Column type change detected for {}: {}", table, sql);

            // Strategy: Create compatibility layer
            // 1. Add new column with new type
            // 2. Populate from old column
            // 3. Deprecate old column
            // 4. Eventually drop old column

            throw new UnsupportedOperationException(
                "Direct column type changes not supported. " +
                "Use add/deprecate strategy instead."
            );
        }
    }
    ```

    **Avro Schema Evolution:**

    ```json
    {
      "type": "record",
      "name": "User",
      "namespace": "com.example.ecommerce",
      "fields": [
        {
          "name": "id",
          "type": "long"
        },
        {
          "name": "name",
          "type": "string"
        },
        {
          "name": "email",
          "type": "string"
        },
        {
          "name": "phone",
          "type": ["null", "string"],
          "default": null,
          "doc": "Added in v2"
        },
        {
          "name": "address",
          "type": [
            "null",
            {
              "type": "record",
              "name": "Address",
              "fields": [
                {"name": "street", "type": "string"},
                {"name": "city", "type": "string"},
                {"name": "country", "type": "string"}
              ]
            }
          ],
          "default": null,
          "doc": "Added in v3"
        }
      ]
    }
    ```

    ---

    ## 3.3 Initial Snapshot + Incremental Sync

    **Snapshot Strategies:**

    ```java
    public enum SnapshotMode {
        /**
         * Always perform initial snapshot
         * Use case: First-time setup
         */
        INITIAL,

        /**
         * Snapshot when no offset exists
         * Use case: Resume after offset lost
         */
        WHEN_NEEDED,

        /**
         * Skip snapshot, start from current binlog
         * Use case: Only capture changes going forward
         */
        NEVER,

        /**
         * Snapshot specific tables only
         * Use case: Selective initial load
         */
        SCHEMA_ONLY,

        /**
         * Incremental snapshot (chunk by chunk)
         * Use case: Large tables, avoid locking
         */
        INCREMENTAL
    }
    ```

    **Lock-Free Snapshot (Incremental):**

    ```java
    public class IncrementalSnapshotEngine {

        /**
         * Incremental snapshot algorithm:
         * 1. Read table in chunks (e.g., 10,000 rows)
         * 2. Continue binlog tailing in parallel
         * 3. De-duplicate changes during snapshot
         * 4. Merge snapshot + incremental changes
         */
        public void performIncrementalSnapshot(String table) {
            // Get table metadata
            TableMetadata metadata = getTableMetadata(table);
            Object minKey = metadata.getMinPrimaryKey();
            Object maxKey = metadata.getMaxPrimaryKey();

            // Calculate chunk boundaries
            List<Range> chunks = calculateChunks(minKey, maxKey, CHUNK_SIZE);

            log.info("Starting incremental snapshot for {} ({} chunks)",
                table, chunks.size());

            // Start binlog tailing (captures concurrent changes)
            BinlogPosition startPosition = getCurrentBinlogPosition();

            // Read chunks
            for (Range chunk : chunks) {
                readChunk(table, chunk);

                // Yield to allow binlog events to be processed
                Thread.sleep(100);
            }

            // De-duplicate: Remove snapshot rows if updated during snapshot
            deduplicateEvents(table, startPosition);

            log.info("Incremental snapshot completed for {}", table);
        }

        private void readChunk(String table, Range range) {
            String query = String.format(
                "SELECT * FROM %s WHERE id >= ? AND id < ? ORDER BY id",
                table
            );

            try (PreparedStatement stmt = conn.prepareStatement(query)) {
                stmt.setObject(1, range.start);
                stmt.setObject(2, range.end);

                try (ResultSet rs = stmt.executeQuery()) {
                    while (rs.next()) {
                        Map<String, Object> row = convertResultSet(rs);

                        // Mark as snapshot event
                        ChangeEvent event = ChangeEvent.builder()
                            .operation(Operation.READ)
                            .after(row)
                            .source(createSource(table, true))
                            .snapshotChunk(range.chunkId)
                            .build();

                        publishEvent(table, row, event);
                    }
                }
            }
        }

        private void deduplicateEvents(String table, BinlogPosition startPos) {
            /**
             * Deduplication strategy:
             *
             * Timeline:
             * T0: Start snapshot
             * T1: Snapshot chunk 1 (includes row X with value A)
             * T2: Row X updated to value B (binlog event)
             * T3: Snapshot chunk 2
             * T4: End snapshot
             *
             * Result:
             * - Snapshot event: X = A (older)
             * - Binlog event: X = B (newer)
             * - Keep: X = B (binlog wins)
             */

            // Approach: Let sink handle deduplication
            // Kafka log compaction keeps latest event per key
            // Sink performs UPSERT/MERGE based on timestamp
        }
    }
    ```

    ---

    ## 3.4 Handling Large Transactions

    **Challenge:** A single transaction with 1M row updates

    ```java
    public class LargeTransactionHandler {

        private final int MAX_TRANSACTION_SIZE = 10000;
        private final Map<String, List<ChangeEvent>> pendingTransactions;

        public void handleTransactionEvent(TransactionEvent event) {
            String txId = event.getTransactionId();

            if (event.isBegin()) {
                // Start new transaction
                pendingTransactions.put(txId, new ArrayList<>());

            } else if (event.isEnd()) {
                // Commit transaction
                List<ChangeEvent> events = pendingTransactions.remove(txId);

                if (events.size() > MAX_TRANSACTION_SIZE) {
                    // Split into multiple batches
                    splitAndPublishTransaction(txId, events);
                } else {
                    // Publish as single transaction
                    publishTransaction(txId, events);
                }
            }
        }

        public void handleChangeEvent(ChangeEvent event) {
            String txId = event.getTransactionId();

            if (pendingTransactions.containsKey(txId)) {
                // Buffer event until transaction commits
                pendingTransactions.get(txId).add(event);
            } else {
                // Standalone event (autocommit)
                publishEvent(event);
            }
        }

        private void splitAndPublishTransaction(String txId, List<ChangeEvent> events) {
            log.warn("Large transaction detected: {} events", events.size());

            // Strategy: Break into smaller batches
            // Trade-off: Lose atomicity, but avoid memory issues

            int batchSize = MAX_TRANSACTION_SIZE;
            for (int i = 0; i < events.size(); i += batchSize) {
                int end = Math.min(i + batchSize, events.size());
                List<ChangeEvent> batch = events.subList(i, end);

                // Publish batch with sub-transaction ID
                String subTxId = txId + "-" + (i / batchSize);
                publishBatch(subTxId, batch);

                log.info("Published sub-transaction {}/{}",
                    (i / batchSize) + 1,
                    (events.size() + batchSize - 1) / batchSize);
            }
        }
    }
    ```

    ---

    ## 3.5 Multi-Region Replication

    **Active-Active Replication:**

    ```mermaid
    graph LR
        subgraph "Region US-East"
            DB1[(MySQL Primary<br/>US-East)]
            CDC1[CDC Connector<br/>US-East]
            Kafka1[Kafka Cluster<br/>US-East]
        end

        subgraph "Region EU-West"
            DB2[(MySQL Primary<br/>EU-West)]
            CDC2[CDC Connector<br/>EU-West]
            Kafka2[Kafka Cluster<br/>EU-West]
        end

        subgraph "Global Data Layer"
            Snowflake[(Snowflake<br/>Multi-region)]
            S3_US[S3 Bucket<br/>US-East]
            S3_EU[S3 Bucket<br/>EU-West]
        end

        DB1 -->|Binlog| CDC1
        CDC1 -->|Publish| Kafka1
        Kafka1 -->|MirrorMaker| Kafka2

        DB2 -->|Binlog| CDC2
        CDC2 -->|Publish| Kafka2
        Kafka2 -->|MirrorMaker| Kafka1

        Kafka1 -->|Sink| Snowflake
        Kafka2 -->|Sink| Snowflake

        Kafka1 -->|Backup| S3_US
        Kafka2 -->|Backup| S3_EU
    ```

    **Conflict Resolution:**

    ```java
    public class ConflictResolver {

        public enum ConflictResolutionStrategy {
            LAST_WRITE_WINS,      // Latest timestamp wins
            SOURCE_PRIORITY,       // Prefer specific region
            CUSTOM_LOGIC          // Application-specific
        }

        public ChangeEvent resolveConflict(
                ChangeEvent event1,  // US-East
                ChangeEvent event2   // EU-West
        ) {
            // Detect conflict: same key, different values
            if (!event1.getKey().equals(event2.getKey())) {
                throw new IllegalArgumentException("Not a conflict");
            }

            // Strategy 1: Last-write-wins (timestamp-based)
            if (event1.getTimestamp() > event2.getTimestamp()) {
                return event1;
            } else if (event2.getTimestamp() > event1.getTimestamp()) {
                return event2;
            } else {
                // Same timestamp: Use source priority
                return resolveBySourcePriority(event1, event2);
            }
        }

        private ChangeEvent resolveBySourcePriority(
                ChangeEvent event1,
                ChangeEvent event2
        ) {
            // Priority: US-East > EU-West
            String source1 = event1.getSource().getRegion();
            String source2 = event2.getSource().getRegion();

            if ("us-east".equals(source1)) {
                return event1;
            } else {
                return event2;
            }
        }
    }
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scalability Bottlenecks

    ### 1. Log Parsing Bottleneck

    **Problem:** Single-threaded binlog reader can't keep up with 100K writes/sec

    **Solution: Parallel Snapshot + Partitioned Incremental**

    ```java
    public class ParallelSnapshotEngine {

        /**
         * Parallelize snapshot using table partitions
         *
         * Example: users table partitioned by id ranges
         * - Worker 1: id 1-1M
         * - Worker 2: id 1M-2M
         * - Worker 3: id 2M-3M
         */
        public void performParallelSnapshot(String table, int parallelism) {
            TableMetadata metadata = getTableMetadata(table);
            List<Range> partitions = partitionByPrimaryKey(
                metadata, parallelism
            );

            // Create snapshot tasks
            List<CompletableFuture<Void>> futures = partitions.stream()
                .map(partition -> CompletableFuture.runAsync(() -> {
                    snapshotPartition(table, partition);
                }, snapshotExecutor))
                .collect(Collectors.toList());

            // Wait for all partitions
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .join();

            log.info("Parallel snapshot completed for {} ({} partitions)",
                table, parallelism);
        }
    }
    ```

    **Solution: Kafka Partitioning**

    ```
    Scaling strategy:
    1. Partition Kafka topics by primary key (user_id, order_id)
    2. Scale Kafka consumers (more partitions = more parallelism)
    3. Co-locate sink connectors with Kafka partitions

    Example:
    - Topic: ecommerce.users (1,000 partitions)
    - Consumers: 100 sink connectors (10 partitions each)
    - Throughput: 100K events/sec / 100 = 1K events/sec per consumer
    ```

    ---

    ### 2. Schema Registry Bottleneck

    **Problem:** Every event requires schema lookup (10K/sec)

    **Solution: Schema Caching**

    ```java
    public class SchemaCache {

        private final LoadingCache<String, Schema> cache;

        public SchemaCache(SchemaRegistryClient registry) {
            this.cache = Caffeine.newBuilder()
                .maximumSize(10_000)
                .expireAfterWrite(1, TimeUnit.HOURS)
                .refreshAfterWrite(5, TimeUnit.MINUTES)
                .buildAsync((key, executor) ->
                    CompletableFuture.supplyAsync(() ->
                        registry.getLatestSchemaMetadata(key).getSchema()
                    )
                );
        }

        public Schema getSchema(String subject) {
            return cache.get(subject).join();
        }
    }
    ```

    ---

    ### 3. State Management (Offsets)

    **Problem:** Frequent offset commits (every 1 second) create contention

    **Solution: Batched Offset Commits**

    ```java
    public class BatchedOffsetManager {

        private final Map<String, Offset> pendingOffsets = new ConcurrentHashMap<>();
        private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        public BatchedOffsetManager() {
            // Commit offsets every 5 seconds
            scheduler.scheduleAtFixedRate(
                this::commitPendingOffsets,
                5, 5, TimeUnit.SECONDS
            );
        }

        public void updateOffset(String connector, Offset offset) {
            // Buffer offset updates
            pendingOffsets.put(connector, offset);
        }

        private void commitPendingOffsets() {
            if (pendingOffsets.isEmpty()) {
                return;
            }

            // Batch commit all pending offsets
            Map<String, Offset> toCommit = new HashMap<>(pendingOffsets);
            pendingOffsets.clear();

            try {
                offsetStore.commitBatch(toCommit);
                log.debug("Committed {} offsets", toCommit.size());
            } catch (Exception e) {
                log.error("Failed to commit offsets", e);
                // Restore pending offsets
                pendingOffsets.putAll(toCommit);
            }
        }
    }
    ```

    ---

    ## Performance Optimization

    ### Kafka Producer Tuning

    ```properties
    # Increase batch size (reduce syscalls)
    batch.size=65536  # 64 KB

    # Wait for batches to fill
    linger.ms=10  # 10ms

    # Compression (3x reduction)
    compression.type=snappy

    # More memory for buffering
    buffer.memory=67108864  # 64 MB

    # In-flight requests (parallelism)
    max.in.flight.requests.per.connection=5

    # Idempotent producer (deduplication)
    enable.idempotence=true

    # Acknowledgment
    acks=all  # Wait for all replicas
    ```

    ### Kafka Consumer Tuning

    ```properties
    # Fetch more data per request
    fetch.min.bytes=1048576  # 1 MB
    fetch.max.wait.ms=500

    # Process more records per poll
    max.poll.records=500

    # Consumer session timeout
    session.timeout.ms=30000
    heartbeat.interval.ms=3000

    # Offset commit strategy
    enable.auto.commit=false  # Manual commit
    auto.offset.reset=earliest
    ```

    ---

    ## Monitoring Metrics

    **Key Metrics:**

    ```yaml
    # CDC Connector metrics
    cdc_binlog_lag_seconds: Time behind binlog (0-60s normal, >300s critical)
    cdc_events_processed_total: Total events processed
    cdc_events_per_second: Event processing rate (target: 100K/sec)
    cdc_snapshot_duration_seconds: Snapshot completion time
    cdc_schema_changes_total: DDL changes detected

    # Kafka metrics
    kafka_producer_record_send_rate: Events published/sec
    kafka_consumer_lag: Consumer lag per partition (target: <1000)
    kafka_log_size_bytes: Topic size

    # Sink metrics
    sink_write_latency_seconds: Latency to write to sink (p95 < 1s)
    sink_error_rate: Failed writes (target: <0.1%)

    # Alerts
    - alert: HighCDCLag
      expr: cdc_binlog_lag_seconds > 300
      annotations:
        summary: "CDC lag > 5 minutes"

    - alert: SnapshotStuck
      expr: cdc_snapshot_duration_seconds > 10800  # 3 hours
      annotations:
        summary: "Snapshot taking too long"

    - alert: SchemaConflict
      expr: rate(cdc_schema_evolution_errors[5m]) > 0
      annotations:
        summary: "Schema evolution failed"
    ```

    **Grafana Dashboard:**

    ```
    Row 1: Overview
    - Total events/sec (gauge)
    - CDC lag (graph, per table)
    - Error rate (graph)

    Row 2: Connector Health
    - Connector status (table)
    - Binlog position (table)
    - Schema versions (table)

    Row 3: Kafka Health
    - Producer throughput (graph)
    - Consumer lag (heatmap)
    - Topic size (graph)

    Row 4: Sink Performance
    - Sink write latency (histogram)
    - Sink error rate (graph)
    - Data freshness (gauge)
    ```

    ---

    ## Cost Optimization

    ```
    Monthly cost (1,000 tables, 100K writes/sec):

    Compute:
    - CDC connectors: 10 nodes × $100 = $1,000
    - Kafka brokers: 20 nodes × $200 = $4,000
    - Kafka Connect: 10 nodes × $100 = $1,000
    - Schema Registry: 3 nodes × $50 = $150
    - Monitoring: 8 nodes × $50 = $400
    - Total compute: $6,550/month

    Storage:
    - Kafka (20 TB compacted): 20 × $23 (S3) = $460
    - Snapshots (25 TB): 25 × $23 = $575
    - Binlog backup (10 TB): 10 × $23 = $230
    - Total storage: $1,265/month

    Network:
    - Egress (10 Gbps × 30 days): 100 TB × $0.09/GB = $9,000

    Total: ~$17K/month

    Optimizations:
    1. Use Kafka log compaction: -$10K storage (keep only latest)
    2. Compress binlog: -$150 storage (3:1 compression)
    3. Deduplicate before Kafka: -$3K network (filter 30% duplicates)
    4. Spot instances for snapshot workers: -$500 compute (70% discount)

    Optimized total: ~$13K/month (24% reduction)
    ```

    ---

    ## Disaster Recovery

    **Backup Strategy:**

    ```
    Tier 1: Kafka topics (primary)
    - Retention: 7 days
    - Replication: 3x within region
    - Recovery: Replay from offset

    Tier 2: S3 backup (secondary)
    - Hourly snapshots to S3
    - Retention: 30 days
    - Recovery: Restore from S3, resume CDC

    Tier 3: Cross-region replication (tertiary)
    - MirrorMaker 2 replication
    - RPO: <1 minute
    - RTO: <5 minutes
    ```

    **Failure Scenarios:**

    ```
    1. CDC Connector crash:
       - Kafka Connect detects failure (heartbeat timeout)
       - Restart connector on different worker
       - Resume from last committed offset
       - No data loss (at-least-once)

    2. Kafka broker crash:
       - Partition leader election (< 1 second)
       - Producers/consumers reconnect
       - Continue processing

    3. Database crash:
       - CDC connector retries connection
       - Exponential backoff (1s, 2s, 4s, ...)
       - Resume from last binlog position after recovery

    4. Schema Registry crash:
       - Schema cache prevents immediate failure
       - Failover to replica (< 5 seconds)
       - Continue processing with cached schemas
    ```

=== "💡 Step 5: Interview Tips"

    ## Common Follow-up Questions

    ### Q1: Log-based CDC vs. Query-based CDC?

    ```
    Log-based CDC (Binlog/WAL):
    Pros:
    - Non-intrusive (< 5% overhead)
    - Captures deletes
    - Low latency (< 1 second)
    - No source schema changes

    Cons:
    - Complex implementation
    - Database-specific
    - Requires permissions (REPLICATION)

    Query-based CDC (Polling):
    Pros:
    - Simple implementation
    - Works with any database
    - No special permissions

    Cons:
    - High source load (10-20% overhead)
    - Cannot capture deletes
    - Higher latency (poll interval)
    - Requires updated_at column

    Recommendation: Use log-based for production, query-based for dev/test
    ```

    ### Q2: How do you handle schema evolution without downtime?

    ```
    Answer:

    1. Use Schema Registry with compatibility modes:
       - BACKWARD: New schema can read old data
       - FORWARD: Old schema can read new data
       - FULL: Both BACKWARD and FORWARD

    2. Safe schema changes:
       - ADD COLUMN with default value: ✓ (backward compatible)
       - DROP COLUMN: ✓ (forward compatible, deprecate first)
       - RENAME COLUMN: ✗ (breaking change, use alias)
       - CHANGE TYPE: ✗ (breaking change, add new column)

    3. Deployment strategy:
       - Deploy sink consumers first (can read new schema)
       - Then deploy CDC connectors (produce new schema)
       - Gradual rollout (canary deployment)

    4. Fallback:
       - Keep old schema version for 7 days
       - Allow sinks to downgrade if needed
    ```

    ### Q3: How do you achieve exactly-once delivery in CDC?

    ```
    Answer:

    CDC provides at-least-once (duplicates possible):
    1. Source: Read each binlog event exactly once
    2. Kafka: Write to topic (at-least-once due to retries)
    3. Sink: Consume from Kafka (at-least-once due to rebalancing)

    To achieve exactly-once:

    Option 1: Idempotent sinks (preferred)
    - Use UPSERT/MERGE (not INSERT)
    - Use primary key or unique constraint
    - Duplicates overwrite with same value

    Option 2: Transactional sinks
    - Use Kafka transactions (slow, complex)
    - 2-phase commit with sink database
    - Guarantee exactly-once end-to-end

    Option 3: Deduplication window
    - Keep sliding window of processed keys (1 hour)
    - Discard duplicates within window
    - Trade-off: Memory usage

    Recommendation: Use idempotent sinks (UPSERT) - simplest and most reliable
    ```

    ### Q4: How do you minimize impact on source database?

    ```
    Answer:

    1. Use binlog (not triggers):
       - Triggers: 10-20% overhead per write
       - Binlog: <5% overhead (read-only)

    2. Read from replica:
       - Snapshot from read replica
       - Binlog from primary (required)
       - Reduces primary load

    3. Throttle snapshot:
       - Read in small chunks (10K rows)
       - Sleep between chunks (100ms)
       - Run during off-peak hours

    4. Use incremental snapshot:
       - Lock-free snapshot algorithm
       - No table locks
       - Concurrent binlog tailing

    5. Connection pooling:
       - Reuse connections
       - Limit concurrent connections (1-2 per table)

    Monitoring:
    - Watch database CPU, memory, I/O
    - Alert if overhead > 5%
    ```

    ### Q5: How do you handle large tables (1B+ rows)?

    ```
    Answer:

    1. Incremental snapshot:
       - Chunk by primary key range
       - Read 10K rows per chunk
       - Parallel snapshot (10 workers)
       - Duration: 1B rows / (10 workers × 1K rows/sec) = 28 hours

    2. Schema-only snapshot:
       - Skip initial snapshot
       - Only capture changes going forward
       - Trade-off: Missing historical data

    3. Partition-based snapshot:
       - If table is partitioned, snapshot one partition at a time
       - Example: Snapshot 2024 partition, then 2025 partition
       - Reduces locking and memory

    4. External snapshot:
       - Use database native export (mysqldump, pg_dump)
       - Load to S3/data lake
       - CDC for incremental changes only

    5. Cold storage offload:
       - Move old data to cold storage (S3)
       - CDC only for hot data (recent partitions)
    ```

    ---

    ## Real-World Examples

    **Debezium at Netflix:**
    ```
    Use case: Real-time analytics on user viewing data

    Architecture:
    - 500+ MySQL databases
    - 10,000+ tables
    - 1M writes/sec (peak)
    - < 500ms latency (DB → data warehouse)

    Tech stack:
    - Debezium for CDC
    - Kafka for streaming
    - Delta Lake for storage
    - Presto for queries

    Lessons learned:
    - Schema evolution is critical (100+ DDL changes/day)
    - Monitor lag per table (alert if > 1 minute)
    - Use log compaction (reduces storage 90%)
    ```

    **AWS DMS at Airbnb:**
    ```
    Use case: Multi-region database replication

    Architecture:
    - PostgreSQL primary (US-East)
    - PostgreSQL replica (EU-West)
    - 200+ tables
    - 50K writes/sec

    Tech stack:
    - AWS DMS for CDC
    - S3 for staging
    - Redshift for analytics

    Challenges:
    - Large transactions (batch bookings)
    - Schema changes during migration
    - Data consistency during cutover

    Solutions:
    - Split large transactions (<10K rows per batch)
    - Blue-green deployment for cutover
    - Compare checksums (source vs. target)
    ```

    ---

    ## Key Takeaways

    1. **Log-based CDC > Query-based:** Lower overhead, captures deletes, low latency
    2. **Initial snapshot + incremental:** Full table copy, then stream changes
    3. **Schema evolution:** Use Schema Registry with compatibility checks
    4. **At-least-once delivery:** Idempotent sinks (UPSERT) handle duplicates
    5. **Offset management:** Track binlog position, LSN, GTID
    6. **Minimize source impact:** < 5% overhead (use binlog, not triggers)
    7. **Monitor lag:** Alert if > 1 minute behind
    8. **Handle failures:** Retry with exponential backoff, resume from offset

    ---

    ## Additional Resources

    - [Debezium Documentation](https://debezium.io/documentation/)
    - [AWS Database Migration Service](https://aws.amazon.com/dms/)
    - [Martin Kleppmann: Change Data Capture](https://www.confluent.io/blog/change-data-capture-with-debezium-and-kafka/)
    - [Designing Data-Intensive Applications (Chapter 11)](https://dataintensive.net/)
    - [MySQL Binlog Format](https://dev.mysql.com/doc/refman/8.0/en/replication-formats.html)
    - [PostgreSQL Logical Replication](https://www.postgresql.org/docs/current/logical-replication.html)
