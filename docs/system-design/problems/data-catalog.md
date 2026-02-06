# Design a Data Catalog System

A centralized metadata management platform that enables discovery, understanding, and governance of data assets across an organization through automated metadata ingestion, powerful search capabilities, data lineage visualization, intelligent tagging, and access control management.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100K+ datasets, 10K daily searches, 1M+ metadata entities, multi-PB data landscape |
| **Key Challenges** | Metadata extraction at scale, search relevance ranking, lineage graph traversal, schema evolution tracking, access control integration |
| **Core Concepts** | Metadata management, search and discovery, data lineage, automated classification, schema registry, usage analytics |
| **Companies** | DataHub, Amundsen, Collibra, Alation, Apache Atlas, Google Data Catalog, AWS Glue Catalog |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Metadata Ingestion** | Extract metadata from SQL/NoSQL/S3/APIs | P0 (Must have) |
    | **Search & Discovery** | Full-text search with ranking and filtering | P0 (Must have) |
    | **Data Lineage** | Track data flow from source to destination | P0 (Must have) |
    | **Schema Management** | Track schema changes and evolution | P0 (Must have) |
    | **Tagging & Classification** | Business glossary, technical tags, PII detection | P0 (Must have) |
    | **Access Control** | RBAC, data ownership, request workflows | P0 (Must have) |
    | **Usage Analytics** | Track queries, popular datasets, user activity | P1 (Should have) |
    | **Data Profiling** | Statistics, data quality metrics | P1 (Should have) |
    | **Documentation** | Descriptions, README, field definitions | P1 (Should have) |
    | **Recommendations** | Suggest similar datasets, related tables | P2 (Nice to have) |
    | **Data Preview** | Sample data, query interface | P2 (Nice to have) |
    | **Alerts & Notifications** | Schema changes, ownership changes | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Data quality validation (use Great Expectations/Soda)
    - ETL pipeline orchestration (use Airflow/Dagster)
    - Data transformation (use dbt/Spark)
    - Master data management (MDM)
    - Real-time streaming metadata (focus on batch)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Search Latency** | < 200ms for search queries | Fast user experience |
    | **Ingestion Throughput** | 1,000+ metadata updates/sec | Handle large-scale ingestion |
    | **Availability** | 99.9% uptime | Critical for data discovery |
    | **Freshness** | < 1 hour metadata lag | Near real-time updates |
    | **Scalability** | 100K+ datasets, 1M+ entities | Enterprise-scale support |
    | **Search Relevance** | > 80% relevant results in top 10 | High-quality search results |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Dataset scale:
    - Total datasets: 100,000 (tables, files, dashboards, APIs)
    - Average metadata entities per dataset: 10 (columns, tags, owners)
    - Total metadata entities: 1,000,000
    - New datasets per day: 500
    - Schema changes per day: 2,000

    Search traffic:
    - Daily active users: 5,000 data professionals
    - Searches per user per day: 2
    - Total searches: 10,000 searches/day
    - Average search QPS: 10,000 / 86,400 = 0.12 QPS
    - Peak QPS (business hours): 5-10 QPS

    Metadata ingestion:
    - Full crawl frequency: Weekly (100K datasets)
    - Incremental updates: Every hour (500 datasets)
    - Crawl rate: 100K / (7 * 24 * 3600) = 0.17 datasets/sec
    - Peak ingestion: 10 datasets/sec (during full crawl)

    Lineage queries:
    - Lineage lookups per day: 1,000
    - Average lineage depth: 5 hops
    - Graph traversal time: 100ms

    Usage tracking:
    - Query logs ingested per day: 1M queries (from warehouses)
    - User activity events: 50K events/day
    ```

    ### Storage Estimates

    ```
    Metadata storage:

    Dataset metadata:
    - 100,000 datasets × 10 KB = 1 GB
    - Fields: name, description, location, schema, owner, tags, created_at

    Column metadata:
    - 100,000 datasets × 50 columns × 2 KB = 10 GB
    - Fields: name, type, description, statistics, tags

    Lineage graph:
    - 100,000 datasets × 5 dependencies = 500K edges
    - Per edge: 1 KB (source, target, transformation, job_id)
    - Total: 500 MB

    Schema history:
    - 100,000 datasets × 52 versions/year = 5.2M schema versions
    - Per version: 5 KB (schema snapshot, diff, timestamp)
    - Total: 26 GB/year
    - With 2-year retention: 52 GB

    Tags and classifications:
    - 100,000 datasets × 10 tags × 100 bytes = 100 MB
    - Tag definitions: 1,000 tags × 1 KB = 1 MB

    Usage statistics:
    - 1M queries/day × 365 days = 365M query logs/year
    - Per log: 500 bytes (query, user, dataset, timestamp)
    - Total: 182.5 GB/year
    - With 1-year retention: 182.5 GB

    Search index (Elasticsearch):
    - Primary data: 11 GB (metadata + columns)
    - Index overhead (3x): 33 GB
    - Total: 44 GB

    Total metadata storage:
    - Relational data: 1 GB + 10 GB + 52 GB + 0.1 GB + 182.5 GB = 245.6 GB
    - Graph data: 500 MB
    - Search index: 44 GB
    - Total: ~290 GB
    With replication (3x): ~870 GB
    ```

    ### Bandwidth Estimates

    ```
    Metadata ingestion:
    - Full crawl: 100K datasets × 10 KB / (7 days) = 143 MB/day = 1.7 KB/sec
    - Incremental: 500 datasets × 10 KB / hour = 5 MB/hour = 1.4 KB/sec
    - Total ingestion: ~3 KB/sec (negligible)

    Search queries:
    - 10 searches/sec × 50 KB result = 500 KB/sec = 4 Mbps
    - Peak (10x): 40 Mbps

    Lineage queries:
    - 1,000 queries/day × 100 KB result = 100 MB/day = 1.2 KB/sec (negligible)

    Usage logs ingestion:
    - 1M logs/day × 500 bytes = 500 MB/day = 5.8 KB/sec (negligible)
    ```

    ### Server Estimates

    ```
    Search cluster (Elasticsearch):
    - 3 nodes (data + master)
    - CPU: 16 cores per node (search query processing)
    - Memory: 64 GB per node (index caching)
    - Storage: 200 GB SSD per node

    Metadata database (PostgreSQL):
    - Primary: 1 node (32 cores, 128 GB RAM, 500 GB SSD)
    - Read replicas: 2 nodes
    - Total: 3 database nodes

    Graph database (Neo4j):
    - 3 nodes for lineage graph
    - CPU: 16 cores per node
    - Memory: 64 GB per node
    - Storage: 100 GB SSD per node

    Metadata ingestion workers:
    - 10 worker nodes
    - CPU: 8 cores per node (crawling, parsing)
    - Memory: 32 GB per node

    API servers:
    - 5 nodes (web UI + REST API)
    - CPU: 8 cores per node
    - Memory: 32 GB per node
    - Load balanced

    ML service (classification):
    - 2 nodes (PII detection, auto-tagging)
    - CPU: 16 cores per node
    - GPU: Optional for deep learning models

    Message queue (Kafka):
    - 3 nodes for metadata events
    - CPU: 8 cores per node
    - Memory: 32 GB per node

    Total infrastructure:
    - Search: 3 nodes
    - Databases: 3 + 3 = 6 nodes
    - Ingestion workers: 10 nodes
    - API servers: 5 nodes
    - ML service: 2 nodes
    - Message queue: 3 nodes
    - Total: ~29 nodes
    ```

    ---

    ## Key Assumptions

    1. 100,000 datasets monitored across organization
    2. 10,000 searches per day (5,000 active users)
    3. 500 new datasets added daily
    4. 2,000 schema changes per day
    5. Weekly full metadata crawl
    6. 1M query logs ingested daily
    7. Average dataset has 50 columns
    8. 2-year retention for schema history

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Metadata as first-class citizen:** Treat metadata with same care as data
    2. **Push and pull ingestion:** Support both scheduled crawls and event-driven updates
    3. **Search-first design:** Optimize for fast, relevant search results
    4. **Graph-based lineage:** Leverage graph database for relationship queries
    5. **Automated classification:** Use ML to detect PII, infer tags
    6. **Open metadata standards:** Support OpenMetadata, DataHub schemas

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Data Sources"
            SQL[(SQL Databases<br/>MySQL, Postgres<br/>Snowflake)]
            NoSQL[(NoSQL<br/>MongoDB, Cassandra<br/>DynamoDB)]
            S3Source[Object Storage<br/>S3, GCS<br/>Azure Blob]
            BI[BI Tools<br/>Tableau, Looker<br/>Power BI]
            ETL[ETL Systems<br/>Airflow, dbt<br/>Fivetran]
        end

        subgraph "Metadata Ingestion Layer"
            Connectors[Source Connectors<br/>JDBC, API clients<br/>S3 scanner]

            subgraph "Crawlers"
                SQLCrawler[SQL Crawler<br/>INFORMATION_SCHEMA<br/>SHOW TABLES]
                S3Crawler[S3 Crawler<br/>List objects<br/>Parquet schema]
                APICrawler[API Crawler<br/>REST/GraphQL<br/>BI tool APIs]
            end

            Scheduler[Ingestion Scheduler<br/>Cron-based<br/>Event-driven]
            EventStream[Event Stream<br/>Kafka<br/>Schema changes<br/>Lineage events]
        end

        subgraph "Processing & Enrichment"
            Parser[Metadata Parser<br/>Extract schema<br/>Normalize format]
            Classifier[ML Classifier<br/>PII detection<br/>Sensitive data<br/>Auto-tagging]
            LineageExtractor[Lineage Extractor<br/>Parse SQL<br/>Extract dependencies]
            StatsCollector[Statistics Collector<br/>Row counts<br/>Column profiles]
        end

        subgraph "Storage Layer"
            MetaDB[(Metadata Store<br/>PostgreSQL<br/>Datasets, schemas<br/>Tags, owners)]
            LineageGraph[(Lineage Graph<br/>Neo4j<br/>Data flow<br/>Relationships)]
            SearchIndex[(Search Index<br/>Elasticsearch<br/>Full-text search<br/>Ranking)]
            SchemaRegistry[(Schema Registry<br/>Versioned schemas<br/>Evolution history)]
            UsageDB[(Usage Store<br/>ClickHouse<br/>Query logs<br/>Analytics)]
        end

        subgraph "API & Services"
            SearchService[Search Service<br/>Query parsing<br/>Relevance ranking<br/>Faceted search]
            LineageService[Lineage Service<br/>Graph traversal<br/>Impact analysis<br/>Root cause]
            MetadataAPI[Metadata API<br/>CRUD operations<br/>REST + GraphQL]
            AccessControl[Access Control<br/>RBAC<br/>Data ownership<br/>Request workflow]
            RecommendationEngine[Recommendation Engine<br/>Similar datasets<br/>Related tables]
        end

        subgraph "User Interface"
            WebUI[Web UI<br/>Search interface<br/>Dataset details<br/>Lineage viz]
            DataPortal[Data Portal<br/>Browse catalog<br/>Request access<br/>Documentation]
            AdminConsole[Admin Console<br/>Manage tags<br/>Configure crawlers<br/>User management]
        end

        subgraph "Integrations"
            Slack[Slack<br/>Search bot<br/>Notifications]
            SSO[SSO/LDAP<br/>Authentication<br/>User sync]
            WarehouseQuery[Query Logs<br/>Snowflake<br/>BigQuery history]
        end

        SQL --> SQLCrawler
        NoSQL --> APICrawler
        S3Source --> S3Crawler
        BI --> APICrawler
        ETL --> EventStream

        Scheduler --> SQLCrawler
        Scheduler --> S3Crawler
        Scheduler --> APICrawler

        SQLCrawler --> Parser
        S3Crawler --> Parser
        APICrawler --> Parser
        EventStream --> Parser

        Parser --> Classifier
        Parser --> LineageExtractor
        Parser --> StatsCollector

        Classifier --> MetaDB
        LineageExtractor --> LineageGraph
        StatsCollector --> MetaDB

        MetaDB --> SearchIndex
        MetaDB --> SchemaRegistry

        WarehouseQuery --> UsageDB

        SearchIndex --> SearchService
        LineageGraph --> LineageService
        MetaDB --> MetadataAPI
        UsageDB --> RecommendationEngine

        SearchService --> WebUI
        LineageService --> WebUI
        MetadataAPI --> WebUI
        AccessControl --> WebUI
        RecommendationEngine --> WebUI

        WebUI --> DataPortal
        DataPortal --> AdminConsole

        MetadataAPI --> Slack
        SSO --> AccessControl

        style SearchIndex fill:#e1f5ff
        style LineageGraph fill:#fff9c4
        style MetaDB fill:#ffe1e1
        style Classifier fill:#e8f5e9
        style SearchService fill:#f3e5f5
        style WebUI fill:#fce4ec
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Elasticsearch for search** | Fast full-text search, relevance ranking, faceted filtering | PostgreSQL (slow full-text), Solr (less cloud-native) |
    | **Neo4j for lineage** | Fast graph traversal, impact analysis, relationship queries | PostgreSQL (slow recursive CTEs), custom graph (complexity) |
    | **PostgreSQL for metadata** | ACID guarantees, complex queries, mature ecosystem | MongoDB (no joins), separate microservices (complexity) |
    | **Kafka for events** | Decoupled ingestion, replay capability, high throughput | Direct writes (tight coupling), SQS (limited ordering) |
    | **ML classification** | Automated PII detection, reduce manual tagging effort | Manual tagging only (doesn't scale), regex rules (brittle) |
    | **ClickHouse for usage** | Fast analytical queries, time-series optimization | PostgreSQL (slower aggregations), Elasticsearch (expensive) |

    **Key Trade-off:** We chose **eventual consistency** for metadata updates (via Kafka) over synchronous writes to enable high ingestion throughput. Search results may be slightly stale (< 1 minute), but this is acceptable for discovery use case.

    ---

    ## API Design

    ### 1. Search Datasets

    **Request:**
    ```bash
    GET /api/v1/search?q=user_events&filters=platform:snowflake,tags:pii&limit=20
    ```

    **Response:**
    ```json
    {
      "query": "user_events",
      "total_results": 15,
      "results": [
        {
          "dataset_id": "snowflake://prod/analytics/user_events",
          "name": "user_events",
          "type": "table",
          "platform": "snowflake",
          "database": "analytics",
          "schema": "public",
          "description": "Event tracking table for user actions",
          "owner": "data-team@company.com",
          "tags": ["pii", "user-data", "production"],
          "last_updated": "2025-02-04T10:30:00Z",
          "row_count": 1250000000,
          "column_count": 35,
          "usage_score": 95,
          "quality_score": 0.92,
          "relevance_score": 0.89,
          "highlighted_fields": {
            "name": "<em>user_events</em>",
            "description": "Event tracking table for <em>user</em> actions"
          }
        },
        {
          "dataset_id": "s3://data-lake/raw/user_events/",
          "name": "user_events",
          "type": "dataset",
          "platform": "s3",
          "location": "s3://data-lake/raw/user_events/",
          "description": "Raw event logs from application",
          "owner": "platform-team@company.com",
          "tags": ["raw", "user-data"],
          "last_updated": "2025-02-05T01:00:00Z",
          "size_bytes": 52428800000,
          "file_count": 15000,
          "usage_score": 78,
          "relevance_score": 0.85
        }
      ],
      "facets": {
        "platform": [
          {"value": "snowflake", "count": 8},
          {"value": "s3", "count": 4},
          {"value": "bigquery", "count": 3}
        ],
        "tags": [
          {"value": "pii", "count": 15},
          {"value": "user-data", "count": 12},
          {"value": "production", "count": 10}
        ],
        "owner": [
          {"value": "data-team@company.com", "count": 6},
          {"value": "platform-team@company.com", "count": 5}
        ]
      },
      "suggestions": [
        "user_sessions",
        "user_profiles",
        "event_stream"
      ]
    }
    ```

    ---

    ### 2. Get Dataset Details

    **Request:**
    ```bash
    GET /api/v1/datasets/snowflake%3A%2F%2Fprod%2Fanalytics%2Fuser_events
    ```

    **Response:**
    ```json
    {
      "dataset_id": "snowflake://prod/analytics/user_events",
      "name": "user_events",
      "type": "table",
      "platform": "snowflake",
      "database": "analytics",
      "schema": "public",
      "description": "Event tracking table for user actions. Updated hourly via ETL pipeline.",
      "owner": {
        "email": "data-team@company.com",
        "name": "Data Platform Team",
        "type": "team"
      },
      "tags": [
        {"name": "pii", "type": "compliance"},
        {"name": "user-data", "type": "domain"},
        {"name": "production", "type": "environment"}
      ],
      "classification": "sensitive",
      "created_at": "2022-06-15T00:00:00Z",
      "last_updated": "2025-02-04T10:30:00Z",
      "last_schema_change": "2024-11-20T14:22:00Z",

      "statistics": {
        "row_count": 1250000000,
        "column_count": 35,
        "size_bytes": 215000000000,
        "partitions": 365
      },

      "columns": [
        {
          "name": "event_id",
          "type": "VARCHAR(255)",
          "nullable": false,
          "primary_key": true,
          "description": "Unique identifier for each event",
          "tags": [],
          "statistics": {
            "unique_count": 1250000000,
            "null_count": 0
          }
        },
        {
          "name": "user_id",
          "type": "BIGINT",
          "nullable": false,
          "foreign_keys": ["analytics.users.user_id"],
          "description": "User who triggered the event",
          "tags": ["pii"],
          "statistics": {
            "unique_count": 45000000,
            "null_count": 0,
            "min": 1,
            "max": 98765432
          }
        },
        {
          "name": "event_type",
          "type": "VARCHAR(100)",
          "nullable": false,
          "description": "Type of event (page_view, click, purchase)",
          "tags": [],
          "statistics": {
            "unique_count": 127,
            "null_count": 0,
            "top_values": [
              {"value": "page_view", "count": 850000000},
              {"value": "click", "count": 300000000},
              {"value": "purchase", "count": 50000000}
            ]
          }
        },
        {
          "name": "timestamp",
          "type": "TIMESTAMP_NTZ",
          "nullable": false,
          "description": "When the event occurred",
          "tags": [],
          "statistics": {
            "min": "2022-06-15T00:00:00Z",
            "max": "2025-02-04T23:59:59Z"
          }
        }
      ],

      "sample_data": [
        {
          "event_id": "evt_abc123",
          "user_id": 12345,
          "event_type": "page_view",
          "timestamp": "2025-02-04T10:30:15Z"
        }
      ],

      "usage": {
        "queries_last_30_days": 5420,
        "unique_users_last_30_days": 234,
        "top_users": [
          {"email": "analyst1@company.com", "query_count": 452},
          {"email": "analyst2@company.com", "query_count": 389}
        ],
        "popular_joins": [
          "analytics.users",
          "analytics.sessions"
        ]
      },

      "quality": {
        "score": 0.92,
        "last_checked": "2025-02-04T12:00:00Z",
        "checks": [
          {"name": "freshness", "status": "passed", "value": "< 1 hour"},
          {"name": "completeness", "status": "passed", "value": "99.8%"},
          {"name": "uniqueness", "status": "passed", "value": "100%"}
        ]
      },

      "access_control": {
        "is_public": false,
        "access_level": "restricted",
        "can_view": true,
        "can_edit": false,
        "can_request_access": true,
        "access_request_approval": "owner"
      }
    }
    ```

    ---

    ### 3. Get Data Lineage

    **Request:**
    ```bash
    GET /api/v1/lineage?dataset=snowflake://prod/analytics/user_events&direction=both&depth=3
    ```

    **Response:**
    ```json
    {
      "dataset": "snowflake://prod/analytics/user_events",
      "lineage": {
        "upstream": [
          {
            "dataset_id": "s3://data-lake/raw/events/",
            "name": "raw_events",
            "platform": "s3",
            "distance": 1,
            "transformation": {
              "job_id": "airflow.etl_user_events",
              "job_type": "airflow_dag",
              "description": "Copy raw events to Snowflake",
              "last_run": "2025-02-04T10:00:00Z"
            }
          },
          {
            "dataset_id": "kafka://prod/events",
            "name": "events_topic",
            "platform": "kafka",
            "distance": 2,
            "transformation": {
              "job_id": "flink.event_processor",
              "job_type": "stream_processing",
              "description": "Stream events to S3"
            }
          },
          {
            "dataset_id": "api://app.company.com/events",
            "name": "application_events",
            "platform": "rest_api",
            "distance": 3,
            "transformation": {
              "job_id": "app.event_emitter",
              "job_type": "application",
              "description": "Application event tracking"
            }
          }
        ],
        "downstream": [
          {
            "dataset_id": "snowflake://prod/analytics/user_daily_metrics",
            "name": "user_daily_metrics",
            "platform": "snowflake",
            "distance": 1,
            "transformation": {
              "job_id": "dbt.user_metrics",
              "job_type": "dbt_model",
              "description": "Aggregate events to daily metrics",
              "sql": "SELECT user_id, DATE(timestamp) as date, COUNT(*) as event_count FROM user_events GROUP BY 1, 2"
            }
          },
          {
            "dataset_id": "tableau://prod/executive_dashboard",
            "name": "executive_dashboard",
            "platform": "tableau",
            "distance": 2,
            "transformation": {
              "job_id": "tableau.dashboard_123",
              "job_type": "visualization",
              "consumers": 150
            }
          }
        ]
      },
      "total_upstream": 8,
      "total_downstream": 15,
      "impact_score": 0.87,
      "criticality": "high"
    }
    ```

    ---

    ### 4. Register Metadata (Push API)

    **Request:**
    ```bash
    POST /api/v1/metadata/register
    Content-Type: application/json

    {
      "dataset_id": "snowflake://prod/analytics/new_table",
      "name": "new_table",
      "type": "table",
      "platform": "snowflake",
      "database": "analytics",
      "schema": "public",
      "description": "Newly created table for analytics",
      "owner": "data-team@company.com",
      "tags": ["analytics", "production"],
      "columns": [
        {
          "name": "id",
          "type": "BIGINT",
          "nullable": false,
          "primary_key": true,
          "description": "Primary key"
        },
        {
          "name": "created_at",
          "type": "TIMESTAMP_NTZ",
          "nullable": false,
          "description": "Record creation time"
        }
      ],
      "upstream_datasets": [
        "snowflake://prod/raw/source_table"
      ],
      "transformation": {
        "job_id": "dbt.new_table_model",
        "sql": "SELECT id, created_at FROM source_table WHERE active = TRUE"
      }
    }
    ```

    **Response:**
    ```json
    {
      "dataset_id": "snowflake://prod/analytics/new_table",
      "status": "registered",
      "version": 1,
      "created_at": "2025-02-05T10:00:00Z",
      "indexed": true
    }
    ```

    ---

    ## Database Schema

    ### Datasets

    ```sql
    CREATE TABLE datasets (
        dataset_id VARCHAR(1000) PRIMARY KEY,
        name VARCHAR(500) NOT NULL,
        type VARCHAR(100) NOT NULL,  -- table, view, dataset, dashboard, ml_model
        platform VARCHAR(100) NOT NULL,  -- snowflake, s3, bigquery, tableau
        database_name VARCHAR(500),
        schema_name VARCHAR(500),
        location VARCHAR(1000),
        description TEXT,
        owner_email VARCHAR(255),
        owner_type VARCHAR(50),  -- user, team, service
        classification VARCHAR(100),  -- public, internal, confidential, sensitive
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        last_crawled_at TIMESTAMP,
        row_count BIGINT,
        size_bytes BIGINT,
        quality_score DECIMAL(5, 4)
    );

    CREATE INDEX idx_datasets_name ON datasets(name);
    CREATE INDEX idx_datasets_platform ON datasets(platform);
    CREATE INDEX idx_datasets_owner ON datasets(owner_email);
    CREATE INDEX idx_datasets_updated ON datasets(updated_at DESC);
    CREATE INDEX idx_datasets_active ON datasets(is_active) WHERE is_active = TRUE;
    ```

    ### Columns

    ```sql
    CREATE TABLE columns (
        column_id BIGSERIAL PRIMARY KEY,
        dataset_id VARCHAR(1000) REFERENCES datasets(dataset_id),
        name VARCHAR(500) NOT NULL,
        data_type VARCHAR(200) NOT NULL,
        nullable BOOLEAN DEFAULT TRUE,
        primary_key BOOLEAN DEFAULT FALSE,
        description TEXT,
        ordinal_position INT,
        statistics JSONB,  -- min, max, unique_count, null_count, top_values
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_columns_dataset ON columns(dataset_id);
    CREATE INDEX idx_columns_name ON columns(name);
    CREATE INDEX idx_columns_stats ON columns USING GIN(statistics);
    ```

    ### Tags

    ```sql
    CREATE TABLE tags (
        tag_id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        type VARCHAR(100) NOT NULL,  -- domain, compliance, environment, technical
        description TEXT,
        color VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE dataset_tags (
        dataset_id VARCHAR(1000) REFERENCES datasets(dataset_id),
        tag_id INT REFERENCES tags(tag_id),
        assigned_by VARCHAR(255),
        assigned_at TIMESTAMP DEFAULT NOW(),
        confidence DECIMAL(5, 4),  -- For ML-assigned tags
        PRIMARY KEY (dataset_id, tag_id)
    );

    CREATE INDEX idx_dataset_tags_dataset ON dataset_tags(dataset_id);
    CREATE INDEX idx_dataset_tags_tag ON dataset_tags(tag_id);
    ```

    ### Schema History

    ```sql
    CREATE TABLE schema_versions (
        version_id BIGSERIAL PRIMARY KEY,
        dataset_id VARCHAR(1000) REFERENCES datasets(dataset_id),
        version_number INT NOT NULL,
        schema_snapshot JSONB NOT NULL,  -- Full schema at this version
        schema_diff JSONB,  -- Changes from previous version
        change_type VARCHAR(100),  -- column_added, column_removed, type_changed
        created_at TIMESTAMP DEFAULT NOW(),
        created_by VARCHAR(255)
    );

    CREATE INDEX idx_schema_versions_dataset ON schema_versions(dataset_id);
    CREATE INDEX idx_schema_versions_created ON schema_versions(created_at DESC);
    CREATE UNIQUE INDEX idx_schema_versions_dataset_version
        ON schema_versions(dataset_id, version_number);
    ```

    ### Data Lineage (Edges for graph)

    ```sql
    CREATE TABLE lineage_edges (
        edge_id BIGSERIAL PRIMARY KEY,
        source_dataset_id VARCHAR(1000) NOT NULL,
        target_dataset_id VARCHAR(1000) NOT NULL,
        lineage_type VARCHAR(100),  -- direct, indirect, derived
        job_id VARCHAR(500),
        job_type VARCHAR(100),  -- airflow_dag, dbt_model, sql_query
        transformation_sql TEXT,
        confidence DECIMAL(5, 4) DEFAULT 1.0,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        UNIQUE(source_dataset_id, target_dataset_id, job_id)
    );

    CREATE INDEX idx_lineage_source ON lineage_edges(source_dataset_id);
    CREATE INDEX idx_lineage_target ON lineage_edges(target_dataset_id);
    CREATE INDEX idx_lineage_job ON lineage_edges(job_id);
    ```

    ### Usage Statistics

    ```sql
    -- ClickHouse schema for usage analytics
    CREATE TABLE query_logs (
        query_id String,
        user_email String,
        dataset_id String,
        query_text String,
        execution_time_ms UInt32,
        rows_returned UInt64,
        timestamp DateTime,
        platform String
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (timestamp, user_email);

    CREATE TABLE dataset_usage_daily (
        dataset_id String,
        date Date,
        query_count UInt32,
        unique_users UInt16,
        total_rows_returned UInt64,
        avg_execution_time_ms UInt32
    ) ENGINE = SummingMergeTree()
    PARTITION BY toYYYYMM(date)
    ORDER BY (dataset_id, date);
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. Metadata Extraction

    **SQL Database Crawler:**

    ```python
    class SQLMetadataCrawler:
        """
        Extract metadata from SQL databases

        Supports:
        - MySQL, PostgreSQL, SQL Server
        - Snowflake, BigQuery, Redshift

        Extracts:
        - Tables, views, columns
        - Data types, constraints
        - Indexes, foreign keys
        """

        def __init__(self, connection_config):
            self.conn = self.create_connection(connection_config)
            self.platform = connection_config['platform']

        def crawl_database(self, database_name):
            """
            Crawl entire database and extract metadata

            Returns metadata for all tables and views
            """
            metadata = {
                'database': database_name,
                'platform': self.platform,
                'datasets': []
            }

            # Get list of tables
            tables = self.get_tables(database_name)

            for table in tables:
                try:
                    table_metadata = self.extract_table_metadata(
                        database_name,
                        table['schema'],
                        table['name']
                    )
                    metadata['datasets'].append(table_metadata)
                except Exception as e:
                    logger.error(f"Failed to extract metadata for {table['name']}: {e}")

            return metadata

        def get_tables(self, database_name):
            """
            Get list of tables using INFORMATION_SCHEMA

            Works across most SQL databases
            """
            if self.platform == 'snowflake':
                query = """
                    SELECT
                        table_catalog as database,
                        table_schema as schema,
                        table_name as name,
                        table_type as type,
                        row_count,
                        bytes
                    FROM information_schema.tables
                    WHERE table_catalog = %s
                      AND table_schema NOT IN ('INFORMATION_SCHEMA', 'PERFORMANCE_SCHEMA')
                    ORDER BY table_schema, table_name
                """
            elif self.platform == 'postgres':
                query = """
                    SELECT
                        table_catalog as database,
                        table_schema as schema,
                        table_name as name,
                        table_type as type
                    FROM information_schema.tables
                    WHERE table_catalog = %s
                      AND table_schema NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY table_schema, table_name
                """

            return self.conn.execute(query, [database_name])

        def extract_table_metadata(self, database, schema, table):
            """
            Extract comprehensive metadata for a single table

            Includes:
            - Basic info (name, type, description)
            - Columns (name, type, nullable, constraints)
            - Statistics (row count, size)
            - Relationships (foreign keys)
            """
            # Get column metadata
            columns = self.get_columns(database, schema, table)

            # Get table statistics
            stats = self.get_table_statistics(database, schema, table)

            # Get foreign keys
            foreign_keys = self.get_foreign_keys(database, schema, table)

            # Get table comment (if available)
            description = self.get_table_comment(database, schema, table)

            # Build dataset ID (unique identifier)
            dataset_id = f"{self.platform}://{database}/{schema}/{table}"

            return {
                'dataset_id': dataset_id,
                'name': table,
                'type': 'table',
                'platform': self.platform,
                'database': database,
                'schema': schema,
                'description': description,
                'columns': columns,
                'statistics': stats,
                'foreign_keys': foreign_keys,
                'crawled_at': datetime.utcnow()
            }

        def get_columns(self, database, schema, table):
            """
            Extract column metadata using INFORMATION_SCHEMA.COLUMNS
            """
            query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    ordinal_position,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    column_comment
                FROM information_schema.columns
                WHERE table_catalog = %s
                  AND table_schema = %s
                  AND table_name = %s
                ORDER BY ordinal_position
            """

            rows = self.conn.execute(query, [database, schema, table])

            columns = []
            for row in rows:
                columns.append({
                    'name': row['column_name'],
                    'data_type': row['data_type'],
                    'nullable': row['is_nullable'] == 'YES',
                    'ordinal_position': row['ordinal_position'],
                    'description': row.get('column_comment'),
                    'max_length': row.get('character_maximum_length'),
                    'precision': row.get('numeric_precision'),
                    'scale': row.get('numeric_scale')
                })

            return columns

        def get_table_statistics(self, database, schema, table):
            """
            Get table-level statistics

            Different queries for different platforms
            """
            if self.platform == 'snowflake':
                query = f"""
                    SELECT
                        COUNT(*) as row_count,
                        SUM(bytes) as size_bytes
                    FROM information_schema.tables
                    WHERE table_catalog = %s
                      AND table_schema = %s
                      AND table_name = %s
                """
            elif self.platform == 'postgres':
                query = f"""
                    SELECT
                        reltuples::BIGINT as row_count,
                        pg_total_relation_size('{schema}.{table}') as size_bytes
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s
                      AND c.relname = %s
                """

            result = self.conn.execute(query, [database, schema, table])

            return {
                'row_count': result[0]['row_count'],
                'size_bytes': result[0]['size_bytes']
            }
    ```

    **S3 Metadata Crawler:**

    ```python
    import boto3
    import pyarrow.parquet as pq

    class S3MetadataCrawler:
        """
        Extract metadata from S3 objects

        Supports:
        - Parquet files (read schema)
        - CSV files (infer schema from sample)
        - JSON files (infer schema)

        Metadata extracted:
        - File location, size, count
        - Schema (columns, types)
        - Partitions (Hive-style: dt=2025-01-15)
        """

        def __init__(self, aws_config):
            self.s3_client = boto3.client('s3', **aws_config)

        def crawl_bucket(self, bucket_name, prefix=''):
            """
            Crawl S3 bucket and discover datasets

            Groups files by common prefix (assumes datasets are in folders)
            """
            datasets = {}

            # List objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']

                    # Identify dataset by parent folder
                    dataset_path = self.get_dataset_path(key)

                    if dataset_path not in datasets:
                        datasets[dataset_path] = {
                            'files': [],
                            'total_size': 0,
                            'partitions': set()
                        }

                    datasets[dataset_path]['files'].append(key)
                    datasets[dataset_path]['total_size'] += obj['Size']

                    # Extract partition info (e.g., dt=2025-01-15)
                    partition = self.extract_partition(key)
                    if partition:
                        datasets[dataset_path]['partitions'].add(partition)

            # Extract schema for each dataset
            metadata_list = []
            for dataset_path, info in datasets.items():
                try:
                    metadata = self.extract_dataset_metadata(
                        bucket_name,
                        dataset_path,
                        info
                    )
                    metadata_list.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to extract schema for {dataset_path}: {e}")

            return metadata_list

        def extract_dataset_metadata(self, bucket, dataset_path, info):
            """
            Extract schema from S3 dataset

            Strategy:
            1. Find first file in dataset
            2. Detect file format (Parquet, CSV, JSON)
            3. Extract schema
            """
            # Get first file
            first_file = info['files'][0]

            # Detect format
            if first_file.endswith('.parquet'):
                schema = self.extract_parquet_schema(bucket, first_file)
            elif first_file.endswith('.csv'):
                schema = self.extract_csv_schema(bucket, first_file)
            elif first_file.endswith('.json'):
                schema = self.extract_json_schema(bucket, first_file)
            else:
                schema = []

            dataset_id = f"s3://{bucket}/{dataset_path}"

            return {
                'dataset_id': dataset_id,
                'name': dataset_path.split('/')[-1],
                'type': 'dataset',
                'platform': 's3',
                'location': f"s3://{bucket}/{dataset_path}",
                'columns': schema,
                'statistics': {
                    'file_count': len(info['files']),
                    'size_bytes': info['total_size'],
                    'partitions': len(info['partitions'])
                },
                'crawled_at': datetime.utcnow()
            }

        def extract_parquet_schema(self, bucket, key):
            """
            Read Parquet schema from S3 file

            Uses pyarrow to read metadata without downloading entire file
            """
            # Download to temp file or use S3 filesystem
            s3_path = f"s3://{bucket}/{key}"

            # Read schema
            parquet_file = pq.ParquetFile(s3_path)
            schema = parquet_file.schema

            columns = []
            for field in schema:
                columns.append({
                    'name': field.name,
                    'data_type': str(field.type),
                    'nullable': field.nullable
                })

            return columns

        def extract_csv_schema(self, bucket, key):
            """
            Infer CSV schema from first 1000 rows
            """
            import pandas as pd
            from io import BytesIO

            # Download first 1MB
            obj = self.s3_client.get_object(
                Bucket=bucket,
                Key=key,
                Range='bytes=0-1048576'
            )

            # Read with pandas
            df = pd.read_csv(BytesIO(obj['Body'].read()), nrows=1000)

            columns = []
            for col in df.columns:
                columns.append({
                    'name': col,
                    'data_type': str(df[col].dtype),
                    'nullable': df[col].isnull().any()
                })

            return columns

        def extract_partition(self, key):
            """
            Extract Hive-style partition from key

            Example: data/dt=2025-01-15/hour=10/file.parquet
            Returns: dt=2025-01-15/hour=10
            """
            parts = key.split('/')
            partition_parts = [p for p in parts if '=' in p]

            if partition_parts:
                return '/'.join(partition_parts)

            return None
    ```

    ---

    ## 2. Search Service with Elasticsearch

    **Search Index Configuration:**

    ```python
    from elasticsearch import Elasticsearch

    class SearchIndexManager:
        """
        Manage Elasticsearch index for data catalog search

        Features:
        - Full-text search across dataset metadata
        - Relevance ranking (BM25 + custom scoring)
        - Faceted filtering (platform, tags, owner)
        - Auto-complete suggestions
        - Search result highlighting
        """

        def __init__(self, es_hosts):
            self.es = Elasticsearch(es_hosts)
            self.index_name = 'datasets'

        def create_index(self):
            """
            Create Elasticsearch index with optimized mappings

            Key features:
            - Text analysis with stemming
            - Keyword fields for exact matching
            - Nested objects for columns
            - Custom analyzers for code/SQL
            """
            mapping = {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 2,
                    "analysis": {
                        "analyzer": {
                            "dataset_name_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "asciifolding"]
                            },
                            "code_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "dataset_id": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "analyzer": "dataset_name_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "suggest": {"type": "completion"}
                            }
                        },
                        "description": {
                            "type": "text",
                            "analyzer": "english"
                        },
                        "platform": {"type": "keyword"},
                        "database": {"type": "keyword"},
                        "schema": {"type": "keyword"},
                        "owner": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "classification": {"type": "keyword"},
                        "columns": {
                            "type": "nested",
                            "properties": {
                                "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                                "data_type": {"type": "keyword"},
                                "description": {"type": "text"}
                            }
                        },
                        "row_count": {"type": "long"},
                        "size_bytes": {"type": "long"},
                        "usage_score": {"type": "float"},
                        "quality_score": {"type": "float"},
                        "last_updated": {"type": "date"},
                        "created_at": {"type": "date"}
                    }
                }
            }

            self.es.indices.create(index=self.index_name, body=mapping)

        def index_dataset(self, dataset_metadata):
            """
            Index dataset metadata to Elasticsearch

            Called after metadata ingestion
            """
            doc = {
                'dataset_id': dataset_metadata['dataset_id'],
                'name': dataset_metadata['name'],
                'description': dataset_metadata.get('description', ''),
                'platform': dataset_metadata['platform'],
                'database': dataset_metadata.get('database'),
                'schema': dataset_metadata.get('schema'),
                'owner': dataset_metadata.get('owner'),
                'tags': dataset_metadata.get('tags', []),
                'classification': dataset_metadata.get('classification'),
                'columns': [
                    {
                        'name': col['name'],
                        'data_type': col['data_type'],
                        'description': col.get('description', '')
                    }
                    for col in dataset_metadata.get('columns', [])
                ],
                'row_count': dataset_metadata.get('statistics', {}).get('row_count'),
                'size_bytes': dataset_metadata.get('statistics', {}).get('size_bytes'),
                'usage_score': dataset_metadata.get('usage_score', 0),
                'quality_score': dataset_metadata.get('quality_score', 1.0),
                'last_updated': dataset_metadata.get('updated_at'),
                'created_at': dataset_metadata.get('created_at')
            }

            self.es.index(
                index=self.index_name,
                id=dataset_metadata['dataset_id'],
                document=doc
            )

        def search(self, query, filters=None, limit=20):
            """
            Search datasets with relevance ranking

            Ranking factors:
            1. Text match score (BM25)
            2. Usage score (popular datasets ranked higher)
            3. Quality score (high-quality datasets preferred)
            4. Freshness (recently updated ranked higher)
            """
            # Build query
            must_clauses = []

            # Multi-field search (name, description, column names)
            if query:
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "name^3",  # Boost name matches 3x
                            "description^2",
                            "columns.name^2",
                            "columns.description"
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                })

            # Apply filters
            filter_clauses = []
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {field: value}})
                    else:
                        filter_clauses.append({"term": {field: value}})

            # Build search body
            search_body = {
                "query": {
                    "function_score": {
                        "query": {
                            "bool": {
                                "must": must_clauses,
                                "filter": filter_clauses
                            }
                        },
                        "functions": [
                            # Boost by usage score
                            {
                                "field_value_factor": {
                                    "field": "usage_score",
                                    "factor": 0.1,
                                    "modifier": "log1p",
                                    "missing": 0
                                }
                            },
                            # Boost by quality score
                            {
                                "field_value_factor": {
                                    "field": "quality_score",
                                    "factor": 0.5,
                                    "modifier": "none",
                                    "missing": 1.0
                                }
                            },
                            # Boost recent datasets
                            {
                                "gauss": {
                                    "last_updated": {
                                        "origin": "now",
                                        "scale": "30d",
                                        "decay": 0.5
                                    }
                                }
                            }
                        ],
                        "score_mode": "sum",
                        "boost_mode": "multiply"
                    }
                },
                "highlight": {
                    "fields": {
                        "name": {},
                        "description": {},
                        "columns.name": {}
                    }
                },
                "aggs": {
                    "platforms": {
                        "terms": {"field": "platform", "size": 20}
                    },
                    "tags": {
                        "terms": {"field": "tags", "size": 50}
                    },
                    "owners": {
                        "terms": {"field": "owner", "size": 20}
                    }
                },
                "size": limit
            }

            # Execute search
            response = self.es.search(index=self.index_name, body=search_body)

            # Parse results
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['relevance_score'] = hit['_score']
                result['highlighted_fields'] = hit.get('highlight', {})
                results.append(result)

            # Parse facets
            facets = {
                'platform': [
                    {'value': b['key'], 'count': b['doc_count']}
                    for b in response['aggregations']['platforms']['buckets']
                ],
                'tags': [
                    {'value': b['key'], 'count': b['doc_count']}
                    for b in response['aggregations']['tags']['buckets']
                ],
                'owners': [
                    {'value': b['key'], 'count': b['doc_count']}
                    for b in response['aggregations']['owners']['buckets']
                ]
            }

            return {
                'total': response['hits']['total']['value'],
                'results': results,
                'facets': facets
            }

        def suggest(self, prefix, limit=10):
            """
            Auto-complete suggestions for dataset names
            """
            body = {
                "suggest": {
                    "dataset-suggest": {
                        "prefix": prefix,
                        "completion": {
                            "field": "name.suggest",
                            "size": limit,
                            "skip_duplicates": True
                        }
                    }
                }
            }

            response = self.es.search(index=self.index_name, body=body)

            suggestions = [
                option['text']
                for option in response['suggest']['dataset-suggest'][0]['options']
            ]

            return suggestions
    ```

    ---

    ## 3. Data Lineage Tracking with Neo4j

    ```python
    from neo4j import GraphDatabase

    class LineageGraph:
        """
        Manage data lineage graph using Neo4j

        Graph model:
        - Nodes: Datasets, Jobs, Users
        - Edges: PRODUCES, CONSUMES, OWNS, READS

        Queries:
        - Upstream lineage (where does this data come from?)
        - Downstream lineage (what depends on this data?)
        - Impact analysis (what breaks if I change this?)
        - Root cause analysis (why is this data stale?)
        """

        def __init__(self, neo4j_uri, auth):
            self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)

        def create_dataset_node(self, dataset_metadata):
            """
            Create or update dataset node in graph
            """
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_dataset_tx,
                    dataset_metadata
                )

        @staticmethod
        def _create_dataset_tx(tx, metadata):
            query = """
            MERGE (d:Dataset {id: $dataset_id})
            SET d.name = $name,
                d.platform = $platform,
                d.type = $type,
                d.updated_at = datetime($updated_at)
            RETURN d
            """

            tx.run(query,
                   dataset_id=metadata['dataset_id'],
                   name=metadata['name'],
                   platform=metadata['platform'],
                   type=metadata['type'],
                   updated_at=metadata.get('updated_at'))

        def create_lineage_edge(self, source_id, target_id, job_id, transformation=None):
            """
            Create lineage relationship between datasets

            Represents: source_dataset PRODUCES target_dataset via job
            """
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_lineage_tx,
                    source_id, target_id, job_id, transformation
                )

        @staticmethod
        def _create_lineage_tx(tx, source_id, target_id, job_id, transformation):
            query = """
            MATCH (source:Dataset {id: $source_id})
            MATCH (target:Dataset {id: $target_id})
            MERGE (source)-[r:PRODUCES {job_id: $job_id}]->(target)
            SET r.transformation = $transformation,
                r.updated_at = datetime()
            RETURN r
            """

            tx.run(query,
                   source_id=source_id,
                   target_id=target_id,
                   job_id=job_id,
                   transformation=transformation)

        def get_upstream_lineage(self, dataset_id, max_depth=10):
            """
            Get all upstream datasets (recursive)

            Traverses PRODUCES edges backwards
            """
            with self.driver.session() as session:
                result = session.read_transaction(
                    self._get_upstream_tx,
                    dataset_id,
                    max_depth
                )
                return result

        @staticmethod
        def _get_upstream_tx(tx, dataset_id, max_depth):
            query = f"""
            MATCH path = (source:Dataset)-[:PRODUCES*1..{max_depth}]->(target:Dataset {{id: $dataset_id}})
            RETURN
                source.id as source_id,
                source.name as source_name,
                source.platform as source_platform,
                target.id as target_id,
                target.name as target_name,
                length(path) as distance,
                [r in relationships(path) | {{
                    job_id: r.job_id,
                    transformation: r.transformation
                }}] as transformations
            ORDER BY distance
            """

            result = tx.run(query, dataset_id=dataset_id)

            upstream = []
            for record in result:
                upstream.append({
                    'source_id': record['source_id'],
                    'source_name': record['source_name'],
                    'source_platform': record['source_platform'],
                    'target_id': record['target_id'],
                    'target_name': record['target_name'],
                    'distance': record['distance'],
                    'transformations': record['transformations']
                })

            return upstream

        def get_downstream_lineage(self, dataset_id, max_depth=10):
            """
            Get all downstream datasets (what depends on this?)

            Traverses PRODUCES edges forward
            """
            with self.driver.session() as session:
                result = session.read_transaction(
                    self._get_downstream_tx,
                    dataset_id,
                    max_depth
                )
                return result

        @staticmethod
        def _get_downstream_tx(tx, dataset_id, max_depth):
            query = f"""
            MATCH path = (source:Dataset {{id: $dataset_id}})-[:PRODUCES*1..{max_depth}]->(target:Dataset)
            RETURN
                source.id as source_id,
                source.name as source_name,
                target.id as target_id,
                target.name as target_name,
                target.platform as target_platform,
                length(path) as distance,
                [r in relationships(path) | {{
                    job_id: r.job_id,
                    transformation: r.transformation
                }}] as transformations
            ORDER BY distance
            """

            result = tx.run(query, dataset_id=dataset_id)

            downstream = []
            for record in result:
                downstream.append({
                    'source_id': record['source_id'],
                    'source_name': record['source_name'],
                    'target_id': record['target_id'],
                    'target_name': record['target_name'],
                    'target_platform': record['target_platform'],
                    'distance': record['distance'],
                    'transformations': record['transformations']
                })

            return downstream

        def compute_impact_score(self, dataset_id):
            """
            Compute impact score based on downstream dependencies

            Higher score = more critical dataset
            """
            with self.driver.session() as session:
                result = session.read_transaction(
                    self._compute_impact_tx,
                    dataset_id
                )
                return result

        @staticmethod
        def _compute_impact_tx(tx, dataset_id):
            query = """
            MATCH (d:Dataset {id: $dataset_id})
            OPTIONAL MATCH (d)-[:PRODUCES*]->(downstream:Dataset)
            WITH d, count(DISTINCT downstream) as downstream_count
            OPTIONAL MATCH (user:User)-[:READS]->(d)
            WITH d, downstream_count, count(DISTINCT user) as user_count
            RETURN
                downstream_count,
                user_count,
                (downstream_count * 0.6 + user_count * 0.4) / 100.0 as impact_score
            """

            result = tx.run(query, dataset_id=dataset_id)
            record = result.single()

            return {
                'downstream_count': record['downstream_count'],
                'user_count': record['user_count'],
                'impact_score': min(1.0, record['impact_score'])
            }
    ```

    ---

    ## 4. Automated Classification with ML

    ```python
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    class AutoClassifier:
        """
        Automated data classification using ML

        Features:
        1. PII Detection (email, SSN, phone, credit card)
        2. Sensitive Data Detection (salary, password, API key)
        3. Domain Classification (user data, financial, operational)
        4. Auto-tagging based on column names and values
        """

        def __init__(self):
            self.pii_patterns = {
                'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            }

            self.sensitive_keywords = {
                'password', 'passwd', 'pwd', 'secret', 'token', 'api_key',
                'access_key', 'private_key', 'ssn', 'social_security',
                'credit_card', 'salary', 'compensation', 'medical'
            }

            # Pre-trained model for domain classification
            self.domain_classifier = None

        def classify_column(self, column_metadata, sample_values=None):
            """
            Classify a single column

            Returns:
            - classification: pii, sensitive, normal
            - tags: detected tags
            - confidence: 0.0 to 1.0
            """
            column_name = column_metadata['name'].lower()
            column_type = column_metadata['data_type'].lower()

            classifications = []
            tags = []
            confidences = []

            # Rule-based classification
            # Check column name for PII keywords
            if any(keyword in column_name for keyword in ['email', 'mail']):
                classifications.append('pii')
                tags.append('email')
                confidences.append(0.95)

            elif any(keyword in column_name for keyword in ['phone', 'mobile', 'tel']):
                classifications.append('pii')
                tags.append('phone')
                confidences.append(0.9)

            elif any(keyword in column_name for keyword in ['ssn', 'social_security']):
                classifications.append('pii')
                tags.append('ssn')
                confidences.append(0.95)

            elif 'user_id' in column_name or 'userid' in column_name:
                classifications.append('pii')
                tags.append('user_id')
                confidences.append(0.8)

            # Check for sensitive data keywords
            if any(keyword in column_name for keyword in self.sensitive_keywords):
                classifications.append('sensitive')
                tags.append('sensitive_data')
                confidences.append(0.9)

            # Pattern-based detection (if sample values provided)
            if sample_values:
                pattern_results = self.detect_patterns(sample_values)
                if pattern_results:
                    classifications.append('pii')
                    tags.extend(pattern_results['tags'])
                    confidences.append(pattern_results['confidence'])

            # Determine final classification
            if classifications:
                # Take most severe classification
                if 'pii' in classifications:
                    final_classification = 'pii'
                elif 'sensitive' in classifications:
                    final_classification = 'sensitive'
                else:
                    final_classification = 'normal'

                avg_confidence = sum(confidences) / len(confidences)
            else:
                final_classification = 'normal'
                avg_confidence = 1.0

            return {
                'classification': final_classification,
                'tags': list(set(tags)),
                'confidence': avg_confidence
            }

        def detect_patterns(self, sample_values):
            """
            Detect PII patterns in sample values using regex

            Sample 1000 values and check for patterns
            """
            if not sample_values:
                return None

            # Convert to strings
            values = [str(v) for v in sample_values[:1000]]

            # Check each pattern
            for pattern_name, pattern_regex in self.pii_patterns.items():
                matches = sum(1 for v in values if re.search(pattern_regex, v))
                match_ratio = matches / len(values)

                # If >50% of values match pattern, classify as that PII type
                if match_ratio > 0.5:
                    return {
                        'tags': [pattern_name],
                        'confidence': match_ratio
                    }

            return None

        def classify_dataset(self, dataset_metadata, sample_data=None):
            """
            Classify entire dataset

            Returns:
            - dataset_classification: highest severity from columns
            - column_classifications: per-column results
            - suggested_tags: dataset-level tags
            """
            column_classifications = []

            for column in dataset_metadata.get('columns', []):
                # Get sample values for this column
                sample_values = None
                if sample_data:
                    sample_values = [row.get(column['name']) for row in sample_data]

                classification = self.classify_column(column, sample_values)
                column_classifications.append({
                    'column_name': column['name'],
                    **classification
                })

            # Determine dataset-level classification
            has_pii = any(c['classification'] == 'pii' for c in column_classifications)
            has_sensitive = any(c['classification'] == 'sensitive' for c in column_classifications)

            if has_pii:
                dataset_classification = 'sensitive'
                suggested_tags = ['pii']
            elif has_sensitive:
                dataset_classification = 'confidential'
                suggested_tags = ['sensitive_data']
            else:
                dataset_classification = 'internal'
                suggested_tags = []

            # Add domain tags based on dataset name
            dataset_name = dataset_metadata['name'].lower()
            if 'user' in dataset_name or 'customer' in dataset_name:
                suggested_tags.append('user-data')
            if 'transaction' in dataset_name or 'payment' in dataset_name:
                suggested_tags.append('financial')
            if 'event' in dataset_name or 'log' in dataset_name:
                suggested_tags.append('events')

            return {
                'dataset_classification': dataset_classification,
                'column_classifications': column_classifications,
                'suggested_tags': list(set(suggested_tags))
            }
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Distributed Metadata Ingestion

    ```
    Horizontal scaling of crawlers:
    - 10+ crawler workers running in parallel
    - Each worker assigned subset of data sources
    - Coordination via distributed queue (Kafka)
    - Incremental crawls (only changed datasets)
    - Avoid full crawls during business hours

    Optimization techniques:
    - Connection pooling to data sources
    - Parallel column metadata extraction
    - Batch writes to metadata store (1000 datasets)
    - Async indexing to Elasticsearch
    - Schema comparison to detect changes only
    ```

    ### Search Performance Tuning

    ```python
    class SearchOptimizations:
        """
        Elasticsearch optimizations for low-latency search

        Techniques:
        1. Query caching (common searches cached)
        2. Shard allocation (distribute load)
        3. Result pagination with search_after
        4. Index aliases (zero-downtime reindexing)
        """

        def optimize_search_query(self, query, filters):
            """
            Apply query optimizations

            - Use filter context for exact matches (cached)
            - Use query context for text search (scored)
            - Limit result size
            """
            return {
                "query": {
                    "bool": {
                        # Query context (scored, not cached)
                        "must": [
                            {"multi_match": {"query": query, "fields": ["name", "description"]}}
                        ],
                        # Filter context (not scored, cached)
                        "filter": [
                            {"term": {"platform": filters.get('platform')}},
                            {"terms": {"tags": filters.get('tags', [])}}
                        ]
                    }
                },
                "size": 20,  # Limit results
                "_source": ["dataset_id", "name", "platform", "description"],  # Fetch only needed fields
                "track_total_hits": 10000  # Don't count beyond 10K (faster)
            }
    ```

    ### Lineage Graph Optimization

    ```
    Neo4j performance tuning:
    - Index on dataset_id for fast lookups
    - Limit lineage depth to 10 hops (prevent long queries)
    - Cache common lineage queries (Redis)
    - Pre-compute impact scores (materialized)
    - Use Cypher query profiling (EXPLAIN)

    Graph partitioning:
    - Separate graphs for different environments (prod, dev)
    - Archive old lineage edges (> 1 year)
    - Denormalize hot paths (frequently queried lineage)
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Incremental crawls** | 95% reduction in crawl time | Complexity in change detection |
    | **Search query caching** | 10x faster repeated searches | Stale results (1 min cache TTL) |
    | **Batch metadata writes** | 100x higher ingestion throughput | Eventual consistency (1 sec delay) |
    | **Lineage caching** | 50x faster lineage queries | May show outdated relationships |
    | **Async indexing** | Decouples ingestion from search | Search index lag (< 1 min) |
    | **Column-level partitioning** | 10x faster column queries | More complex data model |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (100K datasets, 10K searches/day):

    Compute:
    - 3 Elasticsearch nodes × $500 = $1,500
    - 3 PostgreSQL nodes × $400 = $1,200
    - 3 Neo4j nodes × $400 = $1,200
    - 10 crawler workers × $100 = $1,000
    - 5 API servers × $100 = $500
    - 2 ML service nodes × $200 = $400
    - 3 Kafka nodes × $100 = $300
    - Total compute: $6,100/month

    Storage:
    - Metadata (900 GB): 900 GB × $0.10/GB (SSD) = $90
    - Search index (44 GB): Included in ES nodes
    - Graph (500 MB): Included in Neo4j nodes
    - Usage logs (200 GB): 200 GB × $0.02/GB (HDD) = $4
    - Total storage: $94/month

    Network:
    - Minimal (mostly internal traffic): ~$50/month

    Total: ~$6,244/month

    Optimizations:
    1. Reserved instances (1-year): -30% compute = -$1,830
    2. S3 for cold metadata (archive old schemas): -$50
    3. Compress search index (reduce size 50%): -$250
    4. Auto-scale crawlers (only during crawls): -$500
    5. Use Aurora serverless for metadata DB: -$400

    Optimized Total: ~$3,214/month (48% reduction)
    ```

    ---

    ## Monitoring Metrics

    ```python
    # Key metrics for data catalog health

    # Ingestion metrics
    metadata_crawl_duration_seconds{platform, crawler_type}
    metadata_crawl_datasets_count{platform, status}  # status: success, failed
    metadata_updates_per_second
    schema_changes_detected_count

    # Search metrics
    search_query_latency_seconds{type}  # type: full-text, filter, suggest
    search_query_count{status}  # status: success, error
    search_results_count{query}
    search_cache_hit_ratio

    # Lineage metrics
    lineage_query_latency_seconds{direction}  # direction: upstream, downstream
    lineage_depth_max{dataset_id}
    lineage_edge_count

    # Usage metrics
    dataset_views_count{dataset_id}
    dataset_search_rank{dataset_id}  # How often in top 10 results
    unique_users_daily

    # Classification metrics
    auto_classification_accuracy
    pii_detection_count{type}  # type: email, phone, ssn
    classification_confidence_avg

    # System metrics
    elasticsearch_cluster_health
    neo4j_query_latency_seconds
    postgres_connection_pool_size
    kafka_consumer_lag_seconds
    ```

=== "💡 Step 5: Additional Considerations"

    ## Integration with Data Quality

    ```python
    class QualityIntegration:
        """
        Integrate data quality metrics into catalog

        Sources:
        - Great Expectations results
        - dbt test results
        - Custom quality checks

        Display quality score on search results
        """

        def enrich_with_quality_metrics(self, dataset_id):
            """
            Fetch quality metrics from data quality platform

            Display in catalog UI
            """
            quality_metrics = self.quality_platform.get_metrics(dataset_id)

            return {
                'quality_score': quality_metrics['overall_score'],
                'last_checked': quality_metrics['last_run'],
                'checks': quality_metrics['check_results'],
                'issues': quality_metrics['active_issues']
            }
    ```

    ---

    ## Access Control Integration

    ```python
    class AccessControlManager:
        """
        Manage data access permissions

        Features:
        - RBAC (Role-Based Access Control)
        - Data ownership
        - Access request workflow
        - Audit logging
        """

        def check_access(self, user_id, dataset_id, action='read'):
            """
            Check if user has permission to access dataset

            Hierarchy:
            1. Owner: Full access
            2. Team members: Read/write
            3. Others: Request access
            """
            dataset = self.metadata_store.get_dataset(dataset_id)

            # Check ownership
            if dataset['owner_email'] == user_id:
                return True

            # Check team membership (from LDAP/SSO)
            owner_team = self.get_team(dataset['owner_email'])
            user_teams = self.get_user_teams(user_id)

            if owner_team in user_teams:
                return True

            # Check explicit grants
            grants = self.metadata_store.get_access_grants(dataset_id)
            if user_id in grants:
                return True

            return False

        def request_access(self, user_id, dataset_id, justification):
            """
            Submit access request

            Workflow:
            1. Create access request
            2. Notify dataset owner
            3. Owner approves/denies
            4. Grant access if approved
            """
            request = {
                'request_id': str(uuid.uuid4()),
                'user_id': user_id,
                'dataset_id': dataset_id,
                'justification': justification,
                'status': 'pending',
                'created_at': datetime.utcnow()
            }

            self.metadata_store.create_access_request(request)

            # Notify owner
            dataset = self.metadata_store.get_dataset(dataset_id)
            self.send_notification(
                recipient=dataset['owner_email'],
                subject=f"Access request for {dataset['name']}",
                body=f"{user_id} requested access: {justification}"
            )

            return request['request_id']
    ```

    ---

    ## Usage Analytics

    ```python
    class UsageAnalytics:
        """
        Track dataset usage patterns

        Metrics:
        - Query frequency
        - Popular datasets
        - User activity
        - Join patterns

        Used for:
        - Search ranking (boost popular datasets)
        - Recommendations (suggest related datasets)
        - ROI calculation (justify data investments)
        """

        def ingest_query_log(self, query_log):
            """
            Ingest query logs from data warehouses

            Extract:
            - Queried tables
            - User who ran query
            - Query timestamp
            - Query duration
            """
            # Parse SQL to extract table references
            tables = self.extract_tables_from_sql(query_log['query_text'])

            for table in tables:
                self.usage_store.record_query(
                    dataset_id=table,
                    user_id=query_log['user_email'],
                    timestamp=query_log['timestamp'],
                    query_duration_ms=query_log['execution_time_ms']
                )

        def compute_usage_score(self, dataset_id, time_window_days=30):
            """
            Compute usage score (0-100)

            Factors:
            - Query frequency
            - Unique users
            - Recency
            """
            stats = self.usage_store.get_usage_stats(
                dataset_id,
                start_date=datetime.utcnow() - timedelta(days=time_window_days)
            )

            query_count = stats['query_count']
            unique_users = stats['unique_users']

            # Normalize to 0-100
            # Assume 1000+ queries = score 100
            query_score = min(100, (query_count / 1000) * 100)

            # Assume 50+ users = score 100
            user_score = min(100, (unique_users / 50) * 100)

            # Weighted average
            usage_score = (query_score * 0.6) + (user_score * 0.4)

            return int(usage_score)

        def recommend_datasets(self, user_id, limit=10):
            """
            Recommend datasets based on usage patterns

            Algorithm:
            - Collaborative filtering (users who queried A also queried B)
            - Popular in user's team
            - Similar to recently viewed datasets
            """
            # Get user's recent queries
            recent = self.usage_store.get_user_queries(user_id, days=7)
            recent_datasets = [q['dataset_id'] for q in recent]

            # Find datasets commonly queried together
            recommendations = self.usage_store.get_cooccurrence(
                recent_datasets,
                limit=limit
            )

            return recommendations
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you keep metadata fresh?"**
   - Scheduled crawls (weekly full, hourly incremental)
   - Event-driven updates (Kafka events from ETL pipelines)
   - Push API (teams register metadata directly)
   - Schema change detection (compare with previous crawl)
   - Staleness monitoring (alert if > 24 hours old)

2. **"How do you handle schema evolution?"**
   - Store schema history (version control for schemas)
   - Detect breaking changes (column removed, type changed)
   - Alert downstream consumers on breaking changes
   - Schema registry with versioning
   - Backward compatibility checks

3. **"How do you rank search results?"**
   - Text relevance (BM25 algorithm in Elasticsearch)
   - Usage score (popular datasets ranked higher)
   - Quality score (high-quality data preferred)
   - Freshness (recent updates boosted)
   - User context (personalization based on past queries)

4. **"How do you extract lineage automatically?"**
   - Parse SQL queries (extract table references)
   - ETL tool integration (Airflow, dbt metadata APIs)
   - Query log analysis (infer lineage from actual queries)
   - Manual registration (push API)
   - Confidence scoring (higher for explicit vs inferred)

5. **"How do you detect PII automatically?"**
   - Pattern matching (regex for email, phone, SSN)
   - Column name analysis (keywords like "email", "ssn")
   - Sample data analysis (check values for patterns)
   - ML classification (trained on labeled datasets)
   - Confidence scores (flag for human review if < 80%)

6. **"How do you scale to 100K+ datasets?"**
   - Distributed crawlers (10+ workers)
   - Incremental crawls (only changed datasets)
   - Elasticsearch sharding (3+ shards)
   - Database read replicas (scale read queries)
   - Caching (Redis for hot data)

7. **"How do you integrate with BI tools?"**
   - API connectors (Tableau, Looker, Power BI)
   - Metadata extraction via APIs
   - Usage tracking from BI query logs
   - Embedded search (iframe in BI tool)
   - SSO integration (single sign-on)

8. **"How do you measure catalog success?"**
   - Search adoption (% users who search weekly)
   - Time to discovery (how fast users find data)
   - Metadata completeness (% datasets with descriptions)
   - Usage growth (query count trends)
   - Self-service rate (fewer data team tickets)

**Key Points to Mention:**

- Metadata as first-class citizen
- Multi-source ingestion (SQL, NoSQL, S3, APIs)
- Elasticsearch for fast search with relevance ranking
- Neo4j for lineage graph traversal
- Automated classification with ML
- Incremental crawls for scalability
- Event-driven updates for freshness
- RBAC and access control integration
- Usage analytics for recommendations

---

## Real-World Examples

**LinkedIn DataHub:**
- Open-source data catalog
- Graph-based metadata model
- Real-time metadata events via Kafka
- Elasticsearch for search
- Integration with Airflow, dbt, Spark

**Lyft Amundsen:**
- Metadata search and discovery
- Neo4j for lineage
- Apache Atlas integration
- Popularity scoring based on usage
- Table/column-level metadata

**Collibra:**
- Enterprise data governance platform
- Business glossary
- Data lineage visualization
- Policy management
- Workflow engine for data requests

**Alation:**
- Behavioral analysis (usage patterns)
- Auto-generated documentation
- ML-powered recommendations
- Collaboration features (comments, tags)
- Integration with 100+ data sources

---

## Summary

**System Characteristics:**

- **Scale:** 100K+ datasets, 10K searches/day, 1M+ metadata entities
- **Latency:** < 200ms search, < 1 hour metadata freshness
- **Availability:** 99.9% uptime
- **Accuracy:** > 80% search relevance, > 90% PII detection

**Core Components:**

1. **Metadata Ingestion:** SQL/NoSQL/S3 crawlers, event stream
2. **Search Service:** Elasticsearch with relevance ranking
3. **Lineage Graph:** Neo4j for relationship queries
4. **Metadata Store:** PostgreSQL for structured metadata
5. **ML Classifier:** Automated PII and sensitive data detection
6. **Usage Analytics:** ClickHouse for query log analysis

**Key Design Decisions:**

- Elasticsearch for search (fast, scalable full-text search)
- Neo4j for lineage (efficient graph traversal)
- Kafka for events (decoupled, scalable ingestion)
- Incremental crawls (reduce load, improve freshness)
- ML classification (automate tagging, reduce manual effort)
- Usage-based ranking (surface popular datasets)
- RBAC integration (secure access control)

This design provides a scalable, user-friendly data catalog that enables data discovery, understanding, and governance across large organizations with hundreds of thousands of datasets.
