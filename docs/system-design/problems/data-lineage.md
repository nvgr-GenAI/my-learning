# Design a Data Lineage Tracking System

A comprehensive data lineage platform that automatically discovers, tracks, and visualizes data flow across distributed systems, providing column-level lineage, impact analysis, compliance tracking, and governance capabilities through automated extraction from SQL/Spark queries, graph-based storage, and intelligent lineage propagation.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50K+ datasets, 500K+ transformations, 1M+ columns, PB-scale data pipelines |
| **Key Challenges** | Column-level lineage extraction, graph traversal at scale, incremental lineage updates, SQL/Spark parsing, impact analysis queries, lineage versioning |
| **Core Concepts** | Graph database for lineage storage, automated discovery, column-level lineage, impact analysis, compliance tracking, data governance, OpenLineage standard |
| **Companies** | DataHub, Marquez, OpenLineage, Collibra, Amundsen, Apache Atlas, Atlan, LinkedIn, Netflix |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Automated Discovery** | Extract lineage from SQL/Spark/Airflow automatically | P0 (Must have) |
    | **Column-Level Lineage** | Track transformations at column granularity | P0 (Must have) |
    | **Impact Analysis** | Find downstream dependencies (what breaks if I change X?) | P0 (Must have) |
    | **Root Cause Analysis** | Find upstream dependencies (where did this data come from?) | P0 (Must have) |
    | **Lineage Visualization** | Interactive graph visualization of data flow | P0 (Must have) |
    | **Compliance Tracking** | Track PII/sensitive data flow for GDPR/CCPA | P0 (Must have) |
    | **Lineage Versioning** | Track lineage changes over time | P1 (Should have) |
    | **Search & Query** | Query lineage graph (find all paths from A to B) | P1 (Should have) |
    | **Metadata Enrichment** | Add business context, owners, descriptions | P1 (Should have) |
    | **Lineage Validation** | Verify lineage accuracy, detect drift | P1 (Should have) |
    | **API & SDK** | Programmatic access for integrations | P1 (Should have) |
    | **Real-time Updates** | Update lineage as pipelines execute | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Data quality monitoring (use Monte Carlo/Great Expectations)
    - Data catalog and search (use DataHub/Amundsen)
    - ETL orchestration (use Airflow/Prefect)
    - Master data management (MDM)
    - Data transformation execution

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Lineage Query Latency** | < 200ms for 5-hop traversal | Fast impact analysis for users |
    | **Ingestion Throughput** | 10K lineage events/sec | Handle high-volume pipeline execution |
    | **Accuracy** | > 95% correct lineage edges | Trust is critical for compliance |
    | **Availability** | 99.9% uptime | Users depend on lineage for daily work |
    | **Scalability** | 50K datasets, 500K transformations | Enterprise-scale support |
    | **Freshness** | < 5 minutes lineage update lag | Near real-time lineage updates |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Dataset scale:
    - Total datasets: 50,000 (tables, files, APIs, dashboards)
    - Active transformations: 500,000 (ETL jobs, views, models)
    - Columns tracked: 1,000,000 (avg 20 columns per dataset)
    - Pipeline executions: 100,000/day (hourly jobs + on-demand)
    - Lineage events: 100,000 × 5 edges avg = 500K events/day
    - Event ingestion rate: 500K / 86,400 = 6 events/sec (avg)
    - Peak (5x): 30 events/sec

    Lineage queries:
    - Daily queries: 50,000 (10% of users × 10 queries/day)
    - Average query latency: 150ms
    - Query QPS: 50,000 / 86,400 = 0.6 queries/sec
    - Peak (10x): 6 queries/sec

    Impact analysis:
    - Daily impact queries: 5,000 (1% of users investigating changes)
    - Average traversal depth: 5 hops
    - Affected datasets per query: 50 avg
    - Query latency target: 200ms

    Compliance tracking:
    - PII-tagged columns: 10,000 (1% of columns)
    - Daily compliance reports: 100
    - Compliance query latency: 1 second (complex graph traversal)
    ```

    ### Storage Estimates

    ```
    Dataset metadata:
    - 50,000 datasets × 5 KB = 250 MB

    Transformation metadata:
    - 500,000 transformations × 10 KB = 5 GB
    - SQL/Spark query text, owner, schedule

    Lineage edges (graph data):
    - Dataset-level edges: 50,000 × 10 dependencies = 500K edges
    - Column-level edges: 1M columns × 5 dependencies = 5M edges
    - Per edge: 200 bytes (source_id, target_id, transformation_id, type)
    - Total: (500K + 5M) × 200 bytes = 1.1 GB

    Column-level transformations:
    - 5M column lineage edges × 2 KB (SQL fragment, expression) = 10 GB

    Lineage versioning (history):
    - Change rate: 5% of lineage per month
    - Monthly snapshot: 1.1 GB + 10 GB = 11.1 GB
    - 2-year retention: 11.1 GB × 24 months = 266 GB

    OpenLineage events (raw):
    - 500K events/day × 10 KB = 5 GB/day
    - 90-day retention: 5 GB × 90 = 450 GB

    Graph database indexes:
    - Adjacency lists, reverse indexes: 5x data size
    - 11.1 GB × 5 = 55.5 GB

    Total storage: 11.1 GB (graph) + 266 GB (history) + 450 GB (events) + 55.5 GB (indexes) ≈ 783 GB
    With replication (3x for Neo4j): ~2.3 TB
    ```

    ### Bandwidth Estimates

    ```
    Lineage event ingestion:
    - Average: 6 events/sec × 10 KB = 60 KB/sec = 480 Kbps
    - Peak: 30 events/sec × 10 KB = 300 KB/sec = 2.4 Mbps

    Lineage queries:
    - Average: 0.6 queries/sec × 50 KB response = 30 KB/sec = 240 Kbps
    - Peak: 6 queries/sec × 50 KB = 300 KB/sec = 2.4 Mbps

    Graph visualization:
    - 1,000 visualizations/day × 500 KB (JSON graph data) = 500 MB/day
    - Average bandwidth: 500 MB / 86,400 = 6 KB/sec = 48 Kbps
    ```

    ### Server Estimates

    ```
    Lineage collectors (event ingestion):
    - 3 nodes (8 cores, 32 GB RAM each)
    - Handle 100 events/sec (30 events/sec × 3 nodes)
    - Parse SQL/Spark queries, extract lineage

    Graph database (Neo4j):
    - 3 nodes (16 cores, 64 GB RAM, 1 TB SSD each)
    - Primary + 2 read replicas
    - Query latency: < 200ms for 5-hop traversal
    - Write throughput: 10K writes/sec

    SQL/Spark parsers:
    - 5 nodes (8 cores, 32 GB RAM each)
    - Parse complex SQL/Spark queries
    - Extract column-level lineage

    API service:
    - 3 nodes (8 cores, 32 GB RAM each)
    - Handle REST API requests
    - Rate limiting, authentication

    Message queue (Kafka):
    - 3 nodes (16 cores, 64 GB RAM)
    - Topics: lineage-events, lineage-updates

    Object storage (S3/GCS):
    - Managed service
    - 500 GB storage (raw events)

    Total infrastructure:
    - Collectors: 3 nodes
    - Graph DB: 3 nodes
    - Parsers: 5 nodes
    - API: 3 nodes
    - Kafka: 3 nodes
    - Total: ~17 nodes
    ```

    ---

    ## Key Assumptions

    1. 50,000 datasets monitored (tables, files, dashboards, ML models)
    2. 500,000 active transformations (ETL jobs, SQL views, Spark jobs)
    3. Average 10 dependencies per dataset
    4. Average 5 column-level dependencies per column
    5. 100,000 pipeline executions per day
    6. 5% lineage change rate per month
    7. 90-day retention for raw lineage events
    8. 2-year retention for lineage snapshots

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Automated discovery:** Extract lineage without manual input
    2. **Column-level granularity:** Track transformations at column level
    3. **Graph-based storage:** Efficient traversal for impact/root cause analysis
    4. **OpenLineage standard:** Use industry standard for interoperability
    5. **Incremental updates:** Update lineage edges, don't rebuild entire graph
    6. **Versioned lineage:** Track lineage evolution over time

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Data Sources"
            Airflow[Airflow DAGs<br/>Pipeline metadata<br/>Task dependencies]
            Spark[Spark Jobs<br/>Query plans<br/>DataFrame lineage]
            SQL[SQL Queries<br/>SELECT/INSERT/CREATE<br/>DML/DDL]
            dbt[dbt Models<br/>ref() lineage<br/>YAML metadata]
            BI[BI Tools<br/>Tableau, Looker<br/>Dashboard queries]
        end

        subgraph "Lineage Collection"
            Collector[Lineage Collector<br/>OpenLineage events<br/>Event validation]

            subgraph "Parsers"
                SQLParser[SQL Parser<br/>sqlglot, sqlparse<br/>Column lineage]
                SparkParser[Spark Plan Analyzer<br/>LogicalPlan parsing<br/>DataFrame lineage]
                AirflowParser[Airflow Parser<br/>DAG dependencies<br/>Task lineage]
            end

            Queue[Event Queue<br/>Kafka<br/>lineage-events topic]
        end

        subgraph "Lineage Processing"
            Processor[Lineage Processor<br/>Event normalization<br/>Edge construction]

            ColumnResolver[Column Resolver<br/>Resolve column mappings<br/>Handle aliases/renames]

            LineageBuilder[Lineage Graph Builder<br/>Construct edges<br/>Merge updates]

            Validator[Lineage Validator<br/>Check consistency<br/>Detect conflicts]
        end

        subgraph "Storage Layer"
            GraphDB[(Graph Database<br/>Neo4j<br/>Lineage edges<br/>Column mappings)]

            MetaDB[(Metadata Store<br/>PostgreSQL<br/>Dataset metadata<br/>Transformations)]

            EventStore[(Event Store<br/>S3/GCS<br/>Raw OpenLineage<br/>Event history)]

            VersionDB[(Version Store<br/>PostgreSQL<br/>Lineage snapshots<br/>Change history)]
        end

        subgraph "Query & Analysis"
            APIService[API Service<br/>REST/GraphQL<br/>Authentication<br/>Rate limiting]

            ImpactAnalyzer[Impact Analyzer<br/>Downstream traversal<br/>Affected datasets]

            RootCauseAnalyzer[Root Cause Analyzer<br/>Upstream traversal<br/>Data provenance]

            ComplianceTracker[Compliance Tracker<br/>PII/sensitive data<br/>GDPR tracking]

            SearchEngine[Search Engine<br/>Find paths A→B<br/>Pattern matching]
        end

        subgraph "Visualization"
            UI[Web UI<br/>Interactive graph<br/>Lineage explorer]

            GraphRenderer[Graph Renderer<br/>D3.js/Cytoscape<br/>Layout algorithms]

            Reports[Compliance Reports<br/>PII flow reports<br/>Audit trails]
        end

        Airflow -->|OpenLineage events| Collector
        Spark -->|Spark listener| Collector
        SQL -->|Query logs| Collector
        dbt -->|dbt events| Collector
        BI -->|BI API| Collector

        Collector --> SQLParser
        Collector --> SparkParser
        Collector --> AirflowParser

        SQLParser --> Queue
        SparkParser --> Queue
        AirflowParser --> Queue

        Queue --> Processor
        Processor --> ColumnResolver
        ColumnResolver --> LineageBuilder
        LineageBuilder --> Validator

        Validator --> GraphDB
        Validator --> MetaDB
        Collector --> EventStore
        LineageBuilder --> VersionDB

        GraphDB --> APIService
        MetaDB --> APIService

        APIService --> ImpactAnalyzer
        APIService --> RootCauseAnalyzer
        APIService --> ComplianceTracker
        APIService --> SearchEngine

        GraphDB --> ImpactAnalyzer
        GraphDB --> RootCauseAnalyzer
        GraphDB --> ComplianceTracker
        GraphDB --> SearchEngine

        ImpactAnalyzer --> UI
        RootCauseAnalyzer --> UI
        ComplianceTracker --> Reports
        SearchEngine --> UI

        UI --> GraphRenderer

        style SQLParser fill:#e1f5ff
        style SparkParser fill:#e1f5ff
        style GraphDB fill:#f3e5f5
        style ImpactAnalyzer fill:#fff9c4
        style ComplianceTracker fill:#ffe1e1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Graph DB (Neo4j)** | Fast graph traversal (BFS/DFS), native lineage queries | PostgreSQL (slow recursive CTEs), JanusGraph (complex setup) |
    | **sqlglot for SQL parsing** | Dialect-agnostic SQL parsing, column lineage extraction | sqlparse (no semantic analysis), custom parser (high effort) |
    | **OpenLineage standard** | Industry standard, ecosystem compatibility | Custom format (no interoperability), DataHub lineage (vendor lock-in) |
    | **Kafka for events** | High throughput, replay capability, durability | Redis Streams (limited retention), direct ingestion (no buffering) |
    | **Column-level tracking** | Compliance (track PII flow), fine-grained impact analysis | Table-level only (insufficient for compliance), no lineage (non-starter) |
    | **Incremental updates** | Scale to 500K transformations, avoid full graph rebuild | Full rebuild (slow, expensive), no updates (stale lineage) |

    **Key Trade-off:** We chose **column-level lineage** over table-level only, accepting 10x more edges and higher storage costs, because compliance tracking (GDPR/CCPA) requires knowing exactly where PII columns flow.

    ---

    ## API Design

    ### 1. Ingest OpenLineage Event

    **Request:**
    ```json
    POST /api/v1/lineage/events
    Content-Type: application/json

    {
      "eventType": "COMPLETE",
      "eventTime": "2025-02-05T10:00:00Z",
      "run": {
        "runId": "run-123-abc",
        "facets": {
          "nominalTime": {
            "_producer": "airflow-openlineage-provider",
            "nominalStartTime": "2025-02-05T10:00:00Z"
          }
        }
      },
      "job": {
        "namespace": "airflow",
        "name": "user_analytics.daily_aggregation",
        "facets": {
          "sql": {
            "_producer": "airflow-openlineage-provider",
            "query": "INSERT INTO analytics.user_daily_metrics SELECT user_id, COUNT(*) as event_count, MAX(timestamp) as last_event FROM events.user_events GROUP BY user_id"
          }
        }
      },
      "inputs": [
        {
          "namespace": "postgres://prod-db",
          "name": "events.user_events",
          "facets": {
            "schema": {
              "fields": [
                {"name": "user_id", "type": "bigint"},
                {"name": "event_type", "type": "string"},
                {"name": "timestamp", "type": "timestamp"}
              ]
            }
          },
          "inputFacets": {
            "columnLineage": {
              "fields": {
                "user_id": {
                  "inputFields": [
                    {"namespace": "postgres://prod-db", "name": "events.user_events", "field": "user_id"}
                  ]
                },
                "event_count": {
                  "inputFields": [
                    {"namespace": "postgres://prod-db", "name": "events.user_events", "field": "user_id"}
                  ],
                  "transformationType": "AGGREGATION",
                  "transformationDescription": "COUNT(*)"
                },
                "last_event": {
                  "inputFields": [
                    {"namespace": "postgres://prod-db", "name": "events.user_events", "field": "timestamp"}
                  ],
                  "transformationType": "AGGREGATION",
                  "transformationDescription": "MAX(timestamp)"
                }
              }
            }
          }
        }
      ],
      "outputs": [
        {
          "namespace": "postgres://prod-db",
          "name": "analytics.user_daily_metrics",
          "facets": {
            "schema": {
              "fields": [
                {"name": "user_id", "type": "bigint"},
                {"name": "event_count", "type": "bigint"},
                {"name": "last_event", "type": "timestamp"}
              ]
            }
          }
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "eventId": "evt_20250205_100000_abc123",
      "status": "accepted",
      "processedAt": "2025-02-05T10:00:01Z",
      "lineageEdgesCreated": 3,
      "warnings": []
    }
    ```

    ---

    ### 2. Query Column Lineage

    **Request:**
    ```bash
    GET /api/v1/lineage/column?
        dataset=analytics.user_daily_metrics&
        column=event_count&
        direction=upstream&
        maxDepth=5
    ```

    **Response:**
    ```json
    {
      "column": {
        "dataset": "analytics.user_daily_metrics",
        "column": "event_count",
        "type": "bigint"
      },
      "lineage": {
        "upstream": [
          {
            "dataset": "events.user_events",
            "column": "user_id",
            "distance": 1,
            "transformation": {
              "type": "AGGREGATION",
              "description": "COUNT(*) GROUP BY user_id",
              "sql": "COUNT(*)",
              "jobName": "user_analytics.daily_aggregation"
            },
            "path": [
              "analytics.user_daily_metrics.event_count",
              "events.user_events.user_id"
            ]
          },
          {
            "dataset": "raw.user_signups",
            "column": "id",
            "distance": 2,
            "transformation": {
              "type": "DIRECT",
              "description": "user_id = id",
              "jobName": "etl.load_user_events"
            },
            "path": [
              "analytics.user_daily_metrics.event_count",
              "events.user_events.user_id",
              "raw.user_signups.id"
            ]
          }
        ]
      },
      "totalPaths": 2,
      "queryTimeMs": 145
    }
    ```

    ---

    ### 3. Impact Analysis

    **Request:**
    ```bash
    GET /api/v1/lineage/impact?dataset=events.user_events&maxDepth=10
    ```

    **Response:**
    ```json
    {
      "dataset": "events.user_events",
      "impact": {
        "totalDatasetsAffected": 15,
        "totalColumnsAffected": 42,
        "totalJobsAffected": 8,
        "downstreamDatasets": [
          {
            "dataset": "analytics.user_daily_metrics",
            "distance": 1,
            "type": "table",
            "columnsAffected": ["user_id", "event_count", "last_event"],
            "criticalPath": true,
            "consumers": [
              {
                "type": "dashboard",
                "name": "Executive Dashboard",
                "users": 50
              }
            ]
          },
          {
            "dataset": "analytics.user_cohorts",
            "distance": 2,
            "type": "table",
            "columnsAffected": ["cohort_id", "user_count"],
            "criticalPath": false,
            "consumers": [
              {
                "type": "ml_model",
                "name": "Churn Prediction Model",
                "users": 5
              }
            ]
          }
        ],
        "dashboards": [
          {
            "name": "Executive Dashboard",
            "platform": "Tableau",
            "users": 50,
            "distance": 2
          },
          {
            "name": "Marketing Analytics",
            "platform": "Looker",
            "users": 30,
            "distance": 3
          }
        ],
        "mlModels": [
          {
            "name": "Churn Prediction Model",
            "framework": "scikit-learn",
            "distance": 3
          }
        ]
      },
      "impactScore": 0.78,
      "estimatedUsersAffected": 150,
      "queryTimeMs": 189
    }
    ```

    ---

    ### 4. Compliance Tracking (PII Flow)

    **Request:**
    ```bash
    GET /api/v1/lineage/compliance/pii?
        sourceDataset=raw.customer_data&
        sourceColumn=email&
        direction=downstream
    ```

    **Response:**
    ```json
    {
      "source": {
        "dataset": "raw.customer_data",
        "column": "email",
        "piiType": "EMAIL",
        "classification": "SENSITIVE"
      },
      "piiFlow": {
        "totalDestinations": 12,
        "destinations": [
          {
            "dataset": "marketing.email_campaigns",
            "column": "customer_email",
            "distance": 1,
            "transformation": "DIRECT",
            "retentionPolicy": "90 days",
            "encryptionStatus": "encrypted",
            "complianceStatus": "COMPLIANT"
          },
          {
            "dataset": "analytics.user_profiles",
            "column": "contact_email",
            "distance": 2,
            "transformation": "DIRECT",
            "retentionPolicy": "indefinite",
            "encryptionStatus": "plaintext",
            "complianceStatus": "NON_COMPLIANT",
            "violations": [
              "Indefinite retention violates GDPR",
              "PII stored in plaintext"
            ]
          }
        ],
        "externalExports": [
          {
            "destination": "s3://exports/customer-data",
            "exportedAt": "2025-02-04T10:00:00Z",
            "complianceStatus": "REQUIRES_REVIEW"
          }
        ]
      },
      "complianceReport": {
        "gdprCompliance": "NON_COMPLIANT",
        "ccpaCompliance": "REQUIRES_REVIEW",
        "violations": 2,
        "recommendations": [
          "Encrypt PII in analytics.user_profiles",
          "Implement retention policy for analytics.user_profiles"
        ]
      }
    }
    ```

    ---

    ## Database Schema

    ### Datasets (PostgreSQL)

    ```sql
    CREATE TABLE datasets (
        dataset_id BIGSERIAL PRIMARY KEY,
        namespace VARCHAR(500) NOT NULL,  -- postgres://prod-db, s3://bucket
        name VARCHAR(500) NOT NULL,       -- schema.table
        fully_qualified_name VARCHAR(1000) UNIQUE NOT NULL,
        type VARCHAR(100),  -- table, file, dashboard, ml_model
        platform VARCHAR(100),  -- postgres, snowflake, s3, tableau
        owner VARCHAR(255),
        description TEXT,
        tags JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_datasets_namespace ON datasets(namespace);
    CREATE INDEX idx_datasets_name ON datasets(name);
    CREATE INDEX idx_datasets_fqn ON datasets(fully_qualified_name);
    ```

    ### Columns (PostgreSQL)

    ```sql
    CREATE TABLE columns (
        column_id BIGSERIAL PRIMARY KEY,
        dataset_id BIGINT REFERENCES datasets(dataset_id),
        name VARCHAR(500) NOT NULL,
        type VARCHAR(100),
        nullable BOOLEAN,
        description TEXT,
        tags JSONB,  -- PII, sensitive, business_critical
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        UNIQUE(dataset_id, name)
    );

    CREATE INDEX idx_columns_dataset ON columns(dataset_id);
    CREATE INDEX idx_columns_tags ON columns USING GIN(tags);
    ```

    ### Transformations (PostgreSQL)

    ```sql
    CREATE TABLE transformations (
        transformation_id BIGSERIAL PRIMARY KEY,
        job_namespace VARCHAR(500),
        job_name VARCHAR(500),
        run_id VARCHAR(255),
        transformation_type VARCHAR(100),  -- SQL, SPARK, PYTHON, dbt
        query_text TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        executed_at TIMESTAMP
    );

    CREATE INDEX idx_transformations_job ON transformations(job_namespace, job_name);
    CREATE INDEX idx_transformations_run ON transformations(run_id);
    ```

    ### Lineage Edges (Neo4j)

    ```cypher
    // Dataset node
    CREATE (d:Dataset {
        id: 'dataset_123',
        namespace: 'postgres://prod-db',
        name: 'analytics.user_daily_metrics',
        type: 'table',
        platform: 'postgres'
    })

    // Column node
    CREATE (c:Column {
        id: 'column_456',
        dataset_id: 'dataset_123',
        name: 'event_count',
        type: 'bigint',
        tags: ['metric']
    })

    // Dataset-level lineage edge
    CREATE (source:Dataset)-[:DEPENDS_ON {
        transformation_id: 'trans_789',
        job_name: 'user_analytics.daily_aggregation',
        created_at: datetime(),
        lineage_type: 'TABLE'
    }]->(target:Dataset)

    // Column-level lineage edge
    CREATE (source_col:Column)-[:DERIVES_FROM {
        transformation_id: 'trans_789',
        transformation_type: 'AGGREGATION',
        transformation_description: 'COUNT(*)',
        sql_fragment: 'COUNT(*) as event_count',
        created_at: datetime(),
        lineage_type: 'COLUMN'
    }]->(target_col:Column)

    // Indexes for fast traversal
    CREATE INDEX dataset_id_index FOR (d:Dataset) ON (d.id);
    CREATE INDEX column_id_index FOR (c:Column) ON (c.id);
    CREATE INDEX column_dataset_index FOR (c:Column) ON (c.dataset_id);
    ```

    ### OpenLineage Events (S3/GCS - Parquet)

    ```
    event_id: string
    event_type: string  -- START, RUNNING, COMPLETE, FAIL, ABORT
    event_time: timestamp
    producer: string
    job_namespace: string
    job_name: string
    run_id: string
    inputs: array<struct<namespace, name, fields>>
    outputs: array<struct<namespace, name, fields>>
    column_lineage: map<string, array<struct<namespace, name, field>>>
    raw_event: string  -- Full JSON payload
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. SQL Parsing for Column Lineage

    **Using sqlglot for SQL lineage extraction:**

    ```python
    import sqlglot
    from sqlglot.lineage import lineage

    class SQLLineageExtractor:
        """
        Extract column-level lineage from SQL queries using sqlglot

        Supports:
        - SELECT with column expressions
        - JOINs (track which table columns come from)
        - Aggregations (GROUP BY, SUM, COUNT)
        - Subqueries and CTEs
        - Window functions
        - CASE/COALESCE expressions
        """

        def extract_lineage(self, sql: str, dialect: str = "postgres"):
            """
            Extract column-level lineage from SQL query

            Returns:
            {
              "target_table": "analytics.user_daily_metrics",
              "column_lineage": {
                "user_id": [
                  {"source_table": "events.user_events", "source_column": "user_id", "transformation": "DIRECT"}
                ],
                "event_count": [
                  {"source_table": "events.user_events", "source_column": "user_id", "transformation": "AGGREGATION"}
                ]
              }
            }
            """
            # Parse SQL
            parsed = sqlglot.parse_one(sql, dialect=dialect)

            # Extract target table
            target_table = self.extract_target_table(parsed)

            # Extract column lineage
            column_lineage = {}

            # Get all selected columns
            if parsed.find(sqlglot.exp.Select):
                select = parsed.find(sqlglot.exp.Select)

                for column_expr in select.expressions:
                    # Get target column name (alias or column name)
                    target_column = self.get_column_alias(column_expr)

                    # Get source columns for this target column
                    source_columns = self.extract_source_columns(column_expr, parsed)

                    column_lineage[target_column] = source_columns

            return {
                "target_table": target_table,
                "column_lineage": column_lineage,
                "query": sql
            }

        def extract_target_table(self, parsed):
            """Extract target table from INSERT/CREATE statement"""
            if parsed.find(sqlglot.exp.Insert):
                insert = parsed.find(sqlglot.exp.Insert)
                return insert.this.sql()
            elif parsed.find(sqlglot.exp.Create):
                create = parsed.find(sqlglot.exp.Create)
                return create.this.sql()
            else:
                # SELECT only, no target table
                return None

        def get_column_alias(self, column_expr):
            """Get column alias or name"""
            if isinstance(column_expr, sqlglot.exp.Alias):
                return column_expr.alias
            elif isinstance(column_expr, sqlglot.exp.Column):
                return column_expr.name
            else:
                # Complex expression, generate name
                return column_expr.sql()

        def extract_source_columns(self, column_expr, full_query):
            """
            Extract source columns and transformations

            Examples:
            - user_id → DIRECT mapping
            - COUNT(*) → AGGREGATION
            - CASE WHEN ... → CONDITIONAL
            - user_id + 1 → TRANSFORMATION
            """
            source_columns = []

            # Find all column references in expression
            for col in column_expr.find_all(sqlglot.exp.Column):
                table = col.table or self.infer_table(col, full_query)

                source_columns.append({
                    "source_table": table,
                    "source_column": col.name,
                    "transformation": self.classify_transformation(column_expr)
                })

            # Handle special cases
            if not source_columns:
                # COUNT(*), CURRENT_TIMESTAMP, literals
                transformation = self.classify_transformation(column_expr)
                if transformation != "LITERAL":
                    source_columns.append({
                        "source_table": None,
                        "source_column": None,
                        "transformation": transformation
                    })

            return source_columns

        def classify_transformation(self, expr):
            """Classify transformation type"""
            if isinstance(expr, sqlglot.exp.Column):
                return "DIRECT"
            elif isinstance(expr, (sqlglot.exp.Count, sqlglot.exp.Sum, sqlglot.exp.Avg, sqlglot.exp.Max, sqlglot.exp.Min)):
                return "AGGREGATION"
            elif isinstance(expr, sqlglot.exp.Case):
                return "CONDITIONAL"
            elif isinstance(expr, (sqlglot.exp.Add, sqlglot.exp.Sub, sqlglot.exp.Mul, sqlglot.exp.Div)):
                return "TRANSFORMATION"
            elif isinstance(expr, sqlglot.exp.Window):
                return "WINDOW_FUNCTION"
            elif isinstance(expr, (sqlglot.exp.Concat, sqlglot.exp.Substring)):
                return "STRING_MANIPULATION"
            elif isinstance(expr, (sqlglot.exp.Literal, sqlglot.exp.Null)):
                return "LITERAL"
            else:
                return "COMPLEX"

        def infer_table(self, column, full_query):
            """Infer which table a column comes from (for unqualified columns)"""
            # Get all tables in FROM clause
            tables = []
            for table_expr in full_query.find_all(sqlglot.exp.Table):
                tables.append(table_expr.name)

            # If only one table, column must come from it
            if len(tables) == 1:
                return tables[0]

            # Multiple tables, need to resolve ambiguity
            # This requires schema metadata to know which table has this column
            return None  # Mark as ambiguous

    # Example usage
    extractor = SQLLineageExtractor()

    sql = """
    INSERT INTO analytics.user_daily_metrics
    SELECT
        u.user_id,
        COUNT(*) as event_count,
        MAX(e.timestamp) as last_event,
        CASE WHEN COUNT(*) > 10 THEN 'active' ELSE 'inactive' END as user_status
    FROM events.user_events e
    JOIN users.user_profile u ON e.user_id = u.id
    WHERE e.event_date = CURRENT_DATE
    GROUP BY u.user_id
    """

    lineage = extractor.extract_lineage(sql)
    print(lineage)
    ```

    **Output:**
    ```python
    {
      "target_table": "analytics.user_daily_metrics",
      "column_lineage": {
        "user_id": [
          {
            "source_table": "users.user_profile",
            "source_column": "user_id",
            "transformation": "DIRECT"
          }
        ],
        "event_count": [
          {
            "source_table": "events.user_events",
            "source_column": None,
            "transformation": "AGGREGATION"
          }
        ],
        "last_event": [
          {
            "source_table": "events.user_events",
            "source_column": "timestamp",
            "transformation": "AGGREGATION"
          }
        ],
        "user_status": [
          {
            "source_table": "events.user_events",
            "source_column": None,
            "transformation": "CONDITIONAL"
          }
        ]
      }
    }
    ```

    ---

    ## 2. Spark Lineage Extraction

    **Analyzing Spark logical plans:**

    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql.catalog import Table

    class SparkLineageExtractor:
        """
        Extract lineage from Spark logical plans

        Approach:
        - Parse LogicalPlan from DataFrame
        - Walk plan tree (Project, Filter, Join, Aggregate)
        - Extract input/output column mappings
        """

        def __init__(self, spark: SparkSession):
            self.spark = spark

        def extract_lineage(self, df):
            """Extract lineage from Spark DataFrame"""
            # Get logical plan
            logical_plan = df._jdf.queryExecution().logical()

            # Parse plan to extract lineage
            lineage = self.parse_logical_plan(logical_plan)

            return lineage

        def parse_logical_plan(self, plan):
            """
            Parse Spark logical plan

            Plan types:
            - Project: SELECT columns
            - Filter: WHERE conditions
            - Join: JOIN operations
            - Aggregate: GROUP BY aggregations
            - LocalRelation: Literal data
            - HiveTableRelation: Reading from Hive table
            """
            plan_str = plan.toString()
            plan_type = plan.getClass().getSimpleName()

            if "Project" in plan_type:
                return self.parse_project(plan)
            elif "Aggregate" in plan_type:
                return self.parse_aggregate(plan)
            elif "Join" in plan_type:
                return self.parse_join(plan)
            elif "Filter" in plan_type:
                return self.parse_filter(plan)
            elif "HiveTableRelation" in plan_type:
                return self.parse_table_scan(plan)
            else:
                # Recursively parse children
                children = []
                for i in range(plan.children().size()):
                    child = plan.children().apply(i)
                    children.append(self.parse_logical_plan(child))
                return {"type": plan_type, "children": children}

        def parse_project(self, plan):
            """
            Parse Project (SELECT) operation

            Extract column mappings:
            - user_id → user_id (direct)
            - count(1) → event_count (aggregation)
            """
            output_cols = []
            for attr in plan.output():
                col_name = attr.name()
                expr = attr.toString()
                output_cols.append({
                    "name": col_name,
                    "expression": expr
                })

            # Get child plan (source)
            child = plan.children().apply(0)
            child_lineage = self.parse_logical_plan(child)

            return {
                "type": "PROJECT",
                "output_columns": output_cols,
                "source": child_lineage
            }

        def parse_aggregate(self, plan):
            """Parse Aggregate (GROUP BY) operation"""
            group_by_cols = []
            agg_cols = []

            # Extract grouping columns
            for expr in plan.groupingExpressions():
                group_by_cols.append(expr.toString())

            # Extract aggregation expressions
            for expr in plan.aggregateExpressions():
                agg_cols.append({
                    "name": expr.name(),
                    "expression": expr.toString(),
                    "type": "AGGREGATION"
                })

            child = plan.children().apply(0)
            child_lineage = self.parse_logical_plan(child)

            return {
                "type": "AGGREGATE",
                "group_by": group_by_cols,
                "aggregations": agg_cols,
                "source": child_lineage
            }

        def parse_join(self, plan):
            """Parse Join operation"""
            left = plan.children().apply(0)
            right = plan.children().apply(1)

            join_type = plan.joinType().toString()
            join_condition = plan.condition().toString() if plan.condition().isDefined() else None

            return {
                "type": "JOIN",
                "join_type": join_type,
                "condition": join_condition,
                "left": self.parse_logical_plan(left),
                "right": self.parse_logical_plan(right)
            }

        def parse_table_scan(self, plan):
            """Parse table scan (read from table)"""
            table_desc = plan.tableMeta()

            return {
                "type": "TABLE_SCAN",
                "database": table_desc.database(),
                "table": table_desc.identifier().table(),
                "columns": [col.name() for col in plan.output()]
            }

    # Example usage with Spark
    spark = SparkSession.builder.appName("lineage").enableHiveSupport().getOrCreate()

    # Sample DataFrame transformation
    events_df = spark.table("events.user_events")
    users_df = spark.table("users.user_profile")

    result_df = events_df \
        .join(users_df, events_df.user_id == users_df.id) \
        .groupBy("user_id") \
        .agg(
            count("*").alias("event_count"),
            max("timestamp").alias("last_event")
        )

    # Extract lineage
    extractor = SparkLineageExtractor(spark)
    lineage = extractor.extract_lineage(result_df)
    print(lineage)
    ```

    ---

    ## 3. Lineage Graph Construction

    **Building and querying the lineage graph in Neo4j:**

    ```python
    from neo4j import GraphDatabase

    class LineageGraphBuilder:
        """
        Build and query lineage graph in Neo4j

        Graph structure:
        - Nodes: Dataset, Column
        - Edges: DEPENDS_ON (table-level), DERIVES_FROM (column-level)
        """

        def __init__(self, uri, user, password):
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

        def create_dataset_node(self, dataset):
            """Create or update dataset node"""
            with self.driver.session() as session:
                session.execute_write(self._create_dataset, dataset)

        @staticmethod
        def _create_dataset(tx, dataset):
            query = """
            MERGE (d:Dataset {id: $id})
            SET d.namespace = $namespace,
                d.name = $name,
                d.type = $type,
                d.platform = $platform,
                d.updated_at = datetime()
            RETURN d
            """
            tx.run(query,
                   id=dataset['id'],
                   namespace=dataset['namespace'],
                   name=dataset['name'],
                   type=dataset['type'],
                   platform=dataset['platform'])

        def create_column_node(self, column):
            """Create or update column node"""
            with self.driver.session() as session:
                session.execute_write(self._create_column, column)

        @staticmethod
        def _create_column(tx, column):
            query = """
            MERGE (c:Column {id: $id})
            SET c.dataset_id = $dataset_id,
                c.name = $name,
                c.type = $type,
                c.tags = $tags,
                c.updated_at = datetime()
            RETURN c
            """
            tx.run(query,
                   id=column['id'],
                   dataset_id=column['dataset_id'],
                   name=column['name'],
                   type=column['type'],
                   tags=column.get('tags', []))

        def create_table_lineage_edge(self, source_dataset_id, target_dataset_id,
                                      transformation_id, job_name):
            """Create table-level lineage edge"""
            with self.driver.session() as session:
                session.execute_write(
                    self._create_table_edge,
                    source_dataset_id,
                    target_dataset_id,
                    transformation_id,
                    job_name
                )

        @staticmethod
        def _create_table_edge(tx, source_id, target_id, trans_id, job_name):
            query = """
            MATCH (source:Dataset {id: $source_id})
            MATCH (target:Dataset {id: $target_id})
            MERGE (target)-[r:DEPENDS_ON]->(source)
            SET r.transformation_id = $trans_id,
                r.job_name = $job_name,
                r.updated_at = datetime()
            RETURN r
            """
            tx.run(query,
                   source_id=source_id,
                   target_id=target_id,
                   trans_id=trans_id,
                   job_name=job_name)

        def create_column_lineage_edge(self, source_column_id, target_column_id,
                                       transformation_type, transformation_desc):
            """Create column-level lineage edge"""
            with self.driver.session() as session:
                session.execute_write(
                    self._create_column_edge,
                    source_column_id,
                    target_column_id,
                    transformation_type,
                    transformation_desc
                )

        @staticmethod
        def _create_column_edge(tx, source_id, target_id, trans_type, trans_desc):
            query = """
            MATCH (source:Column {id: $source_id})
            MATCH (target:Column {id: $target_id})
            MERGE (target)-[r:DERIVES_FROM]->(source)
            SET r.transformation_type = $trans_type,
                r.transformation_description = $trans_desc,
                r.updated_at = datetime()
            RETURN r
            """
            tx.run(query,
                   source_id=source_id,
                   target_id=target_id,
                   trans_type=trans_type,
                   trans_desc=trans_desc)

        def get_upstream_lineage(self, dataset_id, max_depth=5):
            """
            Get upstream lineage (root cause analysis)

            Returns all datasets that this dataset depends on
            """
            with self.driver.session() as session:
                result = session.execute_read(
                    self._query_upstream_lineage,
                    dataset_id,
                    max_depth
                )
                return result

        @staticmethod
        def _query_upstream_lineage(tx, dataset_id, max_depth):
            query = """
            MATCH path = (target:Dataset {id: $dataset_id})-[:DEPENDS_ON*1..{max_depth}]->(source:Dataset)
            RETURN
                source.id as source_id,
                source.name as source_name,
                source.type as source_type,
                length(path) as distance,
                [rel in relationships(path) | rel.job_name] as jobs
            ORDER BY distance
            """.format(max_depth=max_depth)

            result = tx.run(query, dataset_id=dataset_id)

            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"],
                    "source_type": record["source_type"],
                    "distance": record["distance"],
                    "jobs": record["jobs"]
                }
                for record in result
            ]

        def get_downstream_lineage(self, dataset_id, max_depth=10):
            """
            Get downstream lineage (impact analysis)

            Returns all datasets that depend on this dataset
            """
            with self.driver.session() as session:
                result = session.execute_read(
                    self._query_downstream_lineage,
                    dataset_id,
                    max_depth
                )
                return result

        @staticmethod
        def _query_downstream_lineage(tx, dataset_id, max_depth):
            query = """
            MATCH path = (source:Dataset {id: $dataset_id})<-[:DEPENDS_ON*1..{max_depth}]-(target:Dataset)
            RETURN
                target.id as target_id,
                target.name as target_name,
                target.type as target_type,
                length(path) as distance,
                [rel in relationships(path) | rel.job_name] as jobs
            ORDER BY distance
            """.format(max_depth=max_depth)

            result = tx.run(query, dataset_id=dataset_id)

            return [
                {
                    "target_id": record["target_id"],
                    "target_name": record["target_name"],
                    "target_type": record["target_type"],
                    "distance": record["distance"],
                    "jobs": record["jobs"]
                }
                for record in result
            ]

        def get_column_upstream(self, column_id, max_depth=5):
            """Get column-level upstream lineage"""
            with self.driver.session() as session:
                result = session.execute_read(
                    self._query_column_upstream,
                    column_id,
                    max_depth
                )
                return result

        @staticmethod
        def _query_column_upstream(tx, column_id, max_depth):
            query = """
            MATCH path = (target:Column {id: $column_id})-[:DERIVES_FROM*1..{max_depth}]->(source:Column)
            RETURN
                source.id as source_id,
                source.name as source_name,
                source.dataset_id as source_dataset_id,
                length(path) as distance,
                [rel in relationships(path) | rel.transformation_type] as transformations
            ORDER BY distance
            """.format(max_depth=max_depth)

            result = tx.run(query, column_id=column_id)

            return [
                {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"],
                    "source_dataset_id": record["source_dataset_id"],
                    "distance": record["distance"],
                    "transformations": record["transformations"]
                }
                for record in result
            ]

        def find_all_paths(self, source_dataset_id, target_dataset_id, max_depth=10):
            """Find all paths between two datasets"""
            with self.driver.session() as session:
                result = session.execute_read(
                    self._find_paths,
                    source_dataset_id,
                    target_dataset_id,
                    max_depth
                )
                return result

        @staticmethod
        def _find_paths(tx, source_id, target_id, max_depth):
            query = """
            MATCH path = (source:Dataset {id: $source_id})-[:DEPENDS_ON*1..{max_depth}]-(target:Dataset {id: $target_id})
            RETURN
                [node in nodes(path) | node.name] as nodes,
                length(path) as distance
            ORDER BY distance
            LIMIT 10
            """.format(max_depth=max_depth)

            result = tx.run(query, source_id=source_id, target_id=target_id)

            return [
                {
                    "nodes": record["nodes"],
                    "distance": record["distance"]
                }
                for record in result
            ]

        def close(self):
            self.driver.close()

    # Example usage
    graph = LineageGraphBuilder("bolt://localhost:7687", "neo4j", "password")

    # Create nodes
    graph.create_dataset_node({
        "id": "dataset_123",
        "namespace": "postgres://prod-db",
        "name": "analytics.user_daily_metrics",
        "type": "table",
        "platform": "postgres"
    })

    graph.create_dataset_node({
        "id": "dataset_456",
        "namespace": "postgres://prod-db",
        "name": "events.user_events",
        "type": "table",
        "platform": "postgres"
    })

    # Create lineage edge
    graph.create_table_lineage_edge(
        source_dataset_id="dataset_456",
        target_dataset_id="dataset_123",
        transformation_id="trans_789",
        job_name="user_analytics.daily_aggregation"
    )

    # Query lineage
    upstream = graph.get_upstream_lineage("dataset_123", max_depth=5)
    print("Upstream lineage:", upstream)

    downstream = graph.get_downstream_lineage("dataset_456", max_depth=10)
    print("Downstream impact:", downstream)

    graph.close()
    ```

    ---

    ## 4. Impact Analysis Engine

    ```python
    class ImpactAnalyzer:
        """
        Analyze downstream impact of changes

        Use cases:
        - Schema change: Which datasets/dashboards will break?
        - Data quality issue: Which consumers are affected?
        - Deprecation: What needs to be migrated?
        """

        def __init__(self, graph_builder, metadata_store):
            self.graph = graph_builder
            self.metadata = metadata_store

        def analyze_impact(self, dataset_id, max_depth=10):
            """
            Comprehensive impact analysis

            Returns:
            - Affected datasets
            - Affected dashboards
            - Affected ML models
            - Estimated users impacted
            - Critical path analysis
            """
            # Get downstream lineage from graph
            downstream = self.graph.get_downstream_lineage(dataset_id, max_depth)

            impact = {
                "dataset_id": dataset_id,
                "total_datasets_affected": len(downstream),
                "datasets": [],
                "dashboards": [],
                "ml_models": [],
                "estimated_users_affected": 0,
                "critical_paths": []
            }

            # Analyze each downstream dataset
            for ds in downstream:
                dataset_info = self.metadata.get_dataset(ds["target_id"])

                # Classify by type
                if dataset_info["type"] == "dashboard":
                    impact["dashboards"].append({
                        "name": dataset_info["name"],
                        "platform": dataset_info["platform"],
                        "distance": ds["distance"],
                        "users": dataset_info.get("user_count", 0)
                    })
                    impact["estimated_users_affected"] += dataset_info.get("user_count", 0)

                elif dataset_info["type"] == "ml_model":
                    impact["ml_models"].append({
                        "name": dataset_info["name"],
                        "framework": dataset_info.get("framework"),
                        "distance": ds["distance"]
                    })

                else:
                    impact["datasets"].append({
                        "id": ds["target_id"],
                        "name": ds["target_name"],
                        "type": ds["target_type"],
                        "distance": ds["distance"]
                    })

            # Identify critical paths (shortest path to high-impact consumers)
            for dashboard in impact["dashboards"]:
                if dashboard["users"] > 50:  # High-impact dashboard
                    impact["critical_paths"].append({
                        "target": dashboard["name"],
                        "type": "dashboard",
                        "distance": dashboard["distance"],
                        "users": dashboard["users"],
                        "critical": True
                    })

            # Compute impact score (0.0 to 1.0)
            impact["impact_score"] = self.compute_impact_score(impact)

            return impact

        def compute_impact_score(self, impact):
            """
            Compute impact score (0.0 to 1.0)

            Factors:
            - Number of affected datasets
            - Number of users affected
            - Criticality of consumers (dashboards > tables)
            - Distance (closer = higher impact)
            """
            score = 0.0

            # Dataset count factor (0-0.3)
            dataset_factor = min(0.3, impact["total_datasets_affected"] / 100)
            score += dataset_factor

            # User count factor (0-0.4)
            user_factor = min(0.4, impact["estimated_users_affected"] / 500)
            score += user_factor

            # Dashboard factor (0-0.2)
            dashboard_factor = min(0.2, len(impact["dashboards"]) / 10)
            score += dashboard_factor

            # Critical path factor (0-0.1)
            if impact["critical_paths"]:
                score += 0.1

            return min(1.0, score)

        def analyze_column_impact(self, column_id, max_depth=10):
            """
            Column-level impact analysis

            Use case: Schema change (column rename, type change, deletion)
            """
            # Get downstream column lineage
            downstream_cols = self.graph.get_column_downstream(column_id, max_depth)

            impact = {
                "column_id": column_id,
                "total_columns_affected": len(downstream_cols),
                "affected_columns": []
            }

            for col in downstream_cols:
                col_info = self.metadata.get_column(col["target_id"])
                dataset_info = self.metadata.get_dataset(col_info["dataset_id"])

                impact["affected_columns"].append({
                    "column_name": col_info["name"],
                    "dataset_name": dataset_info["name"],
                    "transformation": col["transformations"],
                    "distance": col["distance"]
                })

            return impact
    ```

    ---

    ## 5. Compliance Tracker (PII Flow)

    ```python
    class ComplianceTracker:
        """
        Track PII and sensitive data flow for compliance

        Regulations:
        - GDPR: Right to erasure, data minimization
        - CCPA: Right to know, opt-out
        - HIPAA: PHI protection
        """

        def __init__(self, graph_builder, metadata_store):
            self.graph = graph_builder
            self.metadata = metadata_store

        def track_pii_flow(self, source_dataset_id, source_column_name):
            """
            Track PII flow from source column to all destinations

            Returns:
            - All downstream columns containing PII
            - Encryption status
            - Retention policies
            - Compliance violations
            """
            # Get source column
            source_column = self.metadata.get_column_by_name(
                source_dataset_id,
                source_column_name
            )

            # Check if column is tagged as PII
            if "PII" not in source_column.get("tags", []):
                return {
                    "error": "Column not tagged as PII",
                    "suggestion": "Tag column with PII type (EMAIL, SSN, PHONE, etc.)"
                }

            # Get downstream column lineage
            downstream = self.graph.get_column_downstream(
                source_column["column_id"],
                max_depth=20  # Traverse entire graph
            )

            pii_flow = {
                "source": {
                    "dataset": source_dataset_id,
                    "column": source_column_name,
                    "pii_type": source_column.get("pii_type", "UNKNOWN")
                },
                "destinations": [],
                "violations": [],
                "compliance_status": "UNKNOWN"
            }

            # Analyze each destination
            for col in downstream:
                col_info = self.metadata.get_column(col["target_id"])
                dataset_info = self.metadata.get_dataset(col_info["dataset_id"])

                destination = {
                    "dataset": dataset_info["name"],
                    "column": col_info["name"],
                    "distance": col["distance"],
                    "transformation": col["transformations"],
                    "encryption_status": dataset_info.get("encryption_status", "UNKNOWN"),
                    "retention_policy": dataset_info.get("retention_policy", "UNKNOWN"),
                    "compliance_status": "COMPLIANT"
                }

                # Check for violations
                violations = []

                # Check encryption
                if destination["encryption_status"] != "encrypted":
                    violations.append({
                        "type": "UNENCRYPTED_PII",
                        "severity": "HIGH",
                        "description": f"PII stored in plaintext in {dataset_info['name']}"
                    })

                # Check retention
                if destination["retention_policy"] == "indefinite":
                    violations.append({
                        "type": "INDEFINITE_RETENTION",
                        "severity": "MEDIUM",
                        "description": "Indefinite retention violates GDPR data minimization"
                    })

                # Check external exports
                if dataset_info.get("platform") in ["s3", "gcs"]:
                    violations.append({
                        "type": "EXTERNAL_EXPORT",
                        "severity": "HIGH",
                        "description": "PII exported to external storage",
                        "requires_review": True
                    })

                if violations:
                    destination["compliance_status"] = "NON_COMPLIANT"
                    destination["violations"] = violations
                    pii_flow["violations"].extend(violations)

                pii_flow["destinations"].append(destination)

            # Overall compliance status
            if pii_flow["violations"]:
                pii_flow["compliance_status"] = "NON_COMPLIANT"
            else:
                pii_flow["compliance_status"] = "COMPLIANT"

            return pii_flow

        def generate_compliance_report(self, pii_types=["EMAIL", "SSN", "PHONE"]):
            """
            Generate compliance report for all PII columns

            Returns:
            - Total PII columns
            - Compliant vs. non-compliant
            - Violations by type
            - Recommendations
            """
            report = {
                "timestamp": datetime.utcnow(),
                "pii_columns": [],
                "total_violations": 0,
                "violations_by_type": {},
                "recommendations": []
            }

            # Get all PII columns
            for pii_type in pii_types:
                columns = self.metadata.get_columns_by_tag(f"PII:{pii_type}")

                for col in columns:
                    pii_flow = self.track_pii_flow(
                        col["dataset_id"],
                        col["name"]
                    )

                    report["pii_columns"].append({
                        "column": f"{col['dataset_id']}.{col['name']}",
                        "pii_type": pii_type,
                        "compliance_status": pii_flow["compliance_status"],
                        "destinations": len(pii_flow["destinations"]),
                        "violations": len(pii_flow["violations"])
                    })

                    # Aggregate violations
                    for violation in pii_flow["violations"]:
                        v_type = violation["type"]
                        report["violations_by_type"][v_type] = \
                            report["violations_by_type"].get(v_type, 0) + 1
                        report["total_violations"] += 1

            # Generate recommendations
            if report["violations_by_type"].get("UNENCRYPTED_PII", 0) > 0:
                report["recommendations"].append({
                    "priority": "HIGH",
                    "action": "Encrypt all datasets containing PII",
                    "affected_datasets": report["violations_by_type"]["UNENCRYPTED_PII"]
                })

            if report["violations_by_type"].get("INDEFINITE_RETENTION", 0) > 0:
                report["recommendations"].append({
                    "priority": "MEDIUM",
                    "action": "Implement retention policies for PII datasets",
                    "affected_datasets": report["violations_by_type"]["INDEFINITE_RETENTION"]
                })

            return report
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scalability Strategies

    ### Incremental Lineage Updates

    ```python
    class IncrementalLineageUpdater:
        """
        Update lineage incrementally instead of rebuilding entire graph

        Strategies:
        - Upsert edges (merge instead of create)
        - Prune stale edges (mark edges with last_seen timestamp)
        - Batch updates (accumulate events, write in batches)
        - Versioned lineage (store snapshots for rollback)
        """

        def __init__(self, graph_builder):
            self.graph = graph_builder
            self.edge_cache = {}  # Cache existing edges

        def update_lineage(self, lineage_event):
            """
            Update lineage graph incrementally

            Steps:
            1. Parse lineage event
            2. Compare with existing lineage
            3. Update only changed edges
            4. Mark edges as updated (last_seen timestamp)
            """
            # Extract edges from event
            new_edges = self.extract_edges(lineage_event)

            # Compare with existing edges
            existing_edges = self.get_existing_edges(lineage_event["job_name"])

            # Compute diff
            edges_to_add = []
            edges_to_update = []
            edges_to_delete = []

            for edge in new_edges:
                edge_key = f"{edge['source']}->{edge['target']}"

                if edge_key not in existing_edges:
                    edges_to_add.append(edge)
                else:
                    # Edge exists, check if transformation changed
                    if edge["transformation"] != existing_edges[edge_key]["transformation"]:
                        edges_to_update.append(edge)

            # Find edges that no longer exist (deleted)
            for edge_key, edge in existing_edges.items():
                if edge_key not in [f"{e['source']}->{e['target']}" for e in new_edges]:
                    edges_to_delete.append(edge)

            # Apply updates
            for edge in edges_to_add:
                self.graph.create_lineage_edge(edge)

            for edge in edges_to_update:
                self.graph.update_lineage_edge(edge)

            for edge in edges_to_delete:
                # Mark as inactive (don't delete, keep history)
                self.graph.mark_edge_inactive(edge)

            return {
                "edges_added": len(edges_to_add),
                "edges_updated": len(edges_to_update),
                "edges_deleted": len(edges_to_delete)
            }

        def prune_stale_edges(self, max_age_days=30):
            """
            Remove edges not seen in last N days

            Use case: Job deleted, lineage should be removed
            """
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            query = """
            MATCH ()-[r:DEPENDS_ON|DERIVES_FROM]->()
            WHERE r.last_seen < $cutoff_date
            DELETE r
            """

            self.graph.execute_query(query, cutoff_date=cutoff_date)
    ```

    ---

    ### Graph Query Optimization

    ```python
    class OptimizedLineageQuery:
        """
        Optimize lineage queries for large graphs

        Techniques:
        - Bidirectional BFS (search from both ends)
        - Early termination (stop when max_depth reached)
        - Result caching (cache frequent queries)
        - Query result pagination
        """

        def __init__(self, graph_builder, cache):
            self.graph = graph_builder
            self.cache = cache

        def get_downstream_with_cache(self, dataset_id, max_depth=10):
            """Get downstream lineage with caching"""
            # Check cache
            cache_key = f"downstream:{dataset_id}:{max_depth}"
            cached = self.cache.get(cache_key)

            if cached:
                return cached

            # Query graph
            result = self.graph.get_downstream_lineage(dataset_id, max_depth)

            # Cache result (TTL: 5 minutes)
            self.cache.set(cache_key, result, ttl=300)

            return result

        def bidirectional_search(self, source_id, target_id, max_depth=10):
            """
            Find shortest path using bidirectional BFS

            More efficient than single-direction search:
            - Time: O(b^(d/2)) vs O(b^d) where b=branching factor, d=depth
            """
            # Search from source (forward)
            forward_visited = {source_id: {"distance": 0, "path": [source_id]}}
            forward_frontier = [source_id]

            # Search from target (backward)
            backward_visited = {target_id: {"distance": 0, "path": [target_id]}}
            backward_frontier = [target_id]

            for depth in range(max_depth // 2):
                # Expand forward frontier
                new_forward_frontier = []
                for node in forward_frontier:
                    neighbors = self.graph.get_downstream_neighbors(node)

                    for neighbor in neighbors:
                        if neighbor not in forward_visited:
                            forward_visited[neighbor] = {
                                "distance": depth + 1,
                                "path": forward_visited[node]["path"] + [neighbor]
                            }
                            new_forward_frontier.append(neighbor)

                            # Check if we've met the backward search
                            if neighbor in backward_visited:
                                # Found path!
                                forward_path = forward_visited[neighbor]["path"]
                                backward_path = backward_visited[neighbor]["path"][::-1]

                                return {
                                    "path": forward_path + backward_path[1:],
                                    "distance": depth + 1 + backward_visited[neighbor]["distance"]
                                }

                forward_frontier = new_forward_frontier

                # Expand backward frontier
                new_backward_frontier = []
                for node in backward_frontier:
                    neighbors = self.graph.get_upstream_neighbors(node)

                    for neighbor in neighbors:
                        if neighbor not in backward_visited:
                            backward_visited[neighbor] = {
                                "distance": depth + 1,
                                "path": backward_visited[node]["path"] + [neighbor]
                            }
                            new_backward_frontier.append(neighbor)

                            # Check if we've met the forward search
                            if neighbor in forward_visited:
                                # Found path!
                                forward_path = forward_visited[neighbor]["path"]
                                backward_path = backward_visited[neighbor]["path"][::-1]

                                return {
                                    "path": forward_path + backward_path[1:],
                                    "distance": forward_visited[neighbor]["distance"] + depth + 1
                                }

                backward_frontier = new_backward_frontier

            # No path found within max_depth
            return None
    ```

    ---

    ### Lineage Versioning

    ```python
    class LineageVersionManager:
        """
        Track lineage changes over time

        Use cases:
        - Audit: When did lineage change?
        - Rollback: Restore previous lineage
        - Comparison: What changed between versions?
        """

        def create_snapshot(self, snapshot_name):
            """
            Create lineage snapshot

            Approach:
            - Export graph to JSON
            - Store in S3/GCS with timestamp
            - Keep metadata in PostgreSQL
            """
            # Export graph
            lineage_graph = self.export_graph()

            # Create snapshot metadata
            snapshot = {
                "snapshot_id": self.generate_snapshot_id(),
                "name": snapshot_name,
                "timestamp": datetime.utcnow(),
                "dataset_count": lineage_graph["dataset_count"],
                "edge_count": lineage_graph["edge_count"],
                "storage_path": f"s3://lineage-snapshots/{snapshot_name}.json"
            }

            # Store graph in S3
            self.upload_to_s3(
                lineage_graph,
                snapshot["storage_path"]
            )

            # Store metadata in PostgreSQL
            self.save_snapshot_metadata(snapshot)

            return snapshot

        def compare_snapshots(self, snapshot1_id, snapshot2_id):
            """
            Compare two lineage snapshots

            Returns:
            - Datasets added/removed
            - Edges added/removed
            - Transformations changed
            """
            # Load snapshots
            snapshot1 = self.load_snapshot(snapshot1_id)
            snapshot2 = self.load_snapshot(snapshot2_id)

            # Compare datasets
            datasets1 = set(snapshot1["datasets"].keys())
            datasets2 = set(snapshot2["datasets"].keys())

            datasets_added = datasets2 - datasets1
            datasets_removed = datasets1 - datasets2

            # Compare edges
            edges1 = set([f"{e['source']}->{e['target']}" for e in snapshot1["edges"]])
            edges2 = set([f"{e['source']}->{e['target']}" for e in snapshot2["edges"]])

            edges_added = edges2 - edges1
            edges_removed = edges1 - edges2

            return {
                "datasets_added": list(datasets_added),
                "datasets_removed": list(datasets_removed),
                "edges_added": list(edges_added),
                "edges_removed": list(edges_removed),
                "total_changes": len(datasets_added) + len(datasets_removed) +
                                len(edges_added) + len(edges_removed)
            }
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Incremental updates** | 10x faster updates | More complex logic |
    | **Result caching** | 5x faster repeated queries | Stale results (5 min lag) |
    | **Bidirectional BFS** | 2x faster path finding | More memory usage |
    | **Neo4j indexes** | 10x faster lookups | Higher storage cost |
    | **Batch event processing** | 5x higher throughput | Higher latency (up to 30s) |
    | **Graph partitioning** | Scale to 1M datasets | Complex query routing |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (50K datasets, 500K transformations):

    Compute:
    - 3 collector nodes × $50 = $150
    - 5 parser nodes × $100 = $500
    - 3 API nodes × $50 = $150
    - 3 Kafka nodes × $100 = $300
    - Total compute: $1,100/month

    Graph Database (Neo4j):
    - 3 nodes (16 cores, 64 GB) × $400 = $1,200
    - Neo4j Enterprise license: $3,000/month
    - Total Neo4j: $4,200/month

    Storage:
    - Graph data (3 TB with replication): 3,000 × $0.023 = $69
    - Event store (500 GB): 500 × $0.023 = $12
    - Total storage: $81/month

    Total: ~$5,381/month

    Optimizations:
    1. Use Neo4j Community Edition (free): -$3,000
    2. Reduce event retention to 30 days: -$8
    3. Use spot instances for parsers (70% discount): -$350
    4. Cache frequent queries (reduce DB load 30%): -$360

    Optimized Total: ~$1,663/month (69% reduction)
    ```

    ---

    ## Monitoring Metrics

    ```python
    # Key metrics for lineage system health

    # Ingestion metrics
    lineage_events_ingested_total{source, status}  # status: success, failure
    lineage_parsing_duration_seconds{parser_type}  # sql, spark, airflow
    lineage_ingestion_lag_seconds  # Time from event to graph update

    # Graph metrics
    lineage_datasets_total
    lineage_edges_total{lineage_type}  # lineage_type: table, column
    lineage_graph_query_duration_seconds{query_type}  # upstream, downstream, impact

    # Query metrics
    lineage_api_requests_total{endpoint, status}
    lineage_query_latency_seconds{query_type}
    lineage_cache_hit_ratio

    # Data quality metrics
    lineage_accuracy_rate  # % of correct lineage edges
    lineage_completeness_rate  # % of datasets with lineage
    lineage_staleness_seconds  # Age of oldest lineage edge

    # Business metrics
    impact_analyses_performed_total
    compliance_reports_generated_total
    pii_violations_detected_total
    ```

=== "💡 Step 5: Additional Considerations"

    ## Real-World Implementations

    ### OpenLineage Standard

    ```python
    """
    OpenLineage: Open standard for data lineage collection

    Key concepts:
    - Events: START, RUNNING, COMPLETE, FAIL, ABORT
    - Facets: Extensible metadata (SQL, schema, data quality)
    - Namespace: Unique identifier for data source
    - Run: Execution instance of a job

    Benefits:
    - Vendor-neutral standard
    - Ecosystem compatibility (Marquez, DataHub, Atlan)
    - Extensible via facets
    """

    # Example OpenLineage event
    openlineage_event = {
        "eventType": "COMPLETE",
        "eventTime": "2025-02-05T10:00:00Z",
        "producer": "airflow-openlineage/1.0.0",
        "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
        "job": {
            "namespace": "airflow",
            "name": "user_analytics.daily_aggregation"
        },
        "inputs": [
            {
                "namespace": "postgres://prod-db",
                "name": "events.user_events",
                "facets": {
                    "schema": {
                        "_producer": "airflow-openlineage/1.0.0",
                        "fields": [
                            {"name": "user_id", "type": "bigint"},
                            {"name": "event_type", "type": "string"}
                        ]
                    },
                    "dataQuality": {
                        "_producer": "great-expectations/0.15.0",
                        "rowCount": 1000000,
                        "bytes": 50000000,
                        "assertions": [
                            {
                                "assertion": "expect_column_values_to_not_be_null",
                                "column": "user_id",
                                "success": true
                            }
                        ]
                    }
                }
            }
        ],
        "outputs": [
            {
                "namespace": "postgres://prod-db",
                "name": "analytics.user_daily_metrics",
                "facets": {
                    "schema": {
                        "_producer": "airflow-openlineage/1.0.0",
                        "fields": [
                            {"name": "user_id", "type": "bigint"},
                            {"name": "event_count", "type": "bigint"}
                        ]
                    },
                    "columnLineage": {
                        "_producer": "airflow-openlineage/1.0.0",
                        "fields": {
                            "user_id": {
                                "inputFields": [
                                    {
                                        "namespace": "postgres://prod-db",
                                        "name": "events.user_events",
                                        "field": "user_id"
                                    }
                                ],
                                "transformationType": "DIRECT"
                            },
                            "event_count": {
                                "inputFields": [
                                    {
                                        "namespace": "postgres://prod-db",
                                        "name": "events.user_events",
                                        "field": "user_id"
                                    }
                                ],
                                "transformationType": "AGGREGATION",
                                "transformationDescription": "COUNT(*)"
                            }
                        }
                    }
                }
            }
        ]
    }
    ```

    ---

    ## Integration with Data Catalog

    ```python
    """
    Integrate lineage with data catalog (DataHub, Amundsen)

    Benefits:
    - Unified metadata view
    - Search includes lineage
    - Rich context (ownership, tags, descriptions)
    """

    class DataCatalogIntegration:
        def enrich_lineage_with_catalog(self, lineage_graph, catalog_client):
            """Add catalog metadata to lineage nodes"""
            for node in lineage_graph["nodes"]:
                # Get dataset metadata from catalog
                catalog_metadata = catalog_client.get_dataset(node["id"])

                # Enrich node
                node["owner"] = catalog_metadata.get("owner")
                node["description"] = catalog_metadata.get("description")
                node["tags"] = catalog_metadata.get("tags")
                node["quality_score"] = catalog_metadata.get("quality_score")

            return lineage_graph
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"How do you handle complex SQL transformations (CTEs, subqueries, window functions)?"**
   - Use advanced SQL parsers (sqlglot with semantic analysis)
   - Parse CTEs as intermediate virtual tables
   - Window functions: track partitioning columns
   - Subqueries: recursively parse inner queries
   - Maintain context (table aliases, CTEs in scope)

2. **"How do you ensure lineage accuracy?"**
   - Automated validation (compare lineage with actual data flow)
   - User feedback (allow users to correct lineage)
   - Confidence scoring (parser confidence + user validation)
   - Regular audits (sample lineage, verify manually)
   - Schema-based validation (check if columns exist)

3. **"How do you handle lineage at PB scale?"**
   - Graph partitioning (shard by namespace/platform)
   - Incremental updates (only update changed edges)
   - Sampling (profile subset of transformations)
   - Federated lineage (separate graphs per region/team)
   - Result caching (cache frequent queries)

4. **"How do you track lineage in real-time streaming pipelines?"**
   - Kafka/Flink integration (emit lineage events)
   - Micro-batch lineage (collect lineage per batch)
   - Stream topology (track Kafka topics, stream processors)
   - Challenge: Column lineage harder in streaming (schema evolution)

5. **"How do you handle schema evolution?"**
   - Track schema versions (link lineage to schema version)
   - Detect breaking changes (column renamed/deleted)
   - Version lineage (separate graphs per schema version)
   - Alert on schema drift (lineage validation fails)

6. **"How do you optimize graph traversal queries?"**
   - Neo4j indexes (index on dataset_id, column_id)
   - Bidirectional BFS (search from both ends)
   - Early termination (stop at max_depth)
   - Query result caching (5-minute TTL)
   - Graph partitioning (reduce search space)

7. **"How do you integrate with BI tools (Tableau, Looker)?"**
   - BI tool APIs (extract dashboard queries)
   - Parse BI SQL (Tableau generates SQL)
   - Manual registration (users register dashboards)
   - Web scraping (extract metadata from BI tool)

8. **"How do you handle privacy and security?"**
   - RBAC (users only see lineage they can access)
   - Redaction (hide sensitive column names/values)
   - Audit logs (track who accessed lineage)
   - Data classification (label PII, sensitive columns)

**Key Points to Mention:**

- OpenLineage standard for interoperability
- Column-level lineage for compliance
- Graph database (Neo4j) for fast traversal
- sqlglot for SQL parsing, Spark plan for Spark lineage
- Incremental updates for scale
- Impact analysis for change management
- Compliance tracking for GDPR/CCPA
- Lineage versioning for audit trails

---

## Real-World Examples

**DataHub (LinkedIn):**
- Metadata platform with lineage
- OpenLineage integration
- Graph-based storage (MySQL + Neo4j)
- Column-level lineage via SQL parsing
- Used internally at LinkedIn for 10K+ datasets

**Marquez (WeWork):**
- OpenLineage reference implementation
- Lineage collection via OpenLineage API
- PostgreSQL for metadata, graph queries via SQL
- Integration with Airflow, dbt, Spark
- Open-source (Apache 2.0)

**Collibra:**
- Enterprise data governance platform
- Automated lineage discovery
- Business glossary integration
- Compliance tracking (GDPR, CCPA)
- Commercial product ($$$)

**Apache Atlas:**
- Metadata management for Hadoop
- Integration with Hive, HBase, Kafka
- Graph database (JanusGraph)
- Type system (entities, classifications, relationships)
- Open-source (Apache 2.0)

---

## Summary

**System Characteristics:**

- **Scale:** 50K datasets, 500K transformations, 1M columns
- **Throughput:** 10K lineage events/sec ingestion
- **Latency:** < 200ms for 5-hop traversal
- **Accuracy:** > 95% correct lineage edges
- **Availability:** 99.9% uptime

**Core Components:**

1. **Lineage Collector:** Ingest OpenLineage events from Airflow/Spark/dbt
2. **SQL/Spark Parsers:** Extract column-level lineage from queries
3. **Graph Database (Neo4j):** Store and query lineage graph
4. **Impact Analyzer:** Downstream dependency analysis
5. **Compliance Tracker:** PII flow tracking for GDPR/CCPA
6. **Lineage Versioning:** Track lineage changes over time

**Key Design Decisions:**

- OpenLineage standard for interoperability
- Neo4j for fast graph traversal (vs. PostgreSQL)
- Column-level lineage for compliance (vs. table-level only)
- sqlglot for SQL parsing (vs. custom parser)
- Incremental updates (vs. full rebuild)
- Result caching (5-min TTL)
- Bidirectional BFS for path finding

This design provides a comprehensive, scalable data lineage tracking system capable of handling enterprise-scale data pipelines, providing column-level lineage for compliance, and enabling fast impact analysis for change management.
