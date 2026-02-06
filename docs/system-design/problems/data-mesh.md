# Design a Data Mesh Platform

A decentralized data architecture platform that treats data as a product, organized by domain ownership with federated computational governance, self-serve data infrastructure, and interoperability standards to enable scalable, discoverable, and high-quality data products across an enterprise.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100+ domains, 1,000+ data products, 10,000+ consumers, petabyte-scale distributed data |
| **Key Challenges** | Domain autonomy vs standardization, federated governance, discovery across domains, data product SLAs, cross-domain data sharing |
| **Core Concepts** | Domain ownership, data products, federated computational governance, self-serve platform, data contracts, interoperability |
| **Companies** | ThoughtWorks, Netflix, Zalando, Spotify, LinkedIn, PayPal, Intuit, modern data teams |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Domain Data Products** | Domains own and publish data products with SLAs | P0 (Must have) |
    | **Data Contracts** | Define schemas, SLAs, quality guarantees | P0 (Must have) |
    | **Self-Serve Infrastructure** | Platform for creating, deploying, and serving data products | P0 (Must have) |
    | **Federated Governance** | Domain-level autonomy with global policies | P0 (Must have) |
    | **Data Catalog** | Discover and search data products across domains | P0 (Must have) |
    | **Policy Engine** | Enforce computational policies (security, quality, compliance) | P0 (Must have) |
    | **Interoperability Standards** | Common schemas, formats, APIs for cross-domain consumption | P0 (Must have) |
    | **Data Product Observability** | Monitor SLAs, quality, usage metrics | P1 (Should have) |
    | **Lineage Tracking** | Track data flow across domains | P1 (Should have) |
    | **Access Control** | Fine-grained permissions for data products | P1 (Should have) |
    | **Data Marketplace** | Browse, request access, rate data products | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Centralized data lake/warehouse (anti-pattern in data mesh)
    - Real-time streaming orchestration (use Kafka/Flink)
    - ML model deployment (use MLOps platforms)
    - BI tool implementation (use Tableau/Looker)
    - Data storage engines (use existing: Snowflake, S3, BigQuery)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Domain Autonomy** | 100+ independent domains | Enable decentralized ownership and scaling |
    | **Data Product Availability** | 99.9% uptime per product | Business-critical analytics workloads |
    | **Discovery Latency** | < 1s catalog search | Fast data product discovery |
    | **Product Creation Time** | < 1 day to deploy new product | Self-serve infrastructure efficiency |
    | **Cross-Domain Query** | < 10s p95 for federated queries | Enable cross-domain analytics |
    | **Policy Compliance** | 100% enforcement | Security, privacy, regulatory requirements |
    | **Scalability** | 1,000+ data products | Support large enterprises |

    ---

    ## Capacity Estimation

    ### Domain & Product Estimates

    ```
    Organization structure:
    - Total domains: 100 domains (e.g., Customer, Orders, Payments, Marketing, etc.)
    - Data products per domain: 10 products (avg)
    - Total data products: 100 × 10 = 1,000 data products
    - Data product versions: 5 versions per product (avg)
    - Total versioned products: 1,000 × 5 = 5,000 versions

    Data product characteristics:
    - Average data product size: 100 GB - 10 TB
    - Total data across all products: 10 PB
    - Daily data updates: 10% of products (100 products/day)
    - Daily data volume updated: 100 products × 500 GB = 50 TB/day

    Domain organization:
    - Domain teams: 100 teams
    - Team size: 5-10 people (data engineers, analytics engineers, domain experts)
    - Products per team: 10 products
    - Consumers per product: 50-100 consumers (other domains, analysts, applications)
    ```

    ### Traffic Estimates

    ```
    Catalog queries (discovery):
    - Daily catalog searches: 10,000 searches
    - Peak searches: 100 searches/sec
    - Search latency: < 1 second (indexed search)
    - Catalog metadata size: 1,000 products × 100 KB = 100 MB

    Data product access:
    - Daily data product queries: 100,000 queries
    - Peak query rate: 1,000 queries/sec
    - Average query size: 1 GB scanned
    - Total daily scan: 100,000 × 1 GB = 100 TB/day

    Cross-domain data sharing:
    - 20% of queries are cross-domain (query multiple domains)
    - Cross-domain queries: 20,000 queries/day
    - Average domains per query: 3 domains
    - Federated query complexity: 3x latency overhead

    Policy enforcement:
    - Policy checks per query: 100,000 checks/day
    - Policy evaluation latency: < 10ms
    - Policy rules: 500 global + domain policies
    - Policy cache hit rate: 95%

    Data product updates:
    - Daily product updates: 100 products updated/day
    - Update frequency: hourly, daily, weekly (varies by product)
    - Update duration: 5 min - 2 hours (depends on size)
    - Concurrent updates: 20 products updating simultaneously
    ```

    ### Storage Estimates

    ```
    Data product storage:
    - Total data products: 10 PB
    - Storage format: Parquet/Iceberg on S3
    - Storage cost: 10 PB × $0.023/GB = $230K/month
    - With compression (5:1): $46K/month

    Metadata storage:
    - Product schemas: 1,000 products × 50 KB = 50 MB
    - SLA definitions: 1,000 × 10 KB = 10 MB
    - Quality metrics: 1,000 × 100 KB = 100 MB
    - Lineage graph: 5,000 edges × 1 KB = 5 MB
    - Total metadata: ~200 MB (negligible)

    Contract registry:
    - Data contracts: 1,000 contracts × 20 KB = 20 MB
    - Contract versions: 5,000 versions × 20 KB = 100 MB

    Logs & observability:
    - Access logs: 100K queries/day × 1 KB = 100 MB/day
    - Quality metrics: 1,000 products × 10 metrics × 1 KB = 10 MB/day
    - Retention: 90 days → 10 GB
    ```

    ### Compute Estimates

    ```
    Self-serve platform:
    - Data product build pipelines: 100 concurrent builds
    - Build workers: 100 Kubernetes pods
    - CPU per pod: 4 cores
    - Memory per pod: 16 GB
    - Total: 400 cores, 1.6 TB RAM

    Policy engine:
    - Policy evaluation nodes: 10 nodes (HA)
    - CPU per node: 8 cores
    - Memory per node: 32 GB
    - Total: 80 cores, 320 GB RAM

    Catalog service:
    - Search nodes: 5 nodes
    - CPU per node: 4 cores
    - Memory per node: 16 GB
    - Total: 20 cores, 80 GB RAM

    Data product serving (compute):
    - Query engines per domain: 1-3 engines (Spark, Presto, dbt)
    - Average cluster: 20 nodes × 16 cores = 320 cores per domain
    - 100 domains: 32,000 cores total (shared across time)
    - Actual usage (20% active): 6,400 cores
    ```

    ---

    ## Key Assumptions

    1. 100 domains with 10 data products each (1,000 total products)
    2. 10 PB total data across all products
    3. 50-100 consumers per data product
    4. 20% of queries are cross-domain (federated)
    5. 10% of products updated daily
    6. Data contracts define SLAs (freshness, quality, availability)
    7. Domains use self-serve platform for product lifecycle
    8. Federated governance with domain autonomy
    9. Interoperability via common standards (Parquet, Avro, REST APIs)
    10. Policy engine enforces computational policies globally

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Domain ownership:** Each domain owns its data products end-to-end
    2. **Data as a product:** Treat data with product thinking (discoverable, addressable, trustworthy, self-describing)
    3. **Self-serve platform:** Enable domains to create and manage products independently
    4. **Federated computational governance:** Domain autonomy with global policies as code
    5. **Interoperability by design:** Common standards for schemas, formats, APIs
    6. **Product-centric architecture:** Not table-centric or pipeline-centric

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Domain: Customer Analytics"
            CDP1[Customer Data Product<br/>Owner: Customer Team<br/>SLA: 99.9%, 1hr freshness]
            CDP2[Customer Segments<br/>Owner: Customer Team<br/>SLA: 99.5%, 24hr freshness]
            CDataStore[(Customer Domain<br/>Data Storage<br/>Snowflake)]
            CDP1 --> CDataStore
            CDP2 --> CDataStore
        end

        subgraph "Domain: Orders"
            ODP1[Orders Data Product<br/>Owner: Orders Team<br/>SLA: 99.99%, 5min freshness]
            ODP2[Order Analytics<br/>Owner: Orders Team<br/>SLA: 99.5%, 1hr freshness]
            ODataStore[(Orders Domain<br/>Data Storage<br/>S3/Iceberg)]
            ODP1 --> ODataStore
            ODP2 --> ODataStore
        end

        subgraph "Domain: Payments"
            PDP1[Payment Transactions<br/>Owner: Payments Team<br/>SLA: 99.99%, real-time]
            PDataStore[(Payments Domain<br/>Data Storage<br/>BigQuery)]
            PDP1 --> PDataStore
        end

        subgraph "Self-Serve Data Platform"
            ProductSDK[Product SDK<br/>Python/Go Library<br/>Create/Deploy/Update]
            CICDPipeline[CI/CD Pipeline<br/>Terraform/Kubernetes<br/>Infrastructure as Code]
            DataCatalog[Data Catalog<br/>OpenMetadata<br/>Search/Discovery]
            ContractRegistry[Contract Registry<br/>Schema Registry<br/>Avro/Protobuf]
            PolicyEngine[Policy Engine<br/>Open Policy Agent<br/>Computational Governance]
            ObservabilityHub[Observability Hub<br/>Prometheus/Grafana<br/>SLA Monitoring]
        end

        subgraph "Federated Governance"
            GlobalPolicies[Global Policies<br/>Security, Privacy<br/>Compliance, Quality]
            DomainPolicies[Domain Policies<br/>Domain-specific rules<br/>Local governance]
        end

        subgraph "Data Consumers"
            Analyst[Data Analysts<br/>Tableau, Looker<br/>Cross-domain queries]
            MLTeam[ML Engineers<br/>Feature stores<br/>Model training]
            Application[Applications<br/>REST APIs<br/>Real-time access]
            OtherDomains[Other Domains<br/>Cross-domain<br/>data sharing]
        end

        subgraph "Infrastructure Layer"
            K8s[Kubernetes<br/>Compute orchestration]
            Storage[Storage Layer<br/>S3, Snowflake, BigQuery<br/>Polyglot persistence]
            Network[Service Mesh<br/>Istio/Envoy<br/>API gateway]
        end

        CDP1 --> ProductSDK
        CDP2 --> ProductSDK
        ODP1 --> ProductSDK
        ODP2 --> ProductSDK
        PDP1 --> ProductSDK

        ProductSDK --> CICDPipeline
        ProductSDK --> DataCatalog
        ProductSDK --> ContractRegistry

        CICDPipeline --> K8s
        K8s --> Storage

        CDP1 --> PolicyEngine
        ODP1 --> PolicyEngine
        PDP1 --> PolicyEngine

        PolicyEngine --> GlobalPolicies
        PolicyEngine --> DomainPolicies

        CDP1 --> ObservabilityHub
        ODP1 --> ObservabilityHub
        PDP1 --> ObservabilityHub

        DataCatalog --> Analyst
        DataCatalog --> MLTeam
        DataCatalog --> Application

        CDP1 --> Analyst
        ODP1 --> MLTeam
        PDP1 --> Application

        CDP1 --> OtherDomains
        ODP1 --> OtherDomains

        ObservabilityHub --> Analyst

        Storage --> Network
        Network --> Application

        style CDP1 fill:#e1f5ff
        style ODP1 fill:#e8f5e9
        style PDP1 fill:#fff9c4
        style ProductSDK fill:#fce4ec
        style PolicyEngine fill:#ffe1e1
        style DataCatalog fill:#f3e5f5
        style ObservabilityHub fill:#e0f2f1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Domain Data Products** | Core principle: domain ownership, product thinking | Centralized data lake (monolith, bottleneck) |
    | **Self-Serve Platform** | Enable domain autonomy, reduce platform team bottleneck | Manual setup (doesn't scale), central team (slow) |
    | **Data Contracts** | Guarantee SLAs, interoperability, backwards compatibility | Ad-hoc schemas (breaks consumers), documentation (not enforced) |
    | **Policy Engine (OPA)** | Enforce policies as code, federated governance | Manual reviews (slow), centralized control (bottleneck) |
    | **Data Catalog (OpenMetadata)** | Discover products across 100 domains | File-based docs (not searchable), wiki (stale) |
    | **Kubernetes** | Self-serve infrastructure, domain isolation | VMs (slow provisioning), serverless (vendor lock-in) |
    | **Polyglot Storage** | Domains choose best storage for use case | Single storage (one-size-fits-all doesn't work) |

    **Key Trade-off:** We chose **domain autonomy with federated policies** over centralized control to achieve scale and agility. This means accepting eventual consistency and complexity in cross-domain queries in exchange for independent domain velocity.

    ---

    ## Data Product Structure

    ### Data Product Definition

    ```yaml
    # data-product.yaml (Infrastructure as Code)
    apiVersion: datamesh.io/v1
    kind: DataProduct
    metadata:
      name: customer-analytics-360
      domain: customer-analytics
      owner: customer-team@company.com
      version: 2.3.0
      tags:
        - customer
        - analytics
        - pii
        - gdpr

    spec:
      description: |
        360-degree view of customer data including demographics,
        behavior, preferences, and lifetime value.

      # Output ports (how consumers access data)
      output_ports:
        - name: customer-360-analytical
          type: batch
          format: parquet
          location: s3://data-mesh/customer-analytics/customer-360/
          schema_ref: customer_360_v2.avsc
          partitioning:
            - field: country
            - field: date
          access_mode: read-only

        - name: customer-360-api
          type: rest-api
          endpoint: https://api.datamesh.company.com/customer-analytics/customer-360/v2
          schema_ref: customer_360_v2_api.yaml
          rate_limit: 1000 req/sec
          authentication: oauth2

        - name: customer-360-streaming
          type: stream
          topic: customer-analytics.customer-360.changes
          schema_ref: customer_360_v2.avsc
          format: avro

      # Input ports (data sources)
      input_ports:
        - source: customer-domain.raw.users
          type: batch
        - source: customer-domain.raw.interactions
          type: stream

      # SLA commitments (contract)
      sla:
        availability: 99.9%
        freshness: 1 hour
        completeness: 99.5%
        timeliness: 95% within 1 hour
        retention: 2 years

      # Quality guarantees
      quality:
        - name: no_null_customer_id
          rule: customer_id IS NOT NULL
          severity: error
        - name: valid_email
          rule: email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
          severity: warning
        - name: revenue_positive
          rule: lifetime_value >= 0
          severity: error

      # Computational policies (applied automatically)
      policies:
        - type: access-control
          rule: |
            allow if user.role in ["analyst", "data-scientist"] and
                     user.completed_pii_training == true
        - type: data-privacy
          rule: |
            mask email if user.region != "EU"
            redact ssn for all users
        - type: data-retention
          rule: |
            delete records older than 2 years

      # Dependencies
      dependencies:
        - data_product: customer-domain.raw-users
          version: ">=1.0.0"
        - data_product: events-domain.user-interactions
          version: "^2.0.0"

      # Observability
      observability:
        metrics:
          - name: rows_processed
            type: counter
          - name: processing_duration
            type: histogram
          - name: data_quality_score
            type: gauge
        alerts:
          - name: freshness_sla_breach
            condition: freshness > 1 hour
            severity: critical
          - name: quality_score_low
            condition: data_quality_score < 0.95
            severity: warning

      # Infrastructure (managed by platform)
      infrastructure:
        compute:
          type: spark
          cluster_size: medium
          schedule: "0 * * * *"  # Hourly
        storage:
          type: s3-iceberg
          compression: snappy
          partitions: auto
    ```

    ---

    ## API Design

    ### 1. Register Data Product

    **Request:**
    ```bash
    POST /api/v1/data-products
    Content-Type: application/yaml

    # (YAML from above)
    ```

    **Response:**
    ```json
    {
      "product_id": "customer-analytics-360",
      "version": "2.3.0",
      "status": "pending_validation",
      "validation_checks": {
        "schema_validation": "passed",
        "policy_compliance": "passed",
        "resource_allocation": "in_progress"
      },
      "deployment_url": "https://console.datamesh.company.com/products/customer-analytics-360/deploy",
      "estimated_deployment_time": "15 minutes"
    }
    ```

    ---

    ### 2. Query Data Product

    **Request:**
    ```bash
    GET /api/v1/data-products/customer-analytics-360/query
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "sql": "SELECT customer_id, country, lifetime_value FROM customer_360 WHERE country = 'US' LIMIT 100",
      "output_format": "json",
      "filters": {
        "country": "US"
      }
    }
    ```

    **Response:**
    ```json
    {
      "query_id": "q_abc123",
      "status": "completed",
      "rows": [
        {
          "customer_id": "cust_001",
          "country": "US",
          "lifetime_value": 1250.50
        }
      ],
      "metadata": {
        "rows_returned": 100,
        "bytes_scanned": 1048576,
        "execution_time_ms": 234,
        "data_product_version": "2.3.0",
        "freshness": "2025-02-05T10:30:00Z"
      },
      "lineage": {
        "source_products": [
          "customer-domain.raw-users",
          "events-domain.user-interactions"
        ]
      }
    }
    ```

    ---

    ### 3. Search Data Catalog

    **Request:**
    ```bash
    GET /api/v1/catalog/search?q=customer+revenue&domain=customer-analytics&tags=pii
    ```

    **Response:**
    ```json
    {
      "results": [
        {
          "product_id": "customer-analytics-360",
          "name": "Customer Analytics 360",
          "domain": "customer-analytics",
          "description": "360-degree view of customer data...",
          "owner": "customer-team@company.com",
          "version": "2.3.0",
          "tags": ["customer", "analytics", "pii", "gdpr"],
          "sla": {
            "availability": 99.9,
            "freshness": "1 hour"
          },
          "quality_score": 0.98,
          "consumers": 87,
          "popularity_score": 9.2
        }
      ],
      "total_results": 1,
      "search_time_ms": 42
    }
    ```

    ---

    ### 4. Enforce Policy

    **Request (Internal API):**
    ```bash
    POST /api/v1/policy/evaluate
    Content-Type: application/json

    {
      "user": {
        "id": "user_123",
        "role": "analyst",
        "region": "US",
        "completed_pii_training": true
      },
      "action": "query",
      "resource": {
        "data_product": "customer-analytics-360",
        "columns": ["customer_id", "email", "lifetime_value"]
      }
    }
    ```

    **Response:**
    ```json
    {
      "decision": "allow",
      "policies_applied": [
        {
          "policy_id": "global.access-control.001",
          "decision": "allow",
          "reason": "User has completed PII training and has analyst role"
        },
        {
          "policy_id": "customer-analytics.data-privacy.002",
          "decision": "transform",
          "transformations": [
            {
              "column": "email",
              "action": "mask",
              "pattern": "***@domain.com"
            }
          ]
        }
      ],
      "evaluation_time_ms": 8
    }
    ```

    ---

    ## Database Schema

    ### Data Product Registry

    ```sql
    CREATE TABLE data_products (
        product_id VARCHAR(255) PRIMARY KEY,
        domain VARCHAR(100) NOT NULL,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        owner VARCHAR(255) NOT NULL,
        version VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL,  -- active, deprecated, archived
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        tags TEXT[],

        -- SLA commitments
        sla_availability DECIMAL(5,2),
        sla_freshness INTERVAL,
        sla_completeness DECIMAL(5,2),

        -- Metadata
        output_ports JSONB,
        input_ports JSONB,
        quality_rules JSONB,
        dependencies JSONB,

        UNIQUE(product_id, version)
    );

    CREATE INDEX idx_data_products_domain ON data_products(domain);
    CREATE INDEX idx_data_products_owner ON data_products(owner);
    CREATE INDEX idx_data_products_tags ON data_products USING GIN(tags);
    ```

    ### Data Contracts

    ```sql
    CREATE TABLE data_contracts (
        contract_id BIGSERIAL PRIMARY KEY,
        product_id VARCHAR(255) REFERENCES data_products(product_id),
        version VARCHAR(50) NOT NULL,
        schema_registry_id INT NOT NULL,
        schema_format VARCHAR(50) NOT NULL,  -- avro, protobuf, json-schema
        schema_definition JSONB NOT NULL,

        -- Contract terms
        backwards_compatible BOOLEAN DEFAULT true,
        breaking_changes TEXT[],

        -- Validation
        validation_rules JSONB,

        created_at TIMESTAMP DEFAULT NOW(),
        deprecated_at TIMESTAMP,

        UNIQUE(product_id, version)
    );
    ```

    ### Governance Policies

    ```sql
    CREATE TABLE governance_policies (
        policy_id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        domain VARCHAR(100),  -- NULL for global policies
        type VARCHAR(50) NOT NULL,  -- access-control, data-privacy, retention, quality
        policy_code TEXT NOT NULL,  -- Rego (OPA) policy
        severity VARCHAR(50) NOT NULL,  -- error, warning, info

        enabled BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_policies_domain ON governance_policies(domain);
    CREATE INDEX idx_policies_type ON governance_policies(type);
    ```

    ### Product Observability

    ```sql
    CREATE TABLE product_metrics (
        metric_id BIGSERIAL PRIMARY KEY,
        product_id VARCHAR(255) REFERENCES data_products(product_id),
        metric_name VARCHAR(255) NOT NULL,
        metric_value DECIMAL(20,5) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        dimensions JSONB,  -- Additional labels

        UNIQUE(product_id, metric_name, timestamp)
    );

    CREATE INDEX idx_product_metrics_timestamp ON product_metrics(timestamp DESC);
    CREATE INDEX idx_product_metrics_product ON product_metrics(product_id);

    -- Time-series optimization
    CREATE TABLE product_metrics_hourly (
        product_id VARCHAR(255),
        metric_name VARCHAR(255),
        hour TIMESTAMP,
        avg_value DECIMAL(20,5),
        min_value DECIMAL(20,5),
        max_value DECIMAL(20,5),
        count BIGINT,

        PRIMARY KEY (product_id, metric_name, hour)
    );
    ```

    ### Lineage Graph

    ```sql
    CREATE TABLE product_lineage (
        lineage_id BIGSERIAL PRIMARY KEY,
        source_product_id VARCHAR(255) REFERENCES data_products(product_id),
        target_product_id VARCHAR(255) REFERENCES data_products(product_id),
        lineage_type VARCHAR(50) NOT NULL,  -- direct, indirect, derived
        transformation_logic TEXT,
        created_at TIMESTAMP DEFAULT NOW(),

        UNIQUE(source_product_id, target_product_id)
    );

    CREATE INDEX idx_lineage_source ON product_lineage(source_product_id);
    CREATE INDEX idx_lineage_target ON product_lineage(target_product_id);
    ```

=== "🔧 Step 3: Deep Dive"

    ## 3.1 Data Product Implementation

    ### Product SDK (Self-Serve)

    ```python
    from datamesh import DataProduct, OutputPort, SLA, QualityRule

    class CustomerAnalytics360(DataProduct):
        """
        Customer 360 data product

        Provides unified view of customer data including:
        - Demographics
        - Behavior patterns
        - Purchase history
        - Lifetime value

        Owner: customer-team@company.com
        """

        def __init__(self):
            super().__init__(
                product_id="customer-analytics-360",
                domain="customer-analytics",
                version="2.3.0"
            )

            # Define SLA contract
            self.sla = SLA(
                availability=99.9,
                freshness_minutes=60,
                completeness=99.5,
                timeliness_percentile=95
            )

            # Define output ports
            self.add_output_port(
                OutputPort(
                    name="analytical",
                    type="batch",
                    format="parquet",
                    location="s3://data-mesh/customer-analytics/customer-360/",
                    schema=self.get_schema(),
                    partitions=["country", "date"]
                )
            )

            self.add_output_port(
                OutputPort(
                    name="api",
                    type="rest",
                    endpoint="/customer-analytics/customer-360/v2",
                    rate_limit=1000
                )
            )

            # Define quality rules
            self.add_quality_rule(
                QualityRule(
                    name="no_null_customer_id",
                    rule="customer_id IS NOT NULL",
                    severity="error"
                )
            )

            self.add_quality_rule(
                QualityRule(
                    name="valid_email",
                    rule="email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'",
                    severity="warning"
                )
            )

        def get_schema(self):
            """Define schema contract (Avro)"""
            return {
                "type": "record",
                "name": "Customer360",
                "namespace": "com.company.datamesh.customer",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "email", "type": ["null", "string"], "default": None},
                    {"name": "country", "type": "string"},
                    {"name": "registration_date", "type": "long", "logicalType": "timestamp-millis"},
                    {"name": "lifetime_value", "type": "double"},
                    {"name": "segment", "type": "string"},
                    {"name": "last_updated", "type": "long", "logicalType": "timestamp-millis"}
                ]
            }

        def transform(self, input_data):
            """
            Transformation logic to create data product

            This runs on self-serve platform (Spark/dbt)
            """
            from pyspark.sql import functions as F

            # Join multiple sources
            customers = input_data["customer-domain.raw.users"]
            interactions = input_data["events-domain.user-interactions"]
            orders = input_data["orders-domain.orders"]

            # Calculate metrics
            customer_360 = customers.alias("c") \
                .join(
                    interactions.alias("i"),
                    F.col("c.customer_id") == F.col("i.customer_id"),
                    "left"
                ) \
                .join(
                    orders.alias("o"),
                    F.col("c.customer_id") == F.col("o.customer_id"),
                    "left"
                ) \
                .groupBy("c.customer_id", "c.email", "c.country", "c.registration_date") \
                .agg(
                    F.sum("o.total_amount").alias("lifetime_value"),
                    F.count("i.interaction_id").alias("interaction_count"),
                    F.max("i.timestamp").alias("last_interaction")
                ) \
                .withColumn(
                    "segment",
                    F.when(F.col("lifetime_value") > 1000, "high_value")
                     .when(F.col("lifetime_value") > 100, "medium_value")
                     .otherwise("low_value")
                ) \
                .withColumn("last_updated", F.current_timestamp())

            return customer_360

        def validate_quality(self, df):
            """
            Run quality checks on output data

            Returns: (passed: bool, metrics: dict)
            """
            total_rows = df.count()

            # Check 1: No null customer_id
            null_customer_id = df.filter(F.col("customer_id").isNull()).count()
            if null_customer_id > 0:
                raise QualityCheckError(f"Found {null_customer_id} rows with null customer_id")

            # Check 2: Valid email format
            invalid_emails = df.filter(
                ~F.col("email").rlike(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
            ).count()
            invalid_email_pct = (invalid_emails / total_rows) * 100

            if invalid_email_pct > 5:
                raise QualityCheckWarning(f"{invalid_email_pct:.2f}% of emails are invalid")

            # Check 3: Positive lifetime value
            negative_ltv = df.filter(F.col("lifetime_value") < 0).count()
            if negative_ltv > 0:
                raise QualityCheckError(f"Found {negative_ltv} rows with negative lifetime_value")

            return True, {
                "total_rows": total_rows,
                "invalid_emails": invalid_emails,
                "quality_score": 1.0 - (invalid_emails / total_rows)
            }

    # Deploy data product
    if __name__ == "__main__":
        product = CustomerAnalytics360()

        # Platform automatically handles:
        # 1. Register in catalog
        # 2. Deploy infrastructure (Spark cluster)
        # 3. Schedule execution
        # 4. Monitor SLAs
        # 5. Enforce policies

        product.deploy()
    ```

    ---

    ### Data Contract Definition

    ```yaml
    # customer-360-contract.yaml
    apiVersion: datamesh.io/v1
    kind: DataContract
    metadata:
      product_id: customer-analytics-360
      version: 2.3.0

    schema:
      format: avro
      registry: schema-registry.datamesh.company.com
      compatibility: backward  # Allows adding optional fields, removing fields

      definition:
        type: record
        name: Customer360
        fields:
          - name: customer_id
            type: string
            doc: "Unique customer identifier"
            required: true
            pii: false

          - name: email
            type: ["null", "string"]
            doc: "Customer email address"
            required: false
            pii: true
            privacy_classification: sensitive

          - name: country
            type: string
            doc: "Customer country (ISO 3166-1 alpha-2)"
            required: true
            pii: false

          - name: lifetime_value
            type: double
            doc: "Total revenue from customer (USD)"
            required: true
            pii: false
            min: 0

          - name: segment
            type: string
            doc: "Customer segment"
            required: true
            pii: false
            enum: ["high_value", "medium_value", "low_value"]

    # SLA guarantees (binding contract)
    sla:
      availability:
        target: 99.9%
        measurement_window: 30 days
        exclusions:
          - planned_maintenance

      freshness:
        target: 1 hour
        measurement: max_age_of_data
        breach_threshold: 2 hours

      completeness:
        target: 99.5%
        measurement: (total_rows / expected_rows) * 100

      accuracy:
        target: 98%
        measurement: data_quality_score
        validation_rules:
          - no_null_required_fields
          - valid_email_format
          - positive_lifetime_value

    # Consumer expectations
    consumer_expectations:
      response_time:
        p50: 500ms
        p95: 2s
        p99: 5s

      throughput:
        queries_per_second: 100

      retention:
        duration: 2 years

    # Deprecation policy
    deprecation:
      notice_period: 90 days
      support_window: 180 days
      migration_guide: https://docs.company.com/migration/customer-360-v3
    ```

    ---

    ## 3.2 Federated Computational Governance

    ### Policy as Code (Open Policy Agent)

    ```rego
    # policies/global/access-control.rego
    package datamesh.access

    import future.keywords.if
    import future.keywords.in

    # Default deny
    default allow = false

    # Allow if user has required training and role
    allow if {
        # Check PII training for PII data products
        input.resource.tags[_] == "pii"
        input.user.completed_pii_training == true

        # Check role
        input.user.role in ["analyst", "data_scientist", "engineer"]
    }

    # Allow if user is product owner
    allow if {
        input.resource.owner == input.user.email
    }

    # Allow if user's team is in allowed teams
    allow if {
        input.user.team in input.resource.allowed_teams
    }

    # Deny if user is from restricted region
    deny[msg] if {
        input.resource.tags[_] == "gdpr"
        input.user.region == "CN"
        msg := "GDPR data cannot be accessed from China"
    }
    ```

    ```rego
    # policies/global/data-privacy.rego
    package datamesh.privacy

    import future.keywords.if

    # Mask email for non-EU users accessing EU data
    mask_email if {
        input.resource.tags[_] == "gdpr"
        input.user.region != "EU"
    }

    # Redact SSN for all users except compliance team
    redact_ssn if {
        input.user.role != "compliance_officer"
    }

    # Apply differential privacy for aggregate queries
    add_noise if {
        input.query.type == "aggregate"
        input.resource.tags[_] == "pii"
        count(input.query.group_by) < 3  # Small groups need noise
    }

    # Column-level transformations
    transformations[t] {
        mask_email
        t := {
            "column": "email",
            "action": "mask",
            "pattern": "***@***.com"
        }
    }

    transformations[t] {
        redact_ssn
        t := {
            "column": "ssn",
            "action": "redact",
            "replacement": "XXX-XX-XXXX"
        }
    }
    ```

    ```rego
    # policies/domain/customer-analytics/data-quality.rego
    package datamesh.domain.customer_analytics.quality

    import future.keywords.if

    # Domain-specific quality rules
    required_quality_score = 0.95

    # Reject data product update if quality score too low
    deny[msg] if {
        input.action == "update_data_product"
        input.metrics.quality_score < required_quality_score
        msg := sprintf("Quality score %.2f below required %.2f", [
            input.metrics.quality_score,
            required_quality_score
        ])
    }

    # Require data validation before publishing
    require_validation if {
        input.action == "publish_data_product"
        not input.validation_results.passed
    }
    ```

    ### Policy Enforcement Engine

    ```python
    from opa import OPAClient
    import logging

    class PolicyEngine:
        """
        Federated policy enforcement using Open Policy Agent

        Enforces:
        - Access control policies
        - Data privacy policies
        - Quality policies
        - Compliance policies
        """

        def __init__(self, opa_url="http://opa:8181"):
            self.opa = OPAClient(opa_url)
            self.policy_cache = {}

        def evaluate_access(self, user, data_product, action="query"):
            """
            Evaluate if user can access data product

            Returns: (allowed: bool, transformations: list)
            """
            # Build policy input
            policy_input = {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "team": user.team,
                    "region": user.region,
                    "completed_pii_training": user.completed_pii_training
                },
                "action": action,
                "resource": {
                    "data_product": data_product.id,
                    "domain": data_product.domain,
                    "owner": data_product.owner,
                    "tags": data_product.tags,
                    "allowed_teams": data_product.allowed_teams
                }
            }

            # Evaluate access control policies
            access_decision = self.opa.evaluate_policy(
                "datamesh/access/allow",
                policy_input
            )

            if not access_decision["result"]:
                # Check deny reasons
                deny_reasons = self.opa.evaluate_policy(
                    "datamesh/access/deny",
                    policy_input
                )
                logging.warning(f"Access denied: {deny_reasons}")
                return False, []

            # Evaluate privacy policies (get transformations)
            privacy_transformations = self.opa.evaluate_policy(
                "datamesh/privacy/transformations",
                policy_input
            )

            return True, privacy_transformations["result"]

        def validate_data_quality(self, data_product, metrics):
            """
            Validate data quality against policies

            Returns: (passed: bool, violations: list)
            """
            policy_input = {
                "action": "update_data_product",
                "data_product": {
                    "id": data_product.id,
                    "domain": data_product.domain
                },
                "metrics": {
                    "quality_score": metrics["quality_score"],
                    "completeness": metrics["completeness"],
                    "freshness_minutes": metrics["freshness_minutes"]
                }
            }

            # Evaluate domain-specific quality policies
            domain_policy = f"datamesh/domain/{data_product.domain}/quality/deny"
            violations = self.opa.evaluate_policy(domain_policy, policy_input)

            if violations["result"]:
                return False, violations["result"]

            return True, []

        def enforce_retention(self, data_product, record):
            """
            Enforce data retention policies

            Returns: should_delete: bool
            """
            policy_input = {
                "data_product": {
                    "id": data_product.id,
                    "retention_days": data_product.sla.retention_days
                },
                "record": {
                    "created_at": record.created_at,
                    "age_days": (datetime.now() - record.created_at).days
                }
            }

            should_delete = self.opa.evaluate_policy(
                "datamesh/retention/should_delete",
                policy_input
            )

            return should_delete["result"]
    ```

    ---

    ## 3.3 Self-Serve Infrastructure

    ### Infrastructure as Code (Terraform)

    ```hcl
    # terraform/data-product-module/main.tf
    # Self-serve module for data product deployment

    variable "product_id" {
      type = string
    }

    variable "domain" {
      type = string
    }

    variable "compute_size" {
      type    = string
      default = "medium"
    }

    variable "schedule" {
      type    = string
      default = "0 * * * *"  # Hourly
    }

    # Create namespace for domain isolation
    resource "kubernetes_namespace" "domain_namespace" {
      metadata {
        name = "domain-${var.domain}"
        labels = {
          domain = var.domain
        }
      }
    }

    # Deploy Spark job for data product transformation
    resource "kubernetes_deployment" "data_product_job" {
      metadata {
        name      = "${var.product_id}-job"
        namespace = kubernetes_namespace.domain_namespace.metadata[0].name
      }

      spec {
        replicas = 1

        selector {
          match_labels = {
            app         = var.product_id
            domain      = var.domain
          }
        }

        template {
          metadata {
            labels = {
              app    = var.product_id
              domain = var.domain
            }
          }

          spec {
            container {
              name  = "spark-driver"
              image = "datamesh/spark:3.5.0"

              env {
                name  = "PRODUCT_ID"
                value = var.product_id
              }

              env {
                name  = "DOMAIN"
                value = var.domain
              }

              resources {
                requests = {
                  cpu    = var.compute_size == "small" ? "2" : "8"
                  memory = var.compute_size == "small" ? "8Gi" : "32Gi"
                }
                limits = {
                  cpu    = var.compute_size == "small" ? "4" : "16"
                  memory = var.compute_size == "small" ? "16Gi" : "64Gi"
                }
              }
            }

            service_account_name = "data-product-sa"
          }
        }
      }
    }

    # Create CronJob for scheduled execution
    resource "kubernetes_cron_job" "data_product_schedule" {
      metadata {
        name      = "${var.product_id}-schedule"
        namespace = kubernetes_namespace.domain_namespace.metadata[0].name
      }

      spec {
        schedule = var.schedule

        job_template {
          metadata {
            labels = {
              app    = var.product_id
              domain = var.domain
            }
          }

          spec {
            template {
              metadata {}

              spec {
                container {
                  name  = "trigger"
                  image = "datamesh/trigger:latest"
                  args  = ["--product-id", var.product_id]
                }

                restart_policy = "OnFailure"
              }
            }
          }
        }
      }
    }

    # Create service for API access
    resource "kubernetes_service" "data_product_api" {
      metadata {
        name      = "${var.product_id}-api"
        namespace = kubernetes_namespace.domain_namespace.metadata[0].name
      }

      spec {
        selector = {
          app = var.product_id
        }

        port {
          port        = 8080
          target_port = 8080
        }

        type = "ClusterIP"
      }
    }

    # Outputs
    output "api_endpoint" {
      value = "https://api.datamesh.company.com/${var.domain}/${var.product_id}"
    }

    output "namespace" {
      value = kubernetes_namespace.domain_namespace.metadata[0].name
    }
    ```

    ### CI/CD Pipeline (GitHub Actions)

    ```yaml
    # .github/workflows/deploy-data-product.yml
    name: Deploy Data Product

    on:
      push:
        branches:
          - main
        paths:
          - 'products/**'

    jobs:
      validate:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3

          - name: Validate Data Contract
            run: |
              # Validate schema compatibility
              datamesh validate-contract \
                --product-id ${{ matrix.product_id }} \
                --schema-file products/${{ matrix.product_id }}/schema.avsc

          - name: Run Policy Checks
            run: |
              # Check policy compliance
              datamesh check-policies \
                --product-id ${{ matrix.product_id }} \
                --policies-dir policies/

          - name: Run Quality Tests
            run: |
              # Run data quality tests
              pytest tests/quality/${{ matrix.product_id }}/

      deploy:
        needs: validate
        runs-on: ubuntu-latest
        steps:
          - name: Deploy Infrastructure
            run: |
              # Deploy using Terraform module
              terraform init
              terraform apply -auto-approve \
                -var="product_id=${{ matrix.product_id }}" \
                -var="domain=${{ matrix.domain }}"

          - name: Register in Catalog
            run: |
              # Register data product in catalog
              datamesh register-product \
                --product-file products/${{ matrix.product_id }}/data-product.yaml

          - name: Update Lineage
            run: |
              # Update lineage graph
              datamesh update-lineage \
                --product-id ${{ matrix.product_id }}

      monitor:
        needs: deploy
        runs-on: ubuntu-latest
        steps:
          - name: Setup SLA Monitoring
            run: |
              # Configure alerts for SLA violations
              datamesh setup-monitoring \
                --product-id ${{ matrix.product_id }} \
                --alert-channel slack
    ```

    ---

    ## 3.4 Data Product Interoperability

    ### Schema Registry Integration

    ```python
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer

    class SchemaRegistry:
        """
        Central schema registry for data product interoperability

        Ensures:
        - Schema compatibility (backward, forward, full)
        - Version management
        - Schema evolution
        """

        def __init__(self, registry_url):
            self.client = SchemaRegistryClient({"url": registry_url})

        def register_schema(self, product_id, schema, compatibility="BACKWARD"):
            """
            Register schema for data product

            Compatibility modes:
            - BACKWARD: Consumers using new schema can read old data
            - FORWARD: Consumers using old schema can read new data
            - FULL: Both backward and forward compatible
            - NONE: No compatibility checks
            """
            subject = f"{product_id}-value"

            # Set compatibility level
            self.client.set_compatibility(subject, compatibility)

            # Register schema
            schema_id = self.client.register_schema(subject, schema)

            return schema_id

        def validate_compatibility(self, product_id, new_schema):
            """
            Validate if new schema is compatible with existing versions

            Returns: (compatible: bool, errors: list)
            """
            subject = f"{product_id}-value"

            try:
                # Check compatibility
                is_compatible = self.client.test_compatibility(
                    subject, new_schema
                )

                if not is_compatible:
                    # Get incompatibility reasons
                    latest_version = self.client.get_latest_version(subject)
                    errors = self._compare_schemas(
                        latest_version.schema, new_schema
                    )
                    return False, errors

                return True, []

            except Exception as e:
                return False, [str(e)]

        def _compare_schemas(self, old_schema, new_schema):
            """
            Compare schemas and identify breaking changes
            """
            errors = []

            old_fields = {f["name"]: f for f in old_schema["fields"]}
            new_fields = {f["name"]: f for f in new_schema["fields"]}

            # Check for removed required fields (breaking change)
            for field_name, field in old_fields.items():
                if field_name not in new_fields:
                    if "default" not in field:
                        errors.append(
                            f"Removed required field '{field_name}' without default value"
                        )

            # Check for type changes (breaking change)
            for field_name in old_fields.keys() & new_fields.keys():
                old_type = old_fields[field_name]["type"]
                new_type = new_fields[field_name]["type"]

                if old_type != new_type:
                    errors.append(
                        f"Changed type of field '{field_name}' from {old_type} to {new_type}"
                    )

            return errors
    ```

    ### Cross-Domain Query Federation

    ```python
    class FederatedQueryEngine:
        """
        Execute queries across multiple data products/domains

        Handles:
        - Query planning across domains
        - Data product discovery
        - Policy enforcement
        - Result aggregation
        """

        def __init__(self, catalog, policy_engine):
            self.catalog = catalog
            self.policy_engine = policy_engine

        def execute_federated_query(self, sql, user):
            """
            Execute query that spans multiple domains

            Example:
            SELECT c.customer_id, c.country, o.total_orders
            FROM customer_analytics.customer_360 c
            JOIN orders_domain.order_summary o
              ON c.customer_id = o.customer_id
            WHERE c.country = 'US'
            """
            # 1. Parse SQL and identify data products
            parsed = self._parse_sql(sql)
            data_products = parsed["data_products"]

            # 2. Check access for each data product
            for product_id in data_products:
                product = self.catalog.get_product(product_id)

                allowed, transformations = self.policy_engine.evaluate_access(
                    user, product, action="query"
                )

                if not allowed:
                    raise PermissionError(
                        f"User {user.id} not allowed to access {product_id}"
                    )

            # 3. Generate query plan
            query_plan = self._generate_query_plan(parsed, data_products)

            # 4. Execute sub-queries on each domain
            results = {}
            for domain, sub_query in query_plan.items():
                domain_engine = self._get_domain_engine(domain)
                results[domain] = domain_engine.execute(sub_query)

            # 5. Aggregate results
            final_result = self._aggregate_results(results, query_plan)

            # 6. Apply transformations (masking, redaction)
            final_result = self._apply_transformations(
                final_result, transformations
            )

            return final_result

        def _generate_query_plan(self, parsed, data_products):
            """
            Generate distributed query plan

            Strategy:
            - Push down filters to source domains
            - Minimize data transfer
            - Execute joins locally where possible
            """
            plan = {}

            for product_id in data_products:
                product = self.catalog.get_product(product_id)
                domain = product.domain

                # Extract filters for this product
                filters = [
                    f for f in parsed["filters"]
                    if f["table"] == product_id
                ]

                # Build sub-query
                sub_query = {
                    "product": product_id,
                    "columns": parsed["columns"][product_id],
                    "filters": filters
                }

                plan[domain] = sub_query

            return plan
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scaling Data Mesh

    ### Domain Autonomy vs. Standardization

    | Aspect | Centralized | Federated (Data Mesh) | Trade-off |
    |--------|-------------|---------------------|-----------|
    | **Ownership** | Central data team | Domain teams | Autonomy vs consistency |
    | **Velocity** | Slow (bottleneck) | Fast (parallel) | Speed vs coordination overhead |
    | **Standards** | Enforced globally | Federated policies | Flexibility vs interoperability |
    | **Technology** | Single stack | Polyglot persistence | Best-fit vs complexity |
    | **Quality** | Centrally validated | Domain responsibility | Speed vs risk |

    **Recommendation:** Start with global interoperability standards (schemas, formats, APIs), allow domain autonomy for implementation details.

    ---

    ## Performance Optimization

    ### Data Product Caching

    ```python
    class DataProductCache:
        """
        Distributed cache for data product outputs

        Benefits:
        - Reduce recomputation (save 80% compute)
        - Faster cross-domain queries (10x speedup)
        - Lower costs (cache hits avoid warehouse queries)
        """

        def __init__(self, redis_cluster):
            self.cache = redis_cluster
            self.ttl_seconds = 3600  # 1 hour default

        def get_cached_result(self, product_id, query_hash):
            """Get cached query result if fresh enough"""
            cache_key = f"product:{product_id}:query:{query_hash}"

            cached = self.cache.get(cache_key)
            if cached:
                result = json.loads(cached)

                # Check freshness against SLA
                product = catalog.get_product(product_id)
                age_seconds = time.time() - result["cached_at"]

                if age_seconds < product.sla.freshness_seconds:
                    return result["data"]

            return None

        def cache_result(self, product_id, query_hash, data):
            """Cache query result with TTL"""
            cache_key = f"product:{product_id}:query:{query_hash}"

            cached_data = {
                "data": data,
                "cached_at": time.time()
            }

            self.cache.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(cached_data)
            )
    ```

    ---

    ## Cost Optimization

    ```
    Monthly Cost (100 domains, 1,000 products):

    Compute (Self-Serve Platform):
    - Kubernetes cluster: 100 nodes × $150 = $15,000
    - Data product builds: 1,000 products × $5 = $5,000
    - Query engines (shared): $20,000
    - Total compute: $40,000/month

    Storage:
    - Data products: 10 PB × $0.023/GB = $230,000
    - With compression (5:1): $46,000
    - Metadata: negligible
    - Total storage: $46,000/month

    Governance & Catalog:
    - Policy engine: $2,000
    - Data catalog: $3,000
    - Schema registry: $1,000
    - Total governance: $6,000/month

    Total: ~$92,000/month

    Optimizations:
    1. Incremental data products (reduce recomputation): -$20K
    2. Spot instances for batch jobs: -$10K
    3. Storage tiering (cold data to Glacier): -$15K
    4. Query result caching (reduce warehouse costs): -$10K

    Optimized Total: ~$37,000/month (60% savings)
    ```

    ---

    ## Monitoring & Observability

    ### Data Product Health Dashboard

    ```python
    # Metrics to track per data product

    # Availability metrics
    product_availability_percent{product_id, domain}
    product_uptime_seconds{product_id}
    product_downtime_incidents{product_id}

    # Freshness metrics
    product_data_age_minutes{product_id}
    product_sla_freshness_breaches{product_id}

    # Quality metrics
    product_quality_score{product_id}
    product_validation_failures{product_id, rule}
    product_completeness_percent{product_id}

    # Usage metrics
    product_queries_total{product_id, consumer_domain}
    product_bytes_scanned{product_id}
    product_consumers_count{product_id}

    # Performance metrics
    product_query_duration_seconds{product_id, percentile}
    product_build_duration_seconds{product_id}

    # Cost metrics
    product_compute_cost_usd{product_id}
    product_storage_cost_usd{product_id}
    ```

    ---

    ## Interview Tips

    **Common Follow-up Questions:**

    1. **"What's the difference between data mesh and data lake?"**
       - Data lake: Centralized storage, single team owns all data
       - Data mesh: Decentralized, domain teams own their data products
       - Data lake: Technology-focused (storage)
       - Data mesh: Organizational pattern (ownership + technology)

    2. **"How do you handle cross-domain queries?"**
       - Federated query engine routes to appropriate domains
       - Each domain enforces its own policies
       - Results aggregated in coordination layer
       - Push down filters to minimize data transfer
       - Cache frequently joined products

    3. **"How do you enforce data quality across domains?"**
       - Data contracts define quality SLAs
       - Computational policies enforce rules automatically
       - Domain teams responsible for meeting SLAs
       - Central observability tracks quality metrics
       - Consumers can see quality scores before using data

    4. **"What if domains use different technologies?"**
       - That's expected! Polyglot persistence
       - Require standard output formats (Parquet, Avro)
       - Standard APIs (REST, GraphQL)
       - Schema registry ensures interoperability
       - Self-serve platform abstracts differences

    5. **"How do you prevent data duplication?"**
       - Domain ownership is clear (single source of truth per domain)
       - Catalog shows lineage (derived vs source products)
       - Products can reference other products (don't copy)
       - Versioning prevents divergence
       - Trade-off: Some duplication OK for performance

    6. **"How do you handle GDPR/compliance?"**
       - Federated computational governance
       - Policies as code (OPA)
       - Automatic enforcement (not manual)
       - Column-level transformations (masking, redaction)
       - Audit logs for all access

    7. **"How do you onboard new domains?"**
       - Self-serve platform (1 day to first product)
       - Product SDK (Python/Go library)
       - Templates and examples
       - CI/CD automation
       - Domain teams are independent

    8. **"What are the biggest challenges?"**
       - Cultural shift (from centralized to distributed)
       - Maintaining interoperability standards
       - Avoiding domain silos
       - Governance complexity
       - Initial platform investment

    **Key Points to Mention:**

    - Four principles: domain ownership, data as product, self-serve platform, federated governance
    - Decentralized architecture (not centralized data lake)
    - Data contracts ensure interoperability
    - Policies as code (computational governance)
    - Self-serve infrastructure (domain autonomy)
    - Observability for SLA monitoring
    - Schema registry for schema evolution
    - Polyglot persistence (domains choose storage)

    ---

    ## Real-World Examples

    **Netflix:**
    - 1,000+ data products across domains
    - Self-serve platform (Metacat, Genie)
    - Federated ownership (domain teams)
    - Polyglot storage (Cassandra, Elasticsearch, S3)

    **Zalando:**
    - Pioneer in data mesh adoption
    - 100+ domains with data products
    - Self-serve platform on Kubernetes
    - Contract-first approach

    **ThoughtWorks:**
    - Coined "data mesh" term (Zhamak Dehghani)
    - Consulting on enterprise implementations
    - Focus on organizational change

    **Intuit:**
    - Data product platform
    - Federated governance with OPA
    - Cross-domain discovery and lineage

    ---

    ## Summary

    **System Characteristics:**

    - **Scale:** 100+ domains, 1,000+ data products, 10 PB data
    - **Latency:** < 1s catalog search, < 10s cross-domain queries
    - **Autonomy:** Domain teams deploy products independently
    - **Governance:** 100% policy compliance with federated control

    **Core Components:**

    1. **Domain Data Products:** Owned by domain teams with SLA contracts
    2. **Self-Serve Platform:** Kubernetes-based infrastructure as code
    3. **Data Catalog:** Centralized discovery (OpenMetadata)
    4. **Policy Engine:** Computational governance (Open Policy Agent)
    5. **Schema Registry:** Interoperability and evolution
    6. **Observability Hub:** SLA monitoring and quality tracking

    **Key Design Decisions:**

    - Decentralized ownership (not centralized data team)
    - Data contracts with SLAs (not just schemas)
    - Federated computational governance (policies as code)
    - Self-serve infrastructure (domain autonomy)
    - Polyglot persistence (best storage per domain)
    - Product-centric (not table-centric)
    - Cross-domain interoperability via standards

    **Benefits:**

    - Scale through decentralization (100+ domains)
    - Faster delivery (domain autonomy)
    - Better data quality (domain expertise)
    - Flexible technology choices (polyglot)
    - Automatic policy enforcement

    **Trade-offs:**

    - Higher coordination overhead
    - More complex governance
    - Potential data duplication
    - Requires cultural change
    - Initial platform investment

    This design provides a scalable, decentralized data architecture that treats data as a product with domain ownership, self-serve infrastructure, and federated computational governance to enable enterprise-wide data sharing and analytics.
