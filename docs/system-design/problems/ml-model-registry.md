# Design an ML Model Registry

A centralized model registry system that enables ML teams to version, manage, approve, and track machine learning models from experimentation through production deployment, with metadata management, approval workflows, staging environments, deployment tracking, and lineage tracing.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K models, 100K versions, 1K daily deployments, 500+ teams |
| **Key Challenges** | Model versioning, approval workflows, artifact management, deployment tracking, lineage from experiment to production, multi-environment support |
| **Core Concepts** | Semantic versioning, state machines, artifact storage (S3), metadata DB, approval workflows, deployment hooks, A/B test integration |
| **Companies** | MLflow Model Registry, Neptune, Weights & Biases Registry, AWS SageMaker Model Registry, Vertex AI Model Registry |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Model Registration** | Register models with name, version, framework, schema | P0 (Must have) |
    | **Version Management** | Semantic versioning (major.minor.patch), auto-incrementing | P0 (Must have) |
    | **Stage Transitions** | Move models through dev/staging/production stages | P0 (Must have) |
    | **Approval Workflows** | Multi-step approval process with reviewers | P0 (Must have) |
    | **Artifact Management** | Store model files, metadata, checksums (S3/GCS) | P0 (Must have) |
    | **Deployment Tracking** | Track where/when models are deployed | P0 (Must have) |
    | **Model Lineage** | Trace model from experiment, dataset, training run | P0 (Must have) |
    | **Model Search** | Find models by name, tags, metrics, stage | P0 (Must have) |
    | **Webhook Integration** | Notify external systems on model events | P1 (Should have) |
    | **A/B Test Integration** | Support traffic splitting between model versions | P1 (Should have) |
    | **Model Aliases** | Human-readable aliases (champion, challenger) | P1 (Should have) |
    | **Access Control** | Role-based permissions per model | P2 (Nice to have) |
    | **Model Cards** | Auto-generated documentation with metrics, schema | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (focus on registry only)
    - Model serving/inference (registry provides metadata to serving systems)
    - Experiment tracking (focus on production models, not experiments)
    - Feature engineering pipelines
    - Model monitoring and drift detection (registry tracks deployment, not runtime metrics)
    - Data labeling and annotation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Critical for deployment pipelines |
    | **Registration Latency** | < 5 seconds for model upload | Fast CI/CD workflows |
    | **Query Latency** | < 100ms for metadata queries | Quick model lookups |
    | **Artifact Download** | 1GB in < 2 minutes | Fast model deployment |
    | **Consistency** | Strong consistency for stage transitions | No deployment conflicts |
    | **Audit Trail** | 100% complete history of changes | Compliance requirements |
    | **Scalability** | 10K models, 100K versions | Enterprise-scale adoption |
    | **Data Durability** | 99.999999999% (S3 standard) | Cannot lose model artifacts |
    | **Concurrent Deployments** | 100+ simultaneous deployments | Large teams working in parallel |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Total models: 10,000 unique models
    - Total versions: 100,000 model versions (avg 10 versions per model)
    - Active teams: 500 teams
    - Data scientists: 2,000 users

    Model operations:
    - New model registrations: 50 registrations/day
    - New version registrations: 500 versions/day
    - Stage transitions: 1,000 transitions/day (dev → staging → production)
    - Model queries (metadata): 10,000 queries/day
    - Artifact downloads: 2,000 downloads/day
    - Approval requests: 200 approvals/day

    Deployment tracking:
    - Production deployments: 100 deployments/day
    - Staging deployments: 300 deployments/day
    - Dev deployments: 600 deployments/day
    - Total: 1,000 deployments/day

    QPS calculations:
    - Model registrations: 500 / 86,400 = 0.006 QPS
    - Metadata queries: 10,000 / 86,400 = 0.12 QPS
    - Artifact downloads: 2,000 / 86,400 = 0.023 QPS
    - Total write QPS: ~0.5 QPS
    - Total read QPS: ~5 QPS
    - Read-to-write ratio: 10:1

    Webhook events:
    - Events per day: 1,500 events (registrations + transitions + deployments)
    - Webhooks per event: 2 webhooks average
    - Total webhook calls: 3,000 calls/day = 0.035 QPS
    ```

    ### Storage Estimates

    ```
    Model artifacts (S3):
    - Average model size: 500 MB (deep learning models)
    - Small models: 20% × 100K versions × 10 MB = 200 GB
    - Medium models: 50% × 100K versions × 500 MB = 25 TB
    - Large models: 30% × 100K versions × 2 GB = 60 TB
    - Total artifacts: 85 TB
    - With deduplication (10% savings): 76.5 TB

    Metadata storage (PostgreSQL):
    - Model metadata: 10K models × 5 KB = 50 MB
    - Version metadata: 100K versions × 10 KB = 1 GB
    - Stage transitions: 100K versions × 5 stages × 500 bytes = 250 MB
    - Approval records: 50K approvals × 2 KB = 100 MB
    - Deployment records: 365K deployments/year × 1 KB = 365 MB
    - Lineage records: 100K versions × 2 KB = 200 MB
    - Audit logs: 500K events/year × 1 KB = 500 MB
    - Total metadata: ~2.5 GB
    - With indexes: ~5 GB

    Checksums and signatures:
    - SHA256 per artifact: 100K × 64 bytes = 6.4 MB
    - Signatures: 100K × 256 bytes = 25.6 MB

    Total storage: 76.5 TB (artifacts) + 5 GB (metadata) ≈ 76.5 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (uploads):
    - Model uploads: 500 versions/day × 500 MB = 250 GB/day
    - Average: 250 GB / 86,400 = 2.9 MB/sec ≈ 23 Mbps
    - Peak (10x): 230 Mbps

    Egress (downloads):
    - Model downloads: 2,000 downloads/day × 500 MB = 1 TB/day
    - Average: 1 TB / 86,400 = 11.6 MB/sec ≈ 93 Mbps
    - Peak (5x): 465 Mbps

    Metadata traffic:
    - Queries: 10,000 queries/day × 10 KB = 100 MB/day
    - Negligible compared to artifact traffic

    Webhook traffic:
    - 3,000 calls/day × 5 KB = 15 MB/day
    - Negligible

    Total bandwidth: ~600 Mbps peak (1 Gbps link sufficient)
    ```

    ### Memory Estimates

    ```
    Application servers:
    - In-memory cache (hot model metadata): 2 GB
    - Connection pools: 500 MB
    - Per-server memory: 8 GB
    - Total servers (3 replicas): 24 GB

    Database (PostgreSQL):
    - Shared buffers (25% of data): 2 GB
    - Connection memory: 2 GB
    - Total DB memory: 8 GB

    Redis cache:
    - Hot model metadata: 500 MB
    - Session storage: 500 MB
    - Total: 2 GB

    Total memory: 24 GB (app) + 8 GB (DB) + 2 GB (Redis) ≈ 34 GB
    ```

    ---

    ## Key Assumptions

    1. Average model size is 500 MB (deep learning models dominate)
    2. 80% of models have < 20 versions, 20% have > 20 versions
    3. Only 10% of registered models reach production
    4. Models stay in staging for 7-14 days average
    5. Approval process takes 1-3 days on average
    6. 10% of artifacts are duplicates (same model, different metadata)
    7. Most queries are for production models (hot data)
    8. Deployments happen during business hours (bursty traffic)
    9. Large models (> 2GB) are 30% of total
    10. Webhook delivery has 3 retries with exponential backoff

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Immutability:** Once registered, model artifacts are immutable (version if changes needed)
    2. **State machine:** Models transition through well-defined stages with validation
    3. **Artifact-metadata separation:** Store artifacts in S3, metadata in PostgreSQL
    4. **Event-driven:** Emit events for all model operations (registration, transition, deployment)
    5. **Audit trail:** Complete history of all model operations for compliance
    6. **Extensibility:** Plugin architecture for custom validators, approval workflows
    7. **Multi-tenancy:** Isolation between teams with shared infrastructure

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            MLflowClient[MLflow Client<br/>Python SDK]
            WebUI[Web UI<br/>React Dashboard]
            CICD[CI/CD Pipeline<br/>GitHub Actions]
        end

        subgraph "API Layer"
            APIGateway[API Gateway<br/>REST/gRPC<br/>Auth + Rate Limiting]
            RegistryAPI[Registry API<br/>FastAPI<br/>CRUD operations]
        end

        subgraph "Core Services"
            VersionManager[Version Manager<br/>Semantic versioning<br/>Auto-increment]
            StageManager[Stage Manager<br/>State machine<br/>Transition rules]
            ApprovalEngine[Approval Engine<br/>Workflow orchestration<br/>Multi-step approval]
            DeploymentTracker[Deployment Tracker<br/>Track deployments<br/>Rollback support]
        end

        subgraph "Supporting Services"
            LineageService[Lineage Service<br/>Track experiment → model<br/>Dataset → model]
            SearchService[Search Service<br/>Elasticsearch<br/>Full-text search]
            WebhookService[Webhook Service<br/>Event dispatch<br/>Retry mechanism]
            ArtifactManager[Artifact Manager<br/>Upload/download<br/>Checksum validation]
        end

        subgraph "Storage Layer"
            MetadataDB[(Metadata DB<br/>PostgreSQL<br/>Models, versions, stages)]
            ArtifactStore[(Artifact Store<br/>S3/GCS<br/>Model files)]
            CacheDB[(Cache<br/>Redis<br/>Hot metadata)]
            SearchIndex[(Search Index<br/>Elasticsearch<br/>Model search)]
        end

        subgraph "Event System"
            EventBus[Event Bus<br/>Kafka/RabbitMQ<br/>Model events]
            AuditLogger[Audit Logger<br/>Complete audit trail]
        end

        subgraph "External Integrations"
            ServingSystem[Model Serving<br/>TensorFlow Serving<br/>SageMaker]
            ExperimentTracking[Experiment Tracking<br/>MLflow/W&B]
            Notifications[Notifications<br/>Slack/Email]
        end

        MLflowClient -->|1. Register model| APIGateway
        WebUI -->|1. Register model| APIGateway
        CICD -->|1. Register model| APIGateway

        APIGateway -->|2. Route request| RegistryAPI

        RegistryAPI -->|3. Create version| VersionManager
        VersionManager -->|4. Generate version| RegistryAPI

        RegistryAPI -->|5. Upload artifact| ArtifactManager
        ArtifactManager -->|6. Store file| ArtifactStore
        ArtifactManager -->|7. Compute checksum| ArtifactStore

        RegistryAPI -->|8. Store metadata| MetadataDB
        RegistryAPI -->|9. Cache metadata| CacheDB
        RegistryAPI -->|10. Index for search| SearchIndex

        RegistryAPI -->|11. Publish event| EventBus
        EventBus -->|12. Process event| WebhookService
        EventBus -->|13. Log to audit| AuditLogger
        WebhookService -->|14. Notify| Notifications

        WebUI -->|15. Request approval| ApprovalEngine
        ApprovalEngine -->|16. Update status| MetadataDB
        ApprovalEngine -->|17. Publish event| EventBus

        WebUI -->|18. Transition stage| StageManager
        StageManager -->|19. Validate transition| MetadataDB
        StageManager -->|20. Update stage| MetadataDB
        StageManager -->|21. Publish event| EventBus

        ServingSystem -->|22. Deploy model| DeploymentTracker
        DeploymentTracker -->|23. Record deployment| MetadataDB
        DeploymentTracker -->|24. Publish event| EventBus

        WebUI -->|25. View lineage| LineageService
        LineageService -->|26. Query lineage| MetadataDB
        LineageService -.->|27. Link to experiments| ExperimentTracking

        WebUI -->|28. Search models| SearchService
        SearchService -->|29. Query index| SearchIndex

        AuditLogger -->|30. Write logs| MetadataDB

        style ArtifactStore fill:#ffe1e1
        style MetadataDB fill:#e1f5ff
        style EventBus fill:#fff4e1
        style ApprovalEngine fill:#f3e5f5
        style StageManager fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PostgreSQL (Metadata)** | ACID transactions for stage transitions, complex queries, strong consistency | DynamoDB (weak consistency, no transactions), MongoDB (no ACID for multi-doc) |
    | **S3 (Artifacts)** | Cheap, durable (11 9's), scalable (PB+), versioning support | HDFS (operational overhead), Database (not for large blobs), EFS (expensive) |
    | **Redis (Cache)** | Sub-millisecond reads for hot models, reduce DB load | Memcached (no persistence), in-memory only (lost on restart) |
    | **Elasticsearch (Search)** | Full-text search, complex filters, aggregations | PostgreSQL full-text (slower), custom indexing (complex) |
    | **Kafka (Events)** | Reliable event streaming, replay capability, ordered delivery | RabbitMQ (no replay), webhooks only (no internal consumers) |
    | **Semantic Versioning** | Clear communication of changes (breaking vs. non-breaking) | Auto-incrementing only (no semantic meaning), timestamps (hard to interpret) |

    **Key Trade-off:** We chose **strong consistency for stage transitions over performance** to prevent deployment conflicts (two teams deploying same model to production simultaneously). This adds latency (~50ms) but ensures correctness.

    ---

    ## Database Schema

    ### Metadata DB (PostgreSQL)

    ```sql
    -- Models (top-level entity)
    CREATE TABLE models (
        model_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        description TEXT,
        team_id VARCHAR(50),
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        tags JSONB,  -- Flexible tags for categorization

        CONSTRAINT unique_model_name UNIQUE (name)
    );
    CREATE INDEX idx_models_team ON models(team_id);
    CREATE INDEX idx_models_created ON models(created_at DESC);
    CREATE INDEX idx_models_tags ON models USING GIN(tags);

    -- Model versions
    CREATE TABLE model_versions (
        version_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(50) REFERENCES models(model_id),
        version VARCHAR(50) NOT NULL,  -- Semantic version: 1.2.3
        stage VARCHAR(20) DEFAULT 'dev',  -- dev, staging, production, archived
        status VARCHAR(20) DEFAULT 'pending_approval',  -- pending_approval, approved, rejected, active, archived

        -- Artifact information
        artifact_uri VARCHAR(1000),  -- S3 path
        artifact_size_bytes BIGINT,
        artifact_hash VARCHAR(64),  -- SHA256 checksum

        -- Model metadata
        framework VARCHAR(50),  -- tensorflow, pytorch, sklearn, onnx
        framework_version VARCHAR(50),
        input_schema JSONB,
        output_schema JSONB,
        signature JSONB,  -- MLflow signature format

        -- Lineage
        source_experiment_id VARCHAR(50),  -- From experiment tracking
        source_run_id VARCHAR(50),
        dataset_id VARCHAR(50),
        parent_version_id VARCHAR(50) REFERENCES model_versions(version_id),

        -- Metadata
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        tags JSONB,
        notes TEXT,

        CONSTRAINT unique_model_version UNIQUE (model_id, version)
    );
    CREATE INDEX idx_versions_model ON model_versions(model_id);
    CREATE INDEX idx_versions_stage ON model_versions(stage);
    CREATE INDEX idx_versions_status ON model_versions(status);
    CREATE INDEX idx_versions_created ON model_versions(created_at DESC);
    CREATE INDEX idx_versions_experiment ON model_versions(source_experiment_id);

    -- Stage transitions (history)
    CREATE TABLE stage_transitions (
        transition_id VARCHAR(50) PRIMARY KEY,
        version_id VARCHAR(50) REFERENCES model_versions(version_id),
        from_stage VARCHAR(20),
        to_stage VARCHAR(20) NOT NULL,
        transitioned_by VARCHAR(100),
        transitioned_at TIMESTAMP DEFAULT NOW(),
        reason TEXT,
        metadata JSONB
    );
    CREATE INDEX idx_transitions_version ON stage_transitions(version_id);
    CREATE INDEX idx_transitions_time ON stage_transitions(transitioned_at DESC);

    -- Approval workflows
    CREATE TABLE approvals (
        approval_id VARCHAR(50) PRIMARY KEY,
        version_id VARCHAR(50) REFERENCES model_versions(version_id),
        approval_type VARCHAR(50),  -- stage_transition, production_deployment
        status VARCHAR(20),  -- pending, approved, rejected

        -- Approvers
        required_approvers VARCHAR(100)[],  -- Array of user IDs
        approved_by VARCHAR(100)[],
        rejected_by VARCHAR(100),

        -- Approval chain
        approval_level INTEGER DEFAULT 1,  -- Multi-level approvals
        total_levels INTEGER DEFAULT 1,

        requested_by VARCHAR(100),
        requested_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP,
        notes TEXT,
        metadata JSONB
    );
    CREATE INDEX idx_approvals_version ON approvals(version_id);
    CREATE INDEX idx_approvals_status ON approvals(status);
    CREATE INDEX idx_approvals_requested ON approvals(requested_at DESC);

    -- Deployments (track where models are deployed)
    CREATE TABLE deployments (
        deployment_id VARCHAR(50) PRIMARY KEY,
        version_id VARCHAR(50) REFERENCES model_versions(version_id),

        -- Deployment target
        environment VARCHAR(50),  -- production, staging, dev
        endpoint VARCHAR(500),  -- API endpoint or serving endpoint
        region VARCHAR(50),  -- us-east-1, eu-west-1, etc.

        -- Deployment info
        deployment_status VARCHAR(20),  -- deploying, active, failed, rolled_back
        deployed_by VARCHAR(100),
        deployed_at TIMESTAMP DEFAULT NOW(),
        undeployed_at TIMESTAMP,

        -- Traffic info
        traffic_percentage INTEGER DEFAULT 100,  -- For A/B testing
        is_primary BOOLEAN DEFAULT true,

        -- Metadata
        deployment_config JSONB,
        notes TEXT
    );
    CREATE INDEX idx_deployments_version ON deployments(version_id);
    CREATE INDEX idx_deployments_environment ON deployments(environment);
    CREATE INDEX idx_deployments_status ON deployments(deployment_status);
    CREATE INDEX idx_deployments_time ON deployments(deployed_at DESC);

    -- Model aliases (human-readable names)
    CREATE TABLE model_aliases (
        alias_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(50) REFERENCES models(model_id),
        version_id VARCHAR(50) REFERENCES model_versions(version_id),
        alias VARCHAR(100) NOT NULL,  -- champion, challenger, latest, etc.

        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),

        CONSTRAINT unique_model_alias UNIQUE (model_id, alias)
    );
    CREATE INDEX idx_aliases_model ON model_aliases(model_id);

    -- Webhooks
    CREATE TABLE webhooks (
        webhook_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(50) REFERENCES models(model_id),

        url VARCHAR(500) NOT NULL,
        events VARCHAR(50)[],  -- Array of event types to subscribe to
        secret VARCHAR(100),  -- For signature verification

        is_active BOOLEAN DEFAULT true,
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),
        last_triggered_at TIMESTAMP,
        failure_count INTEGER DEFAULT 0
    );
    CREATE INDEX idx_webhooks_model ON webhooks(model_id);
    CREATE INDEX idx_webhooks_active ON webhooks(is_active);

    -- Audit logs
    CREATE TABLE audit_logs (
        log_id BIGSERIAL PRIMARY KEY,
        entity_type VARCHAR(50),  -- model, version, deployment, etc.
        entity_id VARCHAR(50),
        action VARCHAR(50),  -- create, update, delete, transition, approve, deploy

        user_id VARCHAR(100),
        timestamp TIMESTAMP DEFAULT NOW(),

        old_value JSONB,
        new_value JSONB,
        metadata JSONB,

        ip_address INET,
        user_agent TEXT
    );
    CREATE INDEX idx_audit_entity ON audit_logs(entity_type, entity_id);
    CREATE INDEX idx_audit_user ON audit_logs(user_id);
    CREATE INDEX idx_audit_time ON audit_logs(timestamp DESC);
    CREATE INDEX idx_audit_action ON audit_logs(action);
    ```

    ---

    ## API Design

    ### 1. Register Model

    **Request:**
    ```python
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Register model from experiment run
    model_uri = "runs:/abc123/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name="fraud-detection",
        tags={
            "team": "risk",
            "use_case": "payment_fraud",
            "framework": "pytorch"
        }
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/models/{model_name}/versions
    {
      "source_run_id": "abc123",
      "artifact_path": "model",
      "description": "Fraud detection model with 95% accuracy",
      "tags": {
        "team": "risk",
        "use_case": "payment_fraud"
      }
    }
    ```

    **Response:**
    ```json
    {
      "version_id": "ver_xyz789",
      "model_id": "model_fraud_detection",
      "version": "1.0.0",
      "stage": "dev",
      "status": "pending_approval",
      "artifact_uri": "s3://ml-models/fraud-detection/1.0.0/model.pkl",
      "artifact_hash": "sha256:abc123...",
      "created_at": "2024-01-15T10:30:00Z"
    }
    ```

    ---

    ### 2. Transition Model Stage

    **Request:**
    ```python
    # Transition to staging
    client.transition_model_version_stage(
        name="fraud-detection",
        version="1.0.0",
        stage="staging",
        archive_existing_versions=True
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/models/{model_name}/versions/{version}/transition
    {
      "to_stage": "staging",
      "reason": "Passed validation tests with 95% accuracy",
      "archive_existing": true
    }
    ```

    **Response:**
    ```json
    {
      "transition_id": "trans_456",
      "version_id": "ver_xyz789",
      "from_stage": "dev",
      "to_stage": "staging",
      "transitioned_at": "2024-01-15T11:00:00Z",
      "approval_required": true,
      "approval_id": "appr_789"
    }
    ```

    ---

    ### 3. Request Approval

    **Request:**
    ```python
    # Request production approval
    approval = client.request_approval(
        model_name="fraud-detection",
        version="1.0.0",
        approval_type="production_deployment",
        required_approvers=["alice@company.com", "bob@company.com"],
        notes="Model shows 95% accuracy on validation set, ready for production"
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/approvals
    {
      "version_id": "ver_xyz789",
      "approval_type": "production_deployment",
      "required_approvers": ["alice@company.com", "bob@company.com"],
      "notes": "Model shows 95% accuracy on validation set"
    }
    ```

    **Response:**
    ```json
    {
      "approval_id": "appr_789",
      "version_id": "ver_xyz789",
      "status": "pending",
      "required_approvers": ["alice@company.com", "bob@company.com"],
      "approved_by": [],
      "requested_at": "2024-01-15T11:00:00Z",
      "approval_url": "https://registry.company.com/approvals/appr_789"
    }
    ```

    ---

    ### 4. Approve Model

    **Request:**
    ```python
    # Approve as reviewer
    client.approve_model_version(
        approval_id="appr_789",
        approved_by="alice@company.com",
        notes="Reviewed metrics and code, approved for production"
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/approvals/{approval_id}/approve
    {
      "approved_by": "alice@company.com",
      "notes": "Reviewed metrics and code, approved for production"
    }
    ```

    **Response:**
    ```json
    {
      "approval_id": "appr_789",
      "status": "approved",
      "approved_by": ["alice@company.com", "bob@company.com"],
      "completed_at": "2024-01-15T12:00:00Z",
      "can_deploy": true
    }
    ```

    ---

    ### 5. Record Deployment

    **Request:**
    ```python
    # Record production deployment
    deployment = client.record_deployment(
        model_name="fraud-detection",
        version="1.0.0",
        environment="production",
        endpoint="https://api.company.com/predict/fraud",
        region="us-east-1",
        traffic_percentage=100
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/deployments
    {
      "version_id": "ver_xyz789",
      "environment": "production",
      "endpoint": "https://api.company.com/predict/fraud",
      "region": "us-east-1",
      "traffic_percentage": 100,
      "deployment_config": {
        "instance_type": "ml.m5.xlarge",
        "min_instances": 2,
        "max_instances": 10
      }
    }
    ```

    **Response:**
    ```json
    {
      "deployment_id": "deploy_123",
      "version_id": "ver_xyz789",
      "environment": "production",
      "deployment_status": "active",
      "deployed_at": "2024-01-15T13:00:00Z",
      "traffic_percentage": 100
    }
    ```

    ---

    ### 6. Search Models

    **Request:**
    ```python
    # Search models by tags and stage
    models = client.search_model_versions(
        filter_string="stage='production' AND tags.team='risk'",
        order_by=["version DESC"],
        max_results=10
    )
    ```

    **REST API:**
    ```http
    POST /api/v1/models/search
    {
      "filter": {
        "stage": "production",
        "tags.team": "risk"
      },
      "order_by": ["version DESC"],
      "limit": 10
    }
    ```

    **Response:**
    ```json
    {
      "models": [
        {
          "version_id": "ver_xyz789",
          "model_name": "fraud-detection",
          "version": "1.0.0",
          "stage": "production",
          "framework": "pytorch",
          "created_at": "2024-01-15T10:30:00Z",
          "tags": {"team": "risk", "use_case": "payment_fraud"}
        }
      ],
      "total_count": 1
    }
    ```

    ---

    ### 7. Get Model Lineage

    **Request:**
    ```python
    # Get lineage for a model version
    lineage = client.get_model_lineage(
        model_name="fraud-detection",
        version="1.0.0"
    )
    ```

    **REST API:**
    ```http
    GET /api/v1/models/{model_name}/versions/{version}/lineage
    ```

    **Response:**
    ```json
    {
      "version_id": "ver_xyz789",
      "lineage": {
        "experiment": {
          "experiment_id": "exp_123",
          "experiment_name": "fraud-detection-tuning",
          "run_id": "run_abc123"
        },
        "dataset": {
          "dataset_id": "ds_456",
          "dataset_name": "fraud_transactions_2024q1",
          "version": "v1.2"
        },
        "parent_model": {
          "model_name": "fraud-detection",
          "version": "0.9.0"
        },
        "training_config": {
          "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
          }
        }
      }
    }
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. Semantic Versioning System

    **Challenge:** Automatic version assignment while supporting semantic versioning (major.minor.patch).

    **Solution: Intelligent Version Manager**

    ```python
    from typing import Optional, Tuple
    import re
    from dataclasses import dataclass

    @dataclass
    class SemanticVersion:
        major: int
        minor: int
        patch: int

        def __str__(self) -> str:
            return f"{self.major}.{self.minor}.{self.patch}"

        @classmethod
        def parse(cls, version: str) -> "SemanticVersion":
            """Parse semantic version string"""
            match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
            if not match:
                raise ValueError(f"Invalid semantic version: {version}")

            return cls(
                major=int(match.group(1)),
                minor=int(match.group(2)),
                patch=int(match.group(3))
            )

        def increment(self, level: str) -> "SemanticVersion":
            """Increment version at specified level"""
            if level == "major":
                return SemanticVersion(self.major + 1, 0, 0)
            elif level == "minor":
                return SemanticVersion(self.major, self.minor + 1, 0)
            elif level == "patch":
                return SemanticVersion(self.major, self.minor, self.patch + 1)
            else:
                raise ValueError(f"Invalid level: {level}")

        def __lt__(self, other: "SemanticVersion") -> bool:
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    class VersionManager:
        def __init__(self, db_client):
            self.db = db_client

        def generate_next_version(
            self,
            model_id: str,
            increment_level: Optional[str] = "patch",
            schema_changed: bool = False,
            breaking_change: bool = False
        ) -> str:
            """Generate next version number based on changes"""

            # Get latest version
            latest_version = self._get_latest_version(model_id)

            if latest_version is None:
                # First version
                return "1.0.0"

            version = SemanticVersion.parse(latest_version)

            # Determine increment level based on changes
            if breaking_change or schema_changed:
                # Breaking change or schema change → major version bump
                new_version = version.increment("major")
            elif increment_level == "minor":
                # New features, non-breaking changes → minor version bump
                new_version = version.increment("minor")
            else:
                # Bug fixes, improvements → patch version bump
                new_version = version.increment("patch")

            return str(new_version)

        def _get_latest_version(self, model_id: str) -> Optional[str]:
            """Get latest version for a model"""
            query = """
                SELECT version FROM model_versions
                WHERE model_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.db.fetch_one(query, (model_id,))

            return result["version"] if result else None

        def validate_version(
            self,
            model_id: str,
            version: str
        ) -> Tuple[bool, Optional[str]]:
            """Validate version doesn't already exist"""

            # Check format
            try:
                SemanticVersion.parse(version)
            except ValueError as e:
                return False, str(e)

            # Check uniqueness
            query = """
                SELECT 1 FROM model_versions
                WHERE model_id = %s AND version = %s
            """
            result = self.db.fetch_one(query, (model_id, version))

            if result:
                return False, f"Version {version} already exists"

            return True, None

        def register_version(
            self,
            model_id: str,
            artifact_uri: str,
            artifact_hash: str,
            framework: str,
            input_schema: dict,
            output_schema: dict,
            source_run_id: Optional[str] = None,
            tags: Optional[dict] = None,
            version: Optional[str] = None
        ) -> dict:
            """Register a new model version"""

            # Detect schema changes
            schema_changed = self._detect_schema_change(
                model_id, input_schema, output_schema
            )

            # Generate version if not provided
            if version is None:
                version = self.generate_next_version(
                    model_id=model_id,
                    schema_changed=schema_changed
                )
            else:
                # Validate provided version
                valid, error = self.validate_version(model_id, version)
                if not valid:
                    raise ValueError(error)

            # Insert into database
            query = """
                INSERT INTO model_versions (
                    version_id, model_id, version, stage, status,
                    artifact_uri, artifact_hash,
                    framework, input_schema, output_schema,
                    source_run_id, tags
                )
                VALUES (
                    gen_random_uuid()::text, %s, %s, 'dev', 'pending_approval',
                    %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING version_id, created_at
            """

            result = self.db.execute(query, (
                model_id, version, artifact_uri, artifact_hash,
                framework, json.dumps(input_schema), json.dumps(output_schema),
                source_run_id, json.dumps(tags or {})
            ))

            return {
                "version_id": result["version_id"],
                "version": version,
                "stage": "dev",
                "status": "pending_approval",
                "created_at": result["created_at"]
            }

        def _detect_schema_change(
            self,
            model_id: str,
            new_input_schema: dict,
            new_output_schema: dict
        ) -> bool:
            """Detect if schema has changed from latest version"""

            query = """
                SELECT input_schema, output_schema
                FROM model_versions
                WHERE model_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """

            result = self.db.fetch_one(query, (model_id,))

            if not result:
                return False  # First version

            old_input = json.loads(result["input_schema"])
            old_output = json.loads(result["output_schema"])

            # Check if schemas are different
            return (
                old_input != new_input_schema or
                old_output != new_output_schema
            )
    ```

    **Usage:**
    ```python
    version_manager = VersionManager(db_client)

    # Register new version (auto-increment patch)
    version_info = version_manager.register_version(
        model_id="model_fraud_detection",
        artifact_uri="s3://ml-models/fraud-detection/model.pkl",
        artifact_hash="sha256:abc123...",
        framework="pytorch",
        input_schema={"features": ["amount", "merchant_id", "timestamp"]},
        output_schema={"prediction": "float", "probability": "float"}
    )
    # Returns: version = "1.0.1" (auto-incremented patch)

    # Register with schema change (auto-increment major)
    version_info = version_manager.register_version(
        model_id="model_fraud_detection",
        artifact_uri="s3://ml-models/fraud-detection/model.pkl",
        artifact_hash="sha256:def456...",
        framework="pytorch",
        input_schema={"features": ["amount", "merchant_id", "timestamp", "user_location"]},  # Added field
        output_schema={"prediction": "float", "probability": "float"}
    )
    # Returns: version = "2.0.0" (schema change triggered major bump)
    ```

    ---

    ## 2. Approval Workflow State Machine

    **Challenge:** Multi-step approval process with role-based approvers.

    **Solution: Approval Workflow Engine**

    ```python
    from enum import Enum
    from typing import List, Optional
    from datetime import datetime

    class ApprovalStatus(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
        EXPIRED = "expired"

    class ApprovalType(Enum):
        STAGE_TRANSITION = "stage_transition"
        PRODUCTION_DEPLOYMENT = "production_deployment"
        MODEL_DELETION = "model_deletion"

    class ApprovalEngine:
        def __init__(self, db_client, notification_service):
            self.db = db_client
            self.notifications = notification_service

        def request_approval(
            self,
            version_id: str,
            approval_type: ApprovalType,
            required_approvers: List[str],
            requested_by: str,
            notes: Optional[str] = None,
            approval_levels: int = 1
        ) -> dict:
            """Request approval for a model version"""

            # Create approval record
            query = """
                INSERT INTO approvals (
                    approval_id, version_id, approval_type, status,
                    required_approvers, requested_by, requested_at,
                    approval_level, total_levels, notes
                )
                VALUES (
                    gen_random_uuid()::text, %s, %s, %s, %s, %s, NOW(), 1, %s, %s
                )
                RETURNING approval_id, requested_at
            """

            result = self.db.execute(query, (
                version_id, approval_type.value, ApprovalStatus.PENDING.value,
                required_approvers, requested_by, approval_levels, notes
            ))

            approval_id = result["approval_id"]

            # Send notifications to approvers
            self._notify_approvers(
                approval_id=approval_id,
                version_id=version_id,
                approvers=required_approvers,
                requested_by=requested_by
            )

            return {
                "approval_id": approval_id,
                "status": ApprovalStatus.PENDING.value,
                "required_approvers": required_approvers,
                "requested_at": result["requested_at"]
            }

        def approve(
            self,
            approval_id: str,
            approved_by: str,
            notes: Optional[str] = None
        ) -> dict:
            """Approve a pending approval"""

            # Get approval info
            approval = self._get_approval(approval_id)

            if approval["status"] != ApprovalStatus.PENDING.value:
                raise ValueError(f"Approval is not pending: {approval['status']}")

            # Check if user is authorized to approve
            if approved_by not in approval["required_approvers"]:
                raise ValueError(f"User {approved_by} is not authorized to approve")

            # Check if already approved by this user
            if approved_by in approval.get("approved_by", []):
                raise ValueError(f"User {approved_by} has already approved")

            # Add approval
            approved_list = approval.get("approved_by", [])
            approved_list.append(approved_by)

            # Check if all required approvals are complete
            all_approved = set(approved_list) == set(approval["required_approvers"])

            if all_approved:
                # Move to next level or complete
                if approval["approval_level"] < approval["total_levels"]:
                    new_status = ApprovalStatus.PENDING.value
                    new_level = approval["approval_level"] + 1
                else:
                    new_status = ApprovalStatus.APPROVED.value
                    new_level = approval["approval_level"]
            else:
                new_status = ApprovalStatus.PENDING.value
                new_level = approval["approval_level"]

            # Update approval
            query = """
                UPDATE approvals
                SET approved_by = %s,
                    status = %s,
                    approval_level = %s,
                    completed_at = CASE WHEN %s = 'approved' THEN NOW() ELSE NULL END,
                    notes = COALESCE(%s, notes)
                WHERE approval_id = %s
                RETURNING status, completed_at
            """

            result = self.db.execute(query, (
                approved_list, new_status, new_level,
                new_status, notes, approval_id
            ))

            # Send notification
            if new_status == ApprovalStatus.APPROVED.value:
                self._notify_approval_complete(approval_id, approval["version_id"])

            return {
                "approval_id": approval_id,
                "status": new_status,
                "approved_by": approved_list,
                "completed_at": result.get("completed_at")
            }

        def reject(
            self,
            approval_id: str,
            rejected_by: str,
            notes: str
        ) -> dict:
            """Reject a pending approval"""

            # Get approval info
            approval = self._get_approval(approval_id)

            if approval["status"] != ApprovalStatus.PENDING.value:
                raise ValueError(f"Approval is not pending: {approval['status']}")

            # Check if user is authorized to reject
            if rejected_by not in approval["required_approvers"]:
                raise ValueError(f"User {rejected_by} is not authorized to reject")

            # Update approval
            query = """
                UPDATE approvals
                SET status = %s,
                    rejected_by = %s,
                    completed_at = NOW(),
                    notes = %s
                WHERE approval_id = %s
                RETURNING completed_at
            """

            result = self.db.execute(query, (
                ApprovalStatus.REJECTED.value, rejected_by, notes, approval_id
            ))

            # Send notification
            self._notify_approval_rejected(
                approval_id, approval["version_id"], rejected_by, notes
            )

            return {
                "approval_id": approval_id,
                "status": ApprovalStatus.REJECTED.value,
                "rejected_by": rejected_by,
                "completed_at": result["completed_at"]
            }

        def check_approval_status(
            self,
            version_id: str,
            approval_type: ApprovalType
        ) -> dict:
            """Check if version has required approvals"""

            query = """
                SELECT approval_id, status, approved_by, rejected_by, completed_at
                FROM approvals
                WHERE version_id = %s AND approval_type = %s
                ORDER BY requested_at DESC
                LIMIT 1
            """

            result = self.db.fetch_one(query, (version_id, approval_type.value))

            if not result:
                return {
                    "has_approval": False,
                    "status": None,
                    "can_proceed": False
                }

            return {
                "has_approval": True,
                "status": result["status"],
                "can_proceed": result["status"] == ApprovalStatus.APPROVED.value,
                "approval_id": result["approval_id"],
                "approved_by": result.get("approved_by", []),
                "completed_at": result.get("completed_at")
            }

        def _get_approval(self, approval_id: str) -> dict:
            """Get approval details"""
            query = """
                SELECT * FROM approvals
                WHERE approval_id = %s
            """

            result = self.db.fetch_one(query, (approval_id,))

            if not result:
                raise ValueError(f"Approval not found: {approval_id}")

            return result

        def _notify_approvers(
            self,
            approval_id: str,
            version_id: str,
            approvers: List[str],
            requested_by: str
        ):
            """Send notifications to approvers"""
            for approver in approvers:
                self.notifications.send(
                    recipient=approver,
                    subject="Model Approval Request",
                    message=f"Approval requested by {requested_by} for model version {version_id}",
                    approval_url=f"https://registry.company.com/approvals/{approval_id}"
                )

        def _notify_approval_complete(self, approval_id: str, version_id: str):
            """Notify when approval is complete"""
            # Implementation depends on notification service
            pass

        def _notify_approval_rejected(
            self,
            approval_id: str,
            version_id: str,
            rejected_by: str,
            notes: str
        ):
            """Notify when approval is rejected"""
            # Implementation depends on notification service
            pass
    ```

    **Usage:**
    ```python
    approval_engine = ApprovalEngine(db_client, notification_service)

    # Request approval for production deployment
    approval = approval_engine.request_approval(
        version_id="ver_xyz789",
        approval_type=ApprovalType.PRODUCTION_DEPLOYMENT,
        required_approvers=["alice@company.com", "bob@company.com"],
        requested_by="charlie@company.com",
        notes="Model ready for production, 95% accuracy"
    )

    # Alice approves
    approval_engine.approve(
        approval_id=approval["approval_id"],
        approved_by="alice@company.com",
        notes="Metrics look good"
    )

    # Bob approves (final approval)
    approval_engine.approve(
        approval_id=approval["approval_id"],
        approved_by="bob@company.com",
        notes="Approved for production"
    )
    # Approval status changes to "approved"

    # Check if can deploy
    status = approval_engine.check_approval_status(
        version_id="ver_xyz789",
        approval_type=ApprovalType.PRODUCTION_DEPLOYMENT
    )
    # Returns: can_proceed = True
    ```

    ---

    ## 3. Stage Transition State Machine

    **Challenge:** Enforce valid stage transitions with validation.

    **Solution: Stage Manager with State Machine**

    ```python
    from enum import Enum
    from typing import Optional, List

    class ModelStage(Enum):
        DEV = "dev"
        STAGING = "staging"
        PRODUCTION = "production"
        ARCHIVED = "archived"

    class StageTransitionRule:
        """Define allowed stage transitions"""
        TRANSITIONS = {
            ModelStage.DEV: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEV, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.STAGING],  # Rollback to staging
            ModelStage.ARCHIVED: []  # No transitions from archived
        }

        APPROVAL_REQUIRED = {
            (ModelStage.STAGING, ModelStage.PRODUCTION): True,
            (ModelStage.DEV, ModelStage.PRODUCTION): True,  # Direct transition requires approval
        }

    class StageManager:
        def __init__(self, db_client, approval_engine):
            self.db = db_client
            self.approval_engine = approval_engine

        def transition_stage(
            self,
            version_id: str,
            to_stage: ModelStage,
            transitioned_by: str,
            reason: Optional[str] = None,
            archive_existing: bool = False
        ) -> dict:
            """Transition model version to new stage"""

            # Get current stage
            current_stage = self._get_current_stage(version_id)
            from_stage = ModelStage(current_stage)

            # Validate transition
            if not self._is_valid_transition(from_stage, to_stage):
                raise ValueError(
                    f"Invalid transition: {from_stage.value} → {to_stage.value}. "
                    f"Allowed transitions: {[s.value for s in StageTransitionRule.TRANSITIONS[from_stage]]}"
                )

            # Check if approval required
            if self._requires_approval(from_stage, to_stage):
                approval_status = self.approval_engine.check_approval_status(
                    version_id=version_id,
                    approval_type=ApprovalType.STAGE_TRANSITION
                )

                if not approval_status["can_proceed"]:
                    raise ValueError(
                        f"Approval required for {from_stage.value} → {to_stage.value} transition. "
                        f"Current approval status: {approval_status['status']}"
                    )

            # Archive existing versions in target stage if requested
            if archive_existing:
                self._archive_existing_versions(version_id, to_stage)

            # Update stage
            query = """
                UPDATE model_versions
                SET stage = %s, updated_at = NOW()
                WHERE version_id = %s
                RETURNING updated_at
            """

            result = self.db.execute(query, (to_stage.value, version_id))

            # Record transition
            self._record_transition(
                version_id=version_id,
                from_stage=from_stage,
                to_stage=to_stage,
                transitioned_by=transitioned_by,
                reason=reason
            )

            return {
                "version_id": version_id,
                "from_stage": from_stage.value,
                "to_stage": to_stage.value,
                "transitioned_at": result["updated_at"]
            }

        def _get_current_stage(self, version_id: str) -> str:
            """Get current stage for version"""
            query = """
                SELECT stage FROM model_versions
                WHERE version_id = %s
            """

            result = self.db.fetch_one(query, (version_id,))

            if not result:
                raise ValueError(f"Version not found: {version_id}")

            return result["stage"]

        def _is_valid_transition(
            self,
            from_stage: ModelStage,
            to_stage: ModelStage
        ) -> bool:
            """Check if transition is valid"""
            allowed_transitions = StageTransitionRule.TRANSITIONS.get(from_stage, [])
            return to_stage in allowed_transitions

        def _requires_approval(
            self,
            from_stage: ModelStage,
            to_stage: ModelStage
        ) -> bool:
            """Check if transition requires approval"""
            return StageTransitionRule.APPROVAL_REQUIRED.get(
                (from_stage, to_stage), False
            )

        def _archive_existing_versions(
            self,
            exclude_version_id: str,
            stage: ModelStage
        ):
            """Archive other versions in the same stage"""
            query = """
                UPDATE model_versions
                SET stage = 'archived', updated_at = NOW()
                WHERE model_id = (
                    SELECT model_id FROM model_versions WHERE version_id = %s
                )
                AND stage = %s
                AND version_id != %s
            """

            self.db.execute(query, (exclude_version_id, stage.value, exclude_version_id))

        def _record_transition(
            self,
            version_id: str,
            from_stage: ModelStage,
            to_stage: ModelStage,
            transitioned_by: str,
            reason: Optional[str]
        ):
            """Record stage transition in history"""
            query = """
                INSERT INTO stage_transitions (
                    transition_id, version_id, from_stage, to_stage,
                    transitioned_by, transitioned_at, reason
                )
                VALUES (
                    gen_random_uuid()::text, %s, %s, %s, %s, NOW(), %s
                )
            """

            self.db.execute(query, (
                version_id, from_stage.value, to_stage.value,
                transitioned_by, reason
            ))

        def get_stage_history(self, version_id: str) -> List[dict]:
            """Get stage transition history"""
            query = """
                SELECT from_stage, to_stage, transitioned_by, transitioned_at, reason
                FROM stage_transitions
                WHERE version_id = %s
                ORDER BY transitioned_at ASC
            """

            return self.db.fetch_all(query, (version_id,))
    ```

    **State Machine Diagram:**
    ```mermaid
    stateDiagram-v2
        [*] --> Dev: Register
        Dev --> Staging: Validation passes
        Dev --> Archived: Discard

        Staging --> Production: Approval granted
        Staging --> Dev: Issues found
        Staging --> Archived: Discard

        Production --> Staging: Rollback
        Production --> Archived: Deprecate

        Archived --> [*]

        note right of Staging
            Approval required for
            Staging → Production
        end note
    ```

    **Usage:**
    ```python
    stage_manager = StageManager(db_client, approval_engine)

    # Transition dev → staging (no approval needed)
    transition = stage_manager.transition_stage(
        version_id="ver_xyz789",
        to_stage=ModelStage.STAGING,
        transitioned_by="alice@company.com",
        reason="Passed validation tests"
    )

    # Request approval for staging → production
    approval = approval_engine.request_approval(
        version_id="ver_xyz789",
        approval_type=ApprovalType.STAGE_TRANSITION,
        required_approvers=["bob@company.com"],
        requested_by="alice@company.com"
    )

    # After approval, transition staging → production
    transition = stage_manager.transition_stage(
        version_id="ver_xyz789",
        to_stage=ModelStage.PRODUCTION,
        transitioned_by="alice@company.com",
        reason="Approved for production deployment",
        archive_existing=True  # Archive old production model
    )

    # Get transition history
    history = stage_manager.get_stage_history("ver_xyz789")
    # Returns: [
    #   {from_stage: "dev", to_stage: "staging", ...},
    #   {from_stage: "staging", to_stage: "production", ...}
    # ]
    ```

    ---

    ## 4. Deployment Tracking with Rollback

    **Challenge:** Track model deployments across environments and support rollback.

    **Solution: Deployment Tracker**

    ```python
    from typing import List, Optional
    from datetime import datetime

    class DeploymentStatus(Enum):
        DEPLOYING = "deploying"
        ACTIVE = "active"
        FAILED = "failed"
        ROLLED_BACK = "rolled_back"
        UNDEPLOYED = "undeployed"

    class DeploymentTracker:
        def __init__(self, db_client, event_bus):
            self.db = db_client
            self.events = event_bus

        def record_deployment(
            self,
            version_id: str,
            environment: str,
            endpoint: str,
            region: str,
            deployed_by: str,
            traffic_percentage: int = 100,
            deployment_config: Optional[dict] = None
        ) -> dict:
            """Record a new deployment"""

            # Check if version is approved for this environment
            if environment == "production":
                stage = self._get_version_stage(version_id)
                if stage != "production":
                    raise ValueError(
                        f"Cannot deploy to production: version is in {stage} stage"
                    )

            # Mark any existing primary deployment as non-primary
            if traffic_percentage == 100:
                self._unset_primary_deployment(version_id, environment)

            # Create deployment record
            query = """
                INSERT INTO deployments (
                    deployment_id, version_id, environment, endpoint, region,
                    deployment_status, deployed_by, deployed_at,
                    traffic_percentage, is_primary, deployment_config
                )
                VALUES (
                    gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s
                )
                RETURNING deployment_id, deployed_at
            """

            result = self.db.execute(query, (
                version_id, environment, endpoint, region,
                DeploymentStatus.ACTIVE.value, deployed_by,
                traffic_percentage, traffic_percentage == 100,
                json.dumps(deployment_config or {})
            ))

            deployment_id = result["deployment_id"]

            # Publish deployment event
            self.events.publish("model.deployed", {
                "deployment_id": deployment_id,
                "version_id": version_id,
                "environment": environment,
                "endpoint": endpoint,
                "traffic_percentage": traffic_percentage
            })

            return {
                "deployment_id": deployment_id,
                "version_id": version_id,
                "environment": environment,
                "deployment_status": DeploymentStatus.ACTIVE.value,
                "deployed_at": result["deployed_at"],
                "traffic_percentage": traffic_percentage
            }

        def rollback_deployment(
            self,
            deployment_id: str,
            rolled_back_by: str,
            reason: str
        ) -> dict:
            """Rollback a deployment"""

            # Get deployment info
            deployment = self._get_deployment(deployment_id)

            if deployment["deployment_status"] != DeploymentStatus.ACTIVE.value:
                raise ValueError(
                    f"Cannot rollback: deployment is {deployment['deployment_status']}"
                )

            # Mark as rolled back
            query = """
                UPDATE deployments
                SET deployment_status = %s,
                    undeployed_at = NOW(),
                    notes = %s
                WHERE deployment_id = %s
                RETURNING undeployed_at
            """

            result = self.db.execute(query, (
                DeploymentStatus.ROLLED_BACK.value,
                f"Rolled back by {rolled_back_by}: {reason}",
                deployment_id
            ))

            # Find previous deployment to restore
            previous = self._get_previous_deployment(
                environment=deployment["environment"],
                region=deployment["region"],
                before_time=deployment["deployed_at"]
            )

            if previous:
                # Restore previous deployment
                restored = self.record_deployment(
                    version_id=previous["version_id"],
                    environment=deployment["environment"],
                    endpoint=deployment["endpoint"],
                    region=deployment["region"],
                    deployed_by=rolled_back_by,
                    traffic_percentage=100
                )

                rollback_info = {
                    "rolled_back_to": previous["version_id"],
                    "restored_deployment_id": restored["deployment_id"]
                }
            else:
                rollback_info = {
                    "rolled_back_to": None,
                    "restored_deployment_id": None
                }

            # Publish rollback event
            self.events.publish("model.rolled_back", {
                "deployment_id": deployment_id,
                "version_id": deployment["version_id"],
                "environment": deployment["environment"],
                "reason": reason,
                **rollback_info
            })

            return {
                "deployment_id": deployment_id,
                "deployment_status": DeploymentStatus.ROLLED_BACK.value,
                "undeployed_at": result["undeployed_at"],
                **rollback_info
            }

        def update_traffic_split(
            self,
            environment: str,
            region: str,
            traffic_splits: List[dict]
        ) -> List[dict]:
            """Update traffic split for A/B testing

            Args:
                traffic_splits: [
                    {"version_id": "ver_1", "percentage": 90},
                    {"version_id": "ver_2", "percentage": 10}
                ]
            """

            # Validate percentages sum to 100
            total_percentage = sum(split["percentage"] for split in traffic_splits)
            if total_percentage != 100:
                raise ValueError(f"Traffic percentages must sum to 100, got {total_percentage}")

            # Update traffic for each version
            updated = []
            for split in traffic_splits:
                query = """
                    UPDATE deployments
                    SET traffic_percentage = %s,
                        is_primary = %s
                    WHERE version_id = %s
                      AND environment = %s
                      AND region = %s
                      AND deployment_status = 'active'
                    RETURNING deployment_id
                """

                result = self.db.execute(query, (
                    split["percentage"],
                    split["percentage"] == 100,
                    split["version_id"],
                    environment,
                    region
                ))

                if result:
                    updated.append({
                        "deployment_id": result["deployment_id"],
                        "version_id": split["version_id"],
                        "traffic_percentage": split["percentage"]
                    })

            # Publish traffic split event
            self.events.publish("model.traffic_split", {
                "environment": environment,
                "region": region,
                "traffic_splits": traffic_splits
            })

            return updated

        def get_active_deployments(
            self,
            environment: Optional[str] = None,
            region: Optional[str] = None
        ) -> List[dict]:
            """Get active deployments"""

            conditions = ["deployment_status = 'active'"]
            params = []

            if environment:
                conditions.append("environment = %s")
                params.append(environment)

            if region:
                conditions.append("region = %s")
                params.append(region)

            query = f"""
                SELECT d.*, mv.version, mv.model_id, m.name as model_name
                FROM deployments d
                JOIN model_versions mv ON d.version_id = mv.version_id
                JOIN models m ON mv.model_id = m.model_id
                WHERE {' AND '.join(conditions)}
                ORDER BY d.deployed_at DESC
            """

            return self.db.fetch_all(query, tuple(params))

        def get_deployment_history(
            self,
            version_id: str
        ) -> List[dict]:
            """Get deployment history for a version"""

            query = """
                SELECT *
                FROM deployments
                WHERE version_id = %s
                ORDER BY deployed_at DESC
            """

            return self.db.fetch_all(query, (version_id,))

        def _get_deployment(self, deployment_id: str) -> dict:
            """Get deployment details"""
            query = """
                SELECT * FROM deployments
                WHERE deployment_id = %s
            """

            result = self.db.fetch_one(query, (deployment_id,))

            if not result:
                raise ValueError(f"Deployment not found: {deployment_id}")

            return result

        def _get_version_stage(self, version_id: str) -> str:
            """Get stage for version"""
            query = """
                SELECT stage FROM model_versions
                WHERE version_id = %s
            """

            result = self.db.fetch_one(query, (version_id,))
            return result["stage"] if result else None

        def _unset_primary_deployment(self, version_id: str, environment: str):
            """Unset primary flag for existing deployments"""
            query = """
                UPDATE deployments
                SET is_primary = false
                WHERE version_id != %s
                  AND environment = %s
                  AND deployment_status = 'active'
            """

            self.db.execute(query, (version_id, environment))

        def _get_previous_deployment(
            self,
            environment: str,
            region: str,
            before_time: datetime
        ) -> Optional[dict]:
            """Get previous active deployment"""
            query = """
                SELECT *
                FROM deployments
                WHERE environment = %s
                  AND region = %s
                  AND deployed_at < %s
                  AND deployment_status = 'active'
                ORDER BY deployed_at DESC
                LIMIT 1
            """

            return self.db.fetch_one(query, (environment, region, before_time))
    ```

    **Usage:**
    ```python
    deployment_tracker = DeploymentTracker(db_client, event_bus)

    # Deploy to production
    deployment = deployment_tracker.record_deployment(
        version_id="ver_xyz789",
        environment="production",
        endpoint="https://api.company.com/predict/fraud",
        region="us-east-1",
        deployed_by="alice@company.com",
        traffic_percentage=100
    )

    # A/B test: split traffic 90/10
    deployment_tracker.update_traffic_split(
        environment="production",
        region="us-east-1",
        traffic_splits=[
            {"version_id": "ver_xyz789", "percentage": 90},  # Current
            {"version_id": "ver_abc123", "percentage": 10}   # New
        ]
    )

    # Rollback deployment
    deployment_tracker.rollback_deployment(
        deployment_id=deployment["deployment_id"],
        rolled_back_by="bob@company.com",
        reason="High error rate detected"
    )
    # Automatically restores previous deployment

    # Get active deployments
    active = deployment_tracker.get_active_deployments(
        environment="production",
        region="us-east-1"
    )
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scaling Strategies

    ### 1. Artifact Deduplication with Content-Addressed Storage

    **Challenge:** Multiple versions share same model artifacts (20-30% duplication).

    **Solution: Content-Addressed Storage with SHA256**

    ```python
    import hashlib
    import boto3
    from typing import Optional

    class ArtifactManager:
        def __init__(self, s3_client, db_client):
            self.s3 = s3_client
            self.db = db_client
            self.bucket = "ml-model-registry"

        def upload_artifact(
            self,
            version_id: str,
            local_path: str,
            artifact_type: str = "model"
        ) -> dict:
            """Upload artifact with deduplication"""

            # 1. Compute content hash
            content_hash = self._compute_hash(local_path)
            file_size = os.path.getsize(local_path)

            # 2. Check if artifact exists (content-addressed)
            existing_key = self._find_by_hash(content_hash)

            if existing_key:
                # Artifact already exists, reuse it
                s3_key = existing_key
                uploaded = False
            else:
                # Upload new artifact (use hash as key for deduplication)
                s3_key = f"artifacts/{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"

                with open(local_path, "rb") as f:
                    self.s3.upload_fileobj(
                        f, self.bucket, s3_key,
                        ExtraArgs={"ServerSideEncryption": "AES256"}
                    )

                uploaded = True

            # 3. Create artifact reference for this version
            artifact_uri = f"s3://{self.bucket}/{s3_key}"

            query = """
                UPDATE model_versions
                SET artifact_uri = %s,
                    artifact_hash = %s,
                    artifact_size_bytes = %s
                WHERE version_id = %s
            """

            self.db.execute(query, (
                artifact_uri, content_hash, file_size, version_id
            ))

            return {
                "artifact_uri": artifact_uri,
                "artifact_hash": content_hash,
                "size_bytes": file_size,
                "uploaded": uploaded,
                "deduplicated": not uploaded
            }

        def download_artifact(
            self,
            version_id: str,
            dst_path: str
        ) -> str:
            """Download artifact for version"""

            # Get artifact URI
            query = """
                SELECT artifact_uri, artifact_hash
                FROM model_versions
                WHERE version_id = %s
            """

            result = self.db.fetch_one(query, (version_id,))

            if not result:
                raise ValueError(f"Version not found: {version_id}")

            artifact_uri = result["artifact_uri"]
            expected_hash = result["artifact_hash"]

            # Parse S3 URI
            s3_key = artifact_uri.replace(f"s3://{self.bucket}/", "")

            # Download from S3
            with open(dst_path, "wb") as f:
                self.s3.download_fileobj(self.bucket, s3_key, f)

            # Verify checksum
            actual_hash = self._compute_hash(dst_path)
            if actual_hash != expected_hash:
                os.remove(dst_path)
                raise ValueError(
                    f"Checksum mismatch: expected {expected_hash}, got {actual_hash}"
                )

            return dst_path

        def _compute_hash(self, file_path: str) -> str:
            """Compute SHA256 hash"""
            sha256 = hashlib.sha256()

            with open(file_path, "rb") as f:
                # Read in 8MB chunks
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    sha256.update(chunk)

            return sha256.hexdigest()

        def _find_by_hash(self, content_hash: str) -> Optional[str]:
            """Check if artifact with hash exists in S3"""
            s3_key = f"artifacts/{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"

            try:
                self.s3.head_object(Bucket=self.bucket, Key=s3_key)
                return s3_key
            except:
                return None
    ```

    **Storage Savings:**
    - Without deduplication: 100K versions × 500 MB = 50 TB
    - With deduplication (30% savings): 35 TB
    - **Savings: 15 TB (30%)**

    ---

    ### 2. Metadata Search Optimization with Elasticsearch

    **Challenge:** Searching 100K model versions with complex filters is slow in PostgreSQL.

    **Solution: Elasticsearch Index**

    ```python
    from elasticsearch import Elasticsearch

    class ModelSearchService:
        def __init__(self, es_client, db_client):
            self.es = es_client
            self.db = db_client
            self.index = "model_registry"

        def index_model_version(self, version_data: dict):
            """Index model version in Elasticsearch"""

            doc = {
                "version_id": version_data["version_id"],
                "model_id": version_data["model_id"],
                "model_name": version_data["model_name"],
                "version": version_data["version"],
                "stage": version_data["stage"],
                "status": version_data["status"],
                "framework": version_data["framework"],
                "tags": version_data.get("tags", {}),
                "created_at": version_data["created_at"],
                "created_by": version_data["created_by"],

                # For filtering
                "team_id": version_data.get("team_id"),
                "source_experiment_id": version_data.get("source_experiment_id")
            }

            self.es.index(
                index=self.index,
                id=version_data["version_id"],
                document=doc
            )

        def search(
            self,
            query: Optional[str] = None,
            filters: Optional[dict] = None,
            sort_by: Optional[str] = None,
            limit: int = 10,
            offset: int = 0
        ) -> dict:
            """Search model versions"""

            # Build Elasticsearch query
            es_query = {"bool": {"must": [], "filter": []}}

            # Full-text search
            if query:
                es_query["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["model_name^2", "tags.description", "framework"]
                    }
                })

            # Filters
            if filters:
                if "stage" in filters:
                    es_query["bool"]["filter"].append({
                        "term": {"stage": filters["stage"]}
                    })

                if "status" in filters:
                    es_query["bool"]["filter"].append({
                        "term": {"status": filters["status"]}
                    })

                if "team_id" in filters:
                    es_query["bool"]["filter"].append({
                        "term": {"team_id": filters["team_id"]}
                    })

                if "tags" in filters:
                    for key, value in filters["tags"].items():
                        es_query["bool"]["filter"].append({
                            "term": {f"tags.{key}": value}
                        })

            # Sorting
            sort = []
            if sort_by:
                field, order = sort_by.split(" ")
                sort.append({field: {"order": order.lower()}})
            else:
                sort.append({"created_at": {"order": "desc"}})

            # Execute search
            response = self.es.search(
                index=self.index,
                query=es_query if es_query["bool"]["must"] or es_query["bool"]["filter"] else {"match_all": {}},
                sort=sort,
                size=limit,
                from_=offset
            )

            return {
                "results": [hit["_source"] for hit in response["hits"]["hits"]],
                "total": response["hits"]["total"]["value"]
            }
    ```

    **Performance:**
    - PostgreSQL LIKE query: 5-10 seconds for 100K rows
    - Elasticsearch: <100ms for same query
    - **50-100x faster!**

    ---

    ### 3. Webhook Reliability with Retry Mechanism

    **Challenge:** External webhook endpoints may be temporarily unavailable.

    **Solution: Webhook Service with Exponential Backoff**

    ```python
    import time
    import hmac
    import hashlib
    import requests
    from typing import List, Optional

    class WebhookService:
        def __init__(self, db_client, max_retries: int = 3):
            self.db = db_client
            self.max_retries = max_retries

        def dispatch_event(
            self,
            event_type: str,
            model_id: str,
            payload: dict
        ):
            """Dispatch event to registered webhooks"""

            # Get webhooks for this model and event type
            webhooks = self._get_webhooks(model_id, event_type)

            for webhook in webhooks:
                self._send_webhook(
                    webhook_id=webhook["webhook_id"],
                    url=webhook["url"],
                    secret=webhook["secret"],
                    event_type=event_type,
                    payload=payload
                )

        def _send_webhook(
            self,
            webhook_id: str,
            url: str,
            secret: str,
            event_type: str,
            payload: dict,
            attempt: int = 1
        ):
            """Send webhook with retry logic"""

            # Create payload with signature
            payload_json = json.dumps(payload)
            signature = self._generate_signature(payload_json, secret)

            headers = {
                "Content-Type": "application/json",
                "X-Registry-Event": event_type,
                "X-Registry-Signature": signature,
                "X-Registry-Delivery": webhook_id
            }

            try:
                response = requests.post(
                    url,
                    data=payload_json,
                    headers=headers,
                    timeout=10
                )

                response.raise_for_status()

                # Success - update last triggered time
                self._update_webhook_status(webhook_id, success=True)

            except Exception as e:
                # Failure - retry with exponential backoff
                if attempt < self.max_retries:
                    # Exponential backoff: 2^attempt seconds
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)

                    self._send_webhook(
                        webhook_id, url, secret, event_type, payload, attempt + 1
                    )
                else:
                    # Max retries exceeded
                    self._update_webhook_status(webhook_id, success=False)
                    print(f"Webhook failed after {self.max_retries} attempts: {url}")

        def _generate_signature(self, payload: str, secret: str) -> str:
            """Generate HMAC signature for webhook"""
            return hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

        def _get_webhooks(self, model_id: str, event_type: str) -> List[dict]:
            """Get webhooks for model and event type"""
            query = """
                SELECT webhook_id, url, secret, events
                FROM webhooks
                WHERE model_id = %s
                  AND is_active = true
                  AND %s = ANY(events)
            """

            return self.db.fetch_all(query, (model_id, event_type))

        def _update_webhook_status(self, webhook_id: str, success: bool):
            """Update webhook status after delivery attempt"""
            if success:
                query = """
                    UPDATE webhooks
                    SET last_triggered_at = NOW(),
                        failure_count = 0
                    WHERE webhook_id = %s
                """
                self.db.execute(query, (webhook_id,))
            else:
                query = """
                    UPDATE webhooks
                    SET failure_count = failure_count + 1,
                        is_active = CASE WHEN failure_count >= 10 THEN false ELSE is_active END
                    WHERE webhook_id = %s
                """
                self.db.execute(query, (webhook_id,))
    ```

    **Reliability:**
    - 3 retries with exponential backoff (2s, 4s, 8s)
    - Signature verification prevents tampering
    - Auto-disable after 10 consecutive failures
    - **Delivery success rate: 99.9%**

    ---

    ### 4. Multi-Region Artifact Replication

    **Challenge:** Deploying models in multiple regions requires artifact access in each region.

    **Solution: S3 Cross-Region Replication**

    ```python
    class MultiRegionArtifactManager:
        def __init__(self, s3_clients: dict):
            self.s3_clients = s3_clients  # {region: boto3_client}
            self.primary_region = "us-east-1"
            self.replica_regions = ["eu-west-1", "ap-southeast-1"]

        def upload_with_replication(
            self,
            version_id: str,
            local_path: str
        ) -> dict:
            """Upload artifact to primary and replicate to other regions"""

            # Upload to primary region
            primary_client = self.s3_clients[self.primary_region]
            content_hash = self._compute_hash(local_path)
            s3_key = f"artifacts/{content_hash[:2]}/{content_hash[2:4]}/{content_hash}"

            with open(local_path, "rb") as f:
                primary_client.upload_fileobj(
                    f,
                    f"ml-models-{self.primary_region}",
                    s3_key,
                    ExtraArgs={
                        "ServerSideEncryption": "AES256",
                        "Metadata": {
                            "version_id": version_id,
                            "hash": content_hash
                        }
                    }
                )

            # Trigger replication (S3 does this automatically with replication rules)
            # Or manually copy to replica regions
            replica_uris = {}
            for region in self.replica_regions:
                replica_client = self.s3_clients[region]
                replica_bucket = f"ml-models-{region}"

                # Copy from primary to replica
                copy_source = {
                    "Bucket": f"ml-models-{self.primary_region}",
                    "Key": s3_key
                }

                replica_client.copy(
                    copy_source,
                    replica_bucket,
                    s3_key
                )

                replica_uris[region] = f"s3://{replica_bucket}/{s3_key}"

            return {
                "primary_uri": f"s3://ml-models-{self.primary_region}/{s3_key}",
                "replica_uris": replica_uris,
                "hash": content_hash
            }

        def download_from_nearest_region(
            self,
            version_id: str,
            dst_path: str,
            preferred_region: str
        ) -> str:
            """Download artifact from nearest available region"""

            # Try preferred region first
            try:
                client = self.s3_clients[preferred_region]
                bucket = f"ml-models-{preferred_region}"

                # Get artifact key from version
                s3_key = self._get_artifact_key(version_id)

                with open(dst_path, "wb") as f:
                    client.download_fileobj(bucket, s3_key, f)

                return preferred_region

            except Exception as e:
                # Fallback to primary region
                client = self.s3_clients[self.primary_region]
                bucket = f"ml-models-{self.primary_region}"

                with open(dst_path, "wb") as f:
                    client.download_fileobj(bucket, s3_key, f)

                return self.primary_region
    ```

    **Benefits:**
    - Faster downloads in each region (< 1 second vs. 10+ seconds cross-region)
    - Resilience to regional outages
    - Lower data transfer costs
    - **Download latency: 90% reduction for cross-region deployments**

    ---

    ## Bottleneck Analysis

    ### Scenario 1: Slow Model Registration (> 30 seconds)

    **Diagnosis:**
    - Large model upload (2GB+) taking 20+ seconds
    - Metadata insertion taking 5+ seconds
    - Checksum computation taking 10+ seconds

    **Solution 1: Parallel Upload + Async Checksum**
    ```python
    # Upload in parallel with checksum computation
    import concurrent.futures

    def register_model_fast(local_path: str, version_id: str):
        # Compute checksum in background while uploading
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            upload_future = executor.submit(upload_to_s3, local_path, version_id)
            checksum_future = executor.submit(compute_hash, local_path)

            # Wait for both
            s3_uri = upload_future.result()
            checksum = checksum_future.result()

        # Store metadata
        store_metadata(version_id, s3_uri, checksum)
    ```

    **Solution 2: Multipart Upload for Large Models**
    - Split 2GB file into 20 × 100MB chunks
    - Upload 10 chunks in parallel
    - **Registration time: 30s → 5s (6x faster)**

    ---

    ### Scenario 2: High Query Latency (> 1 second)

    **Diagnosis:**
    - Complex JOIN queries on large tables
    - No proper indexing
    - Full table scans

    **Solution: Add Indexes + Cache**
    ```sql
    -- Add composite index for common query patterns
    CREATE INDEX idx_versions_model_stage_status
    ON model_versions(model_id, stage, status);

    -- Add covering index for metadata queries
    CREATE INDEX idx_versions_metadata
    ON model_versions(model_id, version, stage, status, created_at)
    INCLUDE (framework, tags);
    ```

    ```python
    # Cache hot model metadata in Redis
    def get_model_metadata(version_id: str):
        # Try cache first
        cached = redis_client.get(f"metadata:{version_id}")
        if cached:
            return json.loads(cached)

        # Fetch from DB
        metadata = db.fetch_metadata(version_id)

        # Cache for 1 hour
        redis_client.setex(
            f"metadata:{version_id}",
            3600,
            json.dumps(metadata)
        )

        return metadata
    ```

    **Performance:**
    - Query time: 1s → 50ms (20x faster)
    - Cache hit rate: 80% for production models

=== "💡 Step 5: Additional Considerations"

    ## Security & Compliance

    ### 1. Model Access Control

    **Role-Based Access Control (RBAC):**
    ```python
    class ModelAccessControl:
        ROLES = {
            "viewer": ["read"],
            "developer": ["read", "register", "update"],
            "approver": ["read", "approve", "reject"],
            "admin": ["read", "register", "update", "approve", "reject", "delete", "deploy"]
        }

        def check_permission(
            self,
            user_id: str,
            model_id: str,
            action: str
        ) -> bool:
            """Check if user has permission for action"""

            # Get user role for this model
            query = """
                SELECT role FROM model_permissions
                WHERE model_id = %s AND user_id = %s
            """
            result = self.db.fetch_one(query, (model_id, user_id))

            if not result:
                return False

            role = result["role"]
            allowed_actions = self.ROLES.get(role, [])

            return action in allowed_actions
    ```

    ---

    ### 2. Model Encryption

    **Encrypt models at rest and in transit:**
    ```python
    # S3 server-side encryption
    s3_client.upload_file(
        local_path,
        bucket,
        s3_key,
        ExtraArgs={
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": "arn:aws:kms:us-east-1:123456789:key/abc-def-ghi"
        }
    )

    # TLS for all API calls
    app = FastAPI()
    app.add_middleware(
        TLSMiddleware,
        minimum_tls_version="1.2"
    )
    ```

    ---

    ### 3. Audit Trail for Compliance

    **Complete audit logging:**
    ```python
    def log_audit_event(
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: str,
        old_value: Optional[dict],
        new_value: Optional[dict],
        request_info: dict
    ):
        query = """
            INSERT INTO audit_logs (
                entity_type, entity_id, action, user_id,
                old_value, new_value, ip_address, user_agent, timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """

        db.execute(query, (
            entity_type, entity_id, action, user_id,
            json.dumps(old_value), json.dumps(new_value),
            request_info["ip_address"], request_info["user_agent"]
        ))
    ```

    ---

    ## Monitoring & Alerting

    **Key Metrics:**
    ```python
    # 1. Registration Metrics
    gauge("registry_total_models")
    gauge("registry_total_versions")
    histogram("registry_registration_latency_ms", buckets=[100, 500, 1000, 5000, 10000])

    # 2. Storage Metrics
    gauge("registry_artifact_storage_tb")
    gauge("registry_metadata_storage_gb")

    # 3. Deployment Metrics
    gauge("registry_production_models")
    gauge("registry_active_deployments")
    counter("registry_deployments_total", labels=["environment", "status"])

    # 4. Approval Metrics
    gauge("registry_pending_approvals")
    histogram("registry_approval_time_hours", buckets=[1, 6, 12, 24, 48, 168])

    # 5. Error Metrics
    counter("registry_errors_total", labels=["endpoint", "error_type"])
    ```

    **Alerts:**
    ```yaml
    - alert: HighRegistrationLatency
      expr: histogram_quantile(0.95, registry_registration_latency_ms) > 10000
      for: 5m
      annotations:
        summary: "Model registration taking > 10 seconds ({{ $value }}ms)"

    - alert: PendingApprovalBacklog
      expr: registry_pending_approvals > 100
      for: 1h
      annotations:
        summary: "Large approval backlog ({{ $value }} pending)"

    - alert: HighErrorRate
      expr: rate(registry_errors_total[5m]) > 10
      for: 5m
      annotations:
        summary: "High error rate ({{ $value }} errors/sec)"
    ```

    ---

    ## Cost Optimization

    ### Monthly Cost Breakdown

    ```
    Infrastructure:
    - API servers (3 × c5.xlarge): 3 × $0.17/hr × 730 hrs = $372
    - PostgreSQL (db.r5.large): $0.24/hr × 730 hrs = $175
    - Redis (cache.r5.large): $0.126/hr × 730 hrs = $92
    - Elasticsearch (3 × r5.large): 3 × $0.126/hr × 730 hrs = $276

    Storage:
    - S3 artifacts (76.5 TB):
      - Standard (25%): 19 TB × $0.023 = $437
      - IA (50%): 38 TB × $0.0125 = $475
      - Glacier (25%): 19 TB × $0.004 = $76
      - Total: $988

    - PostgreSQL storage (100 GB): $10
    - Elasticsearch storage (200 GB): $20

    Data transfer:
    - Egress (10 TB/month): 10,000 GB × $0.09 = $900

    Total: $2,833/month ≈ $34K/year

    Cost per model: $34K / 10K models = $3.40/model/year
    ```

    ---

    ## Disaster Recovery

    **Backup Strategy:**
    ```bash
    # Daily PostgreSQL backup
    pg_dump registry_db | gzip > backup-$(date +%Y%m%d).sql.gz
    aws s3 cp backup-$(date +%Y%m%d).sql.gz s3://registry-backups/

    # S3 versioning enabled (automatic)
    aws s3api put-bucket-versioning \
      --bucket ml-model-registry \
      --versioning-configuration Status=Enabled

    # Cross-region replication for artifacts
    aws s3api put-bucket-replication \
      --bucket ml-model-registry \
      --replication-configuration file://replication.json
    ```

    **Recovery Time:**
    - Metadata DB: < 30 minutes (restore from backup)
    - Artifacts: Instant (S3 is durable)
    - **RTO: 30 minutes, RPO: 1 hour**

=== "🎯 Step 6: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5 minutes)

    **Key Questions:**
    - Scale? (number of models, versions, teams)
    - Approval workflow complexity? (single-step vs. multi-step)
    - Deployment tracking scope? (which environments, how detailed)
    - Multi-region support? (artifact replication needed?)
    - Integration requirements? (with experiment tracking, model serving)

    ---

    ### 2. Start with Use Case (2 minutes)

    "Let's design a model registry that enables ML teams to:
    1. Register models from experiments with semantic versioning
    2. Move models through dev → staging → production stages with approvals
    3. Track where models are deployed (endpoints, regions, traffic splits)
    4. Trace model lineage back to training experiments and datasets"

    ---

    ### 3. High-Level Architecture (10 minutes)

    Draw architecture diagram with:
    - Registry API (FastAPI)
    - Version Manager (semantic versioning)
    - Stage Manager (state machine)
    - Approval Engine (workflow orchestration)
    - Deployment Tracker
    - S3 (artifacts), PostgreSQL (metadata), Elasticsearch (search)

    Explain the rationale for each component.

    ---

    ### 4. Deep Dive (20 minutes)

    Focus on 2-3 areas based on interviewer interest:
    - **Semantic versioning:** Auto-increment logic with schema change detection
    - **Approval workflow:** State machine with multi-level approvals
    - **Stage transitions:** Validation rules and consistency guarantees
    - **Artifact deduplication:** Content-addressed storage with checksums

    ---

    ### 5. Scale & Optimize (10 minutes)

    Discuss:
    - Artifact deduplication (30% storage savings)
    - Elasticsearch for fast search (50-100x faster)
    - Webhook reliability (retry mechanism with exponential backoff)
    - Multi-region replication (90% latency reduction)

    ---

    ## Common Follow-Up Questions

    ### Q1: How do you prevent two teams from deploying different models to production simultaneously?

    **Answer:**
    "We use strong consistency with database transactions. The stage transition to production requires:
    1. Acquiring a lock on the model (SELECT FOR UPDATE)
    2. Checking current production version
    3. Archiving existing production version if archive_existing=true
    4. Setting new version to production
    5. Releasing lock

    This ensures only one version can be in production stage at a time per model."

    ---

    ### Q2: How do you handle model rollback in case of issues?

    **Answer:**
    "We maintain deployment history for each model version. When a rollback is triggered:
    1. Mark current deployment as rolled_back
    2. Query for the previous active deployment
    3. Redeploy the previous version
    4. Publish rollback event to webhooks
    5. Notify serving systems to switch back

    The entire process takes < 30 seconds with automated rollback detection based on error rates."

    ---

    ### Q3: How do you support A/B testing between model versions?

    **Answer:**
    "We track traffic percentage for each deployment:
    1. Deploy version A with 90% traffic
    2. Deploy version B with 10% traffic
    3. Model serving systems route requests based on percentages
    4. We track metrics per version (latency, predictions, business metrics)
    5. After statistical significance test, promote winner to 100%

    The traffic split is stored in the deployments table and consumed by serving systems via API."

    ---

    ### Q4: How do you ensure model artifacts aren't corrupted?

    **Answer:**
    "We use SHA256 checksums:
    1. Compute checksum during upload
    2. Store checksum in metadata DB
    3. Verify checksum on every download
    4. If mismatch detected, reject download and alert
    5. S3 also provides built-in checksum validation

    This prevents silent data corruption and ensures model integrity."

    ---

    ### Q5: How do you handle models that are too large (10GB+)?

    **Answer:**
    "For large models:
    1. Use multipart upload (split into 100MB chunks, upload 10 in parallel)
    2. Store in S3 with intelligent tiering (auto-move to cheaper storage classes)
    3. Provide presigned URLs for direct download (not through API server)
    4. Consider model compression (ONNX, TensorFlow Lite)
    5. Implement lazy loading in serving systems

    This reduces registration time from 30s to 5s and download time from 10min to 2min."

    ---

    ## Red Flags to Avoid

    1. **Don't** store model artifacts in database (use S3)
    2. **Don't** allow uncontrolled stage transitions (enforce state machine)
    3. **Don't** skip approval for production deployments (compliance risk)
    4. **Don't** forget to validate checksums (prevents corruption)
    5. **Don't** ignore lineage tracking (reproducibility requirement)

    ---

    ## Bonus Points

    1. Mention **real-world systems:** MLflow Model Registry, AWS SageMaker Model Registry
    2. Discuss **model cards:** Auto-generated documentation with metrics, schema, intended use
    3. Talk about **model lineage:** Link to experiments, datasets, training jobs
    4. Consider **model aliases:** Champion, challenger, latest (user-friendly references)
    5. Mention **integration with CI/CD:** Automated registration and deployment pipelines

=== "📚 References & Resources"

    ## Real-World Implementations

    ### MLflow Model Registry
    - **Architecture:** Central registry with stage management, UI, REST API
    - **Scale:** Used by 1000+ companies
    - **Key Features:** Model versioning, stage transitions, model serving integration
    - **Docs:** [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

    ---

    ### AWS SageMaker Model Registry
    - **Architecture:** Managed service with versioning, approval workflows, CI/CD integration
    - **Scale:** Enterprise customers, thousands of models
    - **Key Features:** Model packages, approval workflows, deployment tracking
    - **Docs:** [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)

    ---

    ### Weights & Biases Model Registry
    - **Architecture:** Cloud-based with experiment integration, lineage tracking
    - **Scale:** 500K+ users
    - **Key Features:** Model lineage, version comparison, deployment tracking
    - **Website:** [W&B Model Registry](https://docs.wandb.ai/guides/models)

    ---

    ### Neptune Model Registry
    - **Architecture:** Cloud-native with metadata management, collaboration features
    - **Scale:** Enterprise customers
    - **Key Features:** Model versioning, stage management, approval workflows
    - **Website:** [Neptune Model Registry](https://neptune.ai/product/model-registry)

    ---

    ## Open Source Tools

    ### MLflow Model Registry
    ```python
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Register model
    mlflow.register_model(
        model_uri="runs:/abc123/model",
        name="fraud-detection"
    )

    # Transition to production
    client.transition_model_version_stage(
        name="fraud-detection",
        version="1",
        stage="production"
    )
    ```

    ---

    ### BentoML Model Store
    ```python
    import bentoml

    # Save model
    bentoml.pytorch.save_model(
        "fraud_detection",
        model,
        signatures={"predict": {"batchable": True}}
    )

    # Load model
    model = bentoml.pytorch.load_model("fraud_detection:latest")
    ```

    ---

    ## Related System Design Problems

    1. **ML Experiment Tracking** - Tracks experiments that generate models (upstream)
    2. **Model Serving Platform** - Deploys models from registry (downstream)
    3. **Feature Store** - Provides features for model training and serving
    4. **CI/CD Pipeline** - Automates model registration and deployment
    5. **Model Monitoring** - Tracks deployed model performance in production

---

## Summary

An **ML Model Registry** is a centralized system for managing machine learning models from development to production:

**Key Components:**
- **Version Manager:** Semantic versioning with auto-increment and schema change detection
- **Stage Manager:** State machine for dev → staging → production transitions
- **Approval Engine:** Multi-step approval workflows with role-based approvers
- **Deployment Tracker:** Track deployments across environments with rollback support
- **Artifact Manager:** Content-addressed storage with deduplication (S3)
- **Lineage Service:** Trace models back to experiments, datasets, training jobs
- **Search Service:** Fast model discovery with Elasticsearch
- **Webhook Service:** Event-driven integrations with retry mechanism

**Core Challenges:**
- Semantic versioning with automatic increment
- Approval workflows with multi-level approvals
- Stage transitions with validation and consistency
- Artifact deduplication to save storage (30% savings)
- Deployment tracking across multiple environments
- Model lineage from experiment to production
- Search performance for 100K+ model versions

**Architecture Decisions:**
- Strong consistency for stage transitions (prevent conflicts)
- Content-addressed storage for artifact deduplication
- Event-driven architecture with webhooks
- Elasticsearch for fast search (50-100x faster than PostgreSQL)
- Multi-region replication for fast deployments
- Complete audit trail for compliance

This is a **medium difficulty** problem that combines database design, state machines, workflow orchestration, and distributed storage. Focus on versioning, approval workflows, and deployment tracking during interviews.
