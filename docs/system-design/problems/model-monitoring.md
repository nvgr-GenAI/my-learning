# Design an ML Model Monitoring System

A comprehensive production model monitoring platform that detects data drift, concept drift, performance degradation, and bias in real-time, providing alerts and insights to maintain model health and reliability across thousands of deployed models at scale.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1,000+ models, 100M predictions/day, 1K+ features tracked, sub-second drift detection |
| **Key Challenges** | Statistical drift detection (PSI, KL divergence), concept drift, real-time alerting, bias detection, high-volume prediction logging |
| **Core Concepts** | Data drift (PSI, KS test, KL divergence), concept drift detection, performance monitoring, bias metrics (demographic parity), prediction logging |
| **Companies** | Arize AI, WhyLabs, Evidently AI, Fiddler AI, AWS SageMaker Model Monitor, Google Vertex AI Model Monitoring |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Prediction Logging** | Capture predictions, features, actuals, metadata in real-time | P0 (Must have) |
    | **Data Drift Detection** | Detect distribution shifts using PSI, KL divergence, KS test | P0 (Must have) |
    | **Concept Drift Detection** | Monitor accuracy degradation, decision boundary shifts | P0 (Must have) |
    | **Performance Monitoring** | Track accuracy, precision, recall, AUC over time | P0 (Must have) |
    | **Bias Detection** | Detect demographic parity violations, equal opportunity | P0 (Must have) |
    | **Alerting** | Real-time alerts on drift, performance drop, bias detection | P0 (Must have) |
    | **Dashboard** | Interactive visualizations for drift, performance, bias trends | P0 (Must have) |
    | **Feature Importance Drift** | Track changes in feature contributions over time | P1 (Should have) |
    | **Explainability Tracking** | Monitor SHAP values, prediction explanations | P1 (Should have) |
    | **Custom Metrics** | Support business-specific monitoring metrics | P1 (Should have) |
    | **Model Comparison** | Compare drift/performance across model versions | P2 (Nice to have) |
    | **Auto-Remediation** | Trigger retraining workflows on drift detection | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (focus on monitoring only)
    - Model serving/inference (assume existing serving platform)
    - Feature engineering pipelines
    - Data labeling and ground truth collection
    - Model versioning and deployment
    - Edge device monitoring (focus on cloud deployments)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Logging)** | < 10ms p95 for async logging | Should not impact inference latency |
    | **Latency (Drift Detection)** | < 1 second for real-time alerts | Fast feedback for critical issues |
    | **Throughput** | > 100,000 predictions logged/sec | Support high-traffic models |
    | **Availability** | 99.9% uptime | Critical for production monitoring |
    | **Data Retention** | 90 days hot, 1 year cold storage | Compliance and historical analysis |
    | **Alert Latency** | < 5 minutes from drift to alert | Timely notification |
    | **Dashboard Load Time** | < 3 seconds for 30-day view | Interactive exploration |
    | **Scalability** | Support 1,000+ models, 10K+ features | Multi-tenant platform |
    | **Accuracy** | < 1% false positive rate on drift | Avoid alert fatigue |
    | **Storage Efficiency** | < 100 bytes per prediction logged | Cost-effective at scale |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Total deployed models: 1,000 models
    - Predictions per day: 100M predictions/day
    - Average predictions per model: 100K/day
    - Hot models (top 20%): 200 models handling 80% traffic = 400K predictions/day each
    - Features per prediction: 50 features average
    - Protected attributes tracked: 5 attributes (age, gender, race, etc.)

    Prediction logging:
    - Daily predictions: 100M predictions
    - Logs per second: 100M / 86,400 = ~1,160 logs/sec average
    - Peak logging: 5x average = ~5,800 logs/sec
    - Log size: 100 bytes per prediction (compressed)
    - Daily storage: 100M × 100 bytes = 10 GB/day

    Ground truth feedback:
    - Ground truth collection rate: 10% of predictions (delayed)
    - Daily ground truth: 10M labels/day
    - Feedback delay: 1 hour to 30 days (varies by use case)

    Drift detection:
    - Drift checks per model: Every 5 minutes
    - Daily drift checks: 1,000 models × 288 checks/day = 288K checks/day
    - Features analyzed per check: 50 features
    - Statistical tests per check: 3 tests (PSI, KS, KL divergence)
    - Computations per day: 288K × 50 × 3 = 43.2M computations/day

    Alerting:
    - Alert rate (normal): 1% of models drift per day = 10 alerts/day
    - Alert rate (incident): 20% of models = 200 alerts/hour
    - Alert channels: Email, Slack, PagerDuty

    Dashboard queries:
    - Active users: 100 ML engineers
    - Dashboard views per day: 100 users × 10 views = 1,000 views/day
    - QPS for reads: 1,000 / 86,400 = 0.01 QPS (low, interactive)

    Write-to-read ratio: 1000:1 (logging >> dashboard queries)
    ```

    ### Storage Estimates

    ```
    Prediction logs:
    - Daily predictions: 100M × 100 bytes = 10 GB/day
    - 90 days hot storage: 10 GB × 90 = 900 GB
    - 1 year cold storage: 10 GB × 365 × 0.3 (compressed) = 1.1 TB

    Feature distributions (reference):
    - Features per model: 50 features
    - Histogram bins per feature: 100 bins
    - 1,000 models × 50 × 100 × 8 bytes = 40 MB (reference data)
    - Historical distributions (30 days): 40 MB × 30 = 1.2 GB

    Ground truth labels:
    - Daily labels: 10M × 20 bytes = 200 MB/day
    - 90 days: 200 MB × 90 = 18 GB

    Drift metrics (time-series):
    - Metrics per model: 50 features × 3 drift metrics = 150 metrics
    - Samples per day: 288 samples (every 5 minutes)
    - 1,000 models × 150 × 288 × 16 bytes = 691 GB/day
    - 90 days: 691 GB × 90 = 62 TB (downsampled to 10 GB)

    Performance metrics:
    - Metrics per model: 10 metrics (accuracy, precision, recall, etc.)
    - 1,000 models × 10 × 288 × 16 bytes = 46 MB/day
    - 90 days: 46 MB × 90 = 4.1 GB

    SHAP values (explainability):
    - SHAP per prediction: 50 features × 4 bytes = 200 bytes
    - Sample rate: 1% of predictions = 1M/day
    - Daily: 1M × 200 bytes = 200 MB/day
    - 90 days: 200 MB × 90 = 18 GB

    Total storage: 900 GB + 1.1 TB + 18 GB + 10 GB + 4.1 GB + 18 GB ≈ 2.1 TB
    ```

    ### Compute Estimates

    ```
    Prediction logging service:
    - Ingestion throughput: 5,800 logs/sec peak
    - Instances: 10 instances (8 vCPU, 32 GB each)
    - Total: 80 vCPUs, 320 GB RAM

    Drift detection workers:
    - Drift checks: 288K checks/day = 3.3 checks/sec
    - CPU per check: 0.5 seconds (statistical tests)
    - Total CPU: 3.3 × 0.5 = 1.65 CPU-seconds/sec
    - Instances: 5 instances (8 vCPU each)
    - Total: 40 vCPUs

    Performance monitoring:
    - Label joining: 10M labels/day
    - Metric aggregation: Every 5 minutes
    - Instances: 3 instances (4 vCPU each)
    - Total: 12 vCPUs

    Bias detection:
    - Bias checks: 1,000 models × 5 protected attributes × 288/day = 1.44M checks/day
    - CPU per check: 0.1 seconds
    - Total: 1.44M × 0.1 / 86,400 = 1.67 CPU-seconds/sec
    - Instances: 3 instances (4 vCPU each)
    - Total: 12 vCPUs

    API servers:
    - Dashboard queries: 0.01 QPS (low load)
    - Alert management: 10 alerts/day
    - Instances: 5 instances (4 vCPU each)
    - Total: 20 vCPUs

    Database servers:
    - PostgreSQL (metadata): 1 instance (16 vCPU, 64 GB)
    - InfluxDB (time-series): 3 instances (8 vCPU, 32 GB each)
    - Total: 40 vCPUs, 160 GB RAM

    Total compute: 80 + 40 + 12 + 12 + 20 + 40 = 204 vCPUs
    ```

    ### Memory Estimates

    ```
    Prediction buffer (in-memory):
    - Buffer 10 seconds of predictions: 1,160 logs/sec × 10s = 11,600 logs
    - 11,600 × 1 KB (with features) = 11.6 MB per instance
    - 10 instances: 116 MB

    Reference distributions cache:
    - Hot models: 200 models × 50 features × 100 bins × 8 bytes = 8 MB
    - Total: 80 MB

    Drift detection state:
    - Recent predictions (sliding window): 10,000 predictions × 1 KB = 10 MB per model
    - 1,000 models: 10 GB (distributed across workers)

    Alert state:
    - Active alerts: 1,000 models × 10 KB = 10 MB

    Dashboard cache:
    - Recent metrics: 100 MB

    Total memory: 116 MB + 80 MB + 10 GB + 10 MB + 100 MB ≈ 10.3 GB
    ```

    ---

    ## Key Assumptions

    1. Average prediction has 50 features with mixed types (numeric, categorical)
    2. Ground truth labels available for 10% of predictions with variable delay
    3. Drift checks run every 5 minutes (not real-time per prediction)
    4. False positive rate for drift detection is <1% (avoid alert fatigue)
    5. Hot models (20%) handle 80% of traffic (Pareto principle)
    6. Reference distribution updated weekly from training data
    7. Statistical tests use 10,000 prediction sliding window
    8. Bias checks on 5 protected attributes per model
    9. Dashboard queries historical data (last 30 days)
    10. SHAP values computed for 1% of predictions (sampled)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Async logging:** Non-blocking prediction capture via message queue
    2. **Streaming analytics:** Real-time drift detection on prediction streams
    3. **Statistical rigor:** Multiple drift detection methods (PSI, KS, KL divergence)
    4. **Separation of concerns:** Logging, detection, alerting as separate services
    5. **Scalability:** Distributed workers for parallel drift detection
    6. **Observability:** Comprehensive monitoring of monitors (meta-monitoring)
    7. **Configurability:** Per-model thresholds, metrics, alert rules

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Model Serving Layer"
            Model1[Model A<br/>Inference Service]
            Model2[Model B<br/>Inference Service]
            Model3[Model C<br/>Inference Service]
        end

        subgraph "Prediction Logging"
            LogSDK[Logging SDK<br/>Async, Batched]
            LogAPI[Log Ingestion API<br/>FastAPI<br/>Rate limiting]
            Queue[Message Queue<br/>Kafka<br/>Partitioned by model]
        end

        subgraph "Data Processing"
            LogProcessor[Log Processor<br/>Kafka Consumer<br/>Parse & validate]
            FeatureStore[(Feature Store<br/>Predictions + Features<br/>TimescaleDB)]
            LabelStore[(Label Store<br/>Ground truth<br/>PostgreSQL)]
        end

        subgraph "Drift Detection"
            DriftCoordinator[Drift Coordinator<br/>Schedule checks<br/>Per-model config]
            DriftWorker1[Drift Worker 1<br/>PSI, KS, KL<br/>Statistical tests]
            DriftWorker2[Drift Worker 2<br/>PSI, KS, KL<br/>Statistical tests]
            ReferenceStore[(Reference Store<br/>Training distributions<br/>S3 + Cache)]
        end

        subgraph "Performance Monitoring"
            LabelJoiner[Label Joiner<br/>Match predictions<br/>to ground truth]
            PerfCalculator[Performance Calculator<br/>Accuracy, precision<br/>recall, AUC]
            ConceptDrift[Concept Drift Detector<br/>Sequential tests<br/>ADWIN, DDM]
        end

        subgraph "Bias Detection"
            BiasAnalyzer[Bias Analyzer<br/>Demographic parity<br/>Equal opportunity]
            FairnessMetrics[Fairness Metrics<br/>Disparate impact<br/>Equalized odds]
        end

        subgraph "Alerting"
            AlertManager[Alert Manager<br/>Rule engine<br/>Deduplication]
            NotificationService[Notification Service<br/>Slack, Email<br/>PagerDuty]
            AlertStore[(Alert History<br/>PostgreSQL)]
        end

        subgraph "Analytics & Dashboards"
            MetricsDB[(Metrics DB<br/>InfluxDB<br/>Time-series)]
            DashboardAPI[Dashboard API<br/>GraphQL<br/>Aggregations]
            WebUI[Web Dashboard<br/>React<br/>Drift viz, alerts]
            ExplainabilityEngine[Explainability<br/>SHAP tracker<br/>Feature importance]
        end

        Model1 -->|1. Log prediction| LogSDK
        Model2 -->|1. Log prediction| LogSDK
        Model3 -->|1. Log prediction| LogSDK
        LogSDK -->|2. Batch upload| LogAPI
        LogAPI -->|3. Publish| Queue

        Queue -->|4. Consume| LogProcessor
        LogProcessor -->|5a. Store predictions| FeatureStore
        LogProcessor -->|5b. Update metrics| MetricsDB

        DriftCoordinator -->|6. Schedule check| DriftWorker1
        DriftCoordinator -->|6. Schedule check| DriftWorker2
        DriftWorker1 -->|7a. Fetch recent| FeatureStore
        DriftWorker1 -->|7b. Fetch reference| ReferenceStore
        DriftWorker1 -->|8. Compute drift| MetricsDB

        LabelJoiner -->|9a. Fetch predictions| FeatureStore
        LabelJoiner -->|9b. Fetch labels| LabelStore
        LabelJoiner -->|10. Joined data| PerfCalculator
        PerfCalculator -->|11. Performance metrics| MetricsDB
        PerfCalculator -->|12. Check concept drift| ConceptDrift

        BiasAnalyzer -->|13. Fetch predictions| FeatureStore
        BiasAnalyzer -->|14. Compute fairness| FairnessMetrics
        FairnessMetrics -->|15. Bias metrics| MetricsDB

        DriftWorker1 -->|16a. Alert on drift| AlertManager
        ConceptDrift -->|16b. Alert on perf drop| AlertManager
        FairnessMetrics -->|16c. Alert on bias| AlertManager

        AlertManager -->|17. Send notification| NotificationService
        AlertManager -->|18. Store alert| AlertStore

        WebUI -->|19. Query metrics| DashboardAPI
        DashboardAPI -->|20. Fetch data| MetricsDB
        DashboardAPI -->|20. Fetch data| FeatureStore

        ExplainabilityEngine -->|21. Track SHAP| MetricsDB

        style LogSDK fill:#e1f5ff
        style Queue fill:#ffe1e1
        style DriftWorker1 fill:#fff4e1
        style AlertManager fill:#f3e5f5
        style WebUI fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Kafka (Queue)** | High-throughput (1M+ msgs/sec), partitioning by model, replay capability | RabbitMQ (lower throughput), Kinesis (vendor lock-in), SQS (ordering issues) |
    | **TimescaleDB (Predictions)** | Time-series optimized, SQL queries, automatic partitioning | InfluxDB (no SQL joins), Cassandra (complex queries slow), PostgreSQL (time-series slow) |
    | **InfluxDB (Metrics)** | Time-series native, high write throughput, automatic downsampling | Prometheus (pull-based, not for logs), TimescaleDB (redundant), Elasticsearch (expensive) |
    | **S3 (Reference Data)** | Cheap, durable, version historical distributions | Database (expensive for large data), Redis (not durable) |
    | **PostgreSQL (Metadata)** | ACID, complex queries (alerts, labels), strong consistency | MongoDB (no joins), DynamoDB (expensive for queries) |
    | **Redis (Cache)** | Sub-millisecond reads for hot reference distributions | Memcached (no persistence), in-memory only (lost on restart) |

    **Key Trade-off:** We chose **batch drift detection (every 5 minutes) over per-prediction** to reduce compute cost 100x. This introduces 5-minute detection delay but is acceptable for most use cases. For critical models, check interval can be reduced to 1 minute.

    ---

    ## API Design

    ### 1. Log Prediction

    **Request:**
    ```python
    from arize import Client

    arize_client = Client(api_key="xxx", space_key="yyy")

    # Log prediction
    response = arize_client.log(
        model_id="fraud-detection-v2",
        model_version="2.1.0",
        prediction_id="pred_abc123",
        prediction_label="fraud",
        prediction_score=0.87,
        features={
            "transaction_amount": 1250.50,
            "merchant_category": "electronics",
            "user_age": 34,
            "account_age_days": 245,
            "num_transactions_24h": 3
        },
        tags={
            "environment": "production",
            "region": "us-west"
        },
        timestamp=datetime.utcnow()
    )
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "prediction_id": "pred_abc123",
      "ingested_at": "2024-01-15T10:30:00Z"
    }
    ```

    ---

    ### 2. Log Ground Truth

    **Request:**
    ```python
    # Log actual outcome (delayed)
    arize_client.log_actual(
        model_id="fraud-detection-v2",
        prediction_id="pred_abc123",
        actual_label="not_fraud",
        actual_score=0.0,
        timestamp=datetime.utcnow()
    )
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "prediction_id": "pred_abc123",
      "matched": true,
      "label_delay_hours": 24.5
    }
    ```

    ---

    ### 3. Get Drift Metrics

    **Request:**
    ```python
    GET /api/v1/models/fraud-detection-v2/drift?
        feature=transaction_amount&
        start_time=2024-01-01T00:00:00Z&
        end_time=2024-01-15T00:00:00Z&
        metric=psi
    ```

    **Response:**
    ```json
    {
      "model_id": "fraud-detection-v2",
      "feature": "transaction_amount",
      "metric": "psi",
      "data_points": [
        {
          "timestamp": "2024-01-01T00:00:00Z",
          "value": 0.05,
          "status": "normal"
        },
        {
          "timestamp": "2024-01-02T00:00:00Z",
          "value": 0.12,
          "status": "warning"
        },
        {
          "timestamp": "2024-01-03T00:00:00Z",
          "value": 0.28,
          "status": "critical"
        }
      ],
      "threshold_warning": 0.1,
      "threshold_critical": 0.25
    }
    ```

    ---

    ### 4. Get Performance Metrics

    **Request:**
    ```python
    GET /api/v1/models/fraud-detection-v2/performance?
        start_time=2024-01-01T00:00:00Z&
        end_time=2024-01-15T00:00:00Z
    ```

    **Response:**
    ```json
    {
      "model_id": "fraud-detection-v2",
      "metrics": [
        {
          "timestamp": "2024-01-01T00:00:00Z",
          "accuracy": 0.94,
          "precision": 0.89,
          "recall": 0.91,
          "f1": 0.90,
          "auc": 0.96,
          "sample_count": 10000
        }
      ],
      "concept_drift_detected": false,
      "performance_trend": "stable"
    }
    ```

    ---

    ### 5. Create Alert Rule

    **Request:**
    ```json
    POST /api/v1/models/fraud-detection-v2/alert-rules
    {
      "rule_name": "High PSI Alert",
      "condition": {
        "metric": "psi",
        "feature": "transaction_amount",
        "operator": "greater_than",
        "threshold": 0.25
      },
      "severity": "critical",
      "channels": ["slack", "email"],
      "enabled": true
    }
    ```

    **Response:**
    ```json
    {
      "rule_id": "rule_xyz789",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
    ```

    ---

    ## Database Schema

    ### Prediction Store (TimescaleDB)

    ```sql
    -- Predictions table (hypertable for time-series)
    CREATE TABLE predictions (
        prediction_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(100) NOT NULL,
        model_version VARCHAR(50),
        timestamp TIMESTAMPTZ NOT NULL,
        prediction_label VARCHAR(100),
        prediction_score DOUBLE PRECISION,
        features JSONB,  -- All features as JSON
        tags JSONB,      -- Metadata
        environment VARCHAR(50),
        ingested_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Convert to hypertable (TimescaleDB)
    SELECT create_hypertable('predictions', 'timestamp');

    -- Indexes
    CREATE INDEX idx_predictions_model ON predictions(model_id, timestamp DESC);
    CREATE INDEX idx_predictions_ingested ON predictions(ingested_at DESC);
    CREATE INDEX idx_predictions_features ON predictions USING GIN (features);

    -- Automatic partitioning by time (1 week chunks)
    SELECT set_chunk_time_interval('predictions', INTERVAL '7 days');
    ```

    ---

    ### Label Store (PostgreSQL)

    ```sql
    -- Ground truth labels
    CREATE TABLE labels (
        label_id BIGSERIAL PRIMARY KEY,
        prediction_id VARCHAR(50) REFERENCES predictions(prediction_id),
        model_id VARCHAR(100) NOT NULL,
        actual_label VARCHAR(100),
        actual_score DOUBLE PRECISION,
        timestamp TIMESTAMPTZ NOT NULL,
        label_delay_hours DOUBLE PRECISION,  -- Time between prediction and label
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX idx_labels_prediction ON labels(prediction_id);
    CREATE INDEX idx_labels_model ON labels(model_id, timestamp DESC);
    CREATE INDEX idx_labels_timestamp ON labels(timestamp DESC);
    ```

    ---

    ### Alert Store (PostgreSQL)

    ```sql
    -- Alert rules
    CREATE TABLE alert_rules (
        rule_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(100) NOT NULL,
        rule_name VARCHAR(255),
        condition JSONB NOT NULL,  -- Metric, threshold, operator
        severity VARCHAR(20),  -- info, warning, critical
        channels JSONB,  -- ["slack", "email", "pagerduty"]
        enabled BOOLEAN DEFAULT true,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX idx_alert_rules_model ON alert_rules(model_id);

    -- Alert history
    CREATE TABLE alerts (
        alert_id BIGSERIAL PRIMARY KEY,
        rule_id VARCHAR(50) REFERENCES alert_rules(rule_id),
        model_id VARCHAR(100) NOT NULL,
        alert_type VARCHAR(50),  -- drift, performance, bias
        severity VARCHAR(20),
        message TEXT,
        details JSONB,  -- Metric values, thresholds
        triggered_at TIMESTAMPTZ NOT NULL,
        resolved_at TIMESTAMPTZ,
        status VARCHAR(20) DEFAULT 'active',  -- active, resolved, acknowledged
        notified_channels JSONB
    );

    CREATE INDEX idx_alerts_model ON alerts(model_id, triggered_at DESC);
    CREATE INDEX idx_alerts_status ON alerts(status);
    ```

    ---

    ### Reference Distributions (S3 + Metadata)

    ```sql
    -- Reference data metadata (PostgreSQL)
    CREATE TABLE reference_distributions (
        reference_id VARCHAR(50) PRIMARY KEY,
        model_id VARCHAR(100) NOT NULL,
        model_version VARCHAR(50),
        feature_name VARCHAR(255),
        distribution_type VARCHAR(50),  -- histogram, kde, quantiles
        distribution_data JSONB,  -- Bins, counts, quantiles
        s3_path VARCHAR(500),  -- Full distribution data in S3
        training_dataset_id VARCHAR(100),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX idx_reference_model_feature ON reference_distributions(model_id, feature_name);
    ```

    ---

    ### Drift Metrics (InfluxDB)

    **Schema:**
    ```
    Measurement: drift_metrics

    Tags (indexed):
    - model_id
    - feature_name
    - metric_type (psi, ks_statistic, kl_divergence)

    Fields:
    - value (float)
    - p_value (float, for KS test)
    - status (string: normal, warning, critical)

    Timestamp: Unix timestamp (nanoseconds)
    ```

    **Example Query:**
    ```sql
    -- Get PSI for transaction_amount feature
    SELECT value, status FROM drift_metrics
    WHERE model_id = 'fraud-detection-v2'
      AND feature_name = 'transaction_amount'
      AND metric_type = 'psi'
      AND time > now() - 30d
    ORDER BY time DESC
    ```

=== "🔧 Step 3: Deep Dive"

    ## 1. Data Drift Detection: Statistical Methods

    **Challenge:** Detect when input feature distributions shift from training data.

    **Solution: Multiple Statistical Tests**

    ```python
    import numpy as np
    from scipy import stats
    from typing import Dict, Tuple

    class DriftDetector:
        def __init__(self, reference_distribution: Dict[str, np.ndarray]):
            """
            Args:
                reference_distribution: {feature_name: distribution_array}
            """
            self.reference = reference_distribution

        def compute_psi(
            self, feature_name: str, production_data: np.ndarray
        ) -> float:
            """
            Population Stability Index (PSI)

            PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

            Thresholds:
            - PSI < 0.1: No significant change
            - 0.1 < PSI < 0.25: Moderate change
            - PSI > 0.25: Significant change (alert)
            """
            reference = self.reference[feature_name]

            # Create 10 equal-frequency bins from reference
            _, bin_edges = np.histogram(reference, bins=10)

            # Count frequencies in each bin
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            prod_counts, _ = np.histogram(production_data, bins=bin_edges)

            # Convert to percentages (avoid division by zero)
            ref_pct = (ref_counts + 1e-6) / np.sum(ref_counts)
            prod_pct = (prod_counts + 1e-6) / np.sum(prod_counts)

            # Compute PSI
            psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

            return float(psi)

        def compute_ks_test(
            self, feature_name: str, production_data: np.ndarray
        ) -> Tuple[float, float]:
            """
            Kolmogorov-Smirnov Test

            Tests if two distributions are significantly different.

            Returns:
                (ks_statistic, p_value)

            Thresholds:
            - p_value > 0.05: Distributions similar (no drift)
            - p_value < 0.05: Distributions different (drift detected)
            """
            reference = self.reference[feature_name]

            # Two-sample KS test
            ks_stat, p_value = stats.ks_2samp(reference, production_data)

            return float(ks_stat), float(p_value)

        def compute_kl_divergence(
            self, feature_name: str, production_data: np.ndarray
        ) -> float:
            """
            Kullback-Leibler Divergence

            KL(P || Q) = Σ P(x) × log(P(x) / Q(x))

            Measures how one distribution diverges from reference.
            Always non-negative, 0 means identical distributions.

            Thresholds:
            - KL < 0.1: Minor change
            - 0.1 < KL < 0.5: Moderate drift
            - KL > 0.5: Significant drift
            """
            reference = self.reference[feature_name]

            # Create bins
            _, bin_edges = np.histogram(reference, bins=50)

            # Compute probabilities
            ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
            prod_hist, _ = np.histogram(production_data, bins=bin_edges, density=True)

            # Normalize to probability distributions
            ref_prob = ref_hist / np.sum(ref_hist)
            prod_prob = prod_hist / np.sum(prod_hist)

            # Add small epsilon to avoid log(0)
            ref_prob = ref_prob + 1e-10
            prod_prob = prod_prob + 1e-10

            # Compute KL divergence
            kl_div = np.sum(prod_prob * np.log(prod_prob / ref_prob))

            return float(kl_div)

        def detect_drift(
            self, feature_name: str, production_data: np.ndarray
        ) -> Dict:
            """
            Run all drift detection methods and return results.
            """
            # PSI
            psi = self.compute_psi(feature_name, production_data)
            psi_status = self._get_psi_status(psi)

            # KS Test
            ks_stat, p_value = self.compute_ks_test(feature_name, production_data)
            ks_status = "drift" if p_value < 0.05 else "normal"

            # KL Divergence
            kl_div = self.compute_kl_divergence(feature_name, production_data)
            kl_status = self._get_kl_status(kl_div)

            return {
                "feature": feature_name,
                "psi": {"value": psi, "status": psi_status},
                "ks_test": {
                    "statistic": ks_stat,
                    "p_value": p_value,
                    "status": ks_status
                },
                "kl_divergence": {"value": kl_div, "status": kl_status},
                "overall_drift": any([
                    psi_status == "critical",
                    ks_status == "drift",
                    kl_status == "critical"
                ])
            }

        def _get_psi_status(self, psi: float) -> str:
            if psi < 0.1:
                return "normal"
            elif psi < 0.25:
                return "warning"
            else:
                return "critical"

        def _get_kl_status(self, kl: float) -> str:
            if kl < 0.1:
                return "normal"
            elif kl < 0.5:
                return "warning"
            else:
                return "critical"


    # Usage example
    import pandas as pd

    # Load reference distribution (from training data)
    reference_df = pd.read_parquet("s3://models/fraud-detection/reference.parquet")
    reference_distributions = {
        col: reference_df[col].values
        for col in reference_df.columns
    }

    detector = DriftDetector(reference_distributions)

    # Fetch recent production data (last 10,000 predictions)
    production_df = fetch_recent_predictions(model_id="fraud-detection-v2", limit=10000)

    # Check drift for each feature
    drift_results = {}
    for feature in production_df.columns:
        result = detector.detect_drift(feature, production_df[feature].values)
        drift_results[feature] = result

        if result["overall_drift"]:
            print(f"🚨 Drift detected in {feature}!")
            print(f"  PSI: {result['psi']['value']:.4f} ({result['psi']['status']})")
            print(f"  KS test p-value: {result['ks_test']['p_value']:.4f}")
            print(f"  KL divergence: {result['kl_divergence']['value']:.4f}")
    ```

    **Performance:**
    - PSI computation: O(n) where n = sample size, ~1ms for 10K samples
    - KS test: O(n log n), ~5ms for 10K samples
    - KL divergence: O(n), ~2ms for 10K samples
    - Total per feature: ~8ms → 50 features = 400ms per model

    ---

    ## 2. Concept Drift Detection

    **Challenge:** Detect when model accuracy degrades (decision boundary shifts).

    **Solution: Sequential Drift Detection Methods**

    ```python
    from collections import deque
    import numpy as np

    class ConceptDriftDetector:
        """
        DDM (Drift Detection Method) and ADWIN (Adaptive Windowing)

        Monitors error rate and detects significant increases.
        """

        def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
            """
            Args:
                warning_level: Standard deviations for warning (default: 2σ)
                drift_level: Standard deviations for drift alarm (default: 3σ)
            """
            self.warning_level = warning_level
            self.drift_level = drift_level

            # Statistics tracking
            self.error_rate = 0.0
            self.min_error_rate = float('inf')
            self.min_std = float('inf')
            self.n_samples = 0
            self.n_errors = 0

            # State
            self.in_warning = False
            self.drift_detected = False

        def add_result(self, prediction_correct: bool) -> Dict:
            """
            Update detector with new prediction result.

            Args:
                prediction_correct: True if prediction matches ground truth

            Returns:
                Detection status
            """
            self.n_samples += 1
            if not prediction_correct:
                self.n_errors += 1

            # Compute error rate and standard deviation
            self.error_rate = self.n_errors / self.n_samples
            std = np.sqrt(
                self.error_rate * (1 - self.error_rate) / self.n_samples
            )

            # Update minimum error rate (best performance seen)
            if self.error_rate + std < self.min_error_rate + self.min_std:
                self.min_error_rate = self.error_rate
                self.min_std = std

            # Check for drift
            drift_threshold = self.min_error_rate + self.drift_level * self.min_std
            warning_threshold = self.min_error_rate + self.warning_level * self.min_std

            if self.error_rate + std > drift_threshold:
                self.drift_detected = True
                return {
                    "status": "drift",
                    "error_rate": self.error_rate,
                    "threshold": drift_threshold,
                    "samples": self.n_samples
                }
            elif self.error_rate + std > warning_threshold:
                self.in_warning = True
                return {
                    "status": "warning",
                    "error_rate": self.error_rate,
                    "threshold": warning_threshold,
                    "samples": self.n_samples
                }
            else:
                self.in_warning = False
                return {
                    "status": "normal",
                    "error_rate": self.error_rate,
                    "samples": self.n_samples
                }

        def reset(self):
            """Reset detector after handling drift (e.g., model retrain)."""
            self.error_rate = 0.0
            self.min_error_rate = float('inf')
            self.min_std = float('inf')
            self.n_samples = 0
            self.n_errors = 0
            self.in_warning = False
            self.drift_detected = False


    class ADWINDetector:
        """
        ADWIN: Adaptive Windowing

        Automatically adjusts window size and detects changes in data stream.
        More sensitive than DDM.
        """

        def __init__(self, delta: float = 0.002):
            """
            Args:
                delta: Confidence level (smaller = more sensitive)
            """
            self.delta = delta
            self.window = deque()
            self.total = 0.0
            self.variance = 0.0
            self.width = 0

        def add_result(self, error: float) -> bool:
            """
            Add new error value and check for drift.

            Args:
                error: 1 if prediction wrong, 0 if correct

            Returns:
                True if drift detected
            """
            self.window.append(error)
            self.width += 1
            self.total += error

            # Check if window should be split
            drift_detected = self._detect_change()

            if drift_detected:
                # Remove old data from window
                while len(self.window) > 1 and self._detect_change():
                    removed = self.window.popleft()
                    self.width -= 1
                    self.total -= removed

            return drift_detected

        def _detect_change(self) -> bool:
            """Check if two sub-windows have significantly different means."""
            if self.width < 2:
                return False

            # Split window into two parts and compare means
            n = self.width // 2
            window_list = list(self.window)

            mean1 = np.mean(window_list[:n])
            mean2 = np.mean(window_list[n:])

            # Compute threshold
            m = 1.0 / (1.0 / n + 1.0 / (self.width - n))
            threshold = np.sqrt(2.0 / m * np.log(2.0 / self.delta))

            return abs(mean1 - mean2) > threshold


    # Usage: Monitor concept drift
    ddm = ConceptDriftDetector()
    adwin = ADWINDetector()

    # Process predictions as ground truth arrives
    for prediction_id, actual_label in incoming_labels():
        # Fetch original prediction
        pred = get_prediction(prediction_id)

        # Check if correct
        correct = (pred["prediction_label"] == actual_label)
        error = 0.0 if correct else 1.0

        # Update detectors
        ddm_result = ddm.add_result(correct)
        adwin_drift = adwin.add_result(error)

        if ddm_result["status"] == "drift" or adwin_drift:
            # Trigger alert
            send_alert(
                model_id="fraud-detection-v2",
                alert_type="concept_drift",
                message=f"Performance degradation detected. Error rate: {ddm_result['error_rate']:.2%}",
                severity="critical"
            )

            # Reset detectors after retraining
            ddm.reset()
    ```

    ---

    ## 3. Bias Detection & Fairness Metrics

    **Challenge:** Detect unfair treatment of protected groups (age, gender, race).

    **Solution: Demographic Parity & Equal Opportunity**

    ```python
    import pandas as pd
    from typing import Dict, List

    class BiasDetector:
        def __init__(self, protected_attributes: List[str]):
            """
            Args:
                protected_attributes: List of sensitive features (e.g., ["gender", "race"])
            """
            self.protected_attributes = protected_attributes

        def compute_demographic_parity(
            self, df: pd.DataFrame, protected_attr: str, positive_label: str = "1"
        ) -> Dict:
            """
            Demographic Parity (Statistical Parity)

            P(Ŷ=1 | A=0) ≈ P(Ŷ=1 | A=1)

            Measures if positive prediction rate is similar across groups.

            Example:
            - Loan approval rate: 70% for men, 50% for women → Biased
            """
            results = {}

            for group_value in df[protected_attr].unique():
                group_df = df[df[protected_attr] == group_value]

                # Positive prediction rate
                positive_rate = (
                    group_df["prediction_label"] == positive_label
                ).mean()

                results[group_value] = {
                    "positive_rate": positive_rate,
                    "sample_count": len(group_df)
                }

            # Compute disparate impact (min_rate / max_rate)
            rates = [v["positive_rate"] for v in results.values()]
            disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0

            # 80% rule: Disparate impact should be >= 0.8
            is_fair = disparate_impact >= 0.8

            return {
                "protected_attribute": protected_attr,
                "group_metrics": results,
                "disparate_impact": disparate_impact,
                "is_fair": is_fair,
                "threshold": 0.8
            }

        def compute_equal_opportunity(
            self,
            df: pd.DataFrame,
            protected_attr: str,
            positive_label: str = "1"
        ) -> Dict:
            """
            Equal Opportunity

            TPR(A=0) ≈ TPR(A=1)

            True positive rates should be similar across groups.
            (Among qualified individuals, equal chance of positive prediction)
            """
            # Filter to actual positives only
            positives_df = df[df["actual_label"] == positive_label]

            results = {}

            for group_value in positives_df[protected_attr].unique():
                group_df = positives_df[positives_df[protected_attr] == group_value]

                # True positive rate (recall)
                tpr = (
                    group_df["prediction_label"] == positive_label
                ).mean()

                results[group_value] = {
                    "true_positive_rate": tpr,
                    "sample_count": len(group_df)
                }

            # Compute difference in TPR
            tprs = [v["true_positive_rate"] for v in results.values()]
            max_tpr_diff = max(tprs) - min(tprs) if tprs else 0

            # Threshold: Difference should be < 0.1 (10%)
            is_fair = max_tpr_diff < 0.1

            return {
                "protected_attribute": protected_attr,
                "group_metrics": results,
                "max_tpr_difference": max_tpr_diff,
                "is_fair": is_fair,
                "threshold": 0.1
            }

        def compute_equalized_odds(
            self,
            df: pd.DataFrame,
            protected_attr: str,
            positive_label: str = "1"
        ) -> Dict:
            """
            Equalized Odds

            TPR(A=0) ≈ TPR(A=1) AND FPR(A=0) ≈ FPR(A=1)

            Both true positive and false positive rates should be equal.
            """
            results = {}

            for group_value in df[protected_attr].unique():
                group_df = df[df[protected_attr] == group_value]

                # True positives
                tp = (
                    (group_df["actual_label"] == positive_label) &
                    (group_df["prediction_label"] == positive_label)
                ).sum()

                # False positives
                fp = (
                    (group_df["actual_label"] != positive_label) &
                    (group_df["prediction_label"] == positive_label)
                ).sum()

                # Actual positives and negatives
                p = (group_df["actual_label"] == positive_label).sum()
                n = (group_df["actual_label"] != positive_label).sum()

                tpr = tp / p if p > 0 else 0
                fpr = fp / n if n > 0 else 0

                results[group_value] = {
                    "true_positive_rate": tpr,
                    "false_positive_rate": fpr,
                    "sample_count": len(group_df)
                }

            # Compute max differences
            tprs = [v["true_positive_rate"] for v in results.values()]
            fprs = [v["false_positive_rate"] for v in results.values()]

            max_tpr_diff = max(tprs) - min(tprs) if tprs else 0
            max_fpr_diff = max(fprs) - min(fprs) if fprs else 0

            is_fair = max_tpr_diff < 0.1 and max_fpr_diff < 0.1

            return {
                "protected_attribute": protected_attr,
                "group_metrics": results,
                "max_tpr_difference": max_tpr_diff,
                "max_fpr_difference": max_fpr_diff,
                "is_fair": is_fair
            }

        def analyze_bias(self, df: pd.DataFrame) -> Dict:
            """
            Run all bias detection methods.
            """
            bias_report = {}

            for attr in self.protected_attributes:
                if attr not in df.columns:
                    continue

                bias_report[attr] = {
                    "demographic_parity": self.compute_demographic_parity(df, attr),
                    "equal_opportunity": self.compute_equal_opportunity(df, attr),
                    "equalized_odds": self.compute_equalized_odds(df, attr)
                }

            return bias_report


    # Usage example
    bias_detector = BiasDetector(protected_attributes=["gender", "race", "age_group"])

    # Fetch recent predictions with labels
    predictions_df = fetch_predictions_with_labels(
        model_id="fraud-detection-v2",
        limit=10000
    )

    # Analyze bias
    bias_report = bias_detector.analyze_bias(predictions_df)

    # Check for violations
    for attr, metrics in bias_report.items():
        if not metrics["demographic_parity"]["is_fair"]:
            print(f"⚠️ Demographic parity violation for {attr}")
            print(f"  Disparate impact: {metrics['demographic_parity']['disparate_impact']:.2f}")

            # Send alert
            send_alert(
                model_id="fraud-detection-v2",
                alert_type="bias_detected",
                message=f"Bias detected in {attr}. Disparate impact: {metrics['demographic_parity']['disparate_impact']:.2f}",
                severity="high"
            )
    ```

    ---

    ## 4. Prediction Logging: High-Throughput Ingestion

    **Challenge:** Log 100M predictions/day without impacting inference latency.

    **Solution: Async Batching + Kafka**

    ```python
    import asyncio
    from typing import Dict, List
    from kafka import KafkaProducer
    import json
    import time

    class PredictionLogger:
        def __init__(self, kafka_brokers: List[str], batch_size: int = 100):
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',  # Reduce network bandwidth
                linger_ms=10,  # Batch for 10ms
                batch_size=16384  # 16KB batches
            )
            self.batch_size = batch_size
            self.buffer = []
            self.lock = asyncio.Lock()

        async def log_prediction(self, prediction: Dict):
            """
            Non-blocking async logging.

            Called from model serving code:
            await logger.log_prediction({...})
            """
            async with self.lock:
                self.buffer.append(prediction)

                # Flush when batch full
                if len(self.buffer) >= self.batch_size:
                    await self._flush()

        async def _flush(self):
            """Send batch to Kafka."""
            if not self.buffer:
                return

            batch = self.buffer.copy()
            self.buffer.clear()

            # Send to Kafka (async, returns immediately)
            for pred in batch:
                topic = f"predictions.{pred['model_id']}"
                self.producer.send(topic, value=pred)

            # Don't wait for acknowledgment (fire and forget)
            # Producer will handle retries internally

        async def flush_sync(self):
            """Force flush remaining predictions."""
            async with self.lock:
                await self._flush()

            # Wait for all messages to be sent
            self.producer.flush()


    # Usage in model serving
    logger = PredictionLogger(kafka_brokers=["kafka1:9092", "kafka2:9092"])

    async def predict(features: Dict) -> Dict:
        # Model inference
        prediction = model.predict(features)

        # Log prediction (non-blocking, <1ms)
        await logger.log_prediction({
            "model_id": "fraud-detection-v2",
            "prediction_id": generate_id(),
            "prediction_label": prediction["label"],
            "prediction_score": prediction["score"],
            "features": features,
            "timestamp": time.time()
        })

        return prediction
    ```

    **Performance:**
    - Async logging: <1ms overhead
    - Kafka throughput: 1M+ messages/sec per broker
    - Batching reduces network calls by 100x

=== "⚖️ Step 4: Scale & Optimize"

    ## Scaling Strategies

    ### 1. Sampling for High-Volume Models

    **Challenge:** Logging 1M predictions/day per model is expensive for storage/compute.

    **Solution: Stratified Sampling**

    ```python
    import random
    from typing import Dict

    class SmartSampler:
        def __init__(self, base_sample_rate: float = 0.1):
            """
            Args:
                base_sample_rate: Default sampling rate (10%)
            """
            self.base_sample_rate = base_sample_rate

        def should_sample(self, prediction: Dict) -> bool:
            """
            Adaptive sampling based on prediction characteristics.

            Strategy:
            - Always sample: Low confidence predictions, errors, edge cases
            - High sample rate: Recent data (last 24h) for drift detection
            - Low sample rate: Old data, high confidence predictions
            """
            # Always sample low confidence predictions
            score = prediction.get("prediction_score", 1.0)
            if score < 0.6 or score > 0.95:  # Low confidence or very high
                return True

            # Sample more recent predictions
            age_hours = (time.time() - prediction["timestamp"]) / 3600
            if age_hours < 24:
                sample_rate = 1.0  # 100% for last 24h
            elif age_hours < 168:  # 1 week
                sample_rate = 0.5  # 50%
            else:
                sample_rate = self.base_sample_rate  # 10%

            return random.random() < sample_rate

        def should_compute_shap(self, prediction: Dict) -> bool:
            """
            SHAP values are expensive, only compute for 1% of predictions.
            """
            # Always for incorrect predictions (if label available)
            if "actual_label" in prediction:
                if prediction["prediction_label"] != prediction["actual_label"]:
                    return True

            # Random 1% sampling
            return random.random() < 0.01
    ```

    **Storage Savings:**
    - Without sampling: 100M × 100 bytes = 10 GB/day
    - With sampling (30% average): 30M × 100 bytes = 3 GB/day
    - **70% reduction, maintain drift detection accuracy**

    ---

    ### 2. Distributed Drift Detection

    **Challenge:** Computing drift for 1,000 models × 50 features × 3 metrics = 150K computations/day.

    **Solution: Distributed Workers with Task Queue**

    ```python
    from celery import Celery
    import redis

    # Celery app for distributed task processing
    app = Celery('drift_detection', broker='redis://localhost:6379/0')

    @app.task
    def compute_drift_for_model(model_id: str, check_timestamp: int):
        """
        Celery task: Compute drift for all features of a model.
        """
        # Fetch reference distribution
        reference = load_reference_distribution(model_id)

        # Fetch recent production data (last 10K predictions)
        production_data = fetch_recent_predictions(model_id, limit=10000)

        # Initialize detector
        detector = DriftDetector(reference)

        # Compute drift for each feature
        drift_results = []
        for feature in production_data.columns:
            result = detector.detect_drift(feature, production_data[feature].values)
            drift_results.append(result)

            # Store drift metrics in InfluxDB
            store_drift_metric(
                model_id=model_id,
                feature=feature,
                metrics=result,
                timestamp=check_timestamp
            )

            # Check for alerts
            if result["overall_drift"]:
                trigger_alert(model_id, feature, result)

        return {"model_id": model_id, "drift_count": sum(r["overall_drift"] for r in drift_results)}


    # Drift coordinator (scheduled job, runs every 5 minutes)
    def schedule_drift_checks():
        """
        Schedule drift detection tasks for all models.
        """
        active_models = get_active_models()  # 1,000 models
        check_timestamp = int(time.time())

        for model_id in active_models:
            # Enqueue task (distributed to workers)
            compute_drift_for_model.delay(model_id, check_timestamp)

        print(f"Scheduled {len(active_models)} drift detection tasks")


    # Run workers (horizontal scaling)
    # celery -A drift_detection worker --concurrency=10
    ```

    **Performance:**
    - Single worker: 400ms per model × 1,000 models = 400 seconds (too slow)
    - 20 workers: 400 seconds / 20 = 20 seconds per check cycle
    - **Meets 5-minute check interval requirement**

    ---

    ### 3. Real-Time vs Batch Monitoring

    **Trade-offs:**

    | Aspect | Real-Time | Batch (Every 5 min) |
    |--------|-----------|---------------------|
    | **Latency** | <1 second | 5 minutes |
    | **Compute Cost** | 100x higher | Baseline |
    | **Accuracy** | Per-prediction | Aggregate (10K samples) |
    | **Use Case** | Critical models (fraud) | Most models |

    **Hybrid Approach:**
    ```python
    class HybridMonitor:
        def __init__(self):
            self.critical_models = {"fraud-detection", "credit-scoring"}
            self.batch_models = set()

        async def log_prediction(self, prediction: Dict):
            model_id = prediction["model_id"]

            # Log to storage (always)
            await store_prediction(prediction)

            # Real-time drift check for critical models
            if model_id in self.critical_models:
                await self.real_time_drift_check(prediction)

        async def real_time_drift_check(self, prediction: Dict):
            """
            Lightweight drift check per prediction.
            Only compute simple metrics (e.g., feature range check).
            """
            model_id = prediction["model_id"]
            features = prediction["features"]

            # Load reference ranges (cached)
            ranges = get_reference_ranges(model_id)

            # Check if features within expected range
            for feature, value in features.items():
                min_val, max_val = ranges[feature]

                if value < min_val or value > max_val:
                    # Immediate alert
                    await send_alert(
                        model_id=model_id,
                        alert_type="feature_out_of_range",
                        message=f"Feature {feature} = {value} outside range [{min_val}, {max_val}]",
                        severity="warning"
                    )
    ```

    ---

    ### 4. Storage Optimization: Compression & TTL

    **Challenge:** 10 GB/day × 365 days = 3.65 TB/year per year.

    **Solution 1: Columnar Compression (Parquet)**

    ```python
    import pyarrow as pa
    import pyarrow.parquet as pq

    def store_predictions_compressed(predictions: List[Dict], output_path: str):
        """
        Store predictions in compressed Parquet format.

        Compression ratios:
        - JSON: 100 bytes per prediction
        - Parquet (Snappy): 30 bytes per prediction
        - Parquet (ZSTD): 20 bytes per prediction

        **5x compression!**
        """
        df = pd.DataFrame(predictions)

        # Write to Parquet with ZSTD compression
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            output_path,
            compression='zstd',
            compression_level=9
        )
    ```

    **Solution 2: Time-based Retention (TTL)**

    ```sql
    -- TimescaleDB: Automatic data retention
    SELECT add_retention_policy('predictions', INTERVAL '90 days');

    -- InfluxDB: Retention policy
    CREATE RETENTION POLICY "90_days" ON "monitoring"
        DURATION 90d
        REPLICATION 1
        DEFAULT;

    -- Downsample old metrics (reduce resolution)
    CREATE CONTINUOUS QUERY "downsample_1h" ON "monitoring"
    BEGIN
        SELECT mean(value) INTO metrics_1h FROM drift_metrics
        WHERE time < now() - 7d
        GROUP BY time(1h), model_id, feature_name
    END;
    ```

    **Storage Savings:**
    - Raw JSON: 10 GB/day × 365 = 3.65 TB/year
    - Parquet + compression: 2 GB/day × 365 = 730 GB/year
    - With 90-day retention: 2 GB × 90 = 180 GB
    - **95% reduction!**

    ---

    ## Bottleneck Analysis

    ### Scenario 1: Drift Detection is Slow

    **Diagnosis:**
    ```
    Problem: Drift checks take 10 minutes (too slow for 5-minute interval)

    Root cause:
    - Fetching 10K predictions from TimescaleDB: 5 seconds per model
    - Statistical computations: 400ms per model
    - 1,000 models × 5.4 seconds = 5,400 seconds (90 minutes!)
    ```

    **Solution: Materialized Views**
    ```sql
    -- Pre-aggregate features into time buckets
    CREATE MATERIALIZED VIEW feature_stats_5min AS
    SELECT
        model_id,
        feature_name,
        time_bucket('5 minutes', timestamp) as bucket,
        count(*) as count,
        avg((features->feature_name)::float) as mean,
        stddev((features->feature_name)::float) as stddev,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY (features->feature_name)::float) as median
    FROM predictions
    WHERE timestamp > now() - INTERVAL '1 day'
    GROUP BY model_id, feature_name, bucket;

    -- Refresh every 5 minutes
    CREATE INDEX ON feature_stats_5min (model_id, feature_name, bucket);

    -- Now drift detection reads aggregates (10ms instead of 5 seconds)
    -- 1,000 models × 10ms = 10 seconds total ✅
    ```

    ---

    ### Scenario 2: Alert Fatigue (Too Many Alerts)

    **Problem:** 200 drift alerts/day → Engineers ignore alerts.

    **Solution: Smart Alerting with Deduplication**

    ```python
    class SmartAlertManager:
        def __init__(self, cooldown_minutes: int = 60):
            self.cooldown_minutes = cooldown_minutes
            self.recent_alerts = {}  # {(model_id, alert_type): last_sent_time}

        def should_send_alert(
            self, model_id: str, alert_type: str, severity: str
        ) -> bool:
            """
            Deduplicate alerts to avoid fatigue.

            Rules:
            - Critical: Always send (no cooldown)
            - Warning: Max 1 per hour per model
            - Info: Max 1 per day per model
            """
            key = (model_id, alert_type)

            if severity == "critical":
                return True  # Always send critical alerts

            # Check cooldown
            if key in self.recent_alerts:
                last_sent = self.recent_alerts[key]
                elapsed_minutes = (time.time() - last_sent) / 60

                cooldown = self.cooldown_minutes if severity == "warning" else 1440  # 1 day for info

                if elapsed_minutes < cooldown:
                    return False  # Skip (too soon)

            # Send alert and record
            self.recent_alerts[key] = time.time()
            return True

        def aggregate_alerts(self, alerts: List[Dict]) -> Dict:
            """
            Aggregate multiple alerts into digest.

            Instead of:
            - "Feature A drifted on model X"
            - "Feature B drifted on model X"
            - "Feature C drifted on model X"

            Send:
            - "3 features drifted on model X (A, B, C)"
            """
            by_model = {}
            for alert in alerts:
                model_id = alert["model_id"]
                if model_id not in by_model:
                    by_model[model_id] = []
                by_model[model_id].append(alert)

            digests = []
            for model_id, model_alerts in by_model.items():
                feature_names = [a["feature"] for a in model_alerts]
                digests.append({
                    "model_id": model_id,
                    "alert_count": len(model_alerts),
                    "features": feature_names,
                    "message": f"{len(model_alerts)} features drifted: {', '.join(feature_names)}"
                })

            return digests
    ```

    **Result:**
    - Before: 200 alerts/day (ignored)
    - After: 20 aggregated alerts/day (actionable)
    - **90% reduction in alert noise**

=== "💡 Step 5: Additional Considerations"

    ## Security & Compliance

    ### 1. PII Protection

    ```python
    # Hash PII before logging
    import hashlib

    def anonymize_features(features: Dict) -> Dict:
        """Remove/hash PII from logged features."""
        pii_fields = {"email", "phone", "ssn", "address"}

        anonymized = {}
        for key, value in features.items():
            if key in pii_fields:
                # Hash PII
                anonymized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            else:
                anonymized[key] = value

        return anonymized
    ```

    ---

    ### 2. Model Explainability Tracking

    ```python
    import shap

    def log_prediction_with_explanation(model, features: Dict):
        """Log prediction with SHAP explanation (sampled 1%)."""
        prediction = model.predict(features)

        # Log prediction (always)
        await logger.log_prediction({
            "model_id": "fraud-detection-v2",
            "prediction": prediction,
            "features": features
        })

        # Compute SHAP (1% sample rate)
        if random.random() < 0.01:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)

            # Log SHAP values
            await logger.log_explanation({
                "model_id": "fraud-detection-v2",
                "prediction_id": prediction["id"],
                "shap_values": dict(zip(features.keys(), shap_values[0])),
                "base_value": explainer.expected_value[0]
            })
    ```

    ---

    ### 3. Cost Breakdown

    ```
    Monthly costs for 100M predictions/day:

    Compute:
    - Kafka (3 brokers): $500
    - Drift detection workers (20 instances): $1,200
    - API servers (10 instances): $800
    - Total compute: $2,500

    Storage:
    - TimescaleDB (predictions, 90 days): $300
    - InfluxDB (metrics): $200
    - PostgreSQL (metadata, alerts): $100
    - S3 (reference distributions): $50
    - Total storage: $650

    Data transfer:
    - Ingress (free on AWS)
    - Egress (dashboard queries): $100

    Total: $3,250/month ≈ $39K/year

    Cost per prediction: $39K / (100M × 365) = $0.000001 per prediction
    Cost per model: $39K / 1,000 models = $39/model/year
    ```

    ---

    ## Integration Patterns

    ### With Model Serving Platform

    ```python
    # Model serving integration
    from model_monitor import MonitoringClient

    monitor = MonitoringClient(api_key="xxx")

    @app.post("/predict")
    async def predict_endpoint(request: PredictRequest):
        # Run inference
        prediction = model.predict(request.features)

        # Log to monitoring (async, non-blocking)
        await monitor.log_prediction(
            model_id="fraud-detection-v2",
            prediction=prediction,
            features=request.features
        )

        return prediction
    ```

    ---

    ### Auto-Remediation (Trigger Retraining)

    ```python
    # Alert handler with auto-remediation
    @alert_handler("concept_drift")
    async def handle_concept_drift(alert: Alert):
        model_id = alert["model_id"]

        # Check if drift is sustained (not transient)
        recent_alerts = get_recent_alerts(model_id, hours=24)
        if len(recent_alerts) < 3:
            return  # Wait for more evidence

        # Trigger retraining workflow
        await trigger_retraining_job(
            model_id=model_id,
            reason="concept_drift_detected",
            priority="high"
        )

        # Notify team
        await send_notification(
            channel="slack",
            message=f"🔄 Auto-triggered retraining for {model_id} due to sustained concept drift"
        )
    ```

=== "🎯 Step 6: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5 minutes)

    **Key Questions:**
    - Scale? (number of models, predictions per day)
    - Latency requirements? (real-time vs batch monitoring)
    - Drift detection methods? (PSI, KS test, custom metrics)
    - Alert latency? (how fast should alerts trigger?)
    - Ground truth availability? (immediate, delayed, partial)
    - Protected attributes for bias detection?

    ---

    ### 2. Start with Use Case (2 minutes)

    "Let's design a model monitoring system for a company with 1,000 deployed ML models. The system needs to:
    1. Detect data drift when input distributions shift
    2. Alert on performance degradation when accuracy drops
    3. Monitor bias across demographic groups
    4. Support 100M predictions/day without impacting serving latency"

    ---

    ### 3. High-Level Architecture (10 minutes)

    Draw architecture diagram with:
    - Prediction logging (async, Kafka)
    - Drift detection workers (distributed)
    - Performance monitoring (with label joining)
    - Bias analyzer
    - Alert manager
    - Dashboard (React + InfluxDB)

    Explain the rationale for async logging and batch detection.

    ---

    ### 4. Deep Dive (20 minutes)

    Focus on 2-3 areas based on interviewer interest:
    - **Statistical drift detection:** PSI, KS test, KL divergence implementations
    - **Concept drift:** DDM, ADWIN algorithms
    - **Bias detection:** Demographic parity, equal opportunity metrics
    - **High-throughput logging:** Kafka, batching, sampling

    ---

    ### 5. Scale & Optimize (10 minutes)

    Discuss:
    - Sampling strategies (30% reduction)
    - Distributed drift detection (20 workers)
    - Real-time vs batch monitoring trade-offs
    - Storage optimization (Parquet, TTL, 5x compression)
    - Alert deduplication (90% noise reduction)

    ---

    ## Common Follow-Up Questions

    ### Q1: How do you prevent logging from slowing down inference?

    **Answer:**
    "We use async batching. The SDK queues predictions in-memory and a background thread uploads batches to Kafka every 10ms. This reduces logging latency to <1ms and avoids blocking the inference path. Kafka provides high throughput (1M+ msgs/sec) and durability."

    ---

    ### Q2: How do you handle delayed ground truth labels?

    **Answer:**
    "We use a label joiner service that continuously matches predictions to labels as they arrive. For fraud detection, labels might arrive 24 hours later. We track 'label delay' as a metric and compute performance metrics on a rolling window. Concept drift detection uses sequential methods (DDM, ADWIN) that adapt to varying label rates."

    ---

    ### Q3: Why multiple drift detection methods (PSI, KS, KL)?

    **Answer:**
    "Each method has strengths:
    - PSI: Simple, interpretable, works well for categorical features
    - KS test: Statistical significance (p-value), good for continuous features
    - KL divergence: Measures information loss, sensitive to tail changes

    Using multiple methods reduces false positives. We only alert when 2+ methods agree on drift."

    ---

    ### Q4: How do you reduce alert fatigue?

    **Answer:**
    "Three strategies:
    1. Deduplication: Max 1 alert per hour per model (except critical)
    2. Aggregation: Combine multiple feature drifts into single alert
    3. Thresholds: Use warning (PSI>0.1) and critical (PSI>0.25) levels

    This reduces alerts from 200/day to 20/day (90% reduction) while maintaining coverage."

    ---

    ### Q5: How do you scale to 1,000 models?

    **Answer:**
    "Three approaches:
    1. Distributed workers: 20 Celery workers process drift checks in parallel
    2. Materialized views: Pre-aggregate features (5 seconds → 10ms per model)
    3. Sampling: Log 30% of predictions for non-critical models

    This brings total processing time from 90 minutes to 10 seconds per check cycle."

    ---

    ## Red Flags to Avoid

    1. **Don't** compute drift on every prediction (too expensive, use batch checks)
    2. **Don't** store full predictions forever (use sampling + TTL)
    3. **Don't** use only one drift metric (false positives, use multiple)
    4. **Don't** ignore bias detection (regulatory requirement)
    5. **Don't** send raw alerts (deduplicate and aggregate)

    ---

    ## Bonus Points

    1. Mention **real-world systems:** Arize AI, WhyLabs, Evidently AI
    2. Discuss **auto-remediation:** Trigger retraining on sustained drift
    3. Talk about **explainability:** Track SHAP value distributions
    4. Consider **model comparison:** Compare drift across A/B test variants
    5. Mention **regulatory compliance:** GDPR, Fair Lending Act

=== "📚 References & Resources"

    ## Real-World Implementations

    ### Arize AI
    - **Architecture:** SaaS platform, statistical drift detection, bias monitoring
    - **Scale:** 100B+ predictions monitored
    - **Key Innovation:** Embedding drift for NLP/CV models, automated root cause analysis
    - **Website:** [arize.com](https://arize.com/)

    ---

    ### WhyLabs
    - **Architecture:** Lightweight client, privacy-preserving (only profiles sent)
    - **Scale:** On-device monitoring, edge deployments
    - **Key Innovation:** Data sketches (HyperLogLog, KLL), no raw data upload
    - **Website:** [whylabs.ai](https://whylabs.ai/)

    ---

    ### Evidently AI
    - **Architecture:** Open-source, Python library + dashboard
    - **Scale:** Single-node, suitable for small-medium deployments
    - **Key Innovation:** Interactive HTML reports, test suites
    - **Website:** [evidentlyai.com](https://evidentlyai.com/)

    ---

    ### Fiddler AI
    - **Architecture:** Enterprise platform, explainability + monitoring
    - **Scale:** Fortune 500 customers
    - **Key Innovation:** Feature impact analysis, what-if analysis
    - **Website:** [fiddler.ai](https://fiddler.ai/)

    ---

    ## Open Source Tools

    ### Evidently
    ```python
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_df, current_data=prod_df)
    report.save_html("drift_report.html")
    ```

    ---

    ### Great Expectations
    ```python
    import great_expectations as ge

    df = ge.read_csv("predictions.csv")
    df.expect_column_values_to_be_between("transaction_amount", 0, 10000)
    df.expect_column_values_to_be_in_set("merchant_category", ["retail", "food", "electronics"])
    ```

    ---

    ### WhyLogs
    ```python
    import whylogs as why

    # Profile production data
    profile = why.log(production_df).profile()

    # Compare to reference
    reference_profile = why.read("reference_profile.bin")
    diff = profile.diff(reference_profile)

    # Check for drift
    if diff.summary["drift_detected"]:
        print("Drift detected!")
    ```

    ---

    ## Academic Papers

    1. **"A Survey on Concept Drift Adaptation"** - Gama et al. (2014)
    2. **"Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift"** - Rabanser et al. (2019)
    3. **"Fairness and Abstraction in Sociotechnical Systems"** - Selbst et al. (2019)
    4. **"Learning under Concept Drift: A Review"** - Lu et al. (2018)

    ---

    ## Related System Design Problems

    1. **Model Serving Platform** - Provides predictions to monitor (upstream)
    2. **ML Experiment Tracking** - Stores baseline metrics for comparison
    3. **Feature Store** - Monitors feature quality and freshness
    4. **Data Quality Platform** - Similar drift detection for data pipelines
    5. **Observability Platform** - Similar alerting patterns (Prometheus, Grafana)

---

## Summary

An **ML Model Monitoring System** detects drift, performance degradation, and bias in production models:

**Key Components:**
- **Prediction Logging:** Async batching via Kafka (100K+ logs/sec, <1ms overhead)
- **Drift Detection:** Statistical tests (PSI, KS, KL divergence) on 5-minute intervals
- **Concept Drift:** Sequential algorithms (DDM, ADWIN) for accuracy monitoring
- **Bias Detection:** Fairness metrics (demographic parity, equal opportunity)
- **Alert Manager:** Smart deduplication and aggregation (90% noise reduction)
- **Dashboard:** Real-time visualization with InfluxDB + React

**Core Challenges:**
- Non-blocking prediction logging at high throughput
- Statistical rigor in drift detection (low false positive rate)
- Delayed ground truth labels (hours to days)
- Multi-model scalability (1,000+ models)
- Alert fatigue prevention
- Storage cost optimization

**Architecture Decisions:**
- Async logging (Kafka) to avoid inference latency impact
- Batch drift detection (every 5 minutes) vs per-prediction (100x cheaper)
- Multiple drift metrics (PSI, KS, KL) for robustness
- Distributed workers (Celery) for parallel processing
- Smart sampling (30% reduction) for high-volume models
- Columnar compression (Parquet, 5x reduction) with TTL

This is a **medium difficulty** problem that combines statistics, distributed systems, and ML operations. Focus on statistical methods, async logging, and cost optimization during interviews.
