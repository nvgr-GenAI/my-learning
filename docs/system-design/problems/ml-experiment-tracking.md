# Design ML Experiment Tracking System (MLflow/Weights & Biases)

A comprehensive experiment tracking platform that enables data scientists to log, compare, and reproduce machine learning experiments at scale, tracking metrics, parameters, artifacts, and lineage with real-time visualization, collaboration features, and reproducibility guarantees.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K experiments/month, 1GB artifacts per experiment, 100K metric logs/sec, 50+ concurrent users |
| **Key Challenges** | Large artifact storage, real-time metric streaming, experiment comparison, reproducibility, search performance |
| **Core Concepts** | Experiment runs, metrics logging, artifact storage (S3), metadata DB, versioning, lineage tracking |
| **Companies** | MLflow, Weights & Biases, Neptune, Comet, AWS SageMaker, Databricks, TensorBoard |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Experiment Creation** | Create experiments with name, description, tags | P0 (Must have) |
    | **Metrics Logging** | Log training metrics (loss, accuracy) over time | P0 (Must have) |
    | **Parameter Tracking** | Track hyperparameters (learning rate, batch size) | P0 (Must have) |
    | **Artifact Storage** | Store models, datasets, plots, logs (up to 10GB) | P0 (Must have) |
    | **Experiment Comparison** | Compare multiple runs side-by-side | P0 (Must have) |
    | **Search & Filter** | Query experiments by metrics, params, tags | P0 (Must have) |
    | **Run Versioning** | Track experiment history and reproduce runs | P0 (Must have) |
    | **Real-time Visualization** | Live charts for training metrics | P1 (Should have) |
    | **Collaboration** | Share experiments, comments, annotations | P1 (Should have) |
    | **Lineage Tracking** | Track data, code, model dependencies | P1 (Should have) |
    | **Artifact Deduplication** | Avoid storing duplicate artifacts | P2 (Nice to have) |
    | **Auto-logging** | Automatic metric/param capture from frameworks | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (Kubeflow, SageMaker)
    - Hyperparameter optimization (Optuna, Ray Tune)
    - Model serving and deployment
    - Data labeling and annotation
    - Feature engineering pipelines
    - Model monitoring in production (drift detection)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Data scientists rely on it daily |
    | **Metric Logging Latency** | < 100ms per batch (100 metrics) | Should not slow training loops |
    | **Dashboard Load Time** | < 3 seconds for 100 runs | Interactive exploration |
    | **Search Performance** | < 2 seconds for 10K experiments | Fast filtering |
    | **Artifact Upload** | 1GB in < 5 minutes | Large model checkpoints |
    | **Storage Retention** | Unlimited (user-controlled cleanup) | Historical reference |
    | **Concurrent Users** | 50+ users simultaneously | Team collaboration |
    | **Data Durability** | 99.999999999% (S3 standard) | Cannot lose training results |
    | **Scalability** | 100K experiments, 1PB artifacts | Company-wide adoption |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Total data scientists: 100 users
    - Active concurrent users: 20 users (peak hours)
    - Experiments per month: 10,000 experiments
    - Average experiment duration: 2 hours
    - Concurrent training runs: 50 runs

    Metrics logging:
    - Metrics per experiment: 10 metrics (loss, accuracy, etc.)
    - Logging frequency: Every 10 steps (100 steps/epoch, 100 epochs)
    - Total logs per experiment: 10 metrics × 10 logs/epoch × 100 epochs = 10,000 logs
    - Daily logs: 10,000 experiments/month × 10,000 logs / 30 days = 3.3M logs/day
    - Logs per second: 3.3M / 86,400 = ~38 logs/sec average
    - Peak logging: 50 concurrent runs × 10 metrics × 1 log/sec = 500 logs/sec

    Parameter tracking:
    - Parameters per experiment: 50 params (hyperparameters, config)
    - Total params: 10,000 experiments × 50 = 500K params/month
    - Params are logged once per run (not frequent)

    Artifact storage:
    - Artifacts per experiment: 5 artifacts (model, dataset, plots, logs, config)
    - Average artifact size: 1 GB (model checkpoints dominate)
    - Total artifacts: 10,000 experiments × 5 artifacts = 50K artifacts/month
    - Storage per month: 10,000 experiments × 1 GB = 10 TB/month
    - Annual storage: 120 TB/year (cumulative)

    Dashboard/UI traffic:
    - Experiment views per day: 100 users × 20 views = 2,000 views/day
    - QPS for reads: 2,000 / 86,400 = 0.02 QPS (very low, mostly interactive)
    - Search queries: 100 users × 5 searches/day = 500 searches/day

    Total write QPS: ~500 (metrics) + 0.1 (experiments) = ~500 QPS
    Total read QPS: ~50 (dashboard queries, comparisons)
    Write-to-read ratio: 10:1 (write-heavy)
    ```

    ### Storage Estimates

    ```
    Metadata storage (PostgreSQL):
    - Experiments: 100K experiments × 2 KB = 200 MB
    - Runs: 100K runs × 1 KB = 100 MB
    - Metrics: 100K runs × 10K logs × 100 bytes = 100 GB
    - Parameters: 100K runs × 50 params × 200 bytes = 1 GB
    - Tags/metadata: 100K runs × 500 bytes = 50 MB
    - Total metadata: ~102 GB (with indexes: ~150 GB)

    Artifact storage (S3):
    - Annual artifacts: 120 TB/year
    - 3 years retention: 360 TB
    - With compression (2x for models): 180 TB
    - With deduplication (20% savings): 144 TB

    Time-series metrics (InfluxDB/TimescaleDB):
    - Metrics per run: 10K logs × 100 bytes = 1 MB
    - Total metrics: 100K runs × 1 MB = 100 GB
    - With downsampling (older data): 50 GB

    Total storage: 150 GB (metadata) + 144 TB (artifacts) + 50 GB (metrics) ≈ 144.2 TB
    ```

    ### Bandwidth Estimates

    ```
    Ingress (uploads):
    - Metrics: 500 logs/sec × 100 bytes = 50 KB/sec ≈ 0.4 Mbps
    - Artifacts: 10 TB/month / 30 days / 86,400 = 3.86 MB/sec ≈ 31 Mbps
    - Total ingress: ~31.4 Mbps

    Egress (downloads):
    - Dashboard queries: 50 QPS × 10 KB = 500 KB/sec ≈ 4 Mbps
    - Artifact downloads: 1,000 downloads/day × 1 GB = 1 TB/day
    - Average egress: 1 TB / 86,400 = 11.6 MB/sec ≈ 93 Mbps
    - Total egress: ~97 Mbps

    Total bandwidth: ~128 Mbps (1 Gbps link sufficient)
    ```

    ### Memory Estimates

    ```
    Application servers:
    - In-memory cache (recent experiments): 1 GB
    - Connection pools (DB, S3): 500 MB
    - Per-server memory: 4 GB
    - Total servers (5 replicas): 20 GB

    Database (PostgreSQL):
    - Shared buffers (25% of data): 40 GB
    - Connection memory: 5 GB
    - Total DB memory: 64 GB (server RAM)

    Time-series DB (InfluxDB):
    - Cache for recent metrics: 10 GB
    - Total: 16 GB (server RAM)

    Redis cache (dashboard queries):
    - Cache 1% of experiments: 2 GB
    - Session storage: 1 GB
    - Total: 4 GB

    Total memory: 20 GB (app) + 64 GB (DB) + 16 GB (InfluxDB) + 4 GB (Redis) ≈ 104 GB
    ```

    ---

    ## Key Assumptions

    1. Average experiment runs for 2 hours with 100 epochs
    2. 80% of artifacts are model checkpoints (large), 20% are plots/logs (small)
    3. Users primarily search/filter by recent experiments (last 30 days)
    4. Metric logging is bursty (training phase) vs. idle (no training)
    5. Most experiments are never revisited after 1 month (cold storage eligible)
    6. 20% of artifacts are duplicates (same dataset used across runs)
    7. Real-time dashboard updates needed only for active runs (50 concurrent)
    8. Users collaborate within teams (10-20 users per team)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Separation of concerns:** Metadata (DB), metrics (time-series DB), artifacts (object storage)
    2. **SDK-first approach:** Python SDK for seamless integration into training code
    3. **Asynchronous logging:** Non-blocking metric/artifact uploads
    4. **Scalable storage:** S3 for artifacts, distributed DB for metadata
    5. **Real-time streaming:** WebSocket for live metric updates
    6. **Reproducibility:** Track code version, environment, data lineage

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Training Environment"
            TrainingScript[Training Script<br/>Python/PyTorch/TF]
            SDK[MLflow/W&B SDK<br/>Python Client<br/>Async logging]
        end

        subgraph "API Layer"
            APIGateway[API Gateway<br/>REST/gRPC<br/>Rate limiting]
            TrackingServer[Tracking Server<br/>FastAPI<br/>Handles logs/artifacts]
            WebSocketServer[WebSocket Server<br/>Real-time updates]
        end

        subgraph "Storage Layer"
            MetadataDB[(Metadata DB<br/>PostgreSQL<br/>Experiments, runs, params)]
            MetricsDB[(Time-Series DB<br/>InfluxDB/TimescaleDB<br/>Training metrics)]
            ArtifactStore[(Artifact Store<br/>S3/MinIO<br/>Models, datasets, plots)]
            CacheDB[(Cache<br/>Redis<br/>Recent experiments)]
        end

        subgraph "Processing Layer"
            ComparisonService[Comparison Engine<br/>Parallel aggregation<br/>Statistical analysis]
            SearchService[Search Service<br/>Elasticsearch<br/>Full-text + filters]
            LineageService[Lineage Tracker<br/>Graph DB (Neo4j)<br/>Dependencies]
        end

        subgraph "Frontend"
            WebUI[Web Dashboard<br/>React<br/>Charts, comparisons]
            Notebook[Jupyter Integration<br/>Inline visualizations]
        end

        subgraph "Background Jobs"
            DeduplicationWorker[Deduplication Worker<br/>Content hashing<br/>Artifact cleanup]
            DownsamplingWorker[Downsampling Worker<br/>Aggregate old metrics<br/>Reduce storage]
            AlertingService[Alerting Service<br/>Slack/email<br/>Experiment failures]
        end

        TrainingScript -->|1. Log metrics/params| SDK
        SDK -->|2. Batch upload| APIGateway
        APIGateway -->|3. Route request| TrackingServer

        TrackingServer -->|4a. Store metadata| MetadataDB
        TrackingServer -->|4b. Store metrics| MetricsDB
        TrackingServer -->|4c. Upload artifacts| ArtifactStore
        TrackingServer -->|4d. Cache recent| CacheDB

        TrackingServer -->|5. Push updates| WebSocketServer
        WebSocketServer -->|6. Live updates| WebUI

        WebUI -->|7. Search/filter| SearchService
        SearchService -->|8. Query index| MetadataDB

        WebUI -->|9. Compare runs| ComparisonService
        ComparisonService -->|10. Fetch data| MetricsDB
        ComparisonService -->|10. Fetch data| MetadataDB

        WebUI -->|11. View lineage| LineageService

        SDK -->|12. Download artifacts| ArtifactStore
        Notebook -->|13. Load experiments| TrackingServer

        DeduplicationWorker -.->|Detect duplicates| ArtifactStore
        DownsamplingWorker -.->|Compress old data| MetricsDB
        AlertingService -.->|Monitor| MetadataDB

        style SDK fill:#e1f5ff
        style ArtifactStore fill:#ffe1e1
        style MetricsDB fill:#fff4e1
        style TrackingServer fill:#e8f5e9
        style ComparisonService fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **PostgreSQL (Metadata)** | ACID, relations (experiments→runs), complex queries (search/filter) | MongoDB (no joins, weak consistency), DynamoDB (expensive queries) |
    | **InfluxDB (Metrics)** | Time-series optimized, high write throughput, automatic downsampling | PostgreSQL (slow for time-series), Cassandra (no time-series functions) |
    | **S3 (Artifacts)** | Cheap, durable (11 9's), scalable (PB+), lifecycle policies | HDFS (operational overhead), EBS (expensive), DB (not for large blobs) |
    | **Redis (Cache)** | Sub-millisecond reads, reduce DB load for hot experiments | Memcached (no persistence), in-memory (lost on restart) |
    | **Elasticsearch (Search)** | Full-text search, complex filters, aggregations | PostgreSQL LIKE (slow), custom indexing (reinvent wheel) |
    | **WebSocket (Streaming)** | Real-time metric updates, low latency | Polling (high latency, waste bandwidth), SSE (one-way only) |

    **Key Trade-off:** We chose **asynchronous SDK logging over synchronous** to avoid blocking training loops. This introduces eventual consistency (metrics may appear delayed by 1-10 seconds), but prevents training slowdown.

    ---

    ## API Design

    ### 1. Create Experiment

    **Request:**
    ```python
    import mlflow

    # Create experiment
    experiment_id = mlflow.create_experiment(
        name="fraud-detection-v2",
        artifact_location="s3://ml-artifacts/fraud-detection",
        tags={"team": "risk", "priority": "high"}
    )
    ```

    **Response:**
    ```json
    {
      "experiment_id": "exp_1234567890",
      "name": "fraud-detection-v2",
      "artifact_location": "s3://ml-artifacts/fraud-detection",
      "lifecycle_stage": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
    ```

    ---

    ### 2. Start Run (Training Session)

    **Request:**
    ```python
    with mlflow.start_run(experiment_id=experiment_id, run_name="baseline-lr-0.001") as run:
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "epochs": 100,
            "model": "resnet50"
        })

        # Training loop
        for epoch in range(100):
            train_loss, train_acc = train_epoch()
            val_loss, val_acc = validate()

            # Log metrics (batched internally by SDK)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

        # Log artifacts
        mlflow.log_artifact("model.pth")  # Model checkpoint
        mlflow.log_artifact("training_plot.png")  # Loss curves
        mlflow.log_dict({"config": config}, "config.json")
    ```

    **Response:**
    ```json
    {
      "run_id": "run_abcd1234",
      "experiment_id": "exp_1234567890",
      "run_name": "baseline-lr-0.001",
      "status": "RUNNING",
      "start_time": 1705315800000,
      "artifact_uri": "s3://ml-artifacts/fraud-detection/run_abcd1234/artifacts",
      "user_id": "alice@company.com"
    }
    ```

    ---

    ### 3. Log Metrics (Time-Series Data)

    **Internal SDK Request (Batched):**
    ```python
    # SDK batches metrics internally (every 10 seconds or 100 metrics)
    POST /api/2.0/mlflow/runs/log-batch
    {
      "run_id": "run_abcd1234",
      "metrics": [
        {"key": "train_loss", "value": 0.543, "timestamp": 1705315800000, "step": 0},
        {"key": "train_accuracy", "value": 0.812, "timestamp": 1705315800000, "step": 0},
        {"key": "val_loss", "value": 0.598, "timestamp": 1705315800000, "step": 0},
        {"key": "val_accuracy", "value": 0.789, "timestamp": 1705315800000, "step": 0}
      ]
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "metrics_logged": 4
    }
    ```

    **Performance:** Batching reduces network calls by 100x (1 call per 100 metrics vs. 100 calls).

    ---

    ### 4. Search Experiments

    **Request:**
    ```python
    experiments = mlflow.search_runs(
        experiment_ids=["exp_1234567890"],
        filter_string="metrics.val_accuracy > 0.9 AND params.learning_rate < 0.01",
        order_by=["metrics.val_accuracy DESC"],
        max_results=10
    )
    ```

    **REST API:**
    ```http
    POST /api/2.0/mlflow/runs/search
    {
      "experiment_ids": ["exp_1234567890"],
      "filter": "metrics.val_accuracy > 0.9 AND params.learning_rate < 0.01",
      "order_by": ["metrics.val_accuracy DESC"],
      "max_results": 10
    }
    ```

    **Response:**
    ```json
    {
      "runs": [
        {
          "run_id": "run_xyz789",
          "run_name": "tuned-lr-0.005",
          "metrics": {"val_accuracy": 0.932, "val_loss": 0.412},
          "params": {"learning_rate": 0.005, "batch_size": 64},
          "start_time": 1705315800000,
          "end_time": 1705322600000,
          "duration_seconds": 6800
        }
      ],
      "total_count": 15
    }
    ```

    ---

    ### 5. Compare Runs

    **Request:**
    ```python
    comparison = mlflow.compare_runs(
        run_ids=["run_abcd1234", "run_xyz789", "run_def456"]
    )
    ```

    **Response:**
    ```json
    {
      "runs": [
        {
          "run_id": "run_abcd1234",
          "metrics": {"val_accuracy": 0.812},
          "params": {"learning_rate": 0.001}
        },
        {
          "run_id": "run_xyz789",
          "metrics": {"val_accuracy": 0.932},
          "params": {"learning_rate": 0.005}
        }
      ],
      "comparison_table": {
        "val_accuracy": [0.812, 0.932],
        "learning_rate": [0.001, 0.005]
      },
      "best_run": "run_xyz789"
    }
    ```

    ---

    ### 6. Download Artifact

    **Request:**
    ```python
    # Download model from best run
    artifact_path = mlflow.download_artifacts(
        run_id="run_xyz789",
        artifact_path="model.pth",
        dst_path="./models"
    )
    ```

    **Generates Presigned S3 URL:**
    ```http
    GET /api/2.0/mlflow/artifacts?run_id=run_xyz789&path=model.pth

    Response:
    {
      "presigned_url": "https://s3.amazonaws.com/ml-artifacts/...?AWSAccessKeyId=...",
      "expires_in": 3600
    }
    ```

    **Performance:** Direct S3 download (not through server) → 100 MB/s transfer speed.

    ---

    ## Database Schema

    ### Metadata DB (PostgreSQL)

    ```sql
    -- Experiments (top-level container)
    CREATE TABLE experiments (
        experiment_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        artifact_location VARCHAR(500),
        lifecycle_stage VARCHAR(20) DEFAULT 'active',  -- active, deleted
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        created_by VARCHAR(100)
    );
    CREATE INDEX idx_experiments_name ON experiments(name);
    CREATE INDEX idx_experiments_created ON experiments(created_at DESC);

    -- Runs (individual training sessions)
    CREATE TABLE runs (
        run_id VARCHAR(50) PRIMARY KEY,
        experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
        run_name VARCHAR(255),
        user_id VARCHAR(100),
        status VARCHAR(20),  -- RUNNING, FINISHED, FAILED
        start_time BIGINT,  -- Unix timestamp ms
        end_time BIGINT,
        source_type VARCHAR(50),  -- NOTEBOOK, JOB, LOCAL
        source_name VARCHAR(500),  -- Script path
        artifact_uri VARCHAR(500),
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_runs_experiment ON runs(experiment_id);
    CREATE INDEX idx_runs_status ON runs(status);
    CREATE INDEX idx_runs_user ON runs(user_id);
    CREATE INDEX idx_runs_start_time ON runs(start_time DESC);

    -- Parameters (hyperparameters, config)
    CREATE TABLE params (
        run_id VARCHAR(50) REFERENCES runs(run_id),
        key VARCHAR(255) NOT NULL,
        value VARCHAR(500) NOT NULL,
        PRIMARY KEY (run_id, key)
    );
    CREATE INDEX idx_params_key ON params(key);
    CREATE INDEX idx_params_value ON params(value);

    -- Metric summaries (latest value per metric)
    CREATE TABLE latest_metrics (
        run_id VARCHAR(50) REFERENCES runs(run_id),
        key VARCHAR(255) NOT NULL,
        value DOUBLE PRECISION,
        timestamp BIGINT,
        step BIGINT,
        PRIMARY KEY (run_id, key)
    );
    CREATE INDEX idx_metrics_key ON latest_metrics(key);
    CREATE INDEX idx_metrics_value ON latest_metrics(value);

    -- Tags (metadata, team, priority)
    CREATE TABLE tags (
        run_id VARCHAR(50) REFERENCES runs(run_id),
        key VARCHAR(255) NOT NULL,
        value VARCHAR(500),
        PRIMARY KEY (run_id, key)
    );
    CREATE INDEX idx_tags_key ON tags(key);

    -- Artifacts (references to S3)
    CREATE TABLE artifacts (
        artifact_id VARCHAR(50) PRIMARY KEY,
        run_id VARCHAR(50) REFERENCES runs(run_id),
        path VARCHAR(500) NOT NULL,  -- Relative path within run
        size_bytes BIGINT,
        content_hash VARCHAR(64),  -- SHA256 for deduplication
        s3_key VARCHAR(1000),  -- S3 object key
        uploaded_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX idx_artifacts_run ON artifacts(run_id);
    CREATE INDEX idx_artifacts_hash ON artifacts(content_hash);
    ```

    ---

    ### Time-Series DB (InfluxDB)

    **Schema:**
    ```
    Measurement: metrics

    Tags (indexed):
    - run_id
    - key (metric name: train_loss, val_accuracy, etc.)

    Fields:
    - value (float)
    - step (int)

    Timestamp: Unix timestamp (nanoseconds)
    ```

    **Example Query:**
    ```sql
    -- Get training loss for a run
    SELECT value FROM metrics
    WHERE run_id = 'run_abcd1234' AND key = 'train_loss'
    ORDER BY time ASC

    -- Compare val_accuracy across runs
    SELECT MEAN(value) FROM metrics
    WHERE key = 'val_accuracy' AND run_id IN ('run_1', 'run_2', 'run_3')
    GROUP BY run_id
    ```

    **Retention Policy (Downsampling):**
    ```sql
    -- Keep raw data for 30 days
    CREATE RETENTION POLICY raw_30d ON mlflow_db DURATION 30d REPLICATION 1 DEFAULT

    -- Downsample to 1-minute averages, keep for 1 year
    CREATE CONTINUOUS QUERY downsample_1m ON mlflow_db
    BEGIN
      SELECT MEAN(value) INTO metrics_1m FROM metrics
      GROUP BY time(1m), run_id, key
    END
    ```

    ---

    ### Artifact Store (S3)

    **Bucket Structure:**
    ```
    s3://ml-artifacts/
    ├── {experiment_id}/
    │   ├── {run_id}/
    │   │   ├── artifacts/
    │   │   │   ├── model.pth
    │   │   │   ├── training_plot.png
    │   │   │   ├── config.json
    │   │   │   └── logs/
    │   │   │       └── training.log
    │   │   └── metadata/
    │   │       └── run_info.json
    ```

    **S3 Lifecycle Policy:**
    ```json
    {
      "Rules": [
        {
          "Id": "TransitionToIA",
          "Status": "Enabled",
          "Transitions": [
            {
              "Days": 90,
              "StorageClass": "STANDARD_IA"
            }
          ]
        },
        {
          "Id": "TransitionToGlacier",
          "Status": "Enabled",
          "Transitions": [
            {
              "Days": 365,
              "StorageClass": "GLACIER"
            }
          ]
        }
      ]
    }
    ```

    **Cost Savings:**
    - S3 Standard (0-90 days): $0.023/GB/month
    - S3 IA (90-365 days): $0.0125/GB/month (46% cheaper)
    - S3 Glacier (365+ days): $0.004/GB/month (83% cheaper)

=== "🔧 Step 3: Deep Dive"

    ## 1. Asynchronous Metric Logging (SDK Design)

    **Challenge:** Logging metrics synchronously blocks training loops, slowing down training by 10-50%.

    **Solution: Async Batching SDK**

    ```python
    import threading
    import queue
    import time
    import requests
    from typing import Dict, Any

    class MLflowClient:
        def __init__(self, tracking_uri: str):
            self.tracking_uri = tracking_uri
            self.metrics_queue = queue.Queue(maxsize=10000)
            self.batch_size = 100
            self.flush_interval = 10  # seconds

            # Start background thread
            self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self.worker_thread.start()

        def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int):
            """Non-blocking metric logging"""
            timestamp = int(time.time() * 1000)

            for key, value in metrics.items():
                metric_data = {
                    "run_id": run_id,
                    "key": key,
                    "value": value,
                    "step": step,
                    "timestamp": timestamp
                }

                try:
                    # Non-blocking put (timeout 0.1s)
                    self.metrics_queue.put(metric_data, timeout=0.1)
                except queue.Full:
                    # Log warning but don't block training
                    print(f"Warning: Metrics queue full, dropping metric {key}")

        def _batch_worker(self):
            """Background thread that batches and uploads metrics"""
            batch = []
            last_flush = time.time()

            while True:
                try:
                    # Get metric from queue (block up to 1 second)
                    metric = self.metrics_queue.get(timeout=1.0)
                    batch.append(metric)

                    # Flush if batch full or interval elapsed
                    should_flush = (
                        len(batch) >= self.batch_size or
                        (time.time() - last_flush) >= self.flush_interval
                    )

                    if should_flush and batch:
                        self._flush_batch(batch)
                        batch = []
                        last_flush = time.time()

                except queue.Empty:
                    # Flush any pending metrics
                    if batch:
                        self._flush_batch(batch)
                        batch = []
                        last_flush = time.time()

        def _flush_batch(self, batch):
            """Upload batch of metrics to server"""
            try:
                # Group by run_id for efficiency
                runs_data = {}
                for metric in batch:
                    run_id = metric["run_id"]
                    if run_id not in runs_data:
                        runs_data[run_id] = []
                    runs_data[run_id].append({
                        "key": metric["key"],
                        "value": metric["value"],
                        "step": metric["step"],
                        "timestamp": metric["timestamp"]
                    })

                # Send batched request per run
                for run_id, metrics in runs_data.items():
                    response = requests.post(
                        f"{self.tracking_uri}/api/2.0/mlflow/runs/log-batch",
                        json={"run_id": run_id, "metrics": metrics},
                        timeout=5
                    )
                    response.raise_for_status()

                print(f"Flushed {len(batch)} metrics")

            except Exception as e:
                print(f"Error flushing metrics: {e}")
                # Could retry or write to local disk

        def flush(self):
            """Explicit flush (call at end of training)"""
            # Wait for queue to empty
            while not self.metrics_queue.empty():
                time.sleep(0.1)
    ```

    **Usage in Training:**
    ```python
    client = MLflowClient(tracking_uri="http://mlflow-server:5000")

    for epoch in range(100):
        for batch in dataloader:
            loss = train_step(batch)

            # Non-blocking! Returns immediately
            client.log_metrics(
                run_id="run_abcd1234",
                metrics={"train_loss": loss},
                step=global_step
            )
            global_step += 1

    # Wait for all metrics to upload
    client.flush()
    ```

    **Performance:**
    - Synchronous: 10ms per log call × 10,000 calls = 100 seconds overhead
    - Asynchronous: <0.1ms per log call × 10,000 calls = 1 second overhead
    - **100x speedup!**

    ---

    ## 2. Real-Time Metric Streaming (WebSocket)

    **Challenge:** Dashboard needs live updates without polling (wastes bandwidth).

    **Solution: WebSocket Server**

    ```python
    from fastapi import FastAPI, WebSocket
    from typing import Dict, Set
    import asyncio
    import json

    app = FastAPI()

    # Track active connections per run
    active_connections: Dict[str, Set[WebSocket]] = {}

    @app.websocket("/ws/runs/{run_id}")
    async def websocket_endpoint(websocket: WebSocket, run_id: str):
        await websocket.accept()

        # Register connection
        if run_id not in active_connections:
            active_connections[run_id] = set()
        active_connections[run_id].add(websocket)

        try:
            # Keep connection alive, wait for close
            while True:
                await websocket.receive_text()
        except:
            pass
        finally:
            # Unregister on disconnect
            active_connections[run_id].discard(websocket)

    async def broadcast_metric(run_id: str, metric_data: dict):
        """Called when new metric is logged"""
        if run_id in active_connections:
            disconnected = set()

            for websocket in active_connections[run_id]:
                try:
                    await websocket.send_json(metric_data)
                except:
                    disconnected.add(websocket)

            # Remove dead connections
            active_connections[run_id] -= disconnected

    # Hook into tracking server
    @app.post("/api/2.0/mlflow/runs/log-batch")
    async def log_batch(data: dict):
        run_id = data["run_id"]
        metrics = data["metrics"]

        # Store in database (existing logic)
        await store_metrics(run_id, metrics)

        # Broadcast to active WebSocket connections
        for metric in metrics:
            await broadcast_metric(run_id, {
                "type": "metric",
                "key": metric["key"],
                "value": metric["value"],
                "step": metric["step"],
                "timestamp": metric["timestamp"]
            })

        return {"status": "success"}
    ```

    **Frontend (React):**
    ```javascript
    // Connect to WebSocket
    const ws = new WebSocket(`ws://mlflow-server:5000/ws/runs/${runId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "metric") {
        // Update chart in real-time
        chart.addPoint({
          x: data.step,
          y: data.value,
          metric: data.key
        });
      }
    };
    ```

    **Performance:**
    - Polling (1 req/sec): 60 requests/minute × 50 users = 3,000 req/min
    - WebSocket: 50 connections, 0 polling requests
    - **Bandwidth savings: 99%**

    ---

    ## 3. Experiment Comparison Engine

    **Challenge:** Comparing 100 runs with 10K metrics each (1M data points) is slow.

    **Solution: Parallel Aggregation with Sampling**

    ```python
    import asyncio
    from typing import List, Dict
    import pandas as pd
    import numpy as np

    class ComparisonEngine:
        def __init__(self, influxdb_client, postgres_client):
            self.influxdb = influxdb_client
            self.postgres = postgres_client

        async def compare_runs(self, run_ids: List[str]) -> Dict:
            """Compare multiple runs efficiently"""

            # Fetch data in parallel
            tasks = [
                self._fetch_run_metadata(run_id) for run_id in run_ids
            ] + [
                self._fetch_run_metrics(run_id) for run_id in run_ids
            ]

            results = await asyncio.gather(*tasks)

            # Split results
            metadata_list = results[:len(run_ids)]
            metrics_list = results[len(run_ids):]

            # Compute comparison
            comparison = {
                "runs": [],
                "metric_comparison": {},
                "param_comparison": {},
                "statistical_tests": {}
            }

            for i, run_id in enumerate(run_ids):
                metadata = metadata_list[i]
                metrics = metrics_list[i]

                comparison["runs"].append({
                    "run_id": run_id,
                    "run_name": metadata["run_name"],
                    "params": metadata["params"],
                    "final_metrics": self._get_final_metrics(metrics)
                })

            # Compute metric comparison (side-by-side)
            comparison["metric_comparison"] = self._compute_metric_comparison(
                run_ids, metrics_list
            )

            # Statistical significance tests
            comparison["statistical_tests"] = self._compute_significance(
                metrics_list
            )

            return comparison

        async def _fetch_run_metadata(self, run_id: str) -> Dict:
            """Fetch params and metadata from PostgreSQL"""
            query = """
                SELECT r.run_name, r.status, r.start_time, r.end_time,
                       json_object_agg(p.key, p.value) as params
                FROM runs r
                LEFT JOIN params p ON r.run_id = p.run_id
                WHERE r.run_id = %s
                GROUP BY r.run_id
            """
            result = await self.postgres.fetch_one(query, (run_id,))
            return result

        async def _fetch_run_metrics(self, run_id: str) -> pd.DataFrame:
            """Fetch metrics from InfluxDB (downsampled for speed)"""

            # For comparison, we don't need every single point
            # Downsample to 1000 points max
            query = f"""
                SELECT time, key, value, step
                FROM metrics
                WHERE run_id = '{run_id}'
                AND time > now() - 30d
                SAMPLE 1000
            """

            result = await self.influxdb.query(query)
            return pd.DataFrame(result)

        def _get_final_metrics(self, metrics_df: pd.DataFrame) -> Dict:
            """Get final value for each metric"""
            final_metrics = {}

            for metric_name in metrics_df["key"].unique():
                metric_data = metrics_df[metrics_df["key"] == metric_name]
                final_value = metric_data.iloc[-1]["value"]  # Last value
                final_metrics[metric_name] = final_value

            return final_metrics

        def _compute_metric_comparison(
            self, run_ids: List[str], metrics_list: List[pd.DataFrame]
        ) -> Dict:
            """Create side-by-side metric comparison"""
            comparison = {}

            # Find common metrics
            all_metrics = set()
            for metrics_df in metrics_list:
                all_metrics.update(metrics_df["key"].unique())

            for metric_name in all_metrics:
                comparison[metric_name] = {}

                for i, run_id in enumerate(run_ids):
                    metrics_df = metrics_list[i]
                    metric_data = metrics_df[metrics_df["key"] == metric_name]

                    if not metric_data.empty:
                        comparison[metric_name][run_id] = {
                            "final": metric_data.iloc[-1]["value"],
                            "max": metric_data["value"].max(),
                            "min": metric_data["value"].min(),
                            "mean": metric_data["value"].mean()
                        }

            return comparison

        def _compute_significance(self, metrics_list: List[pd.DataFrame]) -> Dict:
            """Compute statistical significance (t-test)"""
            from scipy import stats

            if len(metrics_list) != 2:
                return {}  # Only for 2-way comparison

            results = {}

            # Find common metrics
            metrics1 = set(metrics_list[0]["key"].unique())
            metrics2 = set(metrics_list[1]["key"].unique())
            common_metrics = metrics1.intersection(metrics2)

            for metric_name in common_metrics:
                values1 = metrics_list[0][
                    metrics_list[0]["key"] == metric_name
                ]["value"].values
                values2 = metrics_list[1][
                    metrics_list[1]["key"] == metric_name
                ]["value"].values

                # T-test
                t_stat, p_value = stats.ttest_ind(values1, values2)

                results[metric_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }

            return results
    ```

    **Performance:**
    - Sequential queries: 100 runs × 2 queries × 100ms = 20 seconds
    - Parallel queries: max(100ms, 100ms) = 100ms
    - With sampling (1000 points): 10x faster than full data
    - **Total: <1 second for comparison**

    ---

    ## 4. Artifact Deduplication

    **Challenge:** Multiple experiments use same dataset → wasted storage (20% duplicate artifacts).

    **Solution: Content-Addressed Storage**

    ```python
    import hashlib
    import boto3
    from typing import Optional

    class ArtifactStore:
        def __init__(self, s3_client, postgres_client):
            self.s3 = s3_client
            self.db = postgres_client
            self.bucket = "ml-artifacts"

        def upload_artifact(
            self, run_id: str, local_path: str, artifact_path: str
        ) -> str:
            """Upload artifact with deduplication"""

            # 1. Compute content hash
            content_hash = self._compute_hash(local_path)

            # 2. Check if artifact with same hash exists
            existing_s3_key = self._find_by_hash(content_hash)

            if existing_s3_key:
                # Artifact already exists! Just create reference
                print(f"Deduplication: Artifact {artifact_path} already exists")
                s3_key = existing_s3_key
            else:
                # Upload new artifact
                s3_key = f"{run_id}/artifacts/{artifact_path}"

                with open(local_path, "rb") as f:
                    self.s3.upload_fileobj(f, self.bucket, s3_key)

                print(f"Uploaded new artifact: {s3_key}")

            # 3. Record in database
            file_size = os.path.getsize(local_path)
            self._record_artifact(
                run_id=run_id,
                artifact_path=artifact_path,
                s3_key=s3_key,
                content_hash=content_hash,
                size_bytes=file_size
            )

            return s3_key

        def _compute_hash(self, file_path: str) -> str:
            """Compute SHA256 hash of file"""
            sha256 = hashlib.sha256()

            with open(file_path, "rb") as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

            return sha256.hexdigest()

        def _find_by_hash(self, content_hash: str) -> Optional[str]:
            """Check if artifact with same hash exists"""
            query = """
                SELECT s3_key FROM artifacts
                WHERE content_hash = %s
                LIMIT 1
            """
            result = self.db.fetch_one(query, (content_hash,))

            if result:
                return result["s3_key"]
            return None

        def _record_artifact(
            self, run_id: str, artifact_path: str, s3_key: str,
            content_hash: str, size_bytes: int
        ):
            """Record artifact metadata"""
            query = """
                INSERT INTO artifacts (
                    artifact_id, run_id, path, s3_key, content_hash, size_bytes
                )
                VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)
            """
            self.db.execute(query, (
                run_id, artifact_path, s3_key, content_hash, size_bytes
            ))

        def download_artifact(self, run_id: str, artifact_path: str, dst_path: str):
            """Download artifact"""
            # Get S3 key from database
            query = """
                SELECT s3_key FROM artifacts
                WHERE run_id = %s AND path = %s
            """
            result = self.db.fetch_one(query, (run_id, artifact_path))

            if not result:
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")

            s3_key = result["s3_key"]

            # Download from S3
            with open(dst_path, "wb") as f:
                self.s3.download_fileobj(self.bucket, s3_key, f)
    ```

    **Storage Savings:**
    - Without deduplication: 10,000 runs × 1 GB dataset = 10 TB
    - With deduplication (dataset shared by 50% of runs): 5 TB
    - **50% storage savings for common artifacts**

    ---

    ## 5. Reproducibility: Run Versioning

    **Challenge:** "How do I reproduce this experiment from 6 months ago?"

    **Solution: Capture Environment Snapshot**

    ```python
    import os
    import sys
    import subprocess
    import json
    from datetime import datetime

    class RunContext:
        @staticmethod
        def capture_environment():
            """Capture full environment for reproducibility"""
            context = {
                "timestamp": datetime.utcnow().isoformat(),
                "python_version": sys.version,
                "platform": sys.platform,
                "hostname": os.uname().nodename,

                # Git info
                "git": RunContext._get_git_info(),

                # Python packages
                "packages": RunContext._get_pip_packages(),

                # Environment variables (filtered)
                "env_vars": RunContext._get_safe_env_vars(),

                # Command line
                "command": " ".join(sys.argv),
                "working_directory": os.getcwd()
            }

            return context

        @staticmethod
        def _get_git_info():
            """Capture Git commit, branch, diff"""
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()

                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()

                # Check for uncommitted changes
                diff = subprocess.check_output(
                    ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
                ).decode()

                has_changes = len(diff) > 0

                return {
                    "commit": commit,
                    "branch": branch,
                    "has_uncommitted_changes": has_changes,
                    "diff": diff if has_changes else None
                }
            except:
                return None

        @staticmethod
        def _get_pip_packages():
            """Get installed packages with versions"""
            result = subprocess.check_output([
                sys.executable, "-m", "pip", "freeze"
            ]).decode()

            packages = {}
            for line in result.split("\n"):
                if "==" in line:
                    name, version = line.split("==")
                    packages[name] = version

            return packages

        @staticmethod
        def _get_safe_env_vars():
            """Get environment variables (exclude secrets)"""
            excluded = {"AWS_SECRET", "API_KEY", "PASSWORD", "TOKEN"}

            safe_vars = {}
            for key, value in os.environ.items():
                if not any(secret in key.upper() for secret in excluded):
                    safe_vars[key] = value

            return safe_vars

    # Usage in SDK
    with mlflow.start_run() as run:
        # Automatically capture environment
        context = RunContext.capture_environment()

        # Log as artifact
        mlflow.log_dict(context, "environment.json")

        # Log git commit as tag
        if context["git"]:
            mlflow.set_tag("git.commit", context["git"]["commit"])
            mlflow.set_tag("git.branch", context["git"]["branch"])
    ```

    **Reproduction Script:**
    ```python
    def reproduce_run(run_id: str):
        """Reproduce an experiment run"""

        # 1. Download environment.json
        env_context = mlflow.artifacts.download_artifact(
            run_id, "environment.json"
        )

        # 2. Checkout git commit
        git_commit = env_context["git"]["commit"]
        subprocess.run(["git", "checkout", git_commit])

        # 3. Restore Python environment
        packages = env_context["packages"]
        with open("requirements.txt", "w") as f:
            for name, version in packages.items():
                f.write(f"{name}=={version}\n")

        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        # 4. Restore environment variables
        for key, value in env_context["env_vars"].items():
            os.environ[key] = value

        # 5. Re-run command
        command = env_context["command"]
        subprocess.run(command, shell=True)
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scaling Strategies

    ### 1. Handling Large Artifacts (10GB+ Models)

    **Challenge:** Uploading 10GB model takes 10+ minutes, blocking SDK.

    **Solution 1: Multipart Upload**

    ```python
    import boto3
    from concurrent.futures import ThreadPoolExecutor

    class MultipartUploader:
        def __init__(self, s3_client, bucket: str):
            self.s3 = s3_client
            self.bucket = bucket
            self.chunk_size = 100 * 1024 * 1024  # 100 MB chunks

        def upload_large_file(self, local_path: str, s3_key: str):
            """Upload large file in parallel chunks"""

            file_size = os.path.getsize(local_path)

            if file_size < self.chunk_size:
                # Small file, single upload
                with open(local_path, "rb") as f:
                    self.s3.upload_fileobj(f, self.bucket, s3_key)
                return

            # Initiate multipart upload
            upload_id = self.s3.create_multipart_upload(
                Bucket=self.bucket, Key=s3_key
            )["UploadId"]

            # Split into chunks
            num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

            def upload_chunk(chunk_num):
                with open(local_path, "rb") as f:
                    f.seek(chunk_num * self.chunk_size)
                    data = f.read(self.chunk_size)

                    response = self.s3.upload_part(
                        Bucket=self.bucket,
                        Key=s3_key,
                        PartNumber=chunk_num + 1,
                        UploadId=upload_id,
                        Body=data
                    )

                    return {
                        "PartNumber": chunk_num + 1,
                        "ETag": response["ETag"]
                    }

            # Upload chunks in parallel (10 threads)
            with ThreadPoolExecutor(max_workers=10) as executor:
                parts = list(executor.map(upload_chunk, range(num_chunks)))

            # Complete multipart upload
            self.s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )

            print(f"Uploaded {file_size / (1024**3):.2f} GB in {num_chunks} chunks")
    ```

    **Performance:**
    - Single-threaded: 10 GB @ 100 Mbps = 13 minutes
    - 10 parallel chunks: 10 GB @ 1 Gbps = 1.3 minutes
    - **10x faster!**

    ---

    **Solution 2: Background Upload**

    ```python
    class AsyncArtifactUploader:
        def __init__(self):
            self.upload_queue = queue.Queue()
            self.worker = threading.Thread(target=self._upload_worker, daemon=True)
            self.worker.start()

        def log_artifact_async(self, run_id: str, local_path: str, artifact_path: str):
            """Non-blocking artifact upload"""

            # Add to queue, return immediately
            self.upload_queue.put({
                "run_id": run_id,
                "local_path": local_path,
                "artifact_path": artifact_path
            })

            print(f"Queued artifact for upload: {artifact_path}")

        def _upload_worker(self):
            """Background thread that uploads artifacts"""
            while True:
                item = self.upload_queue.get()

                try:
                    # Upload to S3
                    uploader = MultipartUploader(s3_client, "ml-artifacts")
                    s3_key = f"{item['run_id']}/artifacts/{item['artifact_path']}"

                    uploader.upload_large_file(item["local_path"], s3_key)

                    print(f"Uploaded artifact: {item['artifact_path']}")

                except Exception as e:
                    print(f"Error uploading artifact: {e}")

    # Usage
    uploader = AsyncArtifactUploader()

    # Training loop continues immediately
    uploader.log_artifact_async(run_id, "model.pth", "model.pth")

    # Wait for uploads to complete at end
    uploader.upload_queue.join()
    ```

    ---

    ### 2. Real-Time Metric Streaming at Scale

    **Challenge:** 50 concurrent training runs × 10 metrics/sec = 500 metrics/sec peak.

    **Solution: Metric Aggregation Service**

    ```python
    from collections import defaultdict
    import asyncio

    class MetricAggregator:
        def __init__(self, influxdb_client, flush_interval=1.0):
            self.influxdb = influxdb_client
            self.flush_interval = flush_interval
            self.buffer = defaultdict(list)  # {run_id: [metrics]}
            self.lock = asyncio.Lock()

            # Start background flusher
            asyncio.create_task(self._periodic_flush())

        async def add_metrics(self, run_id: str, metrics: list):
            """Buffer metrics for batch write"""
            async with self.lock:
                self.buffer[run_id].extend(metrics)

        async def _periodic_flush(self):
            """Flush buffered metrics every N seconds"""
            while True:
                await asyncio.sleep(self.flush_interval)

                async with self.lock:
                    if not self.buffer:
                        continue

                    # Copy buffer and clear
                    to_flush = dict(self.buffer)
                    self.buffer.clear()

                # Write to InfluxDB (outside lock)
                await self._flush_to_influxdb(to_flush)

        async def _flush_to_influxdb(self, data):
            """Batch write to InfluxDB"""
            # Convert to InfluxDB line protocol
            lines = []

            for run_id, metrics in data.items():
                for metric in metrics:
                    line = (
                        f"metrics,run_id={run_id},key={metric['key']} "
                        f"value={metric['value']},step={metric['step']}i "
                        f"{metric['timestamp']}000000"
                    )
                    lines.append(line)

            # Batch write (InfluxDB supports 5000+ points per write)
            if lines:
                await self.influxdb.write(lines)
                print(f"Flushed {len(lines)} metrics to InfluxDB")
    ```

    **Performance:**
    - Individual writes: 500 metrics/sec × 10ms = 5 seconds of DB time/sec (unsustainable)
    - Batched writes (1000 metrics): 10ms per batch = 0.005 seconds of DB time/sec
    - **1000x more efficient!**

    ---

    ### 3. Multi-User Collaboration: Access Control

    **Challenge:** 500 teams, each needs isolated experiments with sharing.

    **Solution: Role-Based Access Control**

    ```sql
    -- Teams (multi-tenancy)
    CREATE TABLE teams (
        team_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Team memberships
    CREATE TABLE team_members (
        team_id VARCHAR(50) REFERENCES teams(team_id),
        user_id VARCHAR(100),
        role VARCHAR(20),  -- owner, admin, member, viewer
        joined_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (team_id, user_id)
    );

    -- Experiment ownership
    ALTER TABLE experiments ADD COLUMN team_id VARCHAR(50) REFERENCES teams(team_id);
    ALTER TABLE experiments ADD COLUMN visibility VARCHAR(20) DEFAULT 'team';  -- team, public, private

    CREATE INDEX idx_experiments_team ON experiments(team_id);

    -- Shared experiments
    CREATE TABLE experiment_shares (
        experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
        shared_with_team_id VARCHAR(50) REFERENCES teams(team_id),
        permission VARCHAR(20),  -- read, write
        PRIMARY KEY (experiment_id, shared_with_team_id)
    );
    ```

    **Access Control Logic:**
    ```python
    class AccessControl:
        def __init__(self, db_client):
            self.db = db_client

        def can_access_experiment(
            self, user_id: str, experiment_id: str, required_permission: str = "read"
        ) -> bool:
            """Check if user can access experiment"""

            # Get experiment info
            query = """
                SELECT team_id, visibility, created_by
                FROM experiments
                WHERE experiment_id = %s
            """
            exp = self.db.fetch_one(query, (experiment_id,))

            if not exp:
                return False

            # Public experiments: read-only access for all
            if exp["visibility"] == "public" and required_permission == "read":
                return True

            # Private experiments: only creator
            if exp["visibility"] == "private":
                return exp["created_by"] == user_id

            # Team experiments: check team membership
            query = """
                SELECT role FROM team_members
                WHERE team_id = %s AND user_id = %s
            """
            member = self.db.fetch_one(query, (exp["team_id"], user_id))

            if not member:
                # Check if experiment is shared with user's team
                query = """
                    SELECT permission FROM experiment_shares es
                    JOIN team_members tm ON es.shared_with_team_id = tm.team_id
                    WHERE es.experiment_id = %s AND tm.user_id = %s
                """
                share = self.db.fetch_one(query, (experiment_id, user_id))

                if not share:
                    return False

                # Check permission level
                if required_permission == "write" and share["permission"] != "write":
                    return False

                return True

            # Check role permissions
            role = member["role"]

            if required_permission == "write":
                return role in ("owner", "admin", "member")

            return True  # All roles can read
    ```

    ---

    ### 4. Search Performance Optimization

    **Challenge:** Searching 100K experiments with complex filters is slow.

    **Solution: Elasticsearch Indexing**

    ```python
    from elasticsearch import Elasticsearch

    class ExperimentSearchService:
        def __init__(self, es_client):
            self.es = es_client
            self.index = "experiments"

        def index_experiment(self, run_data):
            """Index run in Elasticsearch"""

            doc = {
                "run_id": run_data["run_id"],
                "experiment_id": run_data["experiment_id"],
                "run_name": run_data["run_name"],
                "user_id": run_data["user_id"],
                "status": run_data["status"],
                "start_time": run_data["start_time"],
                "end_time": run_data["end_time"],

                # Flatten params for searching
                "params": run_data["params"],

                # Latest metrics
                "metrics": run_data["metrics"],

                # Tags
                "tags": run_data["tags"]
            }

            self.es.index(index=self.index, id=run_data["run_id"], document=doc)

        def search_experiments(
            self, query: str = None, filters: dict = None,
            sort_by: str = None, limit: int = 10
        ):
            """Search experiments with complex filters"""

            # Build Elasticsearch query
            es_query = {"bool": {"must": [], "filter": []}}

            # Full-text search
            if query:
                es_query["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["run_name^2", "tags.description"]
                    }
                })

            # Filters
            if filters:
                # Metric filters (e.g., val_accuracy > 0.9)
                if "metrics" in filters:
                    for metric_key, condition in filters["metrics"].items():
                        operator, value = condition  # ("gt", 0.9)

                        es_query["bool"]["filter"].append({
                            "range": {
                                f"metrics.{metric_key}": {operator: value}
                            }
                        })

                # Param filters (e.g., learning_rate < 0.01)
                if "params" in filters:
                    for param_key, condition in filters["params"].items():
                        operator, value = condition

                        es_query["bool"]["filter"].append({
                            "range": {
                                f"params.{param_key}": {operator: value}
                            }
                        })

                # Tag filters
                if "tags" in filters:
                    for tag_key, tag_value in filters["tags"].items():
                        es_query["bool"]["filter"].append({
                            "term": {f"tags.{tag_key}": tag_value}
                        })

            # Sorting
            sort = []
            if sort_by:
                # e.g., "metrics.val_accuracy DESC"
                field, order = sort_by.rsplit(" ", 1)
                sort.append({field: {"order": order.lower()}})

            # Execute search
            response = self.es.search(
                index=self.index,
                query=es_query,
                sort=sort,
                size=limit
            )

            return [hit["_source"] for hit in response["hits"]["hits"]]
    ```

    **Performance:**
    - PostgreSQL LIKE query: 5-10 seconds for 100K rows
    - Elasticsearch: <200ms for same query
    - **25-50x faster!**

    ---

    ## Bottleneck Analysis

    ### Scenario 1: Dashboard Loading is Slow (>10 seconds)

    **Diagnosis:**
    ```python
    import time

    def load_dashboard(experiment_id):
        t0 = time.time()

        # Step 1: Load experiment metadata
        experiment = db.fetch_experiment(experiment_id)
        t1 = time.time()
        print(f"Load experiment: {t1-t0:.2f}s")

        # Step 2: Load all runs
        runs = db.fetch_runs(experiment_id)  # 1000 runs
        t2 = time.time()
        print(f"Load runs: {t2-t1:.2f}s")  # SLOW: 5 seconds

        # Step 3: Load latest metrics for each run
        for run in runs:
            metrics = db.fetch_latest_metrics(run["run_id"])
            run["metrics"] = metrics
        t3 = time.time()
        print(f"Load metrics: {t3-t2:.2f}s")  # SLOW: 8 seconds (1000 queries!)

        return runs
    ```

    **Solution 1: Batch Queries**
    ```python
    # Instead of 1000 individual queries, use JOIN
    query = """
        SELECT r.run_id, r.run_name, r.status,
               json_object_agg(m.key, m.value) as metrics
        FROM runs r
        LEFT JOIN latest_metrics m ON r.run_id = m.run_id
        WHERE r.experiment_id = %s
        GROUP BY r.run_id
    """
    runs = db.fetch_all(query, (experiment_id,))
    # Time: 0.5 seconds (16x faster!)
    ```

    **Solution 2: Pagination**
    ```python
    # Don't load 1000 runs at once
    def load_dashboard_paginated(experiment_id, page=1, page_size=50):
        offset = (page - 1) * page_size

        query = """
            SELECT r.run_id, r.run_name, r.status,
                   json_object_agg(m.key, m.value) as metrics
            FROM runs r
            LEFT JOIN latest_metrics m ON r.run_id = m.run_id
            WHERE r.experiment_id = %s
            GROUP BY r.run_id
            ORDER BY r.start_time DESC
            LIMIT %s OFFSET %s
        """
        runs = db.fetch_all(query, (experiment_id, page_size, offset))

        return runs
    # Time: 0.05 seconds (100x faster!)
    ```

    ---

    ### Scenario 2: Metric Logging Slows Down Training

    **Diagnosis:**
    ```python
    import time

    # Training loop with synchronous logging
    for epoch in range(100):
        for batch in dataloader:  # 1000 batches
            loss = train_step(batch)

            t0 = time.time()
            mlflow.log_metric("train_loss", loss, step=global_step)
            t1 = time.time()

            if t1 - t0 > 0.01:  # Logging takes >10ms
                print(f"WARNING: Metric logging took {(t1-t0)*1000:.1f}ms")
    ```

    **Solution:** Use async SDK (already implemented in Deep Dive section).

    ---

    ### Scenario 3: Artifact Storage Costs are Too High

    **Current Cost:**
    ```
    120 TB/year × $0.023/GB/month = $2,760/month = $33,120/year
    ```

    **Optimization 1: Lifecycle Policies**
    ```
    - First 90 days: S3 Standard ($0.023/GB/month)
    - 90-365 days: S3 IA ($0.0125/GB/month)
    - 365+ days: S3 Glacier ($0.004/GB/month)

    Assuming uniform age distribution:
    - 25% in Standard (30 TB): 30 TB × $0.023 = $690/month
    - 25% in IA (30 TB): 30 TB × $0.0125 = $375/month
    - 50% in Glacier (60 TB): 60 TB × $0.004 = $240/month

    Total: $1,305/month = $15,660/year
    Savings: $17,460/year (53% reduction!)
    ```

    **Optimization 2: Compression**
    ```python
    import gzip

    def upload_compressed_artifact(local_path, s3_key):
        # Compress before upload
        with open(local_path, "rb") as f_in:
            with gzip.open(local_path + ".gz", "wb") as f_out:
                f_out.writelines(f_in)

        # Upload compressed
        s3.upload_file(local_path + ".gz", bucket, s3_key + ".gz")

        # Cleanup
        os.remove(local_path + ".gz")
    ```

    **Compression ratios:**
    - Model checkpoints: 1.5-2x (PyTorch/TF already compressed)
    - Logs: 10x
    - Datasets: 3-5x
    - Average: 2x

    **Final cost with compression:**
    ```
    $15,660/year ÷ 2 = $7,830/year
    Total savings: $25,290/year (76% reduction!)
    ```

=== "💡 Step 5: Additional Considerations"

    ## Security & Compliance

    ### 1. Experiment Access Control

    **Multi-tenancy isolation:**
    ```python
    # Enforce team_id in all queries
    def get_user_experiments(user_id: str):
        query = """
            SELECT e.* FROM experiments e
            JOIN team_members tm ON e.team_id = tm.team_id
            WHERE tm.user_id = %s
        """
        return db.fetch_all(query, (user_id,))
    ```

    ---

    ### 2. Sensitive Data Protection

    ```python
    # Redact sensitive params before logging
    SENSITIVE_KEYS = {"api_key", "password", "token", "secret"}

    def log_params_safe(params: dict):
        safe_params = {}

        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
                safe_params[key] = "[REDACTED]"
            else:
                safe_params[key] = value

        mlflow.log_params(safe_params)
    ```

    ---

    ### 3. Audit Logging

    ```sql
    CREATE TABLE audit_log (
        log_id BIGSERIAL PRIMARY KEY,
        user_id VARCHAR(100),
        action VARCHAR(50),  -- create_run, delete_experiment, etc.
        resource_type VARCHAR(50),
        resource_id VARCHAR(50),
        ip_address INET,
        user_agent TEXT,
        timestamp TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX idx_audit_user ON audit_log(user_id);
    CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
    ```

    ---

    ## Monitoring & Alerting

    **Key Metrics:**
    ```python
    # 1. API Latency
    histogram("mlflow_api_latency_ms", buckets=[10, 50, 100, 500, 1000])

    # 2. Metrics Logging Rate
    gauge("mlflow_metrics_logged_per_sec")

    # 3. Storage Usage
    gauge("mlflow_artifacts_storage_tb")

    # 4. Active Runs
    gauge("mlflow_active_runs")

    # 5. Database Connection Pool
    gauge("mlflow_db_connections_active")
    gauge("mlflow_db_connections_idle")

    # 6. Error Rate
    counter("mlflow_api_errors", labels=["endpoint", "error_type"])
    ```

    **Alerts:**
    ```yaml
    - alert: HighAPILatency
      expr: histogram_quantile(0.95, mlflow_api_latency_ms) > 1000
      for: 5m
      annotations:
        summary: "MLflow API latency high ({{ $value }}ms)"

    - alert: StorageQuotaExceeded
      expr: mlflow_artifacts_storage_tb > 150
      for: 1h
      annotations:
        summary: "Artifact storage exceeded 150TB threshold"

    - alert: DatabaseConnectionPoolExhausted
      expr: mlflow_db_connections_active > 90
      for: 5m
      annotations:
        summary: "Database connection pool nearly exhausted"
    ```

    ---

    ## Cost Optimization

    ### Monthly Cost Breakdown

    ```
    Infrastructure:
    - API servers (5 × c5.2xlarge): 5 × $0.34/hr × 730 hrs = $1,241
    - PostgreSQL (db.r5.2xlarge): $0.48/hr × 730 hrs = $350
    - InfluxDB (c5.2xlarge): $0.34/hr × 730 hrs = $248
    - Redis (cache.r5.large): $0.126/hr × 730 hrs = $92
    - Elasticsearch (3 × r5.xlarge): 3 × $0.252/hr × 730 hrs = $552

    Storage:
    - S3 artifacts (144 TB, lifecycle): $1,305
    - PostgreSQL storage (200 GB): $20
    - InfluxDB storage (100 GB): $10

    Data transfer:
    - Egress (5 TB/month): 5,000 GB × $0.09 = $450

    Total: $4,268/month ≈ $51K/year

    Cost per experiment: $51K / (120K experiments/year) = $0.43/experiment
    ```

    ---

    ## Disaster Recovery

    **Backup Strategy:**
    ```bash
    # Daily PostgreSQL backup
    pg_dump mlflow_db | gzip > backup-$(date +%Y%m%d).sql.gz
    aws s3 cp backup-$(date +%Y%m%d).sql.gz s3://mlflow-backups/

    # S3 versioning enabled (automatic)
    aws s3api put-bucket-versioning \
      --bucket ml-artifacts \
      --versioning-configuration Status=Enabled

    # InfluxDB snapshot
    influxd backup -portable /tmp/influxdb-backup
    tar -czf influxdb-backup.tar.gz /tmp/influxdb-backup
    aws s3 cp influxdb-backup.tar.gz s3://mlflow-backups/
    ```

    **Recovery Time:**
    - Metadata DB: <1 hour (restore from backup)
    - Time-series DB: <2 hours (restore + reindex)
    - Artifacts: Instant (S3 is durable)

=== "🎯 Step 6: Interview Tips"

    ## How to Approach This Problem

    ### 1. Clarify Requirements (5 minutes)

    **Key Questions:**
    - Scale? (number of experiments, users, artifacts per month)
    - Latency requirements? (how fast should logging be?)
    - Storage constraints? (artifact size limits, retention period)
    - Collaboration features? (team sharing, access control)
    - Integration needs? (TensorFlow, PyTorch, Jupyter)

    ---

    ### 2. Start with Use Case (2 minutes)

    "Let's design an experiment tracking system for a data science team. Users need to:
    1. Log metrics during training (loss, accuracy) without slowing down training
    2. Store large model checkpoints (1-10 GB)
    3. Compare experiments side-by-side to find best hyperparameters
    4. Reproduce past experiments from 6 months ago"

    ---

    ### 3. High-Level Architecture (10 minutes)

    Draw architecture diagram with:
    - Python SDK (async logging)
    - API Gateway + Tracking Server
    - PostgreSQL (metadata), InfluxDB (metrics), S3 (artifacts)
    - WebSocket (real-time updates)
    - Web UI (React dashboard)

    Explain the rationale for each component.

    ---

    ### 4. Deep Dive (20 minutes)

    Focus on 2-3 areas based on interviewer interest:
    - **Async SDK:** Batch metric logging to avoid blocking training
    - **Artifact deduplication:** Content-addressed storage
    - **Comparison engine:** Parallel queries with statistical tests
    - **Search optimization:** Elasticsearch indexing

    ---

    ### 5. Scale & Optimize (10 minutes)

    Discuss:
    - Large artifact uploads (multipart, background)
    - Real-time metric streaming (aggregation, WebSocket)
    - Search performance (Elasticsearch vs. PostgreSQL)
    - Cost optimization (S3 lifecycle, compression)

    ---

    ## Common Follow-Up Questions

    ### Q1: How do you prevent metric logging from slowing down training?

    **Answer:**
    "We use asynchronous batching in the SDK. Metrics are queued in-memory and a background thread uploads batches of 100 metrics every 10 seconds. This reduces network calls by 100x and ensures logging takes <0.1ms, not blocking the training loop."

    ---

    ### Q2: How do you handle large model checkpoints (10+ GB)?

    **Answer:**
    "We use S3 multipart upload, splitting the file into 100MB chunks and uploading 10 chunks in parallel. This achieves 1 Gbps upload speed. We also use background uploads so training can continue while artifacts upload asynchronously."

    ---

    ### Q3: How do you compare 100 experiments efficiently?

    **Answer:**
    "We fetch metadata and metrics in parallel using asyncio (100 queries → 100ms). For metrics, we downsample to 1000 points per run to reduce data transfer. We also compute statistical significance (t-tests) to identify meaningful differences."

    ---

    ### Q4: How do you ensure reproducibility?

    **Answer:**
    "We capture a full environment snapshot: Git commit, Python packages, environment variables, and command line. This is stored as environment.json artifact. To reproduce, we checkout the git commit, restore packages, and re-run the command."

    ---

    ### Q5: How do you handle multi-tenancy and access control?

    **Answer:**
    "Each experiment belongs to a team. Users have roles (owner, admin, member, viewer) within teams. We enforce team_id in all queries. Experiments can be shared with other teams with read/write permissions. Public experiments are read-only for all users."

    ---

    ## Red Flags to Avoid

    1. **Don't** store metrics in PostgreSQL (use time-series DB like InfluxDB)
    2. **Don't** store artifacts in database (use S3)
    3. **Don't** use synchronous logging (blocks training)
    4. **Don't** forget about cost optimization (S3 lifecycle, compression)
    5. **Don't** ignore reproducibility (capture git commit, packages)

    ---

    ## Bonus Points

    1. Mention **real-world systems:** MLflow, Weights & Biases, Neptune
    2. Discuss **auto-logging:** Automatic capture from TensorFlow/PyTorch
    3. Talk about **hyperparameter optimization:** Integration with Optuna
    4. Consider **model registry:** Transition to model serving
    5. Mention **experiment templates:** Reusable configurations

=== "📚 References & Resources"

    ## Real-World Implementations

    ### MLflow (Databricks)
    - **Architecture:** Tracking server, artifact store (S3), model registry
    - **Scale:** Open-source, used by 1000+ companies
    - **Key Innovation:** Unified API for TensorFlow, PyTorch, scikit-learn
    - **Docs:** [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

    ---

    ### Weights & Biases
    - **Architecture:** SaaS platform, real-time streaming, collaboration
    - **Scale:** 500K+ users, millions of experiments
    - **Key Innovation:** Live dashboards, experiment reports, sweep (hyperparameter optimization)
    - **Website:** [wandb.ai](https://wandb.ai/)

    ---

    ### Neptune.ai
    - **Architecture:** Cloud-based, metadata store, experiment comparison
    - **Scale:** Enterprise customers, high-security requirements
    - **Key Innovation:** Advanced search, lineage tracking, model registry integration
    - **Website:** [neptune.ai](https://neptune.ai/)

    ---

    ### Comet ML
    - **Architecture:** Hybrid (cloud + on-prem), experiment tracking, model monitoring
    - **Scale:** Fortune 500 companies
    - **Key Innovation:** Code diff tracking, model production monitoring
    - **Website:** [comet.com](https://www.comet.com/)

    ---

    ## Open Source Tools

    ### MLflow
    ```python
    import mlflow

    mlflow.set_tracking_uri("http://mlflow-server:5000")

    with mlflow.start_run():
        mlflow.log_params({"learning_rate": 0.001, "batch_size": 32})

        for epoch in range(100):
            loss = train()
            mlflow.log_metric("loss", loss, step=epoch)

        mlflow.log_artifact("model.pth")
    ```

    ---

    ### TensorBoard
    ```python
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('runs/experiment_1')

    for epoch in range(100):
        loss = train()
        writer.add_scalar('Loss/train', loss, epoch)

    writer.close()
    ```

    ---

    ### Sacred
    ```python
    from sacred import Experiment

    ex = Experiment('my_experiment')

    @ex.config
    def cfg():
        learning_rate = 0.001
        batch_size = 32

    @ex.automain
    def main(learning_rate, batch_size):
        for epoch in range(100):
            loss = train(learning_rate, batch_size)
            ex.log_scalar("loss", loss, epoch)
    ```

    ---

    ## Related System Design Problems

    1. **Feature Store** - Provides features for training (upstream)
    2. **Model Serving Platform** - Deploys models to production (downstream)
    3. **Distributed Training System** - Generates experiments to track
    4. **Data Versioning** - DVC, Pachyderm (similar tracking concepts)
    5. **Observability Platform** - Similar metrics storage (Prometheus, Grafana)

---

## Summary

An **ML Experiment Tracking System** enables data scientists to log, compare, and reproduce experiments at scale:

**Key Components:**
- **Python SDK:** Async batching for non-blocking metric logging
- **Metadata DB (PostgreSQL):** Experiments, runs, parameters, tags
- **Time-Series DB (InfluxDB):** High-throughput metrics storage
- **Artifact Store (S3):** Scalable, durable model/dataset storage
- **WebSocket Server:** Real-time metric streaming to dashboard
- **Search Service (Elasticsearch):** Fast experiment filtering
- **Web UI (React):** Interactive comparison and visualization

**Core Challenges:**
- Async logging to avoid slowing down training loops
- Handling large artifacts (10+ GB model checkpoints)
- Real-time metric streaming for live dashboards
- Efficient experiment comparison across 100+ runs
- Reproducibility (capture full environment snapshot)
- Search performance for 100K+ experiments

**Architecture Decisions:**
- Separate storage: Metadata (PostgreSQL), metrics (InfluxDB), artifacts (S3)
- Async SDK with batching (100x faster than synchronous)
- Content-addressed storage for artifact deduplication (20% savings)
- S3 lifecycle policies for cost optimization (76% reduction)
- WebSocket for real-time updates (99% bandwidth savings vs. polling)

This is a **medium difficulty** problem that combines distributed systems, database design, and ML infrastructure knowledge. Focus on async logging, storage separation, and cost optimization during interviews.
