# Design ML Model Serving Platform (TensorFlow Serving/SageMaker)

A production-grade machine learning model serving platform that enables deployment, serving, and management of thousands of ML models at scale with low-latency inference, autoscaling, A/B testing, canary deployments, and comprehensive monitoring.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10,000+ models, 100M predictions/day, 50K QPS peak, <100ms p95 latency |
| **Key Challenges** | Low-latency inference, GPU optimization, dynamic batching, model versioning, A/B testing, multi-tenancy |
| **Core Concepts** | Docker/Kubernetes, TensorFlow Serving, TorchServe, dynamic batching, canary deployments, model registry |
| **Companies** | AWS SageMaker, Google Vertex AI, Azure ML, Uber Michelangelo, Netflix, Spotify, Airbnb |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Model Deployment** | Deploy models via Docker/Kubernetes with versioning | P0 (Must have) |
    | **Real-time Inference** | Synchronous prediction API with <100ms latency | P0 (Must have) |
    | **Batch Inference** | Asynchronous batch predictions for large datasets | P0 (Must have) |
    | **Model Versioning** | Multiple versions per model with traffic routing | P0 (Must have) |
    | **A/B Testing** | Split traffic between model versions for testing | P0 (Must have) |
    | **Canary Deployments** | Gradual rollout with automatic rollback | P0 (Must have) |
    | **Autoscaling** | Scale based on request rate and latency | P0 (Must have) |
    | **Model Registry** | Central repository for model artifacts and metadata | P0 (Must have) |
    | **Monitoring & Alerting** | Track latency, throughput, accuracy, drift | P1 (Should have) |
    | **GPU Optimization** | Efficient GPU utilization with batching | P1 (Should have) |
    | **Multi-framework Support** | TensorFlow, PyTorch, ONNX, Scikit-learn | P1 (Should have) |
    | **Feature Store Integration** | Real-time feature retrieval for inference | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training infrastructure (focus on serving only)
    - Data labeling and annotation
    - Feature engineering pipelines
    - AutoML capabilities
    - Model debugging/profiling tools
    - Edge device deployment (focus on cloud serving)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Real-time)** | < 100ms p95, < 50ms p50 | Critical for user-facing applications |
    | **Latency (Batch)** | < 1 hour for 1M predictions | Cost-effective for non-time-sensitive tasks |
    | **Throughput** | > 50,000 predictions/sec peak | Support high-traffic applications |
    | **Availability** | 99.95% uptime (4.38 hours/year) | Production SLA requirements |
    | **GPU Utilization** | > 80% average utilization | Expensive resources, maximize efficiency |
    | **Model Load Time** | < 30 seconds | Fast deployment and rollback |
    | **Scalability** | Support 10,000+ models | Multi-tenant platform |
    | **Consistency** | Strong consistency for model versions | No prediction inconsistencies |
    | **Cost Efficiency** | < $0.01 per 1,000 predictions | Competitive pricing |
    | **Security** | Multi-tenant isolation, encryption at rest/transit | Protect sensitive models and data |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Scale:
    - Total models deployed: 10,000 models
    - Active models (receiving traffic): 2,000 models
    - Teams/tenants: 500 teams

    Real-time inference:
    - Daily predictions: 100M predictions/day
    - Average QPS: 100M / 86,400 = ~1,160 req/sec
    - Peak QPS: 5x average = ~5,800 req/sec (hot models)
    - Per-model QPS: 5,800 / 2,000 = ~3 req/sec average
    - Hot models (top 10%): 200 models handling 80% traffic = ~23 req/sec each

    Batch inference:
    - Batch jobs per day: 500 jobs
    - Average batch size: 100,000 records
    - Daily batch predictions: 50M predictions/day
    - Processing time: 1-6 hours per job

    Model deployment:
    - New deployments per day: 200 deployments
    - Model updates: 50 updates/day
    - Rollbacks: 5 rollbacks/day

    A/B testing:
    - Active A/B tests: 100 concurrent tests
    - Traffic split: 50/50 or 90/10
    - Test duration: 7 days average

    Request size:
    - Average input size: 5 KB (features + metadata)
    - Average output size: 1 KB (predictions + scores)
    - Batch request: 10 MB (1,000 records)

    Read/Write ratio: 1000:1 (inference >> model updates)
    ```

    ### Storage Estimates

    ```
    Model registry:
    - Average model size: 500 MB (TensorFlow, PyTorch models)
    - Versions per model: 5 versions (keep recent versions)
    - Total models: 10,000 × 5 × 500 MB = 25 TB
    - With compression (50%): 12.5 TB

    Model metadata:
    - Metadata per model version: 10 KB (framework, input/output schema, metrics)
    - 10,000 models × 5 versions × 10 KB = 500 MB

    Inference logs:
    - Log per prediction: 500 bytes (input hash, output, latency, timestamp)
    - Daily logs: 100M × 500 bytes = 50 GB/day
    - 30 days retention: 50 GB × 30 = 1.5 TB
    - 1 year (compressed): 1.5 TB × 12 × 0.3 = 5.4 TB

    Metrics and monitoring:
    - Time-series metrics: 1 KB per model per minute
    - 10,000 models × 1 KB × 60 × 24 = 14.4 GB/day
    - 90 days retention: 14.4 GB × 90 = 1.3 TB

    Feature store cache:
    - Features per prediction: 2 KB
    - Cache for 10M hot predictions: 10M × 2 KB = 20 GB

    Deployment artifacts:
    - Docker images: 2 GB per model (base image + dependencies)
    - Active models: 2,000 × 2 GB = 4 TB
    - With layer caching: ~1 TB

    Total: 12.5 TB (models) + 5.4 TB (logs) + 1.3 TB (metrics) + 20 GB (cache) + 1 TB (images) ≈ 20 TB
    ```

    ### Compute Estimates

    ```
    Real-time inference (CPU-based models):
    - QPS per model instance: 100 req/sec (optimized)
    - Peak QPS: 5,800 req/sec
    - Instances needed: 5,800 / 100 = 58 instances
    - With 2x buffer: 116 instances
    - Instance type: 8 vCPU, 32 GB RAM
    - Total: 928 vCPUs, 3.7 TB RAM

    Real-time inference (GPU-based models):
    - GPU models: 20% of active models = 400 models
    - QPS per GPU instance: 500 req/sec (with batching)
    - Peak GPU QPS: 5,800 × 0.2 = 1,160 req/sec
    - GPUs needed: 1,160 / 500 = 3 GPUs
    - With redundancy: 6 GPUs (T4 or V100)
    - Cost: 6 × $0.95/hour = $5.70/hour = $4,104/month

    Batch inference:
    - Batch jobs: 500 jobs/day
    - Average job duration: 2 hours
    - Concurrent jobs: 500 × 2 / 24 = ~42 jobs
    - Instances per job: 4 instances (parallelization)
    - Total: 168 instances (spot/preemptible)

    Control plane:
    - API servers: 10 instances (8 vCPU each)
    - Model registry: 5 instances (4 vCPU each)
    - Monitoring: 3 instances (16 vCPU each)
    - Total: 148 vCPUs

    Kubernetes overhead:
    - Master nodes: 3 nodes × 8 vCPU = 24 vCPUs
    - Monitoring (Prometheus): 16 vCPUs
    - Logging (Elasticsearch): 32 vCPUs

    Total: 928 + 148 + 72 = 1,148 vCPUs + 6 GPUs
    ```

    ### Bandwidth Estimates

    ```
    Real-time inference ingress:
    - 1,160 req/sec × 5 KB = 5.8 MB/sec ≈ 46 Mbps

    Real-time inference egress:
    - 1,160 req/sec × 1 KB = 1.16 MB/sec ≈ 9 Mbps

    Batch inference:
    - Input data transfer: 50M predictions × 5 KB / 86,400 = 2.9 GB/sec (peak)
    - Output: 50M × 1 KB / 86,400 = 578 MB/sec (peak)

    Model deployment:
    - 200 deployments/day × 500 MB = 100 GB/day
    - Average: 100 GB / 86,400 = 1.16 MB/sec ≈ 9 Mbps
    - Peak: 10x = 90 Mbps

    Monitoring/metrics:
    - 10,000 models × 1 KB/min = 10 MB/min ≈ 170 KB/sec ≈ 1.4 Mbps

    Logs:
    - 1,160 req/sec × 500 bytes = 580 KB/sec ≈ 4.6 Mbps

    Total ingress: 46 + 90 = 136 Mbps (+ 2.9 GB/sec batch peak)
    Total egress: 9 + 4.6 + 1.4 = 15 Mbps (+ 578 MB/sec batch peak)
    ```

    ### Memory Estimates (Caching)

    ```
    Model cache (in-memory):
    - Hot models: 200 models × 500 MB = 100 GB
    - With model server optimization: 50 GB

    Feature cache:
    - 10M hot predictions × 2 KB = 20 GB

    Prediction cache:
    - Cache deterministic predictions: 1M cached results
    - 1M × 1 KB = 1 GB

    Metadata cache:
    - Model schemas, configs: 10,000 models × 100 KB = 1 GB

    Metrics buffer:
    - In-memory metrics: 5 GB

    Total cache: 50 GB + 20 GB + 1 GB + 1 GB + 5 GB = 77 GB
    ```

    ---

    ## Key Assumptions

    1. Average model size is 500 MB (deep learning models)
    2. 80% of traffic goes to 20% of models (hot models)
    3. 80% of models use CPU inference, 20% require GPU
    4. Batch inference can tolerate 1-6 hour latency
    5. Real-time inference requires <100ms p95 latency
    6. Models are updated weekly on average
    7. A/B tests run for 7 days with 50/50 or 90/10 splits
    8. GPU utilization can reach 80% with dynamic batching
    9. Docker images share base layers (reduces storage)
    10. Feature store provides <10ms feature retrieval

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Containerization:** Package models in Docker for consistency and portability
    2. **Orchestration:** Kubernetes for deployment, scaling, and load balancing
    3. **Multi-tenancy:** Isolated namespaces per team with resource quotas
    4. **Observability:** Comprehensive monitoring, logging, and tracing
    5. **Progressive rollout:** Canary deployments with automatic rollback on errors
    6. **Optimization:** Dynamic batching, GPU sharing, model caching
    7. **Flexibility:** Support multiple frameworks (TensorFlow, PyTorch, ONNX)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client[API Clients]
            BatchClient[Batch Job Clients]
        end

        subgraph "API Gateway Layer"
            Gateway[API Gateway<br/>Rate Limiting, Auth]
            LB[Load Balancer]
        end

        subgraph "Serving Layer"
            subgraph "Model Inference Pods"
                TFS1[TensorFlow Serving<br/>Model A v1]
                TFS2[TensorFlow Serving<br/>Model A v2]
                TS1[TorchServe<br/>Model B]
                Custom[Custom Server<br/>Model C]
            end

            Batcher[Dynamic Batch<br/>Aggregator]
            Router[Traffic Router<br/>A/B Testing]
        end

        subgraph "Control Plane"
            API[Model API<br/>FastAPI/gRPC]
            Deployer[Model Deployer<br/>Kubernetes Operator]
            Registry[Model Registry<br/>MLflow/BentoML]
            Orchestrator[Workflow<br/>Orchestrator]
        end

        subgraph "Storage Layer"
            S3[Object Storage<br/>S3/GCS<br/>Model Artifacts]
            MetaDB[(Metadata DB<br/>PostgreSQL<br/>Model Versions)]
            MetricDB[(Time-Series DB<br/>Prometheus<br/>Metrics)]
            LogDB[(Log Storage<br/>Elasticsearch<br/>Predictions)]
        end

        subgraph "Supporting Services"
            FeatureStore[Feature Store<br/>Real-time Features]
            Monitor[Monitoring<br/>Grafana + Alerting]
            Tracing[Distributed Tracing<br/>Jaeger]
            Cache[Prediction Cache<br/>Redis]
        end

        Client -->|REST/gRPC| Gateway
        BatchClient -->|Batch API| Gateway
        Gateway --> Router
        Router -->|Route by version| TFS1
        Router -->|Route by version| TFS2
        Router --> TS1
        Router --> Custom
        Router --> Batcher
        Batcher --> TFS1

        Client -->|Deploy/Manage| API
        API --> Deployer
        API --> Registry
        Deployer -->|Create/Update Pods| TFS1
        Deployer -->|Create/Update Pods| TFS2
        Registry -->|Pull Models| S3

        BatchClient --> Orchestrator
        Orchestrator --> TFS1
        Orchestrator --> S3

        TFS1 -.->|Query Features| FeatureStore
        TFS2 -.->|Query Features| FeatureStore
        TFS1 -.->|Cache Predictions| Cache

        TFS1 -->|Metrics| MetricDB
        TFS2 -->|Metrics| MetricDB
        TFS1 -->|Logs| LogDB
        Router -->|Routing Metrics| MetricDB

        Monitor -->|Query| MetricDB
        Monitor -->|Query| LogDB
        Tracing -.->|Trace| TFS1

        Registry --> MetaDB
        Registry --> S3
        API --> MetaDB
    ```

    ---

    ## Component Breakdown

    ### 1. API Gateway
    - **Load balancing** across model serving instances
    - **Authentication/Authorization** (API keys, JWT)
    - **Rate limiting** per client/tenant
    - **Request routing** to appropriate model versions
    - **Protocol translation** (REST to gRPC)

    ### 2. Traffic Router & A/B Testing
    - **Version routing** based on traffic split rules
    - **Canary deployment** with gradual rollout (1% → 10% → 50% → 100%)
    - **Shadow traffic** for testing without affecting production
    - **Sticky routing** for consistent user experience
    - **Automatic rollback** on error rate spikes

    ### 3. Model Serving Runtime
    - **TensorFlow Serving:** Optimized for TF models, supports gRPC/REST
    - **TorchServe:** PyTorch model serving with custom handlers
    - **ONNX Runtime:** Cross-framework inference
    - **Custom servers:** FastAPI/Flask for scikit-learn, XGBoost
    - **Model warming:** Pre-load models at startup
    - **Health checks:** Liveness and readiness probes

    ### 4. Dynamic Batching
    - **Request aggregation:** Combine multiple requests into batches
    - **Adaptive batching:** Adjust batch size based on latency targets
    - **Timeout-based flushing:** Process batch after max wait time (10-50ms)
    - **GPU optimization:** Maximize GPU utilization with larger batches
    - **Per-model configuration:** Different batch sizes per model type

    ### 5. Model Registry
    - **Centralized storage** for model artifacts (S3/GCS)
    - **Version management** with semantic versioning
    - **Metadata tracking:** Framework, input/output schema, metrics, lineage
    - **Access control:** Role-based access to models
    - **Model lineage:** Track training job, dataset, hyperparameters
    - **Search & discovery:** Find models by tags, metrics

    ### 6. Model Deployer (Kubernetes Operator)
    - **Automated deployment** from registry to Kubernetes
    - **Rolling updates** with zero downtime
    - **Resource allocation:** CPU/GPU/memory quotas
    - **Namespace isolation:** Multi-tenant separation
    - **Custom Resource Definitions (CRD):** ModelDeployment, ABTest resources
    - **Helm charts/Operators:** Standardized deployment templates

    ### 7. Autoscaling
    - **Horizontal Pod Autoscaler (HPA):** Scale based on CPU/memory/custom metrics
    - **Vertical Pod Autoscaler (VPA):** Adjust resource requests/limits
    - **Custom metrics:** Scale on request queue depth, latency, GPU utilization
    - **Predictive scaling:** ML-based autoscaling for traffic patterns
    - **Scale-to-zero:** Remove idle model instances to save cost

    ### 8. Monitoring & Observability
    - **System metrics:** CPU, memory, GPU utilization, network I/O
    - **Application metrics:** Request rate, latency (p50/p95/p99), error rate
    - **Model metrics:** Prediction distribution, confidence scores, drift detection
    - **Business metrics:** Revenue impact, user engagement
    - **Alerting:** PagerDuty, Slack notifications on SLO violations
    - **Dashboards:** Grafana for real-time visualization

    ### 9. Feature Store Integration
    - **Real-time feature retrieval** (<10ms latency)
    - **Feature caching** for hot features
    - **Feature versioning** aligned with model versions
    - **Point-in-time correctness** for training/serving consistency

    ---

    ## Data Flow

    ### Real-time Inference Flow

    ```
    1. Client sends prediction request to API Gateway
    2. Gateway authenticates, rate limits, routes to Traffic Router
    3. Router selects model version based on A/B test rules
    4. Request queued in Dynamic Batcher (if enabled)
    5. Batcher accumulates requests for 10-50ms or until batch size reached
    6. Batch sent to Model Serving Pod (TensorFlow Serving, TorchServe)
    7. Model server loads model from cache (or pulls from registry if cold start)
    8. (Optional) Server queries Feature Store for additional features
    9. Model performs inference (CPU/GPU)
    10. Response sent back through router to client
    11. Metrics logged to Prometheus, predictions logged to Elasticsearch
    12. Monitor checks latency, error rate, triggers alerts if needed
    ```

    ### Batch Inference Flow

    ```
    1. Client submits batch job via Batch API (input data location, model version)
    2. Workflow Orchestrator (Airflow/Argo) creates batch job
    3. Job reads input data from S3/GCS in chunks
    4. Each chunk processed by model serving instances in parallel
    5. Results written back to S3/GCS
    6. Job completion notification sent to client
    7. Metrics aggregated and stored in monitoring system
    ```

    ### Model Deployment Flow

    ```
    1. Data scientist trains model, saves to Model Registry
    2. Model metadata stored in PostgreSQL (version, schema, metrics)
    3. Model artifact uploaded to S3/GCS
    4. User triggers deployment via Model API
    5. Model Deployer validates model (schema, dependencies)
    6. Deployer creates Kubernetes Deployment/Service
    7. Docker image built (base image + model) or pulled from cache
    8. Pods started, model loaded from registry
    9. Health checks pass, traffic routed to new version
    10. Old version kept for rollback (gradual deprecation)
    ```

    ### A/B Testing Flow

    ```
    1. User creates A/B test via API (model v1 vs v2, 90/10 split)
    2. Traffic Router updated with routing rules
    3. 90% of requests routed to v1, 10% to v2
    4. Both versions log predictions, latency, errors
    5. Monitoring system compares metrics (latency, accuracy, business metrics)
    6. After 7 days, statistical significance test determines winner
    7. Winning version promoted to 100% traffic
    8. Losing version deprecated or kept for rollback
    ```

=== "📡 Step 3: API Design"

    ## REST API Endpoints

    ### Prediction API (Client-facing)

    ```http
    POST /v1/models/{model_name}/predict
    ```

    **Request:**
    ```json
    {
      "version": "v2.1.0",           // Optional, defaults to latest
      "instances": [                  // Single or batch predictions
        {
          "feature1": 0.5,
          "feature2": "category_a",
          "feature3": [1, 2, 3]
        }
      ],
      "parameters": {                 // Optional inference parameters
        "temperature": 0.8,
        "top_k": 5
      },
      "timeout_ms": 1000,
      "explain": false                // Return feature importance
    }
    ```

    **Response:**
    ```json
    {
      "predictions": [
        {
          "output": 0.85,             // Prediction value
          "confidence": 0.92,         // Confidence score
          "class": "positive",
          "probabilities": {          // Class probabilities
            "positive": 0.85,
            "negative": 0.15
          },
          "explanation": {            // If explain=true
            "feature1": 0.3,
            "feature2": 0.5,
            "feature3": 0.2
          }
        }
      ],
      "model_name": "sentiment_model",
      "model_version": "v2.1.0",
      "latency_ms": 45,
      "request_id": "req_abc123"
    }
    ```

    ---

    ### Batch Inference API

    ```http
    POST /v1/batch/predict
    ```

    **Request:**
    ```json
    {
      "model_name": "fraud_detection",
      "model_version": "v3.0.0",
      "input_uri": "s3://bucket/input/transactions.parquet",
      "output_uri": "s3://bucket/output/predictions.parquet",
      "batch_size": 1000,            // Records per batch
      "max_parallelism": 10,         // Parallel workers
      "timeout_minutes": 60,
      "notify_on_completion": "user@example.com"
    }
    ```

    **Response:**
    ```json
    {
      "job_id": "batch_job_xyz789",
      "status": "pending",            // pending, running, completed, failed
      "created_at": "2026-02-05T10:00:00Z",
      "estimated_completion_time": "2026-02-05T12:00:00Z"
    }
    ```

    **Check batch job status:**
    ```http
    GET /v1/batch/jobs/{job_id}
    ```

    **Response:**
    ```json
    {
      "job_id": "batch_job_xyz789",
      "status": "completed",
      "progress": {
        "total_records": 1000000,
        "processed_records": 1000000,
        "failed_records": 0,
        "percent_complete": 100
      },
      "metrics": {
        "avg_latency_ms": 12,
        "throughput_per_sec": 5000
      },
      "output_uri": "s3://bucket/output/predictions.parquet",
      "started_at": "2026-02-05T10:05:00Z",
      "completed_at": "2026-02-05T11:30:00Z"
    }
    ```

    ---

    ### Model Management API (Admin)

    **Register model:**
    ```http
    POST /v1/models
    ```

    **Request:**
    ```json
    {
      "name": "recommendation_model",
      "version": "v1.0.0",
      "framework": "tensorflow",
      "framework_version": "2.15.0",
      "artifact_uri": "s3://bucket/models/recommendation_v1.tar.gz",
      "input_schema": {
        "user_id": "string",
        "item_ids": "array<int>",
        "context": "object"
      },
      "output_schema": {
        "scores": "array<float>",
        "item_ids": "array<int>"
      },
      "metadata": {
        "training_date": "2026-02-01",
        "dataset": "interactions_2026_jan",
        "accuracy": 0.89,
        "tags": ["production", "personalization"]
      }
    }
    ```

    **Deploy model:**
    ```http
    POST /v1/models/{model_name}/deployments
    ```

    **Request:**
    ```json
    {
      "version": "v1.0.0",
      "deployment_name": "prod-us-west",
      "resources": {
        "replicas": 3,
        "cpu": "4",
        "memory": "16Gi",
        "gpu": 0                      // 0 for CPU, 1+ for GPU
      },
      "autoscaling": {
        "enabled": true,
        "min_replicas": 2,
        "max_replicas": 10,
        "target_cpu_percent": 70,
        "target_latency_ms": 100
      },
      "batching": {
        "enabled": true,
        "max_batch_size": 32,
        "batch_timeout_ms": 50
      },
      "traffic_split": {             // For A/B testing
        "v1.0.0": 100
      }
    }
    ```

    **Create A/B test:**
    ```http
    POST /v1/models/{model_name}/ab-tests
    ```

    **Request:**
    ```json
    {
      "name": "test_v2_vs_v1",
      "versions": {
        "v1.0.0": 90,               // 90% traffic
        "v2.0.0": 10                // 10% traffic
      },
      "duration_days": 7,
      "metrics": ["latency", "accuracy", "ctr"],
      "rollback_on_error_rate": 0.05,  // Rollback if error rate > 5%
      "auto_promote": true            // Auto-promote winner after test
    }
    ```

    **Update traffic split (canary deployment):**
    ```http
    PATCH /v1/models/{model_name}/deployments/{deployment_name}/traffic
    ```

    **Request:**
    ```json
    {
      "traffic_split": {
        "v1.0.0": 50,
        "v2.0.0": 50
      },
      "canary_config": {
        "enabled": true,
        "stages": [
          {"percent": 10, "duration_minutes": 30},
          {"percent": 50, "duration_minutes": 60},
          {"percent": 100, "duration_minutes": 0}
        ],
        "success_criteria": {
          "error_rate_max": 0.01,
          "latency_p95_max_ms": 120
        }
      }
    }
    ```

    **Rollback deployment:**
    ```http
    POST /v1/models/{model_name}/rollback
    ```

    **Request:**
    ```json
    {
      "to_version": "v1.0.0",
      "reason": "High error rate in v2.0.0"
    }
    ```

    ---

    ### Monitoring API

    **Get model metrics:**
    ```http
    GET /v1/models/{model_name}/metrics?start_time=2026-02-04T00:00:00Z&end_time=2026-02-05T00:00:00Z
    ```

    **Response:**
    ```json
    {
      "model_name": "fraud_detection",
      "version": "v3.0.0",
      "metrics": {
        "request_count": 1500000,
        "error_count": 150,
        "error_rate": 0.0001,
        "latency": {
          "p50": 45,
          "p95": 98,
          "p99": 145,
          "avg": 52
        },
        "throughput_per_sec": 150,
        "prediction_distribution": {
          "fraud": 0.05,
          "not_fraud": 0.95
        },
        "confidence_distribution": {
          "high": 0.85,
          "medium": 0.12,
          "low": 0.03
        }
      },
      "resource_usage": {
        "cpu_percent": 65,
        "memory_percent": 72,
        "gpu_utilization": 85
      }
    }
    ```

    ---

    ## gRPC API (High-performance)

    For ultra-low latency, provide gRPC API alongside REST:

    ```protobuf
    syntax = "proto3";

    service PredictionService {
      rpc Predict(PredictRequest) returns (PredictResponse);
      rpc StreamPredict(stream PredictRequest) returns (stream PredictResponse);
    }

    message PredictRequest {
      string model_name = 1;
      string version = 2;
      repeated Instance instances = 3;
      map<string, string> parameters = 4;
    }

    message Instance {
      map<string, Feature> features = 1;
    }

    message Feature {
      oneof kind {
        float float_value = 1;
        int64 int_value = 2;
        string string_value = 3;
        FloatList float_list = 4;
      }
    }

    message PredictResponse {
      repeated Prediction predictions = 1;
      string model_version = 2;
      float latency_ms = 3;
    }

    message Prediction {
      float output = 1;
      float confidence = 2;
      map<string, float> probabilities = 3;
    }
    ```

=== "💾 Step 4: Database Schema"

    ## PostgreSQL (Metadata Store)

    ### Models Table
    ```sql
    CREATE TABLE models (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(255) UNIQUE NOT NULL,
        description TEXT,
        framework VARCHAR(50) NOT NULL,  -- tensorflow, pytorch, onnx, sklearn
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        created_by VARCHAR(255) NOT NULL,
        team_id UUID REFERENCES teams(id),
        tags TEXT[],
        INDEX idx_models_name (name),
        INDEX idx_models_team (team_id)
    );
    ```

    ### Model Versions Table
    ```sql
    CREATE TABLE model_versions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        model_id UUID REFERENCES models(id) ON DELETE CASCADE,
        version VARCHAR(50) NOT NULL,          -- v1.0.0, v2.1.3
        artifact_uri TEXT NOT NULL,            -- s3://bucket/models/model.tar.gz
        framework_version VARCHAR(50),         -- 2.15.0
        input_schema JSONB NOT NULL,           -- JSON schema for inputs
        output_schema JSONB NOT NULL,          -- JSON schema for outputs
        model_size_bytes BIGINT,
        training_metrics JSONB,                -- accuracy, loss, etc.
        training_metadata JSONB,               -- dataset, hyperparameters, etc.
        status VARCHAR(20) NOT NULL DEFAULT 'draft',  -- draft, active, deprecated, archived
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        created_by VARCHAR(255) NOT NULL,
        UNIQUE(model_id, version),
        INDEX idx_versions_model (model_id),
        INDEX idx_versions_status (status)
    );
    ```

    ### Deployments Table
    ```sql
    CREATE TABLE deployments (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        model_id UUID REFERENCES models(id),
        version_id UUID REFERENCES model_versions(id),
        deployment_name VARCHAR(255) NOT NULL,
        environment VARCHAR(50) NOT NULL,      -- prod, staging, dev
        cluster VARCHAR(100) NOT NULL,         -- k8s cluster name
        namespace VARCHAR(100) NOT NULL,       -- k8s namespace
        status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, deploying, active, failed, terminating
        replicas INT NOT NULL DEFAULT 1,
        resources JSONB NOT NULL,              -- cpu, memory, gpu
        autoscaling_config JSONB,
        batching_config JSONB,
        endpoint_url TEXT,                     -- http://model-service.namespace.svc
        health_status VARCHAR(20),             -- healthy, unhealthy, degraded
        deployed_at TIMESTAMP,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE(model_id, deployment_name),
        INDEX idx_deployments_model (model_id),
        INDEX idx_deployments_status (status),
        INDEX idx_deployments_environment (environment)
    );
    ```

    ### Traffic Rules Table (A/B Testing)
    ```sql
    CREATE TABLE traffic_rules (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        model_id UUID REFERENCES models(id),
        deployment_id UUID REFERENCES deployments(id),
        rule_name VARCHAR(255) NOT NULL,
        rule_type VARCHAR(20) NOT NULL,        -- ab_test, canary, shadow
        version_weights JSONB NOT NULL,        -- {"v1.0.0": 90, "v2.0.0": 10}
        routing_strategy VARCHAR(50),          -- round_robin, sticky, random
        conditions JSONB,                      -- user_id, region, etc.
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active, paused, completed
        success_criteria JSONB,
        auto_promote BOOLEAN DEFAULT false,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        created_by VARCHAR(255) NOT NULL,
        INDEX idx_traffic_model (model_id),
        INDEX idx_traffic_status (status)
    );
    ```

    ### Inference Logs Table (Partitioned by day)
    ```sql
    CREATE TABLE inference_logs (
        id BIGSERIAL,
        request_id VARCHAR(255) NOT NULL,
        model_id UUID REFERENCES models(id),
        version_id UUID REFERENCES model_versions(id),
        deployment_id UUID REFERENCES deployments(id),
        input_hash VARCHAR(64),                -- SHA256 of input (for caching/dedup)
        output JSONB,
        latency_ms INT NOT NULL,
        error_message TEXT,
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        user_id VARCHAR(255),
        metadata JSONB,                        -- request_size, response_size, etc.
        PRIMARY KEY (id, timestamp)
    ) PARTITION BY RANGE (timestamp);

    -- Create monthly partitions
    CREATE TABLE inference_logs_2026_02 PARTITION OF inference_logs
        FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

    CREATE INDEX idx_logs_model ON inference_logs(model_id, timestamp);
    CREATE INDEX idx_logs_request ON inference_logs(request_id);
    ```

    ### Batch Jobs Table
    ```sql
    CREATE TABLE batch_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_name VARCHAR(255) NOT NULL,
        model_id UUID REFERENCES models(id),
        version_id UUID REFERENCES model_versions(id),
        input_uri TEXT NOT NULL,
        output_uri TEXT NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
        total_records BIGINT,
        processed_records BIGINT DEFAULT 0,
        failed_records BIGINT DEFAULT 0,
        batch_size INT DEFAULT 1000,
        max_parallelism INT DEFAULT 10,
        progress_percent INT DEFAULT 0,
        metrics JSONB,                         -- avg_latency, throughput
        error_message TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        created_by VARCHAR(255) NOT NULL,
        INDEX idx_batch_model (model_id),
        INDEX idx_batch_status (status),
        INDEX idx_batch_created (created_at)
    );
    ```

    ### Model Metrics Table (Time-series aggregates)
    ```sql
    CREATE TABLE model_metrics (
        id BIGSERIAL,
        model_id UUID REFERENCES models(id),
        version_id UUID REFERENCES model_versions(id),
        deployment_id UUID REFERENCES deployments(id),
        metric_name VARCHAR(100) NOT NULL,     -- request_count, latency_p95, error_rate
        metric_value FLOAT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        granularity VARCHAR(20) NOT NULL,      -- minute, hour, day
        dimensions JSONB,                      -- region, user_segment, etc.
        PRIMARY KEY (id, timestamp)
    ) PARTITION BY RANGE (timestamp);

    CREATE INDEX idx_metrics_model_time ON model_metrics(model_id, metric_name, timestamp);
    ```

    ### Teams Table (Multi-tenancy)
    ```sql
    CREATE TABLE teams (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(255) UNIQUE NOT NULL,
        namespace VARCHAR(100) UNIQUE NOT NULL,  -- k8s namespace
        resource_quota JSONB,                  -- cpu, memory, gpu limits
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        INDEX idx_teams_name (name)
    );
    ```

    ---

    ## Redis (Caching)

    ### Prediction Cache
    ```
    Key: pred:{model_name}:{version}:{input_hash}
    Value: {
        "output": {...},
        "confidence": 0.95,
        "cached_at": "2026-02-05T10:00:00Z"
    }
    TTL: 3600 seconds (1 hour)
    ```

    ### Feature Cache
    ```
    Key: features:{user_id}
    Value: {
        "age": 35,
        "country": "US",
        "preferences": [...],
        "computed_at": "2026-02-05T10:00:00Z"
    }
    TTL: 300 seconds (5 minutes)
    ```

    ### Model Metadata Cache
    ```
    Key: model:{model_name}:{version}:metadata
    Value: {
        "input_schema": {...},
        "output_schema": {...},
        "framework": "tensorflow"
    }
    TTL: 3600 seconds
    ```

    ### Traffic Rules Cache
    ```
    Key: traffic:{model_name}:rules
    Value: {
        "v1.0.0": 90,
        "v2.0.0": 10,
        "strategy": "round_robin"
    }
    TTL: 60 seconds
    ```

    ---

    ## Prometheus (Metrics)

    ### Time-series Metrics
    ```
    # Request rate
    model_requests_total{model="fraud_detection", version="v3.0.0", status="success"}

    # Latency histogram
    model_latency_ms{model="fraud_detection", version="v3.0.0", quantile="0.95"}

    # Error rate
    model_errors_total{model="fraud_detection", version="v3.0.0", error_type="timeout"}

    # Throughput
    model_throughput_per_sec{model="fraud_detection", version="v3.0.0"}

    # Resource utilization
    model_cpu_usage{deployment="fraud-prod", pod="fraud-prod-abc123"}
    model_memory_usage_bytes{deployment="fraud-prod", pod="fraud-prod-abc123"}
    model_gpu_utilization{deployment="fraud-prod", pod="fraud-prod-abc123"}

    # Batch size (for dynamic batching)
    model_batch_size{model="recommendation", version="v2.0.0"}

    # Prediction distribution
    model_predictions_by_class{model="sentiment", version="v1.0.0", class="positive"}
    ```

=== "🔍 Step 5: Deep Dive - Core Components"

    ## 1. Dynamic Batching for GPU Optimization

    Dynamic batching aggregates multiple inference requests into a single batch to maximize GPU throughput while meeting latency constraints.

    ### Batching Algorithm

    ```python
    import asyncio
    import time
    from typing import List, Any
    from collections import deque

    class DynamicBatcher:
        def __init__(
            self,
            max_batch_size: int = 32,
            max_wait_ms: int = 50,
            model_inference_fn: callable = None
        ):
            self.max_batch_size = max_batch_size
            self.max_wait_ms = max_wait_ms
            self.model_inference_fn = model_inference_fn
            self.request_queue = deque()
            self.batch_lock = asyncio.Lock()
            self.batch_ready = asyncio.Event()

        async def predict(self, input_data: Any) -> Any:
            """Add request to queue and wait for batched prediction."""
            # Create a future for this request
            request_future = asyncio.Future()

            async with self.batch_lock:
                self.request_queue.append({
                    'input': input_data,
                    'future': request_future,
                    'timestamp': time.time()
                })
                self.batch_ready.set()

            # Wait for result
            result = await request_future
            return result

        async def batch_processor(self):
            """Background task that processes batches."""
            while True:
                await self.batch_ready.wait()

                async with self.batch_lock:
                    if len(self.request_queue) == 0:
                        self.batch_ready.clear()
                        continue

                    # Determine batch size based on queue length and wait time
                    oldest_request_time = self.request_queue[0]['timestamp']
                    wait_time_ms = (time.time() - oldest_request_time) * 1000

                    # Flush if max wait time exceeded or max batch size reached
                    should_flush = (
                        len(self.request_queue) >= self.max_batch_size or
                        wait_time_ms >= self.max_wait_ms
                    )

                    if not should_flush:
                        self.batch_ready.clear()
                        await asyncio.sleep(0.001)  # 1ms
                        continue

                    # Extract batch
                    batch_size = min(len(self.request_queue), self.max_batch_size)
                    batch = [self.request_queue.popleft() for _ in range(batch_size)]

                    if len(self.request_queue) == 0:
                        self.batch_ready.clear()

                # Process batch (outside lock to avoid blocking)
                await self._process_batch(batch)

        async def _process_batch(self, batch: List[dict]):
            """Process a batch of requests."""
            try:
                # Prepare batch input
                batch_input = [req['input'] for req in batch]

                # Run inference
                start_time = time.time()
                batch_output = await self.model_inference_fn(batch_input)
                inference_time_ms = (time.time() - start_time) * 1000

                # Distribute results to futures
                for req, output in zip(batch, batch_output):
                    req['future'].set_result(output)

                # Log metrics
                print(f"Processed batch: size={len(batch)}, latency={inference_time_ms:.2f}ms")

            except Exception as e:
                # Set exception for all requests in batch
                for req in batch:
                    req['future'].set_exception(e)

    # Usage example with TensorFlow model
    import tensorflow as tf

    class ModelServer:
        def __init__(self, model_path: str):
            self.model = tf.keras.models.load_model(model_path)
            self.batcher = DynamicBatcher(
                max_batch_size=32,
                max_wait_ms=50,
                model_inference_fn=self._batch_inference
            )
            # Start batch processor
            asyncio.create_task(self.batcher.batch_processor())

        async def _batch_inference(self, batch_input: List[Any]) -> List[Any]:
            """Run batched inference on GPU."""
            # Convert to tensor
            input_tensor = tf.convert_to_tensor(batch_input)

            # Run inference
            predictions = self.model(input_tensor, training=False)

            # Convert to list
            return predictions.numpy().tolist()

        async def predict(self, input_data: Any) -> Any:
            """Single prediction (automatically batched)."""
            return await self.batcher.predict(input_data)
    ```

    **Key Benefits:**
    - **GPU utilization:** Increases from 30-40% to 70-80%
    - **Throughput:** 3-5x improvement with batching
    - **Latency:** Trade-off: +10-50ms per request, but higher overall throughput
    - **Cost:** Reduce GPU count by 50-70%

    ---

    ## 2. A/B Testing Implementation

    A/B testing enables data-driven model deployment by comparing model versions with statistical rigor.

    ### Traffic Router with A/B Testing

    ```python
    import hashlib
    import random
    from typing import Dict, Optional
    from dataclasses import dataclass
    from enum import Enum

    class RoutingStrategy(Enum):
        RANDOM = "random"
        STICKY = "sticky"          # Same user always gets same version
        WEIGHTED_RANDOM = "weighted_random"

    @dataclass
    class TrafficRule:
        version_weights: Dict[str, int]  # {"v1.0.0": 90, "v2.0.0": 10}
        strategy: RoutingStrategy
        sticky_key: Optional[str] = None  # user_id, session_id, etc.

    class TrafficRouter:
        def __init__(self):
            self.rules = {}  # model_name -> TrafficRule
            self.route_cache = {}  # (model_name, sticky_key) -> version

        def set_traffic_rule(self, model_name: str, rule: TrafficRule):
            """Configure traffic split for a model."""
            self.rules[model_name] = rule
            # Clear cache for this model
            self.route_cache = {
                k: v for k, v in self.route_cache.items()
                if k[0] != model_name
            }

        def route(
            self,
            model_name: str,
            sticky_key: Optional[str] = None
        ) -> str:
            """Determine which model version to route to."""
            if model_name not in self.rules:
                return "latest"  # Default to latest version

            rule = self.rules[model_name]

            if rule.strategy == RoutingStrategy.STICKY and sticky_key:
                return self._sticky_route(model_name, sticky_key, rule)
            else:
                return self._weighted_random_route(rule)

        def _sticky_route(
            self,
            model_name: str,
            sticky_key: str,
            rule: TrafficRule
        ) -> str:
            """Consistent routing for same user/session."""
            cache_key = (model_name, sticky_key)

            # Check cache
            if cache_key in self.route_cache:
                return self.route_cache[cache_key]

            # Hash sticky key to determine version
            hash_value = int(hashlib.md5(sticky_key.encode()).hexdigest(), 16)

            # Convert weights to cumulative distribution
            versions = list(rule.version_weights.keys())
            weights = list(rule.version_weights.values())
            total_weight = sum(weights)

            # Map hash to version
            cumulative = 0
            hash_percent = (hash_value % 100)

            for version, weight in zip(versions, weights):
                cumulative += (weight * 100) // total_weight
                if hash_percent < cumulative:
                    self.route_cache[cache_key] = version
                    return version

            # Fallback
            return versions[-1]

        def _weighted_random_route(self, rule: TrafficRule) -> str:
            """Random routing based on weights."""
            versions = list(rule.version_weights.keys())
            weights = list(rule.version_weights.values())
            return random.choices(versions, weights=weights, k=1)[0]

    # Usage example
    router = TrafficRouter()

    # Set up A/B test: 90% v1, 10% v2
    router.set_traffic_rule(
        "fraud_detection",
        TrafficRule(
            version_weights={"v1.0.0": 90, "v2.0.0": 10},
            strategy=RoutingStrategy.STICKY,
            sticky_key="user_id"
        )
    )

    # Route requests
    user_id = "user_12345"
    version = router.route("fraud_detection", sticky_key=user_id)
    print(f"User {user_id} routed to {version}")
    # Same user will always get same version (sticky routing)
    ```

    ### Statistical Significance Testing

    After A/B test runs for sufficient time, determine winner:

    ```python
    import numpy as np
    from scipy import stats

    class ABTestAnalyzer:
        def __init__(
            self,
            confidence_level: float = 0.95,
            min_sample_size: int = 1000
        ):
            self.confidence_level = confidence_level
            self.min_sample_size = min_sample_size

        def analyze_latency(
            self,
            v1_latencies: np.ndarray,
            v2_latencies: np.ndarray
        ) -> dict:
            """Compare latency distributions (lower is better)."""
            if len(v1_latencies) < self.min_sample_size or \
               len(v2_latencies) < self.min_sample_size:
                return {"conclusion": "insufficient_data"}

            # T-test for means
            t_stat, p_value = stats.ttest_ind(v1_latencies, v2_latencies)

            mean_v1 = np.mean(v1_latencies)
            mean_v2 = np.mean(v2_latencies)
            p95_v1 = np.percentile(v1_latencies, 95)
            p95_v2 = np.percentile(v2_latencies, 95)

            is_significant = p_value < (1 - self.confidence_level)

            winner = None
            if is_significant:
                winner = "v1" if mean_v1 < mean_v2 else "v2"

            return {
                "conclusion": "significant" if is_significant else "not_significant",
                "winner": winner,
                "p_value": p_value,
                "v1_mean_ms": mean_v1,
                "v2_mean_ms": mean_v2,
                "v1_p95_ms": p95_v1,
                "v2_p95_ms": p95_v2,
                "improvement_percent": ((mean_v1 - mean_v2) / mean_v1) * 100
            }

        def analyze_business_metric(
            self,
            v1_conversions: int,
            v1_impressions: int,
            v2_conversions: int,
            v2_impressions: int
        ) -> dict:
            """Compare conversion rates (higher is better)."""
            if v1_impressions < self.min_sample_size or \
               v2_impressions < self.min_sample_size:
                return {"conclusion": "insufficient_data"}

            # Two-proportion z-test
            rate_v1 = v1_conversions / v1_impressions
            rate_v2 = v2_conversions / v2_impressions

            pooled_rate = (v1_conversions + v2_conversions) / (v1_impressions + v2_impressions)
            se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/v1_impressions + 1/v2_impressions))

            z_stat = (rate_v2 - rate_v1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            is_significant = p_value < (1 - self.confidence_level)

            winner = None
            if is_significant:
                winner = "v1" if rate_v1 > rate_v2 else "v2"

            return {
                "conclusion": "significant" if is_significant else "not_significant",
                "winner": winner,
                "p_value": p_value,
                "v1_rate": rate_v1,
                "v2_rate": rate_v2,
                "v1_conversions": v1_conversions,
                "v2_conversions": v2_conversions,
                "relative_improvement": ((rate_v2 - rate_v1) / rate_v1) * 100
            }
    ```

    ---

    ## 3. Canary Deployment Strategy

    Gradually roll out new model versions with automatic rollback on errors.

    ```python
    import asyncio
    from datetime import datetime, timedelta
    from typing import List

    @dataclass
    class CanaryStage:
        percent: int
        duration_minutes: int

    @dataclass
    class SuccessCriteria:
        error_rate_max: float
        latency_p95_max_ms: float

    class CanaryDeployment:
        def __init__(
            self,
            model_name: str,
            old_version: str,
            new_version: str,
            stages: List[CanaryStage],
            success_criteria: SuccessCriteria,
            traffic_router: TrafficRouter,
            metrics_client: 'MetricsClient'
        ):
            self.model_name = model_name
            self.old_version = old_version
            self.new_version = new_version
            self.stages = stages
            self.success_criteria = success_criteria
            self.router = traffic_router
            self.metrics = metrics_client
            self.current_stage = 0

        async def execute(self):
            """Execute canary deployment with automatic rollback."""
            print(f"Starting canary deployment: {self.old_version} -> {self.new_version}")

            for stage_idx, stage in enumerate(self.stages):
                self.current_stage = stage_idx

                # Update traffic split
                new_version_traffic = stage.percent
                old_version_traffic = 100 - stage.percent

                print(f"Stage {stage_idx + 1}: {new_version_traffic}% to {self.new_version}")

                self.router.set_traffic_rule(
                    self.model_name,
                    TrafficRule(
                        version_weights={
                            self.old_version: old_version_traffic,
                            self.new_version: new_version_traffic
                        },
                        strategy=RoutingStrategy.WEIGHTED_RANDOM
                    )
                )

                # Wait for stage duration
                print(f"Monitoring for {stage.duration_minutes} minutes...")
                await asyncio.sleep(stage.duration_minutes * 60)

                # Check success criteria
                is_healthy = await self._check_health(stage.duration_minutes)

                if not is_healthy:
                    print("❌ Canary failed health check. Rolling back...")
                    await self._rollback()
                    return False

                print(f"✅ Stage {stage_idx + 1} passed health check")

            # All stages passed, complete rollout
            print(f"✅ Canary deployment successful. 100% to {self.new_version}")
            self.router.set_traffic_rule(
                self.model_name,
                TrafficRule(
                    version_weights={self.new_version: 100},
                    strategy=RoutingStrategy.WEIGHTED_RANDOM
                )
            )
            return True

        async def _check_health(self, duration_minutes: int) -> bool:
            """Check if new version meets success criteria."""
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=duration_minutes)

            # Get metrics for new version
            metrics = await self.metrics.get_metrics(
                model_name=self.model_name,
                version=self.new_version,
                start_time=start_time,
                end_time=end_time
            )

            error_rate = metrics['error_rate']
            latency_p95 = metrics['latency_p95_ms']

            print(f"  Error rate: {error_rate:.4f} (max: {self.success_criteria.error_rate_max})")
            print(f"  Latency p95: {latency_p95:.1f}ms (max: {self.success_criteria.latency_p95_max_ms}ms)")

            # Check criteria
            if error_rate > self.success_criteria.error_rate_max:
                print(f"  ❌ Error rate too high: {error_rate}")
                return False

            if latency_p95 > self.success_criteria.latency_p95_max_ms:
                print(f"  ❌ Latency too high: {latency_p95}ms")
                return False

            return True

        async def _rollback(self):
            """Rollback to old version."""
            self.router.set_traffic_rule(
                self.model_name,
                TrafficRule(
                    version_weights={self.old_version: 100},
                    strategy=RoutingStrategy.WEIGHTED_RANDOM
                )
            )
            print(f"Rolled back to {self.old_version}")

            # Send alert
            await self._send_alert(
                f"Canary deployment failed for {self.model_name}. "
                f"Rolled back to {self.old_version}"
            )

        async def _send_alert(self, message: str):
            """Send alert to monitoring system."""
            # Integrate with PagerDuty, Slack, etc.
            print(f"ALERT: {message}")

    # Usage example
    async def deploy_with_canary():
        router = TrafficRouter()
        metrics = MetricsClient()

        canary = CanaryDeployment(
            model_name="fraud_detection",
            old_version="v1.0.0",
            new_version="v2.0.0",
            stages=[
                CanaryStage(percent=10, duration_minutes=30),   # 10% for 30 min
                CanaryStage(percent=50, duration_minutes=60),   # 50% for 1 hour
                CanaryStage(percent=100, duration_minutes=0)    # 100% immediately
            ],
            success_criteria=SuccessCriteria(
                error_rate_max=0.01,      # Max 1% error rate
                latency_p95_max_ms=120    # Max 120ms p95 latency
            ),
            traffic_router=router,
            metrics_client=metrics
        )

        success = await canary.execute()
        return success
    ```

    ---

    ## 4. Model Monitoring & Drift Detection

    Monitor model performance and detect when model accuracy degrades over time.

    ```python
    import numpy as np
    from collections import deque
    from typing import Dict, List

    class ModelMonitor:
        def __init__(
            self,
            model_name: str,
            version: str,
            window_size: int = 1000,
            drift_threshold: float = 0.1
        ):
            self.model_name = model_name
            self.version = version
            self.window_size = window_size
            self.drift_threshold = drift_threshold

            # Sliding windows for metrics
            self.latencies = deque(maxlen=window_size)
            self.predictions = deque(maxlen=window_size)
            self.confidences = deque(maxlen=window_size)

            # Baseline statistics (computed from training data)
            self.baseline_prediction_dist = None
            self.baseline_feature_stats = None

        def log_prediction(
            self,
            latency_ms: float,
            prediction: float,
            confidence: float,
            features: Dict[str, float]
        ):
            """Log a single prediction for monitoring."""
            self.latencies.append(latency_ms)
            self.predictions.append(prediction)
            self.confidences.append(confidence)

        def get_metrics(self) -> Dict:
            """Get current metrics."""
            if len(self.latencies) == 0:
                return {}

            latencies_arr = np.array(self.latencies)
            predictions_arr = np.array(self.predictions)
            confidences_arr = np.array(self.confidences)

            return {
                "latency_p50_ms": np.percentile(latencies_arr, 50),
                "latency_p95_ms": np.percentile(latencies_arr, 95),
                "latency_p99_ms": np.percentile(latencies_arr, 99),
                "avg_latency_ms": np.mean(latencies_arr),
                "avg_confidence": np.mean(confidences_arr),
                "low_confidence_rate": np.mean(confidences_arr < 0.5),
                "prediction_mean": np.mean(predictions_arr),
                "prediction_std": np.std(predictions_arr)
            }

        def detect_drift(self) -> Dict:
            """Detect prediction distribution drift."""
            if len(self.predictions) < self.window_size:
                return {"drift_detected": False, "reason": "insufficient_data"}

            current_dist = np.array(self.predictions)

            # Compare to baseline using Kolmogorov-Smirnov test
            if self.baseline_prediction_dist is not None:
                ks_statistic, p_value = stats.ks_2samp(
                    current_dist,
                    self.baseline_prediction_dist
                )

                drift_detected = ks_statistic > self.drift_threshold

                if drift_detected:
                    return {
                        "drift_detected": True,
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "severity": "high" if ks_statistic > 0.2 else "medium"
                    }

            return {"drift_detected": False}

        def detect_anomalies(self) -> List[str]:
            """Detect anomalous behavior."""
            anomalies = []
            metrics = self.get_metrics()

            # High latency
            if metrics.get("latency_p95_ms", 0) > 200:
                anomalies.append(f"High latency: {metrics['latency_p95_ms']:.1f}ms")

            # Low confidence
            if metrics.get("low_confidence_rate", 0) > 0.2:
                anomalies.append(f"High low-confidence rate: {metrics['low_confidence_rate']:.2%}")

            # Drift
            drift_result = self.detect_drift()
            if drift_result.get("drift_detected"):
                anomalies.append(f"Distribution drift detected: {drift_result['severity']}")

            return anomalies
    ```

=== "⚡ Step 6: Scalability & Optimizations"

    ## Horizontal Scaling

    ### Kubernetes Autoscaling

    **Horizontal Pod Autoscaler (HPA):**
    ```yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: fraud-detection-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: fraud-detection-v2
      minReplicas: 2
      maxReplicas: 20
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Pods
        pods:
          metric:
            name: model_latency_p95_ms
          target:
            type: AverageValue
            averageValue: "100"  # Scale if p95 > 100ms
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
          policies:
          - type: Percent
            value: 50
            periodSeconds: 60
        scaleUp:
          stabilizationWindowSeconds: 0  # Scale up immediately
          policies:
          - type: Percent
            value: 100
            periodSeconds: 15
    ```

    **Custom Metrics Autoscaling:**
    ```python
    # Expose custom metrics for HPA
    from prometheus_client import Gauge, start_http_server

    # Metrics
    latency_p95 = Gauge('model_latency_p95_ms', 'Model p95 latency', ['model', 'version'])
    queue_depth = Gauge('model_queue_depth', 'Inference queue depth', ['model'])
    gpu_utilization = Gauge('model_gpu_utilization', 'GPU utilization %', ['model'])

    # Update metrics
    def update_metrics(model_name: str, version: str):
        metrics = get_current_metrics(model_name, version)
        latency_p95.labels(model=model_name, version=version).set(metrics['p95_latency'])
        queue_depth.labels(model=model_name).set(metrics['queue_depth'])
        gpu_utilization.labels(model=model_name).set(metrics['gpu_util'])

    # Start Prometheus metrics server
    start_http_server(8000)
    ```

    ---

    ## GPU Pool Management

    Efficiently manage GPU resources across multiple models:

    ```yaml
    # GPU node pool (separate from CPU nodes)
    apiVersion: v1
    kind: Node
    metadata:
      name: gpu-node-1
      labels:
        node-type: gpu
        gpu-type: nvidia-t4
    spec:
      taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule

    ---
    # Model deployment requesting GPU
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: recommendation-gpu
    spec:
      replicas: 3
      template:
        spec:
          nodeSelector:
            node-type: gpu
            gpu-type: nvidia-t4
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
          containers:
          - name: model-server
            image: tensorflow/serving:latest-gpu
            resources:
              limits:
                nvidia.com/gpu: 1  # Request 1 GPU
                memory: 16Gi
              requests:
                nvidia.com/gpu: 1
                memory: 16Gi
    ```

    **GPU Sharing (Multi-Process Service - MPS):**
    ```yaml
    # Enable GPU sharing for small models
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: nvidia-mps-config
    data:
      mps.conf: |
        # Allow 4 processes to share 1 GPU
        CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
        CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    ```

    ---

    ## Request Routing & Load Balancing

    ### Intelligent Routing

    ```python
    from typing import Dict, List
    import heapq

    class IntelligentRouter:
        """Route requests to least-loaded instance."""

        def __init__(self):
            self.instances = {}  # instance_id -> {load, latency, health}

        def register_instance(
            self,
            instance_id: str,
            endpoint: str,
            capacity: int
        ):
            """Register a model serving instance."""
            self.instances[instance_id] = {
                'endpoint': endpoint,
                'capacity': capacity,
                'current_load': 0,
                'avg_latency_ms': 0,
                'health': 'healthy'
            }

        def route_request(self, model_name: str, version: str) -> str:
            """Route request to best available instance."""
            # Filter healthy instances serving this model version
            candidates = [
                (instance_id, info)
                for instance_id, info in self.instances.items()
                if info['health'] == 'healthy'
            ]

            if not candidates:
                raise Exception("No healthy instances available")

            # Use weighted least connections algorithm
            # Score = load_ratio + latency_weight
            best_instance = min(
                candidates,
                key=lambda x: (
                    x[1]['current_load'] / x[1]['capacity'] +
                    x[1]['avg_latency_ms'] / 1000
                )
            )

            instance_id, info = best_instance

            # Increment load
            self.instances[instance_id]['current_load'] += 1

            return info['endpoint']

        def release_connection(self, instance_id: str):
            """Decrement load after request completes."""
            if instance_id in self.instances:
                self.instances[instance_id]['current_load'] -= 1

        def update_metrics(
            self,
            instance_id: str,
            avg_latency_ms: float,
            health: str
        ):
            """Update instance metrics."""
            if instance_id in self.instances:
                self.instances[instance_id]['avg_latency_ms'] = avg_latency_ms
                self.instances[instance_id]['health'] = health
    ```

    ---

    ## Caching Strategies

    ### Multi-layer Caching

    ```python
    import hashlib
    from functools import lru_cache
    from typing import Any, Optional

    class PredictionCache:
        """Multi-layer cache: L1 (in-memory) -> L2 (Redis) -> Model."""

        def __init__(self, redis_client, ttl_seconds: int = 3600):
            self.redis = redis_client
            self.ttl = ttl_seconds

        @lru_cache(maxsize=10000)
        def _l1_cache(self, cache_key: str) -> Optional[Any]:
            """L1: In-process memory cache (LRU)."""
            # This is automatically cached by @lru_cache
            return None

        def _compute_cache_key(
            self,
            model_name: str,
            version: str,
            input_data: dict
        ) -> str:
            """Compute deterministic cache key from input."""
            # Sort keys for consistency
            sorted_input = str(sorted(input_data.items()))
            input_hash = hashlib.sha256(sorted_input.encode()).hexdigest()[:16]
            return f"pred:{model_name}:{version}:{input_hash}"

        async def get(
            self,
            model_name: str,
            version: str,
            input_data: dict
        ) -> Optional[Any]:
            """Get prediction from cache (L1 -> L2)."""
            cache_key = self._compute_cache_key(model_name, version, input_data)

            # L1: In-memory cache
            result = self._l1_cache(cache_key)
            if result:
                return result

            # L2: Redis cache
            result = await self.redis.get(cache_key)
            if result:
                # Populate L1 cache
                self._l1_cache(cache_key)
                return result

            return None

        async def set(
            self,
            model_name: str,
            version: str,
            input_data: dict,
            prediction: Any
        ):
            """Store prediction in cache (both L1 and L2)."""
            cache_key = self._compute_cache_key(model_name, version, input_data)

            # Store in L1 (automatically via @lru_cache on next get)
            # Store in L2 (Redis)
            await self.redis.setex(cache_key, self.ttl, prediction)

    # Usage
    async def predict_with_cache(
        model_name: str,
        version: str,
        input_data: dict,
        model_inference_fn: callable,
        cache: PredictionCache
    ) -> Any:
        """Predict with caching."""
        # Try cache first
        cached_result = await cache.get(model_name, version, input_data)
        if cached_result:
            return cached_result

        # Cache miss: run inference
        prediction = await model_inference_fn(input_data)

        # Store in cache
        await cache.set(model_name, version, input_data, prediction)

        return prediction
    ```

    ---

    ## Cost Optimization

    ### Spot Instances for Batch Inference

    ```yaml
    # Use spot/preemptible instances for batch jobs (60-90% cost savings)
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: batch-inference-job
    spec:
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-preemptible: "true"  # GKE preemptible
            # Or for AWS: eks.amazonaws.com/capacityType: "SPOT"
          tolerations:
          - key: cloud.google.com/gke-preemptible
            operator: Equal
            value: "true"
            effect: NoSchedule
          restartPolicy: OnFailure  # Restart if preempted
          containers:
          - name: batch-predictor
            image: my-batch-inference:latest
            resources:
              requests:
                cpu: "8"
                memory: 32Gi
    ```

    ### Model Compression

    Reduce model size and inference latency:

    ```python
    import tensorflow as tf

    # Quantization: FP32 -> INT8 (4x smaller, 2-4x faster)
    def quantize_model(model_path: str, output_path: str):
        """Quantize TensorFlow model to INT8."""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model size reduced: {len(tflite_model) / 1024 / 1024:.2f} MB")

    # Pruning: Remove unnecessary weights
    import tensorflow_model_optimization as tfmot

    def prune_model(model, target_sparsity: float = 0.5):
        """Prune model to reduce size."""
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model,
            **pruning_params
        )

        return model_for_pruning
    ```

    ---

    ## Performance Numbers

    ### Latency Breakdown

    ```
    Real-time inference (CPU):
    - Network latency: 5-10ms
    - Load balancing: 1-2ms
    - Feature retrieval: 5-10ms
    - Model inference: 20-50ms
    - Response serialization: 1-2ms
    Total: 32-74ms (within 100ms target)

    Real-time inference (GPU with batching):
    - Network latency: 5-10ms
    - Load balancing: 1-2ms
    - Batch queuing: 10-50ms
    - Feature retrieval: 5-10ms
    - Model inference: 5-15ms (batched)
    - Response serialization: 1-2ms
    Total: 27-89ms

    Batch inference:
    - Data loading: 5-10 min (1M records from S3)
    - Inference: 30-60 min (1M predictions)
    - Data writing: 5-10 min
    Total: 40-80 min for 1M predictions
    ```

    ### Throughput

    ```
    CPU-based model (TensorFlow Serving):
    - Single instance: 100-200 req/sec
    - With 10 instances: 1,000-2,000 req/sec

    GPU-based model (with batching):
    - Single GPU (T4): 500-1,000 req/sec
    - Single GPU (V100): 1,500-3,000 req/sec
    - With 5 GPUs: 7,500-15,000 req/sec

    Batch inference:
    - CPU: 5,000-10,000 predictions/sec
    - GPU: 20,000-50,000 predictions/sec
    ```

=== "💡 Step 7: Interview Tips"

    ## How to Approach This Problem

    ### Time Management (60-75 minutes)

    1. **Requirements (10-15 min)**
        - Clarify use cases: real-time vs batch, model types
        - Define latency targets: <100ms for real-time
        - Estimate scale: models, predictions/day, QPS
        - Discuss multi-tenancy requirements

    2. **High-level Design (15-20 min)**
        - Draw architecture: API Gateway → Router → Model Serving → Storage
        - Explain model deployment flow (Registry → Kubernetes)
        - Discuss A/B testing and canary deployments
        - Mention monitoring and alerting

    3. **Deep Dive (20-25 min)**
        - Pick 2-3 areas based on interviewer interest:
            - Dynamic batching for GPU optimization
            - A/B testing and traffic routing
            - Canary deployments with rollback
            - Model monitoring and drift detection
        - Provide code examples or algorithms
        - Discuss trade-offs

    4. **Scalability (10-15 min)**
        - Horizontal scaling with Kubernetes HPA
        - GPU pool management
        - Caching strategies (L1/L2)
        - Cost optimization (spot instances, compression)

    ---

    ## Common Follow-up Questions

    ### 1. How do you handle cold starts?

    **Answer:**
    - **Model warming:** Pre-load models at pod startup
    - **Keep-alive replicas:** Maintain min replicas > 0
    - **Predictive scaling:** Scale up before traffic arrives
    - **Model caching:** Cache frequently used models in memory
    - **Fast model loading:** Use model server optimizations (TF Serving)

    ### 2. How do you ensure low latency (<100ms)?

    **Answer:**
    - **Dynamic batching:** Aggregate requests for GPU efficiency
    - **Model optimization:** Quantization (INT8), pruning, distillation
    - **Caching:** Cache predictions for deterministic inputs
    - **Feature store:** Pre-compute and cache features
    - **Co-location:** Deploy model close to feature store
    - **gRPC:** Use gRPC instead of REST for lower overhead

    ### 3. How do you handle model versioning?

    **Answer:**
    - **Semantic versioning:** v1.0.0, v1.1.0, v2.0.0
    - **Multiple versions:** Run multiple versions simultaneously
    - **Traffic routing:** Route requests by version
    - **Model registry:** Central repository with metadata
    - **Immutable artifacts:** Never modify deployed models
    - **Rollback:** Keep old versions for quick rollback

    ### 4. How do you implement A/B testing?

    **Answer:**
    - **Traffic splitting:** Route % of traffic to each version
    - **Sticky routing:** Consistent experience per user (hash user_id)
    - **Metrics collection:** Log predictions, latency, business metrics
    - **Statistical testing:** T-test, z-test for significance
    - **Duration:** Run for 7-14 days for significance
    - **Auto-promotion:** Automatically promote winner

    ### 5. How do you detect model degradation?

    **Answer:**
    - **Prediction distribution drift:** KS-test vs baseline
    - **Confidence scores:** Monitor low-confidence predictions
    - **Business metrics:** Track CTR, conversion rate
    - **Latency spikes:** Alert on p95 latency increases
    - **Error rates:** Alert on prediction errors
    - **Scheduled retraining:** Retrain models periodically

    ### 6. How do you optimize GPU utilization?

    **Answer:**
    - **Dynamic batching:** Aggregate requests (50-50ms timeout)
    - **Right-sizing:** Match batch size to GPU memory
    - **GPU sharing:** Use NVIDIA MPS for small models
    - **Mixed precision:** FP16 inference for 2x throughput
    - **Model parallelism:** Split large models across GPUs
    - **Monitoring:** Track GPU util, aim for >80%

    ### 7. How do you handle multi-tenancy?

    **Answer:**
    - **Namespace isolation:** Separate Kubernetes namespaces per team
    - **Resource quotas:** Limit CPU/GPU/memory per team
    - **Network policies:** Isolate network traffic
    - **RBAC:** Role-based access control
    - **Model registry:** Team-scoped model access
    - **Cost allocation:** Track usage per team for chargeback

    ### 8. How do you do canary deployments?

    **Answer:**
    - **Gradual rollout:** 10% → 50% → 100% traffic
    - **Success criteria:** Error rate < 1%, latency < 100ms
    - **Monitoring:** Track metrics at each stage
    - **Automatic rollback:** Rollback if criteria violated
    - **Duration:** 30 min per stage for stability
    - **Shadow traffic:** Test without affecting prod

    ### 9. How do you handle batch inference?

    **Answer:**
    - **Async processing:** Use job queue (Kubernetes Jobs)
    - **Chunking:** Split large datasets into chunks
    - **Parallelization:** Process chunks in parallel
    - **Spot instances:** Use preemptible instances (60-90% savings)
    - **Output:** Write results to S3/GCS
    - **Monitoring:** Track progress, ETA, failures

    ### 10. How do you serve multiple frameworks?

    **Answer:**
    - **Framework-specific servers:**
        - TensorFlow: TensorFlow Serving
        - PyTorch: TorchServe
        - ONNX: ONNX Runtime
        - Scikit-learn: Custom FastAPI server
    - **Unified interface:** REST/gRPC API gateway
    - **Model registry:** Store framework metadata
    - **Docker images:** Framework-specific base images
    - **Auto-deployment:** Deploy based on framework

    ---

    ## Key Metrics to Mention

    - **Latency:** p50, p95, p99 (target: <100ms)
    - **Throughput:** Predictions per second (target: 50K QPS)
    - **Availability:** Uptime % (target: 99.95%)
    - **GPU utilization:** % (target: >80%)
    - **Error rate:** % (target: <0.1%)
    - **Cost per prediction:** $ (target: <$0.01/1K predictions)
    - **Model load time:** seconds (target: <30s)
    - **Cold start time:** seconds (target: <10s)

    ---

    ## Technologies to Mention

    **Model Serving:**
    - TensorFlow Serving, TorchServe, ONNX Runtime, Triton Inference Server
    - BentoML, KServe, Seldon Core

    **Orchestration:**
    - Kubernetes, Docker, Helm
    - AWS EKS, GKE, AKS

    **Model Registry:**
    - MLflow, DVC, Weights & Biases
    - AWS SageMaker Model Registry

    **Monitoring:**
    - Prometheus, Grafana, Datadog
    - ELK Stack (Elasticsearch, Logstash, Kibana)

    **Storage:**
    - S3, GCS, Azure Blob Storage
    - PostgreSQL, Redis

    **Workflow:**
    - Airflow, Argo Workflows, Kubeflow

    ---

    ## Real-world Examples to Mention

    **Uber Michelangelo:**
    - 10,000+ models serving billions of predictions/day
    - Sub-100ms latency for ride pricing, ETAs
    - Canary deployments with automatic rollback
    - Feature store integration for real-time features

    **Netflix:**
    - Personalized recommendations for 200M+ users
    - A/B testing framework for model comparison
    - Multi-region deployment for low latency
    - GPU optimization for deep learning models

    **Spotify:**
    - Music recommendation serving
    - Real-time and batch inference pipelines
    - Feature store with <10ms latency
    - Continuous model training and deployment

    **Airbnb:**
    - Pricing optimization models
    - Search ranking models
    - Dynamic pricing with real-time inference
    - Multi-framework support (TF, PyTorch, XGBoost)

    ---

    ## Trade-offs to Discuss

    | Decision | Option A | Option B | Trade-off |
    |----------|----------|----------|-----------|
    | **Batching** | Enabled | Disabled | Higher throughput vs lower latency |
    | **Caching** | Aggressive | Conservative | Lower latency vs staleness |
    | **GPU vs CPU** | GPU | CPU | Higher throughput vs cost |
    | **Multi-tenancy** | Shared cluster | Isolated clusters | Cost efficiency vs isolation |
    | **Model compression** | Quantized | Full precision | Smaller size vs accuracy |
    | **Autoscaling** | Aggressive | Conservative | Responsiveness vs cost |
    | **Canary rollout** | Fast (1 hour) | Slow (1 day) | Speed vs safety |

=== "🎯 Step 8: Code Examples"

    ## Complete Model Serving Implementation

    ### FastAPI Model Server

    ```python
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    import tensorflow as tf
    import numpy as np
    import time
    import asyncio
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

    # Pydantic models
    class PredictionRequest(BaseModel):
        instances: List[Dict[str, Any]]
        parameters: Optional[Dict[str, Any]] = {}

    class PredictionResponse(BaseModel):
        predictions: List[Dict[str, Any]]
        model_name: str
        model_version: str
        latency_ms: float

    # Prometheus metrics
    prediction_counter = Counter(
        'model_predictions_total',
        'Total predictions',
        ['model', 'version', 'status']
    )
    prediction_latency = Histogram(
        'model_prediction_latency_ms',
        'Prediction latency',
        ['model', 'version']
    )
    active_requests = Gauge(
        'model_active_requests',
        'Active prediction requests',
        ['model', 'version']
    )

    # FastAPI app
    app = FastAPI(title="ML Model Serving API")

    # Model manager
    class ModelManager:
        def __init__(self):
            self.models = {}  # model_name:version -> model
            self.batchers = {}  # model_name:version -> DynamicBatcher

        def load_model(
            self,
            model_name: str,
            version: str,
            model_path: str,
            enable_batching: bool = True
        ):
            """Load model into memory."""
            key = f"{model_name}:{version}"

            # Load TensorFlow model
            model = tf.keras.models.load_model(model_path)
            self.models[key] = model

            # Create batcher if enabled
            if enable_batching:
                async def inference_fn(batch_input):
                    input_tensor = tf.convert_to_tensor(batch_input)
                    predictions = model(input_tensor, training=False)
                    return predictions.numpy().tolist()

                self.batchers[key] = DynamicBatcher(
                    max_batch_size=32,
                    max_wait_ms=50,
                    model_inference_fn=inference_fn
                )
                # Start batch processor
                asyncio.create_task(self.batchers[key].batch_processor())

            print(f"Loaded model: {key}")

        async def predict(
            self,
            model_name: str,
            version: str,
            input_data: Any
        ) -> Any:
            """Run prediction."""
            key = f"{model_name}:{version}"

            if key not in self.models:
                raise ValueError(f"Model not found: {key}")

            # Use batcher if available
            if key in self.batchers:
                return await self.batchers[key].predict(input_data)
            else:
                # Direct inference
                model = self.models[key]
                input_tensor = tf.convert_to_tensor([input_data])
                prediction = model(input_tensor, training=False)
                return prediction.numpy().tolist()[0]

    # Global model manager
    model_manager = ModelManager()

    # API endpoints
    @app.post("/v1/models/{model_name}/predict", response_model=PredictionResponse)
    async def predict(model_name: str, request: PredictionRequest):
        """Prediction endpoint."""
        version = request.parameters.get('version', 'latest')

        # Track active requests
        active_requests.labels(model=model_name, version=version).inc()

        start_time = time.time()
        status = "success"

        try:
            # Run predictions
            predictions = []
            for instance in request.instances:
                pred = await model_manager.predict(model_name, version, instance)
                predictions.append({
                    "output": pred,
                    "confidence": float(np.max(pred)) if isinstance(pred, list) else float(pred)
                })

            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            prediction_counter.labels(
                model=model_name,
                version=version,
                status=status
            ).inc(len(predictions))
            prediction_latency.labels(
                model=model_name,
                version=version
            ).observe(latency_ms)

            return PredictionResponse(
                predictions=predictions,
                model_name=model_name,
                model_version=version,
                latency_ms=latency_ms
            )

        except Exception as e:
            status = "error"
            prediction_counter.labels(
                model=model_name,
                version=version,
                status=status
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            active_requests.labels(model=model_name, version=version).dec()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        """Readiness check endpoint."""
        # Check if models are loaded
        if len(model_manager.models) == 0:
            raise HTTPException(status_code=503, detail="No models loaded")
        return {"status": "ready", "models": list(model_manager.models.keys())}

    # Mount Prometheus metrics
    app.mount("/metrics", make_asgi_app())

    # Startup: Load models
    @app.on_event("startup")
    async def startup_event():
        """Load models on startup."""
        # Load models from environment variables or config
        import os
        model_path = os.getenv("MODEL_PATH", "/models/fraud_detection/v1")
        model_name = os.getenv("MODEL_NAME", "fraud_detection")
        model_version = os.getenv("MODEL_VERSION", "v1.0.0")

        model_manager.load_model(model_name, model_version, model_path)

    # Run server
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    ```

    ---

    ### Kubernetes Deployment

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: fraud-detection-v1
      namespace: ml-models
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: fraud-detection
          version: v1.0.0
      template:
        metadata:
          labels:
            app: fraud-detection
            version: v1.0.0
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "8000"
            prometheus.io/path: "/metrics"
        spec:
          containers:
          - name: model-server
            image: my-registry/fraud-detection:v1.0.0
            ports:
            - containerPort: 8000
              name: http
            env:
            - name: MODEL_NAME
              value: "fraud_detection"
            - name: MODEL_VERSION
              value: "v1.0.0"
            - name: MODEL_PATH
              value: "/models/fraud_detection/v1"
            resources:
              requests:
                cpu: "2"
                memory: 8Gi
              limits:
                cpu: "4"
                memory: 16Gi
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /ready
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 5
            volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true
          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc
              readOnly: true
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: fraud-detection-v1
      namespace: ml-models
    spec:
      selector:
        app: fraud-detection
        version: v1.0.0
      ports:
      - port: 80
        targetPort: 8000
      type: ClusterIP
    ```

    ---

    ### TensorFlow Serving Configuration

    ```bash
    # Start TensorFlow Serving with REST API
    docker run -p 8501:8501 \
      --mount type=bind,source=/models/fraud_detection,target=/models/fraud_detection \
      -e MODEL_NAME=fraud_detection \
      -e MODEL_BASE_PATH=/models/fraud_detection \
      -e REST_API_PORT=8501 \
      -e BATCHING_PARAMETERS_FILE=/config/batching.config \
      tensorflow/serving:latest-gpu
    ```

    **Batching configuration:**
    ```protobuf
    # batching.config
    max_batch_size { value: 32 }
    batch_timeout_micros { value: 50000 }  # 50ms
    max_enqueued_batches { value: 1000 }
    num_batch_threads { value: 8 }
    ```

    ---

    ### Model Deployment Script

    ```python
    import boto3
    import kubernetes
    from kubernetes import client, config
    import yaml
    import time

    class ModelDeployer:
        def __init__(self):
            # Load Kubernetes config
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()

            # AWS S3 client
            self.s3 = boto3.client('s3')

        def deploy_model(
            self,
            model_name: str,
            version: str,
            model_artifact_uri: str,
            namespace: str = "ml-models",
            replicas: int = 3,
            cpu: str = "2",
            memory: str = "8Gi",
            gpu: int = 0
        ):
            """Deploy model to Kubernetes."""
            print(f"Deploying {model_name}:{version}...")

            # Step 1: Download model from S3
            local_path = f"/tmp/{model_name}_{version}"
            self._download_model(model_artifact_uri, local_path)

            # Step 2: Build Docker image
            image_tag = self._build_docker_image(model_name, version, local_path)

            # Step 3: Push to registry
            self._push_image(image_tag)

            # Step 4: Create Kubernetes deployment
            deployment = self._create_deployment(
                model_name, version, image_tag, namespace,
                replicas, cpu, memory, gpu
            )

            # Step 5: Create service
            service = self._create_service(model_name, version, namespace)

            # Step 6: Wait for rollout
            self._wait_for_rollout(deployment.metadata.name, namespace)

            print(f"✅ Deployed {model_name}:{version}")
            return {
                "deployment_name": deployment.metadata.name,
                "service_name": service.metadata.name,
                "endpoint": f"http://{service.metadata.name}.{namespace}.svc.cluster.local"
            }

        def _download_model(self, s3_uri: str, local_path: str):
            """Download model from S3."""
            # Parse S3 URI: s3://bucket/path/to/model.tar.gz
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]

            self.s3.download_file(bucket, key, local_path)
            print(f"Downloaded model from {s3_uri}")

        def _build_docker_image(
            self,
            model_name: str,
            version: str,
            model_path: str
        ) -> str:
            """Build Docker image with model."""
            import subprocess

            image_tag = f"my-registry/{model_name}:{version}"

            # Create Dockerfile
            dockerfile = f"""
            FROM tensorflow/serving:latest
            COPY {model_path} /models/{model_name}/{version}
            ENV MODEL_NAME={model_name}
            ENV MODEL_BASE_PATH=/models/{model_name}
            """

            # Build image
            subprocess.run(["docker", "build", "-t", image_tag, "-"], input=dockerfile.encode())
            print(f"Built Docker image: {image_tag}")

            return image_tag

        def _create_deployment(
            self,
            model_name: str,
            version: str,
            image_tag: str,
            namespace: str,
            replicas: int,
            cpu: str,
            memory: str,
            gpu: int
        ):
            """Create Kubernetes deployment."""
            deployment_name = f"{model_name}-{version.replace('.', '-')}"

            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(
                    name=deployment_name,
                    namespace=namespace
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": model_name, "version": version}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": model_name, "version": version}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model-server",
                                    image=image_tag,
                                    ports=[client.V1ContainerPort(container_port=8501)],
                                    resources=client.V1ResourceRequirements(
                                        requests={"cpu": cpu, "memory": memory},
                                        limits={"cpu": cpu, "memory": memory}
                                    )
                                )
                            ]
                        )
                    )
                )
            )

            # Create deployment
            result = self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )

            print(f"Created deployment: {deployment_name}")
            return result

        def _wait_for_rollout(self, deployment_name: str, namespace: str, timeout: int = 300):
            """Wait for deployment to be ready."""
            start_time = time.time()

            while time.time() - start_time < timeout:
                deployment = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)

                if deployment.status.ready_replicas == deployment.spec.replicas:
                    print(f"✅ Deployment {deployment_name} is ready")
                    return True

                print(f"Waiting for deployment... ({deployment.status.ready_replicas}/{deployment.spec.replicas} ready)")
                time.sleep(5)

            raise Exception(f"Deployment {deployment_name} failed to become ready within {timeout}s")
    ```

---

## Summary

An ML Model Serving Platform is a complex system requiring expertise in:

- **Containerization & Orchestration:** Docker, Kubernetes, Helm
- **Model Serving Frameworks:** TensorFlow Serving, TorchServe, ONNX Runtime
- **Performance Optimization:** Dynamic batching, GPU optimization, caching
- **Deployment Strategies:** A/B testing, canary deployments, rollbacks
- **Monitoring:** Latency, throughput, accuracy, drift detection
- **Scalability:** Horizontal scaling, autoscaling, resource management

**Key Takeaways:**
1. **Low latency** (<100ms) requires batching, caching, and optimization
2. **GPU efficiency** (>80%) comes from dynamic batching and right-sizing
3. **Safe deployments** need canary rollouts with automatic rollback
4. **Multi-tenancy** requires namespace isolation and resource quotas
5. **Cost optimization** uses spot instances, compression, and efficient routing

This system powers ML-driven products at Uber, Netflix, Spotify, Airbnb, and thousands of companies worldwide.
